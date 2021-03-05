/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <fstream>
#include <numeric>
#include <string>

#include "ugu/camera.h"
#include "ugu/common.h"
#include "ugu/image.h"
#include "ugu/mesh.h"

namespace ugu {

bool Depth2PointCloud(const Image1f& depth, const Camera& camera,
                      Image3f* point_cloud, bool gl_coord = false);

bool Depth2PointCloud(const Image1f& depth, const Camera& camera,
                      Mesh* point_cloud, bool gl_coord = false);

bool Depth2PointCloud(const Image1f& depth, const Image3b& color,
                      const Camera& camera, Mesh* point_cloud,
                      bool gl_coord = false);

bool Depth2Mesh(const Image1f& depth, const Camera& camera, Mesh* mesh,
                float max_connect_z_diff, int x_step = 1, int y_step = 1,
                bool gl_coord = false, ugu::Image3f* point_cloud = nullptr,
                ugu::Image3f* normal = nullptr);

bool Depth2Mesh(const Image1f& depth, const Image3b& color,
                const Camera& camera, Mesh* mesh, float max_connect_z_diff,
                int x_step = 1, int y_step = 1, bool gl_coord = false,
                const std::string& material_name = "Depth2Mesh_mat",
                bool with_vertex_color = false,
                ugu::Image3f* point_cloud = nullptr,
                ugu::Image3f* normal = nullptr);

void WriteFaceIdAsText(const Image1i& face_id, const std::string& path);

inline bool WriteBinary(const std::string& path, void* data, size_t size) {
  std::ofstream ofs(path, std::ios::binary);
  ofs.write(reinterpret_cast<char*>(data), size);

  if (ofs.bad()) {
    return false;
  }
  return true;
}

inline bool LoadBinaryBase(const std::string& path, std::vector<char>* data) {
  std::ifstream ifs(path, std::ios::binary);

  ifs.seekg(0, std::ios::end);
  long long int size = ifs.tellg();
  ifs.seekg(0);

  data->resize(size);
  ifs.read(data->data(), size);

  return true;
}

inline bool LoadBinaryBase(const std::string& path, char* data) {
  std::ifstream ifs(path, std::ios::binary);

  ifs.seekg(0, std::ios::end);
  long long int size = ifs.tellg();
  ifs.seekg(0);

  ifs.read(data, size);

  return true;
}

template <typename T>
bool LoadBinary(const std::string& path, std::vector<T>* data) {
  std::vector<char> internal_data;
  LoadBinaryBase(path, &internal_data);
  size_t elem_num = internal_data.size() / sizeof(T);
  data->resize(elem_num);

  std::memcpy(data->data(), internal_data.data(), internal_data.size());

  return true;
}

#ifdef UGU_USE_OPENCV
template <typename T>
bool WriteBinary(const std::string& path, ugu::Image<T>& image) {
  size_t size_in_bytes = image.total() * image.elemSize();
  return WriteBinary(path, image.data, size_in_bytes);
}

template <typename T>
bool LoadBinary(const std::string& path, ugu::Image<T>& image) {
  std::vector<char> internal_data;
  LoadBinaryBase(path, &internal_data);

  size_t size_in_bytes = image.total() * image.elemSize();

  if (size_in_bytes != internal_data.size()) {
    return false;
  }

  std::memcpy(image.data, internal_data.data(), size_in_bytes);

  return true;
}

#endif

inline void NormalizeWeights(const std::vector<float>& weights,
                             std::vector<float>* normalized_weights) {
  assert(!weights.empty());
  normalized_weights->clear();
  std::copy(weights.begin(), weights.end(),
            std::back_inserter(*normalized_weights));
  // Add eps
  const float eps = 0.000001f;
  std::for_each(normalized_weights->begin(), normalized_weights->end(),
                [&](float& x) { x += eps; });
  float sum = std::accumulate(normalized_weights->begin(),
                              normalized_weights->end(), 0.0f);
  if (sum > 0.000001) {
    std::for_each(normalized_weights->begin(), normalized_weights->end(),
                  [&](float& n) { n /= sum; });
  } else {
    // if sum is too small, just set even weights
    float val = 1.0f / static_cast<float>(normalized_weights->size());
    std::fill(normalized_weights->begin(), normalized_weights->end(), val);
  }
}

template <typename T>
Eigen::Matrix<T, 3, 1> WeightedAverage(
    const std::vector<Eigen::Matrix<T, 3, 1>>& data,
    const std::vector<float>& weights) {
  assert(data.size() > 0);
  assert(data.size() == weights.size());

  std::vector<float> normalized_weights;
  NormalizeWeights(weights, &normalized_weights);

  double weighted_average[3];
  for (size_t i = 0; i < data.size(); i++) {
    weighted_average[0] += (data[i][0] * normalized_weights[i]);
    weighted_average[1] += (data[i][1] * normalized_weights[i]);
    weighted_average[2] += (data[i][2] * normalized_weights[i]);
  }

  return Eigen::Matrix<T, 3, 1>(static_cast<float>(weighted_average[0]),
                                static_cast<float>(weighted_average[1]),
                                static_cast<float>(weighted_average[2]));
}

template <typename T>
T WeightedMedian(const std::vector<T>& data,
                 const std::vector<float>& weights) {
  assert(data.size() > 0);
  assert(data.size() == weights.size());

  std::vector<float> normalized_weights;
  NormalizeWeights(weights, &normalized_weights);

  std::vector<std::pair<float, T>> data_weights;
  for (size_t i = 0; i < data.size(); i++) {
    data_weights.push_back(std::make_pair(normalized_weights[i], data[i]));
  }
  std::sort(data_weights.begin(), data_weights.end(),
            [](const std::pair<float, T>& a, const std::pair<float, T>& b) {
              return a.first < b.first;
            });

  float weights_sum{0};
  size_t index{0};
  for (size_t i = 0; i < data_weights.size(); i++) {
    weights_sum += data_weights[i].first;
    if (weights_sum > 0.5f) {
      index = i;
      break;
    }
  }

  return data_weights[index].second;
}


// FINDING OPTIMAL ROTATION AND TRANSLATION BETWEEN CORRESPONDING 3D POINTS
// http://nghiaho.com/?page_id=671
Eigen::Affine3d FindRigidTransformFrom3dCorrespondences(
    const std::vector<Eigen::Vector3d>& src,
    const std::vector<Eigen::Vector3d>& dst);

Eigen::Affine3d FindRigidTransformFrom3dCorrespondences(
    const std::vector<Eigen::Vector3f>& src,
    const std::vector<Eigen::Vector3f>& dst);

}  // namespace ugu
