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

// "Least-squares estimation of transformation parameters between two point
// patterns ", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`
// implementation reference:
// https://github.com/scikit-image/scikit-image/blob/main/skimage/transform/_geometric.py#L63
bool FindSimilarityTransformFromPointCorrespondences(
    const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst, Eigen::MatrixXd& R,
    Eigen::MatrixXd& t, Eigen::MatrixXd& scale, Eigen::MatrixXd& T);

Eigen::Affine3d FindSimilarityTransformFrom3dCorrespondences(
    const std::vector<Eigen::Vector3d>& src,
    const std::vector<Eigen::Vector3d>& dst);

Eigen::Affine3d FindSimilarityTransformFrom3dCorrespondences(
    const std::vector<Eigen::Vector3f>& src,
    const std::vector<Eigen::Vector3f>& dst);

Eigen::Affine3d FindSimilarityTransformFrom3dCorrespondences(
    const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst);

template <typename T>
float EdgeFunction(const T& a, const T& b, const T& c) {
  return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0]);
}

template <typename T>
bool RasterizeTriangle(const std::array<Eigen::Vector3f, 3>& src_vetex_color,
                       const std::array<Eigen::Vector2f, 3>& target_tri,
                       ugu::Image<T>* target, ugu::Image1b* mask,
                       float min_val = 1.f, float max_val = -1.f) {
  // Area could be negative
  float area = EdgeFunction(target_tri[0], target_tri[1], target_tri[2]);
  if (std::abs(area) < std::numeric_limits<float>::min()) {
    area = area > 0 ? std::numeric_limits<float>::min()
                    : -std::numeric_limits<float>::min();
  }
  float inv_area = 1.0f / area;

  // Loop for bounding box of the target triangle
  int xmin = static_cast<int>(
      std::min({target_tri[0].x(), target_tri[1].x(), target_tri[2].x()}) - 1);
  xmin = std::max(0, std::min(xmin, target->cols - 1));
  int xmax = static_cast<int>(
      std::max({target_tri[0].x(), target_tri[1].x(), target_tri[2].x()}) + 1);
  xmax = std::max(0, std::min(xmax, target->cols - 1));

  int ymin = static_cast<int>(
      std::min({target_tri[0].y(), target_tri[1].y(), target_tri[2].y()}) - 1);
  ymin = std::max(0, std::min(ymin, target->rows - 1));
  int ymax = static_cast<int>(
      std::max({target_tri[0].y(), target_tri[1].y(), target_tri[2].y()}) + 1);
  ymax = std::max(0, std::min(ymax, target->rows - 1));

  auto saturate_cast = [=](float x) {
    constexpr bool is_integer = std::numeric_limits<T::value_type>::is_integer;
    if (min_val < max_val) {
      x = std::clamp(x, min_val, max_val);
    }
    if (is_integer) {
      return std::round(x);
    }
    return x;
  };

  for (int y = ymin; y <= ymax; y++) {
    for (int x = xmin; x <= xmax; x++) {
      Eigen::Vector2f pixel_sample(static_cast<float>(x),
                                   static_cast<float>(y));
      float w0 = EdgeFunction(target_tri[1], target_tri[2], pixel_sample);
      float w1 = EdgeFunction(target_tri[2], target_tri[0], pixel_sample);
      float w2 = EdgeFunction(target_tri[0], target_tri[1], pixel_sample);
      // Barycentric in the target triangle
      w0 *= inv_area;
      w1 *= inv_area;
      w2 *= inv_area;

      // Barycentric coordinate should be positive inside of the triangle
      // Skip outside of the target triangle
      if (w0 < 0 || w1 < 0 || w2 < 0) {
        continue;
      }

      // Barycentric to interpolate color
      Eigen::Vector3f color = w0 * src_vetex_color[0] +
                              w1 * src_vetex_color[1] + w2 * src_vetex_color[2];

      target->at<T>(y, x) =
          T({static_cast<T::value_type>(saturate_cast(color[0])),
             static_cast<T::value_type>(saturate_cast(color[1])),
             static_cast<T::value_type>(saturate_cast(color[2]))});

      if (mask != nullptr) {
        mask->at<unsigned char>(y, x) = 255;
      }
    }
  }

  return true;
}

template <typename T>
bool RasterizeVertexAttributeToTexture(
    const std::vector<Eigen::Vector3f>& vertex_colors,
    const std::vector<Eigen::Vector3i>& vertex_color_indices,
    const std::vector<Eigen::Vector2f>& uvs,
    const std::vector<Eigen::Vector3i>& uv_indices, Image<T>& texture,
    int width = 512, int height = 512) {
  if (vertex_color_indices.empty() ||
      vertex_color_indices.size() != uv_indices.size()) {
    return false;
  }

  if (width > 0 && height > 0) {
    texture = Image<T>::zeros(height, width);
  } else if (texture.empty()) {
    return false;
  }

  auto face_num = vertex_color_indices.size();
  for (auto i = 0; i < face_num; i++) {
    const auto vc_face = vertex_color_indices[i];
    const auto uv_face = uv_indices[i];

    std::array<Eigen::Vector3f, 3> src_vetex_color{vertex_colors[vc_face[0]],
                                                   vertex_colors[vc_face[1]],
                                                   vertex_colors[vc_face[2]]};
    std::array<Eigen::Vector2f, 3> target_tri{uvs[uv_face[0]], uvs[uv_face[1]],
                                              uvs[uv_face[2]]};

    for (auto& tri : target_tri) {
      tri.x() = texture.cols * tri.x() - 0.5f;
      tri.y() = texture.rows * (1.f - tri.y()) - 0.5f;
    }

    RasterizeTriangle(src_vetex_color, target_tri, &texture, nullptr);
  }

  return true;
}

inline float U2X(float u, int w) { return u * w - 0.5f; }
inline float V2Y(float v, int h) { return (1.f - v) * h - 0.5f; }

template <typename T>
bool FetchVertexAttributeFromTexture(const Image<T>& texture,
                                     const std::vector<Eigen::Vector2f>& uvs,
                                     std::vector<Eigen::Vector3f>& vertex_attrs,
                                     bool use_bilinear = true) {
  vertex_attrs.clear();

  const int w = texture.cols;
  const int h = texture.rows;
  for (const auto& uv : uvs) {
    float x = U2X(uv.x(), w);
    float y = V2Y(uv.y(), h);
    T attr;
    if (use_bilinear) {
      attr = BilinearInterpolation(x, y, texture);
    } else {
      int x_i = static_cast<int>(
          std::clamp(std::round(x), 0.f, static_cast<float>(w - 1)));
      int y_i = static_cast<int>(
          std::clamp(std::round(y), 0.f, static_cast<float>(h - 1)));
      attr = texture.at<T>(y_i, x_i);
    }
    vertex_attrs.emplace_back(static_cast<float>(attr[0]),
                              static_cast<float>(attr[1]),
                              static_cast<float>(attr[2]));
  }

  return true;
}

}  // namespace ugu
