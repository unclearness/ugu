/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <cassert>
#include <numeric>
#include <vector>

#include "Eigen/Geometry"

namespace ugu {

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

template <typename T, typename TT>
T LinearInterpolation(const T& val1, const T& val2, const TT& r) {
  return (TT(1.0) - r) * val1 + r * val2;
}

template <class Container, class Predicate>
Container Filter(const Container& c, Predicate f) {
  Container r;
  std::copy_if(begin(c), end(c), std::back_inserter(r), f);
  return r;
}

template <class Container>
Container Mask(const Container& c, std::vector<bool> m) {
  Container r;
  for (size_t i = 0; i < c.size(); i++) {
    if (m[i]) {
      r.push_back(c[i]);
    }
  }
  return r;
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

// https://github.com/facebookresearch/pytorch3d/blob/14dd2611eeda6d0f4b43a3cadf90ef3c64eb1d0f/pytorch3d/renderer/mesh/rasterize_meshes.py#L755
template <typename T>
std::tuple<float, T> PointLineSegmentDistance(const T& p, const T& v0,
                                              const T& v1,
                                              bool limit_segment = true,
                                              float eps = 1e-8f) {
  T v1v0 = v1 - v0;
  float l2 = v1v0.dot(v1v0);  // |v1 - v0|^2
  if (l2 <= eps) {
    return std::tuple((p - v1).dot(p - v1), (v0 + v1) * 0.5f);  // v0 == v1
  }
  float t = v1v0.dot(p - v0) / l2;
  // Limit to segment
  if (limit_segment) {
    t = std::clamp(t, 0.f, 1.f);
  }
  T p_proj = v0 + t * v1v0;
  T delta_p = p_proj - p;
  return std::tuple(delta_p.dot(delta_p), p_proj);
}

template <typename T>
std::tuple<float, T> PointTriangleDistance(const T& p, const T& v0, const T& v1,
                                           const T& v2) {
  auto [e01_dist, p_proj01] = PointLineSegmentDistance(p, v0, v1);
  auto [e02_dist, p_proj02] = PointLineSegmentDistance(p, v0, v2);
  auto [e12_dist, p_proj12] = PointLineSegmentDistance(p, v1, v2);
  std::array<float, 3> dists = {e01_dist, e02_dist, e12_dist};
  std::array<T, 3> projs = {p_proj01, p_proj02, p_proj12};
  size_t min_index = std::distance(
      dists.begin(), std::min_element(dists.begin(), dists.end()));
  return std::tuple(dists[min_index], projs[min_index]);
}

}  // namespace ugu
