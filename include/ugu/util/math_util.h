/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <cassert>
#include <numeric>
#include <vector>

#include "Eigen/Geometry"
#include "ugu/util/raster_util.h"

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

  double weighted_average[3] = {0.0, 0.0, 0.0};
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

  return std::tuple(delta_p.norm(), p_proj);
}

template <typename T>
std::tuple<float, T> PointTriangleEdgeDistance(const T& p, const T& v0,
                                               const T& v1, const T& v2) {
  auto [e01_dist, p_proj01] = PointLineSegmentDistance(p, v0, v1);
  auto [e02_dist, p_proj02] = PointLineSegmentDistance(p, v0, v2);
  auto [e12_dist, p_proj12] = PointLineSegmentDistance(p, v1, v2);
  std::array<float, 3> dists = {e01_dist, e02_dist, e12_dist};
  std::array<T, 3> projs = {p_proj01, p_proj02, p_proj12};
  size_t min_index = std::distance(
      dists.begin(), std::min_element(dists.begin(), dists.end()));

  return std::tuple(dists[min_index], projs[min_index]);
}

template <typename T, typename TT>
std::tuple<TT, T> PointPlaneDistance(const T& p, const TT& a, const TT& b,
                                     const TT& c, const TT& d) {
  const T normal(a, b, c);
  // point-plane distance |ax'+by'+cz'+d|
  const TT signed_dist = p.dot(normal) + d;

  T foot = -signed_dist * normal + p;

  return std::tuple(signed_dist, foot);
}

template <typename T, typename TT>
std::tuple<float, T, Eigen::Vector2f> PointTriangleDistance(
    const T& p, const T& v0, const T& v1, const T& v2, const TT& a, const TT& b,
    const TT& c, const TT& d) {
  // Case 1: foot of perpendicular line is inside of target triangle
  auto [signed_dist_plane, plane_foot] = PointPlaneDistance(p, a, b, c, d);

  auto [is_foot_inside, foot_uv] =
      IsPoint3dInsideTriangle(plane_foot, v0, v1, v2);

  if (is_foot_inside) {
    return std::tuple(signed_dist_plane, plane_foot, foot_uv);
  }

  // Case 2: foot of perpendicular line is outside of the triangle
  // Check distance to boundary line segments of triangle
  auto [edge_abs_dist, edge_foot] = PointTriangleEdgeDistance(p, v0, v1, v2);

  auto [is_edge_inside, edge_uv] =
      IsPoint3dInsideTriangle(edge_foot, v0, v1, v2);

  if (!is_edge_inside) {
    // By numerical reason, sometimes becomes little over [0, 1]
    // So just clip
    edge_uv[0] = std::clamp(edge_uv[0], 0.f, 1.f);
    edge_uv[1] = std::clamp(edge_uv[1], 0.f, 1.f);
  }

  float signed_edge_dist =
      signed_dist_plane >= 0 ? edge_abs_dist : -edge_abs_dist;

  return std::tuple(signed_edge_dist, edge_foot, edge_uv);
}

template <typename T>
std::tuple<float, T, Eigen::Vector2f> PointTriangleDistance(const T& p,
                                                            const T& v0,
                                                            const T& v1,
                                                            const T& v2) {
  T normal = (v1 - v0).cross(v2 - v0).normalized();
  typename T::Scalar d = -normal.dot(v0);

  return PointTriangleDistance(p, v0, v1, v2, normal[0], normal[1], normal[2],
                               d);
}

inline Eigen::Vector3f Extract3f(const Eigen::Vector4f& v) {
  return Eigen::Vector3f(v[0], v[1], v[2]);
}

}  // namespace ugu
