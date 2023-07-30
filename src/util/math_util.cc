/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/util/math_util.h"

#include "ugu/util/geom_util.h"

namespace {

template <typename T, int N>
void ComputeAxisForPointsImpl(const std::vector<Eigen::Matrix<T, N, 1>>& points,
                              std::array<Eigen::Matrix<T, N, 1>, N>& axis,
                              std::array<T, N>& weights,
                              Eigen::Matrix<T, N, 1>& means) {
  ugu::Pca<Eigen::MatrixXd> pca;
  Eigen::MatrixXd pca_data;
  pca_data.resize(N, static_cast<Eigen::Index>(points.size()));

  for (size_t i = 0; i < points.size(); i++) {
    for (int j = 0; j < N; j++) {
      pca_data(j, i) = static_cast<double>(points[i][j]);
    }
  }

  pca.Compute(pca_data);

  // ugu::LOGI("\n");
  for (int j = 0; j < N; j++) {
    axis[j] = Eigen::Matrix<T, N, 1>(pca.vecs.col(j).cast<T>());
    weights[j] = T(pca.coeffs(j, 0));
    // ugu::LOGI("%d %f\n", j, weights[j]);
  }

  means = pca.means.cast<T>();
  // ugu::LOGI("\n");
}

}  // namespace

namespace ugu {

Eigen::Vector3f MedianColor(const std::vector<Eigen::Vector3f>& colors) {
  Eigen::Vector3f median;
  std::vector<std::vector<float>> ith_channel_list(3);
  for (const auto& color : colors) {
    for (int i = 0; i < 3; i++) {
      ith_channel_list[i].push_back(color[i]);
    }
  }
  for (int i = 0; i < 3; i++) {
    median[i] = Median(ith_channel_list[i]);
  }
  return median;
}

void ComputeAxisForPoints(const std::vector<Eigen::Vector3f>& points,
                          std::array<Eigen::Vector3f, 3>& axes,
                          std::array<float, 3>& weights,
                          Eigen::Vector3f& means) {
  ComputeAxisForPointsImpl<float, 3>(points, axes, weights, means);
}

void ComputeAxisForPoints(const std::vector<Eigen::Vector2f>& points,
                          std::array<Eigen::Vector2f, 2>& axes,
                          std::array<float, 2>& weights,
                          Eigen::Vector2f& means) {
  ComputeAxisForPointsImpl<float, 2>(points, axes, weights, means);
}

OrientedBoundingBox::OrientedBoundingBox(){};
OrientedBoundingBox::OrientedBoundingBox(
    const std::vector<Eigen::Vector3f>& points) {
  Init(points);
}
OrientedBoundingBox::~OrientedBoundingBox(){};
void OrientedBoundingBox::Init(const std::vector<Eigen::Vector3f>& points) {
  std::array<float, 3> weights;
  Eigen::Vector3f means;
  ComputeAxisForPoints(points, axes, weights, means);
  bb2wld.col(0) = axes[0];
  bb2wld.col(1) = axes[1];
  bb2wld.col(2) = axes[2];
  wld2bb = bb2wld.inverse();
  std::vector<Eigen::Vector3f> points_bb;
  std::transform(points.begin(), points.end(), std::back_inserter(points_bb),
                 [&](const Eigen::Vector3f& p) { return wld2bb * p; });
  max_bb = ComputeMaxBound(points_bb);
  min_bb = ComputeMinBound(points_bb);

  center_wld = bb2wld * (0.5f * (max_bb + min_bb));
  len = max_bb - min_bb;
  offset = len.minCoeff() * 0.001f;
}

bool OrientedBoundingBox::IsInside(const Eigen::Vector3f& p_wld) {
  auto p_bb = wld2bb * p_wld;
  for (int i = 0; i < 3; i++) {
    if (p_bb[i] < min_bb[i] - offset || max_bb[i] + offset < p_bb[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace ugu
