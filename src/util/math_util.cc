/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/util/math_util.h"

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

}  // namespace ugu
