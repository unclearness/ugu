/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 *
 */
#ifdef UGU_USE_NANOFLANN
#pragma once

#include <stdexcept>
#include <unordered_set>

#include "Eigen/Core"
#include "nanoflann.hpp"
#include "ugu/accel/kdtree.h"

namespace ugu {

template <typename Point>
class KdTreeNanoflannEigenX : public KdTree<Point> {
 public:
  KdTreeNanoflannEigenX() {}
  ~KdTreeNanoflannEigenX() {}
  bool Build() override;
  void Clear() override;

  void SetData(const std::vector<Point>& data) override;
  void SetMaxLeafDataNum(int max_leaf_data_num) override;

  KdTreeSearchResults SearchNn(const Point& query) const override;
  KdTreeSearchResults SearchKnn(const Point& query,
                                const size_t& k) const override;
  KdTreeSearchResults SearchRadius(const Point& query,
                                   const double& r) const override;

 private:
  using nf_eigen_adaptor = nanoflann::KDTreeEigenMatrixAdaptor<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>;

  mutable std::shared_ptr<nf_eigen_adaptor> m_mat_index;
  int m_max_leaf_data_num = 10;
  std::vector<Point> m_data;
  Eigen::MatrixXf m_mat;  // TODO: double

  static KdTreeSearchResults RangeQueryKdTree(const Eigen::VectorXf& query,
                                              nf_eigen_adaptor& index,
                                              float epsilon) {
    std::vector<std::pair<Eigen::Index, float>> ret_matches;
    nanoflann::SearchParams params;
    // For squared L2 distance
    const float sq_epsilon = epsilon * epsilon;
    const size_t nMatches = index.index->radiusSearch(query.data(), sq_epsilon,
                                                      ret_matches, params);

    KdTreeSearchResults results;
    for (size_t i = 0; i < nMatches; i++) {
      const auto& m = ret_matches[i];
      results.push_back(
          {static_cast<size_t>(m.first), static_cast<double>(m.second)});
    }

    return results;
  }
};

template <typename Point>
bool KdTreeNanoflannEigenX<Point>::Build() {
  Eigen::MatrixXf mat(m_data.size(), m_data[0].rows());
  for (size_t i = 0; i < m_data.size(); i++) {
    for (Eigen::Index j = 0; j < m_data[0].rows(); j++) {
      mat.coeffRef(i, j) = m_data[i][j];
    }
  }
  m_mat = std::move(mat);
  m_mat_index = std::make_shared<nf_eigen_adaptor>(
      static_cast<size_t>(m_data[0].size()), m_mat, m_max_leaf_data_num);
  // m_mat_index = std::move(tmp);
  m_mat_index->index->buildIndex();
  return true;
}

template <typename Point>
void KdTreeNanoflannEigenX<Point>::Clear() {
  m_data.clear();
}

template <typename Point>
void KdTreeNanoflannEigenX<Point>::SetMaxLeafDataNum(int max_leaf_data_num) {
  m_max_leaf_data_num = max_leaf_data_num;
}

template <typename Point>
void KdTreeNanoflannEigenX<Point>::SetData(const std::vector<Point>& data) {
  m_data = data;
}

template <typename Point>
KdTreeSearchResults KdTreeNanoflannEigenX<Point>::SearchNn(
    const Point& query) const {
  throw std::logic_error("not implemented");
}

template <typename Point>
KdTreeSearchResults KdTreeNanoflannEigenX<Point>::SearchKnn(
    const Point& query, const size_t& k) const {
  throw std::logic_error("not implemented");
}

template <typename Point>
KdTreeSearchResults KdTreeNanoflannEigenX<Point>::SearchRadius(
    const Point& query, const double& r) const {
  return RangeQueryKdTree(query, *m_mat_index, static_cast<float>(r));
}

}  // namespace ugu
#endif