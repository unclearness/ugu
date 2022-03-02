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

  std::unique_ptr<nf_eigen_adaptor> m_mat_index;
  int m_max_leaf_data_num = 10;
  std::vector<Point> m_data;
  Eigen::MatrixXf m_mat;  // TODO: double
};

template <typename Point>
class KdTreeNanoflannVector : public KdTree<Point> {
 public:
  KdTreeNanoflannVector() {}
  ~KdTreeNanoflannVector() {}
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
  template <class VectorOfVectorsType, typename num_t = double, int DIM = -1,
            class Distance = nanoflann::metric_L2, typename IndexType = size_t>
  struct KDTreeVectorOfVectorsAdaptor {
    using self_t =
        KDTreeVectorOfVectorsAdaptor<VectorOfVectorsType, num_t, DIM, Distance>;
    using metric_t =
        typename Distance::template traits<num_t, self_t>::distance_t;
    using index_t =
        nanoflann::KDTreeSingleIndexAdaptor<metric_t, self_t, DIM, IndexType>;

    /** The kd-tree index for the user to call its methods as usual with any
     * other FLANN index */
    index_t* index = nullptr;

    /// Constructor: takes a const ref to the vector of vectors object with the
    /// data points
    KDTreeVectorOfVectorsAdaptor(const size_t /* dimensionality */,
                                 const VectorOfVectorsType& mat,
                                 const int leaf_max_size = 10)
        : m_data(mat) {
      assert(mat.size() != 0 && mat[0].size() != 0);
      const size_t dims = mat[0].size();
      if (DIM > 0 && static_cast<int>(dims) != DIM)
        throw std::runtime_error(
            "Data set dimensionality does not match the 'DIM' template "
            "argument");
      index =
          new index_t(static_cast<int>(dims), *this /* adaptor */,
                      nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size));
      index->buildIndex();
    }

    ~KDTreeVectorOfVectorsAdaptor() { delete index; }

    const VectorOfVectorsType& m_data;

    /** Query for the \a num_closest closest points to a given point
     *  (entered as query_point[0:dim-1]).
     *  Note that this is a short-cut method for index->findNeighbors().
     *  The user can also call index->... methods as desired.
     *
     * \note nChecks_IGNORED is ignored but kept for compatibility with
     * the original FLANN interface.
     */
    inline void query(const num_t* query_point, const size_t num_closest,
                      IndexType* out_indices, num_t* out_distances_sq,
                      const int nChecks_IGNORED = 10) const {
      nanoflann::KNNResultSet<num_t, IndexType> resultSet(num_closest);
      resultSet.init(out_indices, out_distances_sq);
      index->findNeighbors(resultSet, query_point, nanoflann::SearchParams());
    }

    /** @name Interface expected by KDTreeSingleIndexAdaptor
     * @{ */

    const self_t& derived() const { return *this; }
    self_t& derived() { return *this; }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return m_data.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    inline num_t kdtree_get_pt(const size_t idx, const size_t dim) const {
      return m_data[idx][dim];
    }

    // Optional bounding-box computation: return false to default to a standard
    // bbox computation loop.
    // Return true if the BBOX was already computed by the class and returned
    // in "bb" so it can be avoided to redo it again. Look at bb.size() to
    // find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const {
      return false;
    }
    /** @} */

  };  // end of KDTreeVectorOfVectorsAdaptor

  using nf_vector_adaptor =
      KDTreeVectorOfVectorsAdaptor<std::vector<Point>, float>;

  std::unique_ptr<nf_vector_adaptor> m_index;
  int m_max_leaf_data_num = 10;
  std::vector<Point> m_data;
};

template <typename Point, typename Adaptor, typename Index>
KdTreeSearchResults RangeQueryKdTreeNanoflann(const Point& query,
                                              const Adaptor& index,
                                              float epsilon) {
  std::vector<std::pair<Index, float>> ret_matches;
  nanoflann::SearchParams params;
  // For squared L2 distance
  const float sq_epsilon = epsilon * epsilon;
  const size_t nMatches =
      index.index->radiusSearch(query.data(), sq_epsilon, ret_matches, params);

  KdTreeSearchResults results;
  for (size_t i = 0; i < nMatches; i++) {
    const auto& m = ret_matches[i];
    results.push_back({static_cast<size_t>(m.first),
                       static_cast<double>(std::sqrt(m.second))});
  }

  return results;
}

template <typename Point, typename Adaptor, typename Index>
KdTreeSearchResults QueryKnnNanoflann(const Point& query, const Adaptor& index,
                                      size_t k) {
  std::vector<Index> out_indices(k);
  std::vector<float> out_distances_sq(k);
  const size_t nMatches = index.index->knnSearch(
      query.data(), k, out_indices.data(), out_distances_sq.data());

  KdTreeSearchResults results;
  for (size_t i = 0; i < nMatches; i++) {
    results.push_back({static_cast<size_t>(out_indices[i]),
                       static_cast<double>(std::sqrt(out_distances_sq[i]))});
  }

  return results;
}

template <typename Point>
bool KdTreeNanoflannEigenX<Point>::Build() {
  Eigen::MatrixXf mat(m_data.size(), m_data[0].rows());
  for (size_t i = 0; i < m_data.size(); i++) {
    for (Eigen::Index j = 0; j < m_data[0].rows(); j++) {
      mat.coeffRef(i, j) = m_data[i][j];
    }
  }
  m_mat = std::move(mat);
  m_mat_index = std::make_unique<nf_eigen_adaptor>(
      static_cast<size_t>(m_data[0].size()), m_mat, m_max_leaf_data_num);
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
  return QueryKnnNanoflann<Point, nf_eigen_adaptor, Eigen::Index>(
      query, *m_mat_index, 1);
}

template <typename Point>
KdTreeSearchResults KdTreeNanoflannEigenX<Point>::SearchKnn(
    const Point& query, const size_t& k) const {
  return QueryKnnNanoflann<Point, nf_eigen_adaptor, Eigen::Index>(
      query, *m_mat_index, k);
}

template <typename Point>
KdTreeSearchResults KdTreeNanoflannEigenX<Point>::SearchRadius(
    const Point& query, const double& r) const {
  return RangeQueryKdTreeNanoflann<Point, nf_eigen_adaptor, Eigen::Index>(
      query, *m_mat_index, static_cast<float>(r));
}

template <typename Point>
bool KdTreeNanoflannVector<Point>::Build() {
  m_index = std::make_unique<nf_vector_adaptor>(
      static_cast<size_t>(m_data.size()), m_data, m_max_leaf_data_num);
  m_index->index->buildIndex();
  return true;
}

template <typename Point>
void KdTreeNanoflannVector<Point>::Clear() {
  m_data.clear();
}

template <typename Point>
void KdTreeNanoflannVector<Point>::SetMaxLeafDataNum(int max_leaf_data_num) {
  m_max_leaf_data_num = max_leaf_data_num;
}

template <typename Point>
void KdTreeNanoflannVector<Point>::SetData(const std::vector<Point>& data) {
  m_data = data;
}

template <typename Point>
KdTreeSearchResults KdTreeNanoflannVector<Point>::SearchNn(
    const Point& query) const {
  return QueryKnnNanoflann<Point, nf_vector_adaptor, size_t>(query, *m_index,
                                                             1);
}

template <typename Point>
KdTreeSearchResults KdTreeNanoflannVector<Point>::SearchKnn(
    const Point& query, const size_t& k) const {
  return QueryKnnNanoflann<Point, nf_vector_adaptor, size_t>(query, *m_index,
                                                             k);
}

template <typename Point>
KdTreeSearchResults KdTreeNanoflannVector<Point>::SearchRadius(
    const Point& query, const double& r) const {
  return RangeQueryKdTreeNanoflann<Point, nf_vector_adaptor, size_t>(
      query, *m_index, static_cast<float>(r));
}

}  // namespace ugu
#endif