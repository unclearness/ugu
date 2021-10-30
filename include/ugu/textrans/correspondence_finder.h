/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#pragma once

#ifdef UGU_USE_NANOFLANN

#include "ugu/image.h"
#include "ugu/mesh.h"

#ifdef UGU_USE_NANOFLANN
#include "nanoflann.hpp"
#endif

namespace ugu {

struct Corresp {
  int32_t fid = -1;
  Eigen::Vector3f uv = Eigen::Vector3f::Zero();
  Eigen::Vector3f p =
      Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
  float singed_dist = std::numeric_limits<float>::lowest();
  float abs_dist = std::numeric_limits<float>::max();
};

class CorrespFinder {
 public:
  virtual ~CorrespFinder() {}
  virtual bool Init(const std::vector<Eigen::Vector3f>& verts,
                    const std::vector<Eigen::Vector3i>& verts_faces) = 0;
  virtual Corresp Find(const Eigen::Vector3f& src_p,
                       const Eigen::Vector3f& src_n) const = 0;
};
using CorrespFinderPtr = std::shared_ptr<CorrespFinder>;

#ifdef UGU_USE_NANOFLANN
template <class VectorOfVectorsType, typename num_t = double, int DIM = -1,
          class Distance = nanoflann::metric_L2, typename IndexType = size_t>
struct KDTreeVectorOfVectorsAdaptor {
  typedef KDTreeVectorOfVectorsAdaptor<VectorOfVectorsType, num_t, DIM,
                                       Distance>
      self_t;
  typedef
      typename Distance::template traits<num_t, self_t>::distance_t metric_t;
  typedef nanoflann::KDTreeSingleIndexAdaptor<metric_t, self_t, DIM, IndexType>
      index_t;

  index_t* index;  //! The kd-tree index for the user to call its methods as
                   //! usual with any other FLANN index.

  KDTreeVectorOfVectorsAdaptor(){};

  /// Constructor: takes a const ref to the vector of vectors object with the
  /// data points
  KDTreeVectorOfVectorsAdaptor(const size_t /* dimensionality */,
                               const VectorOfVectorsType& mat,
                               const int leaf_max_size = 10)
      : m_data(mat) {
    init();
  }

  ~KDTreeVectorOfVectorsAdaptor() { delete index; }

  void init(const size_t /* dimensionality */, const VectorOfVectorsType& mat,
            const int leaf_max_size = 10) {
    m_data = mat;

    assert(mat.size() != 0 && mat[0].size() != 0);
    const size_t dims = mat[0].size();
    if (DIM > 0 && static_cast<int>(dims) != DIM)
      throw std::runtime_error(
          "Data set dimensionality does not match the 'DIM' template argument");
    index =
        new index_t(static_cast<int>(dims), *this /* adaptor */,
                    nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size));
    index->buildIndex();
  }

  VectorOfVectorsType m_data;

  /** Query for the \a num_closest closest points to a given point (entered as
   * query_point[0:dim-1]). Note that this is a short-cut method for
   * index->findNeighbors(). The user can also call index->... methods as
   * desired. \note nChecks_IGNORED is ignored but kept for compatibility with
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
  //   Return true if the BBOX was already computed by the class and returned in
  //   "bb" so it can be avoided to redo it again. Look at bb.size() to find out
  //   the expected dimensionality (e.g. 2 or 3 for point clouds)
  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /*bb*/) const {
    return false;
  }

  /** @} */

};  // end of KDTreeVectorOfVectorsAdaptor

// using ugu_kdtree_t =
//    nanoflann::KDTreeEigenMatrixAdaptor<Eigen::Matrix<float, 3, 1>>;

using my_vector_of_vectors_t = std::vector<Eigen::Vector3f>;

using ugu_kdtree_t =
    KDTreeVectorOfVectorsAdaptor<my_vector_of_vectors_t, float>;

std::tuple<std::vector<size_t>, std::vector<float>> QueryKdTree(
    const Eigen::Vector3f& p, const ugu_kdtree_t& index, int32_t nn_num);

class KDTreeCorrespFinder : public ugu::CorrespFinder {
 public:
  KDTreeCorrespFinder(){};
  KDTreeCorrespFinder(uint32_t nn_num) { SetNnNum(nn_num); };
  template <class... Args>
  static std::shared_ptr<KDTreeCorrespFinder> Create(Args... args) {
    return std::make_shared<KDTreeCorrespFinder>(args...);
  }
  bool Init(const std::vector<Eigen::Vector3f>& verts,
            const std::vector<Eigen::Vector3i>& verts_faces) override;
  ugu::Corresp Find(const Eigen::Vector3f& src_p,
                    const Eigen::Vector3f& src_n) const override;

  void SetNnNum(uint32_t nn_num);

 private:
  std::shared_ptr<ugu_kdtree_t> m_tree;
  uint32_t m_nn_num = 10;

  std::vector<Eigen::Vector3f> m_verts;
  std::vector<Eigen::Vector3i> m_verts_faces;
  std::vector<Eigen::Vector3f> m_face_centroids;
  // ax + by + cz + d = 0
  std::vector<Eigen::Vector4f> m_face_planes;
};
using KDTreeCorrespFinderPtr = std::shared_ptr<KDTreeCorrespFinder>;

#endif

}  // namespace ugu

#endif