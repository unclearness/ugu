/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */
#ifdef UGU_USE_NANOFLANN

#include "ugu/textrans/texture_transfer.h"

#include "nanoflann.hpp"
#include "ugu/util/math_util.h"
#include "ugu/util/raster_util.h"

namespace {

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

auto ComputeFaceInfo(const std::vector<Eigen::Vector3f>& verts,
                     const std::vector<Eigen::Vector3i>& vert_faces) {
  std::vector<Eigen::Vector3f> face_centroids;
  // ax + by + cz + d = 0
  std::vector<Eigen::Vector4f> face_planes;

  for (const auto& face : vert_faces) {
    const Eigen::Vector3f& v0 = verts[face[0]];
    const Eigen::Vector3f& v1 = verts[face[1]];
    const Eigen::Vector3f& v2 = verts[face[2]];
    const auto centroid = (v0 + v1 + v2) / 3.0;
    face_centroids.emplace_back(centroid);

    Eigen::Vector3f vec10 = v1 - v0;
    Eigen::Vector3f vec20 = v2 - v0;
    Eigen::Vector3f n = vec10.cross(vec20).normalized();
    float d = -1.f * n.dot(v0);
    face_planes.emplace_back(n[0], n[1], n[2], d);
  }

  return std::make_tuple(face_centroids, face_planes);
}

// using ugu_kdtree_t =
//    nanoflann::KDTreeEigenMatrixAdaptor<Eigen::Matrix<float, 3, 1>>;

using my_vector_of_vectors_t = std::vector<Eigen::Vector3f>;

using ugu_kdtree_t =
    KDTreeVectorOfVectorsAdaptor<my_vector_of_vectors_t, float>;

auto QueryKdTree(const Eigen::Vector3f& p, const ugu_kdtree_t& index,
                 int32_t nn_num) {
  std::vector<std::pair<Eigen::Index, float>> ret_matches;
  std::vector<size_t> out_indices(nn_num);
  std::vector<float> out_distance_sq(nn_num);
  index.index->knnSearch(p.data(), nn_num, out_indices.data(),
                         out_distance_sq.data());
  return std::make_tuple(out_indices, out_distance_sq);
}

inline Eigen::Vector3f Extract3f(const Eigen::Vector4f& v) {
  return Eigen::Vector3f(v[0], v[1], v[2]);
}

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
  ugu_kdtree_t m_tree;
  uint32_t m_nn_num = 10;

  std::vector<Eigen::Vector3f> m_verts;
  std::vector<Eigen::Vector3i> m_verts_faces;
  std::vector<Eigen::Vector3f> m_face_centroids;
  // ax + by + cz + d = 0
  std::vector<Eigen::Vector4f> m_face_planes;
};
using KDTreeCorrespFinderPtr = std::shared_ptr<KDTreeCorrespFinder>;

bool KDTreeCorrespFinder::Init(
    const std::vector<Eigen::Vector3f>& verts,
    const std::vector<Eigen::Vector3i>& verts_faces) {
  auto [face_centroids, face_planes] = ComputeFaceInfo(verts, verts_faces);

  m_verts = verts;
  m_verts_faces = verts_faces;

  m_face_centroids = std::move(face_centroids);
  m_face_planes = std::move(face_planes);

  m_tree.init(verts_faces.size(), m_face_centroids, 10 /* max leaf */);

  return true;
}

void KDTreeCorrespFinder::SetNnNum(uint32_t nn_num) { m_nn_num = nn_num; }

ugu::Corresp KDTreeCorrespFinder::Find(const Eigen::Vector3f& src_p,
                                       const Eigen::Vector3f& src_n) const {
  (void)src_n;

  // Get the closest src face
  // Roughly get candidates.NN for face center points.
  auto [indices, distance_sq] = QueryKdTree(src_p, m_tree, m_nn_num);
  // Check point - plane distance and get the smallest
  float min_dist = std::numeric_limits<float>::max();
  float min_signed_dist = std::numeric_limits<float>::infinity();
  int32_t min_index = -1;
  Eigen::Vector3f min_bary(99.f, 99.f, 99.f);
  Eigen::Vector3f min_foot;
  for (const auto& index : indices) {
    const auto& vface = m_verts_faces[index];
    const auto& v0 = m_verts[vface[0]];
    const auto& v1 = m_verts[vface[1]];
    const auto& v2 = m_verts[vface[2]];

    // Case 1: foot of perpendicular line is inside of target triangle
    const auto& plane = m_face_planes[index];
    const Eigen::Vector3f normal = Extract3f(plane);
    // point-plane distance |ax'+by'+cz'+d|
    const float signed_dist = src_p.dot(normal) + plane[3];

    float dist = std::abs(signed_dist);
    Eigen::Vector3f foot = -signed_dist * normal + src_p;
    float foot_dist = foot.dot(normal) + plane[3];
    if (std::abs(foot_dist) > 0.0001f) {
      // ugu::LOGE("wrong dist %f %f\n", foot, foot_dist);
      throw std::runtime_error("wrong dist");
    }

    auto [isInside, bary] = ugu::IsPoint3dInsideTriangle(foot, v0, v1, v2);

    if (dist < min_dist && isInside) {
      min_dist = dist;
      min_signed_dist = signed_dist;
      min_index = static_cast<int32_t>(index);
      min_bary = bary;
      min_foot = foot;
    }
  }

  // Case 2: foot of perpendicular line is outside of the triangle
  if (min_index < 0) {
    for (const auto& index : indices) {
      const auto& vface = m_verts_faces[index];
      const auto& v0 = m_verts[vface[0]];
      const auto& v1 = m_verts[vface[1]];
      const auto& v2 = m_verts[vface[2]];

      const auto& plane = m_face_planes[index];
      const Eigen::Vector3f normal = Extract3f(plane);

      // Check distance to boundary line segments of triangle
      auto [ldist, lfoot] = ugu::PointTriangleDistance(src_p, v0, v1, v2);
      auto [lIsInside, lbary] = ugu::IsPoint3dInsideTriangle(lfoot, v0, v1, v2);
      if (!lIsInside) {
        // By numerical reason, sometimes becomes little over [0, 1]
        // So just clip
        lbary[0] = std::clamp(lbary[0], 0.f, 1.f);
        lbary[1] = std::clamp(lbary[1], 0.f, 1.f);
      }

      if (ldist < min_dist) {
        min_dist = ldist;
        // TODO: Add sign
        min_signed_dist = ldist;
        min_index = static_cast<int32_t>(index);
        min_bary = lbary;
        min_foot = lfoot;
      }
    }
  }

  // IMPORTANT: Without this line, unexpectable noises may appear...
  min_bary[2] = 1.f - min_bary[0] - min_bary[1];

  ugu::Corresp corresp;
  corresp.fid = min_index;
  corresp.p = min_foot;
  corresp.singed_dist = min_signed_dist;
  corresp.abs_dist = min_dist;
  corresp.uv = min_bary;

  return corresp;
}

}  // namespace

namespace ugu {

bool TexTransNoCorresp(const ugu::Image3f& src_tex,
                       const std::vector<Eigen::Vector2f>& src_uvs,
                       const std::vector<Eigen::Vector3i>& src_uv_faces,
                       const std::vector<Eigen::Vector3f>& src_verts,
                       const std::vector<Eigen::Vector3i>& src_verts_faces,
                       const std::vector<Eigen::Vector2f>& dst_uvs,
                       const std::vector<Eigen::Vector3i>& dst_uv_faces,
                       const std::vector<Eigen::Vector3f>& dst_verts,
                       const std::vector<Eigen::Vector3i>& dst_vert_faces,
                       int32_t dst_tex_h, int32_t dst_tex_w,
                       TexTransNoCorrespOutput& output, int32_t interp,
                       int32_t nn_num) {
  if (interp != InterpolationFlags::INTER_LINEAR &&
      interp != InterpolationFlags::INTER_NEAREST) {
    ugu::LOGE("interp is not supported\n");
    return false;
  }

  if (nn_num < 1) {
    ugu::LOGE("nn_num must be larger than 1\n");
    return false;
  }

  output.dst_tex = ugu::Image3f::zeros(dst_tex_h, dst_tex_w);
  output.dst_mask = ugu::Image1b::zeros(dst_tex_h, dst_tex_w);
  output.nn_pos_tex = ugu::Image3f::zeros(dst_tex_h, dst_tex_w);
  output.nn_bary_tex = ugu::Image3f::zeros(dst_tex_h, dst_tex_w);
  output.nn_fid_tex = ugu::Image1i::zeros(dst_tex_h, dst_tex_w);
  output.srcpos_tex = ugu::Image3f::zeros(dst_tex_h, dst_tex_w);

  float src_w = static_cast<float>(src_tex.cols);
  float src_h = static_cast<float>(src_tex.rows);

  CorrespFinderPtr coresp_finder = KDTreeCorrespFinder::Create(nn_num);
  coresp_finder->Init(src_verts, src_verts_faces);

  assert(dst_uv_faces.size() == dst_vert_faces.size());
#pragma omp parallel for
  for (int64_t df_idx = 0; df_idx < static_cast<int64_t>(dst_uv_faces.size());
       df_idx++) {
    const auto& duvface = dst_uv_faces[df_idx];
    const auto& dvface = dst_vert_faces[df_idx];

    // Get bounding box in dst tex
    const auto& duv0 = dst_uvs[duvface[0]];
    const auto& duv1 = dst_uvs[duvface[1]];
    const auto& duv2 = dst_uvs[duvface[2]];
    const auto& dv0 = dst_verts[dvface[0]];
    const auto& dv1 = dst_verts[dvface[1]];
    const auto& dv2 = dst_verts[dvface[2]];
    const auto bb_min_x =
        U2X(std::max({0.f, std::min({duv0[0], duv1[0], duv2[0]})}), dst_tex_w);
    const auto bb_max_x = U2X(std::min({static_cast<float>(dst_tex_w - 1),
                                        std::max({duv0[0], duv1[0], duv2[0]})}),
                              dst_tex_w);

    // Be careful how to get min/max of y. It is an inversion of v
    const auto bb_min_v = std::min({duv0[1], duv1[1], duv2[1]});
    const auto bb_max_v = std::max({duv0[1], duv1[1], duv2[1]});
    const auto bb_min_y = std::max(0.f, V2Y(bb_max_v, dst_tex_h));
    const auto bb_max_y =
        std::min(static_cast<float>(dst_tex_h - 1), V2Y(bb_min_v, dst_tex_h));
    // pixel-wise loop for the bb in dst tex
    float area = ugu::EdgeFunction(duv0, duv1, duv2);
    float inv_area = 1.f / area;

    for (int32_t bb_y = static_cast<int32_t>(bb_min_y);
         bb_y <= static_cast<int32_t>(std::ceil(bb_max_y)); bb_y++) {
      for (int32_t bb_x = static_cast<int32_t>(bb_min_x);
           bb_x <= static_cast<int32_t>(std::ceil(bb_max_x)); bb_x++) {
        Eigen::Vector2f pix_uv(X2U(static_cast<float>(bb_x), dst_tex_w),
                               Y2V(static_cast<float>(bb_y), dst_tex_h));

        float w0 = ugu::EdgeFunction(duv1, duv2, pix_uv) * inv_area;
        float w1 = ugu::EdgeFunction(duv2, duv0, pix_uv) * inv_area;
        float w2 = ugu::EdgeFunction(duv0, duv1, pix_uv) * inv_area;
        // Check if this pixel is on the dst triangle
        if (w0 < 0 || w1 < 0 || w2 < 0) {
          continue;
        }

        // Get corresponding position on the dst face
        Eigen::Vector3f dpos = w0 * dv0 + w1 * dv1 + w2 * dv2;

        // Get the closest src face info
        // auto [foot, min_signed_dist, min_dist, min_index, bary] =
        //   CalcClosestSurfaceInfo(tree, dpos, src_verts, src_verts_faces,
        //                          src_face_planes, nn_num);
        Corresp corresp = coresp_finder->Find(dpos, Eigen::Vector3f::Ones());

        if (corresp.fid < 0) {
          ugu::LOGE("min_index is None %d %d\n", bb_y, bb_x);
          continue;
        }
        output.nn_fid_tex.at<int>(bb_y, bb_x) = corresp.fid;
        output.nn_pos_tex.at<ugu::Vec3f>(bb_y, bb_x) =
            ugu::Vec3f({corresp.p[0], corresp.p[1], corresp.p[2]});
        output.nn_bary_tex.at<ugu::Vec3f>(bb_y, bb_x) =
            ugu::Vec3f({corresp.uv[0], corresp.uv[1], corresp.uv[2]});
        const auto& suvface = src_uv_faces[corresp.fid];
        const auto& suv0 = src_uvs[suvface[0]];
        const auto& suv1 = src_uvs[suvface[1]];
        const auto& suv2 = src_uvs[suvface[2]];

        Eigen::Vector2f suv =
            corresp.uv[0] * suv0 + corresp.uv[1] * suv1 + corresp.uv[2] * suv2;

        // Calc pixel pos in src tex
        float sx = std::clamp(U2X(suv[0], static_cast<int32_t>(src_w)), 0.f,
                              src_w - 1.f - 0.001f);
        float sy = std::clamp(V2Y(suv[1], static_cast<int32_t>(src_h)), 0.f,
                              src_h - 1.f - 0.001f);

        ugu::Vec3f& srcpos = output.srcpos_tex.at<ugu::Vec3f>(bb_y, bb_x);
        srcpos[0] = sx;
        srcpos[1] = sy;

        // Fetch and copy to dst tex
        ugu::Vec3f& src_color = output.dst_tex.at<ugu::Vec3f>(bb_y, bb_x);
        if (interp == InterpolationFlags::INTER_LINEAR) {
          src_color = ugu::BilinearInterpolation(sx, sy, src_tex);
        } else if (interp == InterpolationFlags::INTER_NEAREST) {
          src_color =
              src_tex.at<ugu::Vec3f>(static_cast<int32_t>(std::round(sy)),
                                     static_cast<int32_t>(std::round(sx)));
        } else {
          ugu::LOGE("interp is not supported\n");
        }

        output.dst_mask.at<uint8_t>(bb_y, bb_x) = 255;
      }
    }
  }

  return true;
}

bool TexTransNoCorresp(const ugu::Image3f& src_tex, const ugu::Mesh& src_mesh,
                       const ugu::Mesh& dst_mesh, int32_t dst_tex_h,
                       int32_t dst_tex_w, TexTransNoCorrespOutput& output,
                       int32_t interp, int32_t nn_num) {
  return TexTransNoCorresp(src_tex, src_mesh.uv(), src_mesh.uv_indices(),
                           src_mesh.vertices(), src_mesh.vertex_indices(),
                           dst_mesh.uv(), dst_mesh.uv_indices(),
                           dst_mesh.vertices(), dst_mesh.vertex_indices(),
                           dst_tex_h, dst_tex_w, output, interp, nn_num);
}

}  // namespace ugu

#endif
