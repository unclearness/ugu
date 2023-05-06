/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/registration/nonrigid.h"

#ifdef _WIN32
#pragma warning(push, UGU_EIGEN_WARNING_LEVEL)
#endif

// Tried Eigen::CholmodSimplicialLDLT with SuiteSparse but slower than
// Eigen::SimplicialLDLT
#if 0
#if __has_include(<cholmod.h>)
#include <Eigen/CholmodSupport>
#define UGU_CHOLMOD_SUPPORT
#endif
#endif

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#ifdef _WIN32
#pragma warning(pop)
#endif

#include "ugu/image_io.h"
#include "ugu/timer.h"
#include "ugu/util/geom_util.h"
#include "ugu/util/math_util.h"

// #define UGU_NCIP_IGNORE_WEIGHT_ZERO

namespace {
using namespace ugu;
}  // namespace

namespace ugu {

NonRigidIcp::NonRigidIcp() {
  m_corresp_finder = KDTreeCorrespFinder::Create();

  m_bvh = GetDefaultBvh<Eigen::Vector3f, Eigen::Vector3i>();
};
NonRigidIcp::~NonRigidIcp(){};

void NonRigidIcp::SetThreadNum(int thread_num) { m_num_theads = thread_num; }

void NonRigidIcp::SetSrc(const Mesh& src, const Eigen::Affine3f& transform) {
  m_src_org = Mesh::Create(src);
  m_src = Mesh::Create(src);

  m_transform = transform;
  m_src->Transform(transform);
  m_src->CalcStats();
  m_src_stats = m_src->stats();

  m_src_norm = Mesh::Create(*m_src);
  m_src_norm_deformed = Mesh::Create(*m_src);
}

void NonRigidIcp::SetDst(const Mesh& dst) {
  m_dst = Mesh::Create(dst);
  m_dst_norm = Mesh::Create(dst);
}

void NonRigidIcp::SetSrcLandmarks(const std::vector<PointOnFace>& src_landmarks,
                                  const std::vector<double>& betas) {
  m_src_landmarks = src_landmarks;
  if (betas.size() == src_landmarks.size()) {
    m_betas = betas;
  } else {
    m_betas.resize(src_landmarks.size(), 1.0);
  }
}

void NonRigidIcp::SetDstLandmarkPositions(
    const std::vector<Eigen::Vector3f>& dst_landmark_positions) {
  m_dst_landmark_positions = dst_landmark_positions;
}

bool NonRigidIcp::Init(bool check_self_itersection, float angle_rad_th,
                       bool dst_check_geometry_border,
                       bool src_check_geometry_border) {
  if (m_src == nullptr || m_dst == nullptr || m_corresp_finder == nullptr) {
    LOGE("Data was not set\n");
    return false;
  }

  m_check_self_itersection = check_self_itersection;
  m_angle_rad_th = angle_rad_th;
  m_dst_check_geometry_border = dst_check_geometry_border;
  m_src_check_geometry_border = src_check_geometry_border;

  // Construct edges
  m_edges.clear();
  for (size_t i = 0; i < m_src->vertex_indices().size(); i++) {
    const Eigen::Vector3i& face = m_src->vertex_indices()[i];
    int v0 = face[0];
    int v1 = face[1];
    int v2 = face[2];
    m_edges.push_back(std::pair<int, int>(v0, v1));
    m_edges.push_back(std::pair<int, int>(v1, v2));
    m_edges.push_back(std::pair<int, int>(v2, v0));
  }

  // TODO: Messy...
  m_dst_border_fids.clear();
  m_dst_border_edges.clear();
  if (m_dst_check_geometry_border) {
    auto [boundary_edges_list, boundary_vertex_ids_list] =
        FindBoundaryLoops(m_dst->vertex_indices(),
                          static_cast<int32_t>(m_dst->vertices().size()));
    std::unordered_map<int, std::vector<int>> v2f =
        GenerateVertex2FaceMap(m_dst->vertex_indices(),
                               static_cast<int32_t>(m_dst->vertices().size()));
    for (const auto& boudary : boundary_edges_list) {
      for (const auto& edge : boudary) {
        auto fl0 = v2f[edge.first];
        auto fl1 = v2f[edge.second];
        // Find shared face
        std::unordered_set<int> shared_faces;
        for (const auto& f0 : fl0) {
          for (const auto& f1 : fl1) {
            if (f0 == f1) {
              shared_faces.insert(f0);
            }
          }
        }

        // On boundaries, must be 1
        assert(shared_faces.size() == 1);
        int shared_fid = *shared_faces.begin();

        m_dst_border_fids.insert(shared_fid);

        if (m_dst_border_edges.find(shared_fid) == m_dst_border_edges.end()) {
          m_dst_border_edges.insert(std::make_pair(
              shared_fid, std::vector<std::pair<int, int>>{edge}));
        } else {
          m_dst_border_edges[shared_fid].push_back(edge);
        }
      }
    }
  }

  m_src_border_vids.clear();
  if (m_src_check_geometry_border) {
    auto [boundary_edges_list, boundary_vertex_ids_list] =
        FindBoundaryLoops(m_src->vertex_indices(),
                          static_cast<int32_t>(m_src->vertices().size()));
    for (const auto& boundary_vertex_ids : boundary_vertex_ids_list) {
      for (const auto& vid : boundary_vertex_ids) {
        m_src_border_vids.insert(vid);
      }
    }
  }

  if (m_rescale) {
    // Init anisotropic scale to [0, 1] cube. Center translation was untouched.
    // In the orignal paper, [-1, 1] cube.
    m_norm2org_scale =
        m_src_stats.bb_max - m_src_stats.bb_min +
        Eigen::Vector3f::Constant(std::numeric_limits<float>::epsilon());
  } else {
    m_norm2org_scale.setOnes();
  }

  m_org2norm_scale = m_norm2org_scale.cwiseInverse();

  // Scale mesh to make parameters scale-indepdendent
  m_src_norm->Scale(m_org2norm_scale);
  m_dst_norm->Scale(m_org2norm_scale);

  // Init deformed meshes with intial meshess
  m_src_norm_deformed = Mesh::Create(*m_src_norm);
  m_src_deformed = Mesh::Create(*m_src);

  m_dst_landmark_positions_norm.clear();
  std::transform(m_dst_landmark_positions.begin(),
                 m_dst_landmark_positions.end(),
                 std::back_inserter(m_dst_landmark_positions_norm),
                 [&](const Eigen::Vector3f& v) {
                   return v.cwiseProduct(m_org2norm_scale);
                 });

  // Init correspondences
  m_corresp.resize(m_src_norm->vertices().size());
  m_target.resize(m_corresp.size());
  m_weights_per_node.resize(m_corresp.size(), 1.0);

  // Init KD Tree
  bool ret = m_corresp_finder->Init(m_dst_norm->vertices(),
                                    m_dst_norm->vertex_indices());
  // Init BVH
  if (m_check_self_itersection) {
    // m_bvh->SetData(m_src_norm_deformed->vertices(),
    //                m_src_norm_deformed->vertex_indices());
    m_bvh->SetData(m_dst_norm->vertices(), m_dst_norm->vertex_indices());
    ret &= m_bvh->Build();
  }

  return ret;
}

bool NonRigidIcp::FindCorrespondences() {
  const std::vector<Eigen::Vector3f>& current = m_src_norm_deformed->vertices();
  const std::vector<Eigen::Vector3f>& current_n =
      m_src_norm_deformed->normals();
  auto point2plane_corresp_func = [&](size_t idx) {
    auto corresps = m_corresp_finder->FindKnn(current[idx], m_corresp_nn_num);

    Corresp c = corresps[0];
    for (const auto& corresp : corresps) {
      float vangle =
          std::acos(std::clamp(corresp.n.dot(current_n[idx]), -1.f, 1.f));
      if (vangle > m_angle_rad_th) {
        continue;
      }
      c = corresp;
      break;
    }
    m_target[idx] = c.p;
    m_corresp[idx].resize(1);
    m_corresp[idx][0].dist = c.abs_dist;
    m_corresp[idx][0].index = size_t(~0);

    // Validate correspondence
    if (!ValidateCorrespondence(idx, c)) {
      // Set 0 weight to invalid correspondences
      m_weights_per_node[idx] = 0.0;
    }
  };

  // Init weight as 1
  std::fill(m_weights_per_node.begin(), m_weights_per_node.end(), 1.0);

  // Find coressponding points
  parallel_for(0u, current.size(), point2plane_corresp_func, m_num_theads);

  return true;
}

bool NonRigidIcp::Registrate(double alpha, double gamma, int max_iter,
                             double min_frobenius_norm_diff) {
  using IndexType = Eigen::SparseMatrix<double>::StorageIndex;
  IndexType n = static_cast<IndexType>(m_src->vertices().size());
  IndexType m = static_cast<IndexType>(m_edges.size());
  IndexType l = 0;
  if (m_src_landmarks.size() > 0) {
    if (m_src_landmarks.size() == m_betas.size() &&
        m_src_landmarks.size() == m_dst_landmark_positions.size()) {
      l = static_cast<IndexType>(m_src_landmarks.size());
    } else {
      LOGW("Landmark size mismatch. Will not be used...\n");
    }
  }

  Eigen::MatrixX3d X(4 * n, 3);
  X.setZero();

  int iter = 0;
  Timer<> timer, iter_timer;
  bool verbose = true;
  LogLevel org_loglevel = get_log_level();
  if (!verbose) {
    set_log_level(LogLevel::kWarning);
  }

  m_dst_landmark_positions_norm.clear();
  std::transform(m_dst_landmark_positions.begin(),
                 m_dst_landmark_positions.end(),
                 std::back_inserter(m_dst_landmark_positions_norm),
                 [&](const Eigen::Vector3f& v) {
                   return v.cwiseProduct(m_org2norm_scale);
                 });

  Timer<> timer_mat;
  timer_mat.Start();
  // 1.alpha_M_G
  std::vector<Eigen::Triplet<double>> alpha_M_G;
  for (IndexType i = 0; i < m; ++i) {
    int a = m_edges[i].first;
    int b = m_edges[i].second;

#if 0
      double edge_len = (m_src_norm_deformed->vertices()[a] -
                         m_src_norm_deformed->vertices()[b])
                            .norm();

      for (int j = 0; j < 3; j++) {
        alpha_M_G.push_back(
            Eigen::Triplet<double>(i * 4 + j, a * 4 + j, alpha * edge_len));
        alpha_M_G.push_back(
            Eigen::Triplet<double>(i * 4 + j, b * 4 + j, -alpha * edge_len));
      }
#endif

    for (int j = 0; j < 3; j++) {
      alpha_M_G.push_back(Eigen::Triplet<double>(i * 4 + j, a * 4 + j, alpha));
      alpha_M_G.push_back(Eigen::Triplet<double>(i * 4 + j, b * 4 + j, -alpha));
    }

    alpha_M_G.push_back(
        Eigen::Triplet<double>(i * 4 + 3, a * 4 + 3, alpha * gamma));
    alpha_M_G.push_back(
        Eigen::Triplet<double>(i * 4 + 3, b * 4 + 3, -alpha * gamma));
  }
  timer_mat.End();
  LOGI("1.alpha_M_G: %f ms\n", timer_mat.elapsed_msec());

  while (iter < max_iter) {
    iter_timer.Start();
    LOGI("Registrate(): iter %d\n", iter);

    timer.Start();
    FindCorrespondences();
    timer.End();
    LOGI("FindCorrespondences(): %f ms\n", timer.elapsed_msec());

    timer.Start();
    Eigen::SparseMatrix<double> A(4 * m + n + l, 4 * n);

    // 2.W_D
    timer_mat.Start();
    std::vector<Eigen::Triplet<double>> W_D;
    for (IndexType i = 0; i < n; ++i) {
      const Eigen::Vector3f& vtx = m_src_norm_deformed->vertices()[i];

      double weight = m_weights_per_node[i];
#ifdef UGU_NCIP_IGNORE_WEIGHT_ZERO
      // If weight is 0, set the same position to target and reset weight to 1
      if (weight == 0.0) {
        weight = 1.0;
      }
#endif

      for (int j = 0; j < 3; ++j) {
        W_D.push_back(Eigen::Triplet<double>(
            4 * m + i, i * 4 + j, weight * static_cast<double>(vtx[j])));
      }

      W_D.push_back(Eigen::Triplet<double>(4 * m + i, i * 4 + 3, weight));
    }
    timer_mat.End();
    LOGI("2.W_D: %f ms\n", timer_mat.elapsed_msec());

    // 3.beta_D_L
    timer_mat.Start();
    std::vector<Eigen::Triplet<double>> beta_D_L;
    for (IndexType i = 0; i < l; i++) {
      const PointOnFace& pof = m_src_landmarks[i];
      const Eigen::Vector3f bary(pof.u, pof.v, 1.f - pof.u - pof.v);
      const Eigen::Vector3i& indices =
          m_src_norm_deformed->vertex_indices()[pof.fid];
      double beta = m_betas[i];
      for (int k = 0; k < 3; k++) {
        const int& index = indices[k];
        // Barycentric interpolation
        for (int j = 0; j < 3; j++) {
          beta_D_L.push_back(Eigen::Triplet<double>(
              4 * m + n + i, index * 4 + j,
              beta * bary[k] * m_src_norm_deformed->vertices()[index](j)));
        }
        // Homogeneous w
        beta_D_L.push_back(
            Eigen::Triplet<double>(4 * m + n + i, index * 4 + 3, beta));
      }
    }
    timer_mat.End();
    LOGI("3.beta_D_L: %f ms\n", timer_mat.elapsed_msec());

    timer_mat.Start();
    std::vector<Eigen::Triplet<double>> _A = alpha_M_G;
    _A.insert(_A.end(), W_D.begin(), W_D.end());
    _A.insert(_A.end(), beta_D_L.begin(), beta_D_L.end());
    A.setFromTriplets(_A.begin(), _A.end());
    timer_mat.End();
    LOGI("A.setFromTriplets: %f ms\n", timer_mat.elapsed_msec());

    // for the B
    timer_mat.Start();
    Eigen::MatrixX3d B = Eigen::MatrixX3d::Zero(4 * m + n + l, 3);
    for (IndexType i = 0; i < n; ++i) {
      double weight = m_weights_per_node[i];
      auto& target_pos = m_target[i];
#ifdef UGU_NCIP_IGNORE_WEIGHT_ZERO
      // If weight is 0, set the same position to target
      if (weight == 0.0) {
        weight = 1.0;
        target_pos = m_src_norm_deformed->vertices()[i];
      }
#endif
      for (int j = 0; j < 3; j++) {
        B(4 * m + i, j) = weight * target_pos[j];
      }
    }
    for (int i = 0; i < l; i++) {
      double beta = m_betas[i];
      for (int j = 0; j < 3; j++) {
        // Eq.(12) in the paper seems wrong. beta should be multiplied to B as
        // well as A
        B(4 * m + n + i, j) = beta * m_dst_landmark_positions_norm[i](j);
      }
    }
    timer_mat.End();
    LOGI("B: %f ms\n", timer_mat.elapsed_msec());

    timer_mat.Start();
    // Eigen::SparseMatrix<double> AtA = A.transpose() * A;
    // Eigen::MatrixX3d AtB = A.transpose() * B;
    timer_mat.End();
    LOGI("matmul: %f ms\n", timer_mat.elapsed_msec());

    timer.End();
    LOGI("SparseMatrix Preparation: %f ms\n", timer.elapsed_msec());

    timer.Start();

#if 1
    // Closed-form
#ifdef UGU_CHOLMOD_SUPPORT
    Eigen::CholmodSimplicialLDLT<Eigen::SparseMatrix<double>> solver;
#else
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
#endif

#else
    // Iterative
    Eigen::initParallel();
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double>>
        solver;
    solver.setTolerance(1e-4);
    solver.setMaxIterations(10);
#endif
    solver.compute(A.transpose() * A);

    timer.End();
    LOGI("solver.compute(): %f ms\n", timer.elapsed_msec());

    timer.Start();
    Eigen::MatrixX3d TmpX(4 * n, 3);
    TmpX = X;
    X = solver.solve(A.transpose() * B);
    timer.End();
    LOGI("solve(): %f ms\n", timer.elapsed_msec());

    timer.Start();
    Eigen::Matrix3Xd Xt = X.transpose();
    std::vector<Eigen::Vector3f> updated_vertices;
    for (Eigen::Index i = 0; i < n; ++i) {
      const Eigen::Vector3f& vtx = m_src_norm_deformed->vertices()[i];

      Eigen::Vector4d point(vtx[0], vtx[1], vtx[2], 1.0);
      Eigen::Vector3d result = Xt.block<3, 4>(0, 4 * i) * point;

      updated_vertices.push_back(result.cast<float>());
    }
    m_src_norm_deformed->set_vertices(updated_vertices);
    m_src_norm_deformed->CalcNormal();
    m_src_deformed->set_vertices(updated_vertices);
    m_src_deformed->Scale(m_norm2org_scale);
    m_src_deformed->CalcNormal();

    timer.End();
    LOGI("Update data: %f ms\n", timer.elapsed_msec());

    timer.Start();
    auto diff = (A * X - B);
    auto l2 = diff.cwiseProduct(diff);
    auto stiffness_block = l2.block(0, 0, 4 * m, 3);
    double stiffness_reg = stiffness_block.sum();
    auto point2surface_block = l2.block(4 * m, 0, n, 3);
    double point2surface_loss = point2surface_block.sum();
    auto landmark_block = l2.block(4 * m + n, 0, l, 3);
    double landmark_loss = landmark_block.sum();
    LOGI("Src verts to dst mesh: %lf\n", point2surface_loss);
    LOGI("Landmark loss        : %lf\n", landmark_loss);
    LOGI("Stiffness reg.       : %lf\n", stiffness_reg);
    timer.End();
    LOGI("Calc loss: %f ms\n", timer.elapsed_msec());

    double frobenius_norm_diff = (X - TmpX).norm();
    LOGI("frobenius_norm_diff %f \n", frobenius_norm_diff);

    iter_timer.End();
    LOGI("Registrate(): iter %d %f ms\n", iter, iter_timer.elapsed_msec());

    iter++;
    if (frobenius_norm_diff < min_frobenius_norm_diff) {
      break;
    }
  }
  if (!verbose) {
    set_log_level(org_loglevel);
  }

  return true;
}

MeshPtr NonRigidIcp::GetDeformedSrc() const { return m_src_deformed; }

bool NonRigidIcp::ValidateCorrespondence(size_t src_idx,
                                         const Corresp& corresp) const {
  // 4.4. Missing data and robustness

  // "A correspondence (Xivi, ui) is dropped if 1) ui lies on a border of the
  // target mesh,
  constexpr float eps = 1e-10f;  // std::numeric_limits<float>::epsilon();
  if (m_dst_check_geometry_border && !m_dst_border_fids.empty()) {
    if (m_dst_border_fids.find(corresp.fid) != m_dst_border_fids.end()) {
      for (const std::pair<int, int>& e : m_dst_border_edges.at(corresp.fid)) {
        // Compute distance from a corresponding point to a border edge
        auto [dist, prj_p] =
            PointLineSegmentDistance(corresp.p, m_dst_norm->vertices()[e.first],
                                     m_dst_norm->vertices()[e.second]);
        // If dist is too small, the corresponding point is on the edge.
        if (dist < eps) {
          return false;
        }
      }
    }
  }

  // ORIGINAL
  // Additionaly check src boundary
  if (m_src_check_geometry_border) {
    if (m_src_border_vids.find(static_cast<int>(src_idx)) !=
        m_src_border_vids.end()) {
      return false;
    }
  }

  // 2) the angle between the normals of the meshes at Xivi and ui is
  // larger than a fixed threshold, or
  const Eigen::Vector3f& src_n = m_src_norm_deformed->normals()[src_idx];
  const Eigen::Vector3f& dst_n = corresp.n;
  float dot = std::clamp(src_n.dot(dst_n), -1.f, 1.f);
  if (std::acos(dot) > m_angle_rad_th) {
    return false;
  }

  //  3) the line segment Xivi to ui intersects the deformed template."
  // -> To avoid self intersection
  if (m_check_self_itersection) {
    Ray ray;
    ray.org = m_src_norm_deformed->vertices()[src_idx];
    ray.dir = (corresp.p - ray.org).normalized();
    std::vector<IntersectResult> results = m_bvh->Intersect(ray, false);
    if (!results.empty()) {
      const auto& r = results[0];
      if (r.fid != static_cast<uint32_t>(corresp.fid)) {
        return false;
      }
    }
  }

  return true;
}

}  // namespace ugu