/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/registration/nonrigid.h"

// #include <Eigen/CholmodSupport>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>

#include "ugu/timer.h"
#include "ugu/util/geom_util.h"
#include "ugu/util/math_util.h"

namespace {
using namespace ugu;
}  // namespace

namespace ugu {

NonRigidIcp::NonRigidIcp() {
  m_corresp_finder = KDTreeCorrespFinder::Create();
  m_corresp_finder->SetNnNum(100);

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

void NonRigidIcp::SetSrcLandmakrVertexIds(
    const std::vector<int>& src_landmark_indices) {
  m_src_landmark_indices = src_landmark_indices;
}

void NonRigidIcp::SetDstLandmakrVertexIds(
    const std::vector<int>& dst_landmark_indices) {
  m_dst_landmark_indices = dst_landmark_indices;
}

bool NonRigidIcp::Init(bool check_self_itersection, float angle_cos_th,
                       bool check_geometry_border) {
  if (m_src == nullptr || m_dst == nullptr || m_corresp_finder == nullptr) {
    LOGE("Data was not set\n");
    return false;
  }

  m_check_self_itersection = check_self_itersection;
  m_angle_cos_th = angle_cos_th;
  m_check_geometry_border = check_geometry_border;

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
  if (m_check_geometry_border) {
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
        if (m_dst_border_edges.find(shared_fid) == m_dst_border_edges.end()) {
          m_dst_border_edges.insert(std::make_pair(
              shared_fid, std::vector<std::pair<int, int>>{edge}));
        } else {
          m_dst_border_edges[shared_fid].push_back(edge);
        }
      }
    }
  }

  // Init anisotropic scale to [0, 1] cube. Center translation was untouched.
  // In the orignal paper, [-1, 1] cube.
  m_norm2org_scale =
      m_src_stats.bb_max - m_src_stats.bb_min +
      Eigen::Vector3f::Constant(std::numeric_limits<float>::epsilon());
  m_org2norm_scale = m_norm2org_scale.cwiseInverse();

  // Scale mesh to make parameters scale-indepdendent
  m_src_norm->Scale(m_org2norm_scale);
  m_dst_norm->Scale(m_org2norm_scale);

  // Init deformed meshes with intial meshess
  m_src_norm_deformed = Mesh::Create(*m_src_norm);
  m_src_deformed = Mesh::Create(*m_src);

  // Init correspondences
  m_corresp.resize(m_src_norm->vertices().size());
  m_target.resize(m_corresp.size());
  m_weights_per_node.resize(m_corresp.size(), 1.0);

  // Set landmark positions
  if (!m_src_landmark_indices.empty()) {
    m_src_landmark_positions.resize(m_src_landmark_indices.size());
    for (size_t i = 0; i < m_src_landmark_indices.size(); i++) {
      int idx = m_src_landmark_indices[i];
      m_src_landmark_positions[i] = m_src_norm_deformed->vertices()[idx];
    }
    m_dst_landmark_positions.resize(m_dst_landmark_indices.size());
    for (size_t i = 0; i < m_dst_landmark_indices.size(); i++) {
      int idx = m_dst_landmark_indices[i];
      m_dst_landmark_positions[i] = m_dst_norm->vertices()[idx];
    }
  }

  // Init KD Tree
  bool ret = m_corresp_finder->Init(m_dst_norm->vertices(),
                                    m_dst_norm->vertex_indices());
  // Init BVH
  if (m_check_self_itersection) {
    m_bvh->SetData(m_src_norm_deformed->vertices(),
                   m_src_norm_deformed->vertex_indices());
    ret &= m_bvh->Build();
  }

  return ret;
}

bool NonRigidIcp::FindCorrespondences() {
  const std::vector<Eigen::Vector3f>& current = m_src_norm_deformed->vertices();
  const std::vector<Eigen::Vector3f>& current_n =
      m_src_norm_deformed->normals();
  auto point2plane_corresp_func = [&](size_t idx) {
#if 0
    Corresp c = m_corresp_finder->Find(
        current[idx], current_n[idx],  // Eigen::Vector3f::Zero(),
        CorrespFinderMode::kMinAngleCosDist);
#else
    auto corresps = m_corresp_finder->FindAll(
        current[idx], current_n[idx],  // Eigen::Vector3f::Zero(),
        CorrespFinderMode::kMinAngleCosDist);

    Corresp c = corresps[0];
    for (const auto& corresp : corresps) {
      if (std::cos(corresp.angle) < m_angle_cos_th) {
        continue;
      }
      c = corresp;
      break;
    }
#endif
    m_target[idx] = c.p;
    m_corresp[idx].resize(1);
    m_corresp[idx][0].dist = c.abs_dist;
    m_corresp[idx][0].index = size_t(~0);

#if 1
    // Validate correspondence
    if (!ValidateCorrespondence(idx, c)) {
      // Set 0 weight to invalid correspondences
      m_weights_per_node[idx] = 0.0;
    }
#endif
  };

  // Init weight as 1
  std::fill(m_weights_per_node.begin(), m_weights_per_node.end(), 1.0);

  // Find coressponding points
  parallel_for(0u, current.size(), point2plane_corresp_func, m_num_theads);

  return true;
}

bool NonRigidIcp::Registrate(double alpha, double beta, double gamma) {
  Eigen::Index n = static_cast<Eigen::Index>(m_src->vertices().size());
  Eigen::Index m = static_cast<Eigen::Index>(m_edges.size());
  Eigen::Index l = 0;
  if (m_src_landmark_positions.size() > 0) {
    if (m_src_landmark_positions.size() == m_dst_landmark_positions.size()) {
      l = static_cast<Eigen::Index>(m_src_landmark_positions.size());
    } else {
      LOGW("Landmark size mismatch. Will not be used...\n");
    }
  }

  Eigen::MatrixX3d X(4 * n, 3);
  X.setZero();

  int iter = 0;
  constexpr int max_iter = 10;  // TODO: better terminate criteria
  constexpr double min_frobenius_norm_diff = 2.0;
  Timer<> timer, iter_timer;
  bool verbose = true;
  LogLevel org_loglevel = get_log_level();
  if (!verbose) {
    set_log_level(LogLevel::kWarning);
  }

  if (m_check_self_itersection) {
    // BVH construction is costly.
    // So once per each stiffness
    m_bvh->SetData(m_src_norm_deformed->vertices(),
                   m_src_norm_deformed->vertex_indices());
    m_bvh->Build();
  }

  while (iter < max_iter) {
    iter_timer.Start();
    LOGI("Registrate(): iter %d\n", iter);

    timer.Start();
    FindCorrespondences();
    timer.End();
    LOGI("FindCorrespondences(): %f ms\n", timer.elapsed_msec());

    timer.Start();
    Eigen::SparseMatrix<double> A(4 * m + n + l, 4 * n);

    // 1.alpha_M_G
    std::vector<Eigen::Triplet<double>> alpha_M_G;
    for (Eigen::Index i = 0; i < m; ++i) {
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
        alpha_M_G.push_back(
            Eigen::Triplet<double>(i * 4 + j, a * 4 + j, alpha));
        alpha_M_G.push_back(
            Eigen::Triplet<double>(i * 4 + j, b * 4 + j, -alpha));
      }

      alpha_M_G.push_back(
          Eigen::Triplet<double>(i * 4 + 3, a * 4 + 3, alpha * gamma));
      alpha_M_G.push_back(
          Eigen::Triplet<double>(i * 4 + 3, b * 4 + 3, -alpha * gamma));
    }

    // 2.W_D
    std::vector<Eigen::Triplet<double>> W_D;
    for (Eigen::Index i = 0; i < n; ++i) {
      const Eigen::Vector3f& vtx = m_src_norm_deformed->vertices()[i];

      double weight = m_weights_per_node[i];
      // if (weight == 0) weight = 1;

      for (int j = 0; j < 3; ++j) {
        W_D.push_back(Eigen::Triplet<double>(
            4 * m + i, i * 4 + j, weight * static_cast<double>(vtx[j])));
      }

      W_D.push_back(Eigen::Triplet<double>(4 * m + i, i * 4 + 3, weight * 1.0));
    }

    // 3.beta_D_L
    std::vector<Eigen::Triplet<double>> beta_D_L;
    for (Eigen::Index i = 0; i < l; i++) {
      for (int j = 0; j < 3; j++) {
        beta_D_L.push_back(Eigen::Triplet<double>(
            4 * m + n + i, m_src_landmark_indices[i] * 4 + j,
            beta * m_src_landmark_positions[i](j)));
      }

      beta_D_L.push_back(Eigen::Triplet<double>(
          4 * m + n + i, m_src_landmark_indices[i] * 4 + 3, beta));
    }

    std::vector<Eigen::Triplet<double>> _A = alpha_M_G;
    _A.insert(_A.end(), W_D.begin(), W_D.end());
    _A.insert(_A.end(), beta_D_L.begin(), beta_D_L.end());
    A.setFromTriplets(_A.begin(), _A.end());

    // for the B
    Eigen::MatrixX3d B = Eigen::MatrixX3d::Zero(4 * m + n + l, 3);
    for (Eigen::Index i = 0; i < n; ++i) {
#if 0
      int idx = 0;
      trimesh::point xyz;

      double weight = m_weights_per_node[i];
      if (weight != 0) {
        idx = m_soft_corres[i].second;
        xyz = m_dst->vertices[idx];
      } else {
        weight = 1;
        idx = m_soft_corres[i].first;
        xyz = m_src->vertices[idx];
      }
#endif
      double weight = m_weights_per_node[i];
      const auto& target_pos = m_target[i];
      for (int j = 0; j < 3; j++) {
        B(4 * m + i, j) = weight * target_pos[j];
      }
    }

    for (int i = 0; i < l; i++) {
      for (int j = 0; j < 3; j++) {
        B(4 * m + n + i, j) = beta * m_dst_landmark_positions[i](j);
      }
    }

    Eigen::SparseMatrix<double> AtA =
        Eigen::SparseMatrix<double>(A.transpose()) * A;
    Eigen::MatrixX3d AtB = Eigen::SparseMatrix<double>(A.transpose()) * B;

    timer.End();
    LOGI("SparseMatrix Preparation: %f ms\n", timer.elapsed_msec());

    timer.Start();
    // Eigen::ConjugateGradient< Eigen::SparseMatrix<double> > solver;
    Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> solver;
    // Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double> > solver;
    solver.compute(AtA);

    timer.End();
    LOGI("solver.compute(): %f ms\n", timer.elapsed_msec());

    timer.Start();
    Eigen::MatrixX3d TmpX(4 * n, 3);
    TmpX = X;
    X = solver.solve(AtB);
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

    m_src_landmark_positions.resize(m_src_landmark_indices.size());
    for (size_t i = 0; i < m_src_landmark_indices.size(); i++) {
      int idx = m_src_landmark_indices[i];
      m_src_landmark_positions[i] = m_src_norm_deformed->vertices()[idx];
    }

    timer.End();
    LOGI("Update data: %f ms\n", timer.elapsed_msec());

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
  if (m_check_geometry_border && !m_dst_border_fids.empty()) {
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

  // 2) the angle between the normals of the meshes at Xivi and ui is
  // larger than a fixed threshold, or
  const Eigen::Vector3f& src_n = m_src_norm_deformed->normals()[src_idx];
  const Eigen::Vector3f& dst_n = m_dst_norm->face_normals()[corresp.fid];
  float dot = src_n.dot(dst_n);
  if (dot < m_angle_cos_th) {
    return false;
  }

  //  3) the line segment Xivi to ui intersects the deformed template."
  // -> To avoid self intersection
  if (m_check_self_itersection) {
    Ray ray;
    ray.org = m_src_norm_deformed->vertices()[src_idx];
    ray.dir = (corresp.p - ray.org).normalized();
    std::vector<IntersectResult> results = m_bvh->Intersect(ray);
    double org_dist = (corresp.p - ray.org).norm();
    if (!results.empty()) {
      for (const auto& r : results) {
        // Intersection less than original distance -> self intersection by
        // another face
        if (r.t < org_dist) {
          // std::cout << r.t << " " << org_dist << std::endl;
          return false;
        }
      }
    }
  }

  return true;
}

}  // namespace ugu