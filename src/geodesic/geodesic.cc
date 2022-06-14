/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 *
 */

#include "ugu/geodesic/geodesic.h"

#include <queue>

#include "ugu/face_adjacency.h"

namespace {

struct DijkstraVertexInfo {
  int id = -1;
  // int prev = -1;
  double dist = std::numeric_limits<double>::max();
  DijkstraVertexInfo(int id_, double dist_) {
    id = id_;
    dist = dist_;
  }
  DijkstraVertexInfo(){};
  ~DijkstraVertexInfo(){};
};

bool operator<(const DijkstraVertexInfo& l, const DijkstraVertexInfo& r) {
  return l.dist > r.dist;
};

using DijkstraHeap =
    std::priority_queue<DijkstraVertexInfo, std::vector<DijkstraVertexInfo>>;

void DijkstraUpdate(DijkstraHeap& q, ugu::VertexAdjacency& vertex_adjacency,
                    Eigen::SparseMatrix<float>& edge_dists,
                    std::vector<double>& dists,
                    std::vector<int>& min_path_edges) {
  auto min_v = q.top();
  q.pop();
  // LOGI("%f\n", min_v.dist);
  const auto& connected_v_list = vertex_adjacency[min_v.id];
  for (const auto& v : connected_v_list) {
    double updated_dist = dists[min_v.id] + edge_dists.coeffRef(min_v.id, v);
    if (updated_dist < dists[v]) {
      dists[v] = updated_dist;
      min_path_edges[v] = min_v.id;
      q.push({v, updated_dist});
    }
  }
}

// Implementation of the following paper
// Kimmel, Ron, and James A. Sethian. "Computing geodesic paths on manifolds."
// Proceedings of the national academy of Sciences 95.15 (1998): 8431-8435.
// https://www.pnas.org/content/pnas/95/15/8431.full.pdf
void FmmUpdate(DijkstraHeap& q, ugu::VertexAdjacency& vertex_adjacency,
               const std::unordered_map<int, std::vector<int>>& v2f,
               const std::vector<Eigen::Vector3i>& faces,
               Eigen::SparseMatrix<float>& edge_dists,
               std::vector<double>& dists, std::vector<int>& min_path_edges) {
  auto min_v = q.top();
  q.pop();

  constexpr double EPS = 1e-20;
  // LOGI("%f\n", min_v.dist);
  constexpr double F = 1.0;
  constexpr double F2 = F * F;
  const auto& connected_v_list = vertex_adjacency[min_v.id];
  for (const auto& v : connected_v_list) {
    // Get face id
    std::vector<int> fids = v2f.at(v);
    std::vector<int> vid2_candidates;
    // Get 3rd vertex
    for (const auto& id : fids) {
      const auto& face = faces[id];
      for (int c = 0; c < 3; c++) {
        if (face[c] == min_v.id) {
          // Better code to search 3rd vertex...
          for (int cc = 0; cc < 3; cc++) {
            if (face[cc] != min_v.id && face[cc] != v) {
              vid2_candidates.push_back(face[cc]);
              break;
            }
          }
        }
      }
    }

    if (vid2_candidates.empty()) {
      // Skip if 3rd vertex has not been visited.
      // We need 2 visited vertices to update remaining 1 vertex
      continue;
    }

    double updated_dist = dists[v];

    // Test all candidates
    for (size_t i = 0; i < vid2_candidates.size(); i++) {
      int vid_2 = vid2_candidates[i];

      // Names are consitent with:
      // 4.1. A Construction for Acute Triangulations.
      // Fig. 5.

      int A = min_v.id;
      int B = vid_2;
      if (dists[B] < dists[A]) {
        std::swap(A, B);
      }
      int C = v;

      const double& a = edge_dists.coeffRef(B, C);
      const double& b = edge_dists.coeffRef(A, C);
      const double& c = edge_dists.coeffRef(A, B);
      const double a2 = a * a;
      const double b2 = b * b;
      const double c2 = c * c;
      const double cos_theta =
          (a2 + b2 - c2) / (2 * a * b);  // From Law of cosines
      const double theta = std::acos(cos_theta);

      const double u = dists[B] - dists[A];
      assert(0.0 < u);
      const double u2 = u * u;

      // Eq. [4] in the paper
      const double qa = a2 + b2 - 2 * a * b * std::cos(theta);
      const double qb = 2 * b * u * (a * std::cos(theta) - b);
      const double sin_theta = std::sin(theta);
      const double sin_theta2 = sin_theta * sin_theta;
      const double qc = b2 * (u2 - F2 * a2 * sin_theta2);

      if (std::abs(qa) < EPS) {
        //  continue;
      }
      double qD = qb * qb - 4 * qa * qc;
      if (qD < 0) {
        // I don't why sometimes Discriminator < 0...
        if (0.01 < std::abs(qD)) {
          // ugu::LOGE("something wrong...\n");
          // ugu::LOGE("%f %f %f %f %d \n", qD, dists[A], dists[B], dists[C],
          //          vid2_candidates.size());
          continue;
        } else {
          qD = 0.0;
        }
      }

      const auto t0 = (-qb + std::sqrt(qD)) / (2 * qa);
      const auto t1 = (-qb - std::sqrt(qD)) / (2 * qa);

      std::array<double, 2> answers = {t0, t1};
      // ugu::LOGI("%f %f %f\n", qD, t0, t1);

      const double a_cos_theta = a * cos_theta;
      const double a_inv_cos_theta =
          cos_theta < EPS ? std::numeric_limits<double>::max() : a / cos_theta;

      for (int ii = 0; ii < 2; ii++) {
        const double t = answers[ii];
        const double cond = b * (t - u) / t;

        bool is_satisfied = t > 0;
        is_satisfied &= (u < t);
        is_satisfied &= ((a_cos_theta < cond) && (cond < a_inv_cos_theta));

        if (is_satisfied) {
          updated_dist = std::min(updated_dist, t + dists[A]);
        } else {
          // The last term in the original paper is "c * F + dists[B]".
          // But I think the paper is wrong..."a * F + dists[B]" generates
          // visually better results...
          updated_dist = std::min(updated_dist,
                                  std::min(b * F + dists[A], a * F + dists[B]));
        }
      }
    }

    if (updated_dist < dists[v]) {
      dists[v] = updated_dist;
      min_path_edges[v] = min_v.id;
      q.push({v, updated_dist});
    }
  }
}

bool ComputeGeodesicDistanceDijkstraBase(const ugu::Mesh& mesh,
                                         std::vector<int> src_vids,
                                         Eigen::SparseMatrix<float>& edge_dists,
                                         std::vector<double>& dists,
                                         std::vector<int>& min_path_edges,
                                         bool is_fmm) {
  dists.clear();
  const auto num_vertices = mesh.vertices().size();
  dists.resize(num_vertices, std::numeric_limits<double>::max());
  for (const auto& src_vid : src_vids) {
    dists[src_vid] = 0.0;
  }

  min_path_edges.clear();
  min_path_edges.resize(num_vertices, -1);

  ugu::FaceAdjacency face_adjacency;
  face_adjacency.Init(static_cast<int>(num_vertices), mesh.vertex_indices());

  ugu::VertexAdjacency vertex_adjacency =
      face_adjacency.GenerateVertexAdjacency();

  // Compute edge distances
  edge_dists = Eigen::SparseMatrix<float>(num_vertices, num_vertices);
  {
    std::vector<Eigen::Triplet<float>> triplet_list;
    triplet_list.reserve(mesh.vertices().size() * 2);
    for (size_t i = 0; i < mesh.vertex_indices().size(); i++) {
      const Eigen::Vector3i& face = mesh.vertex_indices()[i];
      const Eigen::Vector3f& v0 = mesh.vertices()[face[0]];
      const Eigen::Vector3f& v1 = mesh.vertices()[face[1]];
      const Eigen::Vector3f& v2 = mesh.vertices()[face[2]];
      const float dist01 = (v0 - v1).norm();
      const float dist12 = (v1 - v2).norm();
      const float dist20 = (v2 - v0).norm();
      triplet_list.push_back(Eigen::Triplet<float>(face[0], face[1], dist01));
      triplet_list.push_back(Eigen::Triplet<float>(face[1], face[0], dist01));

      triplet_list.push_back(Eigen::Triplet<float>(face[1], face[2], dist12));
      triplet_list.push_back(Eigen::Triplet<float>(face[2], face[1], dist12));

      triplet_list.push_back(Eigen::Triplet<float>(face[2], face[0], dist20));
      triplet_list.push_back(Eigen::Triplet<float>(face[0], face[2], dist20));
    }

    edge_dists.setFromTriplets(triplet_list.begin(), triplet_list.end());
  }

  // Naive implementation of Dijkstra's algorithm with priority queue
  // https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
  DijkstraHeap q;

  // Initialization
  for (const auto& src_vid : src_vids) {
    q.push({src_vid, 0.0});
  }
  std::unordered_map<int, std::vector<int>> v2f;

  const auto& faces = mesh.vertex_indices();

  if (is_fmm) {
    v2f = ugu::GenerateVertex2FaceMap(faces, mesh.vertices().size());
  }

  // Main process
  while (!q.empty()) {
    if (is_fmm) {
      FmmUpdate(q, vertex_adjacency, v2f, faces, edge_dists, dists,
                min_path_edges);
    } else {
      DijkstraUpdate(q, vertex_adjacency, edge_dists, dists, min_path_edges);
    }
  }

  return true;
}

}  // namespace

namespace ugu {

bool ComputeGeodesicDistance(const Mesh& mesh, int src_vid,
                             Eigen::SparseMatrix<float>& edge_dists,
                             std::vector<double>& dists,
                             std::vector<int>& min_path_edges,
                             GeodesicComputeMethod method) {
  return ComputeGeodesicDistance(mesh, std::vector(1, src_vid), edge_dists,
                                 dists, min_path_edges, method);
}

bool ComputeGeodesicDistance(const Mesh& mesh, std::vector<int> src_vids,
                             Eigen::SparseMatrix<float>& edge_dists,
                             std::vector<double>& dists,
                             std::vector<int>& min_path_edges,
                             GeodesicComputeMethod method) {
  if (method == GeodesicComputeMethod::DIJKSTRA) {
    return ComputeGeodesicDistanceDijkstraBase(mesh, src_vids, edge_dists,
                                               dists, min_path_edges, false);
  } else if (method == GeodesicComputeMethod::FAST_MARCHING_METHOD) {
    return ComputeGeodesicDistanceDijkstraBase(mesh, src_vids, edge_dists,
                                               dists, min_path_edges, true);
  }

  return false;
}

}  // namespace ugu
