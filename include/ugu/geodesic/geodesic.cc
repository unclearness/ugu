/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 *
 */

#include "geodesic.h"

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

bool ComputeGeodesicDistanceDijkstra(const ugu::Mesh& mesh, int src_vid,
                                     Eigen::SparseMatrix<float>& edge_dists,
                                     std::vector<double>& dists,
                                     std::vector<int>& min_path_edges) {
  dists.clear();
  const auto num_vertices = mesh.vertices().size();
  dists.resize(num_vertices, std::numeric_limits<double>::max());
  dists[src_vid] = 0.0;

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
    for (auto i = 0; i < mesh.vertex_indices().size(); i++) {
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
  q.push({src_vid, 0.0});

  // Main process
  while (!q.empty()) {
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

  return true;
}

}  // namespace

namespace ugu {

bool ComputeGeodesicDistance(const Mesh& mesh, int src_vid,
                             Eigen::SparseMatrix<float>& edge_dists,
                             std::vector<double>& dists,
                             std::vector<int>& min_path_edges,
                             GeodesicComputeMethod method) {
  if (method == GeodesicComputeMethod::DIJKSTRA) {
    return ComputeGeodesicDistanceDijkstra(mesh, src_vid, edge_dists, dists,
                                           min_path_edges);
  }

  return false;
}

}  // namespace ugu
