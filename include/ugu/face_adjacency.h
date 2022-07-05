/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define UGU_FACE_ADJACENCY_USE_SPARSE_MAT

#ifdef UGU_FACE_ADJACENCY_USE_SPARSE_MAT
#include "Eigen/SparseCore"
#else
#include "Eigen/Core"
#endif

namespace ugu {

using VertexAdjacency = std::vector<std::set<int>>;

class FaceAdjacency {
 private:
  // https://qiita.com/shinjiogaki/items/d16abb018a843c09b8c8
  std::vector<Eigen::Vector3i> vertex_indices_;

 public:
#ifdef UGU_FACE_ADJACENCY_USE_SPARSE_MAT
  // Sparse matrix version
  // - Much less memory
  // - Faster Init by setFromTriplets()
  // - Slower GetAdjacentFaces since random access is not O(1)
  Eigen::SparseMatrix<int> mat_;  // Stores face_id + 1 to use SparseMatrix ()

  void Init(int num_vertices,
            const std::vector<Eigen::Vector3i>& vertex_indices) {
    vertex_indices_.clear();
    std::copy(vertex_indices.begin(), vertex_indices.end(),
              std::back_inserter(vertex_indices_));

    mat_ = Eigen::SparseMatrix<int>(num_vertices, num_vertices);

#if 1
    std::vector<Eigen::Triplet<int>> triplet_list;
    triplet_list.reserve(vertex_indices_.size() * 3);
    for (int i = 0; i < static_cast<int>(vertex_indices_.size()); i++) {
      const Eigen::Vector3i& face = vertex_indices_[i];
      triplet_list.push_back(Eigen::Triplet<int>(face[0], face[1], i + 1));
      triplet_list.push_back(Eigen::Triplet<int>(face[1], face[2], i + 1));
      triplet_list.push_back(Eigen::Triplet<int>(face[2], face[0], i + 1));
    }

    mat_.setFromTriplets(triplet_list.begin(), triplet_list.end());

#else
    // insert per elemenet is sloooooooooooooow
    for (int i = 0; i < static_cast<int>(vertex_indices_.size()); i++) {
      const Eigen::Vector3i& face = vertex_indices_[i];
      // i + 1 for SparseMatrix
      mat_.insert(face[0], face[1]) = i + 1;
      mat_.insert(face[1], face[2]) = i + 1;
      mat_.insert(face[2], face[0]) = i + 1;
    }
#endif
  }

#else
  // Dense matrix version
  // - Much more memory
  // - Slower Init by inserting per element
  // - Faster GetAdjacentFaces since random access is O(1)

  Eigen::MatrixXi mat_;  // Stores face_id + 1 to use SparseMatrix ()

  void Init(int num_vertices,
            const std::vector<Eigen::Vector3i>& vertex_indices) {
    vertex_indices_.clear();
    std::copy(vertex_indices.begin(), vertex_indices.end(),
              std::back_inserter(vertex_indices_));

#if 1

    // TODO: initialization with zero is slow
    mat_ = Eigen::MatrixXi::Zero(num_vertices, num_vertices);

    // mat_ = Eigen::MatrixXi(num_vertices, num_vertices); // this line is fast
    // mat_.setZero(); // slow

    // insert per element is slow
    for (int i = 0; i < static_cast<int>(vertex_indices_.size()); i++) {
      const Eigen::Vector3i& face = vertex_indices_[i];
      // i + 1 for SparseMatrix compatibility
      mat_.coeffRef(face[0], face[1]) = i + 1;
      mat_.coeffRef(face[1], face[2]) = i + 1;
      mat_.coeffRef(face[2], face[0]) = i + 1;
    }
#else

    // from SparseMatrix version
    // not fast
    Eigen::SparseMatrix<int> spmat =
        Eigen::SparseMatrix<int>(num_vertices, num_vertices);
    std::vector<Eigen::Triplet<int>> triplet_list;
    triplet_list.reserve(vertex_indices_.size() * 3);
    for (int i = 0; i < static_cast<int>(vertex_indices_.size()); i++) {
      const Eigen::Vector3i& face = vertex_indices_[i];
      triplet_list.push_back(Eigen::Triplet<int>(face[0], face[1], i + 1));
      triplet_list.push_back(Eigen::Triplet<int>(face[1], face[2], i + 1));
      triplet_list.push_back(Eigen::Triplet<int>(face[2], face[0], i + 1));
    }
    spmat.setFromTriplets(triplet_list.begin(), triplet_list.end());

    mat_ = Eigen::MatrixXi(spmat);
#endif
  }

#endif

  void GetAdjacentFaces(int face_id, std::vector<int>* adjacent_face_ids) {
    const Eigen::Vector3i& face = vertex_indices_[face_id];
    adjacent_face_ids->clear();
    const int& m0 = mat_.coeffRef(face[1], face[0]);
    if (0 < m0) {
      adjacent_face_ids->push_back(m0 - 1);
    }
    const int& m1 = mat_.coeffRef(face[2], face[1]);
    if (0 < m1) {
      adjacent_face_ids->push_back(m1 - 1);
    }
    const int& m2 = mat_.coeffRef(face[0], face[2]);
    if (0 < m2) {
      adjacent_face_ids->push_back(m2 - 1);
    }
  }

  void GetAdjacentFacesByVertex(int face_id, const VertexAdjacency& va,
                                std::unordered_map<int, std::vector<int>> v2f,
                                std::vector<int>* adjacent_face_ids) {
    std::unordered_set<int> connecting_vids;
    for (int i = 0; i < 3; i++) {
      const auto& vids = va[vertex_indices_[face_id][i]];
      for (const auto& vid : vids) {
        connecting_vids.insert(vid);
      }
    }

    std::unordered_set<int> connecting;
    const auto& face = vertex_indices_[face_id];
    for (const auto& vid : connecting_vids) {
      const auto& fids = v2f[vid];
      for (const auto& fid : fids) {
        //connecting.insert(fid);
        const auto& face_ = vertex_indices_[fid];
   bool ok = false;
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            if (face[i] == face_[j]) {
              ok = true;
              break;
            }
          }
        }
        if (ok) {
          connecting.insert(fid);
        }

      }
    }
    connecting.erase(face_id);

    adjacent_face_ids->clear();
    std::copy(connecting.begin(), connecting.end(),
              std::back_inserter(*adjacent_face_ids));

#if 0
           std::vector<int> one_ring;
    GetAdjacentFaces(face_id, &one_ring);
    std::unordered_set<int> two_ring;
    for (const auto& fid : one_ring) {
      std::vector<int> tmp;
      GetAdjacentFaces(fid, &tmp);
      for (const auto& t : tmp) {
        two_ring.insert(t);
      }
    }
    two_ring.erase(face_id);

    const auto& face = vertex_indices_[face_id];
    std::unordered_set<int> connecting;

    for (const auto& fid : two_ring) {
      const auto& face_ = vertex_indices_[fid];
      bool ok = false;
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          if (face[i] == face_[j]) {
            ok = true;
            break;
          }
        }
      }
      if (ok) {
        connecting.insert(fid);
      }
    }

    adjacent_face_ids->clear();
    std::copy(connecting.begin(), connecting.end(),
              std::back_inserter(*adjacent_face_ids));
#endif  // 0
  }

  bool RemoveFace(int face_id) {
    const Eigen::Vector3i& face = vertex_indices_[face_id];
    int& m0 = mat_.coeffRef(face[0], face[1]);
    int& m1 = mat_.coeffRef(face[1], face[2]);
    int& m2 = mat_.coeffRef(face[2], face[0]);

    if (m0 == 0 && m1 == 0 && m2 == 0) {
      return false;
    }

    m0 = 0;
    m1 = 0;
    m2 = 0;

    return true;
  }

  bool OverwriteFace(int face_id, const Eigen::Vector3i& face) {
    int& m0 = mat_.coeffRef(face[0], face[1]);
    int& m1 = mat_.coeffRef(face[1], face[2]);
    int& m2 = mat_.coeffRef(face[2], face[0]);

    m0 = face_id + 1;
    m1 = face_id + 1;
    m2 = face_id + 1;
    return true;
  }

  bool HasBoundaryEdge(int face_id,
                       std::vector<std::pair<int, int>>* boundary_edges) {
    const Eigen::Vector3i& face = vertex_indices_[face_id];
    boundary_edges->clear();
    bool ret = false;
    const int& m0 = mat_.coeffRef(face[1], face[0]);
    if (0 == m0) {
      boundary_edges->push_back({face[1], face[0]});
      ret = true;
    }
    const int& m1 = mat_.coeffRef(face[2], face[1]);
    if (0 == m1) {
      boundary_edges->push_back({face[2], face[1]});
      ret = true;
    }
    const int& m2 = mat_.coeffRef(face[0], face[2]);
    if (0 == m2) {
      boundary_edges->push_back({face[0], face[2]});
      ret = true;
    }
    return ret;
  }

  std::tuple<std::vector<std::pair<int, int>>, std::unordered_set<int>>
  GetBoundaryEdges() {
    std::vector<std::pair<int, int>> boundary_edges;

    for (size_t face_id = 0; face_id < vertex_indices_.size(); face_id++) {
      const Eigen::Vector3i& face = vertex_indices_[face_id];
      // boundary_edges should not be shared with other faces.
      // So it should be unique, no need to check duplication
      const int& m0 = mat_.coeffRef(face[1], face[0]);
      if (0 == m0) {
        boundary_edges.push_back({face[1], face[0]});
      }
      const int& m1 = mat_.coeffRef(face[2], face[1]);
      if (0 == m1) {
        boundary_edges.push_back({face[2], face[1]});
      }
      const int& m2 = mat_.coeffRef(face[0], face[2]);
      if (0 == m2) {
        boundary_edges.push_back({face[0], face[2]});
      }
    }

    std::unordered_set<int> boundary_vertex_ids;
    for (const auto& p : boundary_edges) {
      boundary_vertex_ids.insert(p.first);
      boundary_vertex_ids.insert(p.second);
    }

    return {boundary_edges, boundary_vertex_ids};
  }

  VertexAdjacency GenerateVertexAdjacency() {
    const auto vertex_num = mat_.rows();
    VertexAdjacency vertex_adjacency(vertex_num);

    for (size_t face_id = 0; face_id < vertex_indices_.size(); face_id++) {
      const Eigen::Vector3i& face = vertex_indices_[face_id];

      vertex_adjacency[face[0]].insert(face[1]);
      vertex_adjacency[face[1]].insert(face[0]);

      vertex_adjacency[face[2]].insert(face[1]);
      vertex_adjacency[face[1]].insert(face[2]);

      vertex_adjacency[face[0]].insert(face[2]);
      vertex_adjacency[face[2]].insert(face[0]);
    }

    return vertex_adjacency;
  }

  bool Empty() const { return vertex_indices_.empty(); }
};

inline std::unordered_map<int, std::vector<int>> GenerateVertex2FaceMap(
    const std::vector<Eigen::Vector3i>& vertex_indices, size_t num_vertices) {
  std::unordered_map<int, std::vector<int>> v2f;
  for (size_t i = 0; i < num_vertices; i++) {
    // v2f.insert(static_cast<int>(i), {});
  }

  for (size_t i = 0; i < vertex_indices.size(); i++) {
    const auto& index = vertex_indices[i];
    v2f[index[0]].push_back(static_cast<int>(i));
    v2f[index[1]].push_back(static_cast<int>(i));
    v2f[index[2]].push_back(static_cast<int>(i));
  }

  for (auto& p : v2f) {
    std::sort(p.second.begin(), p.second.end());
  }

  return v2f;
}

inline std::pair<std::vector<std::vector<int32_t>>, std::vector<int32_t>>
GenerateVertex2UvMap(const std::vector<Eigen::Vector3i>& vertex_indices,
                     size_t num_v,
                     const std::vector<Eigen::Vector3i>& uv_indices,
                     size_t num_uv) {
  std::vector<std::vector<int32_t>> vid2uvid;
  std::vector<int32_t> uvid2vid;

  vid2uvid.resize(num_v);
  uvid2vid.resize(num_uv);
  for (size_t i = 0; i < vertex_indices.size(); i++) {
    const auto& f = vertex_indices[i];
    const auto& uv_f = uv_indices[i];
    for (int j = 0; j < 3; j++) {
      int32_t vid = f[j];
      int32_t uv_id = uv_f[j];
      vid2uvid[vid].push_back(uv_id);
      uvid2vid[uv_id] = vid;
    }
  }
  for (size_t i = 0; i < num_v; i++) {
    std::sort(vid2uvid[i].begin(), vid2uvid[i].end());
    auto res = std::unique(vid2uvid[i].begin(), vid2uvid[i].end());
    vid2uvid[i].erase(res, vid2uvid[i].end());
  }

  return {vid2uvid, uvid2vid};
}

}  // namespace ugu
