/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <vector>

#define UGU_FACE_ADJACENCY_USE_SPARSE_MAT

#ifdef UGU_FACE_ADJACENCY_USE_SPARSE_MAT
#include "Eigen/SparseCore"
#else
#include "Eigen/Core"
#endif

namespace ugu {

struct FaceAdjacency {
  // https://qiita.com/shinjiogaki/items/d16abb018a843c09b8c8
  std::vector<Eigen::Vector3i> vertex_indices_;

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
    int& m0 = mat_.coeffRef(face[1], face[0]);
    if (0 < m0) {
      adjacent_face_ids->push_back(m0 - 1);
    }
    int& m1 = mat_.coeffRef(face[2], face[1]);
    if (0 < m1) {
      adjacent_face_ids->push_back(m1 - 1);
    }
    int& m2 = mat_.coeffRef(face[0], face[2]);
    if (0 < m2) {
      adjacent_face_ids->push_back(m2 - 1);
    }
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
};
}  // namespace ugu
