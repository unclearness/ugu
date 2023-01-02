/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/registration/rigid.h"

namespace ugu {

// An implementation of the following paper
// Amberg, Brian, Sami Romdhani, and Thomas Vetter. "Optimal step nonrigid ICP
// algorithms for surface registration." 2007 IEEE conference on computer vision
// and pattern recognition. IEEE, 2007.

class NonRigidIcp {
 public:
  NonRigidIcp();
  ~NonRigidIcp();
  void SetThreadNum(int thread_num);
  void SetSrc(const Mesh& src,
              const Eigen::Affine3f& transform = Eigen::Affine3f::Identity());
  void SetDst(const Mesh& dst);

  void SetSrcLandmakrVertexIds(const std::vector<int>& src_landmark_indices);
  void SetDstLandmakrVertexIds(const std::vector<int>& dst_landmark_indices);

  bool Init();  // Initialize KDTree etc.

  bool FindCorrespondences();
  bool Registrate(double alpha = 1000.0, double beta = 10.0,
                  double gamma = 1.0);

  MeshPtr GetDeformedSrc() const;

 private:
  MeshPtr m_src_org = nullptr;
  Eigen::Affine3f m_transform = Eigen::Affine3f::Identity();
  MeshPtr m_src = nullptr;
  MeshPtr m_src_norm = nullptr;
  MeshPtr m_src_norm_deformed = nullptr;
  MeshPtr m_src_deformed = nullptr;
  MeshStats m_src_stats;
  MeshPtr m_dst = nullptr;
  MeshPtr m_dst_norm = nullptr;

  Eigen::Vector3f m_norm2org_scale, m_org2norm_scale;

  KDTreeCorrespFinderPtr m_corresp_finder = nullptr;
  std::vector<KdTreeSearchResults> m_corresp;
  std::vector<Eigen::Vector3f> m_target;

  std::vector<std::pair<int, int> > m_edges;
  std::vector<double> m_weights_per_node;

  int m_num_theads = -1;

  std::vector<int> m_src_landmark_indices;
  std::vector<Eigen::Vector3f> m_src_landmark_positions;
  std::vector<int> m_dst_landmark_indices;
  std::vector<Eigen::Vector3f> m_dst_landmark_positions;
};

}  // namespace ugu