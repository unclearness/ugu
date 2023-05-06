/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/accel/bvh.h"
#include "ugu/point.h"
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

  void SetSrcLandmarks(const std::vector<PointOnFace>& src_landmarks,
                       const std::vector<double>& betas = {});
  void SetDstLandmarkPositions(
      const std::vector<Eigen::Vector3f>& dst_landmark_positions);

  bool Init(bool check_self_itersection = false, float angle_rad_th = 0.65f,
            bool dst_check_geometry_border = false,
            bool src_check_geometry_border = false);  // Initialize KDTree etc.

  bool FindCorrespondences();
  bool Registrate(double alpha = 1000.0, double gamma = 1.0, int max_iter = 10,
                  double min_frobenius_norm_diff = 2.0);

  MeshPtr GetDeformedSrc() const;

 private:
  bool ValidateCorrespondence(size_t src_idx, const Corresp& corresp) const;

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

  std::vector<std::pair<int, int>> m_edges;
  std::vector<double> m_weights_per_node;

  int m_num_theads = -1;

  std::vector<PointOnFace> m_src_landmarks;
  std::vector<Eigen::Vector3f> m_dst_landmark_positions,
      m_dst_landmark_positions_norm;
  std::vector<double> m_betas;

  uint32_t m_corresp_nn_num = 10u;

  float m_angle_rad_th = 0.65f;

  bool m_dst_check_geometry_border = false;
  std::unordered_set<int> m_dst_border_fids;
  // [fid] -> {edge_pair, ...}
  std::unordered_map<int, std::vector<std::pair<int, int>>> m_dst_border_edges;

  bool m_src_check_geometry_border = false;
  std::unordered_set<int> m_src_border_vids;

  bool m_check_self_itersection = false;
  BvhPtr<Eigen::Vector3f, Eigen::Vector3i> m_bvh = nullptr;

  bool m_rescale = true;
};

}  // namespace ugu