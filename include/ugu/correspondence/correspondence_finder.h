/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/accel/kdtree.h"
#include "ugu/common.h"

namespace ugu {

struct Corresp {
  int32_t fid = -1;
  Eigen::Vector2f uv = Eigen::Vector2f::Zero();
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
  std::unique_ptr<KdTree<Eigen::Vector3f>> m_tree;
  uint32_t m_nn_num = 10;

  std::vector<Eigen::Vector3f> m_verts;
  std::vector<Eigen::Vector3i> m_verts_faces;
  std::vector<Eigen::Vector3f> m_face_centroids;
  // ax + by + cz + d = 0
  std::vector<Eigen::Vector4f> m_face_planes;
};
using KDTreeCorrespFinderPtr = std::shared_ptr<KDTreeCorrespFinder>;

}  // namespace ugu
