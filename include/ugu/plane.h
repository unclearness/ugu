/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#pragma once

#include "Eigen/Core"
#include "ugu/common.h"
#include "ugu/line.h"

namespace ugu {

UGU_FLOATING_POINT_ONLY_TEMPLATE
struct Plane {
  // nx + d = 0
  Eigen::Matrix<T, 3, 1> n;
  T d;

  Plane(const Eigen::Matrix<T, 3, 1>& n, T d) : n(n.normalized()), d(d) {}
  Plane(const Eigen::Matrix<T, 3, 1>& p0, const Eigen::Matrix<T, 3, 1>& p1,
        const Eigen::Matrix<T, 3, 1>& p2,
        const Eigen::Matrix<T, 3, 1>& n_hint = Eigen::Matrix<T, 3, 1>::Zero()) {
    Init(p0, p1, p2, n_hint);
  }

  bool IsNormalSide(const Eigen::Matrix<T, 3, 1>& p) const {
    return d > -(n.dot(p));
  }

  bool CalcIntersctionPoint(const Line3<T>& line, T& t,
                            Eigen::Matrix<T, 3, 1>& p) const {
    // https://risalc.info/src/line-plane-intersection-point.html
    T h = -d;
    t = (h - n.dot(line.a)) / (n.dot(line.d));
    p = line.a + t * line.d;
    return true;
  }

  float SignedDist(const Eigen::Matrix<T, 3, 1>& p) const {
    // (p + tn).dot(n) + d = 0
    T t = -d - p.dot(n);
    // positive: upper, negative, lower
    return t;
  }

  Eigen::Matrix<T, 3, 1> Project(const Eigen::Matrix<T, 3, 1>& p) const {
    // (p + tn).dot(n) + d = 0
    return p + SignedDist(p) * n;
  }

  void Init(
      const Eigen::Matrix<T, 3, 1>& p0, const Eigen::Matrix<T, 3, 1>& p1,
      const Eigen::Matrix<T, 3, 1>& p2,
      const Eigen::Matrix<T, 3, 1>& n_hint = Eigen::Matrix<T, 3, 1>::Zero()) {
    n = (p1 - p0).cross(p2 - p0).normalized();
    d = -n.dot(p0);

    if (T(0.01) < n_hint.norm()) {
      if (n_hint.normalized().dot(n) < 0) {
        n = T(-1) * n;
      }
    }
  }
};
using Planef = Plane<float>;
using Planed = Plane<double>;

struct PlaneEstimationStat {
  // Both
  std::vector<size_t> inliers;  // close & similar normal
  float inlier_ratio = -1.f;

  std::vector<size_t> outliers;  // others

  std::vector<size_t> uppers;  // upper points including inliers and others
  float upper_ratio =
      -1.f;  // upper points ratio, possibly for objects on the ground plane

  float area = -1.f;
  float area_ratio = -1.f;
};

struct PlaneEstimationResult {
  Planef estimation;
  PlaneEstimationStat stat;
};

struct EstimateGroundPlaneRansacParam {
  uint32_t max_iter = 100;
  float inlier_angle_th = radians(45.f);
  float inliner_dist_th = -1.f;
  int32_t seed = 0;
  size_t candidates_num = 3;
  Eigen::Vector3f normal_hint = Eigen::Vector3f::Zero();
};

bool EstimateGroundPlaneRansac(const std::vector<Eigen::Vector3f>& points,
                               const std::vector<Eigen::Vector3f>& normals,
                               const EstimateGroundPlaneRansacParam& param,
                               std::vector<PlaneEstimationResult>& candidates);

}  // namespace ugu
