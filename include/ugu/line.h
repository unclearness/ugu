/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#pragma once

#include <limits>
#include <vector>

#include "ugu/camera.h"

namespace ugu {

UGU_FLOATING_POINT_ONLY_TEMPLATE
struct Line3 {
  // p = dt + a
  Eigen::Matrix<T, 3, 1> a, d;

  void Set(const Eigen::Matrix<T, 3, 1>& p0, const Eigen::Matrix<T, 3, 1>& p1) {
    d = (p1 - p0).normalized();
    a = p0;
  }
  const Eigen::Matrix<T, 3, 1> Sample(T t) const { return d * t + a; }
  T Dist(const Line3<T>& l, T* t, Eigen::Matrix<T, 3, 1>* p0, T* l_t,
         Eigen::Matrix<T, 3, 1>* p1) const {
    // http://obelisk2.hatenablog.com/entry/20101228/1293521247
    auto& AB = l.a - this->a;
    auto ABm = AB.dot(this->d);
    auto ABn = AB.dot(l.d);
    auto mn = l.d.dot(this->d);

    if (std::abs(mn - 1.0) < std::numeric_limits<T>::epsilon()) {
      *t = 0;
      *l_t = 0;
    } else {
      *t = (ABm - ABn * mn) / (1.0 - mn * mn);
      *l_t = (ABm * mn - ABn) / (1.0 - mn * mn);
    }

    *p0 = this->Sample(*t);
    *p1 = l.Sample(*l_t);

    return (*p0 - *p1).norm();
  }
};

UGU_FLOATING_POINT_ONLY_TEMPLATE
inline const Line3<T> operator*(const Eigen::Transform<T, 3, Eigen::Affine>& l,
                                const Line3<T>& r) {
  auto out = Line3<T>();
  // apply rotation to direction
  out.d = l.rotation() * r.d;

  // apply translation to position
  out.a = l.translation() + r.a;
  return out;
}

using Line3d = Line3<double>;
using Line3f = Line3<float>;

UGU_FLOATING_POINT_ONLY_TEMPLATE
struct Line2 {
  // y = dx + a
  T a, d;

  Eigen::Matrix<T, 2, 1> p0, p1;
  T GetX(T y) const { return (y - a) / d; }
  T GetY(T x) const { return d * x + a; }
  void Set(const Eigen::Matrix<T, 2, 1>& p_a,
           const Eigen::Matrix<T, 2, 1>& p_b) {
    Set(p_a.x(), p_a.y(), p_b.x(), p_b.y());
  }
  void Set(T x0, T y0, T x1, T y1) {
    p0.x() = x0;
    p0.y() = y0;
    p1.x() = x1;
    p1.y() = y1;

    auto denom = x1 - x0;

    constexpr double EPS = 0.0000000001;
    if (std::abs(denom) < EPS) {
      d = T(999999999);
    } else {
      d = (y1 - y0) / denom;
    }
    a = y0 - d * x0;
  }
};

using Line2d = Line2<double>;
using Line2f = Line2<float>;

// PinholeCamera with line projection
class LinePinholeCamera : public PinholeCamera {
 public:
  using PinholeCamera::Project;
  void Project(const Line3d& camera_l, Line2d* image_l) const;
};

inline void LinePinholeCamera::Project(const Line3d& camera_l,
                                       Line2d* image_l) const {
  // ToDo: Use Plucker line coordinates

  // Get a line parameter large enough
  float sample_t = static_cast<float>(std::max(width_, height_));

  // Sample 2 points on 3D
  Eigen::Vector3f p0_3d = camera_l.Sample(-sample_t).cast<float>();
  Eigen::Vector3f p1_3d = camera_l.Sample(sample_t).cast<float>();

  // Project the 2 points to 2D
  Eigen::Vector2f p0_2d, p1_2d;
  Project(p0_3d, &p0_2d);
  Project(p1_3d, &p1_2d);

  // Calc 2d line equation
  image_l->Set(p0_2d.cast<double>(), p1_2d.cast<double>());
}

UGU_FLOATING_POINT_ONLY_TEMPLATE
std::tuple<bool, Eigen::Matrix<T, 3, 1>> FindExactLineCrossingPoint(
    const Line3<T>& l0, const Line3<T>& l1, T det_th = T(0.0000000001)) {
  Eigen::Matrix<T, 3, 1> point;

  double det = l0.d.x() * (-l1.d.y()) - (-l1.d.x() * l0.d.y());
  if (std::abs(det) < det_th) {
    // If det is zero, do not have the exact crossing point
    return {false, point};
  }

  double inv_det = 1.0 / det;
  double inv00 = -l1.d.y() * inv_det;
  double inv01 = l1.d.x() * inv_det;

  double b0 = l1.a.x() - l0.a.x();
  double b1 = l1.a.y() - l0.a.y();

  double l0_t = b0 * inv00 + b1 * inv01;

  // no need to calculate below.
  // double inv10 = -l0.d.y() * inv_det;
  // double inv11 = l0.d.x() * inv_det;
  // double l1_t = b0 * inv10 + b1 * inv11;
  // auto p1 = l1.Sample(l1_t);

  point = l0.Sample(T(l0_t));

  return {true, point};
}

UGU_FLOATING_POINT_ONLY_TEMPLATE
std::tuple<Eigen::Matrix<T, 3, 1>, std::vector<T>>
FindBestLineCrossingPointLeastSquares(const std::vector<Line3<T>>& lines) {
  // http://tercel-sakuragaoka.blogspot.com/2012/01/vs.html
  std::vector<T> errors;
  Eigen::Matrix<T, 3, 1> p;

  Eigen::MatrixXd lhs = lines.size() * Eigen::MatrixXd::Identity(
                                           lines.size() + 3, lines.size() + 3);
  for (auto i = 0; i < lines.size(); i++) {
    lhs(i, i) = lines[i].d.dot(lines[i].d);
  }

  for (auto i = 0; i < lines.size(); i++) {
    for (auto j = lines.size(); j < lines.size() + 3; j++) {
      lhs(i, j) = -lines[i].d[j - lines.size()];
    }
  }

  for (auto j = 0; j < lines.size(); j++) {
    for (auto i = lines.size(); i < lines.size() + 3; i++) {
      lhs(i, j) = -lines[j].d[i - lines.size()];
    }
  }

  Eigen::VectorXd rhs = Eigen::VectorXd(lines.size() + 3);
  for (auto i = 0; i < lines.size(); i++) {
    rhs(i) = -lines[i].d.dot(lines[i].a);
  }
  for (auto i = lines.size(); i < lines.size() + 3; i++) {
    double sum = 0.0;
    for (auto j = 0; j < lines.size(); j++) {
      sum += lines[j].a[i - lines.size()];
    }
    rhs(i) = sum;
  }

  Eigen::VectorXd ans =
      lhs.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);

  p.x() = ans.cast<T>()[lines.size()];
  p.y() = ans.cast<T>()[lines.size() + 1];
  p.z() = ans.cast<T>()[lines.size() + 2];

  for (auto i = 0; i < lines.size(); i++) {
    auto a = (p.cast<T>() - lines[i].a);
    auto perpendiculer = lines[i].a + a.dot(lines[i].d) * lines[i].d;
    auto error = (p.cast<T>() - perpendiculer).norm();

    errors.push_back(error);
  }

  return {p, errors};
}

UGU_FLOATING_POINT_ONLY_TEMPLATE
struct Plane {
  // nx + d = 0
  Eigen::Matrix<T, 3, 1> n;
  T d;

  Plane(const Eigen::Matrix<T, 3, 1>& n, T d) : n(n.normalized()), d(d) {}

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
};
using Planef = Plane<float>;
using Planed = Plane<double>;

}  // namespace ugu
