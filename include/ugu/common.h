/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <cassert>
#include <iomanip>
#include <limits>
#include <string>

#include "Eigen/Geometry"
#include "ugu/log.h"

namespace ugu {

// TODO: definition in header may be invalid.
static const double pi = 3.14159265358979323846;

// Interpolation method in texture uv space
enum class ColorInterpolation {
  kNn = 0,       // Nearest Neigbor
  kBilinear = 1  // Bilinear interpolation
};

struct Line3d {
  // p = dt + a
  Eigen::Vector3d a, d;

  void Set(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1);
  const Eigen::Vector3d Sample(double t) const;
  const Line3d operator*(const Eigen::Affine3d& T) const;
};
inline void Line3d::Set(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1) {
  d = (p1 - p0).normalized();
  a = p0;
}
inline const Eigen::Vector3d Line3d::Sample(double t) const { return d * t + a; }
inline const Line3d Line3d::operator*(const Eigen::Affine3d& T) const {
  Line3d rfs = *this;
  // apply rotation to direction
  rfs.d = T.rotation() * this->d;

  // apply translation to position
  rfs.a = T.translation() + this->a;

  return rfs;
}

struct Line2d {
  // y = dx + a
  double a, d;

  Eigen::Vector2d p0, p1;
  void Set(const Eigen::Vector2d& p0, const Eigen::Vector2d& p1);
  void Set(double x0, double y0, double x1, double y1);
};
inline void Line2d::Set(const Eigen::Vector2d& p_a,
                        const Eigen::Vector2d& p_b) {
  Set(p_a.x(), p_a.y(), p_b.x(), p_b.y());
}
inline void Line2d::Set(double x0, double y0, double x1, double y1) {
  p0.x() = x0;
  p0.y() = y0;
  p1.x() = x1;
  p1.y() = y1;

  auto denom = x1 - x0;

  constexpr double EPS = 0.0000000001;
  if (std::abs(denom) < EPS) {
    d = 999999999;
  } else {
    d = (y1 - y0) / denom;
  }
  a = y0 - d * x0;
}

// borrow from glm
// radians
template <typename genType>
genType radians(genType degrees) {
  // "'radians' only accept floating-point input"
  assert(std::numeric_limits<genType>::is_iec559);

  return degrees * static_cast<genType>(0.01745329251994329576923690768489);
}

// degrees
template <typename genType>
genType degrees(genType radians) {
  // "'degrees' only accept floating-point input"
  assert(std::numeric_limits<genType>::is_iec559);

  return radians * static_cast<genType>(57.295779513082320876798154814105);
}

// https://stackoverflow.com/questions/13768423/setting-up-projection-model-and-view-transformations-for-vertex-shader-in-eige
template <typename T>
void c2w(const Eigen::Matrix<T, 3, 1>& position,
         const Eigen::Matrix<T, 3, 1>& target, const Eigen::Matrix<T, 3, 1>& up,
         Eigen::Matrix<T, 3, 3>* R) {
  assert(std::numeric_limits<T>::is_iec559);

  R->col(2) = (target - position).normalized();
  R->col(0) = R->col(2).cross(up).normalized();
  R->col(1) = R->col(2).cross(R->col(0));
}

template <typename genType>
void c2w(const Eigen::Matrix<genType, 3, 1>& position,
         const Eigen::Matrix<genType, 3, 1>& target,
         const Eigen::Matrix<genType, 3, 1>& up,
         Eigen::Matrix<genType, 4, 4>* T) {
  assert(std::numeric_limits<genType>::is_iec559);

  *T = Eigen::Matrix<genType, 4, 4>::Identity();

  Eigen::Matrix<genType, 3, 3> R;
  c2w(position, target, up, &R);

  T->topLeftCorner(3, 3) = R;
  T->topRightCorner(3, 1) = position;
}

template <typename T>
std::string zfill(const T& val, int num = 5) {
  std::ostringstream sout;
  sout << std::setfill('0') << std::setw(num) << val;
  return sout.str();
}

// https://docs.opencv.org/4.3.0/d9/d0c/group__calib3d.html
inline void DistortPixelOpencv(float* u, float* v, float fx, float fy, float cx,
                               float cy, float k1, float k2, float p1, float p2,
                               float k3 = 0.0f, float k4 = 0.0f,
                               float k5 = 0.0f, float k6 = 0.0f) {
  double u1 = (*u - cx) / fx;
  double v1 = (*v - cy) / fy;
  double u2 = u1 * u1;
  double v2 = v1 * v1;
  double r2 = u2 + v2;

  // https://github.com/egonSchiele/OpenCV/blob/master/modules/imgproc/src/undistort.cpp#L133
  double _2uv = 2 * u1 * v1;
  double kr = (1 + ((k3 * r2 + k2) * r2 + k1) * r2) /
              (1 + ((k6 * r2 + k5) * r2 + k4) * r2);
  *u = static_cast<float>(fx * (u1 * kr + p1 * _2uv + p2 * (r2 + 2 * u2)) + cx);
  *v = static_cast<float>(fy * (v1 * kr + p1 * (r2 + 2 * v2) + p2 * _2uv) + cy);
}

// https://docs.opencv.org/4.3.0/d9/d0c/group__calib3d.html
inline void UndistortPixelOpencv(float* u, float* v, float fx, float fy,
                                 float cx, float cy, float k1, float k2,
                                 float p1, float p2, float k3 = 0.0f,
                                 float k4 = 0.0f, float k5 = 0.0f,
                                 float k6 = 0.0f) {
  // https://github.com/egonSchiele/OpenCV/blob/master/modules/imgproc/src/undistort.cpp#L345
  // https://github.com/opencv/opencv/blob/master/modules/calib3d/src/undistort.dispatch.cpp#L385
  const double x0 = (*u - cx) / fx;
  const double y0 = (*v - cy) / fy;
  double x = x0;
  double y = y0;
  // Compensate distortion iteratively
  // 5 is from OpenCV code.
  // I don't know theoritical rationale why 5 is enough...
  const int max_iter = 5;
  for (int j = 0; j < max_iter; j++) {
    double r2 = x * x + y * y;
    double icdist = (1 + ((k6 * r2 + k5) * r2 + k4) * r2) /
                    (1 + ((k3 * r2 + k2) * r2 + k1) * r2);
    double deltaX = 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
    double deltaY = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
    x = (x0 - deltaX) * icdist;
    y = (y0 - deltaY) * icdist;
  }

  *u = static_cast<float>(x * fx + cx);
  *v = static_cast<float>(y * fy + cy);
}

}  // namespace ugu
