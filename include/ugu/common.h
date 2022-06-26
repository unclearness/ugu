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

#define UGU_FLOATING_POINT_ONLY_TEMPLATE                              \
  template <typename T,                                               \
            typename std::enable_if<std::is_floating_point<T>::value, \
                                    std::nullptr_t>::type = nullptr>

namespace ugu {

// TODO: definition in header may be invalid.
static const double pi = 3.14159265358979323846;

// Interpolation method in texture uv space
enum class ColorInterpolation {
  kNn = 0,       // Nearest Neigbor
  kBilinear = 1  // Bilinear interpolation
};

// borrow from glm
// radians
template <typename genType>
genType radians(genType degrees) {
  // "'radians' only accept floating-point input"
  static_assert(std::numeric_limits<genType>::is_iec559);

  return degrees * static_cast<genType>(0.01745329251994329576923690768489);
}

// degrees
template <typename genType>
genType degrees(genType radians) {
  // "'degrees' only accept floating-point input"
  static_assert(std::numeric_limits<genType>::is_iec559);

  return radians * static_cast<genType>(57.295779513082320876798154814105);
}

// https://stackoverflow.com/questions/13768423/setting-up-projection-model-and-view-transformations-for-vertex-shader-in-eige
template <typename T>
void c2w(const Eigen::Matrix<T, 3, 1>& position,
         const Eigen::Matrix<T, 3, 1>& target, const Eigen::Matrix<T, 3, 1>& up,
         Eigen::Matrix<T, 3, 3>* R) {
  static_assert(std::numeric_limits<T>::is_iec559);

  R->col(2) = (target - position).normalized();
  R->col(0) = R->col(2).cross(up).normalized();
  R->col(1) = R->col(2).cross(R->col(0));
}

template <typename genType>
void c2w(const Eigen::Matrix<genType, 3, 1>& position,
         const Eigen::Matrix<genType, 3, 1>& target,
         const Eigen::Matrix<genType, 3, 1>& up,
         Eigen::Matrix<genType, 4, 4>* T) {
  static_assert(std::numeric_limits<genType>::is_iec559);

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
  double p1d = p1;
  double p2d = p2;
  for (int j = 0; j < max_iter; j++) {
    double r2 = x * x + y * y;
    double icdist = (1.0 + ((k6 * r2 + k5) * r2 + k4) * r2) /
                    (1.0 + ((k3 * r2 + k2) * r2 + k1) * r2);
    double deltaX = 2.0 * p1d * x * y + p2d * (r2 + 2 * x * x);
    double deltaY = p1d * (r2 + 2.0 * y * y) + 2.0 * p2d * x * y;
    x = (x0 - deltaX) * icdist;
    y = (y0 - deltaY) * icdist;
  }

  *u = static_cast<float>(x * fx + cx);
  *v = static_cast<float>(y * fy + cy);
}

template <typename T, typename TT>
T saturate_cast(const TT& v) {
  return static_cast<T>(
      std::clamp(v, static_cast<TT>(std::numeric_limits<T>::lowest()),
                 static_cast<TT>(std::numeric_limits<T>::max())));
}

enum class Coordinate {
  OpenCV,
  OpenGL
};

}  // namespace ugu
