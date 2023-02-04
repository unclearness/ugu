/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <string>

#define UGU_EIGEN_WARNING_LEVEL 0
#define UGU_OPENCV_WARNING_LEVEL 0

#ifdef _WIN32
#pragma warning(push, UGU_EIGEN_WARNING_LEVEL)
#endif
#include "Eigen/Geometry"
#include "ugu/eigen_util.h"
#ifdef _WIN32
#pragma warning(pop)
#endif

#include "ugu/log.h"

#ifdef UGU_USE_OPENCV
#ifdef _WIN32
#pragma warning(push, UGU_OPENCV_WARNING_LEVEL)
#endif
#include "opencv2/core.hpp"
#ifdef _WIN32
#pragma warning(pop)
#endif
#endif

#define UGU_FLOATING_POINT_ONLY_TEMPLATE                              \
  template <typename T,                                               \
            typename std::enable_if<std::is_floating_point<T>::value, \
                                    std::nullptr_t>::type = nullptr>

namespace ugu {

static inline const double pi = 3.14159265358979323846;

// Interpolation method in texture uv space
enum class ColorInterpolation {
  kNn = 0,       // Nearest Neigbor
  kBilinear = 1  // Bilinear interpolation
};

enum class CoordinateType { OpenCV, OpenGL };

float radians(const float& degrees);
double radians(const double& degrees);

float degrees(const float& radians);
double degrees(const double& radians);

float Fov2FocalPix(float fov, float pix, bool is_deg = false);
float FocalPix2Fov(float f, float pix, bool to_deg = false);

#ifdef UGU_USE_OPENCV
using cv::saturate_cast;
#else
template <typename T, typename TT>
T saturate_cast(const TT& v);

template <typename T, typename TT>
T saturate_cast(const TT& v) {
  return static_cast<T>(
      std::clamp(v, static_cast<TT>(std::numeric_limits<T>::lowest()),
                 static_cast<TT>(std::numeric_limits<T>::max())));
}
#endif
}  // namespace ugu
