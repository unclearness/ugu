/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "ugu/common.h"

namespace {

// borrow from glm
// radians
template <typename genType>
genType radiansImpl(const genType& degrees) {
  // "'radians' only accept floating-point input"
  static_assert(std::numeric_limits<genType>::is_iec559);

  return degrees * static_cast<genType>(0.01745329251994329576923690768489);
}

// degrees
template <typename genType>
genType degreesImpl(const genType& radians) {
  // "'degrees' only accept floating-point input"
  static_assert(std::numeric_limits<genType>::is_iec559);

  return radians * static_cast<genType>(57.295779513082320876798154814105);
}

}  // namespace

namespace ugu {

float radians(const float& degrees) { return radiansImpl(degrees); }
double radians(const double& degrees) { return radiansImpl(degrees); }

float degrees(const float& radians) { return degreesImpl(radians); }
double degrees(const double& radians) { return degreesImpl(radians); }

float Fov2FocalPix(float fov, float pix, bool is_deg) {
  if (is_deg) {
    fov = radians(fov);
  }
  return pix * 0.5f / static_cast<float>(std::tan(fov * 0.5));
}

float FocalPix2Fov(float f, float pix, bool to_deg) {
  float fov = 2.f * std::atan(pix * 0.5f / f);
  if (to_deg) {
    fov = degrees(fov);
  }
  return fov;
}

}  // namespace ugu
