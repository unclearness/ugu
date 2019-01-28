/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "include/pose.h"

#pragma warning(push)
#pragma warning(disable : 4201)
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/euler_angles.hpp"
#pragma warning(pop)

namespace currender {

Pose::Pose() { set_I(); }
Pose::~Pose() {}
Pose::Pose(const Pose& pose) {
  set_I();
  set_T(pose.T());
}
Pose::Pose(const glm::mat4& T) {
  set_I();
  set_T(T);
}
Pose::Pose(const glm::mat3& R, const glm::vec3& t) {
  set_I();
  set_Rt(R, t);
}
void Pose::set_I() {
  T_ = glm::mat4(1.0f);
  R_ = glm::mat3(1.0f);
  t_ = glm::vec3(0.0f);
}

const glm::mat4& Pose::T() const { return T_; }
const glm::mat3& Pose::R() const { return R_; }
const glm::vec3& Pose::t() const { return t_; }
void Pose::set_T(const glm::mat4& T) {
  T_ = T;
  for (int j = 0; j < 3; j++) {
    for (int i = 0; i < 3; i++) {
      R_[i][j] = T_[i][j];
    }
  }
  for (int j = 0; j < 3; j++) {
    t_[j] = T[3][j];
  }
}
void Pose::set_Rt(const glm::mat3& R, const glm::vec3& t) {
  set_I();
  set_R(R);
  set_t(t);
}
void Pose::set_R(const glm::mat3& R) {
  R_ = R;
  for (int j = 0; j < 3; j++) {
    for (int i = 0; i < 3; i++) {
      T_[i][j] = R_[i][j];
    }
  }
}
void Pose::set_t(const glm::vec3& t) {
  t_ = t;

  for (int j = 0; j < 3; j++) {
    T_[3][j] = t_[j];
  }
}

Pose& Pose::operator=(const Pose& pose) {
  set_T(pose.T());
  return *this;
}

void Pose::Transform(glm::vec3* src_dst) const {
  *src_dst = R_ * (*src_dst) + t_;
}

void Pose::Rotate(glm::vec3* src_dst) const { *src_dst = R_ * (*src_dst); }

void Pose::Translate(glm::vec3* src_dst) const { *src_dst = (*src_dst) + t_; }

glm::mat3 EulerAngleDegYXZ(float yaw_deg, float pitch_deg, float roll_deg) {
  return EulerAngleYXZ(glm::radians(yaw_deg), glm::radians(pitch_deg),
                       glm::radians(roll_deg));
}

glm::mat3 EulerAngleYXZ(float yaw_rad, float pitch_rad, float roll_rad) {
  glm::mat4 T = glm::eulerAngleYXZ(yaw_rad, pitch_rad, roll_rad);

  glm::mat3 R;
  for (int j = 0; j < 3; j++) {
    for (int i = 0; i < 3; i++) {
      R[i][j] = T[i][j];
    }
  }

  return R;
}

}  // namespace currender
