/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include "include/common.h"

namespace currender {
class Pose {
  glm::mat4 T_;
  glm::mat3 R_;
  glm::vec3 t_;

 public:
  Pose();
  ~Pose();
  Pose(const Pose& pose);
  explicit Pose(const glm::mat4& T);
  Pose(const glm::mat3& R, const glm::vec3& t);
  void set_I();
  const glm::mat4& T() const;
  const glm::mat3& R() const;
  const glm::vec3& t() const;
  void set_T(const glm::mat4& T);
  void set_Rt(const glm::mat3& R, const glm::vec3& t);
  void set_R(const glm::mat3& R);
  void set_t(const glm::vec3& t);
  Pose& operator=(const Pose& pose);
  void Transform(glm::vec3* src_dst) const;
  void Rotate(glm::vec3* src_dst) const;
  void Translate(glm::vec3* src_dst) const;
};

glm::mat3 EulerAngleDegYXZ(float yaw_deg, float pitch_deg, float roll_deg);
glm::mat3 EulerAngleYXZ(float yaw_rad, float pitch_rad, float roll_rad);

}  // namespace currender
