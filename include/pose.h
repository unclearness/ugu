#pragma once

#include "common.h"

namespace unclearness {
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
  void transform(glm::vec3& src_dst) const;
};

}  // namespace unclearness