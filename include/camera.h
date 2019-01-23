/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

/*
 * right-handed coordinate system
 * z:forward, y:down, x:right
 * same to OpenCV
 */

#pragma once

#include "include/pose.h"

#include "include/common.h"

namespace crender {
class Camera {
 protected:
  int width_;
  int height_;
  Pose c2w_;  // camera -> world
  Pose w2c_;  // world -> camera

 public:
  Camera();
  ~Camera();
  Camera(int width, int height, const Pose& c2w);
  int width();
  int height();
  const Pose& c2w();
  const Pose& w2c();
  void set_size(int width, int height);
  void set_c2w(const Pose& c2w);
  virtual void project(const glm::vec3& camera_p, glm::vec3& image_p) = 0;
  virtual void project(const glm::vec3& camera_p, glm::vec2& image_p) = 0;
  virtual void project(const glm::vec3& camera_p, glm::vec2& image_p,
                       float& d) = 0;
  virtual void unproject(const glm::vec3& image_p, glm::vec3& camera_p) = 0;
  virtual void unproject(const glm::vec2& image_p, float d,
                         glm::vec3& camera_p) = 0;
  virtual void ray_c(float x, float y, glm::vec3& dir) = 0;
  virtual void ray_w(float x, float y, glm::vec3& dir) = 0;  // multiply R
};

class PinholeCamera : public Camera {
  glm::vec2 principal_point_;
  glm::vec2 focal_length_;

 public:
  PinholeCamera();
  ~PinholeCamera();
  PinholeCamera(int width, int height, const Pose& c2w,
                const glm::vec2& principal_point,
                const glm::vec2& focal_length);
  const glm::vec2& principal_point() const;
  const glm::vec2& focal_length() const;
  void set_principal_point(const glm::vec2& principal_point);
  void set_focal_length(const glm::vec2& focal_length);
  void project(const glm::vec3& camera_p, glm::vec3& image_p);
  void project(const glm::vec3& camera_p, glm::vec2& image_p);
  void project(const glm::vec3& camera_p, glm::vec2& image_p, float& d);
  void unproject(const glm::vec3& image_p, glm::vec3& camera_p);
  void unproject(const glm::vec2& image_p, float d, glm::vec3& camera_p);
  void ray_c(float x, float y, glm::vec3& dir);
  void ray_w(float x, float y, glm::vec3& dir);  // multiply R
};

}  // namespace crender
