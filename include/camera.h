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

#include "include/common.h"
#include "include/pose.h"

namespace currender {
class Camera {
 protected:
  int width_;
  int height_;
  Pose c2w_;  // camera -> world, sometimes called as "pose"
  Pose w2c_;  // world -> camera, sometimes called as "extrinsic"

 public:
  Camera();
  virtual ~Camera();
  Camera(int width, int height);
  Camera(int width, int height, const Pose& c2w);
  int width() const;
  int height() const;
  const Pose& c2w() const;
  const Pose& w2c() const;
  void set_size(int width, int height);
  void set_c2w(const Pose& c2w);

  // camera -> image conversion
  virtual void Project(const glm::vec3& camera_p, glm::vec3* image_p) const = 0;
  virtual void Project(const glm::vec3& camera_p, glm::vec2* image_p) const = 0;
  virtual void Project(const glm::vec3& camera_p, glm::vec2* image_p,
                       float* d) const = 0;

  // image -> camera conversion
  // need depth value as input
  virtual void Unproject(const glm::vec3& image_p,
                         glm::vec3* camera_p) const = 0;
  virtual void Unproject(const glm::vec2& image_p, float d,
                         glm::vec3* camera_p) const = 0;

  // position emmiting ray
  virtual void org_ray_c(float x, float y, glm::vec3* org) const = 0;
  virtual void org_ray_w(float x, float y, glm::vec3* org) const = 0;

  // ray direction
  virtual void ray_c(float x, float y,
                     glm::vec3* dir) const = 0;  // ray in camera coordinate
  virtual void ray_w(float x, float y,
                     glm::vec3* dir) const = 0;  // ray in world coordinate
};

// Pinhole camera model with pixel-scale principal point and focal length
// Widely used in computer vision community as perspective camera model
// Valid only if FoV is much less than 180 deg.
class PinholeCamera : public Camera {
  glm::vec2 principal_point_;
  glm::vec2 focal_length_;

 public:
  PinholeCamera();
  ~PinholeCamera();
  PinholeCamera(int width, int height);
  PinholeCamera(int width, int height, float fov_y_deg);
  PinholeCamera(int width, int height, const Pose& c2w);
  PinholeCamera(int width, int height, const Pose& c2w, float fov_y_deg);
  PinholeCamera(int width, int height, const Pose& c2w,
                const glm::vec2& principal_point,
                const glm::vec2& focal_length);

  // FoV (Field of View) in degree interface is provided for convenience
  float fov_x() const;
  float fov_y() const;
  void set_fov_x(float fov_x_deg);
  void set_fov_y(float fov_y_deg);

  // pixel-scale principal point and focal length
  const glm::vec2& principal_point() const;
  const glm::vec2& focal_length() const;
  void set_principal_point(const glm::vec2& principal_point);
  void set_focal_length(const glm::vec2& focal_length);

  void Project(const glm::vec3& camera_p, glm::vec3* image_p) const override;
  void Project(const glm::vec3& camera_p, glm::vec2* image_p) const override;
  void Project(const glm::vec3& camera_p, glm::vec2* image_p,
               float* d) const override;
  void Unproject(const glm::vec3& image_p, glm::vec3* camera_p) const override;
  void Unproject(const glm::vec2& image_p, float d,
                 glm::vec3* camera_p) const override;
  void org_ray_c(float x, float y, glm::vec3* org) const override;
  void org_ray_w(float x, float y, glm::vec3* org) const override;
  void ray_c(float x, float y, glm::vec3* dir) const override;
  void ray_w(float x, float y, glm::vec3* dir) const override;
};

// Orthographic/orthogonal projection camera with no perspective
// Image coordinate is translated camera coordinate
// Different from pinhole camera in particular x and y coordinate in image
class OrthoCamera : public Camera {
 public:
  OrthoCamera();
  ~OrthoCamera();
  OrthoCamera(int width, int height);
  OrthoCamera(int width, int height, const Pose& c2w);

  void Project(const glm::vec3& camera_p, glm::vec3* image_p) const override;
  void Project(const glm::vec3& camera_p, glm::vec2* image_p) const override;
  void Project(const glm::vec3& camera_p, glm::vec2* image_p,
               float* d) const override;
  void Unproject(const glm::vec3& image_p, glm::vec3* camera_p) const override;
  void Unproject(const glm::vec2& image_p, float d,
                 glm::vec3* camera_p) const override;
  void org_ray_c(float x, float y, glm::vec3* org) const override;
  void org_ray_w(float x, float y, glm::vec3* org) const override;
  void ray_c(float x, float y, glm::vec3* dir) const override;
  void ray_w(float x, float y, glm::vec3* dir) const override;
};

}  // namespace currender
