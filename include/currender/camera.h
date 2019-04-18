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

#include <string>
#include <utility>
#include <vector>

#include "currender/common.h"

namespace currender {
class Camera {
 protected:
  int width_;
  int height_;
  Eigen::Affine3d c2w_;  // camera -> world, sometimes called as "pose"
  Eigen::Affine3d w2c_;

  Eigen::Matrix3f c2w_R_f_;
  Eigen::Vector3f c2w_t_f_;

  Eigen::Vector3f x_direc_, y_direc_, z_direc_;

  Eigen::Matrix3f w2c_R_f_;
  Eigen::Vector3f w2c_t_f_;

 public:
  Camera();
  virtual ~Camera();
  Camera(int width, int height);
  Camera(int width, int height, const Eigen::Affine3d& c2w);
  int width() const;
  int height() const;
  const Eigen::Affine3d& c2w() const;
  const Eigen::Affine3d& w2c() const;
  virtual void set_size(int width, int height);
  virtual void set_c2w(const Eigen::Affine3d& c2w);

  // camera -> image conversion
  virtual void Project(const Eigen::Vector3f& camera_p,
                       Eigen::Vector3f* image_p) const = 0;
  virtual void Project(const Eigen::Vector3f& camera_p,
                       Eigen::Vector2f* image_p) const = 0;
  virtual void Project(const Eigen::Vector3f& camera_p,
                       Eigen::Vector2f* image_p, float* d) const = 0;

  // image -> camera conversion
  // need depth value as input
  virtual void Unproject(const Eigen::Vector3f& image_p,
                         Eigen::Vector3f* camera_p) const = 0;
  virtual void Unproject(const Eigen::Vector2f& image_p, float d,
                         Eigen::Vector3f* camera_p) const = 0;

  // position emmiting ray
  virtual void org_ray_c(float x, float y, Eigen::Vector3f* org) const = 0;
  virtual void org_ray_w(float x, float y, Eigen::Vector3f* org) const = 0;
  virtual void org_ray_c(int x, int y, Eigen::Vector3f* org) const = 0;
  virtual void org_ray_w(int x, int y, Eigen::Vector3f* org) const = 0;

  // ray direction
  virtual void ray_c(
      float x, float y,
      Eigen::Vector3f* dir) const = 0;  // ray in camera coordinate
  virtual void ray_w(
      float x, float y,
      Eigen::Vector3f* dir) const = 0;  // ray in world coordinate
  virtual void ray_c(int x, int y, Eigen::Vector3f* dir) const = 0;
  virtual void ray_w(int x, int y, Eigen::Vector3f* dir) const = 0;
};

// Pinhole camera model with pixel-scale principal point and focal length
// Widely used in computer vision community as perspective camera model
// Valid only if FoV is much less than 180 deg.
class PinholeCamera : public Camera {
  Eigen::Vector2f principal_point_;
  Eigen::Vector2f focal_length_;

  std::vector<Eigen::Vector3f> org_ray_c_table_;
  std::vector<Eigen::Vector3f> org_ray_w_table_;
  std::vector<Eigen::Vector3f> ray_c_table_;
  std::vector<Eigen::Vector3f> ray_w_table_;

  void InitRayTable();

 public:
  PinholeCamera();
  ~PinholeCamera();
  PinholeCamera(int width, int height);
  PinholeCamera(int width, int height, float fov_y_deg);
  PinholeCamera(int width, int height, const Eigen::Affine3d& c2w);
  PinholeCamera(int width, int height, const Eigen::Affine3d& c2w,
                float fov_y_deg);
  PinholeCamera(int width, int height, const Eigen::Affine3d& c2w,
                const Eigen::Vector2f& principal_point,
                const Eigen::Vector2f& focal_length);

  void set_size(int width, int height) override;
  void set_c2w(const Eigen::Affine3d& c2w) override;

  // FoV (Field of View) in degree interface is provided for convenience
  float fov_x() const;
  float fov_y() const;
  void set_fov_x(float fov_x_deg);
  void set_fov_y(float fov_y_deg);

  // pixel-scale principal point and focal length
  const Eigen::Vector2f& principal_point() const;
  const Eigen::Vector2f& focal_length() const;
  void set_principal_point(const Eigen::Vector2f& principal_point);
  void set_focal_length(const Eigen::Vector2f& focal_length);

  void Project(const Eigen::Vector3f& camera_p,
               Eigen::Vector3f* image_p) const override;
  void Project(const Eigen::Vector3f& camera_p,
               Eigen::Vector2f* image_p) const override;
  void Project(const Eigen::Vector3f& camera_p, Eigen::Vector2f* image_p,
               float* d) const override;
  void Unproject(const Eigen::Vector3f& image_p,
                 Eigen::Vector3f* camera_p) const override;
  void Unproject(const Eigen::Vector2f& image_p, float d,
                 Eigen::Vector3f* camera_p) const override;
  void org_ray_c(float x, float y, Eigen::Vector3f* org) const override;
  void org_ray_w(float x, float y, Eigen::Vector3f* org) const override;
  void org_ray_c(int x, int y, Eigen::Vector3f* org) const override;
  void org_ray_w(int x, int y, Eigen::Vector3f* org) const override;

  void ray_c(float x, float y, Eigen::Vector3f* dir) const override;
  void ray_w(float x, float y, Eigen::Vector3f* dir) const override;
  void ray_c(int x, int y, Eigen::Vector3f* dir) const override;
  void ray_w(int x, int y, Eigen::Vector3f* dir) const override;
};

// Orthographic/orthogonal projection camera with no perspective
// Image coordinate is translated camera coordinate
// Different from pinhole camera in particular x and y coordinate in image
class OrthoCamera : public Camera {
  std::vector<Eigen::Vector3f> org_ray_c_table_;
  std::vector<Eigen::Vector3f> org_ray_w_table_;
  std::vector<Eigen::Vector3f> ray_c_table_;
  std::vector<Eigen::Vector3f> ray_w_table_;

  void InitRayTable();

 public:
  OrthoCamera();
  ~OrthoCamera();
  OrthoCamera(int width, int height);
  OrthoCamera(int width, int height, const Eigen::Affine3d& c2w);

  void set_size(int width, int height) override;
  void set_c2w(const Eigen::Affine3d& c2w) override;

  void Project(const Eigen::Vector3f& camera_p,
               Eigen::Vector3f* image_p) const override;
  void Project(const Eigen::Vector3f& camera_p,
               Eigen::Vector2f* image_p) const override;
  void Project(const Eigen::Vector3f& camera_p, Eigen::Vector2f* image_p,
               float* d) const override;
  void Unproject(const Eigen::Vector3f& image_p,
                 Eigen::Vector3f* camera_p) const override;
  void Unproject(const Eigen::Vector2f& image_p, float d,
                 Eigen::Vector3f* camera_p) const override;
  void org_ray_c(float x, float y, Eigen::Vector3f* org) const override;
  void org_ray_w(float x, float y, Eigen::Vector3f* org) const override;
  void org_ray_c(int x, int y, Eigen::Vector3f* org) const override;
  void org_ray_w(int x, int y, Eigen::Vector3f* org) const override;

  void ray_c(float x, float y, Eigen::Vector3f* dir) const override;
  void ray_w(float x, float y, Eigen::Vector3f* dir) const override;
  void ray_c(int x, int y, Eigen::Vector3f* dir) const override;
  void ray_w(int x, int y, Eigen::Vector3f* dir) const override;
};

void WriteTumFormat(const std::vector<Eigen::Affine3d>& poses,
                    const std::string& path);
bool LoadTumFormat(const std::string& path,
                   std::vector<Eigen::Affine3d>* poses);
bool LoadTumFormat(const std::string& path,
                   std::vector<std::pair<int, Eigen::Affine3d>>* poses);

}  // namespace currender
