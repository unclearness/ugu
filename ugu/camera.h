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

#include "ugu/common.h"

namespace ugu {

// interface (pure abstract base class with no state or defined methods) for
// camera
class Camera {
 public:
  virtual ~Camera() {}

  virtual int width() const = 0;
  virtual int height() const = 0;
  virtual const Eigen::Affine3d& c2w() const = 0;
  virtual const Eigen::Affine3d& w2c() const = 0;
  virtual void set_size(int width, int height) = 0;
  virtual void set_c2w(const Eigen::Affine3d& c2w) = 0;

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
  int width_;
  int height_;

  Eigen::Affine3d c2w_;  // camera -> world, sometimes called as "pose"
  Eigen::Affine3d w2c_;

  Eigen::Matrix3f c2w_R_f_;
  Eigen::Vector3f c2w_t_f_;
  Eigen::Vector3f x_direc_, y_direc_, z_direc_;

  Eigen::Matrix3f w2c_R_f_;
  Eigen::Vector3f w2c_t_f_;

  Eigen::Vector2f principal_point_;
  Eigen::Vector2f focal_length_;

  std::vector<Eigen::Vector3f> org_ray_c_table_;
  std::vector<Eigen::Vector3f> org_ray_w_table_;
  std::vector<Eigen::Vector3f> ray_c_table_;
  std::vector<Eigen::Vector3f> ray_w_table_;

  void InitRayTable();

  void set_size_no_raytable_update(int width, int height);
  void set_c2w_no_raytable_update(const Eigen::Affine3d& c2w);
  void set_fov_y_no_raytable_update(float fov_y_deg);

 public:
  PinholeCamera();
  ~PinholeCamera();
  PinholeCamera(int width, int height, float fov_y_deg);
  PinholeCamera(int width, int height, const Eigen::Affine3d& c2w,
                float fov_y_deg);
  PinholeCamera(int width, int height, const Eigen::Affine3d& c2w,
                const Eigen::Vector2f& principal_point,
                const Eigen::Vector2f& focal_length);

  int width() const override;
  int height() const override;
  const Eigen::Affine3d& c2w() const override;
  const Eigen::Affine3d& w2c() const override;
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
  int width_;
  int height_;

  Eigen::Affine3d c2w_;  // camera -> world, sometimes called as "pose"
  Eigen::Affine3d w2c_;

  Eigen::Matrix3f c2w_R_f_;
  Eigen::Vector3f c2w_t_f_;
  Eigen::Vector3f x_direc_, y_direc_, z_direc_;

  Eigen::Matrix3f w2c_R_f_;
  Eigen::Vector3f w2c_t_f_;

  std::vector<Eigen::Vector3f> org_ray_c_table_;
  std::vector<Eigen::Vector3f> org_ray_w_table_;
  std::vector<Eigen::Vector3f> ray_c_table_;
  std::vector<Eigen::Vector3f> ray_w_table_;

  void InitRayTable();

  void set_size_no_raytable_update(int width, int height);
  void set_c2w_no_raytable_update(const Eigen::Affine3d& c2w);

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

inline PinholeCamera::PinholeCamera()
    : principal_point_(-1, -1), focal_length_(-1, -1) {
  set_size_no_raytable_update(-1, -1);
  set_c2w_no_raytable_update(Eigen::Affine3d::Identity());
}

inline PinholeCamera::~PinholeCamera() {}

inline int PinholeCamera::width() const { return width_; }

inline int PinholeCamera::height() const { return height_; }

inline const Eigen::Affine3d& PinholeCamera::c2w() const { return c2w_; }

inline const Eigen::Affine3d& PinholeCamera::w2c() const { return w2c_; }

inline PinholeCamera::PinholeCamera(int width, int height, float fov_y_deg) {
  set_size_no_raytable_update(width, height);

  principal_point_[0] = width_ * 0.5f - 0.5f;
  principal_point_[1] = height_ * 0.5f - 0.5f;

  set_fov_y_no_raytable_update(fov_y_deg);

  set_c2w_no_raytable_update(Eigen::Affine3d::Identity());

  InitRayTable();
}

inline PinholeCamera::PinholeCamera(int width, int height,
                                    const Eigen::Affine3d& c2w,
                                    float fov_y_deg) {
  set_size_no_raytable_update(width, height);

  set_c2w_no_raytable_update(c2w);

  principal_point_[0] = width_ * 0.5f - 0.5f;
  principal_point_[1] = height_ * 0.5f - 0.5f;

  set_fov_y_no_raytable_update(fov_y_deg);

  InitRayTable();
}

inline PinholeCamera::PinholeCamera(int width, int height,
                                    const Eigen::Affine3d& c2w,
                                    const Eigen::Vector2f& principal_point,
                                    const Eigen::Vector2f& focal_length)
    : principal_point_(principal_point), focal_length_(focal_length) {
  set_size_no_raytable_update(width, height);
  set_c2w_no_raytable_update(c2w);
  InitRayTable();
}

inline void PinholeCamera::set_size_no_raytable_update(int width, int height) {
  width_ = width;
  height_ = height;
}

inline void PinholeCamera::set_c2w_no_raytable_update(
    const Eigen::Affine3d& c2w) {
  c2w_ = c2w;
  w2c_ = c2w_.inverse();

  c2w_R_f_ = c2w_.matrix().block<3, 3>(0, 0).cast<float>();
  c2w_t_f_ = c2w_.matrix().block<3, 1>(0, 3).cast<float>();
  x_direc_ = c2w_.matrix().block<3, 3>(0, 0).col(0).cast<float>();
  y_direc_ = c2w_.matrix().block<3, 3>(0, 0).col(1).cast<float>();
  z_direc_ = c2w_.matrix().block<3, 3>(0, 0).col(2).cast<float>();

  w2c_R_f_ = w2c_.matrix().block<3, 3>(0, 0).cast<float>();
  w2c_t_f_ = w2c_.matrix().block<3, 1>(0, 3).cast<float>();
}

inline void PinholeCamera::set_fov_y_no_raytable_update(float fov_y_deg) {
  focal_length_[1] =
      height_ * 0.5f /
      static_cast<float>(std::tan(radians<float>(fov_y_deg) * 0.5));
  focal_length_[0] = focal_length_[1];
}

inline void PinholeCamera::set_size(int width, int height) {
  set_size_no_raytable_update(width, height);

  InitRayTable();
}

inline void PinholeCamera::set_c2w(const Eigen::Affine3d& c2w) {
  set_c2w_no_raytable_update(c2w);

  InitRayTable();
}

inline float PinholeCamera::fov_x() const {
  return degrees<float>(2 * std::atan(width_ * 0.5f / focal_length_[0]));
}

inline float PinholeCamera::fov_y() const {
  return degrees<float>(2 * std::atan(height_ * 0.5f / focal_length_[1]));
}

inline const Eigen::Vector2f& PinholeCamera::principal_point() const {
  return principal_point_;
}

inline const Eigen::Vector2f& PinholeCamera::focal_length() const {
  return focal_length_;
}

inline void PinholeCamera::set_principal_point(
    const Eigen::Vector2f& principal_point) {
  principal_point_ = principal_point;
  InitRayTable();
}

inline void PinholeCamera::set_focal_length(
    const Eigen::Vector2f& focal_length) {
  focal_length_ = focal_length;
  InitRayTable();
}

inline void PinholeCamera::set_fov_x(float fov_x_deg) {
  // same focal length per pixel for x and y
  focal_length_[0] =
      width_ * 0.5f /
      static_cast<float>(std::tan(radians<float>(fov_x_deg) * 0.5));
  focal_length_[1] = focal_length_[0];
  InitRayTable();
}

inline void PinholeCamera::set_fov_y(float fov_y_deg) {
  // same focal length per pixel for x and y

  set_fov_y_no_raytable_update(fov_y_deg);

  InitRayTable();
}

inline void PinholeCamera::Project(const Eigen::Vector3f& camera_p,
                                   Eigen::Vector3f* image_p) const {
  (*image_p)[0] =
      focal_length_[0] / camera_p[2] * camera_p[0] + principal_point_[0];
  (*image_p)[1] =
      focal_length_[1] / camera_p[2] * camera_p[1] + principal_point_[1];
  (*image_p)[2] = camera_p[2];
}

inline void PinholeCamera::Project(const Eigen::Vector3f& camera_p,
                                   Eigen::Vector2f* image_p) const {
  (*image_p)[0] =
      focal_length_[0] / camera_p[2] * camera_p[0] + principal_point_[0];
  (*image_p)[1] =
      focal_length_[1] / camera_p[2] * camera_p[1] + principal_point_[1];
}

inline void PinholeCamera::Project(const Eigen::Vector3f& camera_p,
                                   Eigen::Vector2f* image_p, float* d) const {
  (*image_p)[0] =
      focal_length_[0] / camera_p[2] * camera_p[0] + principal_point_[0];
  (*image_p)[1] =
      focal_length_[1] / camera_p[2] * camera_p[1] + principal_point_[1];
  *d = camera_p[2];
}

inline void PinholeCamera::Unproject(const Eigen::Vector3f& image_p,
                                     Eigen::Vector3f* camera_p) const {
  (*camera_p)[0] =
      (image_p[0] - principal_point_[0]) * image_p[2] / focal_length_[0];
  (*camera_p)[1] =
      (image_p[1] - principal_point_[1]) * image_p[2] / focal_length_[1];
  (*camera_p)[2] = image_p[2];
}

inline void PinholeCamera::Unproject(const Eigen::Vector2f& image_p, float d,
                                     Eigen::Vector3f* camera_p) const {
  (*camera_p)[0] = (image_p[0] - principal_point_[0]) * d / focal_length_[0];
  (*camera_p)[1] = (image_p[1] - principal_point_[1]) * d / focal_length_[1];
  (*camera_p)[2] = d;
}

inline void PinholeCamera::org_ray_c(float x, float y,
                                     Eigen::Vector3f* org) const {
  (void)x;
  (void)y;
  (*org)[0] = 0.0f;
  (*org)[1] = 0.0f;
  (*org)[2] = 0.0f;
}

inline void PinholeCamera::org_ray_w(float x, float y,
                                     Eigen::Vector3f* org) const {
  (void)x;
  (void)y;
  *org = c2w_t_f_;
}

inline void PinholeCamera::ray_c(float x, float y, Eigen::Vector3f* dir) const {
  (*dir)[0] = (x - principal_point_[0]) / focal_length_[0];
  (*dir)[1] = (y - principal_point_[1]) / focal_length_[1];
  (*dir)[2] = 1.0f;
  dir->normalize();
}

inline void PinholeCamera::ray_w(float x, float y, Eigen::Vector3f* dir) const {
  ray_c(x, y, dir);
  *dir = c2w_R_f_ * *dir;
}

inline void PinholeCamera::org_ray_c(int x, int y, Eigen::Vector3f* org) const {
  *org = org_ray_c_table_[y * width_ + x];
}
inline void PinholeCamera::org_ray_w(int x, int y, Eigen::Vector3f* org) const {
  *org = org_ray_w_table_[y * width_ + x];
}

inline void PinholeCamera::ray_c(int x, int y, Eigen::Vector3f* dir) const {
  *dir = ray_c_table_[y * width_ + x];
}
inline void PinholeCamera::ray_w(int x, int y, Eigen::Vector3f* dir) const {
  *dir = ray_w_table_[y * width_ + x];
}

inline void PinholeCamera::InitRayTable() {
  org_ray_c_table_.resize(width_ * height_);
  org_ray_w_table_.resize(width_ * height_);
  ray_c_table_.resize(width_ * height_);
  ray_w_table_.resize(width_ * height_);

  for (int y = 0; y < height_; y++) {
    for (int x = 0; x < width_; x++) {
      org_ray_c(static_cast<float>(x), static_cast<float>(y),
                &org_ray_c_table_[y * width_ + x]);
      org_ray_w(static_cast<float>(x), static_cast<float>(y),
                &org_ray_w_table_[y * width_ + x]);

      ray_c(static_cast<float>(x), static_cast<float>(y),
            &ray_c_table_[y * width_ + x]);
      ray_w(static_cast<float>(x), static_cast<float>(y),
            &ray_w_table_[y * width_ + x]);
    }
  }
}

inline OrthoCamera::OrthoCamera() {
  set_size_no_raytable_update(-1, -1);
  set_c2w_no_raytable_update(Eigen::Affine3d::Identity());
}
inline OrthoCamera::~OrthoCamera() {}
inline OrthoCamera::OrthoCamera(int width, int height) {
  set_size_no_raytable_update(width, height);
  set_c2w_no_raytable_update(Eigen::Affine3d::Identity());

  InitRayTable();
}
inline OrthoCamera::OrthoCamera(int width, int height,
                                const Eigen::Affine3d& c2w) {
  set_size_no_raytable_update(width, height);

  set_c2w_no_raytable_update(c2w);

  InitRayTable();
}

inline void OrthoCamera::set_size_no_raytable_update(int width, int height) {
  width_ = width;
  height_ = height;
}

inline void OrthoCamera::set_c2w_no_raytable_update(
    const Eigen::Affine3d& c2w) {
  c2w_ = c2w;
  w2c_ = c2w_.inverse();

  c2w_R_f_ = c2w_.matrix().block<3, 3>(0, 0).cast<float>();
  c2w_t_f_ = c2w_.matrix().block<3, 1>(0, 3).cast<float>();
  x_direc_ = c2w_.matrix().block<3, 3>(0, 0).col(0).cast<float>();
  y_direc_ = c2w_.matrix().block<3, 3>(0, 0).col(1).cast<float>();
  z_direc_ = c2w_.matrix().block<3, 3>(0, 0).col(2).cast<float>();

  w2c_R_f_ = w2c_.matrix().block<3, 3>(0, 0).cast<float>();
  w2c_t_f_ = w2c_.matrix().block<3, 1>(0, 3).cast<float>();
}
inline void OrthoCamera::set_size(int width, int height) {
  set_size_no_raytable_update(width, height);

  InitRayTable();
}

inline void OrthoCamera::set_c2w(const Eigen::Affine3d& c2w) {
  set_c2w_no_raytable_update(c2w);

  InitRayTable();
}

inline void OrthoCamera::Project(const Eigen::Vector3f& camera_p,
                                 Eigen::Vector3f* image_p) const {
  *image_p = camera_p;
}

inline void OrthoCamera::Project(const Eigen::Vector3f& camera_p,
                                 Eigen::Vector2f* image_p) const {
  (*image_p)[0] = camera_p[0];
  (*image_p)[1] = camera_p[1];
}

inline void OrthoCamera::Project(const Eigen::Vector3f& camera_p,
                                 Eigen::Vector2f* image_p, float* d) const {
  (*image_p)[0] = camera_p[0];
  (*image_p)[1] = camera_p[1];
  *d = camera_p[2];
}

inline void OrthoCamera::Unproject(const Eigen::Vector3f& image_p,
                                   Eigen::Vector3f* camera_p) const {
  *camera_p = image_p;
}

inline void OrthoCamera::Unproject(const Eigen::Vector2f& image_p, float d,
                                   Eigen::Vector3f* camera_p) const {
  (*camera_p)[0] = image_p[0];
  (*camera_p)[1] = image_p[1];
  (*camera_p)[2] = d;
}

inline void OrthoCamera::org_ray_c(float x, float y,
                                   Eigen::Vector3f* org) const {
  (*org)[0] = x - width_ / 2;
  (*org)[1] = y - height_ / 2;
  (*org)[2] = 0.0f;
}

inline void OrthoCamera::org_ray_w(float x, float y,
                                   Eigen::Vector3f* org) const {
  *org = c2w_t_f_;

  Eigen::Vector3f offset_x = (x - width_ * 0.5f) * x_direc_;
  Eigen::Vector3f offset_y = (y - height_ * 0.5f) * y_direc_;

  *org += offset_x;
  *org += offset_y;
}

inline void OrthoCamera::ray_c(float x, float y, Eigen::Vector3f* dir) const {
  (void)x;
  (void)y;
  // parallell ray along with z axis
  (*dir)[0] = 0.0f;
  (*dir)[1] = 0.0f;
  (*dir)[2] = 1.0f;
}

inline void OrthoCamera::ray_w(float x, float y, Eigen::Vector3f* dir) const {
  (void)x;
  (void)y;
  // extract z direction of camera pose
  *dir = z_direc_;
}

inline void OrthoCamera::org_ray_c(int x, int y, Eigen::Vector3f* org) const {
  *org = org_ray_c_table_[y * width_ + x];
}

inline void OrthoCamera::org_ray_w(int x, int y, Eigen::Vector3f* org) const {
  *org = org_ray_w_table_[y * width_ + x];
}

inline void OrthoCamera::ray_c(int x, int y, Eigen::Vector3f* dir) const {
  *dir = ray_c_table_[y * width_ + x];
}
inline void OrthoCamera::ray_w(int x, int y, Eigen::Vector3f* dir) const {
  *dir = ray_w_table_[y * width_ + x];
}

inline void OrthoCamera::InitRayTable() {
  org_ray_c_table_.resize(width_ * height_);
  org_ray_w_table_.resize(width_ * height_);
  ray_c_table_.resize(width_ * height_);
  ray_w_table_.resize(width_ * height_);

  for (int y = 0; y < height_; y++) {
    for (int x = 0; x < width_; x++) {
      org_ray_c(static_cast<float>(x), static_cast<float>(y),
                &org_ray_c_table_[y * width_ + x]);
      org_ray_w(static_cast<float>(x), static_cast<float>(y),
                &org_ray_w_table_[y * width_ + x]);

      ray_c(static_cast<float>(x), static_cast<float>(y),
            &ray_c_table_[y * width_ + x]);
      ray_w(static_cast<float>(x), static_cast<float>(y),
            &ray_w_table_[y * width_ + x]);
    }
  }
}

}  // namespace ugu
