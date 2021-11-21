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

  Eigen::Vector2f principal_point_;
  Eigen::Vector2f focal_length_;

  mutable std::vector<Eigen::Vector3f> org_ray_c_table_;
  mutable std::vector<Eigen::Vector3f> org_ray_w_table_;
  mutable std::vector<Eigen::Vector3f> ray_c_table_;
  mutable std::vector<Eigen::Vector3f> ray_w_table_;

  mutable bool need_init_ray_table_ = true;
  void InitRayTable() const;

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

// PinholeCamera with support of OpenCV style distortion/undistortion
class OpenCvCamera : public PinholeCamera {
 private:
  float k1_, k2_, p1_, p2_, k3_, k4_, k5_, k6_;

 public:
  void distortion_coeffs(float* k1, float* k2, float* p1, float* p2,
                         float* k3 = nullptr, float* k4 = nullptr,
                         float* k5 = nullptr, float* k6 = nullptr) const;

  void set_distortion_coeffs(float k1, float k2, float p1, float p2,
                             float k3 = 0.0f, float k4 = 0.0f, float k5 = 0.0f,
                             float k6 = 0.0f);

  // To project 3D points to distorted image coordinate
  void Project(const Eigen::Vector3f& camera_p,
               Eigen::Vector3f* image_p) const override;
  void Project(const Eigen::Vector3f& camera_p,
               Eigen::Vector2f* image_p) const override;
  void Project(const Eigen::Vector3f& camera_p, Eigen::Vector2f* image_p,
               float* d) const override;

  // To recover 3D points in camera coordinate from distorted depth image
  void Unproject(const Eigen::Vector3f& image_p,
                 Eigen::Vector3f* camera_p) const override;
  void Unproject(const Eigen::Vector2f& image_p, float d,
                 Eigen::Vector3f* camera_p) const override;
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

  int width() const override;
  int height() const override;
  const Eigen::Affine3d& c2w() const override;
  const Eigen::Affine3d& w2c() const override;
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
  set_size(-1, -1);
  set_c2w(Eigen::Affine3d::Identity());
}

inline PinholeCamera::~PinholeCamera() {}

inline int PinholeCamera::width() const { return width_; }

inline int PinholeCamera::height() const { return height_; }

inline const Eigen::Affine3d& PinholeCamera::c2w() const { return c2w_; }

inline const Eigen::Affine3d& PinholeCamera::w2c() const { return w2c_; }

inline PinholeCamera::PinholeCamera(int width, int height, float fov_y_deg) {
  set_size(width, height);

  principal_point_[0] = width_ * 0.5f - 0.5f;
  principal_point_[1] = height_ * 0.5f - 0.5f;

  set_fov_y(fov_y_deg);

  set_c2w(Eigen::Affine3d::Identity());

  need_init_ray_table_ = true;
}

inline PinholeCamera::PinholeCamera(int width, int height,
                                    const Eigen::Affine3d& c2w,
                                    float fov_y_deg) {
  set_size(width, height);

  set_c2w(c2w);

  principal_point_[0] = width_ * 0.5f - 0.5f;
  principal_point_[1] = height_ * 0.5f - 0.5f;

  set_fov_y(fov_y_deg);

  need_init_ray_table_ = true;
}

inline PinholeCamera::PinholeCamera(int width, int height,
                                    const Eigen::Affine3d& c2w,
                                    const Eigen::Vector2f& principal_point,
                                    const Eigen::Vector2f& focal_length)
    : principal_point_(principal_point), focal_length_(focal_length) {
  set_size(width, height);
  set_c2w(c2w);
  need_init_ray_table_ = true;
}

inline void PinholeCamera::set_size(int width, int height) {
  width_ = width;
  height_ = height;
  need_init_ray_table_ = true;
}

inline void PinholeCamera::set_c2w(const Eigen::Affine3d& c2w) {
  c2w_ = c2w;
  w2c_ = c2w_.inverse();

  c2w_R_f_ = c2w_.matrix().block<3, 3>(0, 0).cast<float>();
  c2w_t_f_ = c2w_.matrix().block<3, 1>(0, 3).cast<float>();
  x_direc_ = c2w_.matrix().block<3, 3>(0, 0).col(0).cast<float>();
  y_direc_ = c2w_.matrix().block<3, 3>(0, 0).col(1).cast<float>();
  z_direc_ = c2w_.matrix().block<3, 3>(0, 0).col(2).cast<float>();

  w2c_R_f_ = w2c_.matrix().block<3, 3>(0, 0).cast<float>();
  w2c_t_f_ = w2c_.matrix().block<3, 1>(0, 3).cast<float>();

  need_init_ray_table_ = true;
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
  need_init_ray_table_ = true;
}

inline void PinholeCamera::set_focal_length(
    const Eigen::Vector2f& focal_length) {
  focal_length_ = focal_length;
  need_init_ray_table_ = true;
}

inline void PinholeCamera::set_fov_x(float fov_x_deg) {
  // same focal length per pixel for x and y
  focal_length_[0] =
      width_ * 0.5f /
      static_cast<float>(std::tan(radians<float>(fov_x_deg) * 0.5));
  focal_length_[1] = focal_length_[0];

  need_init_ray_table_ = true;
}

inline void PinholeCamera::set_fov_y(float fov_y_deg) {
  // same focal length per pixel for x and y

  focal_length_[1] =
      height_ * 0.5f /
      static_cast<float>(std::tan(radians<float>(fov_y_deg) * 0.5));
  focal_length_[0] = focal_length_[1];

  need_init_ray_table_ = true;
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
  if (need_init_ray_table_) {
    InitRayTable();
    need_init_ray_table_ = false;
  }
  *org = org_ray_c_table_[static_cast<size_t>(y) * static_cast<size_t>(width_) +
                          x];
}
inline void PinholeCamera::org_ray_w(int x, int y, Eigen::Vector3f* org) const {
  if (need_init_ray_table_) {
    InitRayTable();
    need_init_ray_table_ = false;
  }
  *org = org_ray_w_table_[static_cast<size_t>(y) * static_cast<size_t>(width_) +
                          x];
}

inline void PinholeCamera::ray_c(int x, int y, Eigen::Vector3f* dir) const {
  if (need_init_ray_table_) {
    InitRayTable();
    need_init_ray_table_ = false;
  }
  *dir = ray_c_table_[static_cast<size_t>(y) * static_cast<size_t>(width_) + x];
}
inline void PinholeCamera::ray_w(int x, int y, Eigen::Vector3f* dir) const {
  if (need_init_ray_table_) {
    InitRayTable();
    need_init_ray_table_ = false;
  }
  *dir = ray_w_table_[static_cast<size_t>(y) * static_cast<size_t>(width_) + x];
}

inline void PinholeCamera::InitRayTable() const {
  org_ray_c_table_.resize(static_cast<size_t>(width_) *
                          static_cast<size_t>(height_));
  org_ray_w_table_.resize(static_cast<size_t>(width_) *
                          static_cast<size_t>(height_));
  ray_c_table_.resize(static_cast<size_t>(width_) *
                      static_cast<size_t>(height_));
  ray_w_table_.resize(static_cast<size_t>(width_) *
                      static_cast<size_t>(height_));
#if defined(_OPENMP) && defined(UGU_USE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int y = 0; y < height_; y++) {
    for (int x = 0; x < width_; x++) {
      org_ray_c(static_cast<float>(x), static_cast<float>(y),
                &org_ray_c_table_[static_cast<size_t>(y) *
                                      static_cast<size_t>(width_) +
                                  x]);
      org_ray_w(static_cast<float>(x), static_cast<float>(y),
                &org_ray_w_table_[static_cast<size_t>(y) *
                                      static_cast<size_t>(width_) +
                                  x]);

      ray_c(static_cast<float>(x), static_cast<float>(y),
            &ray_c_table_[static_cast<size_t>(y) * static_cast<size_t>(width_) +
                          x]);
      ray_w(static_cast<float>(x), static_cast<float>(y),
            &ray_w_table_[static_cast<size_t>(y) * static_cast<size_t>(width_) +
                          x]);
    }
  }
}

inline void OpenCvCamera::distortion_coeffs(float* k1, float* k2, float* p1,
                                            float* p2, float* k3, float* k4,
                                            float* k5, float* k6) const {
  *k1 = k1_;
  *k2 = k2_;
  *p1 = p1_;
  *p2 = p2_;

  if (k3 != nullptr) {
    *k3 = k3_;
  }
  if (k4 != nullptr) {
    *k4 = k4_;
  }
  if (k5 != nullptr) {
    *k5 = k5_;
  }
  if (k6 != nullptr) {
    *k6 = k6_;
  }
}

inline void OpenCvCamera::set_distortion_coeffs(float k1, float k2, float p1,
                                                float p2, float k3, float k4,
                                                float k5, float k6) {
  k1_ = k1;
  k2_ = k2;
  p1_ = p1;
  p2_ = p2;

  k3_ = k3;
  k4_ = k4;
  k5_ = k5;
  k6_ = k6;
}

inline void OpenCvCamera::Project(const Eigen::Vector3f& camera_p,
                                  Eigen::Vector3f* image_p) const {
  (void)camera_p;
  (void)image_p;
  LOGE("HAVE NOT IMPLEMENTED\n");
}

inline void OpenCvCamera::Project(const Eigen::Vector3f& camera_p,
                                  Eigen::Vector2f* image_p) const {
  (void)camera_p;
  (void)image_p;
  LOGE("HAVE NOT IMPLEMENTED\n");
}

inline void OpenCvCamera::Project(const Eigen::Vector3f& camera_p,
                                  Eigen::Vector2f* image_p, float* d) const {
  (void)camera_p;
  (void)image_p;
  (void)d;
  LOGE("HAVE NOT IMPLEMENTED\n");
}

// To recover 3D points in camera coordinate from distorted depth image
inline void OpenCvCamera::Unproject(const Eigen::Vector3f& image_p,
                                    Eigen::Vector3f* camera_p) const {
  Eigen::Vector3f undistorted_image_p = image_p;

  UndistortPixelOpencv(&undistorted_image_p.x(), &undistorted_image_p.y(),
                       focal_length_.x(), focal_length_.y(),
                       principal_point_.x(), principal_point_.y(), k1_, k2_,
                       p1_, p2_, k3_, k4_, k5_, k6_);

  PinholeCamera::Unproject(undistorted_image_p, camera_p);
}

inline void OpenCvCamera::Unproject(const Eigen::Vector2f& image_p, float d,
                                    Eigen::Vector3f* camera_p) const {
  Eigen::Vector2f undistorted_image_p = image_p;

  UndistortPixelOpencv(&undistorted_image_p.x(), &undistorted_image_p.y(),
                       focal_length_.x(), focal_length_.y(),
                       principal_point_.x(), principal_point_.y(), k1_, k2_,
                       p1_, p2_, k3_, k4_, k5_, k6_);

  PinholeCamera::Unproject(undistorted_image_p, d, camera_p);
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

inline int OrthoCamera::width() const { return width_; }

inline int OrthoCamera::height() const { return height_; }

inline const Eigen::Affine3d& OrthoCamera::c2w() const { return c2w_; }

inline const Eigen::Affine3d& OrthoCamera::w2c() const { return w2c_; }

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
  *org = org_ray_c_table_[static_cast<size_t>(y) * static_cast<size_t>(width_) +
                          x];
}

inline void OrthoCamera::org_ray_w(int x, int y, Eigen::Vector3f* org) const {
  *org = org_ray_w_table_[static_cast<size_t>(y) * static_cast<size_t>(width_) +
                          x];
}

inline void OrthoCamera::ray_c(int x, int y, Eigen::Vector3f* dir) const {
  *dir = ray_c_table_[static_cast<size_t>(y) * static_cast<size_t>(width_) + x];
}
inline void OrthoCamera::ray_w(int x, int y, Eigen::Vector3f* dir) const {
  *dir = ray_w_table_[static_cast<size_t>(y) * static_cast<size_t>(width_) + x];
}

inline void OrthoCamera::InitRayTable() {
  org_ray_c_table_.resize(static_cast<size_t>(width_) *
                          static_cast<size_t>(height_));
  org_ray_w_table_.resize(static_cast<size_t>(width_) *
                          static_cast<size_t>(height_));
  ray_c_table_.resize(static_cast<size_t>(width_) *
                      static_cast<size_t>(height_));
  ray_w_table_.resize(static_cast<size_t>(width_) *
                      static_cast<size_t>(height_));
#if defined(_OPENMP) && defined(UGU_USE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int y = 0; y < height_; y++) {
    for (int x = 0; x < width_; x++) {
      org_ray_c(static_cast<float>(x), static_cast<float>(y),
                &org_ray_c_table_[static_cast<size_t>(y) *
                                      static_cast<size_t>(width_) +
                                  x]);
      org_ray_w(static_cast<float>(x), static_cast<float>(y),
                &org_ray_w_table_[static_cast<size_t>(y) *
                                      static_cast<size_t>(width_) +
                                  x]);

      ray_c(static_cast<float>(x), static_cast<float>(y),
            &ray_c_table_[static_cast<size_t>(y) * static_cast<size_t>(width_) +
                          x]);
      ray_w(static_cast<float>(x), static_cast<float>(y),
            &ray_w_table_[static_cast<size_t>(y) * static_cast<size_t>(width_) +
                          x]);
    }
  }
}

}  // namespace ugu
