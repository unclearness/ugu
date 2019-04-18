/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "currender/camera.h"

#include <fstream>
#include <iomanip>
#include <iostream>

namespace {
std::vector<std::string> Split(const std::string& input, char delimiter) {
  std::istringstream stream(input);
  std::string field;
  std::vector<std::string> result;
  while (std::getline(stream, field, delimiter)) {
    result.push_back(field);
  }
  return result;
}
}  // namespace

namespace currender {

Camera::Camera()
    : width_(-1),
      height_(-1),
      c2w_(Eigen::Affine3d::Identity()),
      w2c_(Eigen::Affine3d::Identity()) {
  set_c2w(c2w_);
}

Camera::~Camera() {}

Camera::Camera(int width, int height)
    : width_(width),
      height_(height),
      c2w_(Eigen::Affine3d::Identity()),
      w2c_(Eigen::Affine3d::Identity()) {
  set_c2w(c2w_);
}

Camera::Camera(int width, int height, const Eigen::Affine3d& c2w)
    : width_(width), height_(height), c2w_(c2w), w2c_(c2w_.inverse()) {
  set_c2w(c2w_);
}

int Camera::width() const { return width_; }

int Camera::height() const { return height_; }

const Eigen::Affine3d& Camera::c2w() const { return c2w_; }

const Eigen::Affine3d& Camera::w2c() const { return w2c_; }

void Camera::set_size(int width, int height) {
  width_ = width;
  height_ = height;
}
void Camera::set_c2w(const Eigen::Affine3d& c2w) {
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

PinholeCamera::PinholeCamera()
    : Camera(), principal_point_(-1, -1), focal_length_(-1, -1) {
  InitRayTable();
}

PinholeCamera::~PinholeCamera() {}

PinholeCamera::PinholeCamera(int width, int height)
    : Camera(width, height), principal_point_(-1, -1), focal_length_(-1, -1) {
  InitRayTable();
}

PinholeCamera::PinholeCamera(int width, int height, float fov_y_deg)
    : Camera(width, height) {
  principal_point_[0] = width_ * 0.5f - 0.5f;
  principal_point_[1] = height_ * 0.5f - 0.5f;

  set_fov_y(fov_y_deg);
  InitRayTable();
}

PinholeCamera::PinholeCamera(int width, int height, const Eigen::Affine3d& c2w)
    : Camera(width, height, c2w),
      principal_point_(-1, -1),
      focal_length_(-1, -1) {
  InitRayTable();
}

PinholeCamera::PinholeCamera(int width, int height, const Eigen::Affine3d& c2w,
                             float fov_y_deg)
    : Camera(width, height, c2w) {
  principal_point_[0] = width_ * 0.5f - 0.5f;
  principal_point_[1] = height_ * 0.5f - 0.5f;

  set_fov_y(fov_y_deg);
  InitRayTable();
}

PinholeCamera::PinholeCamera(int width, int height, const Eigen::Affine3d& c2w,
                             const Eigen::Vector2f& principal_point,
                             const Eigen::Vector2f& focal_length)
    : Camera(width, height, c2w),
      principal_point_(principal_point),
      focal_length_(focal_length) {
  InitRayTable();
}

void PinholeCamera::set_size(int width, int height) {
  Camera::set_size(width, height);

  InitRayTable();
}

void PinholeCamera::set_c2w(const Eigen::Affine3d& c2w) {
  Camera::set_c2w(c2w);

  InitRayTable();
}

float PinholeCamera::fov_x() const {
  return degrees<float>(2 * std::atan(width_ * 0.5f / focal_length_[0]));
}

float PinholeCamera::fov_y() const {
  return degrees<float>(2 * std::atan(height_ * 0.5f / focal_length_[1]));
}

const Eigen::Vector2f& PinholeCamera::principal_point() const {
  return principal_point_;
}

const Eigen::Vector2f& PinholeCamera::focal_length() const {
  return focal_length_;
}

void PinholeCamera::set_principal_point(
    const Eigen::Vector2f& principal_point) {
  principal_point_ = principal_point;
  InitRayTable();
}

void PinholeCamera::set_focal_length(const Eigen::Vector2f& focal_length) {
  focal_length_ = focal_length;
  InitRayTable();
}

void PinholeCamera::set_fov_x(float fov_x_deg) {
  // same focal length per pixel for x and y
  focal_length_[0] =
      width_ * 0.5f /
      static_cast<float>(std::tan(radians<float>(fov_x_deg) * 0.5));
  focal_length_[1] = focal_length_[0];
  InitRayTable();
}

void PinholeCamera::set_fov_y(float fov_y_deg) {
  // same focal length per pixel for x and y
  focal_length_[1] =
      height_ * 0.5f /
      static_cast<float>(std::tan(radians<float>(fov_y_deg) * 0.5));
  focal_length_[0] = focal_length_[1];
  InitRayTable();
}

void PinholeCamera::Project(const Eigen::Vector3f& camera_p,
                            Eigen::Vector3f* image_p) const {
  (*image_p)[0] =
      focal_length_[0] / camera_p[2] * camera_p[0] + principal_point_[0];
  (*image_p)[1] =
      focal_length_[1] / camera_p[2] * camera_p[1] + principal_point_[1];
  (*image_p)[2] = camera_p[2];
}

void PinholeCamera::Project(const Eigen::Vector3f& camera_p,
                            Eigen::Vector2f* image_p) const {
  (*image_p)[0] =
      focal_length_[0] / camera_p[2] * camera_p[0] + principal_point_[0];
  (*image_p)[1] =
      focal_length_[1] / camera_p[2] * camera_p[1] + principal_point_[1];
}

void PinholeCamera::Project(const Eigen::Vector3f& camera_p,
                            Eigen::Vector2f* image_p, float* d) const {
  (*image_p)[0] =
      focal_length_[0] / camera_p[2] * camera_p[0] + principal_point_[0];
  (*image_p)[1] =
      focal_length_[1] / camera_p[2] * camera_p[1] + principal_point_[1];
  *d = camera_p[2];
}

void PinholeCamera::Unproject(const Eigen::Vector3f& image_p,
                              Eigen::Vector3f* camera_p) const {
  (*camera_p)[0] =
      (image_p[0] - principal_point_[0]) * image_p[2] / focal_length_[0];
  (*camera_p)[1] =
      (image_p[1] - principal_point_[1]) * image_p[2] / focal_length_[1];
  (*camera_p)[2] = image_p[2];
}

void PinholeCamera::Unproject(const Eigen::Vector2f& image_p, float d,
                              Eigen::Vector3f* camera_p) const {
  (*camera_p)[0] = (image_p[0] - principal_point_[0]) * d / focal_length_[0];
  (*camera_p)[1] = (image_p[1] - principal_point_[1]) * d / focal_length_[1];
  (*camera_p)[2] = d;
}

void PinholeCamera::org_ray_c(float x, float y, Eigen::Vector3f* org) const {
  (void)x;
  (void)y;
  (*org)[0] = 0.0f;
  (*org)[1] = 0.0f;
  (*org)[2] = 0.0f;
}

void PinholeCamera::org_ray_w(float x, float y, Eigen::Vector3f* org) const {
  (void)x;
  (void)y;
  *org = c2w_t_f_;
}

void PinholeCamera::ray_c(float x, float y, Eigen::Vector3f* dir) const {
  (*dir)[0] = (x - principal_point_[0]) / focal_length_[0];
  (*dir)[1] = (y - principal_point_[1]) / focal_length_[1];
  (*dir)[2] = 1.0f;
  dir->normalize();
}

void PinholeCamera::ray_w(float x, float y, Eigen::Vector3f* dir) const {
  ray_c(x, y, dir);
  *dir = c2w_R_f_ * *dir;
}

void PinholeCamera::org_ray_c(int x, int y, Eigen::Vector3f* org) const {
  *org = org_ray_c_table_[y * width_ + x];
}
void PinholeCamera::org_ray_w(int x, int y, Eigen::Vector3f* org) const {
  *org = org_ray_w_table_[y * width_ + x];
}

void PinholeCamera::ray_c(int x, int y, Eigen::Vector3f* dir) const {
  *dir = ray_c_table_[y * width_ + x];
}
void PinholeCamera::ray_w(int x, int y, Eigen::Vector3f* dir) const {
  *dir = ray_w_table_[y * width_ + x];
}

void PinholeCamera::InitRayTable() {
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

OrthoCamera::OrthoCamera() : Camera() { InitRayTable(); }
OrthoCamera::~OrthoCamera() {}
OrthoCamera::OrthoCamera(int width, int height) : Camera(width, height) {
  InitRayTable();
}
OrthoCamera::OrthoCamera(int width, int height, const Eigen::Affine3d& c2w)
    : Camera(width, height, c2w) {
  InitRayTable();
}

void OrthoCamera::set_size(int width, int height) {
  Camera::set_size(width, height);

  InitRayTable();
}

void OrthoCamera::set_c2w(const Eigen::Affine3d& c2w) {
  Camera::set_c2w(c2w);

  InitRayTable();
}

void OrthoCamera::Project(const Eigen::Vector3f& camera_p,
                          Eigen::Vector3f* image_p) const {
  *image_p = camera_p;
}

void OrthoCamera::Project(const Eigen::Vector3f& camera_p,
                          Eigen::Vector2f* image_p) const {
  (*image_p)[0] = camera_p[0];
  (*image_p)[1] = camera_p[1];
}

void OrthoCamera::Project(const Eigen::Vector3f& camera_p,
                          Eigen::Vector2f* image_p, float* d) const {
  (*image_p)[0] = camera_p[0];
  (*image_p)[1] = camera_p[1];
  *d = camera_p[2];
}

void OrthoCamera::Unproject(const Eigen::Vector3f& image_p,
                            Eigen::Vector3f* camera_p) const {
  *camera_p = image_p;
}

void OrthoCamera::Unproject(const Eigen::Vector2f& image_p, float d,
                            Eigen::Vector3f* camera_p) const {
  (*camera_p)[0] = image_p[0];
  (*camera_p)[1] = image_p[1];
  (*camera_p)[2] = d;
}

void OrthoCamera::org_ray_c(float x, float y, Eigen::Vector3f* org) const {
  (*org)[0] = x - width_ / 2;
  (*org)[1] = y - height_ / 2;
  (*org)[2] = 0.0f;
}

void OrthoCamera::org_ray_w(float x, float y, Eigen::Vector3f* org) const {
  *org = c2w_t_f_;

  Eigen::Vector3f offset_x = (x - width_ * 0.5f) * x_direc_;
  Eigen::Vector3f offset_y = (y - height_ * 0.5f) * y_direc_;

  *org += offset_x;
  *org += offset_y;
}

void OrthoCamera::ray_c(float x, float y, Eigen::Vector3f* dir) const {
  (void)x;
  (void)y;
  // parallell ray along with z axis
  (*dir)[0] = 0.0f;
  (*dir)[1] = 0.0f;
  (*dir)[2] = 1.0f;
}

void OrthoCamera::ray_w(float x, float y, Eigen::Vector3f* dir) const {
  (void)x;
  (void)y;
  // extract z direction of camera pose
  *dir = z_direc_;
}

void OrthoCamera::org_ray_c(int x, int y, Eigen::Vector3f* org) const {
  *org = org_ray_c_table_[y * width_ + x];
}

void OrthoCamera::org_ray_w(int x, int y, Eigen::Vector3f* org) const {
  *org = org_ray_w_table_[y * width_ + x];
}

void OrthoCamera::ray_c(int x, int y, Eigen::Vector3f* dir) const {
  *dir = ray_c_table_[y * width_ + x];
}
void OrthoCamera::ray_w(int x, int y, Eigen::Vector3f* dir) const {
  *dir = ray_w_table_[y * width_ + x];
}

void OrthoCamera::InitRayTable() {
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

void WriteTumFormat(const std::vector<Eigen::Affine3d>& poses,
                    const std::string& path) {
  std::ofstream ofs;
  ofs.open(path, std::ios::out);

  ofs << std::fixed << std::setprecision(6);
  for (size_t i = 0; i < poses.size(); i++) {
    const Eigen::Quaterniond q(poses[i].rotation());
    const Eigen::Vector3d t = poses[i].translation();
    ofs << zfill(i) << " " << t[0] << " " << t[1] << " " << t[2] << " " << q.x()
        << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
  }
}

bool LoadTumFormat(const std::string& path,
                   std::vector<std::pair<int, Eigen::Affine3d>>* poses) {
  poses->clear();

  std::ifstream ifs(path);

  std::string line;
  while (std::getline(ifs, line)) {
    std::vector<std::string> splited = Split(line, ' ');
    if (splited.size() != 8) {
      LOGE("wrong tum format\n");
      return false;
    }

    std::pair<int, Eigen::Affine3d> pose;
    pose.first = std::atoi(splited[0].c_str());

    Eigen::Translation3d t;
    t.x() = std::atof(splited[1].c_str());
    t.y() = std::atof(splited[2].c_str());
    t.z() = std::atof(splited[3].c_str());

    Eigen::Quaterniond q;
    q.x() = std::atof(splited[4].c_str());
    q.y() = std::atof(splited[5].c_str());
    q.z() = std::atof(splited[6].c_str());
    q.w() = std::atof(splited[7].c_str());

    pose.second = t * q;

    poses->push_back(pose);
  }

  return true;
}

bool LoadTumFormat(const std::string& path,
                   std::vector<Eigen::Affine3d>* poses) {
  std::vector<std::pair<int, Eigen::Affine3d>> tmp_poses;
  bool ret = LoadTumFormat(path, &tmp_poses);
  if (!ret) {
    return false;
  }

  poses->clear();
  for (const auto& pose_pair : tmp_poses) {
    poses->push_back(pose_pair.second);
  }

  return true;
}

}  // namespace currender
