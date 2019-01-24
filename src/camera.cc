/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "include/camera.h"

#include <cmath>

namespace currender {
Camera::Camera() : width_(-1), height_(-1), c2w_(Pose()), w2c_(Pose()) {}

Camera::~Camera() {}

Camera::Camera(int width, int height)
    : width_(width), height_(height), c2w_(Pose()), w2c_(Pose()) {}

Camera::Camera(int width, int height, const Pose& c2w)
    : width_(width), height_(height), c2w_(c2w) {
  w2c_.set_T(glm::inverse(c2w_.T()));
}

int Camera::width() const { return width_; }

int Camera::height() const { return height_; }

const Pose& Camera::c2w() const { return c2w_; }

const Pose& Camera::w2c() const { return w2c_; }

void Camera::set_size(int width, int height) {
  width_ = width;
  height_ = height;
}
void Camera::set_c2w(const Pose& c2w) {
  c2w_ = c2w;
  w2c_.set_T(glm::inverse(c2w_.T()));
}

PinholeCamera::PinholeCamera()
    : Camera(), principal_point_(-1, -1), focal_length_(-1, -1) {}

PinholeCamera::~PinholeCamera() {}

PinholeCamera::PinholeCamera(int width, int height)
    : Camera(width, height), principal_point_(-1, -1), focal_length_(-1, -1) {}

PinholeCamera::PinholeCamera(int width, int height, float fov_y_deg)
    : Camera(width, height) {
  principal_point_[0] = width_ * 0.5f;
  principal_point_[1] = height_ * 0.5f;

  set_fov_y(fov_y_deg);
}

PinholeCamera::PinholeCamera(int width, int height, const Pose& c2w)
    : Camera(width, height, c2w),
      principal_point_(-1, -1),
      focal_length_(-1, -1) {}

PinholeCamera::PinholeCamera(int width, int height, const Pose& c2w,
                             float fov_y_deg)
    : Camera(width, height, c2w) {
  principal_point_[0] = width_ * 0.5f;
  principal_point_[1] = height_ * 0.5f;

  set_fov_y(fov_y_deg);
}

PinholeCamera::PinholeCamera(int width, int height, const Pose& c2w,
                             const glm::vec2& principal_point,
                             const glm::vec2& focal_length)
    : Camera(width, height, c2w),
      principal_point_(principal_point),
      focal_length_(focal_length) {}

float PinholeCamera::fov_x() const {
  return glm::degrees<float>(2 * std::atan(width_ * 0.5f / focal_length_[0]));
}

float PinholeCamera::fov_y() const {
  return glm::degrees<float>(2 * std::atan(height_ * 0.5f / focal_length_[1]));
}

const glm::vec2& PinholeCamera::principal_point() const {
  return principal_point_;
}

const glm::vec2& PinholeCamera::focal_length() const { return focal_length_; }

void PinholeCamera::set_principal_point(const glm::vec2& principal_point) {
  principal_point_ = principal_point;
}

void PinholeCamera::set_focal_length(const glm::vec2& focal_length) {
  focal_length_ = focal_length;
}

void PinholeCamera::set_fov_x(float fov_x_deg) {
  focal_length_[0] =
      width_ * 0.5f /
      static_cast<float>(std::tan(glm::radians<float>(fov_x_deg) * 0.5));
  focal_length_[1] = focal_length_[0] / static_cast<float>(width_) *
                     static_cast<float>(height_);
}

void PinholeCamera::set_fov_y(float fov_y_deg) {
  focal_length_[1] =
      height_ * 0.5f /
      static_cast<float>(std::tan(glm::radians<float>(fov_y_deg) * 0.5));
  focal_length_[0] = focal_length_[1] / static_cast<float>(height_) *
                     static_cast<float>(width_);
}

void PinholeCamera::Project(const glm::vec3& camera_p,
                            glm::vec3* image_p) const {
  (*image_p)[0] =
      focal_length_[0] / camera_p[2] * camera_p[0] + principal_point_[0];
  (*image_p)[1] =
      focal_length_[1] / camera_p[2] * camera_p[1] + principal_point_[1];
  (*image_p)[2] = camera_p[2];
}

void PinholeCamera::Project(const glm::vec3& camera_p,
                            glm::vec2* image_p) const {
  (*image_p)[0] =
      focal_length_[0] / camera_p[2] * camera_p[0] + principal_point_[0];
  (*image_p)[1] =
      focal_length_[1] / camera_p[2] * camera_p[1] + principal_point_[1];
}

void PinholeCamera::Project(const glm::vec3& camera_p, glm::vec2* image_p,
                            float* d) const {
  (*image_p)[0] =
      focal_length_[0] / camera_p[2] * camera_p[0] + principal_point_[0];
  (*image_p)[1] =
      focal_length_[1] / camera_p[2] * camera_p[1] + principal_point_[1];
  *d = camera_p[2];
}

void PinholeCamera::Unproject(const glm::vec3& image_p,
                              glm::vec3* camera_p) const {
  (*camera_p)[0] =
      (image_p[0] - principal_point_[0]) * image_p[2] / focal_length_[0];
  (*camera_p)[1] =
      (image_p[1] - principal_point_[1]) * image_p[2] / focal_length_[1];
  (*camera_p)[2] = image_p[2];
}

void PinholeCamera::Unproject(const glm::vec2& image_p, float d,
                              glm::vec3* camera_p) const {
  (*camera_p)[0] = (image_p[0] - principal_point_[0]) * d / focal_length_[0];
  (*camera_p)[1] = (image_p[1] - principal_point_[1]) * d / focal_length_[1];
  (*camera_p)[2] = d;
}

void PinholeCamera::ray_c(float x, float y, glm::vec3* dir) const {
  (*dir)[0] = (x - principal_point_[0]) / focal_length_[0];
  (*dir)[1] = (y - principal_point_[1]) / focal_length_[1];
  (*dir)[2] = 1.0f;
  *dir = glm::normalize(*dir);
}

void PinholeCamera::ray_w(float x, float y, glm::vec3* dir) const {
  ray_c(x, y, dir);
  *dir = c2w_.R() * *dir;
}
}  // namespace currender
