#include "include/camera.h"

namespace crender {
Camera::Camera() {}

Camera::~Camera() {}

Camera::Camera(int width, int height, const Pose& c2w)
    : width_(width), height_(height), c2w_(c2w) {
  w2c_.set_T(glm::inverse(c2w_.T()));
}

int Camera::width() { return width_; }

int Camera::height() { return height_; }

const Pose& Camera::c2w() { return c2w_; }

const Pose& Camera::w2c() { return w2c_; }

void Camera::set_size(int width, int height) {
  width_ = width;
  height_ = height;
}
void Camera::set_c2w(const Pose& c2w) {
  c2w_ = c2w;
  w2c_.set_T(glm::inverse(c2w_.T()));
}

PinholeCamera::PinholeCamera() {}

PinholeCamera::~PinholeCamera() {}

PinholeCamera::PinholeCamera(int width, int height, const Pose& c2w,
                             const glm::vec2& principal_point,
                             const glm::vec2& focal_length)
    : Camera(width, height, c2w),
      principal_point_(principal_point),
      focal_length_(focal_length) {}

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

void PinholeCamera::project(const glm::vec3& camera_p, glm::vec3& image_p) {
  image_p[0] =
      focal_length_[0] / camera_p[2] * camera_p[0] + principal_point_[0];
  image_p[1] =
      focal_length_[1] / camera_p[2] * camera_p[1] + principal_point_[1];
  image_p[2] = camera_p[2];
}

void PinholeCamera::project(const glm::vec3& camera_p, glm::vec2& image_p) {
  image_p[0] =
      focal_length_[0] / camera_p[2] * camera_p[0] + principal_point_[0];
  image_p[1] =
      focal_length_[1] / camera_p[2] * camera_p[1] + principal_point_[1];
}

void PinholeCamera::project(const glm::vec3& camera_p, glm::vec2& image_p,
                            float& d) {
  image_p[0] =
      focal_length_[0] / camera_p[2] * camera_p[0] + principal_point_[0];
  image_p[1] =
      focal_length_[1] / camera_p[2] * camera_p[1] + principal_point_[1];
  d = camera_p[2];
}

void PinholeCamera::unproject(const glm::vec3& image_p, glm::vec3& camera_p) {
  camera_p[0] =
      (image_p[0] - principal_point_[0]) * image_p[2] / focal_length_[0];
  camera_p[1] =
      (image_p[1] - principal_point_[1]) * image_p[2] / focal_length_[1];
  camera_p[2] = image_p[2];
}

void PinholeCamera::unproject(const glm::vec2& image_p, float d,
                              glm::vec3& camera_p) {
  camera_p[0] = (image_p[0] - principal_point_[0]) * d / focal_length_[0];
  camera_p[1] = (image_p[1] - principal_point_[1]) * d / focal_length_[1];
  camera_p[2] = d;
}

void PinholeCamera::ray_c(float x, float y, glm::vec3& dir) {
  dir[0] = (x - principal_point_[0]) / focal_length_[0];
  dir[1] = (y - principal_point_[1]) / focal_length_[1];
  dir[2] = 1.0f;
  dir = glm::normalize(dir);
}

void PinholeCamera::ray_w(float x, float y, glm::vec3& dir) {
  ray_c(x, y, dir);
  dir = c2w_.R() * dir;
}
};  // namespace crender
