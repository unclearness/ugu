/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/util/camera_util.h"

#include <fstream>
#include <iomanip>
#include <iostream>

#include "ugu/log.h"
#include "ugu/util/math_util.h"
#include "ugu/util/string_util.h"

namespace {
// https://stackoverflow.com/questions/13768423/setting-up-projection-model-and-view-transformations-for-vertex-shader-in-eige
template <typename T>
void c2wImpl(const Eigen::Matrix<T, 3, 1>& position,
             const Eigen::Matrix<T, 3, 1>& target,
             const Eigen::Matrix<T, 3, 1>& up, Eigen::Matrix<T, 3, 3>* R) {
  static_assert(std::numeric_limits<T>::is_iec559);

  R->col(2) = (target - position).normalized();
  R->col(0) = R->col(2).cross(up).normalized();
  R->col(1) = R->col(2).cross(R->col(0));
}

template <typename genType>
void c2wImpl(const Eigen::Matrix<genType, 3, 1>& position,
             const Eigen::Matrix<genType, 3, 1>& target,
             const Eigen::Matrix<genType, 3, 1>& up,
             Eigen::Matrix<genType, 4, 4>* T) {
  static_assert(std::numeric_limits<genType>::is_iec559);

  *T = Eigen::Matrix<genType, 4, 4>::Identity();

  Eigen::Matrix<genType, 3, 3> R;
  c2wImpl(position, target, up, &R);

  T->topLeftCorner(3, 3) = R;
  T->topRightCorner(3, 1) = position;
}

}  // namespace

namespace ugu {

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
    if (line.substr(0, 1) == "#") {
      continue;
    }

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

bool LoadTumFormatExtend(
    const std::string& path,
    std::vector<std::tuple<int, Eigen::Affine3d, std::vector<std::string>>>*
        poses) {
  poses->clear();

  std::ifstream ifs(path);

  std::string line;
  while (std::getline(ifs, line)) {
    if (line.substr(0, 1) == "#") {
      continue;
    }

    std::vector<std::string> splited = Split(line, ' ');
    if (splited.size() != 8) {
      LOGE("wrong tum format\n");
      return false;
    }

    int ts = std::atoi(splited[0].c_str());

    Eigen::Translation3d t;
    t.x() = std::atof(splited[1].c_str());
    t.y() = std::atof(splited[2].c_str());
    t.z() = std::atof(splited[3].c_str());

    Eigen::Quaterniond q;
    q.x() = std::atof(splited[4].c_str());
    q.y() = std::atof(splited[5].c_str());
    q.z() = std::atof(splited[6].c_str());
    q.w() = std::atof(splited[7].c_str());

    Eigen::Affine3d c2w = t * q;

    std::vector<std::string> additional;
    for (size_t i = 8; i < splited.size(); i++) {
      additional.push_back(splited[i]);
    }

    poses->push_back(std::make_tuple(ts, c2w, additional));
  }

  return true;
}

void c2w(const Eigen::Vector3f& position, const Eigen::Vector3f& target,
         const Eigen::Vector3f& up, Eigen::Matrix3f* R) {
  c2wImpl(position, target, up, R);
}
void c2w(const Eigen::Vector3d& position, const Eigen::Vector3d& target,
         const Eigen::Vector3d& up, Eigen::Matrix3d* R) {
  c2wImpl(position, target, up, R);
}

void c2w(const Eigen::Vector3f& position, const Eigen::Vector3f& target,
         const Eigen::Vector3f& up, Eigen::Matrix4f* R) {
  c2wImpl(position, target, up, R);
}

void c2w(const Eigen::Vector3d& position, const Eigen::Vector3d& target,
         const Eigen::Vector3d& up, Eigen::Matrix4d* R) {
  c2wImpl(position, target, up, R);
}

Eigen::Affine3d ConvertCvAndGlWldToCam(const Eigen::Affine3d& w2c) {
  const Eigen::Affine3d offset =
      Eigen::Affine3d(Eigen::AngleAxisd(pi, Eigen::Vector3d::UnitX()));
  return offset * w2c;
}

Eigen::Affine3d ConvertCvAndGlCamToWld(const Eigen::Affine3d& c2w) {
  const Eigen::Affine3d offset =
      Eigen::Affine3d(Eigen::AngleAxisd(pi, Eigen::Vector3d::UnitX()))
          .inverse();
  return c2w * offset;
}

Eigen::Affine3d ConvertCvAndGlCamToWldRotOnly(const Eigen::Affine3d& c2w) {
  const Eigen::Affine3d offset =
      Eigen::Affine3d(Eigen::AngleAxisd(pi, Eigen::Vector3d::UnitX()))
          .inverse();
  Eigen::Affine3d rot_only = Eigen::Translation3d(c2w.translation()) *
                             (c2w.rotation() * offset.rotation());
  return rot_only;
}

Eigen::Matrix4f GetProjectionMatrixOpenGl(float l, float r, float b, float t,
                                          float n, float f) {
  Eigen::Matrix4f mat;
  mat.setZero();
  mat.data()[0] = 2 * n / (r - l);
  mat.data()[5] = 2 * n / (t - b);
  mat.data()[8] = (r + l) / (r - l);
  mat.data()[9] = (t + b) / (t - b);
  mat.data()[10] = -(f + n) / (f - n);
  mat.data()[11] = -1;
  mat.data()[14] = -(2 * f * n) / (f - n);
  mat.data()[15] = 0;
  return mat;
}

Eigen::Matrix4f GetProjectionMatrixOpenGl(float fovY, float aspect,
                                          float z_near, float z_far) {
  float tangent = std::tan(ugu::radians(fovY / 2));  // tangent of half fovY
  float height = z_near * tangent;  // half height of near plane
  float width = height * aspect;    // half width of near plane

  // params: left, right, bottom, top, near, far
  return GetProjectionMatrixOpenGl(-width, width, -height, height, z_near,
                                   z_far);
}

Eigen::Matrix4f GetProjectionMatrixOpenGlForPinhole(int width, int height,
                                                    float fx, float fy,
                                                    float cx, float cy,
                                                    float z_near, float z_far) {
  Eigen::Matrix4f mat;
  mat.setZero();
  float w = static_cast<float>(width);
  float h = static_cast<float>(height);

  mat(0, 0) = 2.f * fx / w;
  mat(1, 1) = 2.f * fy / h;

  mat(0, 2) = (w - 2.f * cx) / w;
  mat(1, 2) = -(h - 2.f * cy) / h;
  mat(2, 2) = -(z_far + z_near) / (z_far - z_near);
  mat(2, 3) = -(2.f * z_far * z_near) / (z_far - z_near);

  mat(3, 2) = -1.f;

  return mat;
}

Eigen::Matrix4f GetProjectionMatrixOpenGlForOrtho(float xmin, float xmax,
                                                  float ymin, float ymax,
                                                  float zmin, float zmax) {
  Eigen::Matrix4f mat;
  mat.setZero();

  mat(0, 0) = 2.f / (xmax - xmin);
  mat(1, 1) = 2.f / (ymax - ymin);
  mat(2, 2) = -2.f / (zmax - zmin);

  mat(0, 3) = -(xmax + xmin) / (xmax - xmin);
  mat(1, 3) = -(ymax + ymin) / (ymax - ymin);
  mat(2, 3) = -(zmax + zmin) / (zmax - zmin);
  mat(3, 3) = 1.f;

  return mat;
}

// https://docs.opencv.org/4.3.0/d9/d0c/group__calib3d.html
void DistortPixelOpencv(float* u, float* v, float fx, float fy, float cx,
                        float cy, float k1, float k2, float p1, float p2,
                        float k3, float k4, float k5, float k6) {
  double u1 = (*u - cx) / fx;
  double v1 = (*v - cy) / fy;
  double u2 = u1 * u1;
  double v2 = v1 * v1;
  double r2 = u2 + v2;

  // https://github.com/egonSchiele/OpenCV/blob/master/modules/imgproc/src/undistort.cpp#L133
  double _2uv = 2 * u1 * v1;
  double kr = (1 + ((k3 * r2 + k2) * r2 + k1) * r2) /
              (1 + ((k6 * r2 + k5) * r2 + k4) * r2);
  *u = static_cast<float>(fx * (u1 * kr + p1 * _2uv + p2 * (r2 + 2 * u2)) + cx);
  *v = static_cast<float>(fy * (v1 * kr + p1 * (r2 + 2 * v2) + p2 * _2uv) + cy);
}

// https://docs.opencv.org/4.3.0/d9/d0c/group__calib3d.html
void UndistortPixelOpencv(float* u, float* v, float fx, float fy, float cx,
                          float cy, float k1, float k2, float p1, float p2,
                          float k3, float k4, float k5, float k6) {
  // https://github.com/egonSchiele/OpenCV/blob/master/modules/imgproc/src/undistort.cpp#L345
  // https://github.com/opencv/opencv/blob/master/modules/calib3d/src/undistort.dispatch.cpp#L385
  const double x0 = (*u - cx) / fx;
  const double y0 = (*v - cy) / fy;
  double x = x0;
  double y = y0;
  // Compensate distortion iteratively
  // 5 is from OpenCV code.
  // I don't know theoritical rationale why 5 is enough...
  const int max_iter = 5;
  double p1d = p1;
  double p2d = p2;
  for (int j = 0; j < max_iter; j++) {
    double r2 = x * x + y * y;
    double icdist = (1.0 + ((k6 * r2 + k5) * r2 + k4) * r2) /
                    (1.0 + ((k3 * r2 + k2) * r2 + k1) * r2);
    double deltaX = 2.0 * p1d * x * y + p2d * (r2 + 2 * x * x);
    double deltaY = p1d * (r2 + 2.0 * y * y) + 2.0 * p2d * x * y;
    x = (x0 - deltaX) * icdist;
    y = (y0 - deltaY) * icdist;
  }

  *u = static_cast<float>(x * fx + cx);
  *v = static_cast<float>(y * fy + cy);
}

}  // namespace ugu
