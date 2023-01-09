
/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#pragma once

#include <string>
#include <vector>

#include "Eigen/Geometry"

namespace ugu {
void WriteTumFormat(const std::vector<Eigen::Affine3d>& poses,
                    const std::string& path);
bool LoadTumFormat(const std::string& path,
                   std::vector<Eigen::Affine3d>* poses);
bool LoadTumFormat(const std::string& path,
                   std::vector<std::pair<int, Eigen::Affine3d>>* poses);
bool LoadTumFormatExtend(
    const std::string& path,
    std::vector<std::tuple<int, Eigen::Affine3d, std::vector<std::string>>>*
        poses);

void c2w(const Eigen::Vector3f& position, const Eigen::Vector3f& target,
         const Eigen::Vector3f& up, Eigen::Matrix3f* R);
void c2w(const Eigen::Vector3d& position, const Eigen::Vector3d& target,
         const Eigen::Vector3d& up, Eigen::Matrix3d* R);

void c2w(const Eigen::Vector3f& position, const Eigen::Vector3f& target,
         const Eigen::Vector3f& up, Eigen::Matrix4f* R);
void c2w(const Eigen::Vector3d& position, const Eigen::Vector3d& target,
         const Eigen::Vector3d& up, Eigen::Matrix4d* R);

Eigen::Affine3d ConvertCvAndGlWldToCam(const Eigen::Affine3d& w2c);
Eigen::Affine3d ConvertCvAndGlCamToWld(const Eigen::Affine3d& c2w);
Eigen::Affine3d ConvertCvAndGlCamToWldRotOnly(const Eigen::Affine3d& c2w);

Eigen::Matrix4f GetProjectionMatrixOpenGl(float l, float r, float b, float t,
                                          float n, float f);
Eigen::Matrix4f GetProjectionMatrixOpenGl(float fovY, float aspect,
                                          float z_near, float z_far);
Eigen::Matrix4f GetProjectionMatrixOpenGlForPinhole(int width, int height,
                                                    float fx, float fy,
                                                    float cx, float cy,
                                                    float z_near, float z_far);

void DistortPixelOpencv(float* u, float* v, float fx, float fy, float cx,
                        float cy, float k1, float k2, float p1, float p2,
                        float k3 = 0.0f, float k4 = 0.0f, float k5 = 0.0f,
                        float k6 = 0.0f);

void UndistortPixelOpencv(float* u, float* v, float fx, float fy, float cx,
                          float cy, float k1, float k2, float p1, float p2,
                          float k3 = 0.0f, float k4 = 0.0f, float k5 = 0.0f,
                          float k6 = 0.0f);

}  // namespace ugu