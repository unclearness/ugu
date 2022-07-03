
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

void DistortPixelOpencv(float* u, float* v, float fx, float fy, float cx,
                        float cy, float k1, float k2, float p1, float p2,
                        float k3 = 0.0f, float k4 = 0.0f, float k5 = 0.0f,
                        float k6 = 0.0f);

void UndistortPixelOpencv(float* u, float* v, float fx, float fy, float cx,
                          float cy, float k1, float k2, float p1, float p2,
                          float k3 = 0.0f, float k4 = 0.0f, float k5 = 0.0f,
                          float k6 = 0.0f);

}  // namespace ugu