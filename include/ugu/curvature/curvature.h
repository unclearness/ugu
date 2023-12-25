#pragma once

#include <vector>

#include "ugu/common.h"

namespace ugu {

bool CurvatureGaussian(const std::vector<Eigen::Vector3f>& vertices,
                       const std::vector<Eigen::Vector3i>& faces,
                       std::vector<float>& curvature,
                       std::vector<Eigen::Vector3f>& internal_angles);

std::vector<float> BarycentricCellArea(
    const std::vector<Eigen::Vector3f>& vertices,
    const std::vector<Eigen::Vector3i>& faces);

}