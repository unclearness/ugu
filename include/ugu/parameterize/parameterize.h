/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/mesh.h"

namespace ugu {

enum ParameterizeUvType { kSimpleTriangles = 0, kSmartUv = 1 };

bool Parameterize(
    Mesh& mesh, int tex_w = 1024, int tex_h = 1024,
    ParameterizeUvType type = ParameterizeUvType::kSimpleTriangles);

bool Parameterize(
    const std::vector<Eigen::Vector3f>& vertices,
    const std::vector<Eigen::Vector3i>& faces,
    const std::vector<Eigen::Vector3f>& face_normals,
    std::vector<Eigen::Vector2f>& uvs, std::vector<Eigen::Vector3i>& uv_faces,
    int tex_w = 1024, int tex_h = 1024,
    ParameterizeUvType type = ParameterizeUvType::kSimpleTriangles);

bool OrthoProjectToXY(const Eigen::Vector3f& project_normal,
                      const std::vector<Eigen::Vector3f>& points_3d,
                      std::vector<Eigen::Vector2f>& points_2d,
                      bool align_longest_axis_x = true, bool normalize = true,
                      bool keep_aspect = true, bool align_top_y = true);
}  // namespace ugu