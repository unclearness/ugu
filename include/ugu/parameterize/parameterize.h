/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/mesh.h"

namespace ugu {

enum class ParameterizeUvType { kSimpleTriangles = 0, kSmartUv = 1 };

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

bool PackUvIslands(
    const std::vector<float>& cluster_areas,
    const std::vector<std::vector<Eigen::Vector3i>>& clusters,
    const std::vector<std::vector<Eigen::Vector2f>>& cluster_uvs,
    const std::vector<std::vector<Eigen::Vector3i>>& cluster_sub_faces,
    const std::vector<std::vector<uint32_t>>& cluster_fids, size_t num_faces,
    int tex_w, int tex_h, std::vector<Eigen::Vector2f>& uvs,
    std::vector<Eigen::Vector3i>& uv_faces, bool flip_v = true,
    const std::vector<float>& cluster_weights = std::vector<float>());

}  // namespace ugu