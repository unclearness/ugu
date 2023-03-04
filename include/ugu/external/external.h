/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 *
 */

#pragma once

#include "ugu/mesh.h"
#include "ugu/texturing/texture_mapper.h"

namespace ugu {

bool FastQuadricMeshSimplification(const Mesh& src, int target_face_num,
                                   Mesh* decimated);

bool MvsTexturing(const std::vector<std::shared_ptr<Keyframe>>& keyframes,
                  Mesh* mesh, Mesh* debug_mesh = nullptr,
                  const std::string& save_path = "",
                  const std::string& save_debug_path = "");

bool LibiglLscm(const std::vector<Eigen::Vector3f>& vertices,
                const std::vector<Eigen::Vector3i>& vertex_indices,
                const std::vector<int>& boundary,
                std::vector<Eigen::Vector2f>& uvs);

bool LibiglLscm(const std::vector<Eigen::Vector3f>& vertices,
                const std::vector<Eigen::Vector3i>& vertex_indices, int tex_w,
                int tex_h, const std::vector<float>& cluster_areas,
                const std::vector<std::vector<Eigen::Vector3i>>& clusters,
                const std::vector<std::vector<uint32_t>>& cluster_fids,
                const std::vector<float>& cluster_weights,
                std::vector<Eigen::Vector2f>& uvs,
                std::vector<Eigen::Vector3i>& uv_indices);

bool LibiglLscm(const std::vector<Eigen::Vector3f>& vertices,
                const std::vector<Eigen::Vector3i>& vertex_indices, int tex_w,
                int tex_h, std::vector<Eigen::Vector2f>& uvs,
                std::vector<Eigen::Vector3i>& uv_indices);

bool LibiglLscm(Mesh& mesh, int tex_w, int tex_h);

MeshPtr PoissonRecon(const std::vector<Eigen::Vector3f>& points,
                     const std::vector<Eigen::Vector3f>& normals,
                     const std::vector<Eigen::Vector3f>& colors =
                         std::vector<Eigen::Vector3f>());
MeshPtr PoissonRecon(const MeshPtr& src);

}  // namespace ugu
