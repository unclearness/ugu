/*
 * Copyright (C) 2024, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/mesh.h"

namespace ugu {

MeshPtr PoissonMeshMerging(const MeshPtr pinned,
                           const std::vector<int> pinned_boundary_vids,
                           const MeshPtr floating,
                           const std::vector<int> floating_boundary_vids);
bool PoissonMeshMerging(const std::vector<Eigen::Vector3f> pinned_verts,
                        const std::vector<Eigen::Vector3i> pinned_indices,
                        const std::vector<int> pinned_boundary_vids,
                        const std::vector<Eigen::Vector3f> floating_verts,
                        const std::vector<Eigen::Vector3i> floating_indices,
                        const std::vector<int> floating_boundary_vids,
                        std::vector<Eigen::Vector3f>& merged_verts,
                        std::vector<Eigen::Vector3i>& merged_indices);

}  // namespace ugu
