/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/mesh.h"

namespace ugu {

bool MergeMeshes(const Mesh& src1, const Mesh& src2, Mesh* merged,
                 bool use_src1_material = false);
bool MergeMeshes(const std::vector<std::shared_ptr<Mesh>>& src_meshes,
                 Mesh* merged);

std::tuple<std::vector<std::vector<std::pair<int, int>>>,
           std::vector<std::vector<int>>>
FindBoundaryLoops(const Mesh& mesh);

// make cube with 24 vertices
std::shared_ptr<Mesh> MakeCube(const Eigen::Vector3f& length,
                               const Eigen::Matrix3f& R,
                               const Eigen::Vector3f& t);
std::shared_ptr<Mesh> MakeCube(const Eigen::Vector3f& length);
std::shared_ptr<Mesh> MakeCube(float length, const Eigen::Matrix3f& R,
                               const Eigen::Vector3f& t);
std::shared_ptr<Mesh> MakeCube(float length);

void SetRandomVertexColor(std::shared_ptr<Mesh> mesh, int seed = 0);

}  // namespace ugu
