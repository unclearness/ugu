/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/line.h"
#include "ugu/mesh.h"

namespace ugu {

bool MergeMeshes(const Mesh& src1, const Mesh& src2, Mesh* merged,
                 bool use_src1_material = false);
bool MergeMeshes(const std::vector<MeshPtr>& src_meshes, Mesh* merged);

std::tuple<std::vector<std::vector<std::pair<int, int>>>,
           std::vector<std::vector<int>>>
FindBoundaryLoops(const std::vector<Eigen::Vector3i>& indices, int32_t vnum);

// make cube with 24 vertices
MeshPtr MakeCube(const Eigen::Vector3f& length, const Eigen::Matrix3f& R,
                 const Eigen::Vector3f& t);
MeshPtr MakeCube(const Eigen::Vector3f& length);
MeshPtr MakeCube(float length, const Eigen::Matrix3f& R,
                 const Eigen::Vector3f& t);
MeshPtr MakeCube(float length);

MeshPtr MakePlane(float length,
                  const Eigen::Matrix3f& R = Eigen::Matrix3f::Identity(),
                  const Eigen::Vector3f& t = Eigen::Vector3f::Zero());

void SetRandomVertexColor(MeshPtr mesh, int seed = 0);

int32_t CutByPlane(MeshPtr mesh, const Planef& plane, bool fill_plane = true);

// std::vector<std::vector<int32_t>> ClusterByConnectivity(const
// std::vector<Eigen::Vector3i>& indices, );

}  // namespace ugu
