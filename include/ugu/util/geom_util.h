/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#pragma once

#include <numeric>
#include <optional>
#include <set>

#include "ugu/line.h"
#include "ugu/mesh.h"

namespace ugu {

bool MergeMeshes(const Mesh& src1, const Mesh& src2, Mesh* merged,
                 bool use_src1_material = false);
bool MergeMeshes(const std::vector<MeshPtr>& src_meshes, Mesh* merged);

std::tuple<std::vector<std::vector<std::pair<int, int>>>,
           std::vector<std::vector<int>>>
FindBoundaryLoops(const std::vector<Eigen::Vector3i>& indices, int32_t vnum);

std::tuple<std::vector<std::set<int32_t>>, std::set<int32_t>, std::set<int32_t>,
           std::vector<std::set<int32_t>>>
ClusterByConnectivity(const std::vector<Eigen::Vector3i>& indices,
                      int32_t vnum);

// make cube with 24 vertices
MeshPtr MakeCube(const Eigen::Vector3f& length, const Eigen::Matrix3f& R,
                 const Eigen::Vector3f& t);
MeshPtr MakeCube(const Eigen::Vector3f& length);
MeshPtr MakeCube(float length, const Eigen::Matrix3f& R,
                 const Eigen::Vector3f& t);
MeshPtr MakeCube(float length);

MeshPtr MakeUvSphere(int n_stacks = 10, int n_slices = 10);
MeshPtr MakeUvSphere(const Eigen::Vector3f& center, float r, int n_stacks = 10,
                     int n_slices = 10);

MeshPtr MakePlane(float length,
                  const Eigen::Matrix3f& R = Eigen::Matrix3f::Identity(),
                  const Eigen::Vector3f& t = Eigen::Vector3f::Zero());

void SetRandomUniformVertexColor(MeshPtr mesh, int seed = 0);
void SetRandomVertexColor(MeshPtr mesh, int seed = 0);

int32_t CutByPlane(MeshPtr mesh, const Planef& plane, bool fill_plane = true);

// Success: Vector3(t, u, v), Fail: std::nullopt
std::optional<Eigen::Vector3f> Intersect(const Eigen::Vector3f& origin,
                                         const Eigen::Vector3f& ray,
                                         const Eigen::Vector3f& v0,
                                         const Eigen::Vector3f& v1,
                                         const Eigen::Vector3f& v2,
                                         const float kEpsilon = 1e-6f);

std::optional<Eigen::Vector3d> Intersect(const Eigen::Vector3d& origin,
                                         const Eigen::Vector3d& ray,
                                         const Eigen::Vector3d& v0,
                                         const Eigen::Vector3d& v1,
                                         const Eigen::Vector3d& v2,
                                         const double kEpsilon = 1e-20);

// u, v and t are defined as follows:
// origin + ray * t ==  (1-u-v)*v0 + u*v1+v*v2
struct IntersectResult {
  float t = -1.f;
  float u = -1.f;
  float v = -1.f;
  uint32_t fid = uint32_t(~0);
};

std::vector<IntersectResult> Intersect(
    const Eigen::Vector3f& origin, const Eigen::Vector3f& ray,
    const std::vector<Eigen::Vector3f>& vertices,
    const std::vector<Eigen::Vector3i>& faces, int num_threads = 1,
    const float kEpsilon = 1e-10f);

// https://github.com/isl-org/Open3D/blob/ed30e3b61fbe031e106fa64030bec3f698b316b4/cpp/open3d/geometry/Geometry3D.cpp#L41
template <typename T>
T ComputeMinBound(const std::vector<T>& points) {
  if (points.empty()) {
    return T::Zero();
  }
  return std::accumulate(
      points.begin(), points.end(), points[0],
      [](const T& a, const T& b) { return a.array().min(b.array()).matrix(); });
}

template <typename T>
T ComputeMaxBound(const std::vector<T>& points) {
  if (points.empty()) {
    return T::Zero();
  }
  return std::accumulate(
      points.begin(), points.end(), points[0],
      [](const T& a, const T& b) { return a.array().max(b.array()).matrix(); });
}

}  // namespace ugu
