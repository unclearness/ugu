/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#pragma once

#include <numeric>
#include <optional>
#include <set>

#include "ugu/face_adjacency.h"
#include "ugu/line.h"
#include "ugu/mesh.h"
#include "ugu/plane.h"

namespace ugu {

bool MergeMeshes(const Mesh& src1, const Mesh& src2, Mesh* merged,
                 bool use_src1_material = false,
                 bool merge_same_name_material = true);
bool MergeMeshes(const std::vector<MeshPtr>& src_meshes, Mesh* merged,
                 bool overwrite_material = false,
                 bool merge_same_name_material = true);

std::tuple<std::vector<std::vector<std::pair<int, int>>>,
           std::vector<std::vector<int>>>
FindBoundaryLoops(const std::vector<Eigen::Vector3i>& indices, int32_t vnum);

std::tuple<std::vector<std::set<int32_t>>, std::set<int32_t>, std::set<int32_t>,
           std::vector<std::set<int32_t>>>
ClusterByConnectivity(const std::vector<Eigen::Vector3i>& indices, int32_t vnum,
                      bool vertex_based_adjacency = false);

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

MeshPtr MakePlane(const Eigen::Vector2f& length,
                  const Eigen::Matrix3f& R = Eigen::Matrix3f::Identity(),
                  const Eigen::Vector3f& t = Eigen::Vector3f::Zero(),
                  bool flip_uv_vertical = false);
MeshPtr MakePlane(float length,
                  const Eigen::Matrix3f& R = Eigen::Matrix3f::Identity(),
                  const Eigen::Vector3f& t = Eigen::Vector3f::Zero(),
                  bool flip_uv_vertical = false);
MeshPtr MakeTexturedPlane(
    const ugu::Image3b& texture, float width_scale, float height_scale = -1.f,
    const Eigen::Matrix3f& R = Eigen::Matrix3f::Identity(),
    const Eigen::Vector3f& t = Eigen::Vector3f::Zero());
MeshPtr MakeTexturedPlane(
    const ugu::Image4b& texture, float width_scale, float height_scale = -1.f,
    const Eigen::Matrix3f& R = Eigen::Matrix3f::Identity(),
    const Eigen::Vector3f& t = Eigen::Vector3f::Zero());

MeshPtr MakeCircle(float r, uint32_t n_slices = 20);
MeshPtr MakeCone(float r, float height, uint32_t n_slices = 20);
MeshPtr MakeCylinder(float r, float height, uint32_t n_slices = 20);
MeshPtr MakeArrow(float cylibder_r, float cylinder_height, float cone_r,
                  float cone_height, uint32_t cylinder_slices = 20,
                  uint32_t cone_slices = 20,
                  const ObjMaterial& cylinder_mat = ObjMaterial(),
                  const ObjMaterial& cone_mat = ObjMaterial(),
                  bool use_same_mat_if_possible = true);
MeshPtr MakeOrigin(float size, uint32_t cylinder_slices = 20,
                   uint32_t cone_slices = 20);
MeshPtr MakeTrajectoryGeom(const std::vector<Eigen::Affine3f>& c2w_list,
                           float size, uint32_t cylinder_slices = 20,
                           uint32_t cone_slices = 20);
MeshPtr MakeTrajectoryGeom(const std::vector<Eigen::Affine3d>& c2w_list,
                           float size, uint32_t cylinder_slices = 20,
                           uint32_t cone_slices = 20);

MeshPtr MakeFrustum(float top_w, float top_h, float bottom_w, float bottom_h,
                    float height, const ObjMaterial& top_mat = ObjMaterial(),
                    const ObjMaterial& bottom_mat = ObjMaterial(),
                    const ObjMaterial& side_mat = ObjMaterial(),
                    bool flip_plane_uv = false);

MeshPtr MakeViewFrustum(float fovy_rad, const Eigen::Affine3f& c2w, float z_max,
                        const Image3b& view_image,
                        CoordinateType coord = CoordinateType::OpenCV,
                        float z_min = 0.f, bool attach_view_image_bottom = true,
                        bool attach_view_image_top = true,
                        float aspect = 1.34f);

void SetRandomUniformVertexColor(MeshPtr mesh, int seed = 0);
void SetRandomVertexColor(MeshPtr mesh, int seed = 0);

int32_t CutByPlane(MeshPtr mesh, const Planef& plane, bool fill_plane = true);

// Success: Vector3(t, u, v), Fail: std::nullopt
std::optional<Eigen::Vector3f> Intersect(const Eigen::Vector3f& origin,
                                         const Eigen::Vector3f& ray,
                                         const Eigen::Vector3f& v0,
                                         const Eigen::Vector3f& v1,
                                         const Eigen::Vector3f& v2,
                                         const float kEpsilon = 1e-6f,
                                         bool standard_barycentric = true);

std::optional<Eigen::Vector3d> Intersect(const Eigen::Vector3d& origin,
                                         const Eigen::Vector3d& ray,
                                         const Eigen::Vector3d& v0,
                                         const Eigen::Vector3d& v1,
                                         const Eigen::Vector3d& v2,
                                         const double kEpsilon = 1e-20,
                                         bool standard_barycentric = true);

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
    const float kEpsilon = 1e-10f, bool standard_barycentric = true);

bool UpdateVertexAttrOneRingMost(
    uint32_t num_vertices, const std::vector<Eigen::Vector3i>& indices,
    std::vector<Eigen::Vector3f>& attrs,
    const Adjacency& vert_adjacency = Adjacency(),
    const FaceAdjacency& face_adjacency = FaceAdjacency());

std::tuple<std::vector<Eigen::Vector3f>, std::vector<Eigen::Vector3i>>
ExtractSubGeom(const std::vector<Eigen::Vector3f>& vertices,
               const std::vector<Eigen::Vector3i>& faces,
               const std::vector<uint32_t>& sub_face_ids);

bool CleanGeom(const std::vector<Eigen::Vector3f>& vertices,
               const std::vector<Eigen::Vector3i>& faces,
               std::vector<Eigen::Vector3f>& clean_vertices,
               std::vector<Eigen::Vector3i>& clean_faces);

bool CleanGeom(Mesh& mesh);

bool RemoveSmallConnectedComponents(Mesh& mesh, size_t small_th,
                                    bool pre_clean);

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

bool EstimateNormalsFromPoints(const std::vector<Eigen::Vector3f>& points,
                               std::vector<Eigen::Vector3f>& normals,
                               uint32_t nn_num = 10);

bool EstimateNormalsFromPoints(Mesh* mesh, uint32_t nn_num = 10);

}  // namespace ugu
