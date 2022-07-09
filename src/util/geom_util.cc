/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "ugu/util/geom_util.h"

#include <deque>
#include <random>

#include "ugu/face_adjacency.h"
#include "ugu/util/math_util.h"
#include "ugu/util/thread_util.h"

namespace {

using namespace ugu;

template <typename T>
void CopyVec(const std::vector<T>& src, std::vector<T>* dst,
             bool clear_dst = true) {
  if (clear_dst) {
    dst->clear();
    dst->reserve(src.size());
  }
  std::copy(src.begin(), src.end(), std::back_inserter(*dst));
}

void CalcIntersectionAndRatio(const Eigen::Vector3f& p_st,
                              const Eigen::Vector3f& p_ed,
                              const ugu::Planef& plane,
                              Eigen::Vector3f& interp_p, float& interp_r) {
  // Edges as 3D line equations
  ugu::Line3f l1;
  l1.Set(p_st, p_ed);

  // Get intersection points of edges and plane
  float t1 = -1.f;
  plane.CalcIntersctionPoint(l1, t1, interp_p);

  // Interpolation ratio for uv and vertex color
  interp_r = (p_st - interp_p).norm() / (p_st - p_ed).norm();
}

Eigen::Vector3i GenerateFace(int vid1, int vid2, int vid3, bool flipped) {
  Eigen::Vector3i f(vid1, vid2, vid3);
  if (flipped) {
    f = Eigen::Vector3i(vid2, vid1, vid3);
  }

  return f;
}

// An implementation of Möller-Trumbore algorithm
// https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
// https://pheema.hatenablog.jp/entry/ray-triangle-intersection#%E4%B8%89%E8%A7%92%E5%BD%A2%E3%81%AE%E5%86%85%E9%83%A8%E3%81%AB%E5%AD%98%E5%9C%A8%E3%81%99%E3%82%8B%E7%82%B9%E3%81%AE%E8%A1%A8%E7%8F%BE
template <typename T>
std::optional<T> IntersectImpl(const T& origin, const T& ray, const T& v0,
                               const T& v1, const T& v2,
                               const typename T::Scalar kEpsilon,
                               bool standard_barycentric) {
  using scalar = typename T::Scalar;

  T e1 = v1 - v0;
  T e2 = v2 - v0;

  T alpha = ray.cross(e2);
  scalar det = e1.dot(alpha);

  // if det == 0, ray is parallel to triangle
  // Let's ignore this numerically unstable case
  if (-kEpsilon < det && det < kEpsilon) {
    return std::nullopt;
  }

  scalar invDet = scalar(1.0) / det;
  T r = origin - v0;

  // u, v and t are defined as follows:
  // origin + ray * t == (1−u−v)*v0 + u*v1 +v*v2

  // Must be 0 <= u <= 1
  scalar u = alpha.dot(r) * invDet;
  if (u < 0.0f || u > 1.0f) {
    return std::nullopt;
  }

  T beta = r.cross(e1);

  // Must be 0 <= v <= 1 and u + v <= 1
  // Thus, 0 <= v <= 1 - u
  scalar v = ray.dot(beta) * invDet;
  if (v < 0.0f || u + v > 1.0f) {
    return std::nullopt;
  }

  // The triangle is in the front of ray
  // So, 0 <= t
  scalar t = e2.dot(beta) * invDet;
  if (t < 0.0f) {
    return std::nullopt;
  }

  if (standard_barycentric) {
    // Convert from edge basis vectors to triangle area ratios
    // Original : u*(v1 - v0) + v*(v2 - v0) + v0
    // Converted : u* v0 + v* v1 + (1 - u - v) * v2
    scalar w = 1 - u - v;
    v = u;
    u = w;
  }

  return T(t, u, v);
}

MeshPtr MakeTexturedPlaneImpl(const Image3b& diffuse, const Image4b& with_alpha,
                              float width_scale, float height_scale,
                              const Eigen::Matrix3f& R,
                              const Eigen::Vector3f& t) {
  ObjMaterial mat;
  int w = -1;
  int h = -1;
  if (!diffuse.empty()) {
    mat.diffuse_tex = diffuse;
    mat.diffuse_texname = "diffuse_tex.png";
    w = diffuse.cols;
    h = diffuse.rows;
  }

  if (!with_alpha.empty()) {
    mat.with_alpha_tex = with_alpha;
    mat.with_alpha_texname = "with_alpha_tex.png";
    w = with_alpha.cols;
    h = with_alpha.rows;
  }

  float x_length = width_scale;
  float y_length = height_scale;
  float r = static_cast<float>(h) / static_cast<float>(w);
  if (x_length <= 0.f && y_length <= 0.f) {
    x_length = 1.f;
    y_length = x_length * r;
  } else if (0.f < x_length && y_length <= 0.f) {
    y_length = x_length * r;
  } else if (x_length <= 0.f && 0.f < y_length) {
    x_length = y_length / r;
  } else {
    // do nothing
  }

  auto mesh = MakePlane({x_length, y_length}, R, t);
  mesh->set_single_material(mat);

  return mesh;
}

void MergeMaterialsSolvingNameConflict(
    const std::vector<ugu::ObjMaterial>& src1_materials,
    const std::vector<int>& src1_material_ids,
    std::vector<ugu::ObjMaterial>& src2_materials,
    std::vector<int>& src2_material_ids,
    std::vector<ugu::ObjMaterial>& materials, std::vector<int>& material_ids,
    bool merge_same_name) {
  materials.clear();
  material_ids.clear();
  CopyVec(src1_materials, &materials);

  std::map<int, int> src2_src1_conflict;

  // Check if src2 has the same material name to src1
  for (size_t i = 0; i < src2_materials.size(); i++) {
    auto& mat2 = src2_materials[i];
    bool has_same_name = false;
    for (size_t ii = 0; ii < materials.size(); ii++) {
      const auto& mat = materials[ii];
      if (mat2.name == mat.name) {
        has_same_name = true;
        src2_src1_conflict.insert({static_cast<int>(i), static_cast<int>(ii)});
      }
    }
    // If the same material name was found, update to resolve name confilict
    if (has_same_name) {
      // TODO: rule for modified material name
      int new_name_postfix = 0;
      std::string new_name = mat2.name + "_0";
      while (true) {
        bool is_conflict = false;
        for (size_t j = 0; j < src2_materials.size(); j++) {
          if (new_name == src2_materials[j].name) {
            is_conflict = true;
            break;
          }
        }
        for (size_t j = 0; j < materials.size(); j++) {
          if (new_name == materials[j].name) {
            is_conflict = true;
            break;
          }
        }

        if (!is_conflict) {
          mat2.name = new_name;
          break;
        }

        new_name_postfix++;
        new_name = mat2.name + "_" + std::to_string(new_name_postfix);
      }
    }
  }

  if (merge_same_name && !src2_src1_conflict.empty()) {
    // not merge materials

    // Update src2_material_ids
    for (size_t j = 0; j < src2_material_ids.size(); j++) {
      for (auto& kv : src2_src1_conflict) {
        if (kv.first == src2_material_ids[j]) {
          src2_material_ids[j] = kv.second;
          break;
        }
      }
    }

    CopyVec(src1_material_ids, &material_ids);
    CopyVec(src2_material_ids, &material_ids, false);

  } else {
    CopyVec(src2_materials, &materials, false);

    std::vector<int> offset_material_ids2;
    std::vector<ObjMaterial> src2_materials_ = src2_materials;

    CopyVec(src1_material_ids, &material_ids);
    CopyVec(src2_material_ids, &offset_material_ids2);
    int offset_mi = static_cast<int>(src1_materials.size());
    std::for_each(offset_material_ids2.begin(), offset_material_ids2.end(),
                  [offset_mi](int& i) { i += offset_mi; });
    CopyVec(offset_material_ids2, &material_ids, false);
  }
}

void MergeMaterialsAndIds(const std::vector<ugu::ObjMaterial>& src1_materials,
                          const std::vector<int>& src1_material_ids,
                          const std::vector<ugu::ObjMaterial>& src2_materials,
                          const std::vector<int>& src2_material_ids,
                          std::vector<ugu::ObjMaterial>& materials,
                          std::vector<int>& material_ids,
                          bool use_src1_material, bool merge_same_name) {
  if (use_src1_material) {
    CopyVec(src1_materials, &materials);

    CopyVec(src1_material_ids, &material_ids);

    // Is using original material_ids for src2 right?
    CopyVec(src2_material_ids, &material_ids, false);
  } else {
    std::vector<int> offset_material_ids2;
    std::vector<ObjMaterial> src2_materials_ = src2_materials;
    std::vector<int> src2_material_ids_ = src2_material_ids;
    MergeMaterialsSolvingNameConflict(src1_materials, src1_material_ids,
                                      src2_materials_, src2_material_ids_,
                                      materials, material_ids, merge_same_name);
  }
}

}  // namespace

namespace ugu {

bool MergeMeshes(const Mesh& src1, const Mesh& src2, Mesh* merged,
                 bool use_src1_material, bool merge_same_name_material) {
  std::vector<Eigen::Vector3f> vertices, vertex_colors, vertex_normals;
  std::vector<Eigen::Vector2f> uv;
  std::vector<int> material_ids;
  std::vector<ugu::ObjMaterial> materials;

  std::vector<Eigen::Vector3i> vertex_indices, offset_vertex_indices2;
  std::vector<Eigen::Vector3i> uv_indices, offset_uv_indices2;

  merged->Clear();

  CopyVec(src1.vertices(), &vertices);
  CopyVec(src2.vertices(), &vertices, false);

  CopyVec(src1.vertex_colors(), &vertex_colors);
  CopyVec(src2.vertex_colors(), &vertex_colors, false);

  CopyVec(src1.normals(), &vertex_normals);
  CopyVec(src2.normals(), &vertex_normals, false);

  CopyVec(src1.uv(), &uv);
  CopyVec(src2.uv(), &uv, false);

  CopyVec(src1.vertex_indices(), &vertex_indices);
  CopyVec(src2.vertex_indices(), &offset_vertex_indices2);
  int offset_vi = static_cast<int>(src1.vertices().size());
  std::for_each(offset_vertex_indices2.begin(), offset_vertex_indices2.end(),
                [offset_vi](Eigen::Vector3i& i) {
                  i[0] += offset_vi;
                  i[1] += offset_vi;
                  i[2] += offset_vi;
                });
  CopyVec(offset_vertex_indices2, &vertex_indices, false);

  CopyVec(src1.uv_indices(), &uv_indices);
  CopyVec(src2.uv_indices(), &offset_uv_indices2);
  int offset_uvi = static_cast<int>(src1.uv().size());
  std::for_each(offset_uv_indices2.begin(), offset_uv_indices2.end(),
                [offset_uvi](Eigen::Vector3i& i) {
                  i[0] += offset_uvi;
                  i[1] += offset_uvi;
                  i[2] += offset_uvi;
                });
  CopyVec(offset_uv_indices2, &uv_indices, false);

  MergeMaterialsAndIds(src1.materials(), src1.material_ids(), src2.materials(),
                       src2.material_ids(), materials, material_ids,
                       use_src1_material, merge_same_name_material);

  merged->set_vertices(vertices);
  merged->set_vertex_colors(vertex_colors);
  merged->set_normals(vertex_normals);
  merged->set_uv(uv);
  merged->set_material_ids(material_ids);
  merged->set_materials(materials);
  merged->set_vertex_indices(vertex_indices);
  merged->set_uv_indices(uv_indices);

  merged->CalcNormal();
  merged->CalcStats();

  return true;
}

bool MergeMeshes(const std::vector<MeshPtr>& src_meshes, Mesh* merged,
                 bool overwrite_material, bool merge_same_name_material) {
  if (src_meshes.empty()) {
    return false;
  }

  if (src_meshes.size() == 1) {
    *merged = *src_meshes[0];
    return true;
  }

  Mesh tmp0, tmp2;
  tmp0 = *src_meshes[0];
  for (size_t i = 1; i < src_meshes.size(); i++) {
    const auto& src = src_meshes[i];
    ugu::MergeMeshes(tmp0, *src, &tmp2, overwrite_material,
                     merge_same_name_material);
    tmp0 = Mesh(tmp2);
  }
  *merged = tmp0;

  return true;
}

std::tuple<std::vector<std::vector<std::pair<int, int>>>,
           std::vector<std::vector<int>>>
FindBoundaryLoops(const std::vector<Eigen::Vector3i>& indices, int32_t vnum) {
  std::vector<std::vector<std::pair<int, int>>> boundary_edges_list;
  std::vector<std::vector<int>> boundary_vertex_ids_list;

  ugu::FaceAdjacency face_adjacency;
  face_adjacency.Init(vnum, indices);

  auto [boundary_edges, boundary_vertex_ids] =
      face_adjacency.GetBoundaryEdges();

  if (boundary_edges.empty()) {
    return {boundary_edges_list, boundary_vertex_ids_list};
  }

  auto cur_edge = boundary_edges[0];
  boundary_edges.erase(boundary_edges.begin());

  std::vector<std::pair<int, int>> cur_edges;
  cur_edges.push_back(cur_edge);

  while (true) {
    // The same loop
    // Find connecting vertex
    bool found_connected = false;
    int connected_index = -1;
    for (size_t i = 0; i < boundary_edges.size(); i++) {
      const auto& e = boundary_edges[i];
      if (cur_edge.second == e.first) {
        found_connected = true;
        connected_index = static_cast<int>(i);
        cur_edge = e;
        cur_edges.push_back(cur_edge);
        break;
      }
    }

    if (found_connected) {
      boundary_edges.erase(boundary_edges.begin() + connected_index);
    } else {
      // May be the end of loop
#if 0
      bool loop_closed = false;
      for (size_t i = 0; i < cur_edges.size(); i++) {
        const auto& e = cur_edges[i];
        if (cur_edge.second == e.first) {
          loop_closed = true;
          break;
        }
      }
#endif  // 0
      bool loop_closed = (cur_edge.second == cur_edges[0].first);

      if (!loop_closed) {
        ugu::LOGE("FindBoundaryLoops failed. Maybe non-manifold mesh?");
        boundary_edges_list.clear();
        boundary_vertex_ids_list.clear();
        return {boundary_edges_list, boundary_vertex_ids_list};
      }

      boundary_edges_list.push_back(cur_edges);

      std::vector<int> cur_boundary;
      for (const auto& e : cur_edges) {
        cur_boundary.push_back(e.first);
      }
      assert(3 <= cur_edges.size());
      assert(3 <= cur_boundary.size());
      boundary_vertex_ids_list.push_back(cur_boundary);

      cur_edges.clear();

      // Go to another loop
      if (boundary_edges.empty()) {
        break;
      }

      cur_edge = boundary_edges[0];
      boundary_edges.erase(boundary_edges.begin());
      cur_edges.push_back(cur_edge);
    }
  }

  return {boundary_edges_list, boundary_vertex_ids_list};
}

std::tuple<std::vector<std::set<int32_t>>, std::set<int32_t>, std::set<int32_t>,
           std::vector<std::set<int32_t>>>
ClusterByConnectivity(const std::vector<Eigen::Vector3i>& indices,
                      int32_t vnum) {
  std::vector<std::set<int32_t>> clusters;
  std::vector<std::set<int32_t>> clusters_f;

  std::set<int32_t> non_orphans;

  ugu::FaceAdjacency fa;
  fa.Init(vnum, indices);

  std::unordered_set<int32_t> to_process;
  // std::unordered_set<int32_t> processed;
  for (size_t i = 0; i < indices.size(); i++) {
    to_process.insert(static_cast<int32_t>(i));
  }

  std::deque<int32_t> q;

  while (!to_process.empty()) {
    q.push_back(*to_process.begin());
    std::set<int32_t> cluster_f;
    cluster_f.insert(q.back());
    to_process.erase(q.back());

    while (!q.empty()) {
      int fid = q.front();
      q.pop_front();
      std::vector<int> adjacent_face_ids;

      fa.GetAdjacentFaces(fid, &adjacent_face_ids);
      // fa.RemoveFace(fid);

      for (const auto& afid : adjacent_face_ids) {
        if (to_process.find(afid) != to_process.end()) {
          // processed.insert(afid);
          to_process.erase(afid);
          q.push_back(afid);
          cluster_f.insert(afid);
        }
      }
    }

    std::set<int32_t> cluster;
    for (const auto& fid : cluster_f) {
      cluster.insert(indices[fid][0]);
      cluster.insert(indices[fid][1]);
      cluster.insert(indices[fid][2]);
    }

    non_orphans.insert(cluster.begin(), cluster.end());

    clusters.push_back(cluster);
    clusters_f.push_back(cluster_f);
  }

  std::set<int32_t> orphans, all_vids;
  for (int32_t i = 0; i < vnum; i++) {
    all_vids.insert(i);
  }

  std::set_difference(all_vids.begin(), all_vids.end(), non_orphans.begin(),
                      non_orphans.end(), std::inserter(orphans, orphans.end()));

  return {clusters, non_orphans, orphans, clusters_f};
}

MeshPtr MakeCube(const Eigen::Vector3f& length, const Eigen::Matrix3f& R,
                 const Eigen::Vector3f& t) {
  MeshPtr cube(new Mesh);
  std::vector<Eigen::Vector3f> vertices(24);
  std::vector<Eigen::Vector3i> vertex_indices(12);
  std::vector<Eigen::Vector3f> vertex_colors(24);

  const float h_x = length.x() / 2;
  const float h_y = length.y() / 2;
  const float h_z = length.z() / 2;

  vertices[0] = Eigen::Vector3f(-h_x, h_y, -h_z);
  vertices[1] = Eigen::Vector3f(h_x, h_y, -h_z);
  vertices[2] = Eigen::Vector3f(h_x, h_y, h_z);
  vertices[3] = Eigen::Vector3f(-h_x, h_y, h_z);
  vertex_indices[0] = Eigen::Vector3i(0, 2, 1);
  vertex_indices[1] = Eigen::Vector3i(0, 3, 2);

  vertices[4] = Eigen::Vector3f(-h_x, -h_y, -h_z);
  vertices[5] = Eigen::Vector3f(h_x, -h_y, -h_z);
  vertices[6] = Eigen::Vector3f(h_x, -h_y, h_z);
  vertices[7] = Eigen::Vector3f(-h_x, -h_y, h_z);
  vertex_indices[2] = Eigen::Vector3i(4, 5, 6);
  vertex_indices[3] = Eigen::Vector3i(4, 6, 7);

  vertices[8] = vertices[1];
  vertices[9] = vertices[2];
  vertices[10] = vertices[6];
  vertices[11] = vertices[5];
  vertex_indices[4] = Eigen::Vector3i(8, 9, 10);
  vertex_indices[5] = Eigen::Vector3i(8, 10, 11);

  vertices[12] = vertices[0];
  vertices[13] = vertices[3];
  vertices[14] = vertices[7];
  vertices[15] = vertices[4];
  vertex_indices[6] = Eigen::Vector3i(12, 14, 13);
  vertex_indices[7] = Eigen::Vector3i(12, 15, 14);

  vertices[16] = vertices[0];
  vertices[17] = vertices[1];
  vertices[18] = vertices[5];
  vertices[19] = vertices[4];
  vertex_indices[8] = Eigen::Vector3i(16, 17, 18);
  vertex_indices[9] = Eigen::Vector3i(16, 18, 19);

  vertices[20] = vertices[3];
  vertices[21] = vertices[2];
  vertices[22] = vertices[6];
  vertices[23] = vertices[7];
  vertex_indices[10] = Eigen::Vector3i(20, 22, 21);
  vertex_indices[11] = Eigen::Vector3i(20, 23, 22);

  // set default color
  for (int i = 0; i < 24; i++) {
#ifdef UGU_USE_OPENCV
    // BGR
    vertex_colors[i][2] = (-vertices[i][0] + h_x) / length.x() * 255;
    vertex_colors[i][1] = (-vertices[i][1] + h_y) / length.y() * 255;
    vertex_colors[i][0] = (-vertices[i][2] + h_z) / length.z() * 255;
#else
    // RGB
    vertex_colors[i][0] = (-vertices[i][0] + h_x) / length.x() * 255;
    vertex_colors[i][1] = (-vertices[i][1] + h_y) / length.y() * 255;
    vertex_colors[i][2] = (-vertices[i][2] + h_z) / length.z() * 255;
#endif
  }

  cube->set_vertices(vertices);
  cube->set_vertex_indices(vertex_indices);
  cube->set_vertex_colors(vertex_colors);

  std::vector<ugu::ObjMaterial> materials(1);
  cube->set_materials(materials);
  std::vector<int> material_ids(vertex_indices.size(), 0);
  cube->set_material_ids(material_ids);

  cube->Transform(R, t);

  cube->CalcNormal();

  return cube;
}

MeshPtr MakeCube(const Eigen::Vector3f& length) {
  const Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
  const Eigen::Vector3f t(0.0f, 0.0f, 0.0f);
  return MakeCube(length, R, t);
}

MeshPtr MakeCube(float length, const Eigen::Matrix3f& R,
                 const Eigen::Vector3f& t) {
  Eigen::Vector3f length_xyz{length, length, length};
  return MakeCube(length_xyz, R, t);
}

MeshPtr MakeCube(float length) {
  const Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
  const Eigen::Vector3f t(0.0f, 0.0f, 0.0f);
  return MakeCube(length, R, t);
}

MeshPtr MakeUvSphere(int n_stacks, int n_slices) {
  // https://www.danielsieger.com/blog/2021/03/27/generating-spheres.html

  std::vector<Eigen::Vector3f> vertices;
  std::vector<Eigen::Vector3i> indices;
  std::vector<Eigen::Vector2f> uvs;

  // add top vertex
  int v0 = 0;
  auto v0_ = Eigen::Vector3f(0.f, 1.f, 0.f);
  vertices.push_back(v0_);

  // generate vertices per stack / slice
  for (int i = 0; i < n_stacks - 1; i++) {
    auto phi = ugu::pi * double(int64_t(i) + 1) / double(n_stacks);
    for (int j = 0; j < n_slices; j++) {
      double theta = 2.0 * ugu::pi * double(j) / double(n_slices);
      double x = std::sin(phi) * std::cos(theta);
      double y = std::cos(phi);
      double z = std::sin(phi) * std::sin(theta);
      vertices.push_back(Eigen::Vector3f(
          static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)));
      uvs.push_back({x, y});
    }
  }

  // add bottom vertex
  int v1 = static_cast<int>(vertices.size());
  auto v1_ = Eigen::Vector3f(0.f, -1.f, 0.f);
  vertices.push_back(v1_);

  // add top / bottom triangles
  for (int i = 0; i < n_slices; ++i) {
    auto i0 = i + 1;
    auto i1 = (i + 1) % n_slices + 1;
    indices.push_back({v0, i1, i0});
    i0 = i + n_slices * (n_stacks - 2) + 1;
    i1 = (i + 1) % n_slices + n_slices * (n_stacks - 2) + 1;
    indices.push_back({v1, i0, i1});
  }

  // add triangles per stack / slice
  for (int j = 0; j < n_stacks - 2; j++) {
    auto j0 = j * n_slices + 1;
    auto j1 = (j + 1) * n_slices + 1;
    for (int i = 0; i < n_slices; i++) {
      auto i0 = j0 + i;
      auto i1 = j0 + (i + 1) % n_slices;
      auto i2 = j1 + (i + 1) % n_slices;
      auto i3 = j1 + i;
      indices.push_back({i0, i1, i2});
      indices.push_back({i0, i2, i3});
    }
  }

  auto mesh = Mesh::Create();

  mesh->set_vertices(vertices);
  mesh->set_uv(uvs);
  mesh->set_vertex_indices(indices);
  mesh->set_uv_indices(indices);

  return mesh;
}

MeshPtr MakeUvSphere(const Eigen::Vector3f& center, float r, int n_stacks,
                     int n_slices) {
  auto mesh = MakeUvSphere(n_stacks, n_slices);
  mesh->Scale(r);
  mesh->Translate(center);
  return mesh;
}

MeshPtr MakeTexturedPlane(const Image3b& texture, float width_scale,
                          float height_scale, const Eigen::Matrix3f& R,
                          const Eigen::Vector3f& t) {
  return MakeTexturedPlaneImpl(texture, Image4b(), width_scale, height_scale, R,
                               t);
}

MeshPtr MakeTexturedPlane(const Image4b& texture, float width_scale,
                          float height_scale, const Eigen::Matrix3f& R,
                          const Eigen::Vector3f& t) {
  return MakeTexturedPlaneImpl(Image3b(), texture, width_scale, height_scale, R,
                               t);
}

MeshPtr MakePlane(const Eigen::Vector2f& length, const Eigen::Matrix3f& R,
                  const Eigen::Vector3f& t, bool flip_uv_vertical) {
  MeshPtr plane = std::make_shared<Mesh>();
  std::vector<Eigen::Vector3f> vertices(4);
  std::vector<Eigen::Vector3i> vertex_indices(2);
  std::vector<Eigen::Vector2f> uvs{
      {1.f, 1.f}, {0.f, 1.f}, {1.f, 0.f}, {0.f, 0.f}};

  if (flip_uv_vertical) {
    for (auto& uv : uvs) {
      uv[1] = 1.f - uv[1];
    }
  }

  const float h_x = length[0] / 2;
  const float h_y = length[1] / 2;

  vertices[0] = Eigen::Vector3f(h_x, h_y, 0.f);
  vertices[1] = Eigen::Vector3f(-h_x, h_y, 0.f);
  vertices[2] = Eigen::Vector3f(h_x, -h_y, 0.f);
  vertices[3] = Eigen::Vector3f(-h_x, -h_y, 0.f);

  for (auto& v : vertices) {
    v = R * v + t;
  }

  vertex_indices[0] = Eigen::Vector3i(0, 1, 2);
  vertex_indices[1] = Eigen::Vector3i(2, 1, 3);

  plane->set_vertices(vertices);
  plane->set_vertex_indices(vertex_indices);
  plane->set_uv(uvs);
  plane->set_uv_indices(vertex_indices);

  plane->set_default_material();

  plane->CalcNormal();

  plane->CalcStats();

  return plane;
}

MeshPtr MakePlane(float length, const Eigen::Matrix3f& R,
                  const Eigen::Vector3f& t, bool flip_uv_vertical) {
  return MakePlane(Eigen::Vector2f(length, length), R, t, flip_uv_vertical);
}

MeshPtr MakeCircle(float r, uint32_t n_slices) {
  MeshPtr mesh = std::make_shared<Mesh>();
  std::vector<Eigen::Vector3f> vertices;
  std::vector<Eigen::Vector2f> uvs;
  std::vector<Eigen::Vector3i> indices;

  vertices.emplace_back(Eigen::Vector3f(0.f, 0.f, 0.f));
  uvs.emplace_back(Eigen::Vector2f(0.5f, 0.5f));

  // Generate vertices per stack / slice
  for (int j = 0; j < static_cast<int>(n_slices); j++) {
    double theta = 2.0 * ugu::pi * double(j) / double(n_slices);
    double x = std::cos(theta);
    double y = std::sin(theta);
    vertices.push_back(Eigen::Vector3f(static_cast<float>(x) * r,
                                       static_cast<float>(y) * r, 0.f));
    uvs.push_back({x + 1.f, y + 1.f});
  }

  for (uint32_t i = 0; i < n_slices - 1; i++) {
    int i0 = 0;
    int i1 = static_cast<int>(i + 1) + 0;
    int i2 = static_cast<int>(i + 1) + 1;
    indices.push_back({i0, i1, i2});
  }
  indices.push_back({0, static_cast<int>(n_slices), 1});

  mesh->set_vertices(vertices);
  mesh->set_vertex_indices(indices);
  mesh->set_uv(uvs);
  mesh->set_uv_indices(indices);

  mesh->set_default_material();

  mesh->CalcNormal();

  mesh->CalcStats();

  return mesh;
}

MeshPtr MakeCone(float r, float height, uint32_t n_slices) {
  MeshPtr cone = std::make_shared<Mesh>();
  MeshPtr circle = MakeCircle(r, n_slices);
  circle->FlipFaces();
  std::vector<Eigen::Vector3f> vertices = circle->vertices();
  std::vector<Eigen::Vector2f> uvs = circle->uv();
  std::vector<Eigen::Vector3i> indices = circle->vertex_indices();

  vertices.emplace_back(Eigen::Vector3f(0.f, 0.f, height));
  uvs.emplace_back(Eigen::Vector2f(0.5f, 0.5f));

  // Side
  for (uint32_t i = 0; i < n_slices - 1; i++) {
    int i0 = static_cast<int>(n_slices + 1);
    int i1 = static_cast<int>(i + 1) + 0;
    int i2 = static_cast<int>(i + 1) + 1;
    indices.push_back({i0, i1, i2});
  }
  indices.push_back(
      {1, static_cast<int>(n_slices + 1), static_cast<int>(n_slices)});

  cone->set_vertices(vertices);
  cone->set_vertex_indices(indices);
  cone->set_uv(uvs);
  cone->set_uv_indices(indices);

  cone->set_default_material();

  cone->CalcNormal();

  cone->CalcStats();

  return cone;
}

MeshPtr MakeCylinder(float r, float height, uint32_t n_slices) {
  auto top = MakeCircle(r, n_slices);
  auto bottom = MakeCircle(r, n_slices);
  top->FlipFaces();
  bottom->Translate({0.f, 0.f, height});

#if 1
  std::vector<Eigen::Vector3f> vertices = top->vertices();
  std::copy(bottom->vertices().begin(), bottom->vertices().end(),
            std::back_inserter(vertices));

  std::vector<Eigen::Vector3i> indices = top->vertex_indices();
  size_t offset = top->vertices().size();
  for (const auto& bi : bottom->vertex_indices()) {
    Eigen::Vector3i new_i = bi;
    new_i += Eigen::Vector3i::Constant(static_cast<int>(offset));
    indices.push_back(new_i);
  }
#endif  // 0

  // Side
  for (int i = 0; i < static_cast<int>(n_slices - 1); i++) {
    int i0 = i + 1;
    int i1 = i + 1 + 1;
    int i2 = static_cast<int>(offset) + i + 1;
    int i3 = static_cast<int>(offset) + i + 1 + 1;

    indices.push_back({i0, i1, i2});
    indices.push_back({i2, i1, i3});
  }
  indices.push_back({static_cast<int>(n_slices), static_cast<int>(1),
                     static_cast<int>(offset + 1)});
  indices.push_back({static_cast<int>(n_slices), static_cast<int>(offset + 1),
                     static_cast<int>(offset + n_slices)});

  MeshPtr mesh = Mesh::Create();
  mesh->set_vertices(vertices);
  mesh->set_vertex_indices(indices);

  mesh->set_default_material();

  mesh->CalcNormal();

  mesh->CalcStats();

  return mesh;
}

MeshPtr MakeArrow(float cylibder_r, float cylinder_height, float cone_r,
                  float cone_height, uint32_t cylinder_slices,
                  uint32_t cone_slices, const ObjMaterial& cylinder_mat,
                  const ObjMaterial& cone_mat, bool use_same_mat_if_possible) {
  auto cylinder = MakeCylinder(cylibder_r, cylinder_height, cylinder_slices);
  auto cone = MakeCone(cone_r, cone_height, cone_slices);
  cone->Translate({0.f, 0.f, cylinder_height});
  cone->set_uv({});
  cone->set_uv_indices({});

  cylinder->set_single_material(cylinder_mat);
  cone->set_single_material(cone_mat);

  auto mesh = Mesh::Create();

  MergeMeshes(*cylinder, *cone, mesh.get(), use_same_mat_if_possible);

  return mesh;
}

MeshPtr MakeOrigin(float size, uint32_t cylinder_slices, uint32_t cone_slices) {
  const float cylibder_r = size * 0.05f;
  const float cylinder_height = size * 0.8f;
  const float cone_r = size * 0.1f;
  const float cone_height = size * 0.2f;

  ObjMaterial x_mat, y_mat, z_mat;

  x_mat.diffuse = {1.f, 0.f, 0.f};
  x_mat.name = "x_axis";
  y_mat.diffuse = {0.f, 1.f, 0.f};
  y_mat.name = "y_axis";
  z_mat.diffuse = {0.f, 0.f, 1.f};
  z_mat.name = "z_axis";

  auto x = MakeArrow(cylibder_r, cylinder_height, cone_r, cone_height,
                     cylinder_slices, cone_slices, x_mat, x_mat);
  x->Rotate(Eigen::AngleAxisf(radians(90.f), Eigen::Vector3f(0.f, 1.f, 0.f))
                .matrix());

  auto y = MakeArrow(cylibder_r, cylinder_height, cone_r, cone_height,
                     cylinder_slices, cone_slices, y_mat, y_mat);
  y->Rotate(Eigen::AngleAxisf(radians(-90.f), Eigen::Vector3f(1.f, 0.f, 0.f))
                .matrix());

  auto z = MakeArrow(cylibder_r, cylinder_height, cone_r, cone_height,
                     cylinder_slices, cone_slices, z_mat, z_mat);

  auto mesh = Mesh::Create();

  MergeMeshes({x, y, z}, mesh.get());

  return mesh;
}

MeshPtr MakeTrajectoryGeom(const std::vector<Eigen::Affine3f>& c2w_list,
                           float size, uint32_t cylinder_slices,
                           uint32_t cone_slices) {
  auto trajectory = Mesh::Create();
  std::vector<MeshPtr> meshes;
  for (const auto& c2w : c2w_list) {
    auto cam = MakeOrigin(size, cylinder_slices, cone_slices);

    Eigen::Affine3f T = c2w;

    cam->Transform(T);

    meshes.push_back(cam);
  }

  MergeMeshes(meshes, trajectory.get(), true);

  return trajectory;
}

MeshPtr MakeTrajectoryGeom(const std::vector<Eigen::Affine3d>& c2w_list,
                           float size, uint32_t cylinder_slices,
                           uint32_t cone_slices) {
  std::vector<Eigen::Affine3f> c2w_list_f;
  for (const auto& c2w : c2w_list) {
    c2w_list_f.push_back(c2w.cast<float>());
  }
  return MakeTrajectoryGeom(c2w_list_f, size, cylinder_slices, cone_slices);
}

MeshPtr MakeFrustum(float top_w, float top_h, float bottom_w, float bottom_h,
                    float height, const ObjMaterial& top_mat,
                    const ObjMaterial& bottom_mat, const ObjMaterial& side_mat,
                    bool flip_plane_uv) {
  MeshPtr mesh = Mesh::Create();
  auto top = MakePlane({top_w, top_h}, Eigen::Matrix3f::Identity(),
                       Eigen::Vector3f::Zero(), flip_plane_uv);
  top->Translate({0.f, 0.f, height});
  top->set_single_material(top_mat);

  auto bottom = MakePlane({bottom_w, bottom_h}, Eigen::Matrix3f::Identity(),
                          Eigen::Vector3f::Zero(), flip_plane_uv);
  bottom->FlipFaces();
  bottom->set_single_material(bottom_mat);

  MergeMeshes({top, bottom}, mesh.get());

  auto vertex_indices = mesh->vertex_indices();

  vertex_indices.push_back({0, 4, 1});
  vertex_indices.push_back({1, 4, 5});

  vertex_indices.push_back({1, 5, 3});
  vertex_indices.push_back({3, 5, 7});

  vertex_indices.push_back({2, 3, 7});
  vertex_indices.push_back({2, 7, 6});

  vertex_indices.push_back({0, 2, 6});
  vertex_indices.push_back({4, 0, 6});

  auto uv = mesh->uv();
  uv.push_back({0.f, 0.f});    // 8
  uv.push_back({0.f, 0.25f});  // 9
  uv.push_back({0.f, 0.5f});   // 10
  uv.push_back({0.f, 1.f});    // 11

  uv.push_back({1.f, 0.f});    // 12
  uv.push_back({1.f, 0.25f});  // 13
  uv.push_back({1.f, 0.5f});   // 14
  uv.push_back({1.f, 1.f});    // 15

  auto uv_indices = mesh->uv_indices();
  uv_indices.push_back({8, 9, 12});
  uv_indices.push_back({9, 12, 13});

  uv_indices.push_back({9, 10, 13});
  uv_indices.push_back({10, 13, 14});

  uv_indices.push_back({10, 11, 14});
  uv_indices.push_back({11, 14, 15});

  uv_indices.push_back({11, 8, 15});
  uv_indices.push_back({8, 15, 12});

  std::vector<ObjMaterial> materials;
  std::vector<ObjMaterial> tmp_mat{side_mat};

  std::vector<int> material_ids;
  std::vector<int> tmp_material_ids(8, 0);

  MergeMaterialsAndIds(mesh->materials(), mesh->material_ids(), tmp_mat,
                       tmp_material_ids, materials, material_ids, false, true);

  mesh->set_vertex_indices(vertex_indices);

  mesh->set_uv(uv);
  mesh->set_uv_indices(uv_indices);

  mesh->set_materials(materials);
  mesh->set_material_ids(material_ids);

  mesh->CalcNormal();
  mesh->CalcStats();

  return mesh;
}

MeshPtr MakeViewFrustum(float fovy_rad, const Eigen::Affine3f& c2w, float z_max,
                        const Image3b& view_image, CoordinateType coord,
                        float z_min, bool attach_view_image_bottom,
                        bool attach_view_image_top, float aspect) {
  fovy_rad = std::clamp(fovy_rad, radians(1.f), radians(179.f));

  Eigen::Vector3f view_offset =
      c2w.rotation().col(2) * z_min;  // view direction

  bool flip_plane_uv = false;
  if (coord == CoordinateType::OpenCV) {
    flip_plane_uv = true;
  }

  float height = std::max(z_max - z_min, 1e-10f);

  float aspect_ = aspect;
  if (!view_image.empty()) {
    aspect_ = static_cast<float>(view_image.cols) /
              static_cast<float>(view_image.rows);
  }
  if (aspect_ <= 0.f) {
    aspect_ = 1.34f;  // default 4:3
  }

  float top_h = z_max / std::cos(fovy_rad * 0.5f);
  float top_w = top_h * aspect_;

  float bottom_h = z_min / std::cos(fovy_rad * 0.5f);
  float bottom_w = bottom_h * aspect_;

  ObjMaterial top_mat, bottom_mat, side_mat;
  if (attach_view_image_bottom) {
    bottom_mat.name = "bottom_mat";
    bottom_mat.diffuse_tex = view_image;
    bottom_mat.diffuse_texname = "view_image.jpg";
  }
  if (attach_view_image_top) {
    top_mat.name = "top_mat";
    top_mat.diffuse_tex = view_image;
    top_mat.diffuse_texname = "view_image.jpg";
  }

  side_mat.name = "side_mat";
  auto mesh = MakeFrustum(top_w, top_h, bottom_w, bottom_h, height, top_mat,
                          bottom_mat, side_mat, flip_plane_uv);

  mesh->Transform(c2w.rotation(), c2w.translation() + view_offset);

  return mesh;
}

void SetRandomVertexColor(MeshPtr mesh, int seed) {
  std::mt19937 mt(seed);
  std::uniform_int_distribution<int> random_color(0, 255);

  std::vector<Eigen::Vector3f> vertex_colors(mesh->vertices().size());
  for (auto& vc : vertex_colors) {
    vc[0] = static_cast<float>(random_color(mt));
    vc[1] = static_cast<float>(random_color(mt));
    vc[2] = static_cast<float>(random_color(mt));
  }

  mesh->set_vertex_colors(vertex_colors);
}

void SetRandomUniformVertexColor(MeshPtr mesh, int seed) {
  std::mt19937 mt(seed);
  std::uniform_int_distribution<int> random_color(0, 255);

  std::vector<Eigen::Vector3f> vertex_colors(
      mesh->vertices().size(), {static_cast<float>(random_color(mt)),
                                static_cast<float>(random_color(mt)),
                                static_cast<float>(random_color(mt))});

  mesh->set_vertex_colors(vertex_colors);
}

int32_t CutByPlane(MeshPtr mesh, const Planef& plane, bool fill_plane) {
  int32_t num_removed{0};

  mesh->CalcFaceNormal();

  std::vector<Eigen::Vector3f> valid_vertices, valid_vertex_colors;
  std::vector<Eigen::Vector2f> valid_uv;
  std::vector<Eigen::Vector3i> valid_indices, valid_uv_indices;
  bool with_uv = !mesh->uv().empty() && !mesh->uv_indices().empty();
  bool with_vertex_color = !mesh->vertex_colors().empty();

  std::vector<int> added_table(mesh->vertices().size(), -1);
  std::vector<int> added_uv_table(mesh->uv().size(), -1);

  int valid_face_count{0};
  std::vector<int> valid_face_table(mesh->vertex_indices().size(), -1);
  for (size_t i = 0; i < mesh->vertex_indices().size(); i++) {
    Eigen::Vector3i face, uv_face;
    std::vector<int32_t> valid_vid, invalid_vid;
    std::vector<int32_t> valid_uv_vid, invalid_uv_vid;
    for (int j = 0; j < 3; j++) {
      auto vid = mesh->vertex_indices()[i][j];
      int new_index = -1;
      if (plane.IsNormalSide(mesh->vertices()[vid])) {
        if (added_table[vid] < 0) {
          // Newly add
          valid_vertices.push_back(mesh->vertices()[vid]);
          added_table[vid] = static_cast<int>(valid_vertices.size() - 1);
          if (with_vertex_color) {
            valid_vertex_colors.push_back(mesh->vertex_colors()[vid]);
          }
        }
        new_index = added_table[vid];
      }

      if (new_index < 0) {
        invalid_vid.push_back(vid);
        if (with_uv) {
          auto uv_vid = mesh->uv_indices()[i][j];
          invalid_uv_vid.push_back(uv_vid);
        }
        continue;
      }
      face[j] = new_index;
      valid_vid.push_back(vid);
      if (with_uv) {
        auto uv_vid = mesh->uv_indices()[i][j];
        if (added_uv_table[uv_vid] < 0) {
          // Newly add
          valid_uv.push_back(mesh->uv()[uv_vid]);
          added_uv_table[uv_vid] = static_cast<int>(valid_uv.size() - 1);
        }
        valid_uv_vid.push_back(uv_vid);
        uv_face[j] = added_uv_table[uv_vid];
      }
    }
    if (valid_vid.size() == 0) {
      // All vertices are removed
      continue;
    } else if (valid_vid.size() == 1) {
      // Add two new vertices to make a face
      //     original triangle
      //      -----------
      //     |           /
      //     |          /   outside
      //     |         /
      //     |        /
      //     |-------/    plane
      //     | new  /
      //     |face /
      //     |    /     inside
      //     |   /
      //     |  /
      //     | /

      Eigen::Vector3f p1, p2;
      float interp1, interp2;
      CalcIntersectionAndRatio(mesh->vertices()[valid_vid[0]],
                               mesh->vertices()[invalid_vid[0]], plane, p1,
                               interp1);
      CalcIntersectionAndRatio(mesh->vertices()[valid_vid[0]],
                               mesh->vertices()[invalid_vid[1]], plane, p2,
                               interp2);

      valid_vertices.push_back(p1);
      valid_vertices.push_back(p2);

      Eigen::Vector3f tmp_n = (p1 - mesh->vertices()[valid_vid[0]])
                                  .cross(p2 - mesh->vertices()[valid_vid[0]]);
      bool flipped = tmp_n.dot(mesh->face_normals()[i]) > 0;

      Eigen::Vector3i f =
          GenerateFace(added_table[valid_vid[0]],
                       static_cast<int>(valid_vertices.size() - 1),
                       static_cast<int>(valid_vertices.size() - 2), flipped);

      valid_indices.emplace_back(f);

      if (with_uv) {
        Eigen::Vector2f uv1 =
            LinearInterpolation(mesh->uv()[valid_uv_vid[0]],
                                mesh->uv()[invalid_uv_vid[0]], interp1);
        Eigen::Vector2f uv2 =
            LinearInterpolation(mesh->uv()[valid_uv_vid[0]],
                                mesh->uv()[invalid_uv_vid[1]], interp2);

        valid_uv.push_back(uv1);
        valid_uv.push_back(uv2);

        Eigen::Vector3i uv_f =
            GenerateFace(added_uv_table[valid_uv_vid[0]],
                         static_cast<int>(valid_uv.size() - 1),
                         static_cast<int>(valid_uv.size() - 2), flipped);

        valid_uv_indices.push_back(uv_f);
      }

      if (with_vertex_color) {
        Eigen::Vector3f vc1 = LinearInterpolation(
            mesh->vertex_colors()[valid_uv_vid[0]],
            mesh->vertex_colors()[invalid_uv_vid[0]], interp1);
        Eigen::Vector3f vc2 = LinearInterpolation(
            mesh->vertex_colors()[valid_uv_vid[0]],
            mesh->vertex_colors()[invalid_uv_vid[1]], interp2);

        valid_vertex_colors.push_back(vc1);
        valid_vertex_colors.push_back(vc2);
      }

    } else if (valid_vid.size() == 2) {
      // Add two new vertices to make two faces
      //      -----------
      //     |∖new face 1/
      //     |  ∖       /   inside
      //     |new ∖    /
      //     |face 2∖ /
      //     |-------/    plane
      //     |      /
      //     |     /
      //     |    /     outside
      //     |   /
      //     |  /
      //     | /
      //     original triangle

      Eigen::Vector3f p1, p2;
      float interp1, interp2;
      CalcIntersectionAndRatio(mesh->vertices()[valid_vid[0]],
                               mesh->vertices()[invalid_vid[0]], plane, p1,
                               interp1);
      CalcIntersectionAndRatio(mesh->vertices()[valid_vid[1]],
                               mesh->vertices()[invalid_vid[0]], plane, p2,
                               interp2);

      valid_vertices.push_back(p1);
      valid_vertices.push_back(p2);

      Eigen::Vector3f tmp_n1 = (p1 - mesh->vertices()[valid_vid[0]])
                                   .cross(p2 - mesh->vertices()[valid_vid[0]]);
      bool flipped1 = tmp_n1.dot(mesh->face_normals()[i]) > 0;
      Eigen::Vector3i f1 =
          GenerateFace(added_table[valid_vid[0]],
                       static_cast<int>(valid_vertices.size() - 1),
                       static_cast<int>(valid_vertices.size() - 2), flipped1);
      valid_indices.emplace_back(f1);

      Eigen::Vector3f tmp_n2 = -(mesh->vertices()[valid_vid[0]] - p2)
                                    .cross(mesh->vertices()[valid_vid[1]] - p2);
      bool flipped2 = tmp_n2.dot(mesh->face_normals()[i]) > 0;
      Eigen::Vector3i f2 =
          GenerateFace(added_table[valid_vid[0]], added_table[valid_vid[1]],
                       static_cast<int>(valid_vertices.size() - 1), flipped2);
      valid_indices.emplace_back(f2);

      if (with_uv) {
        Eigen::Vector2f uv1 =
            LinearInterpolation(mesh->uv()[valid_uv_vid[0]],
                                mesh->uv()[invalid_uv_vid[0]], interp1);
        Eigen::Vector2f uv2 =
            LinearInterpolation(mesh->uv()[valid_uv_vid[1]],
                                mesh->uv()[invalid_uv_vid[0]], interp2);

        valid_uv.push_back(uv1);
        valid_uv.push_back(uv2);

        Eigen::Vector3i uv_f1 =
            GenerateFace(added_uv_table[valid_uv_vid[0]],
                         static_cast<int>(valid_uv.size() - 1),
                         static_cast<int>(valid_uv.size() - 2), flipped1);

        valid_uv_indices.push_back(uv_f1);

        Eigen::Vector3i uv_f2 = GenerateFace(
            added_uv_table[valid_uv_vid[0]], added_uv_table[valid_uv_vid[1]],
            static_cast<int>(valid_uv.size() - 1), flipped2);

        valid_uv_indices.push_back(uv_f2);
      }

      if (with_vertex_color) {
        Eigen::Vector3f vc1 = LinearInterpolation(
            mesh->vertex_colors()[valid_uv_vid[0]],
            mesh->vertex_colors()[invalid_uv_vid[0]], interp1);
        Eigen::Vector3f vc2 = LinearInterpolation(
            mesh->vertex_colors()[valid_uv_vid[1]],
            mesh->vertex_colors()[invalid_uv_vid[0]], interp2);

        valid_vertex_colors.push_back(vc1);
        valid_vertex_colors.push_back(vc2);
      }

    } else {
      // All vertices are kept
      valid_indices.push_back(face);
      valid_face_table[i] = valid_face_count;
      valid_face_count++;
      if (with_uv) {
        valid_uv_indices.push_back(uv_face);
      }
    }
  }

  mesh->set_vertices(valid_vertices);
  mesh->set_vertex_indices(valid_indices);
  if (with_uv) {
    mesh->set_uv(valid_uv);
    mesh->set_uv_indices(valid_uv_indices);
  }
  if (with_vertex_color) {
    mesh->set_vertex_colors(valid_vertex_colors);
  }
  mesh->CalcNormal();

  std::vector<int> new_material_ids(valid_indices.size(), 0);
  const std::vector<int>& old_material_ids = mesh->material_ids();

  for (size_t i = 0; i < old_material_ids.size(); i++) {
    int org_f_idx = static_cast<int>(i);
    int new_f_idx = valid_face_table[org_f_idx];
    if (new_f_idx < 0) {
      continue;
    }
    new_material_ids[new_f_idx] = old_material_ids[org_f_idx];
  }

  mesh->set_material_ids(new_material_ids);

  if (fill_plane) {
    // TODO: fill plane by triangulation
  }

  return num_removed;
}

std::optional<Eigen::Vector3f> Intersect(const Eigen::Vector3f& origin,
                                         const Eigen::Vector3f& ray,
                                         const Eigen::Vector3f& v0,
                                         const Eigen::Vector3f& v1,
                                         const Eigen::Vector3f& v2,
                                         const float kEpsilon,
                                         bool standard_barycentric) {
  return IntersectImpl(origin, ray, v0, v1, v2, kEpsilon, standard_barycentric);
}

std::optional<Eigen::Vector3d> Intersect(const Eigen::Vector3d& origin,
                                         const Eigen::Vector3d& ray,
                                         const Eigen::Vector3d& v0,
                                         const Eigen::Vector3d& v1,
                                         const Eigen::Vector3d& v2,
                                         const double kEpsilon,
                                         bool standard_barycentric) {
  return IntersectImpl(origin, ray, v0, v1, v2, kEpsilon, standard_barycentric);
}

std::vector<IntersectResult> Intersect(
    const Eigen::Vector3f& origin, const Eigen::Vector3f& ray,
    const std::vector<Eigen::Vector3f>& vertices,
    const std::vector<Eigen::Vector3i>& faces, int num_threads,
    const float kEpsilon, bool standard_barycentric) {
  std::vector<IntersectResult> results;

  if (num_threads != 1) {
    // Multi-thread version with mutex
    //  Slower for bunny(#face=30k) but can become faster if geometry is pretty
    //  large?
    std::mutex mtx;
    auto func = [&](size_t i) {
      const auto& f = faces[i];
      const auto& v0 = vertices[f[0]];
      const auto& v1 = vertices[f[1]];
      const auto& v2 = vertices[f[2]];
      // Use double for numerical reason
      auto ret =
          Intersect(origin.cast<double>(), ray.cast<double>(),
                    v0.cast<double>(), v1.cast<double>(), v2.cast<double>(),
                    double(kEpsilon), standard_barycentric);
      if (ret) {
        const auto& tuv = ret.value().cast<float>();
        std::lock_guard<std::mutex> lock(mtx);
        IntersectResult result;
        result.fid = static_cast<uint32_t>(i);
        result.t = tuv[0];
        result.u = tuv[1];
        result.v = tuv[2];
        results.emplace_back(result);
      }
    };
    parallel_for(size_t(0), faces.size(), func, num_threads);
  } else {
    // Single thead version without mutex
    // Faster for bunny(#face=30k) and can be parallelized on the upper level.
    for (size_t i = 0; i < faces.size(); i++) {
      const auto& f = faces[i];
      const auto& v0 = vertices[f[0]];
      const auto& v1 = vertices[f[1]];
      const auto& v2 = vertices[f[2]];
      // Use double for numerical reason
      auto ret =
          Intersect(origin.cast<double>(), ray.cast<double>(),
                    v0.cast<double>(), v1.cast<double>(), v2.cast<double>(),
                    double(kEpsilon), standard_barycentric);
      if (ret) {
        const auto& tuv = ret.value().cast<float>();
        IntersectResult result;
        result.fid = static_cast<uint32_t>(i);
        result.t = tuv[0];
        result.u = tuv[1];
        result.v = tuv[2];
        results.emplace_back(result);
      }
    }
  }

  return results;
}

bool UpdateVertexAttrOneRingMost(uint32_t num_vertices,
                                 const std::vector<Eigen::Vector3i>& indices,
                                 std::vector<Eigen::Vector3f>& attrs,
                                 const Adjacency& vert_adjacency,
                                 const FaceAdjacency& face_adjacency) {
  Adjacency vert_adjacency_;
  if (vert_adjacency.empty()) {
    FaceAdjacency face_adjacency_;
    if (face_adjacency.Empty()) {
      face_adjacency_.Init(num_vertices, indices);
    } else {
      face_adjacency_ = face_adjacency;
    }
    vert_adjacency_ = face_adjacency_.GenerateVertexAdjacency();
  } else {
    vert_adjacency_ = vert_adjacency;
  }

  std::vector<Eigen::Vector3f> org_attrs = attrs;

  for (uint32_t i = 0; i < num_vertices; i++) {
    const auto& va = vert_adjacency_[i];
    if (va.empty()) {
      continue;
    }
    int mode_frequency = -1;
    std::unordered_map<Eigen::Vector3f, int> occurrence;

    std::vector<Eigen::Vector3f> one_ring_attrs;
    std::transform(va.begin(), va.end(), std::back_inserter(one_ring_attrs),
                   [&](const int32_t& ii) { return org_attrs[ii]; });

    Mode(one_ring_attrs.begin(), one_ring_attrs.end(), attrs[i], mode_frequency,
         occurrence);
  }

  return true;
}

std::tuple<std::vector<Eigen::Vector3f>, std::vector<Eigen::Vector3i>>
ExtractSubGeom(const std::vector<Eigen::Vector3f>& vertices,
               const std::vector<Eigen::Vector3i>& faces,
               const std::vector<uint32_t>& sub_face_ids) {
  std::vector<Eigen::Vector3f> sub_vertices;
  std::vector<Eigen::Vector3i> sub_faces;

  std::unordered_map<int32_t, int32_t> org2seg;
  int32_t count = 0;
  for (const auto& org_fid : sub_face_ids) {
    Eigen::Vector3i face;
    for (int i = 0; i < 3; i++) {
      int org_vid = faces[org_fid][i];
      int vid = -1;
      if (org2seg.find(org_vid) != org2seg.end()) {
        // Found
        vid = org2seg[org_vid];
      } else {
        // New vid
        org2seg.insert({org_vid, count});
        sub_vertices.push_back(vertices[org_vid]);
        vid = count;
        count++;
      }
      face[i] = vid;
    }
    sub_faces.push_back(face);
  }

  return {sub_vertices, sub_faces};
}

}  // namespace ugu
