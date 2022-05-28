/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "ugu/util/geom_util.h"

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
                               const typename T::Scalar kEpsilon) {
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

}  // namespace

namespace ugu {

bool MergeMeshes(const Mesh& src1, const Mesh& src2, Mesh* merged,
                 bool use_src1_material) {
  std::vector<Eigen::Vector3f> vertices, vertex_colors, vertex_normals;
  std::vector<Eigen::Vector2f> uv;
  std::vector<int> material_ids, offset_material_ids2;
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

  if (use_src1_material) {
    CopyVec(src1.materials(), &materials);

    CopyVec(src1.material_ids(), &material_ids);

    // Is using original material_ids for src2 right?
    CopyVec(src2.material_ids(), &material_ids, false);
  } else {
    CopyVec(src1.materials(), &materials);

    std::vector<ObjMaterial> src2_materials = src2.materials();
    // Check if src2 has the same material name to src1
    for (size_t i = 0; i < src2_materials.size(); i++) {
      auto& mat2 = src2_materials[i];
      bool has_same_name = false;
      for (const auto& mat : materials) {
        if (mat2.name == mat.name) {
          has_same_name = true;
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

    CopyVec(src2_materials, &materials, false);

    CopyVec(src1.material_ids(), &material_ids);
    CopyVec(src2.material_ids(), &offset_material_ids2);
    int offset_mi = static_cast<int>(src1.materials().size());
    std::for_each(offset_material_ids2.begin(), offset_material_ids2.end(),
                  [offset_mi](int& i) { i += offset_mi; });
    CopyVec(offset_material_ids2, &material_ids, false);
  }

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

bool MergeMeshes(const std::vector<MeshPtr>& src_meshes, Mesh* merged) {
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
    ugu::MergeMeshes(tmp0, *src, &tmp2);
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

  auto [boundary_edges_list, boundary_vertex_ids_list] =
      FindBoundaryLoops(indices, vnum);

  clusters.resize(boundary_vertex_ids_list.size());
  clusters_f.resize(boundary_vertex_ids_list.size());

  auto v2f = ugu::GenerateVertex2FaceMap(indices, vnum);

  std::set<int32_t> non_orphans;

  for (size_t i = 0; i < boundary_vertex_ids_list.size(); i++) {
    auto& cluster = clusters[i];

    for (const auto& vid : boundary_vertex_ids_list[i]) {
      cluster.insert(vid);
    }

    while (true) {
      std::set<int32_t> new_vids;
      for (const auto& vid : cluster) {
        const auto& f_list = v2f[vid];

        for (const auto& f : f_list) {
          for (int32_t j = 0; j < 3; j++) {
            const auto& new_vid = indices[f][j];

            if (cluster.count(new_vid) == 0) {
              new_vids.insert(new_vid);
            }
          }
        }
      }

      if (new_vids.empty()) {
        break;
      }

      cluster.insert(new_vids.begin(), new_vids.end());
    }

    non_orphans.insert(cluster.begin(), cluster.end());
    for (const auto& vid : cluster) {
      const auto& f_list = v2f[vid];
      for (const auto& f : f_list) {
        clusters_f[i].insert(f);
      }
    }
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
                  const Eigen::Vector3f& t) {
  MeshPtr plane = std::make_shared<Mesh>();
  std::vector<Eigen::Vector3f> vertices(4);
  std::vector<Eigen::Vector3i> vertex_indices(2);
  const std::vector<Eigen::Vector2f> uvs{
      {1.f, 1.f}, {0.f, 1.f}, {1.f, 0.f}, {0.f, 0.f}};

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

  plane->CalcNormal();

  plane->CalcStats();

  return plane;
}

MeshPtr MakePlane(float length, const Eigen::Matrix3f& R,
                  const Eigen::Vector3f& t) {
  return MakePlane(Eigen::Vector2f(length, length), R, t);
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
                                         const float kEpsilon) {
  return IntersectImpl(origin, ray, v0, v1, v2, kEpsilon);
}

std::optional<Eigen::Vector3d> Intersect(const Eigen::Vector3d& origin,
                                         const Eigen::Vector3d& ray,
                                         const Eigen::Vector3d& v0,
                                         const Eigen::Vector3d& v1,
                                         const Eigen::Vector3d& v2,
                                         const double kEpsilon) {
  return IntersectImpl(origin, ray, v0, v1, v2, kEpsilon);
}

std::vector<IntersectResult> Intersect(
    const Eigen::Vector3f& origin, const Eigen::Vector3f& ray,
    const std::vector<Eigen::Vector3f>& vertices,
    const std::vector<Eigen::Vector3i>& faces, int num_threads,
    const float kEpsilon) {
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
      auto ret = Intersect(origin.cast<double>(), ray.cast<double>(),
                           v0.cast<double>(), v1.cast<double>(),
                           v2.cast<double>(), double(kEpsilon));
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
      auto ret = Intersect(origin.cast<double>(), ray.cast<double>(),
                           v0.cast<double>(), v1.cast<double>(),
                           v2.cast<double>(), double(kEpsilon));
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

}  // namespace ugu
