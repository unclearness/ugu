/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "ugu/util/geom_util.h"

#include <random>

#include "ugu/face_adjacency.h"

namespace {

template <typename T>
void CopyVec(const std::vector<T>& src, std::vector<T>* dst,
             bool clear_dst = true) {
  if (clear_dst) {
    dst->clear();
    dst->reserve(src.size());
  }
  std::copy(src.begin(), src.end(), std::back_inserter(*dst));
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

bool MergeMeshes(const std::vector<std::shared_ptr<Mesh>>& src_meshes,
                 Mesh* merged) {
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
FindBoundaryLoops(const Mesh& mesh) {
  std::vector<std::vector<std::pair<int, int>>> boundary_edges_list;
  std::vector<std::vector<int>> boundary_vertex_ids_list;

  ugu::FaceAdjacency face_adjacency;
  face_adjacency.Init(static_cast<int>(mesh.vertices().size()),
                      mesh.vertex_indices());

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
    for (auto i = 0; i < boundary_edges.size(); i++) {
      const auto& e = boundary_edges[i];
      if (cur_edge.second == e.first) {
        found_connected = true;
        connected_index = i;
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
      for (auto i = 0; i < cur_edges.size(); i++) {
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

std::shared_ptr<Mesh> MakeCube(const Eigen::Vector3f& length,
                               const Eigen::Matrix3f& R,
                               const Eigen::Vector3f& t) {
  std::shared_ptr<Mesh> cube(new Mesh);
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

std::shared_ptr<Mesh> MakeCube(const Eigen::Vector3f& length) {
  const Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
  const Eigen::Vector3f t(0.0f, 0.0f, 0.0f);
  return MakeCube(length, R, t);
}

std::shared_ptr<Mesh> MakeCube(float length, const Eigen::Matrix3f& R,
                               const Eigen::Vector3f& t) {
  Eigen::Vector3f length_xyz{length, length, length};
  return MakeCube(length_xyz, R, t);
}

std::shared_ptr<Mesh> MakeCube(float length) {
  const Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
  const Eigen::Vector3f t(0.0f, 0.0f, 0.0f);
  return MakeCube(length, R, t);
}

void SetRandomVertexColor(std::shared_ptr<Mesh> mesh, int seed) {
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

}  // namespace ugu
