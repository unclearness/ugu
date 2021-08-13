/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "ugu/util/geom_util.h"

#include <random>

#include "ugu/face_adjacency.h"
#include "ugu/util/math_util.h"

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

}  // namespace ugu
