/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 *
 */

#include "ugu/decimation/decimation.h"

#include <iostream>
#include <queue>
#include <random>
#include <unordered_set>

#include "ugu/face_adjacency.h"
#include "ugu/util/geom_util.h"
#include "ugu/util/math_util.h"

using QSlimEdge = std::pair<int32_t, int32_t>;

struct QSlimUvEdge {
  QSlimEdge edge;
  QSlimEdge uv_edge;

  QSlimUvEdge(int32_t e1, int32_t e2, int32_t uv_e1, int32_t uv_e2) {
    bool flipped = false;

    if (e1 < e2) {
      edge.first = e1;
      edge.second = e2;
    } else {
      edge.first = e2;
      edge.second = e1;
      flipped = true;
    }

    if (!flipped) {
      uv_edge.first = uv_e1;
      uv_edge.second = uv_e2;
    } else {
      uv_edge.first = uv_e2;
      uv_edge.second = uv_e1;
    }
  }

  bool operator==(const QSlimUvEdge& value) const {
    return edge.first == value.edge.first && edge.second == value.edge.second;
  }
};

namespace std {

template <>
struct hash<QSlimUvEdge> {
  std::size_t operator()(const QSlimUvEdge& k) const {
    const auto hash1 = std::hash<int>()(k.edge.first);
    const auto hash2 = std::hash<int>()(k.edge.second);

    return hash1 ^ (hash2 << 1);
  }
};

}  // namespace std

namespace {

#if 1
struct DecimatedMesh {
  ugu::MeshPtr mesh;

  std::vector<Eigen::Vector3f> vertices;
  std::vector<Eigen::Vector3f> vertex_colors;   // optional, RGB order
  std::vector<Eigen::Vector3i> vertex_indices;  // face

  std::vector<Eigen::Vector3f> normals;  // normal per vertex
  // std::vector<Eigen::Vector3f> face_normals;  // normal per face
  // std::vector<Eigen::Vector3i> normal_indices;

  std::vector<Eigen::Vector2f> uv;
  std::vector<Eigen::Vector3i> uv_indices;

  std::vector<bool> valid_vertices;
  std::vector<bool> valid_faces;

  std::vector<std::vector<int32_t>> vid2uvid;
  std::vector<int32_t> uvid2vid;
  std::unordered_map<int, std::vector<int>> v2f, uv_v2f;

  // std::set<QSlimEdge> valid_pairs;
  // std::unordered_set<int32_t> invalid_vids;

  ugu::FaceAdjacency face_adjacency, uv_face_adjacency;

  std::unordered_set<int> unified_boundary_vertex_ids;

  bool use_uv = false;

  DecimatedMesh(ugu::MeshPtr mesh) : mesh(mesh) {
    valid_vertices.resize(mesh->vertices().size(), true);
    valid_faces.resize(mesh->vertex_indices().size(), true);

    vertices = mesh->vertices();
    vertex_colors = mesh->vertex_colors();
    vertex_indices = mesh->vertex_indices();
    normals = mesh->normals();

    if (vertex_colors.empty()) {
      vertex_colors.resize(vertices.size());
    }

    uv = mesh->uv();
    uv_indices = mesh->uv_indices();

    use_uv = !uv.empty() && !uv_indices.empty();

    if (use_uv) {
      vid2uvid.resize(vertices.size());
      uvid2vid.resize(uv.size());
      for (size_t i = 0; i < vertex_indices.size(); i++) {
        const auto& f = vertex_indices[i];
        const auto& uv_f = uv_indices[i];
        for (int j = 0; j < 3; j++) {
          int32_t vid = f[j];
          int32_t uv_id = uv_f[j];
          vid2uvid[vid].push_back(uv_id);
          uvid2vid[uv_id] = vid;
        }
      }

      uv_face_adjacency.Init(mesh->uv().size(), mesh->uv_indices());
      uv_v2f =
          ugu::GenerateVertex2FaceMap(mesh->uv_indices(), mesh->uv().size());
    }

    face_adjacency.Init(mesh->vertices().size(), mesh->vertex_indices());
    v2f = ugu::GenerateVertex2FaceMap(mesh->vertex_indices(),
                                      mesh->vertices().size());

    auto [boundary_edges_list, boundary_vertex_ids_list] =
        ugu::FindBoundaryLoops(*mesh);

    for (const auto& list : boundary_vertex_ids_list) {
      for (const auto& id : list) {
        unified_boundary_vertex_ids.insert(id);
      }
    }
  }

  int32_t VertexNum() const {
    return std::count(valid_vertices.begin(), valid_vertices.end(), true);
  }

  int32_t FaceNum() const {
    return std::count(valid_faces.begin(), valid_faces.end(), true);
  }

  auto FacesContactingEdges(int32_t v0, int32_t v1) {
    const auto& v0_f = v2f[v0];
    std::unordered_set<int32_t> union_fids(v0_f.begin(), v0_f.end());
    const auto& v1_f = v2f[v1];
    for (const auto& fid : v1_f) {
      union_fids.insert(fid);
    }

    for (const auto& fid : union_fids) {
      assert(valid_faces[fid]);
    }

    assert(valid_vertices[v0] && valid_vertices[v1]);

    std::unordered_set<int32_t> no_intersection = union_fids;
    std::unordered_set<int32_t> intersection;
    std::set_intersection(v0_f.begin(), v0_f.end(), v1_f.begin(), v1_f.end(),
                          std::inserter(intersection, intersection.end()));
#if 0
				    if (intersection.size()){
      std::cout << "v0f" << std::endl;
      for (auto f : v0_f) {
        std::cout << f << " " << vertex_indices[f] << std::endl;
      }

      std::cout << "v1f" << std::endl;
      for (auto f : v1_f) {
        std::cout << f << " " << vertex_indices[f] << std::endl;
      }

    }
#endif  // 0

    // For non-manifold meshes, always 2
    assert(intersection.size() == 2);

    for (const auto& fid : intersection) {
      no_intersection.erase(fid);
    }

    return std::make_tuple(union_fids, intersection, no_intersection);
  }

  std::tuple<bool, Eigen::Vector3i, Eigen::Vector3i> RemoveFace(int32_t fid) {
    if (!valid_faces[fid]) {
      throw std::exception("Invalid");
      // return {false, vertex_indices[fid], uv_indices[fid]};
    }

    // Remove from v2f

    // std::cout << "fid " << fid << std::endl;
    for (int32_t i = 0; i < 3; i++) {
      int32_t vid = vertex_indices[fid][i];
      assert(valid_vertices[vid]);
      // std::cout << vid << " before " << v2f[vid].size() << std::endl;
      auto result = std::remove(v2f[vid].begin(), v2f[vid].end(), fid);
      v2f[vid].erase(result, v2f[vid].end());
      // std::cout << vid << " after " << v2f[vid].size() << std::endl;
    }

    valid_faces[fid] = false;
    vertex_indices[fid].setConstant(99999);
    // face_adjacency.RemoveFace(fid);
    // uv_face_adjacency.RemoveFace(fid);

    return {true, vertex_indices[fid], uv_indices[fid]};
  }

  void RemoveVertex(int32_t vid) {
    // Remove a vetex
    if (!valid_vertices[vid]) {
      // If vertex is alreay invalid, do nothing
      // return {face_ids, edge_count};
      throw std::exception("");
    }
    valid_vertices[vid] = false;
    vertices[vid].setConstant(99999);

    // Clear v2f
    v2f[vid].clear();

    // Remove from boundary
    // unified_boundary_vertex_ids.erase(vid);
  }

  bool CollapseEdge(int32_t v1, int32_t v2, const Eigen::Vector3f& org_normal,
                    const Eigen::Vector3f& new_pos,
                    const Eigen::Vector3f& new_normal,
                    const Eigen::Vector3f& new_color,
                    const Eigen::Vector2f& new_uv) {
    std::cout << "decimate " << v1 << " " << v2 << std::endl;

    // Get faces connecting the 2 vertices (A, B)
    auto [union_fids, intersection, no_intersection] =
        FacesContactingEdges(v1, v2);

    std::unordered_set<int32_t> to_remove_face_ids = std::move(intersection);
    std::unordered_set<int32_t> to_keep_face_ids = std::move(no_intersection);

    // Replace of B among the faces with A
    for (const auto& fid : to_keep_face_ids) {
      for (int32_t i = 0; i < 3; i++) {
        int32_t vid = vertex_indices[fid][i];
        if (vid == v2) {
          vertex_indices[fid][i] = v1;
        }
        assert(valid_faces[fid]);
      }
    }

    // Update the vertex A's attribute by new ones
    // valid_vertices[v1] = true;
    vertices[v1] = new_pos;
    normals[v1] = new_normal;
    vertex_colors[v1] = new_color;
    if (use_uv) {
      uv[vid2uvid[v1][0]] = new_uv;
    }

    // Remove 2 faces which share edge A-B from the faces
    for (const auto& fid : to_remove_face_ids) {
      RemoveFace(fid);
    }

    // Merge v2f of B to A
    std::copy(v2f[v2].begin(), v2f[v2].end(), std::back_inserter(v2f[v1]));
    std::sort(v2f[v1].begin(), v2f[v1].end());

    // Remove the vertex B with older id
    RemoveVertex(v2);

    return true;
  }

  void Finalize() {
    // Appy valid mask
#if 0
				    vertices = ugu::Mask(vertices, valid_vertices);
    vertex_colors = ugu::Mask(vertex_colors, valid_vertices);
    normals = ugu::Mask(normals, valid_vertices);



    vertex_indices = ugu::Mask(vertex_indices, valid_faces);
    std::vector<bool> valid_uv(uv.size(), false);
    std::vector<bool> valid_uv_faces(uv_indices.size(), false);

    for (size_t i = 0; i < vid2uvid.size(); i++) {
      for (size_t j = 0; j < vid2uvid[i].size(); j++) {
        valid_uv[vid2uvid[i][j]] = valid_vertices[i];
      }
    }

    for (size_t i = 0; i < uv_indices.size(); i++) {
      bool valid = true;
      for (int j = 0; j < 3; j++) {
        if (!valid_uv[uv_indices[i][j]]) {
          valid = false;
          break;
        }
      }
      valid_uv_faces[i] = valid;
    }

    uv = ugu::Mask(uv, valid_uv);
    uv_indices = ugu::Mask(uv_indices, valid_uv_faces);

#endif  // 0

    mesh->Clear();

    auto material_ids = mesh->material_ids();

    RemoveVertices(valid_vertices, valid_faces);

    mesh->set_material_ids(material_ids);

#if 0
    // copy
    mesh->set_vertices(vertices);

    mesh->set_vertex_colors(vertex_colors);
    mesh->set_vertex_indices(vertex_indices);
    mesh->set_normals(normals);
    mesh->set_uv(uv);
    mesh->set_uv_indices(uv_indices);
#endif
  }

  int RemoveVertices(const std::vector<bool>& valid_vertex_table,
                     const std::vector<bool>& valid_face_table_) {
    if (valid_vertex_table.size() != vertices.size()) {
      ugu::LOGE("valid_vertex_table must be same size to vertices");
      return -1;
    }

    int num_removed{0};
    std::vector<int> valid_table(vertices.size(), -1);
    std::vector<Eigen::Vector3f> valid_vertices, valid_vertex_colors;
    std::vector<Eigen::Vector2f> valid_uv;
    std::vector<Eigen::Vector3i> valid_indices;
    bool with_uv = !uv.empty() && !uv_indices.empty();
    bool with_vertex_color = !vertex_colors.empty();
    int valid_count = 0;
    for (size_t i = 0; i < vertices.size(); i++) {
      if (valid_vertex_table[i]) {
        valid_table[i] = valid_count;
        valid_vertices.push_back(vertices[i]);
        if (with_uv) {
          valid_uv.push_back(uv[i]);
        }
        if (with_vertex_color) {
          valid_vertex_colors.push_back(vertex_colors[i]);
        }
        valid_count++;
      } else {
        num_removed++;
      }
    }

    int valid_face_count{0};
    std::vector<int> valid_face_table(vertex_indices.size(), -1);
    for (size_t i = 0; i < vertex_indices.size(); i++) {
      Eigen::Vector3i face;
      if (!valid_face_table_[i]) {
        continue;
      }
      bool valid{true};
      for (int j = 0; j < 3; j++) {
        int new_index = valid_table[vertex_indices[i][j]];
        if (new_index < 0) {
          valid = false;
          break;
        }
        face[j] = new_index;
      }
      if (!valid) {
        continue;
      }
      valid_indices.push_back(face);
      valid_face_table[i] = valid_face_count;
      valid_face_count++;
    }

    mesh->set_vertices(valid_vertices);
    mesh->set_vertex_indices(valid_indices);
    if (with_uv) {
      mesh->set_uv(valid_uv);
      mesh->set_uv_indices(valid_indices);
    }
    if (with_vertex_color) {
      mesh->set_vertex_colors(valid_vertex_colors);
    }
    mesh->CalcNormal();

#if 0
				    std::vector<int> new_material_ids(valid_indices.size(), 0);
    const std::vector<int>& old_material_ids = material_ids_;

    for (size_t i = 0; i < old_material_ids.size(); i++) {
      int org_f_idx = static_cast<int>(i);
      int new_f_idx = valid_face_table[org_f_idx];
      if (new_f_idx < 0) {
        continue;
      }
      new_material_ids[new_f_idx] = old_material_ids[org_f_idx];
    }

    set_material_ids(new_material_ids);

#endif  // 0

    return num_removed;
  }

#if 0
  std::set<QSlimEdge> PrepareValidPairs(bool keep_geom_boundary,
                                        bool keep_uv_boundary) {
    face_adjacency.Init(mesh->vertices().size(), mesh->vertex_indices());
    if (keep_geom_boundary) {
      auto [boundary_edges, boundary_vertex_ids] =
          face_adjacency.GetBoundaryEdges();

      for (const auto& vid : boundary_vertex_ids) {
        invalid_vids.insert(vid);
      }
    }

    if (keep_uv_boundary) {
      // Treat uv vids by converting them to corresponding geom vids
      // TODO: direct solution?
      uv_face_adjacency.Init(mesh->uv().size(), mesh->uv_indices());
      auto [uv_boundary_edges, uv_boundary_vertex_ids] =
          uv_face_adjacency.GetBoundaryEdges();
      uv_v2f =
          ugu::GenerateVertex2FaceMap(mesh->uv_indices(), mesh->uv().size());

      for (const auto& uv_vid : uv_boundary_vertex_ids) {
        // Find the vertex id for geometry face (not uv)
        const auto uv_f_id = uv_v2f[uv_vid][0];
        const auto uv_f = mesh->vertex_indices()[uv_f_id];
        // Get index in uv face
        int index = 0;
        for (int i = 0; i < 3; i++) {
          if (uv_f[i] == uv_vid) {
            index = i;
            break;
          }
        }

        // Convert to geom face
        const auto geom_f = mesh->vertex_indices()[uv_f_id];
        // Convert to geom vid
        int vid = geom_f[index];

        invalid_vids.insert(vid);
      }
    }

    for (const auto& f : mesh->vertex_indices()) {
      size_t v0c = invalid_vids.count(f[0]);
      size_t v1c = invalid_vids.count(f[1]);
      size_t v2c = invalid_vids.count(f[2]);

      // Pair keeps always (smaller, bigger)
      // Is this okay for geometry face reconstruction?

      if (v0c == 0 && v1c == 0) {
        int32_t v0 = std::min(f[0], f[1]);
        int32_t v1 = std::max(f[0], f[1]);
        valid_pairs.insert(std::make_pair(v0, v1));
      }

      if (v1c == 0 && v2c == 0) {
        int32_t v1 = std::min(f[1], f[2]);
        int32_t v2 = std::max(f[1], f[2]);
        valid_pairs.insert(std::make_pair(v1, v2));
      }

      if (v2c == 0 && v0c == 0) {
        int32_t v2 = std::min(f[0], f[2]);
        int32_t v0 = std::max(f[0], f[2]);
        valid_pairs.insert(std::make_pair(v2, v0));
      }
    }

    return valid_pairs;
  }
#endif

  DecimatedMesh() {}
  ~DecimatedMesh() {}
};
#endif  // 1

}  // namespace

namespace ugu {

bool QSlim(MeshPtr mesh, QSlimType type, int32_t target_face_num,
           int32_t target_vertex_num, bool keep_geom_boundary,
           bool keep_uv_boundary) {
#if 0
				  // Set up valid pairs (edge and non-dege)
  DecimatedMesh decimated_mesh(mesh);
  decimated_mesh.PrepareValidPairs(keep_geom_boundary, keep_uv_boundary);

  // Initialize quadrics
  // QuadricsPtr quadrics = Ini
  // decimated_mesh.quadrics = quadrics;
  decimated_mesh.InitializeQuadrics();

  // Initialize heap
  QSlimHeap heap;

  // Add the valid pairs  to heap
  for (const auto& p : decimated_mesh.valid_pairs) {
    QSlimEdgeInfo info(p, decimated_mesh.vert_attrs, decimated_mesh.quadrics,
                       false);
    heap.push(info);
  }

  // Main loop
  while (!heap.empty()) {
    if (target_vertex_num > 0 &&
        target_vertex_num <= decimated_mesh.VertexNum()) {
      break;
    }

    if (target_face_num > 0 && target_face_num <= decimated_mesh.FaceNum()) {
      break;
    }

    // Find the lowest error pair
    auto min_e = heap.top();
    heap.pop();

    // Decimate the pair
    auto new_edges = decimated_mesh.UpdateDecimatedVertex(min_e, type);

    // Add new pairs
    for (const auto& e : new_edges) {
      heap.push(e);
    }
  }

  decimated_mesh.Finalize();

#endif  // 0

#if 1
  DecimatedMesh decimated_mesh(mesh);
  std::mt19937 engine(0);
  std::uniform_real_distribution<double> dist1(0, 1.0);

  while (true) {
    if (target_vertex_num > 0 &&
        target_vertex_num >= decimated_mesh.VertexNum()) {
      break;
    }

    if (target_face_num > 0 && target_face_num >= decimated_mesh.FaceNum()) {
      break;
    }

#if 0
				  int32_t vid_org = (decimated_mesh.VertexNum() - 1) * dist1(engine);
    int32_t vid = -1;
    int32_t count = 0;
    for (size_t i = 0; i < decimated_mesh.valid_vertices.size(); i++) {
      if (decimated_mesh.valid_vertices[i]) {
        count++;
        if (count == vid_org) {
          vid = i;
        }
      }
    }
#endif  // 0

    int32_t fid_org = (decimated_mesh.FaceNum() - 1) * dist1(engine);
    int32_t fid = -1;
    int32_t count = 0;
    for (size_t i = 0; i < decimated_mesh.valid_faces.size(); i++) {
      if (decimated_mesh.valid_faces[i]) {
        count++;
        if (count == fid_org) {
          fid = i;
          break;
        }
      }
    }
    if (fid < 0) {
      continue;
    }

    if (!decimated_mesh.valid_faces[fid]) {
      continue;
    }

    int32_t vid1 = decimated_mesh.vertex_indices[fid][0];
    int32_t vid2 = decimated_mesh.vertex_indices[fid][1];

    if (vid1 > vid2) {
      std::swap(vid1, vid2);
    }

    if (!decimated_mesh.valid_vertices[vid1] ||
        !decimated_mesh.valid_vertices[vid2]) {
      continue;
    }

    if (decimated_mesh.unified_boundary_vertex_ids.count(vid1) > 0 ||
        decimated_mesh.unified_boundary_vertex_ids.count(vid2) > 0) {
      continue;
    }

    Eigen::Vector3f new_pos, new_normal, new_color;
    Eigen::Vector2f new_uv;

    new_pos =
        0.5f * (decimated_mesh.vertices[vid1] + decimated_mesh.vertices[vid2]);

    decimated_mesh.CollapseEdge(vid1, vid2, decimated_mesh.normals[vid1],
                                new_pos, new_normal, new_color, new_uv);

    std::cout << decimated_mesh.FaceNum() << " " << decimated_mesh.VertexNum()
              << std::endl;
  }
#endif

  decimated_mesh.Finalize();

  return true;
}

}  // namespace ugu
