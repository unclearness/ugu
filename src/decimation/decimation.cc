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
#include "ugu/util/math_util.h"

namespace {

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

  void EnsureOrder() {
    //  if (edge.first > edge.second) {
    //   std::swap(edge.first, edge.second);
    //  }
    //  if (uv_edge.first > uv_edge.second) {
    //    std::swap(uv_edge.first, uv_edge.second);
    // }
  }

  bool operator<(const QSlimUvEdge& value) const {
    return edge.first < value.edge.first;
  }
};

// bool operator==(const QSlimEdges& v1, const QSlimEdges& v2) {
//  return v1.edge == v2.edge && v1.uv_edge == v2.uv_edge;
//}

#if 0
				class QSlimEdge : public std::pair<int32_t, int32_t> {
  QSlimEdge(int32_t a, int32_t b) {
    if (a < b) {
      this->first = a;
      this->second = b;
    }
    else {
      this->first = b;
      this->second = a;
    }
  }

};
#endif  // 0

#if 0
				
using VertexAttr = Eigen::VectorXd;
using VertexAttrs = std::vector<VertexAttr>;
using VertexAttrsPtr = std::shared_ptr<VertexAttrs>;



using Quadric = Eigen::MatrixXd;
using Quadrics = std::vector<Quadric>;
using QuadricPtr = std::shared_ptr<Quadric>;
using QuadricsPtr = std::shared_ptr<Quadrics>;

bool ComputeOptimalContraction(const VertexAttr& v1, const Quadric& q1,
                               const VertexAttr& v2, const Quadric& q2,
                               VertexAttr& v, double& error) {
  VertexAttr zero(v1.size());
  zero.setZero();
  zero[zero.size() - 1] = 1.0;

  bool ret = true;

  Quadric q = q1 + q2;

  if (std::abs(q.determinant()) < 0.00001) {
    // Not ivertible case
    ret = false;

    // Select best one from v1, v2 and (v1+v2)/2
    std::array<VertexAttr, 3> candidates = {v1, v2, (v1 + v2) * 0.5};
    double min_error = std::numeric_limits<double>::max();
    VertexAttr min_vert = v1;

    for (int i = 0; i < 3; i++) {
      double tmp_error = candidates[i].transpose() * q * candidates[i];
      if (tmp_error < min_error) {
        min_error = tmp_error;
        candidates[i];
        min_vert = candidates[i];
      }
    }

    error = min_error;
    v = min_vert;

  } else {
    // Invertible case
    // Eq. (1) in the paper
    Quadric q_inv = q.inverse();
    v = q_inv * zero;
    error = v.transpose() * q * v;
  }

  return ret;
}

struct QSlimEdgeInfo {
  QSlimEdge edge = {-1, -1};
  // QSlimEdge edge_uv = {-1, -1};
  // int org_vid = -1;
  // QSlimEdge org_edge = {-1, -1};
  double error = std::numeric_limits<double>::max();
  QuadricsPtr quadrics;
  VertexAttrsPtr vert_attrs;
  VertexAttr decimated_v;
  bool keep_this_edge = false;
  QSlimEdgeInfo(QSlimEdge edge_, VertexAttrsPtr vert_attrs_,
                QuadricsPtr quadrics_, bool keep_this_edge_) {
    edge = edge_;
    // org_vid = org_vid_;
    // error = error_;
    quadrics = quadrics_;
    vert_attrs = vert_attrs_;
    decimated_v.setConstant(std::numeric_limits<double>::max());
    keep_this_edge = keep_this_edge_;

    ComputeError();
  }

  void ComputeError() {
    if (keep_this_edge) {
      error = std::numeric_limits<double>::max();
    } else {
      int v0 = edge.first;
      int v1 = edge.second;
      ComputeOptimalContraction(vert_attrs->at(v0), quadrics->at(v0),
                                vert_attrs->at(v1), quadrics->at(v1),
                                decimated_v, error);
    }
  }

  QSlimEdgeInfo() {}
  ~QSlimEdgeInfo() {}
};

bool operator<(const QSlimEdgeInfo& l, const QSlimEdgeInfo& r) {
  return l.error > r.error;
};

using QSlimHeap =
    std::priority_queue<QSlimEdgeInfo, std::vector<QSlimEdgeInfo>>;

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



  std::unordered_map<int, std::vector<int>> v2f, uv_v2f;

  VertexAttrsPtr vert_attrs;
  QuadricsPtr quadrics;

  std::set<QSlimEdge> valid_pairs;
  std::unordered_set<int32_t> invalid_vids;

  ugu::FaceAdjacency face_adjacency, uv_face_adjacency;

  DecimatedMesh(ugu::MeshPtr mesh) : mesh(mesh) {
    valid_vertices.resize(mesh->vertices().size(), true);

    vertices = mesh->vertices();
    vertex_colors = mesh->vertex_colors();
    vertex_indices = mesh->vertex_indices();

    normals = mesh->normals();

    uv = mesh->uv();
    uv_indices = mesh->uv_indices();
  }

  int32_t VertexNum() const {
    return std::count(valid_vertices.begin(), valid_vertices.end(), true);
  }

  int32_t FaceNum() const {
    return std::count(valid_faces.begin(), valid_faces.end(), true);
  }


  bool RemoveFace(int32_t fid) {

    return true;

  }

  std::vector<int32_t> RemoveVertex(int32_t vid) {
    // Remove a vetex

    // Remove faces connected to the vertex

    std::vector<int32_t> face_ids;

    return face_ids;
  }

  int AddVertex(const Eigen::Vector3f& v) {

    // Search empty vid

    int vid;
    return vid;
  }


 int AddFace(const Eigen::Vector3i& face, const Eigen::Vector3i& uv_face) {

  // Search empty fid

    int fid;
    return fid;

  }



  void Finalize() {
    // Appy valid mask



    // copy
    mesh->set_vertices(vertices);
    mesh->set_vertex_colors(vertex_colors);
    mesh->set_vertex_indices(vertex_indices);

    mesh->set_normals(normals);

    mesh->set_uv(uv);
    mesh->set_uv_indices(uv_indices);
  }

  void Update(int vid, const QSlimEdgeInfo& e, VertexAttr& vertex_attr,
              double scale, ugu::QSlimType type) {
    // Extract vertex attributes from quadrics matrix
    // Eigen::Vector3f vertex, normal, vertex_color;
    // Eigen::Vector3f uv;
    if (type == ugu::QSlimType::XYZ) {
      vertex_attr[0] = e.decimated_v[0];
      vertex_attr[1] = e.decimated_v[1];
      vertex_attr[2] = e.decimated_v[2];
    }

    // Recover original scale of the vertex attributes

    // Set except uv and uv_indices

    // Set uv and uv_indices by index conversion
  }

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

  void InitializeQuadrics() {}

  std::vector<QSlimEdgeInfo> UpdateDecimatedVertex(QSlimEdgeInfo& e,
                                                   ugu::QSlimType type) {
    int32_t v1 = e.edge.first;
    int32_t v2 = e.edge.second;

    // Remove original two vertex attributes
    // Mark them as invalid
    valid_vertices[v1] = false;
    valid_vertices[v2] = false;

    // Add the decimated vertex attribtues
    // Set decimated vertex attributes to the one of the invalidated
    valid_vertices[v1] = true;
    //Eigen::Vector3f vertex, normal, vertex_color;
    //Eigen::Vector3f uv;
    Update(v1, e, vert_attrs->at(v1), 1.0, type);

    // Reconstruct new faces
    // Remove original face indices

    

    //face_adjacency.RemoveFace();  


    // Add new face indices
  


    std::vector<QSlimEdgeInfo> new_edges;
    return new_edges;
  }

  DecimatedMesh() {}
  ~DecimatedMesh() {}
};

#endif  // 0

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

    face_adjacency.Init(mesh->vertices().size(), mesh->vertex_indices());
    v2f = ugu::GenerateVertex2FaceMap(mesh->vertex_indices(),
                                      mesh->vertices().size());

    uv_face_adjacency.Init(mesh->uv().size(), mesh->uv_indices());
    uv_v2f = ugu::GenerateVertex2FaceMap(mesh->uv_indices(), mesh->uv().size());
  }

  int32_t VertexNum() const {
    return std::count(valid_vertices.begin(), valid_vertices.end(), true);
  }

  int32_t FaceNum() const {
    return std::count(valid_faces.begin(), valid_faces.end(), true);
  }

  std::tuple<bool, Eigen::Vector3i, Eigen::Vector3i> RemoveFace(int32_t fid) {
    if (!valid_faces[fid]) {
      // throw std::exception("Invalid");
      return {false, vertex_indices[fid], uv_indices[fid]};
    }
    valid_faces[fid] = false;
    face_adjacency.RemoveFace(fid);
    uv_face_adjacency.RemoveFace(fid);
    return {true, vertex_indices[fid], uv_indices[fid]};
  }

  std::pair<std::unordered_set<int32_t>, std::map<QSlimUvEdge, std::int32_t>>
  RemoveVertex(int32_t vid) {
    std::unordered_set<int32_t> face_ids;
    // std::unordered_set<int32_t> connected_vids;
    // std::unordered_set<int32_t> connected_uvids;

    // std::set<QSlimEdge> shared_edges;
    // std::set<QSlimEdge> shared_uv_edges;
    // std::set<QSlimEdge> left_edges;
    // std::unordered_set<QSlimUvEdge> and_edges;
    // std::unordered_set<QSlimUvEdge> or_edges;
    // std::unordered_set<QSlimEdges> left_edges; // edges included removed
    // faces but not shared among faces

    std::map<QSlimUvEdge, std::int32_t> edge_count;

    // Remove a vetex
    if (!valid_vertices[vid]) {
      // If vertex is alreay invalid, do nothing
      // return;
      throw std::exception("");
    }
    valid_vertices[vid] = false;

    // Remove faces connected to the vertex
    for (const auto& fid : v2f[vid]) {
      //std::vector<int32_t> adjacent_fids, adjacent_uv_fids;
      //face_adjacency.GetAdjacentFaces(fid, &adjacent_fids);
      // uv_face_adjacency.GetAdjacentFaces(fid, &adjacent_uv_fids);

      auto [success, face, uv_face] = RemoveFace(fid);

      if (!success) {
        continue;
      }

      const std::array<std::array<int, 2>, 3> order = {
          {{{0, 1}}, {{1, 2}}, {{0, 2}}}};

      //QSlimUvEdge e1(face[0], face[1], uv_face[0], uv_face[1]);
      //QSlimUvEdge e2(face[1], face[2], uv_face[1], uv_face[2]);
      //QSlimUvEdge e3(face[2], face[0], uv_face[2], uv_face[0]);

      for (int i = 0; i < 3; i++) {
        QSlimUvEdge e1(face[order[i][0]], face[order[i][1]],
                       uv_face[order[i][0]], uv_face[order[i][1]]);
        if (edge_count.find(e1) != edge_count.end()) {
          edge_count[e1] += 1;
        } else {
          edge_count[e1] = 1;
        }
      }

      face_ids.insert(fid);
    }

    // Clear v2f
    v2f[vid].clear();

    return {face_ids, edge_count};
  }

  std::pair<int32_t, int32_t> AddVertex(const Eigen::Vector3f& new_pos,
                                        const Eigen::Vector3f& new_normal,
                                        const Eigen::Vector3f& new_color,
                                        const Eigen::Vector2f& new_uv) {
    int vid = -1;
    int uv_vid = -1;

    // Search empty vid
    auto result =
        std::find(valid_vertices.begin(), valid_vertices.end(), false);
    if (result == valid_vertices.end()) {
      return {vid, uv_vid};
    } else {
      vid = static_cast<int32_t>(*result);
      std::vector<int> uv_vids = vid2uvid[vid];

      assert(!uv_vids.empty());

      uv_vid = uv_vids[0];

      valid_vertices[vid] = true;

      vertices[vid] = new_pos;
      normals[vid] = new_normal;
      vertex_colors[vid] = new_color;

      uv[uv_vid] = new_uv;
    }

    return {vid, uv_vid};
  }

  int AddFace(const Eigen::Vector3i& face, const Eigen::Vector3i& uv_face) {
    // Search empty fid
    int fid = -1;
    // Search empty vid
    auto result = std::find(valid_faces.begin(), valid_faces.end(), false);
    if (result == valid_faces.end()) {
      return fid;
    } else {
      fid = static_cast<int32_t>(*result);

      valid_faces[fid] = true;

      vertex_indices[fid] = face;
      uv_indices[fid] = uv_face;
    }


    for (int i = 0; i < 3; i++) {
      v2f[face[i]].push_back(fid);
      uv_v2f[uv_face[i]].push_back(fid);
    }

    return fid;
  }

  bool CollapseEdge(int32_t v1, int32_t v2, const Eigen::Vector3f& org_normal,
                    const Eigen::Vector3f& new_pos,
                    const Eigen::Vector3f& new_normal,
                    const Eigen::Vector3f& new_color,
                    const Eigen::Vector2f& new_uv) {
    std::unordered_set<int32_t> removed_face_ids;
    std::map<QSlimUvEdge, std::int32_t> removed_edge_count;

    // Remove vertices
    auto [fids1, ecount1] = RemoveVertex(v1);
    auto [fids2, ecount2] = RemoveVertex(v2);

    // Count removed edges
    removed_edge_count = std::move(ecount1);
    for (const auto& p : ecount2) {
      const auto& edge = p.first;
      const int32_t count = p.second;
      if (removed_edge_count.find(edge) != removed_edge_count.end()) {
        removed_edge_count[edge] += count;
      } else {
        removed_edge_count[edge] = count;
      }
    }

    // Find count == 1 edges
    // They are used for triangulation
    std::vector<QSlimUvEdge> candidate_edges;
    for (const auto& p : removed_edge_count) {
      if (p.second == 1) {
        candidate_edges.push_back(p.first);
      }
    }

    // Add vertex
    auto [added_vid, added_uv_vid] =
        AddVertex(new_pos, new_normal, new_color, new_uv);

    // Make faces
    for (const auto& e : candidate_edges) {
      bool fliped = false;
      Eigen::Vector3i face{added_vid, e.edge.first, e.edge.second};
      Eigen::Vector3i uv_face{added_uv_vid, e.uv_edge.first, e.uv_edge.second};

      Eigen::Vector3f n =
          (vertices[added_vid] - vertices[e.edge.first])
              .cross(vertices[e.edge.second] - vertices[e.edge.first])
              .normalized();

      // order check
      if (n.dot(org_normal) < 0) {
        fliped = true;
        std::swap(face[0], face[1]);
        std::swap(uv_face[0], uv_face[1]);
      }

      AddFace(face, uv_face);

    }

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

    RemoveVertices(valid_vertices);

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

  int RemoveVertices(const std::vector<bool>& valid_vertex_table) {
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

    int32_t vid1 = decimated_mesh.vertex_indices[fid][0];
    int32_t vid2 = decimated_mesh.vertex_indices[fid][1];

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
