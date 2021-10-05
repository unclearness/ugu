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

#if 0
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

}  // namespace

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

#endif  // 0

namespace {

using QSlimEdge = std::pair<int32_t, int32_t>;

QSlimEdge MakeQSlimEdge(int32_t v0, int32_t v1) {
  if (v0 < v1) {
    return QSlimEdge(v0, v1);
  }

  return QSlimEdge(v1, v0);
}

using VertexAttr = Eigen::VectorXd;
using VertexAttrs = std::vector<VertexAttr>;
using VertexAttrsPtr = std::shared_ptr<VertexAttrs>;

void ConvertVertexAttr2Vectors(const VertexAttr& attr, Eigen::Vector3f& new_pos,
                               Eigen::Vector3f& new_normal,
                               Eigen::Vector3f& new_color,
                               Eigen::Vector2f& new_uv, ugu::QSlimType type) {
  if (type == ugu::QSlimType::XYZ) {
    new_pos[0] = attr[0];
    new_pos[1] = attr[1];
    new_pos[2] = attr[2];

  } else if (type == ugu::QSlimType::XYZ_UV) {
    new_pos[0] = attr[0];
    new_pos[1] = attr[1];
    new_pos[2] = attr[2];

    new_uv[0] = attr[3];
    new_uv[1] = attr[4];
  }
}

void ConvertVertexAttrs2Vectors(
    const VertexAttrsPtr attrs, std::vector<Eigen::Vector3f>& vertices,
    std::vector<Eigen::Vector3f>& normals,
    std::vector<Eigen::Vector3f>& vertex_colors,
    const std::vector<Eigen::Vector2f>& org_uv,
    std::vector<Eigen::Vector2f>& uv,
    const std::vector<std::vector<int32_t>>& vid2uvid, ugu::QSlimType type) {
  uv = org_uv;
  size_t vnum = attrs->size();
  for (size_t i = 0; i < vnum; i++) {
    ConvertVertexAttr2Vectors(attrs->at(i), vertices[i], normals[i],
                              vertex_colors[i], uv[vid2uvid[i][0]], type);
  }
}

void ConvertVectors2VertexAttr(VertexAttr& attr, const Eigen::Vector3f& pos,
                               const Eigen::Vector3f& normal,
                               const Eigen::Vector3f& color,
                               const Eigen::Vector2f& uv, ugu::QSlimType type) {
  if (type == ugu::QSlimType::XYZ) {
    attr.resize(3, 1);
    attr[0] = pos[0];
    attr[1] = pos[1];
    attr[2] = pos[2];

  } else if (type == ugu::QSlimType::XYZ_UV) {
    attr.resize(5, 1);
    attr[0] = pos[0];
    attr[1] = pos[1];
    attr[2] = pos[2];

    attr[3] = uv[0];
    attr[4] = uv[1];
  }
}

void ConvertVectors2VertexAttrs(
    VertexAttrsPtr attrs, const std::vector<Eigen::Vector3f>& vertices,
    const std::vector<Eigen::Vector3f>& normals,
    const std::vector<Eigen::Vector3f>& vertex_colors,
    const std::vector<Eigen::Vector2f>& uv,
    const std::vector<std::vector<int32_t>>& vid2uvid, ugu::QSlimType type) {
  size_t vnum = attrs->size();
  for (size_t i = 0; i < vnum; i++) {
    ConvertVectors2VertexAttr(attrs->at(i), vertices[i], normals[i],
                              vertex_colors[i], uv[vid2uvid[i][0]], type);
  }
}

using Quadric = Eigen::MatrixXd;
using Quadrics = std::vector<Quadric>;
using QuadricPtr = std::shared_ptr<Quadric>;
using QuadricsPtr = std::shared_ptr<Quadrics>;

bool ComputeOptimalConstraction(const VertexAttr& v1, const Quadric& q1,
                                const VertexAttr& v2, const Quadric& q2,
                                VertexAttr& v, double& error) {
  auto org_size = v1.rows();
  //VertexAttr zero(org_size  + 1);
  //zero.setZero();
  //zero[zero.size() - 1] = 1.0;

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
      candidates[i].conservativeResize(org_size + 1, 1);
      candidates[i][org_size] = 1.0;
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
#if 0
				    Quadric q_inv = q.inverse();
    v = q_inv * zero;
    error = v.transpose() * q * v;
    v.conservativeResize(org_size, 1);
#endif  // 0

    const Eigen::MatrixXd& A = q.topLeftCorner(org_size, org_size);
    const VertexAttr& b = q.topRightCorner(org_size, 1);

    v = -A.inverse() * b;

    error = b.transpose() * v + q(org_size, org_size);
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
      ComputeOptimalConstraction(vert_attrs->at(v0), quadrics->at(v0),
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

#if 1
// To handle topoloy
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
  std::unordered_set<int> unified_boundary_uv_ids;

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

      auto [uv_boundary_edges, uv_boundary_vertex_ids] =
          uv_face_adjacency.GetBoundaryEdges();
      unified_boundary_uv_ids = std::move(uv_boundary_vertex_ids);
    }
  }

  int32_t VertexNum() const {
    return std::count(valid_vertices.begin(), valid_vertices.end(), true);
  }

  int32_t FaceNum() const {
    return std::count(valid_faces.begin(), valid_faces.end(), true);
  }

  std::set<QSlimEdge> ConnectingEdges(int32_t vid) {
    std::set<QSlimEdge> edges;

    for (const auto& f : v2f[vid]) {
      if (vertex_indices[f][0] == vid) {
        edges.emplace(MakeQSlimEdge(vid, vertex_indices[f][1]));
        edges.emplace(MakeQSlimEdge(vid, vertex_indices[f][2]));

      } else if (vertex_indices[f][1] == vid) {
        edges.emplace(MakeQSlimEdge(vid, vertex_indices[f][0]));
        edges.emplace(MakeQSlimEdge(vid, vertex_indices[f][2]));
      } else if (vertex_indices[f][2] == vid) {
        edges.emplace(MakeQSlimEdge(vid, vertex_indices[f][0]));
        edges.emplace(MakeQSlimEdge(vid, vertex_indices[f][1]));
      } else {
        throw std::exception("something wrong");
      }
    }

    return edges;
  }

  auto FacesContactingEdges(int32_t v0, int32_t v1) {
    const auto& v0_f = v2f[v0];
    std::unordered_set<int32_t> union_fids(v0_f.begin(), v0_f.end());
    const auto& v1_f = v2f[v1];
    for (const auto& fid : v1_f) {
      union_fids.insert(fid);
    }

    std::unordered_set<int32_t> no_intersection = union_fids;
    std::unordered_set<int32_t> intersection;
    std::set_intersection(v0_f.begin(), v0_f.end(), v1_f.begin(), v1_f.end(),
                          std::inserter(intersection, intersection.end()));

    // For non-manifold meshes, always 2
    assert(intersection.size() == 2);
    if (intersection.size() != 2) {
        throw std::exception("something wrong");
    }

    for (const auto& fid : intersection) {
      no_intersection.erase(fid);
    }

    return std::make_tuple(union_fids, intersection, no_intersection);
  }

  void RemoveFace(int32_t fid) {
    if (!valid_faces[fid]) {
      throw std::exception("Something wrong");
    }

    // Remove from v2f
    for (int32_t i = 0; i < 3; i++) {
      int32_t vid = vertex_indices[fid][i];
      assert(valid_vertices[vid]);
      auto result = std::remove(v2f[vid].begin(), v2f[vid].end(), fid);
      v2f[vid].erase(result, v2f[vid].end());
    }

    valid_faces[fid] = false;
    vertex_indices[fid].setConstant(99999);
  }

  void RemoveVertex(int32_t vid) {
    // Remove a vetex
    if (!valid_vertices[vid]) {
      // If vertex is alreay invalid, something wrong
      throw std::exception("something wrong");
    }
    valid_vertices[vid] = false;
    vertices[vid].setConstant(99999);

    // Clear v2f
    v2f[vid].clear();

    // Remove from boundary
    // unified_boundary_vertex_ids.erase(vid);
  }

  bool CollapseEdge(int32_t v1, int32_t v2, bool update_vertex = false,
                    const Eigen::Vector3f& new_pos = Eigen::Vector3f::Zero(),
                    const Eigen::Vector3f& new_normal = Eigen::Vector3f::Zero(),
                    const Eigen::Vector3f& new_color = Eigen::Vector3f::Zero(),
                    const Eigen::Vector2f& new_uv = Eigen::Vector2f::Zero()) {
    // std::cout << "decimate " << v1 << " " << v2 << std::endl;

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
    if (update_vertex) {
      vertices[v1] = new_pos;
      normals[v1] = new_normal;
      vertex_colors[v1] = new_color;
      if (use_uv) {
        uv[vid2uvid[v1][0]] = new_uv;
      }
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

  void Finalize(VertexAttrsPtr vert_attrs, ugu::QSlimType type) {
    ConvertVertexAttrs2Vectors(vert_attrs, vertices, normals, vertex_colors,
                               mesh->uv(), uv, vid2uvid, type);
    Finalize();
  }

  void Finalize() {
    // Appy valid mask
    auto material_ids = mesh->material_ids();

    mesh->Clear();

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

  DecimatedMesh() {}
  ~DecimatedMesh() {}
};
#endif  // 1

struct QSlimHandler {
  QuadricsPtr quadrics;
  VertexAttrsPtr vert_attrs;
  ugu::QSlimType type;

  Quadric ComputeInitialQuadric(int32_t vid0, int32_t vid1, int32_t vid2) {
    // Same notation to the paper
    const VertexAttr& p = vert_attrs->at(vid0);
    const VertexAttr& q = vert_attrs->at(vid1);
    const VertexAttr& r = vert_attrs->at(vid2);

    VertexAttr e1 = (q - p).normalized();
    VertexAttr e2 = (r - p - e1.dot(r - p) * e1).normalized();

    const Eigen::Index A_size = vert_attrs->at(vid0).rows();
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(A_size, A_size) -
                        e1 * e1.transpose() - e2 * e2.transpose();
    const VertexAttr b = p.dot(e1) * e1 + p.dot(e2) * e2 - p;
    const double pe1 = p.dot(e1);
    const double pe2 = p.dot(e2);
    const double c = p.dot(p) - pe1 * pe1 - pe2 * pe2;

    const Eigen::Index Q_size = A_size + 1;
    Quadric Q = Eigen::MatrixXd::Zero(Q_size, Q_size);
    Q.topLeftCorner(A_size, A_size) = A;
    Q.topRightCorner(A_size, 1) = b;
    Q.bottomLeftCorner(1, A_size) = b.transpose();
    Q(A_size, A_size) = c;

    return Q;
  }

  void ComputeInitialQuadric(int32_t vid,
                             const std::vector<int32_t>& neighbor_fids,
                             const std::vector<Eigen::Vector3i>& faces) {
    for (const auto& fid : neighbor_fids) {
      const auto face = faces[fid];
      std::vector<int32_t> dst_vids;

      for (int32_t i = 0; i < 3; i++) {
        if (face[i] != vid) {
          dst_vids.push_back(face[i]);
        }
      }
      // Summation for all faces (planes) connecting to a vertex
      quadrics->at(vid) += ComputeInitialQuadric(vid, dst_vids[0], dst_vids[1]);
    }
  }

  void InitializeQuadrics(const DecimatedMesh& mesh,
                          const std::unordered_map<int, std::vector<int>>& v2f,
                          const std::vector<std::vector<int32_t>>& vid2uvid,
                          ugu::QSlimType type) {
    this->type = type;
    quadrics = std::make_shared<Quadrics>();
    vert_attrs = std::make_shared<VertexAttrs>();

    vert_attrs->resize(mesh.vertices.size());
    quadrics->resize(mesh.vertices.size());

    ConvertVectors2VertexAttrs(vert_attrs, mesh.vertices, mesh.normals,
                               mesh.vertex_colors, mesh.uv, vid2uvid, type);

    const Eigen::Index qsize = vert_attrs->at(0).rows() + 1;
    for (auto& q : *quadrics) {
      q.resize(qsize, qsize);
      q.setZero();
    }

    for (size_t i = 0; i < mesh.vertices.size(); i++) {
      ComputeInitialQuadric(i, v2f.at(i), mesh.vertex_indices);
    }
  }
};

std::set<QSlimEdge> PrepareValidEdges(
    const std::vector<Eigen::Vector3i>& faces,
    const std::unordered_set<int>& unified_boundary_vertex_ids,
    const std::unordered_set<int>& unified_boundary_uv_ids,
    const std::vector<int32_t>& uvid2vid, bool keep_geom_boundary,
    bool keep_uv_boundary) {
  std::set<QSlimEdge> valid_edges;
  std::unordered_set<int> invalid_vids;

  if (keep_geom_boundary) {
    invalid_vids = unified_boundary_vertex_ids;
  }

  if (keep_uv_boundary) {
    for (const auto& uv_vid : unified_boundary_uv_ids) {
      invalid_vids.insert(uvid2vid.at(uv_vid));
    }
  }

  for (const auto& f : faces) {
    size_t v0c = invalid_vids.count(f[0]);
    size_t v1c = invalid_vids.count(f[1]);
    size_t v2c = invalid_vids.count(f[2]);

    // Pair keeps always (smaller, bigger)
    // Is this okay for geometry face reconstruction?

    if (v0c == 0 && v1c == 0) {
      int32_t v0 = std::min(f[0], f[1]);
      int32_t v1 = std::max(f[0], f[1]);
      valid_edges.insert(std::make_pair(v0, v1));
    }

    if (v1c == 0 && v2c == 0) {
      int32_t v1 = std::min(f[1], f[2]);
      int32_t v2 = std::max(f[1], f[2]);
      valid_edges.insert(std::make_pair(v1, v2));
    }

    if (v2c == 0 && v0c == 0) {
      int32_t v2 = std::min(f[0], f[2]);
      int32_t v0 = std::max(f[0], f[2]);
      valid_edges.insert(std::make_pair(v2, v0));
    }
  }
  return valid_edges;
}

std::vector<QSlimEdgeInfo> CollapseEdgeAndUpdateQuadrics(DecimatedMesh& mesh,
                                                         QSlimHandler& handler,
                                                         QSlimEdgeInfo& e,
                                                         ugu::QSlimType type) {
  int32_t v1 = e.edge.first;
  int32_t v2 = e.edge.second;

  // Update topology
  mesh.CollapseEdge(v1, v2);

  // Update vertex attributes and quadrics
  handler.quadrics->at(v1) =
      handler.quadrics->at(v1) + handler.quadrics->at(v2);
  handler.vert_attrs->at(v1) = e.decimated_v;

  // Construct edges to add heap
  std::vector<QSlimEdgeInfo> new_edges;
  auto raw_edges = mesh.ConnectingEdges(v1);
  bool keep_this_edge = true;
  for (const auto& e : raw_edges) {
    auto new_edge =
        QSlimEdgeInfo(e, handler.vert_attrs, handler.quadrics, keep_this_edge);
    new_edges.emplace_back(new_edge);
  }

  return new_edges;
}

}  // namespace

namespace ugu {

bool RandomDecimation(MeshPtr mesh, QSlimType type, int32_t target_face_num,
                      int32_t target_vertex_num, bool keep_geom_boundary,
                      bool keep_uv_boundary) {
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

    decimated_mesh.CollapseEdge(vid1, vid2, true, new_pos, new_normal,
                                new_color, new_uv);

    std::cout << decimated_mesh.FaceNum() << " " << decimated_mesh.VertexNum()
              << std::endl;
  }
#endif

  decimated_mesh.Finalize();

  return true;
}

bool QSlim(MeshPtr mesh, QSlimType type, int32_t target_face_num,
           int32_t target_vertex_num, bool keep_geom_boundary,
           bool keep_uv_boundary) {
  // Set up valid edges (we do not consider non-edges yet)
  DecimatedMesh decimated_mesh(mesh);
  auto valid_edges = PrepareValidEdges(
      decimated_mesh.vertex_indices, decimated_mesh.unified_boundary_vertex_ids,
      decimated_mesh.unified_boundary_uv_ids, decimated_mesh.uvid2vid,
      keep_geom_boundary, keep_uv_boundary);

  // Initialize quadrics
  QSlimHandler handler;
  handler.InitializeQuadrics(decimated_mesh, decimated_mesh.v2f,
                             decimated_mesh.vid2uvid, type);

  // Initialize heap
  QSlimHeap heap;

  // Add the valid pairs  to heap
  for (const auto& p : valid_edges) {
    QSlimEdgeInfo info(p, handler.vert_attrs, handler.quadrics, false);
    heap.push(info);
  }

  std::unordered_set<int32_t> removed_vids;

  // Main loop
  while (!heap.empty()) {
    if (target_vertex_num > 0 &&
        target_vertex_num >= decimated_mesh.VertexNum()) {
      break;
    }

    if (target_face_num > 0 && target_face_num >= decimated_mesh.FaceNum()) {
      break;
    }

    // Find the lowest error pair
    auto min_e = heap.top();
    heap.pop();

    // Skip if it has been decimated
    if (removed_vids.count(min_e.edge.first) > 0 ||
        removed_vids.count(min_e.edge.second) > 0) {
      continue;
    }

    std::cout << "decimate " << min_e.edge.first << ", " << min_e.edge.second
             << std::endl;

    // Decimate the pair
    auto new_edges =
        CollapseEdgeAndUpdateQuadrics(decimated_mesh, handler, min_e, type);

    // Memory removed vid
    removed_vids.insert(min_e.edge.second);

    // Add new pairs
    for (const auto& e : new_edges) {
      heap.push(e);
    }

    std::cout << decimated_mesh.FaceNum() << " " <<
    decimated_mesh.VertexNum()
              << std::endl;
  }

  decimated_mesh.Finalize(handler.vert_attrs, type);

  return true;
}

}  // namespace ugu
