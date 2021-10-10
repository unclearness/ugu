/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 *
 */

#include "ugu/decimation/decimation.h"

#include <iostream>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_set>

#include "ugu/face_adjacency.h"
#include "ugu/util/geom_util.h"
#include "ugu/util/math_util.h"
#include "ugu/util/raster_util.h"

namespace {

using QSlimEdge = std::tuple<int32_t, int32_t, int32_t>;

QSlimEdge MakeQSlimEdge(int32_t v0, int32_t v1, int32_t fid) {
  if (v0 < v1) {
    return QSlimEdge(v0, v1, fid);
  }

  return QSlimEdge(v1, v0, fid);
}

using VertexAttr = Eigen::VectorXd;
using VertexAttrList = std::vector<VertexAttr>;
using VertexAttrListPtr = std::shared_ptr<VertexAttrList>;

struct UniqueVertex {
  int32_t vid = -1;
  int32_t uvid = -1;
  int32_t fid = -1;
  UniqueVertex() {}
  ~UniqueVertex() {}

  UniqueVertex(int32_t vid, int32_t uvid, int32_t fid)
      : vid(vid), uvid(uvid), fid(fid) {}

  bool operator<(const UniqueVertex& obj) const {
    if (vid != obj.vid) {
      return vid < obj.vid;
    }

    if (uvid != obj.uvid) {
      return uvid < obj.uvid;
    }

    return fid < obj.fid;
  }

  bool operator==(const UniqueVertex& obj) const {
    return vid == obj.vid && uvid == obj.uvid && fid == obj.fid;
  }
};

std::pair<std::vector<UniqueVertex>,
          std::unordered_map<int32_t, std::vector<int32_t>>>
GenerateUniqueVertices(const std::vector<Eigen::Vector3i>& vertex_indices,
                       const std::vector<Eigen::Vector3i>& uv_indices) {
  std::vector<UniqueVertex> unique_vertices;
  std::unordered_map<int32_t, std::vector<int32_t>> vid2unique;

  for (size_t i = 0; i < vertex_indices.size(); i++) {
    for (int32_t j = 0; j < 3; j++) {
      unique_vertices.push_back(UniqueVertex(
          vertex_indices[i][j], uv_indices[i][j], static_cast<int32_t>(i)));
    }
  }

  std::sort(unique_vertices.begin(), unique_vertices.end());
  auto res = std::unique(unique_vertices.begin(), unique_vertices.end());
  unique_vertices.erase(res, unique_vertices.end());
  for (size_t i = 0; i < unique_vertices.size(); i++) {
    const auto& u = unique_vertices[i];
    vid2unique[u.vid].push_back(static_cast<int32_t>(i));
  }

  for (auto& p : vid2unique) {
    std::sort(p.second.begin(), p.second.end());
    auto res = std::unique(p.second.begin(), p.second.end());
    p.second.erase(res, p.second.end());
  }


  return {unique_vertices, vid2unique};
}

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
    const VertexAttrListPtr attrs,
    const std::vector<UniqueVertex>& unique_vertices,
    std::vector<Eigen::Vector3f>& vertices,
    std::vector<Eigen::Vector3f>& normals,
    std::vector<Eigen::Vector3f>& vertex_colors,
    std::vector<Eigen::Vector2f>& uv, ugu::QSlimType type) {
  size_t vnum = attrs->size();
  for (size_t i = 0; i < vnum; i++) {
    auto unique_v = unique_vertices[i];
    auto vid = unique_v.vid;
    auto uvid = unique_v.uvid;
    ConvertVertexAttr2Vectors(attrs->at(i), vertices[vid], normals[vid],
                              vertex_colors[vid], uv[uvid], type);
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
    VertexAttrListPtr attrs, const std::vector<UniqueVertex>& unique_vertices,
    const std::vector<Eigen::Vector3f>& vertices,
    const std::vector<Eigen::Vector3f>& normals,
    const std::vector<Eigen::Vector3f>& vertex_colors,
    const std::vector<Eigen::Vector2f>& uv, ugu::QSlimType type) {
  size_t vnum = attrs->size();
  for (size_t i = 0; i < vnum; i++) {
    auto unique_v = unique_vertices[i];
    auto vid = unique_v.vid;
    auto uvid = unique_v.uvid;
    ConvertVectors2VertexAttr(attrs->at(i), vertices[vid], normals[vid],
                              vertex_colors[vid], uv[uvid], type);
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
  // VertexAttr zero(org_size  + 1);
  // zero.setZero();
  // zero[zero.size() - 1] = 1.0;

  bool ret = true;

  const Quadric q = q1 + q2;
  const Eigen::MatrixXd& A = q.topLeftCorner(org_size, org_size);
  const Eigen::VectorXd& b = q.topRightCorner(org_size, 1);
  const double c = q(org_size, org_size);
  if (std::abs(q.determinant()) < 0.00001) {
    // Not ivertible case
    ret = false;

    // Select best one from v1, v2 and (v1+v2)/2
    std::array<VertexAttr, 3> candidates = {v1, v2, (v1 + v2) * 0.5};
    double min_error = std::numeric_limits<double>::max();
    VertexAttr min_vert = v1;

    for (int i = 0; i < 3; i++) {
      double vav = candidates[i].transpose() * A * candidates[i];
      double tmp_error = vav + 2.0 * b.transpose() * candidates[i] + c;
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
    v = -A.inverse() * b;
    error = b.transpose() * v + c;
  }

  if (error < 0) {
    assert(std::abs(error) < 0.00001);
    error = 0;
  }

  return ret;
}

struct QSlimEdgeInfo {
  QSlimEdge edge = {-1, -1, -1};
  // QSlimEdge edge_uv = {-1, -1};
  // int org_vid = -1;
  // QSlimEdge org_edge = {-1, -1};
  double error = std::numeric_limits<double>::max();
  QuadricsPtr quadrics;
  VertexAttrListPtr vert_attrs;
  VertexAttr decimated_v;
  bool keep_this_edge = false;
  QSlimEdgeInfo(QSlimEdge edge_, VertexAttrListPtr vert_attrs_,
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
      int v0 = std::get<0>(edge);
      int v1 = std::get<1>(edge);
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

  std::vector<bool> valid_vertices, valid_faces;
  std::vector<bool> valid_uvs, valid_uv_faces;

  std::vector<std::vector<int32_t>> vid2uvid;
  std::vector<int32_t> uvid2vid;
  std::unordered_map<int, std::vector<int>> v2f, uv_v2f;

  // std::set<QSlimEdge> valid_pairs;
  // std::unordered_set<int32_t> invalid_vids;

  ugu::FaceAdjacency face_adjacency, uv_face_adjacency;

  std::unordered_set<int> unified_boundary_vertex_ids;
  std::unordered_set<int> unified_boundary_uv_ids;

  std::unordered_set<int32_t> ignore_vids, ignore_fids, ignore_uv_ids;

  bool use_uv = false;

  DecimatedMesh(ugu::MeshPtr mesh) : mesh(mesh) {
    valid_vertices.resize(mesh->vertices().size(), true);
    valid_faces.resize(mesh->vertex_indices().size(), true);

    vertices = mesh->vertices();
    vertex_colors = mesh->vertex_colors();
    vertex_indices = mesh->vertex_indices();

    if (mesh->normals().size() != vertices.size()) {
      mesh->CalcNormal();
    }

    normals = mesh->normals();

    if (vertex_colors.empty()) {
      vertex_colors.resize(vertices.size());
    }

    valid_uvs.resize(mesh->uv().size(), true);
    valid_uv_faces.resize(mesh->uv_indices().size(), true);
    uv = mesh->uv();
    uv_indices = mesh->uv_indices();

    use_uv = !uv.empty() && !uv_indices.empty() &&
             uv_indices.size() == vertex_indices.size();

    face_adjacency.Init(mesh->vertices().size(), mesh->vertex_indices());
    v2f = ugu::GenerateVertex2FaceMap(mesh->vertex_indices(),
                                      mesh->vertices().size());

    auto [boundary_edges_list, boundary_vertex_ids_list] =
        ugu::FindBoundaryLoops(mesh->vertex_indices(),
                               static_cast<int32_t>(mesh->vertices().size()));

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
      for (size_t i = 0; i < vertices.size(); i++) {
        std::sort(vid2uvid[i].begin(), vid2uvid[i].end());
        auto res = std::unique(vid2uvid[i].begin(), vid2uvid[i].end());
        vid2uvid[i].erase(res, vid2uvid[i].end());
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
    return std::count(valid_vertices.begin(), valid_vertices.end(), true) +
           ignore_vids.size();
  }

  int32_t FaceNum() const {
    return std::count(valid_faces.begin(), valid_faces.end(), true) +
           ignore_fids.size();
  }

  std::set<QSlimEdge> ConnectingEdges(int32_t vid) {
    std::set<QSlimEdge> edges;

    for (const auto& f : v2f[vid]) {
      bool ignore_v0 = ignore_vids.count(vertex_indices[f][0]) > 0;
      bool ignore_v1 = ignore_vids.count(vertex_indices[f][1]) > 0;
      bool ignore_v2 = ignore_vids.count(vertex_indices[f][2]) > 0;

      if (vertex_indices[f][0] == vid) {
        if (!ignore_v1) {
          edges.emplace(MakeQSlimEdge(vid, vertex_indices[f][1], f));
        }
        if (!ignore_v2) {
          edges.emplace(MakeQSlimEdge(vid, vertex_indices[f][2], f));
        }
      } else if (vertex_indices[f][1] == vid) {
        if (!ignore_v0) {
          edges.emplace(MakeQSlimEdge(vid, vertex_indices[f][0], f));
        }
        if (!ignore_v2) {
          edges.emplace(MakeQSlimEdge(vid, vertex_indices[f][2], f));
        }
      } else if (vertex_indices[f][2] == vid) {
        if (!ignore_v0) {
          edges.emplace(MakeQSlimEdge(vid, vertex_indices[f][0], f));
        }
        if (!ignore_v1) {
          edges.emplace(MakeQSlimEdge(vid, vertex_indices[f][1], f));
        }
      } else {
        std::cout << vertex_indices[f] << std::endl; 
        throw std::runtime_error("something wrong");
      }
    }

    return edges;
  }

  auto FacesContactingEdges(int32_t v0, int32_t v1) {
    //const auto& v0_f = uv_v2f[v0];
    //const auto& v1_f = uv_v2f[v1];
    const auto& v0_f = v2f[v0];
    const auto& v1_f = v2f[v1];
    std::unordered_set<int32_t> union_fids(v0_f.begin(), v0_f.end());

    for (const auto& fid : v1_f) {
      union_fids.insert(fid);
    }

    std::unordered_set<int32_t> no_intersection = union_fids;
    std::unordered_set<int32_t> intersection;
    std::set_intersection(v0_f.begin(), v0_f.end(), v1_f.begin(), v1_f.end(),
                          std::inserter(intersection, intersection.end()));

#if 0
    // For non-manifold meshes, always 2
    assert(intersection.size() == 2);
    if (intersection.size() != 2) {
      for (auto f : v0_f) {
        std::cout << vertex_indices[f] << std::endl;
        std::cout << valid_faces[f] << std::endl;
      }
      for (auto f : v1_f) {
        std::cout << vertex_indices[f] << std::endl;
        std::cout << valid_faces[f] << std::endl;
      }

      throw std::runtime_error("something wrong");
    }
#endif  // 0

    for (const auto& fid : intersection) {
      no_intersection.erase(fid);
    }

    return std::make_tuple(union_fids, intersection, no_intersection);
  }

  void RemoveFace(int32_t fid) {
    if (!valid_faces[fid]) {
      throw std::runtime_error("Something wrong");
    }

    // Remove from v2f
    for (int32_t i = 0; i < 3; i++) {
      int32_t vid = vertex_indices[fid][i];
      if (!valid_vertices[vid]) {
        assert(ignore_vids.count(vid) > 0);
      }
      auto result = std::remove(v2f[vid].begin(), v2f[vid].end(), fid);
      v2f[vid].erase(result, v2f[vid].end());
    }

    for (int32_t i = 0; i < 3; i++) {
      int32_t uvid = uv_indices[fid][i];
      if (!valid_uvs[uvid]) {
        //assert(ignore_vids.count(vid) > 0);
      }
      auto result = std::remove(uv_v2f[uvid].begin(), uv_v2f[uvid].end(), fid);
      uv_v2f[uvid].erase(result, uv_v2f[uvid].end());
    }


    valid_faces[fid] = false;
    vertex_indices[fid].setConstant(99999);

    valid_uv_faces[fid] = false;
    uv_indices[fid].setConstant(999999);
  }

  void RemoveVertex(int32_t vid) {
    // Remove a vetex
    if (!valid_vertices[vid]) {
      // If vertex is alreay invalid, something wrong
      throw std::runtime_error("something wrong");
    }
    valid_vertices[vid] = false;
    vertices[vid].setConstant(99999);

    // Clear v2f
    v2f[vid].clear();

    // Remove from boundary
    // unified_boundary_vertex_ids.erase(vid);
  }

  void RemoveUv(int32_t uvid) {
    // Remove a vetex
    if (!valid_uvs[uvid]) {
      // If vertex is alreay invalid, something wrong
      throw std::runtime_error("something wrong");
    }
    valid_uvs[uvid] = false;
    uv[uvid].setConstant(99999);


    // Clear v2f
    uv_v2f[uvid].clear();

    // Remove from boundary
    // unified_boundary_vertex_ids.erase(vid);
  }

#if 0
				
  bool PrepareCollapseEdge(int32_t v1, int32_t v2, const VertexAttr& vertattr,
                           ugu::QSlimType type) {
    // Get faces connecting the 2 vertices (A, B)
    auto [union_fids, intersection, no_intersection] =
        FacesContactingEdges(v1, v2);

    std::unordered_set<int32_t> to_remove_face_ids = std::move(intersection);
    std::unordered_set<int32_t> to_keep_face_ids = std::move(no_intersection);

    assert(to_remove_face_ids.size() == 2);

    // Vertex attribute generation
    Eigen::Vector3f new_pos, new_color, new_normal;
    Eigen::Vector2f new_uv;
    int32_t interp_fid = *to_remove_face_ids.begin();
    int32_t vid0 = vertex_indices[interp_fid][0];
    int32_t vid1 = vertex_indices[interp_fid][1];
    int32_t vid2 = vertex_indices[interp_fid][2];
    int32_t v1_index_pos = -1;
    if (vid0 == v1) {
      v1_index_pos = 0;
    } else if (vid1 == v1) {
      v1_index_pos = 1;
    } else if (vid2 == v1) {
      v1_index_pos = 2;
    }

    int32_t v2_index_pos = -1;
    if (vid0 == v2) {
      v2_index_pos = 0;
    } else if (vid1 == v2) {
      v2_index_pos = 1;
    } else if (vid2 == v2) {
      v2_index_pos = 2;
    }

    assert(v1_index_pos >= 0);
    assert(v2_index_pos >= 0);

    ConvertVertexAttr2Vectors(vertattr, new_pos, new_normal, new_color, new_uv,
                              type);
  }

  bool ValidateBeforeCollapseEdge(int32_t v1, int32_t v2, const VertexAttr& vertattr,
                    ugu::QSlimType type) {


    // Validation before collapse
    // Ensure normal is not flipped
    for (const auto& fid : to_keep_face_ids) {
      int32_t vid0 = vertex_indices[fid][0];
      int32_t vid1 = vertex_indices[fid][1];
      int32_t vid2 = vertex_indices[fid][2];
      // Replace v2
      // This is a test before real replacing
      if (vid0 == v2) {
        vid0 = v1;
      }
      if (vid1 == v2) {
        vid1 = v1;
      }
      if (vid2 == v2) {
        vid2 = v1;
      }

      Eigen::Vector3f vpos0 = vertices[vid0];
      Eigen::Vector3f vpos1 = vertices[vid1];
      Eigen::Vector3f vpos2 = vertices[vid2];

      if (v1 == vid0) {
        vpos0 = new_pos;
      }

      if (v1 == vid1) {
        vpos1 = new_pos;
      }

      if (v1 == vid2) {
        vpos2 = new_pos;
      }

      Eigen::Vector3f v10 = (vpos1 - vpos0).normalized();
      Eigen::Vector3f v20 = (vpos2 - vpos0).normalized();

      // Skip if it becomes one line
      if (std::abs(v10.dot(v20)) > 0.999) {
        return false;
      }
      // Skip if normal is fiipped
      Eigen::Vector3f face_n = v10.cross(v20).normalized();
      if (face_n.dot(normals[v1]) < 0.0) {
        return false;
      }
    }

    if (use_uv) {
      int32_t uv1 = uv_indices[interp_fid][v1_index_pos];
      int32_t uv2 = uv_indices[interp_fid][v2_index_pos];

      for (const auto& fid : to_keep_face_ids) {
        int32_t uvid0 = uv_indices[fid][0];
        int32_t uvid1 = uv_indices[fid][1];
        int32_t uvid2 = uv_indices[fid][2];
        Eigen::Vector2f uvpos0 = uv[uvid0];
        Eigen::Vector2f uvpos1 = uv[uvid1];
        Eigen::Vector2f uvpos2 = uv[uvid2];
        float area = ugu::EdgeFunction(uvpos0, uvpos1, uvpos2);
#if 0
				        if (v1_index_pos == 0) {
          uvpos0 = new_uv;
        }

        else if (v1_index_pos == 1) {
          uvpos1 = new_uv;
        }

        else if (v1_index_pos == 2) {
          uvpos2 = new_uv;
        }
#endif  // 0

        if (uvid0 == uv1 || uvid0 == uv2) {
          uvpos0 = new_uv;
        } else if (uvid1 == uv1 || uvid1 == uv2) {
          uvpos1 = new_uv;
        } else if (uvid2 == uv1 || uvid2 == uv2) {
          uvpos2 = new_uv;
        } else {
          throw std::runtime_error("");
        }

        float area_update = ugu::EdgeFunction(uvpos0, uvpos1, uvpos2);
        if (area * area_update < 0) {
          return false;
        }
      }
    }
}
#endif  // 0

  bool CollapseEdge(int32_t v1, int32_t uv1, int32_t v2, int32_t uv2,
             const VertexAttr& vertattr,
                    ugu::QSlimType type) {
    //std::cout << "decimate " << v1 << " " << v2 << " " << std::endl;


    // Get faces connecting the 2 vertices (A, B)
    auto [union_fids, intersection, no_intersection] =
        FacesContactingEdges(v1, v2);

    std::unordered_set<int32_t> to_remove_face_ids = std::move(intersection);
    std::unordered_set<int32_t> to_keep_face_ids = std::move(no_intersection);

    // Vertex attribute generation
    Eigen::Vector3f new_pos, new_color, new_normal;
    Eigen::Vector2f new_uv;
    ConvertVertexAttr2Vectors(vertattr, new_pos, new_normal, new_color, new_uv,
                              type);
#if 0
				
    std::cout << "to_keep_face_ids" << std::endl;
    for (auto id : to_keep_face_ids) {
      std::cout << id << std::endl;
    }
    std::cout << std::endl;


  std::cout << "to_remove_face_ids" << std::endl;
    for (auto id : to_remove_face_ids) {
      std::cout << id << std::endl;
    }
    std::cout << std::endl;
#endif  // 0


    // assert(to_remove_face_ids.size() == 2);

   // assert(to_remove_face_ids.count(interp_fid) > 0);


#if 1
    int32_t interp_fid = -1;  // *to_remove_face_ids.begin();
    int32_t v1_index_pos = -1;
    bool uv_suc = false;
    for (int32_t interp_fid_ : to_remove_face_ids) {
    int32_t vid0 = vertex_indices[interp_fid_][0];
    int32_t vid1 = vertex_indices[interp_fid_][1];
    int32_t vid2 = vertex_indices[interp_fid_][2];
    int32_t v1_index_pos_ = -1;
    if (vid0 == v1) {
      v1_index_pos_ = 0;
    } else if (vid1 == v1) {
      v1_index_pos_ = 1;
    } else if (vid2 == v1) {
      v1_index_pos_ = 2;
    }

    int32_t v2_index_pos = -1;
    if (vid0 == v2) {
      v2_index_pos = 0;
    } else if (vid1 == v2) {
      v2_index_pos = 1;
    } else if (vid2 == v2) {
      v2_index_pos = 2;
    }

    assert(v1_index_pos_ >= 0);
    assert(v2_index_pos >= 0);

#if 0
    // auto [u, v, w] = ugu::Barycentric(new_pos, vertices[vid0],
    // vertices[vid1],
    //                                  vertices[vid2]);

    float area = ugu::TriArea(vertices[vid0], vertices[vid1], vertices[vid2]);
    if (std::abs(area) < std::numeric_limits<float>::min()) {
      area = area > 0 ? std::numeric_limits<float>::min()
                      : -std::numeric_limits<float>::min();
    }
    float inv_area = 1.0f / area;

    float u = ugu::TriArea(vertices[vid1], vertices[vid2], new_pos);
    float v = ugu::TriArea(vertices[vid2], vertices[vid0], new_pos);
    float w = ugu::TriArea(vertices[vid0], vertices[vid1], new_pos);
    // Barycentric in the target triangle
    u *= inv_area;
    v *= inv_area;
    w *= inv_area;

    new_color = u * vertex_colors[vid0] + v * vertex_colors[vid1] +
                w * vertex_colors[vid2];
    new_normal = (u * normals[vid0] + v * normals[vid1] + w * normals[vid2])
                     .normalized();
    if (use_uv) {
      int32_t uvid0 = uv_indices[interp_fid][0];
      int32_t uvid1 = uv_indices[interp_fid][1];
      int32_t uvid2 = uv_indices[interp_fid][2];

      new_uv = u * uv[uvid0] + v * uv[uvid1] + w * uv[uvid2];
    }
#endif  // 0

    // Validation before collapse
    // Ensure normal is not flipped
    for (const auto& fid : to_keep_face_ids) {
      int32_t vid0 = vertex_indices[fid][0];
      int32_t vid1 = vertex_indices[fid][1];
      int32_t vid2 = vertex_indices[fid][2];
      // Replace v2
      // This is a test before real replacing
#if 1
      if (vid0 == v2) {
        vid0 = v1;
      } else if (vid1 == v2) {
        vid1 = v1;
      } else if (vid2 == v2) {
        vid2 = v1;
      } else {
        // throw std::runtime_error("");
      }
#endif  // 0

      Eigen::Vector3f vpos0 = vertices[vid0];
      Eigen::Vector3f vpos1 = vertices[vid1];
      Eigen::Vector3f vpos2 = vertices[vid2];

      if (v1 == vid0) {
        vpos0 = new_pos;
      } else if (v1 == vid1) {
        vpos1 = new_pos;
      } else if (v1 == vid2) {
        vpos2 = new_pos;
      } else {
        throw std::runtime_error("");
      }

      Eigen::Vector3f v10 = (vpos1 - vpos0).normalized();
      Eigen::Vector3f v20 = (vpos2 - vpos0).normalized();

      // Skip if it becomes one line
      if (std::abs(v10.dot(v20)) > 0.99999) {
        return false;
      }
      // Skip if normal is fiipped
      Eigen::Vector3f face_n = v10.cross(v20).normalized();
      if (face_n.dot(normals[v1]) < 0.0) {
        // Why is this condition not required?
        // return false;
      }
    }

    if (use_uv) {
      bool uv_suc_ = true;
      for (const auto& fid : to_keep_face_ids) {
        int32_t uvid0 = uv_indices[fid][0];
        int32_t uvid1 = uv_indices[fid][1];
        int32_t uvid2 = uv_indices[fid][2];
        Eigen::Vector2f uvpos0 = uv[uvid0];
        Eigen::Vector2f uvpos1 = uv[uvid1];
        Eigen::Vector2f uvpos2 = uv[uvid2];
        float area = ugu::EdgeFunction(uvpos0, uvpos1, uvpos2);
#if 0
				        if (v1_index_pos == 0) {
          uvpos0 = new_uv;
        }

        else if (v1_index_pos == 1) {
          uvpos1 = new_uv;
        }

        else if (v1_index_pos == 2) {
          uvpos2 = new_uv;
        }
#endif  // 0

        if (uvid0 == uv1 || uvid0 == uv2) {
          uvpos0 = new_uv;
        } else if (uvid1 == uv1 || uvid1 == uv2) {
          uvpos1 = new_uv;
        } else if (uvid2 == uv1 || uvid2 == uv2) {
          uvpos2 = new_uv;
        } else {
          //throw std::runtime_error("");
          // return false;
          //std::cout << fid << " " << uvid0 << " " << uvid1 << " " << uvid2 << " " << new_uv << std::endl;
          uv_suc_ = false;
        }

        float area_update = ugu::EdgeFunction(uvpos0, uvpos1, uvpos2);
        if (area * area_update < 0) {
          // return false;
          uv_suc_ = false;
        }

        if (!uv_suc_) {
            break;
        }

      }
      uv_suc = uv_suc_;
      if (uv_suc) {
        v1_index_pos = v1_index_pos_;
        interp_fid = interp_fid_;
        break;
      }

    } else {
     uv_suc = true;
     break;
    }
    }

    if (!uv_suc) {
      return false;
    }

#endif  // 0

#if 0
				    // Validation before collapse
    // Ensure normal is not flipped
    bool uv_suc = false;
    for (const auto& fid : to_keep_face_ids) {
      int32_t vid0 = vertex_indices[fid][0];
      int32_t vid1 = vertex_indices[fid][1];
      int32_t vid2 = vertex_indices[fid][2];
      // Replace v2
      // This is a test before real replacing
#if 1
      if (vid0 == v2) {
        vid0 = v1;
      } else if (vid1 == v2) {
        vid1 = v1;
      } else if (vid2 == v2) {
        vid2 = v1;
      } else {
        //throw std::runtime_error("");
      }
#endif  // 0

      Eigen::Vector3f vpos0 = vertices[vid0];
      Eigen::Vector3f vpos1 = vertices[vid1];
      Eigen::Vector3f vpos2 = vertices[vid2];

      if (v1 == vid0) {
        vpos0 = new_pos;
      } else if (v1 == vid1) {
        vpos1 = new_pos;
      } else if (v1 == vid2) {
        vpos2 = new_pos;
      } else {
        throw std::runtime_error("");
      }

      Eigen::Vector3f v10 = (vpos1 - vpos0).normalized();
      Eigen::Vector3f v20 = (vpos2 - vpos0).normalized();

      // Skip if it becomes one line
      if (std::abs(v10.dot(v20)) > 0.99999) {
        return false;
      }
      // Skip if normal is fiipped
      Eigen::Vector3f face_n = v10.cross(v20).normalized();
      if (face_n.dot(normals[v1]) < 0.0) {
        // Why is this condition not required?
        // return false;
      }

      if (use_uv) {
        bool uv_suc_ = true;
        // for (const auto& fid : to_keep_face_ids) {
        int32_t uvid0 = uv_indices[fid][0];
        int32_t uvid1 = uv_indices[fid][1];
        int32_t uvid2 = uv_indices[fid][2];
        Eigen::Vector2f uvpos0 = uv[uvid0];
        Eigen::Vector2f uvpos1 = uv[uvid1];
        Eigen::Vector2f uvpos2 = uv[uvid2];
        float area = ugu::EdgeFunction(uvpos0, uvpos1, uvpos2);
#if 0
				        if (v1_index_pos == 0) {
          uvpos0 = new_uv;
        }

        else if (v1_index_pos == 1) {
          uvpos1 = new_uv;
        }

        else if (v1_index_pos == 2) {
          uvpos2 = new_uv;
        }
#endif  // 0

        if (uvid0 == uv1 || uvid0 == uv2) {
          uvpos0 = new_uv;
        } else if (uvid1 == uv1 || uvid1 == uv2) {
          uvpos1 = new_uv;
        } else if (uvid2 == uv1 || uvid2 == uv2) {
          uvpos2 = new_uv;
        } else {
          // throw std::runtime_error("");
          // return false;
          std::cout << fid << " " << uvid0 << " " << uvid1 << " " << uvid2
                    << " " << new_uv << std::endl;
          uv_suc_ = false;
        }

        float area_update = ugu::EdgeFunction(uvpos0, uvpos1, uvpos2);
        if (area * area_update < 0) {
          // return false;
          uv_suc_ = false;
        }

        if (!uv_suc_) {
          //break;
        }

        uv_suc = uv_suc_;
        if (uv_suc) {
          //v1_index_pos = v1_index_pos_;
          //interp_fid = interp_fid_;
         // break;
        }

      } else {
        uv_suc = true;
        //break;
      }

    }

    if (!uv_suc) {
      return false;
    }
#endif  // 0


    // Update vertex attributes
    vertex_colors[v1] = new_color;
    normals[v1] = new_normal;
    vertices[v1] = new_pos;
    uv[uv1] = new_uv;

    // Update vertex/uv id in face
    {
      // Replace of B among the faces with A
      for (const auto& fid : to_keep_face_ids) {
        for (int32_t i = 0; i < 3; i++) {
          int32_t& vid = vertex_indices[fid][i];
          if (vid == v2) {
            vid = v1;
          }
          assert(valid_faces[fid]);

          if (use_uv) {
            // We ignore boundary uv
            // So #uv should be always 1
            if (vid2uvid[v1].size() != 1) {
              //   throw std::runtime_error("");
            }
            if (vid2uvid[v2].size() != 1) {
              //   throw std::runtime_error("");
            }
            int32_t& uvid = uv_indices[fid][i];
            if (uvid == uv2) {
              uvid = uv1;
            }
          }
        }
      }
    }

    // Update topology
    {
      // Remove 2 faces which share edge A-B from the faces
      for (const auto& fid : to_remove_face_ids) {
        //std::cout << "hoge " << fid << " " << interp_fid << std::endl;
        RemoveFace(fid);
      }

      // Merge v2f of B to A
      std::copy(v2f[v2].begin(), v2f[v2].end(), std::back_inserter(v2f[v1]));
      std::sort(v2f[v1].begin(), v2f[v1].end());
      auto result = std::unique(v2f[v1].begin(), v2f[v1].end());
      v2f[v1].erase(result, v2f[v1].end());

      {
        std::copy(uv_v2f[uv2].begin(), uv_v2f[uv2].end(),
                  std::back_inserter(uv_v2f[uv1]));
        std::sort(uv_v2f[uv1].begin(), uv_v2f[uv1].end());
        auto result = std::unique(uv_v2f[uv1].begin(), uv_v2f[uv1].end());
        uv_v2f[uv1].erase(result, uv_v2f[uv1].end());
      }

      // Remove the vertex B with older id
      RemoveVertex(v2);
      RemoveUv(uv2);
    }

    return true;
  }

  void Finalize(VertexAttrListPtr vert_attrs,
                const std::vector<UniqueVertex>& unique_vertices,
                ugu::QSlimType type) {
    ConvertVertexAttrs2Vectors(vert_attrs, unique_vertices, vertices, normals,
                               vertex_colors, uv, type);

    Finalize();
  }

  void Finalize() {
    // Appy valid mask
    auto material_ids = mesh->material_ids();

    material_ids.resize(FaceNum());
    auto materials = mesh->materials();

    mesh->Clear();

    RemoveVertices(valid_vertices, valid_faces, valid_uvs, valid_uv_faces);

    mesh->set_material_ids(material_ids);
    mesh->set_materials(materials);
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
                     const std::vector<bool>& valid_face_table_,
                     const std::vector<bool>& valid_uv_table_,
                     const std::vector<bool>& valid_uv_face_table_) {
    if (valid_vertex_table.size() != vertices.size()) {
      ugu::LOGE("valid_vertex_table must be same size to vertices");
      return -1;
    }

    int num_removed{0};
    std::vector<int> valid_table(vertices.size(), -1);
    std::vector<Eigen::Vector3f> valid_vertices, valid_vertex_colors;
    std::vector<Eigen::Vector2f> valid_uv;
    std::vector<Eigen::Vector3i> valid_indices, valid_uv_indices;
    // bool with_uv = !uv.empty() && !uv_indices.empty();
    bool with_vertex_color = !vertex_colors.empty();
    int valid_count = 0;
    for (size_t i = 0; i < vertices.size(); i++) {
      if (valid_vertex_table[i]) {
        valid_table[i] = valid_count;
        valid_vertices.push_back(vertices[i]);
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

    if (use_uv) {
      std::vector<int> valid_uv_table(uv.size(), -1);
      int valid_uv_count = 0;
      for (size_t i = 0; i < uv.size(); i++) {
        if (valid_uv_table_[i]) {
          valid_uv_table[i] = valid_uv_count;
          valid_uv.push_back(uv[i]);
          valid_uv_count++;
        } else {
          // num_removed++;
        }
      }

      int valid_uv_face_count{0};
      std::vector<int> valid_uv_face_table(uv_indices.size(), -1);
      for (size_t i = 0; i < uv_indices.size(); i++) {
        Eigen::Vector3i face;
        if (!valid_uv_face_table_[i]) {
          continue;
        }
        bool valid{true};
        for (int j = 0; j < 3; j++) {
          int new_index = valid_uv_table[uv_indices[i][j]];
          if (new_index < 0) {
            valid = false;
            break;
          }
          face[j] = new_index;
        }
        if (!valid) {
          continue;
        }
        valid_uv_indices.push_back(face);
        valid_uv_face_table[i] = valid_uv_face_count;
        valid_uv_face_count++;
      }
    }

    mesh->set_vertices(valid_vertices);
    mesh->set_vertex_indices(valid_indices);
    if (use_uv) {
      mesh->set_uv(valid_uv);
      mesh->set_uv_indices(valid_uv_indices);
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
  VertexAttrListPtr vert_attrs;
  ugu::QSlimType type;
  std::vector<UniqueVertex> unique_vertices;
  std::unordered_map<int32_t, std::vector<int32_t>> vid2unique;

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

    // std::cout << Q << std::endl;

    return Q;
  }

  void ComputeInitialQuadric(int32_t uniqid, int32_t vid,
                             const std::vector<int32_t>& neighbor_fids,
                             const std::vector<Eigen::Vector3i>& faces) {
    // for (const auto& vid : uni)

    for (const auto& fid : neighbor_fids) {
      const auto& face = faces[fid];
      // std::vector<int32_t> dst_vids;
      std::vector<std::vector<int32_t>> dst_uniqids;
      int32_t count = 0;
      for (int32_t i = 0; i < 3; i++) {
        if (face[i] != vid) {
          std::vector<int32_t> dst_uniq;
          for (const auto& uniqid_ : vid2unique[face[i]]) {
            dst_uniq.push_back(uniqid_);
          }
          dst_uniqids.push_back(dst_uniq);
        }
      }
      // Summation for all faces (planes) connecting to a vertex
      for (size_t j = 0; j < dst_uniqids[0].size(); j++) {
        for (size_t k = 0; k < dst_uniqids[1].size(); k++) {
          quadrics->at(uniqid) += ComputeInitialQuadric(
              uniqid, dst_uniqids[0][j], dst_uniqids[1][k]);
        }
      }
    }
  }

  void InitializeQuadrics(const DecimatedMesh& mesh,
                          const std::unordered_map<int, std::vector<int>>& v2f,
                          const std::vector<std::vector<int32_t>>& vid2uvid,
                          ugu::QSlimType type) {
    this->type = type;
    quadrics = std::make_shared<Quadrics>();
    vert_attrs = std::make_shared<VertexAttrList>();

    auto [unique_vertices_, vid2unique_] =
        GenerateUniqueVertices(mesh.vertex_indices, mesh.uv_indices);

    unique_vertices = std::move(unique_vertices_);
    vid2unique = std::move(vid2unique_);

    vert_attrs->resize(unique_vertices.size());
    quadrics->resize(unique_vertices.size());

    ConvertVectors2VertexAttrs(vert_attrs, unique_vertices, mesh.vertices,
                               mesh.normals, mesh.vertex_colors, mesh.uv, type);

    const Eigen::Index qsize = vert_attrs->at(0).rows() + 1;
    for (auto& q : *quadrics) {
      q.resize(qsize, qsize);
      q.setZero();
    }

#pragma omp parallel for
    for (int64_t i = 0; i < static_cast<int64_t>(vert_attrs->size()); i++) {
      // v2f.at(unique_vertices[i].vid);
      std::vector<int> adjacent_face_ids;
      mesh.uv_face_adjacency.GetAdjacentFaces(unique_vertices[i].fid,
                                              &adjacent_face_ids);

      ComputeInitialQuadric(i, unique_vertices[i].vid, adjacent_face_ids,
                            mesh.vertex_indices);
    }
  }
};

std::pair<std::set<QSlimEdge>, std::unordered_set<int32_t>> PrepareValidEdges(
    //  const DecimatedMesh& mesh,
    const std::vector<Eigen::Vector3i>& faces,
    const std::unordered_set<int>& unified_boundary_vertex_ids,
    const std::unordered_set<int>& unified_boundary_uv_ids,
    const std::vector<int32_t>& uvid2vid,
    const std::unordered_map<int, std::vector<int>>& v2f,
    bool keep_geom_boundary, bool keep_uv_boundary) {
  std::set<QSlimEdge> valid_edges;
  std::unordered_set<int32_t> invalid_vids;

  if (keep_geom_boundary) {
    invalid_vids = unified_boundary_vertex_ids;
#if 0
    std::unordered_set<int32_t> neigbor_vids;
    for (const auto& vid : invalid_vids) {
      for (const auto& fid : v2f.at(vid)) {
        neigbor_vids.insert(faces[fid][0]);
        neigbor_vids.insert(faces[fid][1]);
        neigbor_vids.insert(faces[fid][2]);
      }
    }
    invalid_vids.insert(neigbor_vids.begin(), neigbor_vids.end());
#endif
  }

  if (keep_uv_boundary) {
    for (const auto& uv_vid : unified_boundary_uv_ids) {
      invalid_vids.insert(uvid2vid.at(uv_vid));
    }
#if 0
    std::unordered_set<int32_t> neigbor_vids;
    for (const auto& uv_vid : unified_boundary_uv_ids) {
      for (const auto& fid : v2f.at(uvid2vid[uv_vid])) {
        neigbor_vids.insert(faces[fid][0]);
        neigbor_vids.insert(faces[fid][1]);
        neigbor_vids.insert(faces[fid][2]);
      }
    }
    invalid_vids.insert(neigbor_vids.begin(), neigbor_vids.end());
#endif
  }

  for (int32_t i = 0; i < static_cast<int32_t>(faces.size()); i++) {
    const auto& f = faces[i];
#if 0
	  size_t v0c = invalid_vids.count(f[0]);
    size_t v1c = invalid_vids.count(f[1]);
    size_t v2c = invalid_vids.count(f[2]);

    // Pair keeps always (smaller, bigger)

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

#endif  // 0

    size_t v0c = invalid_vids.count(f[0]);
    size_t v1c = invalid_vids.count(f[1]);
    size_t v2c = invalid_vids.count(f[2]);

    if (v0c > 0 || v1c > 0 || v2c > 0) {
      continue;
    }

    valid_edges.emplace(MakeQSlimEdge(f[0], f[1], i));
    valid_edges.emplace(MakeQSlimEdge(f[1], f[2], i));
    valid_edges.emplace(MakeQSlimEdge(f[2], f[0], i));
  }
  return {valid_edges, invalid_vids};
}

std::pair<bool, std::vector<QSlimEdgeInfo>> CollapseEdgeAndUpdateQuadrics(
    DecimatedMesh& mesh, QSlimHandler& handler, QSlimEdgeInfo& e,
    ugu::QSlimType type) {
  int32_t v1 = std::get<0>(e.edge);
  int32_t v2 = std::get<1>(e.edge);
  std::vector<QSlimEdgeInfo> new_edges;

#if 0
				  const std::vector<int32_t>& uniq1_list = handler.vid2unique[v1];
  const std::vector<int32_t>& uniq2_list = handler.vid2unique[v2];

  for (const auto& uniq1 : uniq1_list) {
    for (const auto& uniq2 : uniq2_list) {
    }
   }

#endif  // 0

  assert(handler.unique_vertices[v1].fid == handler.unique_vertices[v2].fid);

  bool ret = mesh.CollapseEdge(
      handler.unique_vertices[v1].vid, handler.unique_vertices[v1].uvid,
      handler.unique_vertices[v2].vid, handler.unique_vertices[v2].uvid, e.decimated_v, type);
  if (!ret) {
    return {false, new_edges};
  }

  // Update vertex attributes and quadrics
  handler.quadrics->at(v1) =
      handler.quadrics->at(v1) + handler.quadrics->at(v2);
  handler.vert_attrs->at(v1) = e.decimated_v;

  // Construct edges to add heap
  auto raw_edges = mesh.ConnectingEdges(handler.unique_vertices[v1].vid);
  bool keep_this_edge = false;

#if 0
				  const std::vector<int32_t>& uniq1_list = handler.vid2unique[v1];
  const std::vector<int32_t>& uniq2_list = handler.vid2unique[v2];

  for (const auto& uniq1 : uniq1_list) {
    for (const auto& uniq2 : uniq2_list) {
    }
  }

#endif  // 0

  for (const auto& e : raw_edges) {
    const std::vector<int32_t>& uniq1_list =
        handler.vid2unique[std::get<0>(e)];
    const std::vector<int32_t>& uniq2_list =
        handler.vid2unique[std::get<1>(e)];
    for (const auto& uniq1 : uniq1_list) {
      const auto& uvf1 = mesh.uv_v2f[handler.unique_vertices[uniq1].uvid];
      if (std::find(uvf1.begin(), uvf1.end(), std::get<2>(e)) == uvf1.end()) {
        continue;
      }
      for (const auto& uniq2 : uniq2_list) {
#if 1
				        const auto& uvf2 = mesh.uv_v2f[handler.unique_vertices[uniq2].uvid];


      if (std::find(uvf2.begin(), uvf2.end(),
                                                      std::get<2>(e)) ==
                                            uvf2.end()) {
                                          continue;
                                        }
        if (mesh.use_uv) {
          std::vector<int32_t> intersection;
          std::set_intersection(uvf1.begin(), uvf1.end(), uvf2.begin(),
                                uvf2.end(), std::back_inserter(intersection));
          // Check connection in uv
          if (intersection.empty()) {
            continue;
          }
        }
#endif  // 

#if 0
				        if (handler.unique_vertices[uniq1].fid !=
            handler.unique_vertices[uniq2].fid) {
          continue;
        }
#endif  // 0


        auto new_edge =
            QSlimEdgeInfo(MakeQSlimEdge(uniq1, uniq2, std::get<2>(e)), handler.vert_attrs,
                          handler.quadrics, keep_this_edge);
        new_edges.emplace_back(new_edge);
      }
    }
  }

  return {true, new_edges};
}

}  // namespace

namespace ugu {

bool RandomDecimation(MeshPtr mesh, QSlimType type, int32_t target_face_num,
                      int32_t target_vertex_num, bool keep_geom_boundary,
                      bool keep_uv_boundary) {
#if 0
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

    decimated_mesh.CollapseEdge(vid1, vid2, new_pos, type);

    std::cout << decimated_mesh.FaceNum() << " " << decimated_mesh.VertexNum()
              << std::endl;
  }


  decimated_mesh.Finalize();
#endif
  return true;
}

bool QSlim(MeshPtr mesh, QSlimType type, int32_t target_face_num,
           int32_t target_vertex_num, bool keep_geom_boundary,
           bool keep_uv_boundary) {
  // Set up valid edges (we do not consider non-edges yet)
  DecimatedMesh decimated_mesh(mesh);
  auto [valid_edges, ignore_vids_] = PrepareValidEdges(
      decimated_mesh.vertex_indices, decimated_mesh.unified_boundary_vertex_ids,
      decimated_mesh.unified_boundary_uv_ids, decimated_mesh.uvid2vid,
      decimated_mesh.v2f, keep_geom_boundary, keep_uv_boundary);

  decimated_mesh.ignore_vids = std::move(ignore_vids_);

  for (const auto& vid : decimated_mesh.ignore_vids) {
    for (const auto& uvid : decimated_mesh.vid2uvid[vid]) {
      decimated_mesh.ignore_uv_ids.insert(uvid);
    }
  }

  // Initialize quadrics
  QSlimHandler handler;
  handler.InitializeQuadrics(decimated_mesh, decimated_mesh.v2f,
                             decimated_mesh.vid2uvid, type);

  // Initialize heap
  QSlimHeap heap;

  // Add the valid pairs  to heap
  for (const auto& p : valid_edges) {
    const std::vector<int32_t>& uniq1_list = handler.vid2unique[std::get<0>(p)];
    const std::vector<int32_t>& uniq2_list = handler.vid2unique[std::get<1>(p)];
    for (const auto& uniq1 : uniq1_list) {
      const auto& uvf1 =
          decimated_mesh.uv_v2f[handler.unique_vertices[uniq1].uvid];

      assert(std::find(uvf1.begin(), uvf1.end(),std::get<2>(p)) != uvf1.end() );

      if (std::find(uvf1.begin(), uvf1.end(), std::get<2>(p)) == uvf1.end()) {
          continue;
      }

      for (const auto& uniq2 : uniq2_list) {
        const auto& uvf2 =
            decimated_mesh.uv_v2f[handler.unique_vertices[uniq2].uvid];
        if (std::find(uvf2.begin(), uvf2.end(), std::get<2>(p)) == uvf2.end()) {
          continue;
        }

        if (decimated_mesh.use_uv) {
#if 1
				          std::vector<int32_t> intersection;
          std::set_intersection(uvf1.begin(), uvf1.end(), uvf2.begin(),
                                uvf2.end(), std::back_inserter(intersection));
          // Check connection in uv
          if (intersection.empty()) {
            continue;
          }
#endif  // 0
        }
#if 0
				
        if (handler.unique_vertices[uniq1].fid !=
            handler.unique_vertices[uniq2].fid) {
          continue;
        }
#endif  // 0


        auto new_edge =
            QSlimEdgeInfo(MakeQSlimEdge(uniq1, uniq2, std::get<2>(p)), handler.vert_attrs,
                          handler.quadrics, false);
        heap.push(new_edge);
      }
    }

    // QSlimEdgeInfo info(p, handler.vert_attrs, handler.quadrics, false);
  }

  std::unordered_set<int32_t> removed_vids;
  std::unordered_set<int32_t> removed_uvids;

#if 0
				  for (const auto& vid : decimated_mesh.ignore_vids) {
    decimated_mesh.valid_vertices[vid] = false;
    for (const auto& fid : decimated_mesh.v2f[vid]) {
      decimated_mesh.valid_faces[fid] = false;
      decimated_mesh.ignore_fids.insert(fid);
    }
    decimated_mesh.v2f[vid].clear();
  }

  for (auto& p : decimated_mesh.v2f) {
    for (const auto fid : decimated_mesh.ignore_fids) {
      p.second.erase(std::remove(p.second.begin(), p.second.end(), fid),
                     p.second.end());
    }
  }
#endif  // 0

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

    assert(!min_e.keep_this_edge);

    heap.pop();

#if 0
				    if (decimated_mesh.unified_boundary_vertex_ids.count(min_e.edge.first) >
            0 ||
        decimated_mesh.unified_boundary_vertex_ids.count(min_e.edge.second) >
            0) {
      continue;
    }

#endif  // 0

    // Skip if it has been decimated
    if (removed_vids.count(std::get<0>(min_e.edge)) > 0 ||
        removed_vids.count(std::get<1>(min_e.edge)) > 0) {
      if (removed_vids.count(std::get<0>(min_e.edge)) > 0) {
   //     assert(!decimated_mesh.valid_vertices
    //                [handler.unique_vertices[min_e.edge.first].vid]);
      }
      if (removed_vids.count(std::get<1>(min_e.edge)) > 0) {
//        assert(!decimated_mesh.valid_vertices
 //                   [handler.unique_vertices[min_e.edge.second].vid]);
      }
      continue;
    }

   if (removed_uvids.count(handler.unique_vertices[std::get<0>(min_e.edge)].uvid) >
            0 ||
        removed_uvids.count(handler.unique_vertices[std::get<1>(min_e.edge)].uvid) >
            0) {
     // continue;
  }

    // std::cout << "decimate " << min_e.edge.first << ", " << min_e.edge.second
    //         << " " << min_e.error << std::endl;

    // Decimate the pair
    auto [is_collapsed, new_edges] =
        CollapseEdgeAndUpdateQuadrics(decimated_mesh, handler, min_e, type);

    if (!is_collapsed) {
      continue;
    }

    // Memory removed vid
    removed_vids.insert(std::get<1>(min_e.edge));

    //removed_uvids.insert(handler.unique_vertices[min_e.edge.second].uvid);

    // Add new pairs
    for (const auto& e : new_edges) {
      heap.push(e);
    }

    // std::cout << decimated_mesh.FaceNum() << " " <<
    // decimated_mesh.VertexNum()
    //           << std::endl;
  }

#if 1
  // Recover
  for (const auto& vid : decimated_mesh.ignore_vids) {
    decimated_mesh.valid_vertices[vid] = true;
  }
  for (const auto& fid : decimated_mesh.ignore_fids) {
    decimated_mesh.valid_faces[fid] = true;
  }
  for (const auto& vid : decimated_mesh.ignore_uv_ids) {
    decimated_mesh.valid_uvs[vid] = true;
  }

#endif  // 0

  decimated_mesh.Finalize(handler.vert_attrs, handler.unique_vertices, type);

  return true;
}

}  // namespace ugu
