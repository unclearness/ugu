/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/parameterize/parameterize.h"

#include "ugu/clustering/clustering.h"
#include "ugu/line.h"
#include "ugu/util/geom_util.h"
#include "ugu/util/math_util.h"

namespace {

bool ParameterizeSimpleTriangles(const std::vector<Eigen::Vector3i>& faces,
                                 std::vector<Eigen::Vector2f>& uvs,
                                 std::vector<Eigen::Vector3i>& uv_faces,
                                 int tex_w, int tex_h) {
  // Padding must be at least 2
  // to pad right/left and up/down
  const int padding_tri = 2;

  int rect_num = static_cast<int>((faces.size() + 1) / 2);
  int pix_per_rect = tex_h * tex_w / rect_num;
  if (pix_per_rect < 6) {
    return false;
  }
  int max_rect_edge_len = 100;
  int sq_len =
      std::min(static_cast<int>(std::sqrt(pix_per_rect)), max_rect_edge_len);
  /*
   * example. rect_w = 4
   * * is padding on diagonal (fixed)
   * + is upper triangle, - is lower triangle
   * ++++**
   * +++**-
   * ++**--
   * +**---
   * **----
   *
   */

  int max_rect_num = (tex_w / (sq_len + 2 + padding_tri)) *
                     (tex_h / (sq_len + 1 + padding_tri));
  while (max_rect_num < rect_num) {
    sq_len--;
    if (sq_len < 3) {
      return false;
    }
    max_rect_num = (tex_w / (sq_len + 2 + padding_tri)) *
                   (tex_h / (sq_len + 1 + padding_tri));
  }

  int rect_w = sq_len + 2;
  int rect_h = sq_len + 1;

  // Loop per face
  int rect_w_num = tex_w / (rect_w + padding_tri);
  for (int i = 0; i < static_cast<int>(faces.size()); i++) {
    int rect_id = i / 2;
    int rect_x = rect_id % rect_w_num;
    int rect_y = rect_id / rect_w_num;

    std::array<Eigen::Vector2f, 3> target_tri, target_tri_uv;
    bool lower = i % 2 == 0;
    if (lower) {
      int rect_x_min = (rect_w + padding_tri) * rect_x + 2;
      int rect_x_max = rect_x_min + sq_len - 1;
      int rect_y_min = (rect_h + padding_tri) * rect_y;
      int rect_y_max = rect_y_min + sq_len - 1;

      target_tri[0] = Eigen::Vector2f{rect_x_min, rect_y_min};
      target_tri[1] = Eigen::Vector2f{rect_x_max, rect_y_min};
      target_tri[2] = Eigen::Vector2f{rect_x_max, rect_y_max};
    } else {
      int rect_x_min = (rect_w + padding_tri) * rect_x;
      int rect_x_max = rect_x_min + sq_len - 1;
      int rect_y_min = (rect_h + padding_tri) * rect_y + 1;
      int rect_y_max = rect_y_min + sq_len - 1;

      target_tri[0] = Eigen::Vector2f{rect_x_min, rect_y_min};
      target_tri[1] = Eigen::Vector2f{rect_x_min, rect_y_max};
      target_tri[2] = Eigen::Vector2f{rect_x_max, rect_y_max};
    }

    for (int j = 0; j < 3; j++) {
      target_tri_uv[j].x() = (target_tri[j].x() + 0.5f) / tex_w;
      target_tri_uv[j].y() = 1.0f - ((target_tri[j].y() + 0.5f) / tex_h);
    }

    uvs.push_back(target_tri_uv[0]);
    uvs.push_back(target_tri_uv[1]);
    uvs.push_back(target_tri_uv[2]);
    int uv_size = static_cast<int>(uvs.size());
    uv_faces.push_back(Eigen::Vector3i(uv_size - 3, uv_size - 2, uv_size - 1));
  }

  return true;
}

bool ParameterizeSmartUv(const std::vector<Eigen::Vector3f>& vertices,
                         const std::vector<Eigen::Vector3i>& faces,
                         const std::vector<Eigen::Vector3f>& face_normals,
                         std::vector<Eigen::Vector2f>& uvs,
                         std::vector<Eigen::Vector3i>& uv_faces, int tex_w,
                         int tex_h) {
  // Segment mesh
  ugu::SegmentMeshResult res;
  ugu::SegmentMesh(vertices, faces, face_normals, res);

  // Parameterize per segment
  std::vector<std::vector<Eigen::Vector2f>> cluster_uvs(
      res.cluster_fids.size());
  std::vector<std::vector<Eigen::Vector3i>> cluster_uv_faces(
      res.cluster_fids.size());
  for (size_t cid = 0; cid < res.clusters.size(); ++cid) {
    auto prj_n = res.cluster_representative_normals[cid];
    std::vector<Eigen::Vector3f> cluster_vtx;
    std::vector<Eigen::Vector3i> cluster_face;
    std::unordered_map<int32_t, int32_t> org2seg;
    int32_t count = 0;
    for (const auto& org_fid : res.cluster_fids[cid]) {
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
          cluster_vtx.push_back(vertices[org_vid]);
          vid = count;
          count++;
        }
        face[i] = vid;
      }
      cluster_face.push_back(face);
    }
    ugu::OrthoProjectToXY(prj_n, cluster_vtx, cluster_uvs[cid], true, true,
                          true);

    auto uv_img =
        ugu::DrawUv(cluster_uvs[cid], cluster_face, {255, 255, 255}, {0, 0, 0});
    uv_img.WritePng(std::to_string(cid) + ".png");
  }

  // Pack segments

  return true;
}

}  // namespace

namespace ugu {

bool Parameterize(Mesh& mesh, int tex_w, int tex_h, ParameterizeUvType type) {
  std::vector<Eigen::Vector2f> uvs;
  std::vector<Eigen::Vector3i> uv_faces;
  bool ret =
      Parameterize(mesh.vertices(), mesh.vertex_indices(), mesh.face_normals(),
                   uvs, uv_faces, tex_w, tex_h, type);
  if (ret) {
    mesh.set_uv(uvs);
    mesh.set_uv_indices(uv_faces);
  }

  return ret;
}

bool Parameterize(const std::vector<Eigen::Vector3f>& vertices,
                  const std::vector<Eigen::Vector3i>& faces,
                  const std::vector<Eigen::Vector3f>& face_normals,
                  std::vector<Eigen::Vector2f>& uvs,
                  std::vector<Eigen::Vector3i>& uv_faces, int tex_w, int tex_h,
                  ParameterizeUvType type) {
  if (type == ParameterizeUvType::kSimpleTriangles) {
    return ParameterizeSimpleTriangles(faces, uvs, uv_faces, tex_w, tex_h);
  } else if (type == ParameterizeUvType::kSmartUv) {
    return ParameterizeSmartUv(vertices, faces, face_normals, uvs, uv_faces,
                               tex_w, tex_h);
  }

  return false;
}

bool OrthoProjectToXY(const Eigen::Vector3f& project_normal,
                      const std::vector<Eigen::Vector3f>& points_3d,
                      std::vector<Eigen::Vector2f>& points_2d,
                      bool align_longest_axis_x, bool normalize,
                      bool keep_aspect, bool align_top_y) {
  constexpr float d = 0.f;  // any value is ok.
  Planef plane(project_normal, d);
  const Eigen::Vector3f z_vec(0.f, 0.f, 1.f);

  const float angle = std::acos(project_normal.dot(z_vec));
  const Eigen::Vector3f axis = (project_normal.cross(z_vec)).normalized();

  const Eigen::Matrix3f R = Eigen::AngleAxisf(angle, axis).matrix();

  for (const auto& p3d : points_3d) {
    Eigen::Vector3f p = plane.Project(p3d);
    p = R * p;
    points_2d.push_back({p[0], p[1]});
  }

  if (align_longest_axis_x) {
    std::array<Eigen::Vector2f, 2> axes;
    std::array<float, 2> weights;

    // Find the dominant axis
    ComputeAxisForPoints(points_2d, axes, weights);
    float rad = std::acos(
        axes[0].normalized().dot(Eigen::Vector2f(1.f, 0.f).transpose()));

    // Rotate points around the axis
    Eigen::Matrix2f R_2d;
    R_2d(0, 0) = std::cos(rad);
    R_2d(0, 1) = -std::sin(rad);
    R_2d(1, 0) = std::sin(rad);
    R_2d(1, 1) = std::cos(rad);
    for (auto& p2d : points_2d) {
      p2d = R_2d * p2d;
    }
  }

  if (normalize) {
    auto max_bound = ComputeMaxBound(points_2d);
    auto min_bound = ComputeMinBound(points_2d);
    auto len = max_bound - min_bound;
    Eigen::Vector2f inv_len(1.f / len[0], 1.f / len[1]);
    if (keep_aspect) {
      float smaller_inv_len = inv_len[1];
      Eigen::Vector2f smaller_min_bound =
          Eigen::Vector2f::Constant(min_bound[1]);
      if (len[1] < len[0]) {
        smaller_inv_len = inv_len[0];
        smaller_min_bound = Eigen::Vector2f::Constant(min_bound[0]);
      }
      for (auto& p2d : points_2d) {
        p2d = (p2d - smaller_min_bound) * smaller_inv_len;
      }
    } else {
      for (auto& p2d : points_2d) {
        p2d = (p2d - min_bound).cwiseProduct(inv_len);
      }
    }
    if (align_top_y) {
      auto max_bound2 = ComputeMaxBound(points_2d);
      float offset_y = 1.f - max_bound2[1];
      for (auto& p2d : points_2d) {
        p2d[1] = p2d[1] + offset_y;
      }
    }
  }

  return true;
}
}  // namespace ugu