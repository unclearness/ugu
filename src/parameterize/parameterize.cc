/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/parameterize/parameterize.h"

#include "ugu/clustering/clustering.h"
#include "ugu/discrete/bin_packer_2d.h"
#include "ugu/plane.h"
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
  constexpr bool debug = false;

  // Step 1: Segment mesh
  ugu::SegmentMeshResult res;
  ugu::SegmentMesh(vertices, faces, face_normals, res);

  // Step 2: Parameterize per segment
  std::vector<std::vector<Eigen::Vector2f>> cluster_uvs(
      res.cluster_fids.size());
  std::vector<std::vector<Eigen::Vector3i>> cluster_sub_faces;
  for (size_t cid = 0; cid < res.clusters.size(); ++cid) {
    auto prj_n = res.cluster_representative_normals[cid];
    auto [cluster_vtx, cluster_face] =
        ugu::ExtractSubGeom(vertices, faces, res.cluster_fids[cid]);

    cluster_sub_faces.push_back(cluster_face);

    ugu::OrthoProjectToXY(prj_n, cluster_vtx, cluster_uvs[cid], true, true,
                          true);

    if (debug) {
      auto uv_img = ugu::DrawUv(cluster_uvs[cid], cluster_face, {255, 255, 255},
                                {0, 0, 0});
      uv_img.WritePng(std::to_string(cid) + ".png");
    }
  }

  // Step 3: Pack segments
  ugu::PackUvIslands(res.cluster_areas, res.clusters, cluster_uvs,
                     cluster_sub_faces, res.cluster_fids, faces.size(), tex_w,
                     tex_h, uvs, uv_faces, true);

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

  points_2d.clear();
  for (const auto& p3d : points_3d) {
    Eigen::Vector3f p = plane.Project(p3d);
    p = R * p;
    points_2d.push_back({p[0], p[1]});
  }

  if (align_longest_axis_x && points_2d.size() > 1) {
    std::array<Eigen::Vector2f, 2> axes;
    std::array<float, 2> weights;
    Eigen::Vector2f means;

    if (points_2d.size() <= 3) {
      // PCA may be unstable
      // Find the longest manually...
      float max_len = -1.f;
      for (size_t i = 0; i < points_2d.size(); i++) {
        for (size_t j = i + 1; j < points_2d.size(); j++) {
          Eigen::Vector2f axis_ = points_2d[i] - points_2d[j];
          float len = (axis_).norm();
          if (max_len < len) {
            max_len = len;
            axes[0] = axis_.normalized();
          }
        }
      }

    } else {
      // Find the dominant axis by PCA
      ComputeAxisForPoints(points_2d, axes, weights, means);
    }

    // Angle from X-axis, Y-axis is down in UV
    Eigen::Vector2f dominant_axis = axes[0].normalized();
    float rad = std::atan2(-dominant_axis[1], dominant_axis[0]);

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
    Eigen::Vector2f max_bound = ComputeMaxBound(points_2d);
    Eigen::Vector2f min_bound = ComputeMinBound(points_2d);
    Eigen::Vector2f len = max_bound - min_bound;
    Eigen::Vector2f inv_len(1.f / len[0], 1.f / len[1]);

    float aspect = 1.f;
    if (keep_aspect && len[0] > 0.f) {
      aspect = len[1] / len[0];
    }

    // TODO:
    // sometimes aspect is largaer than 1 with align_longest_axis_x
    // Why?
    aspect = std::min(1.f, aspect);

    for (auto& p2d : points_2d) {
      p2d = (p2d - min_bound).cwiseProduct(inv_len);
      p2d[1] *= aspect;
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

bool PackUvIslands(
    const std::vector<float>& cluster_areas,
    const std::vector<std::vector<Eigen::Vector3i>>& clusters,
    const std::vector<std::vector<Eigen::Vector2f>>& cluster_uvs,
    const std::vector<std::vector<Eigen::Vector3i>>& cluster_sub_faces,
    const std::vector<std::vector<uint32_t>>& cluster_fids, size_t num_faces,
    int tex_w, int tex_h, std::vector<Eigen::Vector2f>& uvs,
    std::vector<Eigen::Vector3i>& uv_faces, bool flip_v,
    const std::vector<float>& cluster_weights) {
  std::vector<float> normalized_areas;
  double sum_area = 0.0;
  for (const auto& a : cluster_areas) {
    sum_area += a;
  }

  for (const auto& a : cluster_areas) {
    normalized_areas.push_back(
        static_cast<float>(static_cast<double>(a) / sum_area));
  }

  auto indices = ugu::argsort(normalized_areas, true);
  const float one_pix_area =
      std::pow(1.f / static_cast<float>(std::max(tex_w, tex_h)), 2.f);
  const float smallest_area =
      std::max(normalized_areas[indices.back()], one_pix_area);
  const float padding =
      std::min(std::sqrt(smallest_area), std::sqrt(one_pix_area));

  for (auto& a : normalized_areas) {
    a = std::max(a, one_pix_area);
  }

  std::vector<ugu::Rect2f> rects, packed_pos, available_rects;

  std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> min_max_local_uvs(
      clusters.size());

  std::vector<float> scales(indices.size());
  bool with_weights = cluster_areas.size() == cluster_weights.size();
  for (const auto& idx : indices) {
    const auto& uvs = cluster_uvs[idx];
    Eigen::Vector2f max_bound = ugu::ComputeMaxBound(uvs);
    Eigen::Vector2f min_bound = ugu::ComputeMinBound(uvs);

    min_max_local_uvs[idx] = {min_bound, max_bound};

    Eigen::Vector2f len = max_bound - min_bound;

    float r = std::sqrt(normalized_areas[idx]);
    if (with_weights) {
      r *= cluster_weights[idx];
    }
    scales[idx] = r;
    float w = len[0] * r + padding;
    float h = len[1] * r + padding;
    rects.push_back(ugu::Rect2f(0.f, 0.f, w, h));
  }

  int bin_packing_try_num = 0;
  const float start_scale = 2.f;
  const float pyramid_ratio = 0.95f;

  // TODO: Check fill rate
  // float best_fill_rate = 0.f;

  std::vector<ugu::Rect2f> start_rects = rects;
  float current_scale = start_scale;

  auto rescale_rects = [&]() {
    rects = start_rects;
    current_scale = start_scale * std::pow(pyramid_ratio, bin_packing_try_num);
    for (auto& rect : rects) {
      rect.width *= current_scale;
      rect.height *= current_scale;
    }
  };

  rescale_rects();
  while (!ugu::BinPacking2D(rects, &packed_pos, &available_rects, padding,
                            1.f - padding, padding, 1.f - padding)) {
    bin_packing_try_num++;

    // Rescale rects
    rescale_rects();
  }

  bool debug = false;
  if (debug) {
    auto vis = ugu::DrawPackedRects(packed_pos, tex_w, tex_h);
    vis.WriteJpg("tmp.jpg");
  }

  uvs.clear();
  uv_faces.clear();
  uv_faces.resize(num_faces);

  for (int k = 0; k < static_cast<int>(indices.size()); k++) {
    int uv_index_offset = static_cast<int>(uvs.size());

    const ugu::Rect2f& rect = packed_pos[k];
    size_t cid = indices[k];

    auto [min_bound, max_bound] = min_max_local_uvs[cid];
    float scale = scales[cid] * current_scale;

    Eigen::Vector2f uv_offset(rect.x, rect.y);
    for (const auto& cuv : cluster_uvs[cid]) {
      Eigen::Vector2f tmp;
      tmp[0] = (cuv[0] - min_bound[0]) * scale + uv_offset[0];

      if (flip_v) {
        // Flip V
        tmp[1] = (1.f - cuv[1]) * scale + uv_offset[1];
        tmp[1] = 1.f - tmp[1];
      } else {
        // V is from top, y image coordinate used in BinPacking2D
        tmp[1] = (cuv[1] - min_bound[1]) * scale + uv_offset[1];
      }

      uvs.push_back(tmp);
    }

    for (uint32_t j = 0; j < cluster_fids[cid].size(); j++) {
      uint32_t fid = cluster_fids[cid][j];

      // Add to global uv
      Eigen::Vector3i global_index;
      const Eigen::Vector3i& local_index = cluster_sub_faces[cid][j];
      global_index[0] = uv_index_offset + local_index[0];
      global_index[1] = uv_index_offset + local_index[1];
      global_index[2] = uv_index_offset + local_index[2];
      uv_faces[fid] = global_index;
    }
  }

  return true;
}

}  // namespace ugu