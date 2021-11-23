/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include <stdio.h>

#include <fstream>

#include "ugu/geodesic/geodesic.h"
#include "ugu/inpaint/inpaint.h"
#include "ugu/timer.h"
#include "ugu/util/image_util.h"
#include "ugu/util/raster_util.h"

namespace {

void SaveGeodesicDistance(const std::string& data_dir,
                          const std::string& prefix, ugu::Mesh& mesh,
                          std::vector<double>& dists) {
  auto max_dist = *std::max_element(dists.begin(), dists.end());
  ugu::LOGI("max dist: %f\n", max_dist);
  int contour_line_num = 10;
  double contour_line_step = max_dist / contour_line_num;
  double contour_line_width = contour_line_step * 0.15;

  auto dist2color = [&](double d) -> Eigen::Vector3f {
    bool in_contour_line = false;
    double dist_to_contour = 0.0;
    for (int i = 0; i < contour_line_num; i++) {
      double contour_min = i * contour_line_step - contour_line_width;
      double contour_max = i * contour_line_step + contour_line_width;
      if (contour_min < d && d < contour_max) {
        in_contour_line = true;
        dist_to_contour =
            std::min(std::abs(d - contour_min), std::abs(d - contour_max));
        break;
      }
    }

    float r = static_cast<float>(255.0 * d / max_dist);

    if (in_contour_line) {
      double alpha = dist_to_contour / contour_line_width;
      return {static_cast<float>(r * (1.0 - alpha)), 0.f,
              static_cast<float>(255.0 * alpha)};
      // return {0.f, 0.f, 255.f};
    }

    return {r, 0.f, 0.f};
  };

  std::vector<Eigen::Vector3f> vertex_colors;
  std::transform(dists.begin(), dists.end(), std::back_inserter(vertex_colors),
                 dist2color);

  mesh.set_vertex_colors(vertex_colors);
  mesh.WritePly(data_dir + prefix + "bunny_geodesic_distance_vertex.ply");

  ugu::ObjMaterial distance_mat;
  distance_mat.diffuse_texname = prefix + "geodesic_distance.png";
  int tex_len = 512;
  distance_mat.diffuse_tex = ugu::Image3b::zeros(tex_len, tex_len);

  ugu::RasterizeVertexAttributeToTexture(
      mesh.vertex_colors(), mesh.vertex_indices(), mesh.uv(), mesh.uv_indices(),
      distance_mat.diffuse_tex);

  distance_mat.diffuse_texname =
      prefix + "geodesic_distance_vertex_rasterized.png";
  mesh.set_materials({distance_mat});
  mesh.WriteObj(data_dir, prefix + "bunny_geodesic_distance_vertex_rasterized");

  distance_mat.diffuse_texname = prefix + "geodesic_distance.png";
  distance_mat.diffuse_tex = ugu::Image3b::zeros(tex_len, tex_len);
  auto& geodesic_tex = distance_mat.diffuse_tex;

  ugu::Image1b mask = ugu::Image1b::zeros(tex_len, tex_len);
  for (size_t i = 0; i < mesh.vertex_indices().size(); i++) {
    const auto& f = mesh.vertex_indices()[i];

    std::array<Eigen::Vector2f, 3> target_tri;
    for (int j = 0; j < 3; j++) {
      auto uv_idx = mesh.uv_indices()[i][j];
      auto uv = mesh.uv()[uv_idx];
      target_tri[j][0] = uv[0] * tex_len - 0.5f;
      target_tri[j][1] = (1.0f - uv[1]) * tex_len - 0.5f;
    }

    float area = ugu::EdgeFunction(target_tri[0], target_tri[1], target_tri[2]);
    if (std::abs(area) < std::numeric_limits<float>::min()) {
      area = area > 0 ? std::numeric_limits<float>::min()
                      : -std::numeric_limits<float>::min();
    }
    float inv_area = 1.0f / area;

    // Loop for bounding box of the target triangle
    int xmin = static_cast<int>(
        std::min({target_tri[0].x(), target_tri[1].x(), target_tri[2].x()}) -
        1);
    xmin = std::max(0, std::min(xmin, geodesic_tex.cols - 1));
    int xmax = static_cast<int>(
        std::max({target_tri[0].x(), target_tri[1].x(), target_tri[2].x()}) +
        1);
    xmax = std::max(0, std::min(xmax, geodesic_tex.cols - 1));

    int ymin = static_cast<int>(
        std::min({target_tri[0].y(), target_tri[1].y(), target_tri[2].y()}) -
        1);
    ymin = std::max(0, std::min(ymin, geodesic_tex.rows - 1));
    int ymax = static_cast<int>(
        std::max({target_tri[0].y(), target_tri[1].y(), target_tri[2].y()}) +
        1);
    ymax = std::max(0, std::min(ymax, geodesic_tex.rows - 1));

    for (int y = ymin; y <= ymax; y++) {
      for (int x = xmin; x <= xmax; x++) {
        Eigen::Vector2f pixel_sample(static_cast<float>(x),
                                     static_cast<float>(y));
        float w0 =
            ugu::EdgeFunction(target_tri[1], target_tri[2], pixel_sample);
        float w1 =
            ugu::EdgeFunction(target_tri[2], target_tri[0], pixel_sample);
        float w2 =
            ugu::EdgeFunction(target_tri[0], target_tri[1], pixel_sample);
        // Barycentric coordinate
        w0 *= inv_area;
        w1 *= inv_area;
        w2 *= inv_area;

        auto& c = geodesic_tex.at<ugu::Vec3b>(y, x);
        // Barycentric coordinate should be positive inside of the triangle
        // Skip outside of the target triangle
        if (w0 < 0 || w1 < 0 || w2 < 0) {
          continue;
        }

        // Barycentric interpolation of geodesic distance
        double d_interp =
            w0 * dists[f[0]] + w1 * dists[f[1]] + w2 * dists[f[2]];
        auto color = dist2color(d_interp);
        c[0] = static_cast<unsigned char>(color[0]);
        c[1] = static_cast<unsigned char>(color[1]);
        c[2] = static_cast<unsigned char>(color[2]);

        mask.at<unsigned char>(y, x) = 255;
      }
    }
  }

  // Inpaint to avoid color bleeding
  ugu::Image1b inpaint_mask;
  ugu::Not(mask, &inpaint_mask);
  ugu::Inpaint(inpaint_mask, geodesic_tex);

  mesh.set_materials({distance_mat});
  mesh.WriteObj(data_dir, prefix + "bunny_geodesic_distance");
}

}  // namespace

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  std::string data_dir = "../data/bunny/";
  std::string in_obj_path = data_dir + "bunny.obj";

  ugu::Mesh mesh;
  int src_vid = 0;
  Eigen::SparseMatrix<float> edge_dists;
  std::vector<double> dists;
  std::vector<int> min_path_edges;

  ugu::Timer<> timer;

  mesh.LoadObj(in_obj_path, data_dir);
  timer.Start();
  ugu::ComputeGeodesicDistance(mesh, src_vid, edge_dists, dists, min_path_edges,
                               ugu::GeodesicComputeMethod::DIJKSTRA);
  timer.End();
  ugu::LOGI("dijsktra %f\n", timer.elapsed_msec());
  SaveGeodesicDistance(data_dir, "dijsktra_", mesh, dists);

  mesh.LoadObj(in_obj_path, data_dir);
  timer.Start();
  ugu::ComputeGeodesicDistance(
      mesh, src_vid, edge_dists, dists, min_path_edges,
      ugu::GeodesicComputeMethod::FAST_MARCHING_METHOD);
  timer.End();
  ugu::LOGI("fmm %f\n", timer.elapsed_msec());
  SaveGeodesicDistance(data_dir, "fmm_", mesh, dists);

  return 0;
}
