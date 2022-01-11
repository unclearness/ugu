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

  // This is rasterization for visualized colors.
  // So this does not accurately interpolate distances and the corresponding
  // colors and may have artifacts on blue border lines.
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

  ugu::Image3f geodesic_tex_f = ugu::Image3f::zeros(tex_len, tex_len);
  std::vector<Eigen::Vector3f> dists3;
  std::transform(dists.begin(), dists.end(), std::back_inserter(dists3),
                 [&](const double& d) {
                   float d_f = static_cast<float>(d);
                   return Eigen::Vector3f(d_f, d_f, d_f);
                 });
  // Rasterize float distance to float texture
  // This does not cause artifacts
  ugu::RasterizeVertexAttributeToTexture(
      dists3, mesh.vertex_indices(), mesh.uv(), mesh.uv_indices(),
      geodesic_tex_f, tex_len, tex_len, &mask);
  // Convert to uchar from float
  geodesic_tex_f.forEach<ugu::Vec3f>([&](ugu::Vec3f& p, const int position[2]) {
    auto& c = geodesic_tex.at<ugu::Vec3b>(position[1], position[0]);
    auto color = dist2color(p[0]);
    c[0] = static_cast<unsigned char>(color[0]);
    c[1] = static_cast<unsigned char>(color[1]);
    c[2] = static_cast<unsigned char>(color[2]);
  });

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
