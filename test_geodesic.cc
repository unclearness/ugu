/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include <stdio.h>

#include <fstream>

#include "ugu/geodesic/geodesic.h"

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  std::string data_dir = "../data/bunny/";
  std::string in_obj_path = data_dir + "bunny.obj";

  ugu::Mesh mesh;

  mesh.LoadObj(in_obj_path, data_dir);

  int src_vid = 0;
  Eigen::SparseMatrix<float> edge_dists;
  std::vector<double> dists;
  std::vector<int> min_path_edges;
  ugu::ComputeGeodesicDistance(mesh, src_vid, edge_dists, dists,
                               min_path_edges);

  auto max_dist = *std::max_element(dists.begin(), dists.end());
  int contour_line_num = 10;
  double contour_line_step = max_dist / contour_line_num;
  double contour_line_width = contour_line_step * 0.15;
  std::vector<Eigen::Vector3f> vertex_colors;
  for (const auto& d : dists) {
    bool in_contour_line = false;
    for (int i = 0; i < contour_line_num; i++) {
      double contour_min = i * contour_line_step - contour_line_width;
      double contour_max = i * contour_line_step + contour_line_width;
      if (contour_min < d && d < contour_max) {
        in_contour_line = true;
      }
    }

    if (in_contour_line) {
      vertex_colors.push_back({0.0, 0.0, 255.0});
      continue;
    }

    vertex_colors.push_back(
        {static_cast<float>(255.0 * d / max_dist), 0.0, 0.0});
  }

  mesh.set_vertex_colors(vertex_colors);
  mesh.WritePly(data_dir + "bunny_geodesic_distance.ply");

  return 0;
}
