/*
 * Copyright (C) 2023, unclearness
 * All rights reserved.
 */

#include "ugu/curvature/curvature.h"
#include "ugu/mesh.h"
#include "ugu/timer.h"
#include "ugu/util/image_util.h"

using namespace ugu;

int main() {
  Timer<> timer;

  MeshPtr mesh = Mesh::Create();
  mesh->LoadObj("../data/bunny/bunny.obj");

  std::vector<float> curvature;
  std::vector<Eigen::Vector3f> internal_angles;
  CurvatureGaussian(mesh->vertices(), mesh->vertex_indices(), curvature,
                    internal_angles);
  std::vector<float> area =
      BarycentricCellArea(mesh->vertices(), mesh->vertex_indices());
  std::vector<float> normalized_curvature(curvature.size());
  float max_c = std::numeric_limits<float>::lowest();
  float min_c = std::numeric_limits<float>::max();
  for (size_t i = 0; i < curvature.size(); i++) {
    normalized_curvature[i] = curvature[i] / (std::max(area[i], 1e-10f));
    if (normalized_curvature[i] < min_c) {
      min_c = normalized_curvature[i];
    }
    if (max_c < normalized_curvature[i]) {
      max_c = normalized_curvature[i];
    }
  }
  std::cout << "max/min: " << max_c << "/" << min_c << std::endl;
  std::vector<float> sorted_normalized_curvature = normalized_curvature;
  float r = 0.1f;
  std::sort(sorted_normalized_curvature.begin(),
            sorted_normalized_curvature.end());
  max_c = sorted_normalized_curvature[static_cast<size_t>(
      (1.f - r) * sorted_normalized_curvature.size())];
  min_c = sorted_normalized_curvature[static_cast<size_t>(
      r * sorted_normalized_curvature.size())];
  std::cout << "90%/10%: " << max_c << " " << min_c << std::endl;
  for (auto& c : normalized_curvature) {
    c = std::clamp((c - min_c) / (max_c - min_c), 0.f, 1.f);
  }

  std::vector<Eigen::Vector3f> colors = Colorize(normalized_curvature);

  mesh->set_vertex_colors(colors);

  mesh->WriteObj("../data_out/ex28.obj");

  return 0;
}
