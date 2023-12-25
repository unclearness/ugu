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
    normalized_curvature[i] = curvature[i] * area[i];
    if (normalized_curvature[i] < min_c) {
      min_c = normalized_curvature[i];
    }
    if (max_c < normalized_curvature[i]) {
      max_c = normalized_curvature[i];
    }
  }
  if (std::abs(min_c) < max_c) {
    max_c = std::abs(min_c);
  }
  min_c = -3.f;
  max_c = 3.f;
  for (auto& c : normalized_curvature) {
    c = std::clamp((c - min_c) / (max_c - min_c), 0.f, 1.f);
  }

  std::vector<Eigen::Vector3f> colors = Colorize(normalized_curvature);

  mesh->set_vertex_colors(colors);

  mesh->WriteObj("../data_out/ex28.obj");

  return 0;
}
