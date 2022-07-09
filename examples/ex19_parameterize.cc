/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/parameterize/parameterize.h"

namespace {}  // namespace

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;
  // std::string data_dir = "../data/sphere/";
  // std::string obj_path = data_dir + "icosphere3_smart_uv.obj";

  std::string data_dir = "../data/bunny/";
  std::string obj_path = data_dir + "bunny.obj";

  // load mesh
  ugu::MeshPtr input_mesh = ugu::Mesh::Create();
  input_mesh->LoadObj(obj_path, data_dir);

  ugu::Parameterize(*input_mesh, 512, 512, ugu::ParameterizeUvType::kSmartUv);

  return 0;
  {
    std::vector<Eigen::Vector2f> points_2d;
    ugu::OrthoProjectToXY(Eigen::Vector3f(0.f, 0.f, 1.f),
                          input_mesh->vertices(), points_2d, false, false,
                          false);

    std::vector<Eigen::Vector3f> points_3d;
    for (const auto& p2d : points_2d) {
      points_3d.push_back(Eigen::Vector3f(p2d[0], p2d[1], 0.f));
    }
    auto out_mesh = ugu::Mesh(*input_mesh);

    out_mesh.set_vertices(points_3d);
    out_mesh.WriteObj(data_dir, "bunny_projected_xy");
  }

  return 0;
}
