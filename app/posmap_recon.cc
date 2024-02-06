/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include <iostream>

#include "ugu/image_io.h"
#include "ugu/mesh.h"
#include "ugu/timer.h"
#include "ugu/util/raster_util.h"
#include "ugu/util/string_util.h"

#ifdef _WIN32
#pragma warning(push, 0)
#endif
#include "cxxopts.hpp"
#ifdef _WIN32
#pragma warning(pop)
#endif

int main(int argc, char* argv[]) {
  cxxopts::Options options("textrans",
                           "Texture transfer for almost aligned meshes");
  options.positional_help("[optional args]").show_positional_help();

  options.add_options()("s,src", "Src posmap (.exr or .bin)",
                        cxxopts::value<std::string>())(
      "m,mesh", "Src mesh that has indices(.obj)",
      cxxopts::value<std::string>())("o,out", "Output mesh (.obj)",
                                     cxxopts::value<std::string>())(
      "h,help", "Print usage");

  options.parse_positional({"src", "mesh", "out"});

  auto result = options.parse(argc, argv);

  if (result.count("help") > 0 || result.count("src") < 1 ||
      result.count("mesh") < 1 || result.count("out") < 1) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  const std::string& src_posmap_path = result["src"].as<std::string>();
  const std::string& src_mesh_path = result["mesh"].as<std::string>();
  const std::string& out_path = result["out"].as<std::string>();

  std::string ext = ugu::ExtractExt(src_posmap_path);
#ifdef UGU_USE_OPENCV
  if (ext != "exr") {
    std::cout << "only .exr is supported" << std::endl;
    return 1;
  }
#else
  {
    std::cout << "Not supported" << std::endl;
    return 1;
    // if (ext != "bin") {
    //   std::cout << "only .bin is supported" << std::endl;
    //   return 1;
    // }
  }
#endif

  ugu::Image3f posmap = ugu::imread(src_posmap_path, -1);

  ugu::Timer<> timer;
  ugu::Mesh src_mesh;

  src_mesh.LoadObj(src_mesh_path);

  std::vector<Eigen::Vector3f> vertices(src_mesh.vertices().size());

  for (size_t fid = 0; fid < src_mesh.vertex_indices().size(); fid++) {
    for (int j = 0; j < 3; j++) {
      auto vid = src_mesh.vertex_indices()[fid][j];
      auto uvid = src_mesh.uv_indices()[fid][j];
      auto x = ugu::U2X(src_mesh.uv()[uvid].x(), posmap.cols);
      auto y = ugu::V2Y(src_mesh.uv()[uvid].y(), posmap.rows);

      auto val = ugu::BilinearInterpolation(x, y, posmap);

      vertices[vid][0] = val[0];
      vertices[vid][1] = val[1];
      vertices[vid][2] = val[2];
    }
  }

  src_mesh.set_vertices(vertices);

  src_mesh.WriteObj(out_path);

  return 0;
}