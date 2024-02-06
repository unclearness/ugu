/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include <iostream>

#include "ugu/image_io.h"
#include "ugu/inpaint/inpaint.h"
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

  options.add_options()("s,src", "Src mesh (.obj)",
                        cxxopts::value<std::string>())(
      "o,out", "Output posmap (.exr or .bin)", cxxopts::value<std::string>())(
      "width", "Output texture width",
      cxxopts::value<int>()->default_value("1024"))(
      "height", "Output texture height",
      cxxopts::value<int>()->default_value("1024"))(
      "i,inpaint", "Enable inpaint",
      cxxopts::value<bool>()->default_value("false"))("h,help", "Print usage");

  options.parse_positional({"src", "out"});

  auto result = options.parse(argc, argv);

  if (result.count("help") > 0 || result.count("src") < 1 ||
      result.count("out") < 1) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  const std::string& src_obj_path = result["src"].as<std::string>();
  const std::string& out_path = result["out"].as<std::string>();
  int width = result["width"].as<int>();
  int height = result["height"].as<int>();
  bool to_inpaint = result["inpaint"].as<bool>();

  std::string ext = ugu::ExtractExt(out_path);
#ifdef UGU_USE_OPENCV
  if (ext != "exr") {
    std::cout << "only .exr is supported" << std::endl;
    return 1;
  }
#else
  if (ext != "bin") {
    std::cout << "only .bin is supported" << std::endl;
    return 1;
  }
#endif

  ugu::Timer<> timer;
  ugu::Mesh src_mesh;

  src_mesh.LoadObj(src_obj_path);

  ugu::Image3f posmap;
  ugu::Image1b mask = ugu::Image1b::zeros(width, height);
  ugu::RasterizeVertexAttributeToTexture(
      src_mesh.vertices(), src_mesh.vertex_indices(), src_mesh.uv(),
      src_mesh.uv_indices(), posmap, width, height, &mask);

  if (to_inpaint) {
    ugu::Image1b inv_mask;
    ugu::Not(mask, &inv_mask);
    ugu::Inpaint(inv_mask, posmap, 3.f, ugu::InpaintMethod::NAIVE);
  }

  ugu::imwrite(out_path, posmap);

  return 0;
}