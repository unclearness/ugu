/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include <iostream>

#include "cxxopts.hpp"
#include "ugu/inpaint/inpaint.h"
#include "ugu/textrans/texture_transfer.h"
#include "ugu/texturing/texture_mapper.h"
#include "ugu/timer.h"
#include "ugu/util/raster_util.h"

namespace {

std::string Path2Dir(const std::string& path) {
  std::string::size_type pos = std::string::npos;
  const std::string::size_type unix_pos = path.find_last_of('/');
  const std::string::size_type windows_pos = path.find_last_of('\\');
  if (unix_pos != std::string::npos) {
    if (pos == std::string::npos) {
      pos = unix_pos;
    } else {
      pos = std::max(pos, unix_pos);
    }
  }
  if (windows_pos != std::string::npos) {
    if (pos == std::string::npos) {
      pos = windows_pos;
    } else {
      pos = std::max(pos, unix_pos);
    }
  }
  return (pos == std::string::npos) ? "./" : path.substr(0, pos + 1);
}

std::string GetExt(const std::string& path) {
  size_t ext_i = path.find_last_of(".");
  std::string extname = path.substr(ext_i, path.size() - ext_i);
  return extname;
}

}  // namespace

int main(int argc, char* argv[]) {
  cxxopts::Options options(
      "vc2tex", "Vertex color to texture with simple parameterization");
  options.positional_help("[optional args]").show_positional_help();

  options.add_options()("s,src", "Src mesh with vertex color (.ply)",
                        cxxopts::value<std::string>())(
      "o,out", "Output directory",
      cxxopts::value<std::string>()->default_value("./"))(
      "b,base", "Output basename",
      cxxopts::value<std::string>()->default_value("out"))(
      "width", "Output texture width",
      cxxopts::value<int>()->default_value("1024"))(
      "height", "Output texture height",
      cxxopts::value<int>()->default_value("1024"))(
      "i,inpaint", "Enable inpaint",
      cxxopts::value<bool>()->default_value("true"))(
      "v,verbose", "Verbose", cxxopts::value<bool>()->default_value("false"))(
      "h,help", "Print usage");

  options.parse_positional({"src", "dst"});

  auto result = options.parse(argc, argv);

  if (result.count("help") > 0 || result.count("src") < 1) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  const std::string& src_ply_path = result["src"].as<std::string>();
  const std::string& basename = result["base"].as<std::string>();
  const std::string& out_dir = result["out"].as<std::string>();
  int width = result["width"].as<int>();
  int height = result["height"].as<int>();
  bool inpaint = result["inpaint"].as<bool>();
  bool verbose = result["verbose"].as<bool>();

  ugu::Timer<> timer;
  ugu::Mesh src_mesh, dst_mesh;

  src_mesh.LoadPly(src_ply_path);

  std::string out_basename = out_dir + "/" + basename;

  ugu::Parameterize(src_mesh);

  ugu::Image1b mask = ugu::Image1b::zeros(height, width);
  auto rasterized = ugu::Image3b::zeros(height, width);
  ugu::RasterizeVertexAttributeToTexture(
      src_mesh.vertex_colors(), src_mesh.vertex_indices(), src_mesh.uv(),
      src_mesh.uv_indices(), rasterized, width, height, &mask);
  if (inpaint) {
    ugu::Image1b inv_mask = mask.clone();
    ugu::Not(mask, &inv_mask);
    ugu::Inpaint(inv_mask, rasterized, 3.f, ugu::InpaintMethod::TELEA);
  }
  std::vector<ugu::ObjMaterial> mat = {ugu::ObjMaterial()};
  mat[0].diffuse_tex = rasterized;
  mat[0].diffuse_texname = basename + ".jpg";

  src_mesh.set_materials(mat);

  std::vector<int> material_ids(src_mesh.uv_indices().size(), 0);
  src_mesh.set_material_ids(material_ids);

  src_mesh.WriteObj(out_dir, basename);

  return 0;
}