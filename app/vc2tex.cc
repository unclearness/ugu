/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include <iostream>

#include "ugu/inpaint/inpaint.h"
#include "ugu/parameterize/parameterize.h"
#include "ugu/textrans/texture_transfer.h"
#include "ugu/timer.h"
#include "ugu/util/path_util.h"
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
  ugu::Timer<> timer;
  timer.Start();

  cxxopts::Options options("vc2tex",
                           "UV parameterization including rasteriztaion from "
                           "vertex color to texture");
  options.positional_help("[optional args]").show_positional_help();

  options.add_options()("s,src", "Src mesh with vertex color (.ply or .obj)",
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
      "t,type", "Output UV type (smart or tri)",
      cxxopts::value<std::string>()->default_value("smart"))(
      "v,verbose", "Verbose", cxxopts::value<bool>()->default_value("false"))(
      "h,help", "Print usage");

  options.parse_positional({"src"});

  auto result = options.parse(argc, argv);

  if (result.count("help") > 0 || result.count("src") < 1) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  const std::string& src_path = result["src"].as<std::string>();
  const std::string& basename = result["base"].as<std::string>();
  const std::string& out_dir = result["out"].as<std::string>();
  const std::string& uv_type_str = result["type"].as<std::string>();
  int width = result["width"].as<int>();
  int height = result["height"].as<int>();
  bool inpaint = result["inpaint"].as<bool>();
  bool verbose = result["verbose"].as<bool>();
  bool with_vc = true;

  if (verbose) {
    ugu::set_log_level(ugu::LogLevel::kVerbose);
  } else {
    ugu::set_log_level(ugu::LogLevel::kError);
  }

  ugu::Mesh src_mesh, dst_mesh;

  std::string ext = ugu::ExtractExt(src_path);
  if (ext == "ply") {
    src_mesh.LoadPly(src_path);
  } else if (ext == "obj") {
    std::string mtl_dir = ugu::ExtractDir(src_path);
    src_mesh.LoadObj(src_path, mtl_dir);

    // 0-1 to 0-255
    auto org_vc = src_mesh.vertex_colors();
    for (auto& c : org_vc) {
      for (int i = 0; i < 3; i++) {
        c[i] =
            static_cast<float>(ugu::saturate_cast<std::uint8_t>(c[i] * 255.f));
      }
    }
    src_mesh.set_vertex_colors(org_vc);

  } else {
    ugu::LOGE("src must be .ply or .obj\n");
    return 1;
  }

  if (uv_type_str != "smart" && uv_type_str != "tri") {
    ugu::LOGE("type must be smart or tri\n");
    return 1;
  }

  if (src_mesh.vertex_colors().empty()) {
    ugu::LOGW("src vertex color is empty\n");
    with_vc = false;
  }

  if (src_mesh.vertex_colors().size() != src_mesh.vertices().size()) {
    ugu::LOGW("#vertices %d mismatches #vertex color %d\n",
              src_mesh.vertices().size(), src_mesh.vertex_colors().size());
    with_vc = false;
  }

  std::string out_basename = out_dir + "/" + basename;
  timer.End();
  if (verbose) {
    ugu::LOGI("Preprocess %fms\n", timer.elapsed_msec());
  }

  timer.Start();
  ugu::ParameterizeUvType uv_type = ugu::ParameterizeUvType::kSmartUv;
  if (uv_type_str == "smart") {
    uv_type = ugu::ParameterizeUvType::kSmartUv;
  } else if (uv_type_str == "tri") {
    uv_type = ugu::ParameterizeUvType::kSimpleTriangles;
  }
  ugu::Parameterize(src_mesh, width, height, uv_type);
  timer.End();
  if (verbose) {
    ugu::LOGI("Parameterize %fms\n", timer.elapsed_msec());
  }

  if (with_vc) {
    timer.Start();
    ugu::Image1b mask = ugu::Image1b::zeros(height, width);
    ugu::Image3b rasterized = ugu::Image3b::zeros(height, width);
    ugu::RasterizeVertexAttributeToTexture(
        src_mesh.vertex_colors(), src_mesh.vertex_indices(), src_mesh.uv(),
        src_mesh.uv_indices(), rasterized, width, height, &mask);
    timer.End();
    if (verbose) {
      ugu::LOGI("Rasterize %fms\n", timer.elapsed_msec());
    }

    if (inpaint) {
      timer.Start();
      ugu::Image1b inv_mask = mask.clone();
      ugu::Not(mask, &inv_mask);
      ugu::Inpaint(inv_mask, rasterized, 3.f, ugu::InpaintMethod::TELEA);
      timer.End();
      if (verbose) {
        ugu::LOGI("Inpaint %fms\n", timer.elapsed_msec());
      }
    }

    std::vector<ugu::ObjMaterial> mat = {ugu::ObjMaterial()};
    mat[0].diffuse_tex = rasterized;
    std::string tex_ext = ".jpg";
    if (!src_mesh.materials().empty()) {
      std::string tmp_ext =
          ugu::ExtractExt(src_mesh.materials()[0].diffuse_texpath, false);
      if (tmp_ext.size() == 4) {
        tex_ext = tmp_ext;
      }
    }
    mat[0].diffuse_texname = basename + tex_ext;
    src_mesh.set_materials(mat);
  }

  timer.Start();
  std::vector<int> material_ids(src_mesh.uv_indices().size(), 0);
  src_mesh.set_material_ids(material_ids);
  ugu::EnsureDirExists(out_dir);
  src_mesh.WriteObj(out_dir, basename);
  timer.End();

  if (verbose) {
    ugu::LOGI("Writing %fms\n", timer.elapsed_msec());
  }

  return 0;
}