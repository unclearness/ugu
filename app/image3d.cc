/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include <algorithm>
#include <iostream>

#include "cxxopts.hpp"
#include "ugu/mesh.h"
#include "ugu/timer.h"
#include "ugu/util/geom_util.h"
#include "ugu/util/image_util.h"
#include "ugu/util/path_util.h"
#include "ugu/util/string_util.h"

int main(int argc, char* argv[]) {
  cxxopts::Options options("image3d", "tmp");
  options.positional_help("[optional args]").show_positional_help();

  options.add_options()("s,src", "Src image path (.png, .jpg, .jpeg)",
                        cxxopts::value<std::string>())(
      "o,out", "Output directory",
      cxxopts::value<std::string>()->default_value("./"))(
      "b,base", "Output basename",
      cxxopts::value<std::string>()->default_value("out"))(
      "w,width", "Output texture width",
      cxxopts::value<int>()->default_value("-1"))(
      "scale", "Output scale of width",
      cxxopts::value<float>()->default_value("1.0"))(
      "glb", "GLB output", cxxopts::value<bool>()->default_value("true"))(
      "v,verbose", "Verbose", cxxopts::value<bool>()->default_value("false"))(
      "h,help", "Print usage");

  options.parse_positional({"src"});

  auto result = options.parse(argc, argv);

  if (result.count("help") > 0 || result.count("src") < 1) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  std::string src_path = result["src"].as<std::string>();
  const std::string& basename = result["base"].as<std::string>();
  const std::string& out_dir = result["out"].as<std::string>();
  int width = result["width"].as<int>();
  float scale = result["scale"].as<float>();
  bool is_glb = result["glb"].as<bool>();
  bool verbose = result["verbose"].as<bool>();

  ugu::Timer<> timer;

  auto print_time = [&](const std::string& name) {
    if (!verbose) {
      return;
    }
    timer.End();
    ugu::LOGI("%s: %f ms\n", name.c_str(), timer.elapsed_msec());
    timer.Start();
  };

  timer.Start();

  if (!ugu::FileExists(src_path)) {
    ugu::LOGE("src does not exist: %s\n", src_path.c_str());
    return 1;
  }

  auto ext = ugu::ExtractExt(src_path);
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  if (ext != "jpg" && ext != "jpeg" && ext != "png") {
    ugu::LOGE("src must be .jpg, .jpeg or .png but %s\n", src_path.c_str());
    return 1;
  }

  ugu::Image3b image;

  if (ext == "png") {
    image = ugu::imread<ugu::Image3b>(src_path);
    if (image.empty()) {
      auto image4b = ugu::imread<ugu::Image4b>(src_path);
      if (!image4b.empty()) {
        ugu::AlignChannels(image4b, image);
      }
    }
  } else {
    // jpeg is always 3 channel
    image = ugu::imread<ugu::Image3b>(src_path);
  }

  if (image.empty()) {
    ugu::LOGE("Failed to load src: %s\n", src_path.c_str());
    return 1;
  }

  bool to_resize = width > 0 && image.cols != width;
  if (to_resize) {
    ugu::Size dsize;
    dsize.width = width;
    float r = static_cast<float>(image.rows) / static_cast<float>(image.cols);
    dsize.height = static_cast<int>(r * width);
    ugu::resize(image.clone(), image, dsize);
    // Write tmp image & update path
    std::string tmp_path = out_dir + "/tmp.png";
    src_path = tmp_path;
    ugu::imwrite(src_path, image);
  }
  print_time("image load");

  auto mesh = ugu::MakeTexturedPlane(image, scale);

  print_time("mesh generation");

  // For gltf generation
  auto mats = mesh->materials();
  mats[0].diffuse_texpath = src_path;
  mesh->set_single_material(mats[0]);

  if (is_glb) {
    mesh->WriteGlb(out_dir, basename + ".glb");
  } else {
    mesh->WriteGltfSeparate(out_dir, basename);
  }

  if (to_resize) {
    ugu::RmFile(src_path);
  }

  print_time("write gltf");

  return 0;
}