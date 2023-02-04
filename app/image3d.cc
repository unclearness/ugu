/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include <algorithm>
#include <iostream>

#ifdef _WIN32
#pragma warning(push, 0)
#endif
#include "cxxopts.hpp"
#ifdef _WIN32
#pragma warning(pop)
#endif

#include "ugu/image_io.h"
#include "ugu/image_proc.h"
#include "ugu/mesh.h"
#include "ugu/timer.h"
#include "ugu/util/geom_util.h"
#include "ugu/util/image_util.h"
#include "ugu/util/path_util.h"
#include "ugu/util/string_util.h"

namespace {

void LoadMask(const std::string& path, ugu::Image1b& mask) {
  if (path.empty()) {
    return;
  }
  auto ext = ugu::ExtractExt(path);
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [&](const char c) { return static_cast<char>(::tolower(c)); });
  if (ext != "jpg" && ext != "jpeg" && ext != "png") {
    ugu::LOGW("mask must be .jpg, .jpeg or .png but %s\n", path.c_str());
    return;
  }

  ugu::Image3b tmp3b;
  ugu::Image4b tmp4b;
  std::vector<ugu::Image1b> planes;
  if (ext == "png") {
    mask = ugu::Imread<ugu::Image1b>(path);
    if (mask.empty()) {
      tmp3b = ugu::Imread<ugu::Image3b>(path);
      if (tmp3b.empty()) {
        tmp4b = ugu::Imread<ugu::Image4b>(path);
        if (tmp4b.empty()) {
          return;
        } else {
          ugu::Split(tmp4b, planes);
        }
      } else {
        ugu::Split(tmp3b, planes);
      }
    }
  } else {
    // jpeg is always 3 channel
    tmp3b = ugu::Imread<ugu::Image3b>(path);
    ugu::Split(tmp3b, planes);
  }

  if (mask.empty()) {
    mask = planes[0];
  }
}

}  // namespace

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
      "scale_width",
      "Output scale of width. < 0 will be ignored and automatically calculate "
      "from scale_height",
      cxxopts::value<float>()->default_value("1.0"))(
      "scale_height",
      "Output scale of height. < 0 will be ignored and automatically calculate "
      "from scale_width",
      cxxopts::value<float>()->default_value("-1.0"))(
      "m,mask", "Foreground mask path",
      cxxopts::value<std::string>()->default_value(""))(
      "gltf", "GLTF output", cxxopts::value<bool>()->default_value("false"))(
      "t,threads", "#Threads", cxxopts::value<int>()->default_value("-1"))(
      "v,verbose", "Verbose", cxxopts::value<bool>()->default_value("false"))(
      "no_enlarge_tex",
      "Texutre scale is applied only if actual texture size is smaller than "
      "specified size",
      cxxopts::value<bool>()->default_value("false"))(
      "unlit", "Unlit material",
      cxxopts::value<bool>()->default_value("false"))("h,help", "Print usage");

  options.parse_positional({"src"});

  auto result = options.parse(argc, argv);

  if (result.count("help") > 0 || result.count("src") < 1) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  const std::string src_path = result["src"].as<std::string>();
  const std::string& basename = result["base"].as<std::string>();
  const std::string& out_dir = result["out"].as<std::string>();
  int width = result["width"].as<int>();
  float scale_width = result["scale_width"].as<float>();
  float scale_height = result["scale_height"].as<float>();
  bool is_glb = !result["gltf"].as<bool>();
  bool verbose = result["verbose"].as<bool>();
  const std::string mask_path = result["mask"].as<std::string>();
  int threads_num = result["threads"].as<int>();
  bool no_enlarge_tex = result["no_enlarge_tex"].as<bool>();
  bool is_unlit = result["unlit"].as<bool>();

  if (threads_num > 0) {
    ugu::UGU_THREADS_NUM = threads_num;
  }

  ugu::Timer<> timer, timer_total;

  auto print_time = [verbose](ugu::Timer<>& timer, const std::string& name) {
    if (!verbose) {
      return;
    }
    timer.End();
    ugu::LOGI("%s: %f ms\n", name.c_str(), timer.elapsed_msec());
    timer.Start();
  };

  timer_total.Start();
  timer.Start();

  if (!ugu::FileExists(src_path)) {
    ugu::LOGE("src does not exist: %s\n", src_path.c_str());
    return 1;
  }

  auto ext = ugu::ExtractExt(src_path);
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [&](const char c) { return static_cast<char>(::tolower(c)); });
  if (ext != "jpg" && ext != "jpeg" && ext != "png") {
    ugu::LOGE("src must be .jpg, .jpeg or .png but %s\n", src_path.c_str());
    return 1;
  }

  ugu::Image3b image;
  ugu::Image4b alpha_image;
  ugu::Image1b mask;
  int w = -1;
  int h = -1;
  //int c = -1;
  if (ext == "png") {
    image = ugu::Imread<ugu::Image3b>(src_path);
    w = image.cols;
    h = image.rows;
    //c = image.channels();
    if (image.empty()) {
      alpha_image = ugu::Imread<ugu::Image4b>(src_path);
      w = alpha_image.cols;
      h = alpha_image.rows;
      //c = alpha_image.channels();
    }
    if (image.empty() && alpha_image.empty()) {
      ugu::Image1b gray_image = ugu::Imread<ugu::Image1b>(src_path);
      if (!gray_image.empty()) {
        image = ugu::Merge(gray_image, gray_image, gray_image);
        w = image.cols;
        h = image.rows;
        //c = image.channels();
      }
    }
  } else {
    image = ugu::Imread<ugu::Image3b>(src_path);
    w = image.cols;
    h = image.rows;
    //c = image.channels();
    if (image.empty()) {
      ugu::Image1b gray_image = ugu::Imread<ugu::Image1b>(src_path);
      if (!gray_image.empty()) {
        image = ugu::Merge(gray_image, gray_image, gray_image);
        w = image.cols;
        h = image.rows;
        //c = image.channels();
      }
    }
  }

  if (w < 0) {
    ugu::LOGE("Failed to load src: %s\n", src_path.c_str());
    return 1;
  }

  if (mask.empty() && !mask_path.empty()) {
    LoadMask(mask_path, mask);
  }

  bool to_resize = width > 0 && w != width;
  if (to_resize && no_enlarge_tex) {
    to_resize = width < w;
  }

  if (to_resize) {
    ugu::Size dsize;
    dsize.width = width;
    float r = static_cast<float>(h) / static_cast<float>(w);
    dsize.height = static_cast<int>(r * width);
    if (!image.empty()) {
      ugu::resize(image.clone(), image, dsize);
    }
    if (!mask.empty()) {
      ugu::resize(mask.clone(), mask, dsize);
    }
    if (!alpha_image.empty()) {
      ugu::resize(alpha_image.clone(), alpha_image, dsize);
    }
  }
  print_time(timer, "image load");

  ugu::MeshPtr mesh = nullptr;
  if (!image.empty()) {
    mesh = ugu::MakeTexturedPlane(image, scale_width, scale_height);
  } else if (!alpha_image.empty()) {
    mesh = ugu::MakeTexturedPlane(alpha_image, scale_width, scale_height);
  }
  // Align bottum
  const auto& stats = mesh->stats();
  mesh->Translate({0.f, -stats.bb_min.y(), 0.f});

  print_time(timer, "mesh generation");

  // For gltf generation
  auto mats = mesh->materials();
  bool with_alpha = !alpha_image.empty();

  if (with_alpha) {
    mats[0].with_alpha_tex = alpha_image;
    mats[0].with_alpha_texname = "with_alpha_tex.png";
    mats[0].with_alpha_texpath = out_dir + "/with_alpha.png";

    mats[0].with_alpha_compressed = ugu::PngData(mats[0].with_alpha_tex);
  } else {
    mats[0].diffuse_texpath = src_path;

    if (ext == "png") {
      mats[0].diffuse_compressed = ugu::PngData(mats[0].diffuse_tex);
    } else {
      mats[0].diffuse_compressed = ugu::JpgData(mats[0].diffuse_tex);
    }
  }
  print_time(timer, "texture process");

  mesh->set_single_material(mats[0]);

  print_time(timer, "material process");

  if (is_glb) {
    mesh->WriteGlb(out_dir, basename + ".glb", is_unlit);
  } else {
    mesh->WriteGltfSeparate(out_dir, basename, is_unlit);
  }

  print_time(timer, "write gltf");

  print_time(timer_total, "total");

  return 0;
}