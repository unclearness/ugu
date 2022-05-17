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
#include "ugu/util/string_util.h"

int main(int argc, char* argv[]) {
  cxxopts::Options options("gif3d", "tmp");
  options.positional_help("[optional args]").show_positional_help();

  options.add_options()("s,src", "Src directory that contains images (.png, .jpg, .jpeg)",
                        cxxopts::value<std::string>())(
      "o,out", "Output directory",
      cxxopts::value<std::string>()->default_value("./"))(
      "b,base", "Output basename",
      cxxopts::value<std::string>()->default_value("out"))(
      "t,time", "Time (ms) per an image",
      cxxopts::value<int>()->default_value("100"))(
      "w,width", "Output width",
      cxxopts::value<int>()->default_value("256"))(
      "scale", "Output scale of width",
      cxxopts::value<float>()->default_value("1.0"))(
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
  int width = result["width"].as<int>();
  int time = result["time"].as<int>();
  float scale = result["scale"].as<float>();
  // bool verbose = result["verbose"].as<bool>();

  ugu::Timer<> timer;

  // Load images

  // Make a set of meshes

  // Output

  return 0;
}