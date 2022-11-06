/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#if __has_include("../third_party/nanopm/nanopm.h")
#include <stdio.h>

#include <fstream>
#include <iostream>

#include "ugu/image.h"
#include "ugu/image_io.h"
#include "ugu/synthesis/bdsim.h"

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  std::string data_dir = "../data/synthesis/";
  std::string data_name = "217_s.png";
  ugu::Image3b src = ugu::Imread<ugu::Image3b>(data_dir + data_name);
  ugu::Image3b dst;

  ugu::BdsimParams params;
  params.patch_size = 11;
  params.verbose = true;
  params.debug_dir = "./";
  params.target_size.height = static_cast<int>(src.rows * 0.4);
  params.target_size.width = static_cast<int>(src.cols * 0.4);

  ugu::Synthesize(src, dst, params);

  ugu::imwrite("out.png", dst);

  return 0;
}
#else
int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;
  return 0;
}
#endif