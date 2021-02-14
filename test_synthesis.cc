/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include <stdio.h>
#include <fstream>

#include "ugu/image.h"
#include "ugu/synthesis/bdsim.h"

// test by bunny data with 6 views
int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  std::string data_dir = "../data/synthesis/";
  std::string data_name = "217_s.png";
  ugu::Image3b src = ugu::imread<ugu::Image3b>(data_dir + data_name);
  ugu::Image3b dst;

  ugu::BdsimParams params;
  params.verbose = true;
  params.iteration_in_scale = 10;
  params.target_size.height = src.rows * 2;
  params.target_size.width = src.cols * 2;
  ugu::Synthesize(src, dst, params);

  ugu::imwrite("out.png", dst);

  return 0;
}
