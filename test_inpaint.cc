/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include <stdio.h>

#include <fstream>

#include "ugu/inpaint/inpaint.h"

// test by bunny data with 6 views
int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  std::string data_dir = "../data/bunny/";
  std::string mask_path = data_dir + "00000_mask.png";

  ugu::Image1b mask = ugu::imread<ugu::Image1b>(mask_path);

  // FMM
  ugu::Image1f fmm_dist;
  ugu::FastMarchingMethod(mask, &fmm_dist);
  ugu::Image1b vis_fmm_dist;
  ugu::Depth2Gray(fmm_dist, &vis_fmm_dist, 0.f, 100.f);
  ugu::imwrite(data_dir + "00000_fmm_dist.png", vis_fmm_dist);

  return 0;
}
