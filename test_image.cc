/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include <stdio.h>

#include <fstream>

#include "ugu/image.h"

// test by bunny data with 6 views
int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  std::string data_dir = "../data/bunny/";
  std::string mask_path = data_dir + "00000_mask.png";

  ugu::Image1b mask = ugu::imread<ugu::Image1b>(mask_path);

  // 2D SDF
  ugu::Image1f sdf;
  ugu::MakeSignedDistanceField(mask, &sdf, true, false, -1.0f);
  ugu::Image3b vis_sdf;
  ugu::SignedDistance2Color(sdf, &vis_sdf, -1.0f, 1.0f);
  ugu::imwrite(data_dir + "00000_sdf.png", vis_sdf);

  return 0;
}
