/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include <stdio.h>

#include <fstream>

#include "ugu/image.h"
#include "ugu/util/image_util.h"

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

  ugu::circle(vis_sdf, {200, 200}, 20, {255, 0, 255}, 3);
  ugu::circle(vis_sdf, {100, 100}, 10, {0, 0, 0}, -1);
  ugu::line(vis_sdf, {0, 0}, {50, 50}, {255, 0, 0}, 1);
  ugu::line(vis_sdf, {10, 200}, {100, 200}, {0, 0, 255}, 5);
  ugu::imwrite(data_dir + "00000_sdf_circle.png", vis_sdf);

  return 0;
}
