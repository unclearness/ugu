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
  std::string mask_path = data_dir + "circle.png";

  ugu::Image1b mask = ugu::imread<ugu::Image1b>(mask_path);

#if 0
				  // 2D SDF
  ugu::Image1f sdf;
  ugu::MakeSignedDistanceField(mask, &sdf, true, false, -1.0f);
  ugu::Image3b vis_sdf;
  ugu::SignedDistance2Color(sdf, &vis_sdf, -1.0f, 1.0f);
  ugu::imwrite(data_dir + "00000_sdf.png", vis_sdf);
#endif  // 0


  // FMM
  ugu::Image1f fmm_dist;
  ugu::FastMarchingMethod(mask, &fmm_dist);
  ugu::Image1b vis_fmm_dist;
  ugu::Depth2Gray(fmm_dist, &vis_fmm_dist, 0.f, 50.f);
  ugu::imwrite(data_dir + "00000_fmm_dist.png", vis_fmm_dist);

  return 0;
}
