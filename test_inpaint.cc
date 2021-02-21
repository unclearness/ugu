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

  std::string data_dir = "../data/inpaint/";
  std::string color_path = data_dir + "fruits.jpg";
  std::string mask_path = data_dir + "fruits_scrabble.png";

  ugu::Image1b mask = ugu::imread<ugu::Image1b>(mask_path);
  ugu::Image3b color = ugu::imread<ugu::Image3b>(color_path);
  ugu::Image3b color_gt = color;
  ugu::Image3b color_scrabbled = color;

  // FMM
  ugu::Image1f fmm_dist;
  ugu::FastMarchingMethod(mask, fmm_dist);
  ugu::Image1b vis_fmm_dist;
  ugu::Depth2Gray(fmm_dist, &vis_fmm_dist, 0.f, 10.f);
  ugu::imwrite(data_dir + "00000_fmm_dist.png", vis_fmm_dist);

  for (int j = 0; j < color.rows; j++) {
    for (int i = 0; i < color.cols; i++) {
      if (mask.at<unsigned char>(j, i) == 255) {
        auto& c = color_scrabbled.at<ugu::Vec3b>(j, i);
        c[0] = 0;
        c[1] = 0;
        c[2] = 0;
      }
    }
  }

  // Inpaint
  ugu::imwrite(data_dir + "fruits_scrabbled.png", color_scrabbled);
  ugu::Image3b tmp = color_scrabbled;
  ugu::Inpaint(mask, tmp, 5.f, ugu::InpaintMethod::TELEA);
  ugu::imwrite(data_dir + "fruits_inpainted_telea.png", tmp);

  tmp = color_scrabbled;
  ugu::Inpaint(mask, color, 5.f, ugu::InpaintMethod::NAIVE);
  ugu::imwrite(data_dir + "fruits_inpainted_naive.png", color);

  return 0;
}
