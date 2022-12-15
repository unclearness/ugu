/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include <stdio.h>

#include <fstream>

#include "ugu/image.h"
#include "ugu/image_io.h"
#include "ugu/image_proc.h"
#include "ugu/util/image_util.h"

// test by bunny data with 6 views
int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  std::string data_dir = "../data/bunny/";
  std::string mask_path = data_dir + "00000_mask.png";

  ugu::Image1b mask = ugu::Imread<ugu::Image1b>(mask_path, -1);

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

  // GIF load
  auto [images, delays] = ugu::LoadGif("../data/gif/dancing.gif");
  for (size_t i = 0; i < images.size(); i++) {
    ugu::imwrite("../data/gif/" + std::to_string(i) + "_" +
                     std::to_string(delays[i]) + "ms.png",
                 images[i]);
  }

  {
    ugu::ImageBase refer =
        ugu::imread("../data/color_transfer/reference_00.jpg");
    ugu::ImageBase target = ugu::imread("../data/color_transfer/target_00.jpg");
    ugu::Image3b res = ugu::ColorTransfer(refer, target);
    ugu::imwrite("../data/color_transfer/result_00.jpg", res);
  }

  {
    ugu::ImageBase source =
        ugu::imread("../data/poisson_blending/source.png");
    ugu::ImageBase target = ugu::imread("../data/poisson_blending/target.png");
    ugu::ImageBase mask = ugu::imread("../data/poisson_blending/mask.png");
    ugu::Image3b res = ugu::PoissonBlend(mask, source, target, -35, 35);
    ugu::imwrite("../data/poisson_blending/result.png", res);

  }

  return 0;
}
