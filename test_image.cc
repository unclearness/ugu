/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include <stdio.h>

#include <fstream>

#include "ugu/image.h"
#include "ugu/inflation/inflation.h"
#include "ugu/util.h"
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

  // Inflation
  ugu::Image1f height;
  ugu::Inflation(mask, height, true);
  ugu::Image1b vis_height;
  ugu::Depth2Gray(height, &vis_height, 0.f, 300.f);
  ugu::imwrite(data_dir + "00000_height.png", vis_height);

  std::shared_ptr<ugu::OrthoCamera> camera =
      std::make_shared<ugu::OrthoCamera>();
  camera->set_size(mask.cols, mask.rows);
  camera->set_c2w(Eigen::Affine3d::Identity());

  ugu::Mesh mesh;
  ugu::Depth2Mesh(height, *camera, &mesh, 999999.9f);
  mesh.WritePly(data_dir + "00000_height_mesh.ply");

  return 0;
}
