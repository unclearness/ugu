/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include <fstream>

#include "ugu/inflation/inflation.h"
#include "ugu/mesh.h"

// test by bunny data with 6 views
int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  std::string data_dir = "../data/bunny/";
  std::string mask_path = data_dir + "00000_mask.png";
  std::string color_path = data_dir + "00000_color.png";

  ugu::Image1b mask = ugu::imread<ugu::Image1b>(mask_path);
  // Inflation
  ugu::Image1f height;
  ugu::Mesh mesh;
  ugu::Inflation(mask, height, mesh);
  ugu::Image1b vis_height;
  ugu::Depth2Gray(height, &vis_height, 0.f, 300.f);
  ugu::imwrite(data_dir + "00000_height.png", vis_height);

  mesh.WritePly(data_dir + "00000_height.ply");

  ugu::Image3b color = ugu::imread<ugu::Image3b>(color_path);
  ugu::InflationParams params;
  params.texture = &color;
  ugu::Inflation(mask, height, mesh, params);
  mesh.WriteObj(data_dir, "00000_height_single");

  return 0;
}
