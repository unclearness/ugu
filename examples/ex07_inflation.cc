/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include <fstream>

#include "ugu/inflation/inflation.h"
#include "ugu/mesh.h"
#include "ugu/util/image_util.h"

// test by bunny data with 6 views
int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  std::string data_dir = "../data/anime/";
  std::string mask_path = data_dir + "shion2_mask.png";
  std::string color_path = data_dir + "shion2.jpg";

  ugu::Image1b mask = ugu::imread<ugu::Image1b>(mask_path);
  ugu::Erode(mask.clone(), &mask, 3);
  ugu::Erode(mask.clone(), &mask, 3);
  ugu::resize(mask.clone(), mask, ugu::Size(-1, -1), 0.5f, 0.5f);

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

  params.generate_back = true;
  params.back_texture_type = ugu::InflationBackTextureType::INPAINT;
  ugu::Inflation(mask, height, mesh, params);
  mesh.WriteObj(data_dir, "00000_height_double");

  return 0;
}
