/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include <fstream>

#include "ugu/image_io.h"
#include "ugu/image_proc.h"
#include "ugu/inflation/inflation.h"
#include "ugu/mesh.h"
#include "ugu/util/image_util.h"
#include "ugu/util/path_util.h"

// test by bunny data with 6 views
int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  // https://www.bandainamco-mirai.com/images/miraikomachi/miraikomachi_pose_neon03.png
  std::string data_dir = "../data/character/";
  std::string img_path = data_dir + "miraikomachi_pose_neon03.png";
  ugu::EnsureDirExists(data_dir);
  if (!ugu::FileExists(img_path)) {
    ugu::LOGE(
        "Please download "
        "https://www.bandainamco-mirai.com/images/miraikomachi/"
        "miraikomachi_pose_neon03.png and put it as %s\nSee Mirai Komachi "
        "informaiton https://www.miraikomachi.com/download/\n",
        img_path.c_str());
    return 1;
  }

  ugu::Image4b rgba_org = ugu::imread(img_path, -1);

  ugu::Image4b rgba;
  ugu::resize(rgba_org, rgba,
              {480, static_cast<int>(480 * static_cast<float>(rgba_org.rows) /
                                     static_cast<float>(rgba_org.cols))},
              0.0, 0.0);

  std::vector<ugu::Image1b> planes;
  ugu::Split(rgba, planes);

  ugu::Image1b mask = planes[3];
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

  ugu::Image3b color = ugu::Merge(planes[0], planes[1], planes[2]);
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
