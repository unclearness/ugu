/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include <stdio.h>

#include <fstream>

#include "ugu/camera.h"
#include "ugu/image_io.h"
#include "ugu/sfs/voxel_carver.h"
#include "ugu/util/image_util.h"
#include "ugu/util/string_util.h"

// test by bunny data with 6 views
int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  std::string data_dir{"../data/sfs/"};
  std::vector<Eigen::Affine3d> poses;
  ugu::LoadTumFormat(data_dir + "tumpose.txt", &poses);

  ugu::VoxelCarver carver;
  ugu::VoxelCarverOption option;

  // exact mesh bounding box computed in advacne
  option.bb_min = Eigen::Vector3f(-250.000000f, -344.586151f, -129.982697f);
  option.bb_max = Eigen::Vector3f(250.000000f, 150.542343f, 257.329224f);

  // add offset to the bounding box to keep boundary clean
  float bb_offset = 20.0f;
  option.bb_min[0] -= bb_offset;
  option.bb_min[1] -= bb_offset;
  option.bb_min[2] -= bb_offset;

  option.bb_max[0] += bb_offset;
  option.bb_max[1] += bb_offset;
  option.bb_max[2] += bb_offset;

  // voxel resolution is 10mm
  option.resolution = 10.0f;

  carver.set_option(option);

  carver.Init();

  // image size and intrinsic parameters
  int width = 320;
  int height = 240;
  Eigen::Vector2f principal_point(159.3f, 127.65f);
  Eigen::Vector2f focal_length(258.65f, 258.25f);
  std::shared_ptr<ugu::Camera> camera = std::make_shared<ugu::PinholeCamera>(
      width, height, Eigen::Affine3d::Identity(), principal_point,
      focal_length);

  for (size_t i = 0; i < 6; i++) {
    camera->set_c2w(poses[i]);

    std::string num = ugu::zfill(i);

    ugu::Image1b silhouette =
        ugu::Imread<ugu::Image1b>(data_dir + "/mask_" + num + ".png");

    ugu::Image1f sdf;
    // Carve() is the main process to update voxels. Corresponds to the fusion
    // step in KinectFusion
    carver.Carve(*camera, silhouette, &sdf);

    // save SDF visualization
    ugu::Image3b vis_sdf;
    ugu::SignedDistance2Color(sdf, &vis_sdf, -1.0f, 1.0f);
    ugu::imwrite(data_dir + "/sdf_" + num + ".png", vis_sdf);

    ugu::Mesh mesh;
    // voxel extraction
    // slow for algorithm itself and saving to disk
    carver.ExtractVoxel(&mesh);
    // mesh.WritePly(data_dir + "/voxel_" + num + ".ply");

    // marching cubes
    // smoother and faster
    carver.ExtractIsoSurface(&mesh, 0.0);
    mesh.WritePly(data_dir + "/surface_" + num + ".ply");
  }

  return 0;
}
