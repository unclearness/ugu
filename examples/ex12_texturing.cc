/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include <stdio.h>
#include <fstream>

#include "ugu/camera.h"
#include "ugu/texturing/texture_mapper.h"
#include "ugu/texturing/vertex_colorizer.h"
#include "ugu/texturing/visibility_tester.h"
#include "ugu/external/external.h"

// test by bunny data with 6 views
int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  std::string data_dir = "../data/bunny/";
  std::string obj_path = data_dir + "bunny.obj";
  std::string tumpose_path = data_dir + "tumpose.txt";

  {
    std::ifstream ifs(obj_path);
    if (!ifs.is_open()) {
      printf("Please put %s\n", obj_path.c_str());
      return -1;
    }
  }

  // load mesh
  std::shared_ptr<ugu::Mesh> input_mesh = std::make_shared<ugu::Mesh>();
  input_mesh->LoadObj(obj_path, data_dir);

  // load pose
  {
    std::ifstream ifs(tumpose_path);
    if (!ifs.is_open()) {
      printf("Please run test_renderer to generate tumpose.txt\n");
      return -1;
    }
  }

  std::vector<std::pair<int, Eigen::Affine3d>> poses;
  ugu::LoadTumFormat(tumpose_path, &poses);

  // Make PinholeCamera
  // borrow KinectV1 intrinsics of Freiburg 1 RGB
  // https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
  float r = 0.5f;  // scale to smaller size from VGA
  int width = static_cast<int>(640 * r);
  int height = static_cast<int>(480 * r);
  Eigen::Vector2f principal_point(318.6f * r, 255.3f * r);
  Eigen::Vector2f focal_length(517.3f * r, 516.5f * r);

  // convert to ugu::Keyframe
  ugu::VisibilityTesterOption option;
  option.use_mask = false;
  option.use_depth = false;
  ugu::VisibilityTester tester(option);
  ugu::VisibilityInfo info;
  std::vector<std::shared_ptr<ugu::Keyframe>> keyframes(poses.size());

  for (int i = 0; i < static_cast<int>(poses.size()); i++) {
    keyframes[i] = std::make_shared<ugu::Keyframe>();
    keyframes[i]->id = i;

    keyframes[i]->camera = std::make_shared<ugu::PinholeCamera>(
        width, height, poses[i].second, principal_point, focal_length);

    std::string color_path =
        data_dir + ugu::zfill(poses[i].first) + "_color.png";
    keyframes[i]->color_path = color_path;
    keyframes[i]->color = ugu::imread<ugu::Image3b>(keyframes[i]->color_path);
  }

  tester.set_mesh(input_mesh);
  tester.PrepareMesh();

  info.vertex_info_list.resize(input_mesh->vertices().size());
  info.face_info_list.resize(input_mesh->vertex_indices().size());

  tester.Test(keyframes, &info);

  std::shared_ptr<ugu::Mesh> output_mesh =
      std::make_shared<ugu::Mesh>(*input_mesh.get());

  ugu::TextureMappingOption tmoption;

  ugu::VertexColorizer vertex_colorizer;
  vertex_colorizer.Colorize(info, output_mesh.get());
  std::string output_ply_path = data_dir + "bunny_vertex_color.ply";
  output_mesh->WritePly(output_ply_path);

  tmoption.uv_type = ugu::OutputUvType::kUseOriginalMeshUv;
  ugu::TextureMapping(keyframes, info, output_mesh.get(), tmoption);
  output_mesh->WriteObj(data_dir, "bunny_textured_orguv");

  tmoption.uv_type = ugu::OutputUvType::kGenerateSimpleTile;
  ugu::TextureMapping(keyframes, info, output_mesh.get(), tmoption);
  output_mesh->WriteObj(data_dir, "bunny_textured_tileuv");

  tmoption.uv_type = ugu::OutputUvType::kGenerateSimpleTriangles;
  ugu::TextureMapping(keyframes, info, output_mesh.get(), tmoption);
  output_mesh->WriteObj(data_dir, "bunny_textured_triuv");

  tmoption.uv_type = ugu::OutputUvType::kGenerateSimpleCharts;
  tmoption.tex_h = 512;
  tmoption.tex_w = 512;
  ugu::TextureMapping(keyframes, info, output_mesh.get(), tmoption);
  output_mesh->WriteObj(data_dir, "bunny_textured_charts");

  // mvs-texturing
  ugu::MvsTexturing(keyframes, output_mesh.get());

  return 0;
}
