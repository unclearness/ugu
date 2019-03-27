/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include <stdio.h>
#include <fstream>

#include "currender/renderer.h"

#ifdef CURRENDER_USE_TINYOBJLOADER
#ifndef CURRENDER_TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#endif
#include "tinyobjloader/tiny_obj_loader.h"
#undef TINYOBJLOADER_IMPLEMENTATION
#endif

#ifdef CURRENDER_USE_STB
#pragma warning(push)
#pragma warning(disable : 4100)
#ifndef CURRENDER_STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#include "stb/stb_image.h"
#pragma warning(pop)
#undef STB_IMAGE_IMPLEMENTATION

#pragma warning(push)
#pragma warning(disable : 4996)
#ifndef CURRENDER_STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include "stb/stb_image_write.h"
#pragma warning(pop)
#undef STB_IMAGE_WRITE_IMPLEMENTATION
#endif

using currender::Camera;
using currender::Depth2Gray;
using currender::Image1b;
using currender::Image1f;
using currender::Image3b;
using currender::Image3f;
using currender::Mesh;
using currender::MeshStats;
using currender::Normal2Color;
using currender::PinholeCamera;
using currender::Renderer;
using currender::RendererOption;

namespace {
void Test(const std::string& out_dir, std::shared_ptr<Mesh> mesh,
          std::shared_ptr<Camera> camera, const Renderer& renderer) {
  // images
  Image3b color;
  Image1f depth;
  Image3f normal;
  Image1b mask;
  Image1b vis_depth;
  Image3b vis_normal;

  MeshStats stats = mesh->stats();
  Eigen::Vector3f center = stats.center;
  Eigen::Vector3f eye;
  Eigen::Matrix4f c2w;

  // translation offset is the largest edge length of bounding box * 1.5
  Eigen::Vector3f diff = stats.bb_max - stats.bb_min;
  float offset = std::max(diff[0], std::max(diff[1], diff[2])) * 1.5f;

  // from front
  eye = center;
  eye[2] -= offset;
  currender::c2w(eye, center, Eigen::Vector3f(0, -1, 0), &c2w);

  camera->set_c2w(Eigen::Affine3d(c2w.cast<double>()));
  renderer.Render(&color, &depth, &normal, &mask);
  color.WritePng(out_dir + "front_color.png");
  mask.WritePng(out_dir + "front_mask.png");
  Depth2Gray(depth, &vis_depth);
  vis_depth.WritePng(out_dir + "front_vis_depth.png");
  Normal2Color(normal, &vis_normal);
  vis_normal.WritePng(out_dir + "front_vis_normal.png");

  // from back
  eye = center;
  eye[2] += offset;
  currender::c2w(eye, center, Eigen::Vector3f(0, -1, 0), &c2w);
  camera->set_c2w(Eigen::Affine3d(c2w.cast<double>()));
  renderer.Render(&color, &depth, &normal, &mask);
  color.WritePng(out_dir + "back_color.png");
  mask.WritePng(out_dir + "back_mask.png");
  Depth2Gray(depth, &vis_depth);
  vis_depth.WritePng(out_dir + "back_vis_depth.png");
  Normal2Color(normal, &vis_normal);
  vis_normal.WritePng(out_dir + "back_vis_normal.png");

  // from right
  eye = center;
  eye[0] += offset;
  currender::c2w(eye, center, Eigen::Vector3f(0, -1, 0), &c2w);
  camera->set_c2w(Eigen::Affine3d(c2w.cast<double>()));
  renderer.Render(&color, &depth, &normal, &mask);
  color.WritePng(out_dir + "right_color.png");
  mask.WritePng(out_dir + "right_mask.png");
  Depth2Gray(depth, &vis_depth);
  vis_depth.WritePng(out_dir + "right_vis_depth.png");
  Normal2Color(normal, &vis_normal);
  vis_normal.WritePng(out_dir + "right_vis_normal.png");

  // from left
  eye = center;
  eye[0] -= offset;
  currender::c2w(eye, center, Eigen::Vector3f(0, -1, 0), &c2w);
  camera->set_c2w(Eigen::Affine3d(c2w.cast<double>()));
  renderer.Render(&color, &depth, &normal, &mask);
  color.WritePng(out_dir + "left_color.png");
  mask.WritePng(out_dir + "left_mask.png");
  Depth2Gray(depth, &vis_depth);
  vis_depth.WritePng(out_dir + "left_vis_depth.png");
  Normal2Color(normal, &vis_normal);
  vis_normal.WritePng(out_dir + "left_vis_normal.png");

  // from top
  eye = center;
  eye[1] -= offset;
  currender::c2w(eye, center, Eigen::Vector3f(0, 0, 1), &c2w);
  camera->set_c2w(Eigen::Affine3d(c2w.cast<double>()));
  renderer.Render(&color, &depth, &normal, &mask);
  color.WritePng(out_dir + "top_color.png");
  mask.WritePng(out_dir + "top_mask.png");
  Depth2Gray(depth, &vis_depth);
  vis_depth.WritePng(out_dir + "top_vis_depth.png");
  Normal2Color(normal, &vis_normal);
  vis_normal.WritePng(out_dir + "top_vis_normal.png");

  // from bottom
  eye = center;
  eye[1] += offset;
  currender::c2w(eye, center, Eigen::Vector3f(0, 0, -1), &c2w);
  camera->set_c2w(Eigen::Affine3d(c2w.cast<double>()));
  renderer.Render(&color, &depth, &normal, &mask);
  color.WritePng(out_dir + "bottom_color.png");
  mask.WritePng(out_dir + "bottom_mask.png");
  Depth2Gray(depth, &vis_depth);
  vis_depth.WritePng(out_dir + "bottom_vis_depth.png");
  Normal2Color(normal, &vis_normal);
  vis_normal.WritePng(out_dir + "bottom_vis_normal.png");
}

void AlignMesh(std::shared_ptr<Mesh> mesh) {
  // move center as origin
  MeshStats stats = mesh->stats();
  mesh->Translate(-stats.center);

  // rotate 180 deg. around x_axis to align z:forward, y:down, x:right,
  Eigen::Matrix3f R =
      Eigen::AngleAxisf(currender::radians(180.0f), Eigen::Vector3f::UnitX())
          .toRotationMatrix();
  mesh->Rotate(R);

  // recover original translation
  mesh->Translate(stats.center);
}
}  // namespace

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  // CMake downloads and unzip this data automatically
  // Please do manually if it got wrong
  // http://www.kunzhou.net/tex-models/bunny.zip
  /*
   *  @article{Texturemontage05,
   *  author = "Kun Zhou and Xi Wang and Yiying Tong and Mathieu Desbrun and
   *  Baining Guo and Heung-Yeung Shum", title = "Texturemontage: Seamless
   *  Texturing of Arbitrary Surfaces From Multiple Images", journal = "ACM
   *  Transactions on Graphics", volume = "24", number = "3", year="2005", pages
   *  = "1148-1155"
   */
  std::string data_dir = "../data/bunny/";
  std::string obj_path = data_dir + "bunny.obj";

  std::ifstream ifs(obj_path);
  if (!ifs.is_open()) {
    printf("Please put %s\n", obj_path.c_str());
    return -1;
  }

  // load mesh
  std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>();
  mesh->LoadObj(obj_path, data_dir);

  // original mesh with z:backward, y:up, x:right, like OpenGL
  // align z:forward, y:down, x:right
  AlignMesh(mesh);

  // initialize renderer with diffuse texture color and lambertian shading 
  RendererOption option;
  option.diffuse_color = currender::DiffuseColor::kTexture;
  option.diffuse_shading = currender::DiffuseShading::kLambertian;
  Renderer renderer(option);

  // set mesh
  renderer.set_mesh(mesh);

  // prepare mesh for rendering (e.g. make BVH)
  renderer.PrepareMesh();

  // Make PinholeCamera
  // borrow KinectV1 intrinsics of Freiburg 1 RGB
  // https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
  float r = 0.5f;  // scale to smaller size from VGA
  int width = static_cast<int>(640 * r);
  int height = static_cast<int>(480 * r);
  Eigen::Vector2f principal_point(318.6f * r, 255.3f * r);
  Eigen::Vector2f focal_length(517.3f * r, 516.5f * r);
  std::shared_ptr<Camera> camera = std::make_shared<PinholeCamera>(
      width, height, Eigen::Affine3d::Identity(), principal_point,
      focal_length);

  // set camera
  renderer.set_camera(camera);

  // test
  Test(data_dir, mesh, camera, renderer);

  return 0;
}
