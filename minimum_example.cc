/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

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

namespace {

std::shared_ptr<currender::Mesh> MakeExampleCube() {
  // cube mesh parameters
  float length = 200;                 // cube length
  Eigen::Vector3f center{0, 0, 600};  // cube center position

  // mesh rotation matrix from eular angle. rotate cube -30 deg. around y axis
  // and then rotate further 30 deg. around x axis
  Eigen::Matrix3f R =
      (Eigen::AngleAxisf(currender::radians(-30.0f), Eigen::Vector3f::UnitY()) *
       Eigen::AngleAxisf(currender::radians(30.0f), Eigen::Vector3f::UnitX()) *
       Eigen::AngleAxisf(currender::radians(0.0f), Eigen::Vector3f::UnitZ()))
          .toRotationMatrix();
  // make cube mesh with the above paramters and default vertex color
  return currender::MakeCube(length, R, center);
}

void SaveImages(const currender::Image3b& color,
                const currender::Image1f& depth,
                const currender::Image3f& normal,
                const currender::Image1b& mask) {
  // dir to save images
  std::string save_dir = "../data/minimum_example/";

  // save color
  color.WritePng(save_dir + "color.png");

  // save mask
  mask.WritePng(save_dir + "mask.png");

  // convert depth to gray and save
  currender::Image1b vis_depth;
  currender::Depth2Gray(depth, &vis_depth);
  vis_depth.WritePng(save_dir + "vis_depth.png");

  // convert normal to color and save
  currender::Image3b vis_normal;
  currender::Normal2Color(normal, &vis_normal);
  vis_normal.WritePng(save_dir + "vis_normal.png");

  printf("images are saved in %s\n", save_dir.c_str());
}

void CalcIntrinsics(int width, int height, float fov_y_deg,
                    Eigen ::Vector2f* principal_point,
                    Eigen::Vector2f* focal_length) {
  (*principal_point)[0] = width * 0.5f - 0.5f;
  (*principal_point)[1] = height * 0.5f - 0.5f;

  (*focal_length)[1] =
      height * 0.5f /
      static_cast<float>(std::tan(currender::radians<float>(fov_y_deg) * 0.5));
  (*focal_length)[0] = (*focal_length)[1];
}

}  // namespace

int main() {
  // make an inclined cube mesh with vertex color
  auto mesh = MakeExampleCube();

  // initialize renderer enabling vertex color rendering and lambertian shading
  currender::RendererOption option;
  option.diffuse_color = currender::DiffuseColor::kVertex;
  option.diffuse_shading = currender::DiffuseShading::kLambertian;
  currender::Renderer renderer(option);

  // set mesh
  renderer.set_mesh(mesh);

  // prepare mesh for rendering (e.g. make BVH)
  renderer.PrepareMesh();

  // make PinholeCamera (perspective camera) at origin.
  // its image size is 160 * 120 and its y (vertical) FoV is 50 deg.
  int width = 160;
  int height = 120;
  float fov_y_deg = 50.0f;
  Eigen ::Vector2f principal_point, focal_length;
  CalcIntrinsics(width, height, fov_y_deg, &principal_point, &focal_length);
  auto camera = std::make_shared<currender::PinholeCamera>(
      width, height, Eigen::Affine3d::Identity(), principal_point,
      focal_length);

  // set camera
  renderer.set_camera(camera);

  // render images
  currender::Image3b color;
  currender::Image1f depth;
  currender::Image3f normal;
  currender::Image1b mask;
  renderer.Render(&color, &depth, &normal, &mask);

  // save images
  SaveImages(color, depth, normal, mask);

  return 0;
}
