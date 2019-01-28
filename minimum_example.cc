/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "include/renderer.h"

namespace {

std::shared_ptr<currender::Mesh> MakeExampleCube() {
  // cube mesh parameters
  float length = 200;           // cube length
  glm::vec3 center{0, 0, 600};  // cube center position

  // mesh rotation matrix from eular angle. rotate cube -30 deg. around y axis
  // and then rotate further 30 deg. around x axis
  glm::mat3 R = currender::EulerAngleDegYXZ(-30.0f, 30.0f, 0.0f);

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
}  // namespace

int main() {
  // make an inclined cube mesh with vertex color
  auto mesh = MakeExampleCube();

  // initialize renderer enabling vertex color rendering
  currender::RendererOption option;
  option.diffuse_color = currender::DiffuseColor::kVertex;
  currender::Renderer renderer(option);

  // set mesh
  renderer.set_mesh(mesh);

  // prepare mesh for rendering (e.g. make BVH)
  renderer.PrepareMesh();

  // make PinholeCamera (perspective camera) at origin.
  // its image size is 160 * 120 and its y (vertical) FoV is 50 deg.
  auto camera = std::make_shared<currender::PinholeCamera>(160, 120, 50.0f);

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
