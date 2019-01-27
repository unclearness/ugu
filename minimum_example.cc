/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "include/renderer.h"

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  float length = 200;           // cube length
  glm::vec3 center{0, 0, 600};  // cube center
  // rotate cube 30 deg. around y axis
  glm::mat3 R;
  float deg = 30;
  R[0][0] = cos(glm::radians(deg));
  R[0][1] = 0;
  R[0][2] = sin(glm::radians(deg));
  R[1][0] = 0;
  R[1][1] = 1;
  R[1][2] = 0;
  R[2][0] = -sin(glm::radians(deg));
  R[2][1] = 0;
  R[2][2] = cos(glm::radians(deg));
  // make cube mesh with default vertex color
  std::shared_ptr<currender::Mesh> mesh =
      currender::MakeCube(length, R, center);

  // initialize renderer enabling vertex color rendering
  currender::RendererOption option;
  option.diffuse_color = currender::DiffuseColor::kVertex;
  currender::Renderer renderer(option);

  // set mesh
  renderer.set_mesh(mesh);

  // prepare mesh for rendering (e.g. make BVH)
  renderer.PrepareMesh();

  // make PinholeCamera (perspective camera) at origin
  int width = 320;
  int height = 240;
  float fov_y_deg = 50.0f;
  std::shared_ptr<currender::Camera> camera =
      std::make_shared<currender::PinholeCamera>(width, height, fov_y_deg);

  // set camera
  renderer.set_camera(camera);

  // render
  currender::Image3b color;
  currender::Image1f depth;
  currender::Image3f normal;
  currender::Image1b mask;
  renderer.Render(&color, &depth, &normal, &mask);

  // save images
  std::string save_dir = "../data/minimum_example/";
  color.WritePng(save_dir + "color.png");
  mask.WritePng(save_dir + "mask.png");
  currender::Image1b vis_depth;
  currender::Depth2Gray(depth, &vis_depth);
  vis_depth.WritePng(save_dir + "vis_depth.png");
  currender::Image3b vis_normal;
  currender::Normal2Color(normal, &vis_normal);
  vis_normal.WritePng(save_dir + "vis_normal.png");

  printf("images are saved in %s\n", save_dir.c_str());

  return 0;
}
