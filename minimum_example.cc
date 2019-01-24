/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "include/renderer.h"

using currender::GrayFromDepth;
using currender::Image1b;
using currender::Image1f;
using currender::Image3b;
using currender::Mesh;
using currender::PinholeCamera;
using currender::Pose;
using currender::Renderer;
using currender::RendererOption;

namespace {
void SetCube(std::shared_ptr<Mesh> mesh) {
  std::vector<glm::vec3> vertices(8);
  std::vector<glm::vec3> vertex_colors(8);
  std::vector<glm::ivec3> vertex_indices(12);

  // make vertices
  float length = 200;
  glm::vec3 center{0, 0, 600};
  const float half_len = length / 2;
  vertices[0] = glm::vec3(-half_len, half_len, -half_len);
  vertices[1] = glm::vec3(half_len, half_len, -half_len);
  vertices[2] = glm::vec3(half_len, half_len, half_len);
  vertices[3] = glm::vec3(-half_len, half_len, half_len);
  vertices[4] = glm::vec3(-half_len, -half_len, -half_len);
  vertices[5] = glm::vec3(half_len, -half_len, -half_len);
  vertices[6] = glm::vec3(half_len, -half_len, half_len);
  vertices[7] = glm::vec3(-half_len, -half_len, half_len);

  // rotate cube 30 deg. around y axis
  // and translate
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
  for (auto& v : vertices) {
    v = R * v + center;
  }

  // set vertex color
  vertex_colors[0] = glm::vec3(255, 0, 0);
  vertex_colors[1] = glm::vec3(0, 255, 0);
  vertex_colors[4] = glm::vec3(0, 0, 255);
  vertex_colors[5] = glm::vec3(255, 255, 255);
  vertex_colors[2] = glm::vec3(255, 0, 255);
  vertex_colors[3] = glm::vec3(0, 255, 255);
  vertex_colors[6] = glm::vec3(0, 0, 0);
  vertex_colors[7] = glm::vec3(255, 255, 0);

  // make triangular faces
  vertex_indices[0] = glm::ivec3(0, 1, 2);
  vertex_indices[1] = glm::ivec3(0, 2, 3);
  vertex_indices[2] = glm::ivec3(4, 5, 6);
  vertex_indices[3] = glm::ivec3(4, 6, 7);
  vertex_indices[4] = glm::ivec3(1, 2, 5);
  vertex_indices[5] = glm::ivec3(2, 6, 5);
  vertex_indices[6] = glm::ivec3(0, 4, 3);
  vertex_indices[7] = glm::ivec3(3, 4, 7);
  vertex_indices[8] = glm::ivec3(0, 1, 5);
  vertex_indices[9] = glm::ivec3(0, 5, 4);
  vertex_indices[10] = glm::ivec3(2, 3, 7);
  vertex_indices[11] = glm::ivec3(2, 7, 6);

  // set to mesh
  mesh->set_vertices(vertices);
  mesh->set_vertex_colors(vertex_colors);
  mesh->set_vertex_indices(vertex_indices);
}

}  // namespace

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  // make cube mesh
  std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>();
  SetCube(mesh);

  // initialize renderer enabling vertex color rendering
  RendererOption option;
  option.use_vertex_color = true;
  Renderer renderer(option);

  // set mesh
  renderer.set_mesh(mesh);

  // prepare mesh for rendering (e.g. make BVH)
  renderer.PrepareMesh();

  // make PinholeCamera at origin
  int width = 640;
  int height = 480;
  float fov_y_deg = 50.0f;
  std::shared_ptr<PinholeCamera> camera =
      std::make_shared<PinholeCamera>(width, height, fov_y_deg);

  // set camera
  renderer.set_camera(camera);

  // render
  Image3b color;
  Image1f depth;
  Image1b mask;
  renderer.Render(&color, &depth, &mask);

  // save images
  std::string save_dir = "../data/minimum_example/";
  color.WritePng(save_dir + "color.png");
  mask.WritePng(save_dir + "mask.png");
  Image1b vis_depth;
  GrayFromDepth(depth, &vis_depth);
  vis_depth.WritePng(save_dir + "vis_depth.png");

  printf("images are saved in %s\n", save_dir.c_str());

  return 0;
}
