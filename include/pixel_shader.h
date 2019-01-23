/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <vector>

#include "nanort/nanort.h"
#include "include/common.h"
#include "include/image.h"

namespace crender {

void default_shader(crender::Image3b* color, int x, int y,
                    const nanort::TriangleIntersection<>& isect,
                    const std::vector<glm::ivec3>& faces,
                    const std::vector<glm::ivec3>& uv_indices,
                    const std::vector<glm::vec2>& uv,
                    const std::vector<glm::vec3>& vertex_colors,
                    const crender::Image3b& diffuse_texture) {
  (void)isect;
  (void)faces;
  (void)uv_indices;
  (void)uv;
  (void)vertex_colors;
  (void)diffuse_texture;

  // set Green
  color->at(x, y, 0) = 0;
  color->at(x, y, 1) = 255;
  color->at(x, y, 2) = 0;
}

void vertex_color_shader(crender::Image3b* color, int x, int y,
                         const nanort::TriangleIntersection<>& isect,
                         const std::vector<glm::ivec3>& faces,
                         const std::vector<glm::ivec3>& uv_indices,
                         const std::vector<glm::vec2>& uv,
                         const std::vector<glm::vec3>& vertex_colors,
                         const crender::Image3b& diffuse_texture) {
  (void)uv_indices;
  (void)uv;
  (void)diffuse_texture;

  glm::vec3 interp_color;
  unsigned int fid = isect.prim_id;
  float u = isect.u;
  float v = isect.v;
  // barycentric interpolation of vertex color
  interp_color = (1.0f - u - v) * vertex_colors[faces[fid][0]] +
                 u * vertex_colors[faces[fid][1]] +
                 v * vertex_colors[faces[fid][2]];

  for (int k = 0; k < 3; k++) {
    color->at(x, y, k) = static_cast<unsigned char>(interp_color[k]);
  }
}

void diffuse_nn_shader(crender::Image3b* color, int x, int y,
                       const nanort::TriangleIntersection<>& isect,
                       const std::vector<glm::ivec3>& faces,
                       const std::vector<glm::ivec3>& uv_indices,
                       const std::vector<glm::vec2>& uv,
                       const std::vector<glm::vec3>& vertex_colors,
                       const crender::Image3b& diffuse_texture) {
  (void)faces;
  (void)vertex_colors;

  glm::vec3 interp_color;
  unsigned int fid = isect.prim_id;
  float u = isect.u;
  float v = isect.v;

  // barycentric interpolation of uv
  glm::vec2 interp_uv = (1.0f - u - v) * uv[uv_indices[fid][0]] +
                        u * uv[uv_indices[fid][1]] + v * uv[uv_indices[fid][2]];
  float f_tex_pos[2];
  f_tex_pos[0] = interp_uv[0] * (diffuse_texture.width() - 1);
  f_tex_pos[1] = (1.0f - interp_uv[1]) * (diffuse_texture.height() - 1);

  int tex_pos[2] = {0, 0};
  tex_pos[0] = static_cast<int>(std::round(f_tex_pos[0]));
  tex_pos[1] = static_cast<int>(std::round(f_tex_pos[1]));
  for (int k = 0; k < 3; k++) {
    interp_color[k] = diffuse_texture.at(tex_pos[0], tex_pos[1], k);
  }

  for (int k = 0; k < 3; k++) {
    color->at(x, y, k) = static_cast<unsigned char>(interp_color[k]);
  }
}

void diffuse_bilinear_shader(crender::Image3b* color, int x, int y,
                             const nanort::TriangleIntersection<>& isect,
                             const std::vector<glm::ivec3>& faces,
                             const std::vector<glm::ivec3>& uv_indices,
                             const std::vector<glm::vec2>& uv,
                             const std::vector<glm::vec3>& vertex_colors,
                             const crender::Image3b& diffuse_texture) {
  (void)faces;
  (void)vertex_colors;

  glm::vec3 interp_color;
  unsigned int fid = isect.prim_id;
  float u = isect.u;
  float v = isect.v;

  // barycentric interpolation of uv
  glm::vec2 interp_uv = (1.0f - u - v) * uv[uv_indices[fid][0]] +
                        u * uv[uv_indices[fid][1]] + v * uv[uv_indices[fid][2]];
  float f_tex_pos[2];
  f_tex_pos[0] = interp_uv[0] * (diffuse_texture.width() - 1);
  f_tex_pos[1] = (1.0f - interp_uv[1]) * (diffuse_texture.height() - 1);

  int tex_pos_min[2] = {0, 0};
  int tex_pos_max[2] = {0, 0};
  tex_pos_min[0] = static_cast<int>(std::floor(f_tex_pos[0]));
  tex_pos_min[1] = static_cast<int>(std::floor(f_tex_pos[1]));
  tex_pos_max[0] = tex_pos_min[0] + 1;
  tex_pos_max[1] = tex_pos_min[1] + 1;

  float local_u = f_tex_pos[0] - tex_pos_min[0];
  float local_v = f_tex_pos[1] - tex_pos_min[1];

  for (int k = 0; k < 3; k++) {
    // bilinear interpolation of pixel color
    interp_color[k] =
        (1.0f - local_u) * (1.0f - local_v) *
            diffuse_texture.at(tex_pos_min[0], tex_pos_min[1], k) +
        local_u * (1.0f - local_v) *
            diffuse_texture.at(tex_pos_max[0], tex_pos_min[1], k) +
        (1.0f - local_u) * local_v *
            diffuse_texture.at(tex_pos_min[0], tex_pos_max[1], k) +
        local_u * local_v *
            diffuse_texture.at(tex_pos_max[0], tex_pos_max[1], k);

    assert(0.0f <= interp_color[k] && interp_color[k] <= 255.0f);
  }

  for (int k = 0; k < 3; k++) {
    color->at(x, y, k) = static_cast<unsigned char>(interp_color[k]);
  }
}

}  // namespace crender
