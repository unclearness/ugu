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

void DefaultShader(crender::Image3b* color, int x, int y,
                   const nanort::TriangleIntersection<>& isect,
                   const std::vector<glm::ivec3>& faces,
                   const std::vector<glm::ivec3>& uv_indices,
                   const std::vector<glm::vec2>& uv,
                   const std::vector<glm::vec3>& vertex_colors,
                   const crender::Image3b& diffuse_texture);

void VertexColorShader(crender::Image3b* color, int x, int y,
                       const nanort::TriangleIntersection<>& isect,
                       const std::vector<glm::ivec3>& faces,
                       const std::vector<glm::ivec3>& uv_indices,
                       const std::vector<glm::vec2>& uv,
                       const std::vector<glm::vec3>& vertex_colors,
                       const crender::Image3b& diffuse_texture);

void DiffuseNnShader(crender::Image3b* color, int x, int y,
                     const nanort::TriangleIntersection<>& isect,
                     const std::vector<glm::ivec3>& faces,
                     const std::vector<glm::ivec3>& uv_indices,
                     const std::vector<glm::vec2>& uv,
                     const std::vector<glm::vec3>& vertex_colors,
                     const crender::Image3b& diffuse_texture);

void DiffuseBilinearShader(crender::Image3b* color, int x, int y,
                           const nanort::TriangleIntersection<>& isect,
                           const std::vector<glm::ivec3>& faces,
                           const std::vector<glm::ivec3>& uv_indices,
                           const std::vector<glm::vec2>& uv,
                           const std::vector<glm::vec3>& vertex_colors,
                           const crender::Image3b& diffuse_texture);

}  // namespace crender
