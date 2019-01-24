/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <vector>

#include "include/common.h"
#include "include/image.h"
#include "nanort/nanort.h"

namespace currender {

void DefaultShader(Image3b* color, int x, int y,
                   const nanort::TriangleIntersection<>& isect,
                   const std::vector<glm::ivec3>& faces,
                   const std::vector<glm::ivec3>& uv_indices,
                   const std::vector<glm::vec2>& uv,
                   const std::vector<glm::vec3>& vertex_colors,
                   const Image3b& diffuse_texture);

void VertexColorShader(Image3b* color, int x, int y,
                       const nanort::TriangleIntersection<>& isect,
                       const std::vector<glm::ivec3>& faces,
                       const std::vector<glm::ivec3>& uv_indices,
                       const std::vector<glm::vec2>& uv,
                       const std::vector<glm::vec3>& vertex_colors,
                       const Image3b& diffuse_texture);

void DiffuseNnShader(Image3b* color, int x, int y,
                     const nanort::TriangleIntersection<>& isect,
                     const std::vector<glm::ivec3>& faces,
                     const std::vector<glm::ivec3>& uv_indices,
                     const std::vector<glm::vec2>& uv,
                     const std::vector<glm::vec3>& vertex_colors,
                     const Image3b& diffuse_texture);

void DiffuseBilinearShader(Image3b* color, int x, int y,
                           const nanort::TriangleIntersection<>& isect,
                           const std::vector<glm::ivec3>& faces,
                           const std::vector<glm::ivec3>& uv_indices,
                           const std::vector<glm::vec2>& uv,
                           const std::vector<glm::vec3>& vertex_colors,
                           const Image3b& diffuse_texture);

}  // namespace currender
