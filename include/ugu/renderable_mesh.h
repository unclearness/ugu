/*
 * Copyright (C) 2023, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/mesh.h"
#include "ugu/shader/shader.h"

namespace ugu {

struct Vertex {
  Eigen::Vector3f pos;
  Eigen::Vector3f nor;
  Eigen::Vector3f col;
  Eigen::Vector2f uv;
};

class RenderableMesh : public Mesh {
 public:
  using Mesh::Mesh;

  void Draw(const Shader &shader) const;

  void BindTextures();
  void SetupMesh();

 private:
  uint32_t VAO = ~0;
  uint32_t VBO = ~0;
  uint32_t EBO = ~0;

  std::vector<Vertex> renderable_vertices;
  std::vector<uint32_t> flatten_indices;
  std::vector<uint32_t> texture_ids;
};

}  // namespace ugu
