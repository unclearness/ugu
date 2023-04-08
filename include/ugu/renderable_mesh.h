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
  Eigen::Vector2f uv;
  Eigen::Vector3f col;
  Eigen::Vector3f id;  // [0]: face id, [1]: geom id, [2]: for extension
};

class RenderableMesh;
using RenderableMeshPtr = std::shared_ptr<RenderableMesh>;

class RenderableMesh : public Mesh {
 public:
  static RenderableMeshPtr Create() {
    return std::make_shared<RenderableMesh>();
  }

  void Draw(const Shader &shader) const;

  void BindTextures();
  void SetupMesh(int geo_id);
  void ClearGlState() const;

  uint32_t VAO = ~0u;
  uint32_t VBO = ~0u;
  uint32_t EBO = ~0u;

  std::vector<Vertex> renderable_vertices;
  std::vector<Eigen::Vector3i> renderable_indices;
  std::vector<uint32_t> flatten_indices;
  std::vector<uint32_t> texture_ids;
};

}  // namespace ugu
