/*
 * Copyright (C) 2023, unclearness
 * All rights reserved.
 */

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "ugu/camera.h"
#include "ugu/renderable_mesh.h"
#include "ugu/renderer/base.h"
#include "ugu/shader/shader.h"

namespace ugu {

class RendererGl {
 public:
  RendererGl();
  ~RendererGl();

  bool Init();

  bool Draw(double tic = -1.0);

  void SetCamera(const CameraPtr cam);
  void SetMesh(RenderableMeshPtr mesh,
               const Eigen::Affine3f& trans = Eigen::Affine3f::Identity());
  void ClearMesh();
  void SetFragType(const FragShaderType& frag_type);

 private:
  float m_near_z = 0.01f;
  float m_far_z = 1000.f;
  int m_view_loc = -1;
  int m_prj_loc = -1;
  CameraPtr m_cam = nullptr;
  // std::vector<RenderGlNode> m_nodes;

  std::unordered_map<RenderableMeshPtr, int> m_node_locs;
  std::unordered_map<RenderableMeshPtr, Eigen::Affine3f> m_nodes;
  Shader m_shader;
};

using RendererGlPtr = std::shared_ptr<RendererGl>;

}  // namespace ugu
