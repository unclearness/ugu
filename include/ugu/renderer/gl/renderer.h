/*
 * Copyright (C) 2023, unclearness
 * All rights reserved.
 */

#pragma once

#include <array>
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

  bool ClearGlState();
  bool Init();

  bool Draw(double tic = -1.0);

  void SetCamera(const CameraPtr cam);
  void SetMesh(RenderableMeshPtr mesh,
               const Eigen::Affine3f& trans = Eigen::Affine3f::Identity());
  void ClearMesh();
  void SetFragType(const FragShaderType& frag_type);
  void SetNearFar(float near_z, float far_z);
  void SetSize(uint32_t width, uint32_t height);

  bool ReadGbuf();
  void GetGbuf(GBuffer& gbuf) const;

  void SetShowWire(bool show_wire);
  bool GetShowWire() const;
  void SetWireColor(const Eigen::Vector3f& wire_col);
  const Eigen::Vector3f& GetWireColor() const;
  void SetBackgroundColor(const Eigen::Vector3f& bkg_col);

  bool AddSelectedPos(const Eigen::Vector3f& pos);

 private:
  bool m_initialized = false;
  float m_near_z = 0.01f;
  float m_far_z = 1000.f;
  CameraPtr m_cam = nullptr;

  uint32_t m_width = 1024;
  uint32_t m_height = 720;

  uint32_t gBuffer = ~0u, gPosition = ~0u, gNormal = ~0u, gAlbedoSpec = ~0u,
           gId = ~0u, gFace = ~0u;
  std::array<uint32_t, 5> attachments;
  uint32_t rboDepth = ~0u;
  uint32_t quadVAO = 0;
  uint32_t quadVBO = ~0u;

  std::unordered_map<RenderableMeshPtr, int> m_node_locs;
  std::unordered_map<RenderableMeshPtr, Eigen::Affine3f> m_node_trans;
  std::vector<RenderableMeshPtr> m_geoms;
  Shader m_gbuf_shader;
  Shader m_deferred_shader;

  bool m_show_wire = true;
  Eigen::Vector3f m_wire_col;
  Eigen::Vector3f m_bkg_col;

  Eigen::Vector3f m_bb_max;
  Eigen::Vector3f m_bb_min;

  const uint32_t MAX_SELECTED_POS = 32;  // Sync with GLSL
  std::vector<Eigen::Vector3f> m_selected_positions;
  // std::vector<Eigen::Vector3f> m_selected_positions_1;

  GBuffer m_gbuf;
};

using RendererGlPtr = std::shared_ptr<RendererGl>;

}  // namespace ugu
