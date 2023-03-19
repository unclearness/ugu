/*
 * Copyright (C) 2023, unclearness
 * All rights reserved.
 */

#pragma once

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include "ugu/accel/bvh.h"
#include "ugu/camera.h"
#include "ugu/renderable_mesh.h"
#include "ugu/renderer/base.h"
#include "ugu/shader/shader.h"

namespace ugu {

/// Holds all state information relevant to a character as loaded using FreeType
struct RendererCharacter {
  unsigned int TextureID;   // ID handle of the glyph texture
  Eigen::Vector2i Size;     // size of glyph
  Eigen::Vector2i Bearing;  // offset from baseline to left/top of glyph
  uint32_t Advance;         // horizontal offset to advance to next glyph
};

// A renderer class for rendering text displayed by a font loaded using the
// FreeType library. A single font is loaded, processed into a list of Character
// items for later rendering.
class TextRendererGl {
 public:
  // holds a list of pre-compiled Characters
  std::map<char, RendererCharacter> Characters;
  // shader used for text rendering
  Shader TextShader;
  // constructor
  TextRendererGl(unsigned int width, unsigned int height);
  // pre-compiles a list of characters from the given font
  void Load(std::string font, unsigned int fontSize);
  // renders a string of text using the precompiled list of characters
  void RenderText(
      const std::string& text, float x, float y, float scale,
      const Eigen::Vector3f& color = Eigen::Vector3f::Constant(1.f));
  struct Text {
    std::string body;
    float x;
    float y;
    float scale;
    Eigen::Vector3f color;
  };
  void RenderText(const Text& text);

 private:
  // render state
  uint32_t VAO = ~0u, VBO = ~0u;
};

class RendererGl {
 public:
  RendererGl();
  ~RendererGl();

  bool ClearGlState();
  bool Init();

  bool Draw(double tic = -1.0);

  void SetCamera(const CameraPtr cam);
  CameraPtr GetCamera() const;
  void SetMesh(RenderableMeshPtr mesh,
               const Eigen::Affine3f& trans = Eigen::Affine3f::Identity());
  void ClearMesh();
  void SetFragType(const FragShaderType& frag_type);
  void SetNearFar(float near_z, float far_z);
  void GetNearFar(float& near_z, float& far_z) const;
  void SetSize(uint32_t width, uint32_t height);
  void SetViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height);

  bool ReadGbuf();
  void GetGbuf(GBuffer& gbuf) const;

  void SetShowWire(bool show_wire);
  bool GetShowWire() const;
  void SetWireColor(const Eigen::Vector3f& wire_col);
  const Eigen::Vector3f& GetWireColor() const;
  void SetBackgroundColor(const Eigen::Vector3f& bkg_col);
  const Eigen::Vector3f& GetBackgroundColor() const;

  bool AddSelectedPosition(const RenderableMeshPtr& geom,
                           const Eigen::Vector3f& pos);
  bool AddSelectedPositions(const RenderableMeshPtr& geom,
                            const std::vector<Eigen::Vector3f>& pos_list);
  bool AddSelectedPositionColor(const RenderableMeshPtr& geom,
                                const Eigen::Vector3f& color);
  const Eigen::Vector3f& GetSelectedPositionColor(
      const RenderableMeshPtr& geom) const;
  void ClearSelectedPositions();

  void SetVisibility(const RenderableMeshPtr& geom, bool is_visible);
  bool GetVisibility(const RenderableMeshPtr& geom) const;

  void GetMergedBoundingBox(Eigen::Vector3f& bb_max, Eigen::Vector3f& bb_min);

  std::vector<std::vector<IntersectResult>> Intersect(const Ray& ray) const;

  std::pair<uint32_t, std::vector<std::vector<IntersectResult>>> TestVisibility(
      const Eigen::Vector3f& point) const;

  uint32_t GetMeshId(const RenderableMeshPtr& mesh) const;

  void SetText(const TextRendererGl::Text& text);
  void SetTexts(const std::vector<TextRendererGl::Text>& texts);
  const std::vector<TextRendererGl::Text>& GetTexts() const;
  std::vector<TextRendererGl::Text>& GetTexts();

  static const uint32_t MAX_SELECTED_POS = 128;  // Sync with GLSL
  static const uint32_t MAX_GEOM = 4;            // Sync with GLSL

 private:
  bool m_initialized = false;

  float m_near_z = 0.01f;
  float m_far_z = 1000.f;
  CameraPtr m_cam = nullptr;
  uint32_t m_width = 1024;
  uint32_t m_height = 720;

  uint32_t m_viewport_x = 0;
  uint32_t m_viewport_y = 0;
  uint32_t m_viewport_width = m_width;
  uint32_t m_viewport_height = m_height;

  uint32_t gBuffer = ~0u, gPosition = ~0u, gNormal = ~0u, gAlbedoSpec = ~0u,
           gId = ~0u, gFace = ~0u;
  std::array<uint32_t, 5> attachments = {~0u, ~0u, ~0u, ~0u, ~0u};
  uint32_t rboDepth = ~0u;
  uint32_t quadVAO = ~0u;
  uint32_t quadVBO = ~0u;

  std::unordered_map<RenderableMeshPtr, int> m_node_locs;
  std::unordered_map<RenderableMeshPtr, Eigen::Affine3f> m_node_trans;
  std::vector<RenderableMeshPtr> m_geoms;
  std::unordered_map<RenderableMeshPtr,
                     BvhPtr<Eigen::Vector3f, Eigen::Vector3i>>
      m_bvhs;

  std::unordered_map<RenderableMeshPtr, bool> m_visibility;

  Shader m_gbuf_shader;
  Shader m_deferred_shader;

  bool m_show_wire = true;
  Eigen::Vector3f m_wire_col;
  Eigen::Vector3f m_bkg_col;

  Eigen::Vector3f m_bb_max;
  Eigen::Vector3f m_bb_min;

  std::unordered_map<RenderableMeshPtr, std::vector<Eigen::Vector3f>>
      m_selected_positions;
  std::unordered_map<RenderableMeshPtr, Eigen::Vector3f>
      m_selected_position_colors;

  GBuffer m_gbuf;

  std::shared_ptr<TextRendererGl> m_text_renderer;
  std::string m_font = "default";
  uint32_t m_font_size = 32;
  std::vector<TextRendererGl::Text> m_texts;
};

using RendererGlPtr = std::shared_ptr<RendererGl>;

}  // namespace ugu
