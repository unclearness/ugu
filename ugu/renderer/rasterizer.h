/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <memory>

#include "ugu/renderer/base.h"

namespace ugu {

class Rasterizer : public Renderer {
  class Impl;
  std::unique_ptr<Impl> pimpl_;

 public:
  Rasterizer();
  ~Rasterizer() override;

  // Set option
  explicit Rasterizer(const RendererOption& option);
  void set_option(const RendererOption& option) override;

  // Set mesh
  void set_mesh(std::shared_ptr<const Mesh> mesh) override;

  // Should call after set_mesh() and before Render()
  // Don't modify mesh outside after calling PrepareMesh()
  bool PrepareMesh() override;

  // Set camera
  void set_camera(std::shared_ptr<const Camera> camera) override;

  // Rendering all images
  // If you don't need some of them, pass nullptr
  bool Render(Image3b* color, Image1f* depth, Image3f* normal, Image1b* mask,
              Image1i* face_id) const override;

  // Rendering a image
  bool RenderColor(Image3b* color) const override;
  bool RenderDepth(Image1f* depth) const override;
  bool RenderNormal(Image3f* normal) const override;
  bool RenderMask(Image1b* mask) const override;
  bool RenderFaceId(Image1i* face_id) const override;

  // These Image1w* depth interfaces are prepared for widely used 16 bit
  // (unsigned short) and mm-scale depth image format
  bool RenderW(Image3b* color, Image1w* depth, Image3f* normal, Image1b* mask,
               Image1i* face_id) const override;
  bool RenderDepthW(Image1w* depth) const override;
};

}  // namespace ugu

#ifndef UGU_STATIC_LIBRARY
#include "rasterizer.cc"
#endif