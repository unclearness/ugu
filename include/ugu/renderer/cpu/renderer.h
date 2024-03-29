/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <memory>
#include <vector>

#include "ugu/camera.h"
#include "ugu/mesh.h"
#include "ugu/renderer/base.h"

namespace ugu {

struct RendererCpuOption {
  DiffuseColor diffuse_color{DiffuseColor::kNone};
  // Meaningful only if DiffuseColor::kTexture is specified otherwise ignored
  ColorInterpolation interp{ColorInterpolation::kBilinear};
  ShadingNormal shading_normal{ShadingNormal::kVertex};
  DiffuseShading diffuse_shading{DiffuseShading::kNone};

  float depth_scale{1.0f};       // Multiplied to output depth
  bool backface_culling{true};   // Back-face culling flag
  float oren_nayar_sigma{0.3f};  // Oren-Nayar's sigma

  RendererCpuOption() {}
  ~RendererCpuOption() {}
  void CopyTo(RendererCpuOption* dst) const {
    dst->diffuse_color = diffuse_color;
    dst->depth_scale = depth_scale;
    dst->interp = interp;
    dst->shading_normal = shading_normal;
    dst->diffuse_shading = diffuse_shading;
    dst->backface_culling = backface_culling;
  }
};

// interface (pure abstract base class with no state or defined methods) for
// renderer
class RendererCpu {
 public:
  virtual ~RendererCpu() {}

  // Set option
  virtual void set_option(const RendererCpuOption& option) = 0;

  // Set mesh
  virtual void set_mesh(std::shared_ptr<const Mesh> mesh) = 0;

  // Should call after set_mesh() and before Render()
  // Don't modify mesh outside after calling PrepareMesh()
  virtual bool PrepareMesh() = 0;

  // Set camera
  virtual void set_camera(std::shared_ptr<const Camera> camera) = 0;

  // Rendering all images
  // If you don't need some of them, pass nullptr
  virtual bool Render(Image3b* color, Image1f* depth, Image3f* normal,
                      Image1b* mask, Image1i* face_id) const = 0;

  // Rendering a image
  virtual bool RenderColor(Image3b* color) const = 0;
  virtual bool RenderDepth(Image1f* depth) const = 0;
  virtual bool RenderNormal(Image3f* normal) const = 0;
  virtual bool RenderMask(Image1b* mask) const = 0;
  virtual bool RenderFaceId(Image1i* face_id) const = 0;

  // These Image1w* depth interfaces are prepared for widely used 16 bit
  // (unsigned short) and mm-scale depth image format
  virtual bool RenderW(Image3b* color, Image1w* depth, Image3f* normal,
                       Image1b* mask, Image1i* face_id) const = 0;
  virtual bool RenderDepthW(Image1w* depth) const = 0;
};

using RendererCpuPtr = std::shared_ptr<RendererCpu>;

}  // namespace ugu
