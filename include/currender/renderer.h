/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <memory>
#include <vector>

#include "currender/camera.h"
#include "currender/mesh.h"

namespace currender {

// Diffuse color
enum class DiffuseColor {
  kNone = 0,     // Default white color
  kTexture = 1,  // From diffuse uv texture
  kVertex = 2    // From vertex color
};

// Normal used for shading
// Also returned as output normal
enum class ShadingNormal {
  kFace = 0,   // Face normal
  kVertex = 1  // Vertex normal. Maybe average of face normals
};

// Diffuse shading
// Light ray same to viewing ray is used for shading
enum class DiffuseShading {
  kNone = 0,        // No shading
  kLambertian = 1,  // Lambertian reflectance model
  kOrenNayar =
      2  // Simplified Oren-Nayar reflectatnce model described in wikipedia
         // https://en.wikipedia.org/wiki/Oren%E2%80%93Nayar_reflectance_model
};

// Interpolation method in texture uv space
// Meaningful only if DiffuseColor::kTexture is specified otherwise ignored
enum class ColorInterpolation {
  kNn = 0,       // Nearest Neigbor
  kBilinear = 1  // Bilinear interpolation
};

struct RendererOption {
  DiffuseColor diffuse_color{DiffuseColor::kNone};
  ColorInterpolation interp{ColorInterpolation::kBilinear};
  ShadingNormal shading_normal{ShadingNormal::kVertex};
  DiffuseShading diffuse_shading{DiffuseShading::kNone};

  float depth_scale{1.0f};       // Multiplied to output depth
  bool backface_culling{true};   // Back-face culling flag
  float oren_nayar_sigma{0.3f};  // Oren-Nayar's sigma

  RendererOption() {}
  ~RendererOption() {}
  void RendererOption::CopyTo(RendererOption* dst) const {
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
class Renderer {
 public:
  virtual ~Renderer() {}

  // Set option
  virtual void set_option(const RendererOption& option) = 0;

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

}  // namespace currender
