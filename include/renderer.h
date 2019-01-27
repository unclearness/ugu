/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <memory>
#include <vector>

#include "include/camera.h"
#include "include/mesh.h"
#include "nanort/nanort.h"

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
  DiffuseColor diffuse_color{DiffuseColor::kTexture};
  ColorInterpolation interp{ColorInterpolation::kBilinear};
  ShadingNormal shading_normal{ShadingNormal::kVertex};
  DiffuseShading diffuse_shading{DiffuseShading::kLambertian};

  float depth_scale{1.0f};       // Multiplied to output depth
  bool backface_culling{true};   // Back-face culling flag
  float oren_nayar_sigma{0.3f};  // Oren-Nayar's sigma

  RendererOption();
  ~RendererOption();
  void CopyTo(RendererOption* dst) const;
};

class Renderer {
  bool mesh_initialized_{false};
  std::shared_ptr<const Camera> camera_{nullptr};
  std::shared_ptr<const Mesh> mesh_{nullptr};
  RendererOption option_;

  std::vector<float> flatten_vertices_;
  std::vector<unsigned int> flatten_faces_;

  nanort::BVHBuildOptions<float> build_options_;
  std::unique_ptr<nanort::TriangleMesh<float>> triangle_mesh_;
  std::unique_ptr<nanort::TriangleSAHPred<float>> triangle_pred_;
  nanort::BVHAccel<float> accel_;
  nanort::BVHBuildStatistics stats_;
  float bmin_[3], bmax_[3];

  bool ValidateAndInitBeforeRender(Image3b* color, Image1f* depth,
                                   Image3f* normal, Image1b* mask,
                                   std::vector<uint32_t>* visible_faces) const;

 public:
  Renderer();
  ~Renderer();

  // Set option
  explicit Renderer(const RendererOption& option);
  void set_option(const RendererOption& option);

  // Set mesh
  void set_mesh(std::shared_ptr<const Mesh> mesh);

  // Should call after set_mesh() and before Render()
  // Don't modify mesh outside after calling PrepareMesh()
  bool PrepareMesh();

  // Set camera
  void set_camera(std::shared_ptr<const Camera> camera);

  // Rendering all images and get face visibility
  bool Render(Image3b* color, Image1f* depth, Image3f* normal, Image1b* mask,
              std::vector<uint32_t>* visible_faces = nullptr) const;

  // Rendering a image
  bool RenderColor(Image3b* color) const;
  bool RenderDepth(Image1f* depth) const;
  bool RenderNormal(Image3f* normal) const;
  bool RenderMask(Image1b* mask) const;
  bool VisibilityTest(std::vector<uint32_t>* visible_faces) const;

  // These Image1w* depth interfaces are prepared for widely used unsigned 16
  // bit (unsigned short) and mm scale depth image format
  bool RenderW(Image3b* color, Image1w* depth, Image3f* normal, Image1b* mask,
               std::vector<uint32_t>* visible_faces = nullptr) const;
  bool RenderDepthW(Image1w* depth) const;
};

}  // namespace currender
