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

enum class DiffuseColor { kNone = 0, kTexture = 1, kVertex = 2 };
enum class ShadingNormal { kFace = 0, kVertex = 1 };
enum class DiffuseShading { kNone = 0, kLambertian = 1, kOrenNayar = 2 };
enum class ColorInterpolation { kNn = 0, kBilinear = 1 };

struct RendererOption {
  DiffuseColor diffuse_color{DiffuseColor::kTexture};
  float depth_scale{1.0f};
  ColorInterpolation interp{ColorInterpolation::kBilinear};
  ShadingNormal shading_normal{ShadingNormal::kVertex};
  DiffuseShading diffuse_shading{DiffuseShading::kLambertian};
  bool backface_culling{true};
  float oren_nayar_sigma{0.3f};
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
                                   Image3f* normal, Image1b* mask) const;

 public:
  Renderer();
  ~Renderer();
  explicit Renderer(const RendererOption& option);
  void set_option(const RendererOption& option);
  void set_mesh(std::shared_ptr<const Mesh> mesh);
  bool PrepareMesh();
  void set_camera(std::shared_ptr<const Camera> camera);
  bool Render(Image3b* color, Image1f* depth, Image3f* normal,
              Image1b* mask) const;

  // This Image1w* depth interface is prepared for widely used 16 bit (unsigned
  // short), mm scale depth image format
  bool Render(Image3b* color, Image1w* depth, Image3f* normal,
              Image1b* mask) const;
};

}  // namespace currender
