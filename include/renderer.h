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

namespace crender {

struct RendererOption {
  bool use_vertex_color{false};
  float depth_scale{1.0f};
  enum ColorInterpolation { kNn = 0, kBilinear = 1 };
  ColorInterpolation interp{kBilinear};
  bool backface_culling{true};

  RendererOption();
  ~RendererOption();
  void CopyTo(RendererOption* dst) const;
};

class Renderer {
  bool mesh_initialized_{false};
  std::shared_ptr<Camera> camera_;
  std::shared_ptr<Mesh> mesh_;
  RendererOption option_;

  std::vector<float> flatten_vertices_;
  std::vector<unsigned int> flatten_faces_;

  nanort::BVHBuildOptions<float> build_options_;
  std::unique_ptr<nanort::TriangleMesh<float>> triangle_mesh_;
  std::unique_ptr<nanort::TriangleSAHPred<float>> triangle_pred_;
  nanort::BVHAccel<float> accel_;
  nanort::BVHBuildStatistics stats_;
  float bmin_[3], bmax_[3];

 public:
  Renderer();
  ~Renderer();
  explicit Renderer(const RendererOption& option);
  void set_option(const RendererOption& option);
  void set_mesh(std::shared_ptr<Mesh> mesh);
  bool PrepareMesh();
  void set_camera(std::shared_ptr<Camera> camera);
  bool Render(Image3b* color, Image1f* depth, Image1b* mask) const;
  bool Render(Image3b* color, Image1w* depth, Image1b* mask) const;
};

}  // namespace crender
