#pragma once

#include <memory>

#include "camera.h"
#include "common.h"
#include "mesh.h"

#include "nanort.h"

namespace unclearness {

class CpuRendererOption {
 public:
  bool use_vertex_color{false};
  float depth_scale{1.0f};
  enum ColorInterpolation{
      NN = 0,
      BILINEAR = 1
  };
  ColorInterpolation interp {BILINEAR};
};

class CpuRenderer {
  bool mesh_initialized_{false};
  std::shared_ptr<Camera> camera_;
  std::shared_ptr<Mesh> mesh_;
  CpuRendererOption option_;

  std::vector<float> flatten_vertices;
  std::vector<unsigned int> flatten_faces;

  nanort::BVHBuildOptions<float> build_options;
  std::unique_ptr<nanort::TriangleMesh<float>> triangle_mesh;
  std::unique_ptr<nanort::TriangleSAHPred<float>> triangle_pred;
  nanort::BVHAccel<float> accel;
  nanort::BVHBuildStatistics stats;
  float bmin[3], bmax[3];

 public:
  CpuRenderer();
  ~CpuRenderer();
  void set_mesh(std::shared_ptr<Mesh> mesh);
  bool prepare_mesh();
  void set_camera(std::shared_ptr<Camera> camera);
  bool render(Image3b& color, Image1w& depth, Image1b& mask);
};

}  // namespace unclearness