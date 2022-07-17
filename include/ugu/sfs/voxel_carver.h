/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <memory>
#include <vector>

#include "ugu/camera.h"
#include "ugu/common.h"
#include "ugu/image.h"
#include "ugu/mesh.h"
#include "ugu/voxel/voxel.h"

namespace ugu {

struct VoxelCarverOption {
  Eigen::Vector3f bb_max;
  Eigen::Vector3f bb_min;
  float resolution{0.1f};  // default is 10cm if input is m-scale
  bool sdf_minmax_normalize{true};
  VoxelUpdateOption update_option;
};

class VoxelCarver {
  VoxelCarverOption option_;
  std::unique_ptr<VoxelGrid> voxel_grid_;

 public:
  VoxelCarver();
  ~VoxelCarver();
  explicit VoxelCarver(VoxelCarverOption option);
  void set_option(VoxelCarverOption option);
  bool Init();
  bool Carve(const Camera& camera, const Image1b& silhouette,
             const Eigen::Vector2i& roi_min, const Eigen::Vector2i& roi_max,
             Image1f* sdf);
  bool Carve(const Camera& camera, const Image1b& silhouette, Image1f* sdf);
  bool Carve(const Camera& camera, const Image1b& silhouette);
  bool Carve(const std::vector<Camera>& cameras,
             const std::vector<Image1b>& silhouettes);
  void ExtractVoxel(Mesh* mesh, bool inside_empty = false);
  void ExtractIsoSurface(Mesh* mesh, double iso_level = 0.0);
};

}  // namespace ugu
