/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <vector>

#include "ugu/camera.h"
#include "ugu/image.h"

namespace ugu {

// Voxel update type
enum class VoxelUpdate {
  kMax = 0,             // take max. naive voxel carving
  kWeightedAverage = 1  // weighted average like KinectFusion. truncation is
                        // necessary to get good result
};

// Interpolation method for 2D SDF
enum class SdfInterpolation {
  kNn = 0,       // Nearest Neigbor
  kBilinear = 1  // Bilinear interpolation
};

// The way to update voxels which are projected to outside of the current image
enum class UpdateOutsideImage {
  kNone = 0,  // Do nothing
  kMax = 1    // Fill by max sdf of the current image. This is valid only when
              // silhouette is not protruding over the current image edge
};

struct VoxelUpdateOption {
  // Followed OpenCV kinfu
  // https://github.com/opencv/opencv_contrib/blob/9d0a451bee4cdaf9d3f76912e5abac6000865f1a/modules/rgbd/src/kinfu.cpp#L66
  static inline float kDefaultResToTruncFactor = 7.f;

  VoxelUpdate voxel_update{VoxelUpdate::kMax};
  SdfInterpolation sdf_interp{SdfInterpolation::kBilinear};
  UpdateOutsideImage update_outside{UpdateOutsideImage::kNone};
  int voxel_max_update_num{
      255};  // After updating voxel_max_update_num, no sdf update
  float voxel_update_weight{1.0f};  // only valid if kWeightedAverage is set
  bool use_truncation{false};
  float truncation_band{0.1f};  // only positive value is valid

  VoxelUpdateOption() = default;
  VoxelUpdateOption(const VoxelUpdateOption& option) = default;

  VoxelUpdateOption(const VoxelUpdate& voxel_update,
                    const SdfInterpolation& sdf_interp,
                    const UpdateOutsideImage& update_outside,
                    int voxel_max_update_num, float voxel_update_weight,
                    bool use_truncation, float truncation_band)
      : voxel_update(voxel_update),
        sdf_interp(sdf_interp),
        update_outside(update_outside),
        voxel_max_update_num(voxel_max_update_num),
        voxel_update_weight(voxel_update_weight),
        use_truncation(use_truncation),
        truncation_band(truncation_band){};

  ~VoxelUpdateOption() = default;
};

struct Voxel {
  Eigen::Vector3i index{-1, -1, -1};  // voxel index
  int id{-1};
  Eigen::Vector3f pos{0.0f, 0.0f, 0.0f};  // center of voxel
  Eigen::Vector3f col{0.0f, 0.0f, 0.0f};
  float sdf{0.0f};  // Signed Distance Function (SDF) value
  int update_num{0};
  bool outside{false};
  bool on_surface{false};
  Voxel();
  ~Voxel();
};

class VoxelGrid {
  std::vector<Voxel> voxels_;
  Eigen::Vector3f bb_max_;
  Eigen::Vector3f bb_min_;
  float resolution_{-1.0f};
  Eigen::Vector3i voxel_num_{0, 0, 0};
  int xy_slice_num_{0};

  // index to pos
  std::vector<float> x_pos_list;
  std::vector<float> y_pos_list;
  std::vector<float> z_pos_list;

 public:
  VoxelGrid();
  ~VoxelGrid();
  bool Init(const Eigen::Vector3f& bb_max, const Eigen::Vector3f& bb_min,
            float resolution);
  const Eigen::Vector3i& voxel_num() const;
  const Voxel& get(int x, int y, int z) const;
  Voxel* get_ptr(int x, int y, int z);
  float resolution() const;
  void ResetOnSurface();
  bool initialized() const;
  Eigen::Vector3i get_index(const Eigen::Vector3f& p) const;
  void Clear();
};

VoxelUpdateOption GenFuseDepthDefaultOption(float resolution);

bool FuseDepth(const Camera& camera, const Image1f& depth,
               const VoxelUpdateOption& option, VoxelGrid& voxel_grid);

bool FuseDepth(const Camera& camera, const Image1f& depth,
               const Image1f& normal, const VoxelUpdateOption& option,
               VoxelGrid& voxel_grid, int sample_num = 1,
               const Image3b& color = Image3b());

bool FusePoints(const std::vector<Eigen::Vector3f>& points,
                const std::vector<Eigen::Vector3f>& normals,
                const VoxelUpdateOption& option, VoxelGrid& voxel_grid,
                const std::vector<Eigen::Vector3f>& colors = {},
                int sample_num = 1);

float SdfInterpolationNn(const Eigen::Vector2f& image_p,
                         const ugu::Image1f& sdf,
                         const Eigen::Vector2i& roi_min,
                         const Eigen::Vector2i& roi_max);

float SdfInterpolationBiliner(const Eigen::Vector2f& image_p,
                              const ugu::Image1f& sdf,
                              const Eigen::Vector2i& roi_min,
                              const Eigen::Vector2i& roi_max);

void UpdateVoxelMax(ugu::Voxel* voxel, const ugu::VoxelUpdateOption& option,
                    float sdf, bool with_color = false,
                    const Eigen::Vector3f& col = Eigen::Vector3f::Zero());

void UpdateVoxelWeightedAverage(
    ugu::Voxel* voxel, const ugu::VoxelUpdateOption& option, float sdf,
    bool with_color = false,
    const Eigen::Vector3f& col = Eigen::Vector3f::Zero());

}  // namespace ugu
