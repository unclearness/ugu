/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "ugu/sfs/voxel_carver.h"

#include <array>

#include "ugu/timer.h"
#include "ugu/util/image_util.h"
#include "ugu/util/rgbd_util.h"
#include "ugu/voxel/extract_voxel.h"
#include "ugu/voxel/marching_cubes.h"

namespace {}  // namespace

namespace ugu {

VoxelCarver::VoxelCarver() {}

VoxelCarver::~VoxelCarver() {}

VoxelCarver::VoxelCarver(VoxelCarverOption option) { set_option(option); }

void VoxelCarver::set_option(VoxelCarverOption option) { option_ = option; }

bool VoxelCarver::Init() {
  if (option_.update_option.voxel_max_update_num < 1) {
    LOGE("voxel_max_update_num must be positive");
    return false;
  }
  if (option_.update_option.voxel_update_weight <
      std::numeric_limits<float>::min()) {
    LOGE("voxel_update_weight must be positive");
    return false;
  }
  if (option_.update_option.truncation_band <
      std::numeric_limits<float>::min()) {
    LOGE("truncation_band must be positive");
    return false;
  }
  voxel_grid_ = std::make_unique<VoxelGrid>();
  return voxel_grid_->Init(option_.bb_max, option_.bb_min, option_.resolution);
}

bool VoxelCarver::Carve(const Camera& camera, const Image1b& silhouette,
                        const Eigen::Vector2i& roi_min,
                        const Eigen::Vector2i& roi_max, Image1f* sdf) {
  if (!voxel_grid_->initialized()) {
    LOGE("VoxelCarver::Carve voxel grid has not been initialized\n");
    return false;
  }

  Timer<> timer;
  timer.Start();
  // make signed distance field
  MakeSignedDistanceField(silhouette, roi_min, roi_max, sdf,
                          option_.sdf_minmax_normalize,
                          option_.update_option.use_truncation,
                          option_.update_option.truncation_band);
  timer.End();
  LOGI("VoxelCarver::Carve make SDF %02f\n", timer.elapsed_msec());
  std::function<float(const Eigen::Vector2f&, const ugu::Image1f&,
                      const Eigen::Vector2i&, const Eigen::Vector2i&)>
      interpolate_sdf;
  if (option_.update_option.sdf_interp == SdfInterpolation::kNn) {
    interpolate_sdf = SdfInterpolationNn;
  } else if (option_.update_option.sdf_interp == SdfInterpolation::kBilinear) {
    interpolate_sdf = SdfInterpolationBiliner;
  }

  std::function<void(Voxel*, const VoxelUpdateOption&, float)> update_voxel;
  if (option_.update_option.voxel_update == VoxelUpdate::kMax) {
    update_voxel = UpdateVoxelMax;
  } else if (option_.update_option.voxel_update ==
             VoxelUpdate::kWeightedAverage) {
    update_voxel = UpdateVoxelWeightedAverage;
  }

  timer.Start();
  double min_sdf = std::numeric_limits<double>::max(),
         max_sdf = std::numeric_limits<double>::lowest();
  ugu::minMaxLoc(*sdf, &min_sdf, &max_sdf);
  const Eigen::Vector3i& voxel_num = voxel_grid_->voxel_num();
  const Eigen::Affine3f& w2c = camera.w2c().cast<float>();
#if defined(_OPENMP) && defined(VACANCY_USE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int z = 0; z < voxel_num.z(); z++) {
    for (int y = 0; y < voxel_num.y(); y++) {
      for (int x = 0; x < voxel_num.x(); x++) {
        Voxel* voxel = voxel_grid_->get_ptr(x, y, z);

        if (voxel->outside ||
            voxel->update_num > option_.update_option.voxel_max_update_num) {
          continue;
        }

        Eigen::Vector2f image_p_f;
        Eigen::Vector3f voxel_pos_c = w2c * voxel->pos;

        // skip if the voxel is in the back of the camera
        if (voxel_pos_c.z() < 0) {
          continue;
        }

        camera.Project(voxel_pos_c, &image_p_f);

        float dist = InvalidSdf::kVal;

        if (image_p_f.x() < roi_min.x() || image_p_f.y() < roi_min.y() ||
            roi_max.x() < image_p_f.x() || roi_max.y() < image_p_f.y()) {
          if (option_.update_option.update_outside ==
              UpdateOutsideImage::kNone) {
            continue;
          } else if (option_.update_option.update_outside ==
                     UpdateOutsideImage::kMax) {
            dist = static_cast<float>(max_sdf);
          }
        } else {
          dist = interpolate_sdf(image_p_f, *sdf, roi_min, roi_max);
        }

        // skip if dist is truncated
        if (option_.update_option.use_truncation && dist < -1.0f) {
          continue;
        }

        if (voxel->update_num < 1) {
          voxel->sdf = dist;
          voxel->update_num++;
          continue;
        }

        update_voxel(voxel, option_.update_option, dist);
      }
    }
  }
  timer.End();
  LOGI("VoxelCarver::Carve main loop %02f\n", timer.elapsed_msec());

  return true;
}

bool VoxelCarver::Carve(const Camera& camera, const Image1b& silhouette) {
  Image1f sdf = Image1f::zeros(camera.height(), camera.width());
  return Carve(camera, silhouette, &sdf);
}

bool VoxelCarver::Carve(const Camera& camera, const Image1b& silhouette,
                        Image1f* sdf) {
  Eigen::Vector2i roi_min{0, 0};
  Eigen::Vector2i roi_max{silhouette.cols - 1, silhouette.rows - 1};
  return Carve(camera, silhouette, roi_min, roi_max, sdf);
}

bool VoxelCarver::Carve(const std::vector<Camera>& cameras,
                        const std::vector<Image1b>& silhouettes) {
  assert(cameras.size() == silhouettes.size());

  for (size_t i = 0; i < cameras.size(); i++) {
    bool ret = Carve(cameras[i], silhouettes[i]);
    if (!ret) {
      return false;
    }
  }

  return true;
}

void VoxelCarver::ExtractVoxel(Mesh* mesh, bool inside_empty) {
  Timer<> timer;
  timer.Start();

  ugu::ExtractVoxel(voxel_grid_.get(), mesh, inside_empty);

  timer.End();
  LOGI("VoxelCarver::ExtractVoxel %02f\n", timer.elapsed_msec());
}

void VoxelCarver::ExtractIsoSurface(Mesh* mesh, double iso_level) {
  MarchingCubes(*voxel_grid_, mesh, iso_level);
}

}  // namespace ugu
