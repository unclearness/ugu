/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "ugu/sfs/voxel_carver.h"

#include <array>

#include "ugu/sfs/extract_voxel.h"
#include "ugu/sfs/marching_cubes.h"
#include "ugu/timer.h"
#include "ugu/util/image_util.h"
#include "ugu/util/rgbd_util.h"

namespace {

inline float SdfInterpolationNn(const Eigen::Vector2f& image_p,
                                const ugu::Image1f& sdf,
                                const Eigen::Vector2i& roi_min,
                                const Eigen::Vector2i& roi_max) {
  Eigen::Vector2i image_p_i(static_cast<int>(std::round(image_p.x())),
                            static_cast<int>(std::round(image_p.y())));

  // really need these?
  if (image_p_i.x() < roi_min.x()) {
    image_p_i.x() = roi_min.x();
  }
  if (image_p_i.y() < roi_min.y()) {
    image_p_i.y() = roi_min.y();
  }
  if (roi_max.x() < image_p_i.x()) {
    image_p_i.x() = roi_max.x();
  }
  if (roi_max.y() < image_p_i.y()) {
    image_p_i.y() = roi_max.y();
  }

  return sdf.at<float>(image_p_i.y(), image_p_i.x());
}

inline float SdfInterpolationBiliner(const Eigen::Vector2f& image_p,
                                     const ugu::Image1f& sdf,
                                     const Eigen::Vector2i& roi_min,
                                     const Eigen::Vector2i& roi_max) {
  std::array<int, 2> pos_min = {{0, 0}};
  std::array<int, 2> pos_max = {{0, 0}};
  pos_min[0] = static_cast<int>(std::floor(image_p[0]));
  pos_min[1] = static_cast<int>(std::floor(image_p[1]));
  pos_max[0] = pos_min[0] + 1;
  pos_max[1] = pos_min[1] + 1;

  // really need these?
  if (pos_min[0] < roi_min.x()) {
    pos_min[0] = roi_min.x();
  }
  if (pos_min[1] < roi_min.y()) {
    pos_min[1] = roi_min.y();
  }
  if (roi_max.x() < pos_max[0]) {
    pos_max[0] = roi_max.x();
  }
  if (roi_max.y() < pos_max[1]) {
    pos_max[1] = roi_max.y();
  }

  float local_u = image_p[0] - pos_min[0];
  float local_v = image_p[1] - pos_min[1];

  // bilinear interpolation of sdf
  float dist =
      (1.0f - local_u) * (1.0f - local_v) *
          sdf.at<float>(pos_min[1], pos_min[0]) +
      local_u * (1.0f - local_v) * sdf.at<float>(pos_max[1], pos_min[0]) +
      (1.0f - local_u) * local_v * sdf.at<float>(pos_min[1], pos_max[0]) +
      local_u * local_v * sdf.at<float>(pos_max[1], pos_max[0]);

  return dist;
}

inline void UpdateVoxelMax(ugu::Voxel* voxel,
                           const ugu::VoxelUpdateOption& option, float sdf) {
  (void)option;
  if (sdf > voxel->sdf) {
    voxel->sdf = sdf;
    voxel->update_num++;
  }
}

inline void UpdateVoxelWeightedAverage(ugu::Voxel* voxel,
                                       const ugu::VoxelUpdateOption& option,
                                       float sdf) {
  const float& w = option.voxel_update_weight;
  const float inv_denom = 1.0f / (w * (voxel->update_num + 1));
  voxel->sdf = (w * voxel->update_num * voxel->sdf + w * sdf) * inv_denom;
  voxel->update_num++;
}
}  // namespace

namespace ugu {

Voxel::Voxel() {}
Voxel::~Voxel() {}

VoxelGrid::VoxelGrid() {}

VoxelGrid::~VoxelGrid() {}

bool VoxelGrid::Init(const Eigen::Vector3f& bb_max,
                     const Eigen::Vector3f& bb_min, float resolution) {
  if (resolution < std::numeric_limits<float>::min()) {
    LOGE("resolution must be positive %f\n", resolution);
    return false;
  }
  if (bb_max.x() <= bb_min.x() || bb_max.y() <= bb_min.y() ||
      bb_max.z() <= bb_min.z()) {
    LOGE("input bounding box is invalid\n");
    return false;
  }

  bb_max_ = bb_max;
  bb_min_ = bb_min;
  resolution_ = resolution;

  Eigen::Vector3f diff = bb_max_ - bb_min_;

  for (int i = 0; i < 3; i++) {
    voxel_num_[i] = static_cast<int>(diff[i] / resolution_);
  }

  if (voxel_num_.x() * voxel_num_.y() * voxel_num_.z() >
      std::numeric_limits<int>::max()) {
    LOGE("too many voxels\n");
    return false;
  }

  xy_slice_num_ = voxel_num_[0] * voxel_num_[1];

  voxels_.clear();
  voxels_.resize(voxel_num_.x() * voxel_num_.y() * voxel_num_.z());

  float offset = resolution_ * 0.5f;

  x_pos_list.resize(voxel_num_.x());
  y_pos_list.resize(voxel_num_.y());
  z_pos_list.resize(voxel_num_.z());

#if defined(_OPENMP) && defined(VACANCY_USE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int z = 0; z < voxel_num_.z(); z++) {
    float z_pos = diff.z() * (static_cast<float>(z) /
                              static_cast<float>(voxel_num_.z())) +
                  bb_min_.z() + offset;
    z_pos_list[z] = z_pos;
    for (int y = 0; y < voxel_num_.y(); y++) {
      float y_pos = diff.y() * (static_cast<float>(y) /
                                static_cast<float>(voxel_num_.y())) +
                    bb_min_.y() + offset;
      if (z == 0) {
        y_pos_list[y] = y_pos;
      }
      for (int x = 0; x < voxel_num_.x(); x++) {
        float x_pos = diff.x() * (static_cast<float>(x) /
                                  static_cast<float>(voxel_num_.x())) +
                      bb_min_.x() + offset;

        if (z == 0 && y == 0) {
          x_pos_list[x] = x_pos;
        }

        Voxel* voxel = get_ptr(x, y, z);
        voxel->index.x() = x;
        voxel->index.y() = y;
        voxel->index.z() = z;

        voxel->id = z * xy_slice_num_ + (y * voxel_num_.x() + x);

        voxel->pos.x() = x_pos;
        voxel->pos.y() = y_pos;
        voxel->pos.z() = z_pos;

        voxel->sdf = InvalidSdf::kVal;
      }
    }
  }

  return true;
}

const Eigen::Vector3i& VoxelGrid::voxel_num() const { return voxel_num_; }

const Voxel& VoxelGrid::get(int x, int y, int z) const {
  return voxels_[z * xy_slice_num_ + (y * voxel_num_.x() + x)];
}

Voxel* VoxelGrid::get_ptr(int x, int y, int z) {
  return &voxels_[z * xy_slice_num_ + (y * voxel_num_.x() + x)];
}

float VoxelGrid::resolution() const { return resolution_; }

void VoxelGrid::ResetOnSurface() {
  for (Voxel& v : voxels_) {
    v.on_surface = false;
  }
}

bool VoxelGrid::initialized() const { return !voxels_.empty(); }

Eigen::Vector3i VoxelGrid::get_index(const Eigen::Vector3f& p) const {
  // TODO: Better way to handle voxel indices
  // Real-time 3D reconstruction at scale using voxel hashing
  // https://niessnerlab.org/papers/2013/4hashing/niessner2013hashing.pdf

  Eigen::Vector3i index{-1, -1, -1};

  // Return -1 if any of xyz is out of range
  auto x_it = std::lower_bound(x_pos_list.begin(), x_pos_list.end(), p.x());
  if (x_it == x_pos_list.end()) {
    return index;
  }
  auto y_it = std::lower_bound(y_pos_list.begin(), y_pos_list.end(), p.y());
  if (y_it == y_pos_list.end()) {
    return index;
  }
  auto z_it = std::lower_bound(z_pos_list.begin(), z_pos_list.end(), p.z());
  if (z_it == z_pos_list.end()) {
    return index;
  }

  if (x_it != x_pos_list.begin()) {
    float x0 = std::abs(*x_it - p.x());
    float x1 = std::abs(*(x_it - 1) - p.x());
    if (x1 < x0) {
      x_it--;
    }
  }

  if (y_it != y_pos_list.begin()) {
    float y0 = std::abs(*y_it - p.y());
    float y1 = std::abs(*(y_it - 1) - p.y());
    if (y1 < y0) {
      y_it--;
    }
  }

  if (z_it != z_pos_list.begin()) {
    float z0 = std::abs(*z_it - p.z());
    float z1 = std::abs(*(z_it - 1) - p.z());
    if (z1 < z0) {
      z_it--;
    }
  }

  index.x() = std::distance(x_pos_list.begin(), x_it);
  index.y() = std::distance(y_pos_list.begin(), y_it);
  index.z() = std::distance(z_pos_list.begin(), z_it);

  return index;
}

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

bool FuseDepth(const Camera& camera, const Image1f& depth,
               const VoxelUpdateOption& option, VoxelGrid& voxel_grid) {
  if (!voxel_grid.initialized()) {
    return false;
  }

#if 0
  Mesh pc;
  Depth2PointCloud(depth, camera, &pc);
  pc.Transform(camera.c2w().cast<float>());

  Eigen::Vector3f ray;
  // TODO: Set principal point properly
  camera.ray_w(static_cast<float>(camera.width()) * 0.5f,
               static_cast<float>(camera.height()) * 0.5f, &ray);

  for (const auto& p : pc.vertices()) {
    auto index = voxel_grid.get_index(p);
    if (index.x() < 0 || index.y() < 0 || index.z() < 0) {
      continue;
    }
    auto voxel = voxel_grid.get_ptr(index.x(), index.y(), index.z());

    const Eigen::Vector3f diff = voxel->pos - p;
    // If dot is negative, point is behind the voxel
    float sdf = diff.dot(ray) < 0 ? diff.norm() : -diff.norm();

    // skip if dist is truncated
    if (option.use_truncation) {
      if (-option.truncation_band >= sdf) {
        sdf = InvalidSdf::kVal;
      } else {
        sdf = std::min(1.0f, sdf / option.truncation_band);
      }

      if (sdf < -1.f) {
        continue;
      }
    }

    if (voxel->update_num < 1) {
      voxel->sdf = sdf;
      voxel->update_num++;
      continue;
    }

    UpdateVoxelWeightedAverage(voxel, option, sdf);
  }
#endif

  Eigen::Affine3f w2c = camera.w2c().cast<float>();

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif

  for (int z = 0; z < voxel_grid.voxel_num().z(); z++) {
    for (int y = 0; y < voxel_grid.voxel_num().y(); y++) {
      for (int x = 0; x < voxel_grid.voxel_num().x(); x++) {
        Voxel* voxel = voxel_grid.get_ptr(x, y, z);

        if (voxel->outside || voxel->update_num > option.voxel_max_update_num) {
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
        float d = 0.f;
        if (image_p_f.x() < 0 || image_p_f.y() < 0 ||
            depth.cols - 1 < image_p_f.x() || depth.rows - 1 < image_p_f.y()) {
          continue;
        } else {
          d = SdfInterpolationNn(image_p_f, depth, {0, 0},
                                 {depth.cols - 1, depth.rows - 1});
        }

        if (d < std::numeric_limits<float>::epsilon()) {
          continue;
        }

        float sign = std::signbit(d - voxel_pos_c.z()) ? -1.f : 1.f;
#if 0
        dist = d - voxel_pos_c.z();
#else
        // Use distance along ray
        Eigen::Vector3f camera_p;
        camera.Unproject({image_p_f.x(), image_p_f.y(), d}, &camera_p);
        dist = (voxel_pos_c - camera_p).norm() * sign;
#endif

        // skip if dist is truncated
        if (option.use_truncation) {
          if (dist >= -option.truncation_band) {
            dist = std::min(1.0f, dist / option.truncation_band);
          } else {
            continue;
          }
        }

        if (voxel->update_num < 1) {
          voxel->sdf = dist;
          voxel->update_num++;
          continue;
        }

        UpdateVoxelWeightedAverage(voxel, option, dist);
      }
    }
  }

  return true;
}

}  // namespace ugu
