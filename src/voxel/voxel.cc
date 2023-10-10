/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "ugu/voxel/voxel.h"

#include <array>

#include "ugu/timer.h"
#include "ugu/util/image_util.h"
#include "ugu/util/rgbd_util.h"

namespace {
using namespace ugu;
inline void UpdateVoxel(const Eigen::Vector3i& voxel_idx,
                        const VoxelUpdateOption& option,
                        const Eigen::Vector3f& p, const Eigen::Vector3f& n,
                        const Eigen::Vector3f& c, bool with_color,
                        VoxelGrid& voxel_grid) {
  if (voxel_idx[0] < 0 || voxel_grid.voxel_num()[0] <= voxel_idx[0] ||
      voxel_idx[1] < 0 || voxel_grid.voxel_num()[1] <= voxel_idx[1] ||
      voxel_idx[2] < 0 || voxel_grid.voxel_num()[2] <= voxel_idx[2]) {
    return;
  }
  auto voxel = voxel_grid.get_ptr(voxel_idx[0], voxel_idx[1], voxel_idx[2]);
  const auto diff = voxel->pos - p;
  float sign = std::signbit(diff.dot(n)) ? -1.f : 1.f;
  float dist = diff.norm() * sign;

  // skip if dist is truncated
  if (option.use_truncation) {
    if (dist >= -option.truncation_band) {
      dist = std::min(1.0f, dist / option.truncation_band);
    } else {
      return;
    }
  }

  if (voxel->update_num < 1) {
    voxel->sdf = dist;
    voxel->col = c;
    voxel->update_num++;
    return;
  }

  if (option.voxel_update == VoxelUpdate::kMax) {
    UpdateVoxelMax(voxel, option, dist, with_color, c);
  } else if (option.voxel_update == VoxelUpdate::kWeightedAverage) {
    UpdateVoxelWeightedAverage(voxel, option, dist, with_color, c);
  }
}

inline void FusePointBase(const Eigen::Vector3f& p, const Eigen::Vector3f& n,
                          bool with_color, const Eigen::Vector3f& c,
                          const VoxelUpdateOption& option,
                          VoxelGrid& voxel_grid, int sample_num) {
  const auto& voxel_idx = voxel_grid.get_index(p);
  if (voxel_idx[0] < 0 || voxel_grid.voxel_num()[0] <= voxel_idx[0] ||
      voxel_idx[1] < 0 || voxel_grid.voxel_num()[1] <= voxel_idx[1] ||
      voxel_idx[2] < 0 || voxel_grid.voxel_num()[2] <= voxel_idx[2]) {
    return;
  }

  if (sample_num < 1) {
    // Splat to 26-neighbors
    for (int z = -1; z <= 1; z++) {
      for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
          UpdateVoxel(voxel_idx + Eigen::Vector3i(x, y, z), option, p, n, c,
                      with_color, voxel_grid);
        }
      }
    }
  } else {
    float step = voxel_grid.resolution();
    // Sample along with normal direction
    for (int k = -sample_num; k < sample_num + 1; k++) {
      Eigen::Vector3f offset = n * k * step;
      const auto& voxel_idx_ = voxel_grid.get_index(p + offset);
      if (voxel_idx_[0] < 0 || voxel_idx_[1] < 0 || voxel_idx_[2] < 0) {
        return;
      }
      auto voxel =
          voxel_grid.get_ptr(voxel_idx_[0], voxel_idx_[1], voxel_idx_[2]);
      const auto diff = voxel->pos - p;
      float sign = std::signbit(diff.dot(n)) ? -1.f : 1.f;
      float dist = diff.norm() * sign;

      // skip if dist is truncated
      if (option.use_truncation) {
        if (dist >= -option.truncation_band) {
          dist = std::min(1.0f, dist / option.truncation_band);
        } else {
          return;
        }
      }

      if (voxel->update_num < 1) {
        voxel->sdf = dist;
        voxel->col = c;
        voxel->update_num++;
        return;
      }

      if (option.voxel_update == VoxelUpdate::kMax) {
        UpdateVoxelMax(voxel, option, dist, with_color, c);
      } else if (option.voxel_update == VoxelUpdate::kWeightedAverage) {
        UpdateVoxelWeightedAverage(voxel, option, dist, with_color, c);
      }
    }
  }
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

#if defined(_OPENMP) && defined(UGU_USE_OPENMP)
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

std::vector<Voxel>& VoxelGrid::get_all() { return voxels_; }

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

  index.x() = static_cast<int>(std::distance(x_pos_list.begin(), x_it));
  index.y() = static_cast<int>(std::distance(y_pos_list.begin(), y_it));
  index.z() = static_cast<int>(std::distance(z_pos_list.begin(), z_it));

  return index;
}

void VoxelGrid::Clear() {
#pragma omp parallel for
  for (int64_t i = 0; i < static_cast<int64_t>(voxels_.size()); i++) {
    voxels_[i].sdf = 0.f;
    voxels_[i].update_num = 0;
  }
}

VoxelUpdateOption GenFuseDepthDefaultOption(float resolution) {
  VoxelUpdateOption option;

  option.update_outside = UpdateOutsideImage::kNone;
  option.sdf_interp = SdfInterpolation::kNn;
  option.voxel_update = ugu::VoxelUpdate::kWeightedAverage;
  option.use_truncation = true;
  option.truncation_band =
      resolution * VoxelUpdateOption::kDefaultResToTruncFactor;

  return option;
}

bool FuseDepth(const Camera& camera, const Image1f& depth,
               const VoxelUpdateOption& option, VoxelGrid& voxel_grid) {
  if (!voxel_grid.initialized()) {
    return false;
  }

  Eigen::Affine3f w2c = camera.w2c().cast<float>();

  // https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ismar2011.pdf
  // 3.3 Mapping as Surface Reconstruction
  // "Instead, we use a projective truncated signed distance function that is
  // readily computed and trivially parallelisable."
#if defined(_OPENMP) && defined(UGU_USE_OPENMP)
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
        // Don't use depth difference as distance
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

bool FuseDepth(const Camera& camera, const Image1f& depth,
               const Image3f& normal, const VoxelUpdateOption& option,
               VoxelGrid& voxel_grid, int sample_num, const Image3b& color) {
  if (!voxel_grid.initialized()) {
    return false;
  }

  bool with_color = depth.cols == color.cols && depth.rows == color.rows;

  Eigen::Affine3f c2w = camera.c2w().cast<float>();
  Eigen::Matrix3f c2w_R = c2w.rotation();

  for (int y = 0; y < camera.height(); y++) {
    for (int x = 0; x < camera.width(); x++) {
      const float& d = depth.at<float>(y, x);
      if (d < std::numeric_limits<float>::min()) {
        continue;
      }
      const auto& n = normal.at<Vec3f>(y, x);
      if (std::abs(1.f - (n[0] * n[0] + n[1] * n[1] + n[2] * n[2])) > 0.01f) {
        continue;
      }

      Eigen::Vector2f img_p{static_cast<float>(x), static_cast<float>(y)};
      Eigen::Vector3f camera_p;
      camera.Unproject(img_p, d, &camera_p);
      Eigen::Vector3f wld_p = c2w * camera_p;

      Eigen::Vector3f camera_n{n[0], n[1], n[2]};
      Eigen::Vector3f wld_n = c2w_R * camera_n;

      Eigen::Vector3f c = Eigen::Vector3f::Zero();
      FusePointBase(wld_p, wld_n, with_color, c, option, voxel_grid,
                    sample_num);
    }
  }
  return true;
}

bool FusePointCloudImage(const Eigen::Affine3f c2w, const Image3f& point_cloud,
                         const Image3f& normal, const VoxelUpdateOption& option,
                         VoxelGrid& voxel_grid, int sample_num,
                         const Image3b& color) {
  if (!voxel_grid.initialized()) {
    return false;
  }

  bool with_color =
      point_cloud.cols == color.cols && point_cloud.rows == color.rows;

  Eigen::Matrix3f c2w_R = c2w.rotation();

  for (int y = 0; y < point_cloud.rows; y++) {
    for (int x = 0; x < point_cloud.cols; x++) {
      const auto& n = normal.at<Vec3f>(y, x);
      if (std::abs(1.f - (n[0] * n[0] + n[1] * n[1] + n[2] * n[2])) > 0.01f) {
        continue;
      }

      const auto& cam_p = point_cloud.at<Vec3f>(y, x);
      Eigen::Vector3f camera_p{cam_p[0], cam_p[1], cam_p[2]};
      Eigen::Vector3f wld_p = c2w * camera_p;

      Eigen::Vector3f camera_n{n[0], n[1], n[2]};
      Eigen::Vector3f wld_n = c2w_R * camera_n;

      Eigen::Vector3f c = Eigen::Vector3f::Zero();
      FusePointBase(wld_p, wld_n, with_color, c, option, voxel_grid,
                    sample_num);
    }
  }
  return true;
}

bool FusePoints(const std::vector<Eigen::Vector3f>& points,
                const std::vector<Eigen::Vector3f>& normals,
                const VoxelUpdateOption& option, VoxelGrid& voxel_grid,
                const std::vector<Eigen::Vector3f>& colors, int sample_num) {
  bool with_color = points.size() == colors.size();

  if (!voxel_grid.initialized()) {
    return false;
  }

  // #pragma omp parallel for schedule(dynamic, 1)
  for (int64_t i = 0; i < static_cast<int64_t>(points.size()); i++) {
    const auto& p = points[i];
    const auto& n = normals[i];
    Eigen::Vector3f c = Eigen::Vector3f::Zero();
    if (with_color) {
      c = colors[i];
    }
    FusePointBase(p, n, with_color, c, option, voxel_grid, sample_num);
  }

  return true;
}

float SdfInterpolationNn(const Eigen::Vector2f& image_p,
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

float SdfInterpolationBiliner(const Eigen::Vector2f& image_p,
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

void UpdateVoxelMax(ugu::Voxel* voxel, const ugu::VoxelUpdateOption& option,
                    float sdf, bool with_color, const Eigen::Vector3f& col) {
  (void)option;
  if (sdf > voxel->sdf) {
    voxel->sdf = sdf;
    if (with_color) {
      voxel->col = col;
    }
    voxel->update_num++;
  }
}

void UpdateVoxelWeightedAverage(ugu::Voxel* voxel,
                                const ugu::VoxelUpdateOption& option, float sdf,
                                bool with_color, const Eigen::Vector3f& col) {
  const float& w = option.voxel_update_weight;
  const float inv_denom = 1.0f / (w * (voxel->update_num + 1));
  voxel->sdf = (w * voxel->update_num * voxel->sdf + w * sdf) * inv_denom;

  if (with_color) {
    voxel->col = (w * voxel->update_num * voxel->col + w * col) * inv_denom;
  }

  voxel->update_num++;
}

}  // namespace ugu
