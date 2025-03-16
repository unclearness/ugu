#pragma once

#include <memory>
#include <vector>

#include "Eigen/Core"

namespace ugu {

struct KNnGridSearchResult {
  size_t index;
  double dist;
};

class KNnGridCuda {
 public:
  KNnGridCuda();
  ~KNnGridCuda();

  void SetData(const std::vector<Eigen::Vector3f>& data);
  void SetVoxelLen(float voxel_len);
  void Build();
  std::vector<std::vector<KNnGridSearchResult>> SearchKnn(
      const std::vector<Eigen::Vector3f>& queries, uint32_t k) const;
  std::vector<KNnGridSearchResult> SearchKnn(const Eigen::Vector3f& query,
                                             uint32_t k) const;

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl_;
};
}  // namespace ugu
