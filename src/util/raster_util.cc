/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/util/raster_util.h"

namespace ugu {
bool GenerateUvMask(const std::vector<Eigen::Vector2f>& uvs,
                    const std::vector<Eigen::Vector3i>& uv_indices,
                    Image1b& mask, uint8_t color, int width, int height,
                    int num_threads) {
  ugu::Image3b mask3b = ugu::Image3b::zeros(height, width);
  float c_ = static_cast<float>(color);
  bool ret = GenerateUvMask(uvs, uv_indices, mask3b, {c_, c_, c_}, width,
                            height, num_threads);
  ugu::Image1b mask_ = ugu::Image1b::zeros(height, width);
  for (int y = 0; y < mask_.rows; y++) {
    for (int x = 0; x < mask_.cols; x++) {
      mask_.at<uint8_t>(y, x) = mask3b.at<Vec3b>(y, x)[0];
    }
  }
  mask = mask_;
  return ret;
}

}  // namespace ugu
