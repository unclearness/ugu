/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/image.h"

namespace ugu {

struct StereoParam {
  float fx, fy;
  float lcx, lcy, rcx, rcy;
  float baseline;

  float mind = 0.0f;
  float maxd = 10000.0f;
};

bool Disparity2Depth(const Image1f& disparity, Image1f* depth, float baseline,
                     float fx, float lcx, float rcx, float mind, float maxd);

bool ComputeStereoBruteForce(const Image1b& left, const Image1b& right,
                             Image1f* disparity, Image1f* cost, Image1f* depth,
                             const StereoParam& param, int kernel = 11,
                             float max_disparity = -1.0f);

bool ComputePatchMatchStereo(const Image3b& left, const Image3b& right,
                             Image1f* disparity, Image1f* depth,
                             bool temporal = false, bool refinement = true);

}  // namespace ugu