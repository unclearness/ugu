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

  float max_disparity = -1.0f;
  int kernel = 11;

  float subpixel_step = 0.1f;
};

struct PatchMatchStereoParam {
  StereoParam base_param;

  int iter = 5;
  int random_seed = 0;

  float alpha;
  bool fronto_parallel_window = false;
  float initial_random_disparity_range = -1.0f;

  bool view_propagation = true;
  bool temporal_propagation = false;
  bool plane_refinement = true;

  bool left_right_consistency = true;
};

bool Disparity2Depth(const Image1f& disparity, Image1f* depth, float baseline,
                     float fx, float lcx, float rcx, float mind, float maxd);

bool ComputeStereoBruteForce(const Image1b& left, const Image1b& right,
                             Image1f* disparity, Image1f* cost, Image1f* depth,
                             const StereoParam& param);

bool ComputeStereoBruteForce(const Image3b& left, const Image3b& right,
                             Image1f* disparity, Image1f* cost, Image1f* depth,
                             const StereoParam& param);

bool ComputePatchMatchStereo(const Image3b& left, const Image3b& right,
                             Image1f* disparity, Image1f* cost, Image1f* depth,
                             const PatchMatchStereoParam& param);

}  // namespace ugu