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

  int iter = 3;
  int random_seed = 0;
  int patch_size = 11;

  float gamma = 10.0f;
  float alpha = 0.9f;
  float tau_col = 10.0f;
  float tau_grad = 2.0f;

  bool fronto_parallel_window = false;
  float initial_random_disparity_range = -1.0f;
  float theta_deg_min = 0.0f;
  float theta_deg_max = 75.0f;
  float phi_deg_min = 0.0f;
  float phi_deg_max = 360.0f;

  bool initial_disparity_integer = false;
  bool alternately_reverse = true;

  bool view_propagation = true;
  bool temporal_propagation = false;
  bool plane_refinement = true;

  bool left_right_consistency = true;
  float left_right_consistency_th = 1.0f;
  bool fill_hole_nn = true;
  bool weighted_median_for_filled = true;

  bool debug = true;
  int debug_step = 80;
};

bool Disparity2Depth(const Image1f& disparity, Image1f* depth, float baseline,
                     float fx, float lcx, float rcx, float mind, float maxd);

bool VisualizeCost(const Image1f& cost, Image3b* vis_cost, float minc = -1.0f,
                   float maxc = -1.0f);

bool ComputeStereoBruteForce(const Image1b& left, const Image1b& right,
                             Image1f* disparity, Image1f* cost, Image1f* depth,
                             const StereoParam& param);

bool ComputeStereoBruteForce(const Image3b& left, const Image3b& right,
                             Image1f* disparity, Image1f* cost, Image1f* depth,
                             const StereoParam& param);

bool ComputePatchMatchStereo(const Image3b& left, const Image3b& right,
                             Image1f* ldisparity, Image1f* lcost,
                             Image1f* rdisparity, Image1f* rcost,
                             Image1f* depth,
                             const PatchMatchStereoParam& param);

}  // namespace ugu