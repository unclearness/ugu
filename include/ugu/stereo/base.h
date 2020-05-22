/*
 * Copyright (C) 2020, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/image.h"

namespace ugu {

enum class StereoCost {
  SSD = 0,
  SAD = 1,
  HAMMING = 2  // only with census_transform
};

enum class HoleFilling { NONE = 0, NN = 1 };

enum class FilterForFilled { NONE = 0, MEDIAN = 1 };

template <typename T, typename TT>
TT Sad(const ugu::Image<T>& a, const ugu::Image<T>& b, int minx, int maxx,
       int miny, int maxy, int offsetx, int offsety) {
  TT cost = TT(0);
  for (int jj = miny; jj <= maxy; jj++) {
    for (int ii = minx; ii <= maxx; ii++) {
      const T& a_val = a.at<T>(jj, ii);
      const T& b_val = b.at<T>(jj + offsety, ii + offsetx);
      for (int c = 0; c < a.channels(); c++) {
        cost += static_cast<TT>(std::abs(a_val[c] - b_val[c]));
      }
    }
  }
  return cost;
}

template <typename T, typename TT>
TT Ssd(const ugu::Image<T>& a, const ugu::Image<T>& b, int minx, int maxx,
       int miny, int maxy, int offsetx, int offsety) {
  TT cost = TT(0);
  for (int jj = miny; jj <= maxy; jj++) {
    for (int ii = minx; ii <= maxx; ii++) {
      const T& a_val = a.at<T>(jj, ii);
      const T& b_val = b.at<T>(jj + offsety, ii + offsetx);
      for (int c = 0; c < a.channels(); c++) {
        cost += static_cast<TT>((a_val[c] - b_val[c]) * (a_val[c] - b_val[c]));
      }
    }
  }
  return cost;
}

template <typename T>
T Hamming(T n1, T n2) {
  T x = n1 ^ n2;
  T bits = T(0);

  while (x > 0) {
    bits += x & 1;
    x >>= 1;
  }

  return bits;
}

struct StereoParam {
  float fx, fy;
  float lcx, lcy, rcx, rcy;
  float baseline;

  float mind = 0.0f;
  float maxd = 10000.0f;

  float max_disparity = -1.0f;
  int kernel = 11;

  StereoCost cost = StereoCost::SSD;
  bool census_transform = false;

  bool compute_right = false;
  bool left_right_consistency = true;
  float left_right_consistency_th = 1.0f;

  HoleFilling hole_filling = HoleFilling::NN;
  FilterForFilled filter_for_filled = FilterForFilled::MEDIAN;

  // float subpixel_step = 0.1f;
};

struct SgmParam {
  StereoParam base_param;

  float p1 = 3.0f;
  float p2 = 20.0f;
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
  float theta_deg_max = 89.0f;
  float phi_deg_min = 0.0f;
  float phi_deg_max = 360.0f;

  bool initial_disparity_integer = false;
  bool alternately_reverse = true;

  bool view_propagation = true;
  bool temporal_propagation = false;
  bool plane_refinement = true;

  bool fill_hole_nn = true;
  bool weighted_median_for_filled = true;

  bool debug = true;
  int debug_step = 80;
};

// all_cost[j][i][d]: cost of disparity d at (i, j)
using AllDisparityCost = std::vector<std::vector<std::vector<float>>>;
void InitAllDisparityCost(AllDisparityCost* all_cost, int h, int w,
                          int max_disparity_i, float init_val);
struct Path8Cost {
  int x, y;
  AllDisparityCost cost;
};
using Path8Costs = std::array<Path8Cost, 8>;
void InitPath8Costs(Path8Costs* cost, int h, int w, int max_disparity_i,
                    float init_val);

bool ValidateDisparity(int x, float d, int half_patch_size, int width,
                       bool is_right);

bool AssertDisparity(const ugu::Image1f& disparity, int half_patch_size,
                     bool is_right);

bool Disparity2Depth(const Image1f& disparity, Image1f* depth, float baseline,
                     float fx, float lcx, float rcx, float mind, float maxd);

bool VisualizeCost(const Image1f& cost, Image3b* vis_cost, float minc = -1.0f,
                   float maxc = -1.0f);

bool CensusTransform8u(const Image1b& gray, Image1b* census);

bool ComputeStereoBruteForce(const Image1b& left, const Image1b& right,
                             Image1f* disparity, Image1f* cost, Image1f* depth,
                             const StereoParam& param);

bool ComputeStereoBruteForce(const Image3b& left, const Image3b& right,
                             Image1f* disparity, Image1f* cost, Image1f* depth,
                             const StereoParam& param);

bool ComputeStereoBruteForceCensus(const Image1b& lcensus,
                                   const Image1b& rcensus, Image1f* disparity,
                                   Image1f* cost, Image1f* depth,
                                   const StereoParam& param);

bool ComputeStereoSgm(const Image1b& left, const Image1b& right,
                      Image1f* disparity, Image1f* cost, Image1f* depth,
                      const SgmParam& param);

bool ComputeStereoSgm(const Image3b& left, const Image3b& right,
                      Image1f* disparity, Image1f* cost, Image1f* depth,
                      const SgmParam& param);

bool ComputePatchMatchStereo(const Image3b& left, const Image3b& right,
                             Image1f* ldisparity, Image1f* lcost,
                             Image1f* rdisparity, Image1f* rcost,
                             Image1f* depth,
                             const PatchMatchStereoParam& param);

struct ErrorStat {
  float mean;
  float stdev;
  float min;
  float max;
};

bool ComputeDisparityErrorStat(const Image1f& computed, const Image1f& gt,
                               Image1f* error, ErrorStat* stat,
                               bool singned_dist = false,
                               float truncation_th = -1.0f);

}  // namespace ugu