/*
 * Copyright (C) 2020, unclearness
 * All rights reserved.
 */

#include <random>

#include "ugu/stereo/base.h"
#include "ugu/stereo/patchmatch_stereo_impl.h"
//#include "ugu/stereo/sgm_impl.h"

namespace ugu {

bool ValidateDisparity(int x, float d, int half_patch_size, int width,
                       bool is_right) {
  float disparity_max = 0.0f;

  if (is_right) {
    if (d > 0) {
      return false;
    }

    disparity_max = static_cast<float>(width - x - half_patch_size + 1);
    d *= -1.0f;

  } else {
    if (d < 0) {
      return false;
    }
    disparity_max = static_cast<float>(x - half_patch_size + 1);
  }

  if (disparity_max < d) {
    return false;
  }

  return true;
}

bool AssertDisparity(const ugu::Image1f& disparity, int half_patch_size,
                     bool is_right) {
#if _DEBUG

  for (int j = half_patch_size; j < disparity.rows - half_patch_size; j++) {
    for (int i = half_patch_size; i < disparity.cols - half_patch_size; i++) {
      if (!ValidateDisparity(i, disparity.at<float>(j, i), half_patch_size,
                             disparity.cols, is_right)) {
        printf("%d %f %d %d %d\n", i, disparity.at<float>(j, i),
               half_patch_size, disparity.cols, is_right);
        assert(false);
        return false;
      }
    }
  }
#else

  (void)disparity;
  (void)half_patch_size;
  (void)is_right;

#endif

  return true;
}

bool Disparity2Depth(const Image1f& disparity, Image1f* depth, float baseline,
                     float fx, float lcx, float rcx, float mind, float maxd) {
  const int pix_num = disparity.rows * disparity.cols;
  const float* src = reinterpret_cast<float*>(disparity.data);
  float* dst = reinterpret_cast<float*>(depth->data);

  for (int i = 0; i < pix_num; i++) {
    dst[i] = baseline * fx / (src[i] - lcx + rcx);
    if (dst[i] < mind || maxd < dst[i]) {
      dst[i] = 0.0f;
    }
  }

  return true;
}

bool VisualizeCost(const Image1f& cost, Image3b* vis_cost, float minc,
                   float maxc) {
  if (minc < 0.0 || maxc < 0 || minc > maxc) {
    double minVal, maxVal;
    ugu::minMaxLoc(cost, &minVal, &maxVal);
    minc = static_cast<float>(minVal);
    maxc = static_cast<float>(maxVal);
  }

  if (vis_cost->cols != cost.cols || vis_cost->rows != cost.rows) {
    *vis_cost = Image3b::zeros(cost.rows, cost.cols);
  }

#ifdef UGU_USE_TINYCOLORMAP
  Depth2Color(cost, vis_cost, minc, maxc);
#else
  // TODO
#endif

  return true;
}

// DUMMY
bool CensusTransform8u(const Image3b& color, Image1b* census) {
  ugu::LOGE("CensusTransform8u is invalid for Image3b\n");
  (void)color;
  (void)census;
  return false;
}

bool CensusTransform8u(const Image1b& gray, Image1b* census) {
  Timer<> timer;
  timer.Start();

  const int w = gray.cols;
  const int h = gray.rows;
  const int kernel = 3;
  const int hk = kernel / 2;

  if (census->rows != h || census->cols != w) {
    *census = Image1b::zeros(h, w);
  }

#ifdef UGU_USE_OPENMP
#pragma omp parallel for
#endif
  for (int j = hk; j < h - hk; j++) {
    for (int i = hk; i < w - hk; i++) {
      const unsigned char& center = gray.at<unsigned char>(j, i);
      unsigned char& val = census->at<unsigned char>(j, i);
      val = 0;
      unsigned char mask = 1;
      for (int jj = j - hk; jj <= j + hk; jj++) {
        for (int ii = i - hk; ii <= i + hk; ii++) {
          if (jj == 0 && ii == 0) {
            continue;
          }
          if (center < gray.at<unsigned char>(jj, ii)) {
            val += mask;
          }
          mask <<= 1;
        }
      }
    }
  }

  timer.End();
  ugu::LOGI("CensusTransform8u: %.2f ms\n", timer.elapsed_msec());

  return true;
}

bool ComputeStereoBruteForceCensusImpl(
    const Image1b& lcensus, const Image1b& rcensus, Image1f* disparity,
    AllDisparityCost* all_cost, Image1f* cost, Image1f* depth,
    const StereoParam& param, bool keep_all_cost) {
  Timer<> timer;
  timer.Start();
  if (lcensus.rows != rcensus.rows || lcensus.cols != rcensus.cols) {
    LOGE("Left and right size missmatch\n");
    return false;
  }

  const int hk = param.kernel / 2;
  float max_disparity = param.max_disparity;
  if (max_disparity < 0) {
    max_disparity = static_cast<float>(lcensus.cols - 1);
  }
  const int max_disparity_i = static_cast<int>(max_disparity);
  const int w = lcensus.cols;
  const int h = lcensus.rows;
  if (disparity->rows != h || disparity->cols != w) {
    *disparity = Image1f::zeros(h, w);
  } else {
    disparity->setTo(0.0f);
  }
  if (cost->rows != h || cost->cols != w) {
    *cost = Image1f::zeros(h, w);
  }
  cost->setTo(std::numeric_limits<float>::max());

  if (keep_all_cost && all_cost->size() != h) {
    all_cost->clear();
    all_cost->resize(h);
    for (auto& row : *all_cost) {
      row.resize(w);
      for (auto& p : row) {
        p.resize(max_disparity_i);
      }
    }
  }

  if (depth->rows != h || depth->cols != w) {
    *depth = Image1f::zeros(h, w);
  } else {
    depth->setTo(0.0f);
  }

  timer.End();
  ugu::LOGI("preparation: %.2f ms\n", timer.elapsed_msec());

  timer.Start();
#ifdef UGU_USE_OPENMP
#pragma omp parallel for
#endif
  for (int j = hk; j < h - hk; j++) {
    for (int i = hk; i < w - hk; i++) {
      float& disp = disparity->at<float>(j, i);
      float& c = cost->at<float>(j, i);
      int mink = 0;
      // Find the best match in the same row
      // Integer pixel (not sub-pixel) accuracy
      int d_range = std::min(max_disparity_i, i);
      for (int k = 0; k <= d_range; k++) {
        unsigned char current_cost =
            Hamming<unsigned char>(lcensus.at<unsigned char>(j, i),
                                   rcensus.at<unsigned char>(j, i - k));
        if (keep_all_cost) {
          all_cost->at(j)[i][k] = static_cast<float>(current_cost);
        }
        if (current_cost < c) {
          c = static_cast<float>(current_cost);
          disp = static_cast<float>(k);
          mink = k;
          if (c == 0 && !keep_all_cost) {
            // if cost is 0, prefer to smaller disparity
            break;
          }
        }
      }
    }
  }

  timer.End();
  ugu::LOGI("main loop: %.2f ms\n", timer.elapsed_msec());

  Disparity2Depth(*disparity, depth, param.baseline, param.fx, param.lcx,
                  param.rcx, param.mind, param.maxd);

  return true;
}

void InitAllDisparityCost(AllDisparityCost* all_cost, int h, int w,
                          int max_disparity_i, float init_val) {
  all_cost->clear();
  all_cost->resize(h);
  for (auto& row : *all_cost) {
    row.resize(w);
    for (auto& p : row) {
      p.resize(max_disparity_i + 1, init_val);
    }
  }
}

void InitPath8Costs(Path8Costs* costs, int h, int w, int max_disparity_i,
                    float init_val) {
  for (auto& c : *costs) {
    InitAllDisparityCost(&c.cost, h, w, max_disparity_i, init_val);
  }

  /*
   012
   3+4
   567
  */

  // From upper left
  (*costs)[0].x = -1;
  (*costs)[0].y = -1;

  (*costs)[1].x = 0;
  (*costs)[1].y = -1;

  (*costs)[2].x = 1;
  (*costs)[2].y = -1;

  (*costs)[3].x = -1;
  (*costs)[3].y = 0;

  // From lower right
  (*costs)[4].x = 1;
  (*costs)[4].y = 0;

  (*costs)[5].x = -1;
  (*costs)[5].y = 1;

  (*costs)[6].x = 0;
  (*costs)[6].y = 1;

  (*costs)[7].x = 1;
  (*costs)[7].y = 1;
}

template <typename T>
bool ComputeStereoBruteForceImpl(const Image<T>& left, const Image<T>& right,
                                 Image1f* disparity, AllDisparityCost* all_cost,
                                 Image1f* best_cost, Image1f* depth,
                                 const StereoParam& param, bool keep_all_cost) {
  Timer<> timer;
  timer.Start();
  if (left.rows != right.rows || left.cols != right.cols) {
    LOGE("Left and right size missmatch\n");
    return false;
  }

  const int hk = param.kernel / 2;
  float max_disparity = param.max_disparity;
  if (max_disparity < 0) {
    max_disparity = static_cast<float>(left.cols - 1);
  }
  const int max_disparity_i = static_cast<int>(max_disparity);
  const int w = left.cols;
  const int h = left.rows;
  if (disparity->rows != h || disparity->cols != w) {
    *disparity = Image1f::zeros(h, w);
  } else {
    disparity->setTo(0.0f);
  }
  if (best_cost->rows != h || best_cost->cols != w) {
    *best_cost = Image1f::zeros(h, w);
  }
  best_cost->setTo(std::numeric_limits<float>::max());

  if (keep_all_cost && all_cost->size() != h) {
    InitAllDisparityCost(all_cost, h, w, max_disparity_i,
                         std::numeric_limits<float>::max());
  }

  if (depth->rows != h || depth->cols != w) {
    *depth = Image1f::zeros(h, w);
  } else {
    depth->setTo(0.0f);
  }

  std::function<int(const ugu::Image<T>& a, const ugu::Image<T>& b, int minx,
                    int maxx, int miny, int maxy, int offsetx, int offsety)>
      cost_func;
  if (param.cost == StereoCost::SSD) {
    cost_func = Ssd<T, int>;
  } else if (param.cost == StereoCost::SAD) {
    cost_func = Sad<T, int>;
  } else if (param.cost == StereoCost::HAMMING && param.census_transform) {
    if (param.kernel != 3) {
      LOGE("Census transform is only valid for kernel == 3: actual %d\n",
           param.kernel);
      return false;
    }
    if (left.channels() != 1) {
      LOGE("Census transform is only valid for gray image\n");
      return false;
    }
    Image1b lcensus, rcensus;
    ugu::CensusTransform8u(left, &lcensus);
    ugu::CensusTransform8u(right, &rcensus);
    return ComputeStereoBruteForceCensusImpl(lcensus, rcensus, disparity,
                                             all_cost, best_cost, depth, param,
                                             keep_all_cost);
  } else {
    LOGE("This cost has not been implemented: %d\n", param.cost);
    return false;
  }
  timer.End();
  ugu::LOGI("preparation: %.2f ms\n", timer.elapsed_msec());

  timer.Start();
#ifdef UGU_USE_OPENMP
#pragma omp parallel for
#endif
  for (int j = hk; j < h - hk; j++) {
    for (int i = hk; i < w - hk; i++) {
      float& disp = disparity->at<float>(j, i);
      float& c = best_cost->at<float>(j, i);
      int mink = 0;
      // Find the best match in the same row
      // Integer pixel (not sub-pixel) accuracy
      int d_range = std::min(max_disparity_i, i);
      for (int k = 0; k <= d_range; k++) {
        int current_cost =
            cost_func(left, right, i - hk, i + hk, j - hk, j + hk, -k, 0);
        if (keep_all_cost) {
          all_cost->at(j)[i][k] = static_cast<float>(current_cost);
        }
        if (current_cost < c) {
          c = static_cast<float>(current_cost);
          disp = static_cast<float>(k);
          mink = k;
        }
      }

#if 0
      // Sub-pixel
      if (param.subpixel_step < 0.0f || 1.0f <= param.subpixel_step) {
        continue;
      }
      for (float k = mink - 0.5f; k <= mink + 0.5f; k += param.subpixel_step) {
        float current_cost = 0.0f;
        for (int jj = -hk; jj <= hk; jj++) {
          for (int ii = -hk; ii <= hk; ii++) {
            // SAD
            double rval = BilinearInterpolation(
                i + ii - k, static_cast<float>(j + jj), 0, right);
            float sad = std::abs(static_cast<float>(
                (left.at<unsigned char>(j + jj, i + ii) - rval)));
            current_cost += sad;
          }
        }
        if (current_cost < c) {
          c = current_cost;
          disp = static_cast<float>(k);
        }
      }
#endif
    }
  }
  timer.End();
  ugu::LOGI("main loop: %.2f ms\n", timer.elapsed_msec());

  Disparity2Depth(*disparity, depth, param.baseline, param.fx, param.lcx,
                  param.rcx, param.mind, param.maxd);

  return true;
}

float AggregateCost(int x, int y, int now_d, int path_num,
                    const AllDisparityCost& all_cost, Path8Costs* path8_costs,
                    float p1, float p2, int max_disparity) {
  int offsetx = path8_costs->at(path_num).x;
  int offsety = path8_costs->at(path_num).y;

  const float& now_cost = all_cost[y][x][now_d];

  float v0, v1, v2, v3;
  v0 = std::numeric_limits<float>::max();
  v1 = std::numeric_limits<float>::max();
  v2 = std::numeric_limits<float>::max();
  v3 = std::numeric_limits<float>::max();

  // Search disparity
  for (int d = 0; d <= max_disparity; d++) {
    const float& prev_cost = all_cost[y + offsety][x + offsetx][d];

    if (now_d == d) {
      v0 = prev_cost;
    } else if (now_d == d + 1) {
      v1 = prev_cost + p1;
    } else if (now_d == d - 1) {
      v2 = prev_cost + p1;
    } else {
      float tmp_v = prev_cost + p2;
      if (tmp_v < v3) {
        v3 = tmp_v;
      }
    }
  }

  // Select minimum cost for current pix.
  path8_costs->at(path_num).cost[y][x][now_d] =
      std::min({v0, v1, v2, v3}) + now_cost;

  return path8_costs->at(path_num).cost[y][x][now_d];
}

template <typename T>
bool ComputeStereoSgmImpl(const Image<T>& left, const Image<T>& right,
                          Image1f* disparity, Image1f* cost, Image1f* depth,
                          const SgmParam& param) {
  Timer<> timer;
  timer.Start();

  const int hk = param.base_param.kernel / 2;
  float max_disparity = param.base_param.max_disparity;
  if (max_disparity < 0) {
    max_disparity = static_cast<float>(left.cols - 1);
  }
  const int max_disparity_i = static_cast<int>(max_disparity);
  const int w = left.cols;
  const int h = left.rows;

  AllDisparityCost all_cost, sum_cost;
  bool ret = ComputeStereoBruteForceImpl(left, right, disparity, &all_cost,
                                         cost, depth, param.base_param, true);
  if (!ret) {
    return false;
  }
  timer.End();
  ugu::LOGI("Brute force initialization: %.2f ms\n", timer.elapsed_msec());

  timer.Start();
  Path8Costs path8_costs;
  // Aggregate cost
  InitAllDisparityCost(&sum_cost, h, w, max_disparity_i, 0.0f);
  InitPath8Costs(&path8_costs, h, w, max_disparity_i, 0.0f);

  // Init sum_cost
  for (int j = 0; j < h; j++) {
    for (int i = 0; i < hk; i++) {
      for (int d = 0; d <= max_disparity_i; d++) {
        sum_cost[j][i][d] = all_cost[j][i][d];
        for (int k = 0; k < 8; k++) {
          path8_costs[k].cost[j][i][d] = all_cost[j][i][d];
        }
      }
    }
  }
  for (int j = 0; j < hk; j++) {
    for (int i = 0; i < w; i++) {
      for (int d = 0; d <= max_disparity_i; d++) {
        sum_cost[j][i][d] = all_cost[j][i][d];
        for (int k = 0; k < 8; k++) {
          path8_costs[k].cost[j][i][d] = all_cost[j][i][d];
        }
      }
    }
  }

  for (int j = 0; j < h; j++) {
    for (int i = w - hk; i < w; i++) {
      for (int d = 0; d <= max_disparity_i; d++) {
        sum_cost[j][i][d] = all_cost[j][i][d];
        for (int k = 0; k < 8; k++) {
          path8_costs[k].cost[j][i][d] = all_cost[j][i][d];
        }
      }
    }
  }
  for (int j = h - hk; j < h; j++) {
    for (int i = 0; i < w; i++) {
      for (int d = 0; d <= max_disparity_i; d++) {
        sum_cost[j][i][d] = all_cost[j][i][d];
        for (int k = 0; k < 8; k++) {
          path8_costs[k].cost[j][i][d] = all_cost[j][i][d];
        }
      }
    }
  }
  timer.End();
  ugu::LOGI("SGM init: %.2f ms\n", timer.elapsed_msec());

  timer.Start();
  // Cost aggregation from upper left
  for (int j = hk + 1; j < h - hk - 1; j++) {
    for (int i = hk + 1; i < w - hk - 1; i++) {
      // int d_range = std::min(max_disparity_i, i);
      for (int d = 0; d <= max_disparity_i; d++) {
        for (int k = 0; k < 4; k++) {
          sum_cost[j][i][d] +=
              AggregateCost(i, j, d, k, all_cost, &path8_costs, param.p1,
                            param.p2, max_disparity_i);
        }
      }
    }
  }
  // Cost aggregation from lower right
  for (int j = h - hk - 1; j >= hk + 1; j--) {
    for (int i = w - hk - 1; i >= hk + 1; i--) {
      // int d_range = std::min(max_disparity_i, i);
      for (int d = 0; d <= max_disparity_i; d++) {
        for (int k = 4; k < 8; k++) {
          sum_cost[j][i][d] +=
              AggregateCost(i, j, d, k, all_cost, &path8_costs, param.p1,
                            param.p2, max_disparity_i);
        }
      }
    }
  }
  timer.End();
  ugu::LOGI("Cost aggregation: %.2f ms\n", timer.elapsed_msec());

  timer.Start();
  // Winner-Takes-All
  for (int j = hk; j < h - hk; j++) {
    for (int i = hk; i < w - hk; i++) {
      float& c = cost->at<float>(j, i);
      float& disp = disparity->at<float>(j, i);
      c = sum_cost[j][i][0];
      disp = 0.0f;
      for (int d = 1; d <= max_disparity_i; d++) {
        if (sum_cost[j][i][d] < c) {
          c = sum_cost[j][i][d];
          disp = static_cast<float>(d);
        }
      }
    }
  }

  timer.End();
  ugu::LOGI("Winner-Takes-All: %.2f ms\n", timer.elapsed_msec());

  Disparity2Depth(*disparity, depth, param.base_param.baseline,
                  param.base_param.fx, param.base_param.lcx,
                  param.base_param.rcx, param.base_param.mind,
                  param.base_param.maxd);

  return true;
}

bool ComputeStereoBruteForce(const Image1b& left, const Image1b& right,
                             Image1f* disparity, Image1f* cost, Image1f* depth,
                             const StereoParam& param) {
  AllDisparityCost all_cost_stub;
  return ComputeStereoBruteForceImpl(left, right, disparity, &all_cost_stub,
                                     cost, depth, param, false);
}

bool ComputeStereoBruteForce(const Image3b& left, const Image3b& right,
                             Image1f* disparity, Image1f* cost, Image1f* depth,
                             const StereoParam& param) {
  AllDisparityCost all_cost_stub;
  return ComputeStereoBruteForceImpl(left, right, disparity, &all_cost_stub,
                                     cost, depth, param, false);
}

bool ComputeStereoBruteForceCensus(const Image1b& lcensus,
                                   const Image1b& rcensus, Image1f* disparity,
                                   Image1f* cost, Image1f* depth,
                                   const StereoParam& param) {
  AllDisparityCost all_cost_stub;
  return ComputeStereoBruteForceCensusImpl(
      lcensus, rcensus, disparity, &all_cost_stub, cost, depth, param, false);
}

bool ComputeStereoSgm(const Image1b& left, const Image1b& right,
                      Image1f* disparity, Image1f* cost, Image1f* depth,
                      const SgmParam& param) {
  return ComputeStereoSgmImpl<Vec1b>(left, right, disparity, cost, depth,
                                     param);
}

bool ComputeStereoSgm(const Image3b& left, const Image3b& right,
                      Image1f* disparity, Image1f* cost, Image1f* depth,
                      const SgmParam& param) {
  return ComputeStereoSgmImpl<Vec3b>(left, right, disparity, cost, depth,
                                     param);
}

bool ComputePatchMatchStereo(const Image3b& left, const Image3b& right,
                             Image1f* ldisparity, Image1f* lcost,
                             Image1f* rdisparity, Image1f* rcost,
                             Image1f* depth,
                             const PatchMatchStereoParam& param) {
  return ComputePatchMatchStereoImpl(left, right, ldisparity, lcost, rdisparity,
                                     rcost, depth, param);
}

}  // namespace ugu
