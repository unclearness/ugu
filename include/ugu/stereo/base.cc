/*
 * Copyright (C) 2020, unclearness
 * All rights reserved.
 */

#include <random>

#include "ugu/stereo/base.h"
#include "ugu/stereo/patchmatch_stereo_impl.h"

namespace {

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

#if 0
int Hamming(const ugu::Image1i& a, const ugu::Image1i& b, int minx, int maxx,
            int miny, int maxy, int offsetx, int offsety) {
  int cost = 0;
  for (int jj = miny; jj <= maxy; jj++) {
    for (int ii = minx; ii <= maxx; ii++) {
      const int& a_val = a.at<int>(jj, ii);
      const int& b_val = b.at<int>(jj + offsety, ii + offsetx);
      cost += Hamming(a_val, b_val);
    }
  }
  return cost;
}
#endif

}  // namespace

namespace ugu {

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

#pragma omp parallel for
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

template <typename T>
bool ComputeStereoBruteForceImpl(const Image<T>& left, const Image<T>& right,
                                 Image1f* disparity, Image1f* cost,
                                 Image1f* depth, const StereoParam& param) {
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
  if (cost->rows != h || cost->cols != w) {
    *cost = Image1f::zeros(h, w);
    cost->setTo(std::numeric_limits<float>::max());
  } else {
    cost->setTo(std::numeric_limits<float>::max());
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
  } else {
    LOGE("This cost has not been implemented: %d\n", param.cost);
    return false;
  }
  timer.End();
  ugu::LOGI("preparation: %.2f ms\n", timer.elapsed_msec());

  timer.Start();
#pragma omp parallel for
  for (int j = hk; j < h - hk; j++) {
    for (int i = hk; i < w - hk; i++) {
      float& disp = disparity->at<float>(j, i);
      float& c = cost->at<float>(j, i);
      int mink = 0;
      // Find the best match in the same row
      // Integer pixel (not sub-pixel) accuracy
      int d_range = std::min(max_disparity_i, i);
      for (int k = 0; k < d_range; k++) {
        int current_cost =
            cost_func(left, right, i - hk, i + hk, j - hk, j + hk, -k, 0);
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

bool ComputeStereoBruteForceCensus(const Image1b& lcensus,
                                   const Image1b& rcensus, Image1f* disparity,
                                   Image1f* cost, Image1f* depth,
                                   const StereoParam& param) {
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
    cost->setTo(std::numeric_limits<float>::max());
  } else {
    cost->setTo(std::numeric_limits<float>::max());
  }
  if (depth->rows != h || depth->cols != w) {
    *depth = Image1f::zeros(h, w);
  } else {
    depth->setTo(0.0f);
  }

  timer.End();
  ugu::LOGI("preparation: %.2f ms\n", timer.elapsed_msec());

  timer.Start();
#pragma omp parallel for
  for (int j = hk; j < h - hk; j++) {
    for (int i = hk; i < w - hk; i++) {
      float& disp = disparity->at<float>(j, i);
      float& c = cost->at<float>(j, i);
      int mink = 0;
      // Find the best match in the same row
      // Integer pixel (not sub-pixel) accuracy
      int d_range = std::min(max_disparity_i, i);
      for (int k = 0; k < d_range; k++) {
        unsigned char current_cost =
            Hamming<unsigned char>(lcensus.at<unsigned char>(j, i),
                                   rcensus.at<unsigned char>(j, i - k));
        if (current_cost < c) {
          c = static_cast<float>(current_cost);
          disp = static_cast<float>(k);
          mink = k;
          if (c == 0) {
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

bool ComputeStereoBruteForce(const Image1b& left, const Image1b& right,
                             Image1f* disparity, Image1f* cost, Image1f* depth,
                             const StereoParam& param) {
  return ComputeStereoBruteForceImpl(left, right, disparity, cost, depth,
                                     param);
}

bool ComputeStereoBruteForce(const Image3b& left, const Image3b& right,
                             Image1f* disparity, Image1f* cost, Image1f* depth,
                             const StereoParam& param) {
  return ComputeStereoBruteForceImpl(left, right, disparity, cost, depth,
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
