/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include <random>

#include "ugu/stereo/base.h"
#include "ugu/stereo/patchmatch_stereo_impl.h"

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

bool ComputeStereoBruteForce(const Image1b& left, const Image1b& right,
                             Image1f* disparity, Image1f* cost, Image1f* depth,
                             const StereoParam& param) {
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
  }
  if (cost->rows != h || cost->cols != w) {
    *cost = Image1f::zeros(h, w);
    cost->setTo(std::numeric_limits<float>::max());
  }
  if (depth->rows != h || depth->cols != w) {
    *depth = Image1f::zeros(h, w);
  }

  for (int j = hk; j < h - hk; j++) {
    for (int i = hk; i < w - hk; i++) {
      float& disp = disparity->at<float>(j, i);
      float& c = cost->at<float>(j, i);
      int mink = 0;
      // Find the best match in the same row
      // Integer pixel (not sub-pixel) accuracy
      for (int k = 0; k < max_disparity_i - i; k++) {
        float current_cost = 0.0f;
        for (int jj = -hk; jj <= hk; jj++) {
          for (int ii = -hk; ii <= hk; ii++) {
            // SAD
            float sad = std::abs(static_cast<float>(
                (left.at<unsigned char>(j + jj, i + ii) -
                 right.at<unsigned char>(j + jj, i + ii + k))));

            current_cost += sad;
          }
        }
        if (current_cost < c) {
          c = current_cost;
          disp = static_cast<float>(k);
          mink = k;
        }
      }

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
                i + ii + k, static_cast<float>(j + jj), 0, right);
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
    }
  }

  Disparity2Depth(*disparity, depth, param.baseline, param.fx, param.lcx,
                  param.rcx, param.mind, param.maxd);

  return true;
}  // namespace ugu

bool ComputeStereoBruteForce(const Image3b& left, const Image3b& right,
                             Image1f* disparity, Image1f* cost, Image1f* depth,
                             const StereoParam& param) {
  Image1b left_gray, right_gray;
  Color2Gray(left, &left_gray);
  Color2Gray(right, &right_gray);
  return ComputeStereoBruteForce(left_gray, right_gray, disparity, cost, depth,
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
