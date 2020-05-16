/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include <functional>
#include <random>

#include "ugu/stereo/base.h"

namespace ugu {

using PlaneImage = Image3f;

inline float CalcPatchMatchCost(const Vec3f& plane, const Image3b& left,
                                const Image3b& right, const Image1b& left_gray,
                                const Image1b& right_gray,
                                const Image1f& left_grad,
                                const Image1f& right_grad, int minx, int maxx,
                                int miny, int maxy, float gamma, float alpha,
                                float tau_col, float tau_grad) {
  const float inverse_gamma = 1.0f / gamma;

  std::function<float(int, int, float)> rho = [&](int lx, int y,
                                                  float rx) -> float {
    int rxmin = static_cast<int>(std::floor(rx));
    int rxmax = rxmin + 1;
    float w[2] = {rx - rxmin, rxmax - rx};

    float absdiff_gray0 =
        std::abs(static_cast<float>(left_gray.at<unsigned char>(y, lx) -
                                    right_gray.at<unsigned char>(y, rxmin)));
    float absdiff_gray1 =
        std::abs(static_cast<float>(left_gray.at<unsigned char>(y, lx) -
                                    right_gray.at<unsigned char>(y, rxmax)));
    float absdiff_gray = w[0] * absdiff_gray0 + w[1] * absdiff_gray1;

    float color_term = (1.0f - alpha) * std::min(absdiff_gray, tau_col);
    float absdiff_grad0 =
        std::abs(left_grad.at<float>(y, lx) - right_grad.at<float>(y, rxmin));
    float absdiff_grad1 =
        std::abs(left_grad.at<float>(y, lx) - right_grad.at<float>(y, rxmax));
    float absdiff_grad = w[0] * absdiff_grad0 + w[1] * absdiff_grad1;

    float grad_term = alpha * std::min(absdiff_grad, tau_grad);

    return color_term + grad_term;
  };

  float cost_val = 0.0f;
  for (int j = miny; j <= maxy; j++) {
    for (int i = minx; i <= maxx; i++) {
      const Vec3b& lc = left.at<Vec3b>(j, i);
      const Vec3b& rc = right.at<Vec3b>(j, i);
      float absdiff = 0.0f;
      for (int k = 0; k < 3; k++) {
        absdiff += std::abs(lc[k] - rc[k]);
      }
      float w = std::exp(-absdiff * inverse_gamma);

      float rx = plane[0] * i + plane[1] * j + plane[2];

      float rho_val = rho(i, j, rx);

      cost_val += (w * rho_val);
    }
  }

  return cost_val;
}

inline bool CalcPatchMatchCost(const Image3f& plane, const Image3b& left,
                               const Image3b& right, const Image1b& left_gray,
                               const Image1b& right_gray,
                               const Image1f& left_grad,
                               const Image1f& right_grad, Image1f* cost,
                               const PatchMatchStereoParam& param) {
  const int half_ps = param.patch_size / 2;
  for (int j = half_ps; j < plane.rows - half_ps; j++) {
    for (int i = half_ps; i < plane.cols - half_ps; i++) {
      int minx = i - half_ps;
      int maxx = i + half_ps;
      int miny = j - half_ps;
      int maxy = j + half_ps;
      const Vec3f& p = plane.at<Vec3f>(j, i);
      float c =
          CalcPatchMatchCost(p, left, right, left_gray, right_gray, left_grad,
                             right_grad, minx, maxx, miny, maxy, param.gamma,
                             param.alpha, param.tau_col, param.tau_grad);
      cost->at<float>(j, i) = c;
    }
  }

  return true;
}

inline bool InitPlaneRandom(Image3f* plane_image,
                            std::default_random_engine& engine,
                            float initial_random_disparity_range,
                            bool fronto_parallel, int half_patch_size,
                            bool is_right) {
  initial_random_disparity_range =
      initial_random_disparity_range > 0.0f
          ? initial_random_disparity_range
          : static_cast<float>(plane_image->rows - 1);

  std::uniform_real_distribution<float> normalxy_dist(-1.0f, 1.0f);

  // z must be facing (negative)
  std::uniform_real_distribution<float> normalz_dist(-1.0f, -0.00001f);

  for (int j = half_patch_size; j < plane_image->rows - half_patch_size; j++) {
    for (int i = half_patch_size; i < plane_image->cols - half_patch_size;
         i++) {
      float disparity_max = 1.0f;
      // Maximaum disparity depends on left or right
      if (is_right) {
        disparity_max = std::min(initial_random_disparity_range,
                                 static_cast<float>(i - half_patch_size + 1));
      } else {
        disparity_max = std::min(
            initial_random_disparity_range,
            static_cast<float>(plane_image->cols - i - half_patch_size + 1));
      }
      std::uniform_real_distribution<float> disparity_dist(0.000001f,
                                                           disparity_max);
      float d = disparity_dist(engine);
      // Set disparity as negative for right image
      if (is_right) {
        d *= -1.0f;
      }
      Eigen::Vector3f n(0.0f, 0.0f, -1.0f);
      if (!fronto_parallel) {
        n[0] = normalxy_dist(engine);
        n[1] = normalxy_dist(engine);
        n[2] = normalz_dist(engine);
        n.normalize();
      }
      Vec3f& p = plane_image->at<Vec3f>(j, i);
      float inverse_normal_z = 1.0f / n[2];
      p[0] = -n[0] * inverse_normal_z;
      p[1] = -n[1] * inverse_normal_z;
      p[2] = (n[0] * i + n[1] * j + n[2] * d) * inverse_normal_z;
    }
  }

  return true;
}

inline bool ComputePatchMatchStereoImplBodyFromUpperLeft(
    const Image3b& first, const Image3b& second, Image1f* disparity1,
    Image1f* cost1, PlaneImage* plane1, Image1f* disparity2, Image1f* cost2,
    PlaneImage* plane2, const PatchMatchStereoParam& param) {
  const int half_ps = param.patch_size / 2;
  const int w = first.cols;
  const int h = first.rows;

  for (int j = half_ps; j < h - half_ps; j++) {
    for (int i = half_ps; i < w - half_ps; i++) {
      // Spacial propagation

      // View propagation
      if (param.view_propagation) {
      }

      // Temporal propagation
      if (param.temporal_propagation) {
      }

      // Plane refinement (random re-initialization)
      if (param.plane_refinement) {
      }
    }
  }

  return true;
}

inline bool ComputePatchMatchStereoImplBodyFromLowerRight(
    const Image3b& first, const Image3b& second, Image1f* disparity1,
    Image1f* cost1, PlaneImage* plane1, Image1f* disparity2, Image1f* cost2,
    PlaneImage* plane2, const PatchMatchStereoParam& param) {
  const int half_ps = param.patch_size / 2;
  const int w = first.cols;
  const int h = first.rows;

  for (int j = h - half_ps - 1; half_ps < j; j--) {
    for (int i = w - half_ps - 1; half_ps < i; i--) {
      // Spacial propagation

      // View propagation
      if (param.view_propagation) {
      }

      // Temporal propagation
      if (param.temporal_propagation) {
      }

      // Plane refinement (random re-initialization)
      if (param.plane_refinement) {
      }
    }
  }

  return true;
}

inline bool ComputePatchMatchStereoImpl(const Image3b& left,
                                        const Image3b& right,
                                        Image1f* ldisparity, Image1f* lcost,
                                        Image1f* rdisparity, Image1f* rcost,
                                        Image1f* depth,
                                        const PatchMatchStereoParam& param) {
  if (left.rows != right.rows || left.cols != right.cols) {
    LOGE("Left and right size missmatch\n");
    return false;
  }

  const int w = left.cols;
  const int h = left.rows;
  if (ldisparity->rows != h || ldisparity->cols != w) {
    *ldisparity = Image1f::zeros(h, w);
  }
  if (lcost->rows != h || lcost->cols != w) {
    *lcost = Image1f::zeros(h, w);
    lcost->setTo(std::numeric_limits<float>::max());
  }
  if (rdisparity->rows != h || rdisparity->cols != w) {
    *rdisparity = Image1f::zeros(h, w);
  }
  if (rcost->rows != h || rcost->cols != w) {
    *rcost = Image1f::zeros(h, w);
    rcost->setTo(std::numeric_limits<float>::max());
  }
  if (depth->rows != h || depth->cols != w) {
    *depth = Image1f::zeros(h, w);
  }

  std::default_random_engine engine(param.random_seed);

  PlaneImage lplane = Image3f::zeros(left.rows, left.cols);
  PlaneImage rplane = Image3f::zeros(left.rows, left.cols);

  Image1b left_gray, right_gray;
  Color2Gray(left, &left_gray);
  Color2Gray(right, &right_gray);

  Image1f left_grad, right_grad;
  left_grad = Image1f::zeros(left.rows, left.cols);
  right_grad = Image1f::zeros(left.rows, left.cols);
  SobelX(left_gray, &left_grad);
  SobelX(right_gray, &right_grad);

#if 0
  Image1b vis_lgrad, vis_rgrad;
  Depth2Gray(left_grad, &vis_lgrad, -200, 200);
  Depth2Gray(right_grad, &vis_rgrad, -200, 200);
  ugu::imwrite("lgrad.png", vis_lgrad);
  ugu::imwrite("rgrad.png", vis_rgrad);
#endif

  // Random Initilalization
  InitPlaneRandom(&lplane, engine, param.initial_random_disparity_range,
                  param.fronto_parallel_window, param.patch_size / 2, false);
  CalcPatchMatchCost(lplane, left, right, left_gray, right_gray, left_grad,
                     right_grad, lcost, param);
  InitPlaneRandom(&rplane, engine, param.initial_random_disparity_range,
                  param.fronto_parallel_window, param.patch_size / 2, true);
  CalcPatchMatchCost(rplane, right, left, right_gray, left_gray, right_grad,
                     left_grad, rcost, param);

  for (int i = 0; i < param.iter; i++) {
    if (param.alternately_reverse && i % 2 == 1) {
      // Left
      ComputePatchMatchStereoImplBodyFromLowerRight(left, right, ldisparity,
                                                    lcost, &lplane, rdisparity,
                                                    rcost, &rplane, param);

      // Right
      ComputePatchMatchStereoImplBodyFromLowerRight(right, left, rdisparity,
                                                    rcost, &rplane, ldisparity,
                                                    lcost, &lplane, param);

    } else {
      // Left
      ComputePatchMatchStereoImplBodyFromUpperLeft(left, right, ldisparity,
                                                   lcost, &lplane, rdisparity,
                                                   rcost, &rplane, param);

      // Right
      ComputePatchMatchStereoImplBodyFromUpperLeft(right, left, rdisparity,
                                                   rcost, &rplane, ldisparity,
                                                   lcost, &lplane, param);
    }
  }

  // Post-processing
  if (param.left_right_consistency) {
  }

  return true;
}

}  // namespace ugu
