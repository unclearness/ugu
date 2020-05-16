/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include <functional>
#include <random>

#include "ugu/stereo/base.h"

namespace ugu {

using PlaneImage = Image3f;

inline float L1(const Vec3b& a, const Vec3b& b) {
  float val = 0.0f;
  for (int k = 0; k < 3; k++) {
    val += std::abs(a[k] - b[k]);
  }
  return val;
}

inline float CalcPatchMatchCost(const Vec3f& plane, const Image3b& left,
                                const Image3b& right, const Image1f& left_grad,
                                const Image1f& right_grad, int posx, int posy,
                                int minx, int maxx, int miny, int maxy,
                                float gamma, float alpha, float tau_col,
                                float tau_grad) {
  const float inverse_gamma = 1.0f / gamma;

  std::function<float(int, int, float)> rho = [&](int lx, int y,
                                                  float rx) -> float {
    int rxmin = static_cast<int>(std::floor(rx));
    int rxmax = rxmin + 1;
    float w[2] = {rx - rxmin, rxmax - rx};

    float l1_color0 = L1(left.at<Vec3b>(y, lx), right.at<Vec3b>(y, rxmin));
    float l1_color1 = L1(left.at<Vec3b>(y, lx), right.at<Vec3b>(y, rxmax));
    float l1_color = w[0] * l1_color0 + w[1] * l1_color1;
    float color_term = (1.0f - alpha) * std::min(l1_color, tau_col);

    float absdiff_grad0 =
        std::abs(left_grad.at<float>(y, lx) - right_grad.at<float>(y, rxmin));
    float absdiff_grad1 =
        std::abs(left_grad.at<float>(y, lx) - right_grad.at<float>(y, rxmax));
    float absdiff_grad = w[0] * absdiff_grad0 + w[1] * absdiff_grad1;

    float grad_term = alpha * std::min(absdiff_grad, tau_grad);

    return color_term + grad_term;
  };

  float cost_val = 0.0f;
  const Vec3b& lc = left.at<Vec3b>(posy, posx);
  const float w = static_cast<float>(left.cols - 1);
  for (int j = miny; j <= maxy; j++) {
    for (int i = minx; i <= maxx; i++) {
      const Vec3b& rc = right.at<Vec3b>(j, i);
      float l1 = L1(lc, rc);
      float w = std::exp(-l1 * inverse_gamma);

      float d = plane[0] * i + plane[1] * j + plane[2];
      float rx = static_cast<float>(i - d);

      // rx may be outside of image.
      // Because original plane is defined for (posx, posy)
      // but we are evaluating its surrounding pixels.
      // TODO: better guard
      rx = std::max(std::min(rx, w), 0.0f);

      float rho_val = rho(i, j, rx);

      cost_val += (w * rho_val);
    }
  }

  return cost_val;
}

inline bool CalcPatchMatchCost(const Image3f& plane, const Image3b& left,
                               const Image3b& right, const Image1f& left_grad,
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
      float c = CalcPatchMatchCost(p, left, right, left_grad, right_grad, i, j,
                                   minx, maxx, miny, maxy, param.gamma,
                                   param.alpha, param.tau_col, param.tau_grad);
      cost->at<float>(j, i) = c;
    }
  }

  return true;
}

inline bool InitPlaneRandom(Image3f* plane_image, Image1f* disparity,
                            std::default_random_engine& engine,
                            float initial_random_disparity_range,
                            bool fronto_parallel, int half_patch_size,
                            bool is_right) {
  initial_random_disparity_range =
      initial_random_disparity_range > 0.0f
          ? initial_random_disparity_range
          : static_cast<float>(plane_image->rows - 1);

  // z must be facing (negative), cos(theta) < 0
  const double pi = 3.14159265358979323846;
  std::uniform_real_distribution<float> theta_dist(pi / 2, 3 * pi / 2);
  std::uniform_real_distribution<float> phi_dist(0.0f, 2 * pi);

  for (int j = half_patch_size; j < plane_image->rows - half_patch_size; j++) {
    for (int i = half_patch_size; i < plane_image->cols - half_patch_size;
         i++) {
      float disparity_max = 1.0f;
      // Maximaum disparity depends on left or right
      if (is_right) {
        disparity_max = std::min(
            initial_random_disparity_range,
            static_cast<float>(plane_image->cols - i - half_patch_size + 1));

      } else {
        disparity_max = std::min(initial_random_disparity_range,
                                 static_cast<float>(i - half_patch_size + 1));
      }
      std::uniform_real_distribution<float> disparity_dist(0.000001f,
                                                           disparity_max);
      float&d = disparity->at<float>(j, i);
      d = disparity_dist(engine);
      // Set disparity as negative for right image
      if (is_right) {
        d *= -1.0f;
      }
      Eigen::Vector3f n(0.0f, 0.0f, -1.0f);
      if (!fronto_parallel) {
        float theta = theta_dist(engine);
        float phi = phi_dist(engine);
        n[0] = sin(theta) * cos(phi);
        n[1] = sin(theta) * sin(phi);
        n[2] = cos(theta);
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

#if 1
  Image1b vis_lgrad, vis_rgrad;
  Depth2Gray(left_grad, &vis_lgrad, -200, 200);
  Depth2Gray(right_grad, &vis_rgrad, -200, 200);
  ugu::imwrite("lgrad.png", vis_lgrad);
  ugu::imwrite("rgrad.png", vis_rgrad);
#endif

  // Random Initilalization
  // printf("left\n");
  InitPlaneRandom(&lplane, ldisparity, engine, param.initial_random_disparity_range,
                  param.fronto_parallel_window, param.patch_size / 2, false);
  CalcPatchMatchCost(lplane, left, right, left_grad, right_grad, lcost, param);
  // printf("right\n");
  InitPlaneRandom(&rplane, rdisparity, engine, param.initial_random_disparity_range,
                  param.fronto_parallel_window, param.patch_size / 2, true);
  CalcPatchMatchCost(rplane, right, left, right_grad, left_grad, rcost, param);

  Image3b vis_lcost, vis_rcost;
  VisualizeCost(*lcost, &vis_lcost, 0, 100);
  VisualizeCost(*rcost, &vis_rcost, 0, 100);
  ugu::imwrite("lcost.png", vis_lcost);
  ugu::imwrite("rcost.png", vis_rcost);

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

  Disparity2Depth(*ldisparity, depth, param.base_param.baseline,
                  param.base_param.fx, param.base_param.lcx,
                  param.base_param.rcx,
                  param.base_param.mind, param.base_param.maxd);

#if 0
  Disparity2Depth(*rdisparity, depth, -param.base_param.baseline,
                  param.base_param.fx, param.base_param.lcx,
                  param.base_param.rcx, param.base_param.mind,
                  param.base_param.maxd);
#endif

  return true;
}

}  // namespace ugu
