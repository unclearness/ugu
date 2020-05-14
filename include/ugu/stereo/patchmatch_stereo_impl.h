/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include <functional>
#include <random>

#include "ugu/stereo/base.h"

namespace ugu {

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

inline bool InitPatchMatchStereo(Image3f* plane_image,
                                 std::default_random_engine& engine,
                                 float initial_random_disparity_range,
                                 bool fronto_parallel, int half_patch_size) {
  std::uniform_real_distribution<float> disparity_dist(
      0.000001f, initial_random_disparity_range);

  std::uniform_real_distribution<float> normalxy_dist(-1.0f, 1.0f);

  // z must be facing (negative)
  std::uniform_real_distribution<float> normalz_dist(-1.0f, -0.00001f);

  for (int j = half_patch_size; j < plane_image->rows - half_patch_size; j++) {
    for (int i = half_patch_size; i < plane_image->cols - half_patch_size;
         i++) {
      float d = disparity_dist(engine);
      Eigen::Vector3f n(0.0f, 0.0f, -1.0f);
      if (fronto_parallel) {
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

inline bool ComputePatchMatchStereoImpl(const Image3b& left,
                                        const Image3b& right,
                                        Image1f* disparity, Image1f* cost,
                                        Image1f* depth,
                                        const PatchMatchStereoParam& param) {
  std::default_random_engine engine(param.random_seed);

  Image3f plane_image = Image3f::zeros(left.rows, right.cols);

  return true;
}

}  // namespace ugu
