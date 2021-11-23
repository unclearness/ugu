/*
 * Copyright (C) 2020, unclearness
 * All rights reserved.
 *
 *
 * Implementation of the following paper
 *
 * Bleyer, Michael, Christoph Rhemann, and Carsten Rother. "PatchMatch
 * Stereo-Stereo Matching with Slanted Support Windows." Bmvc. Vol. 11. 2011.
 */

#include <functional>
#include <random>

#include "ugu//util/math_util.h"
#include "ugu/stereo/base.h"
#include "ugu/timer.h"
#include "ugu/util/image_util.h"

namespace {

inline void RecoverNormalFromPlane(const ugu::Vec3f& p, ugu::Vec3f* n) {
  (*n)[2] = -1.0f / std::sqrt((p[0] * p[0] + p[1] * p[1] + 1.0f));
  (*n)[0] = -p[0] * (*n)[2];
  (*n)[1] = -p[1] * (*n)[2];
}

inline bool ValidatePlane(int x, int y, const ugu::Vec3f& plane,
                          int half_patch_size, int width, bool is_right) {
  ugu::Vec3f normal;
  RecoverNormalFromPlane(plane, &normal);
  if (normal[2] > 0) {
    return false;
  }

  float len = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
                        normal[2] * normal[2]);

  if (std::abs(len - 1.0f) > 0.001f) {
    return false;
  }

  float d = x * plane[0] + y * plane[1] + plane[2];

  if (!ugu::ValidateDisparity(x, d, half_patch_size, width, is_right)) {
    return false;
  }

  return true;
}

inline bool AssertPlaneImage(const ugu::Image3f& plane, int half_patch_size,
                             bool is_right) {
#if _DEBUG

  for (int j = half_patch_size; j < plane.rows - half_patch_size; j++) {
    for (int i = half_patch_size; i < plane.cols - half_patch_size; i++) {
      const auto& p = plane.at<ugu::Vec3f>(j, i);
      if (!ValidatePlane(i, j, p, half_patch_size, plane.cols, is_right)) {
        printf("%d (%f %f %f) %d %d %d\n", i, p[0], p[1], p[2], half_patch_size,
               plane.cols, is_right);
        ugu::Vec3f normal;
        RecoverNormalFromPlane(p, &normal);
        float len = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
                              normal[2] * normal[2]);
        printf("(%f %f %f) %f\n", normal[0], normal[1], normal[2], len);
        float d = i * p[0] + j * p[1] + p[2];
        printf("%f %d\n", d,
               ugu::ValidateDisparity(i, d, half_patch_size, plane.cols,
                                      is_right));
        assert(false);
        return false;
      }
    }
  }
#else
  (void)plane;
  (void)half_patch_size;
  (void)is_right;
#endif

  return true;
}

}  // namespace

namespace ugu {

using PlaneImage = Image3f;

inline float L1(const Vec3b& a, const Vec3b& b) {
  int val = 0;
  val += std::abs(a[0] - b[0]);
  val += std::abs(a[1] - b[1]);
  val += std::abs(a[2] - b[2]);
  return static_cast<float>(val);
}

struct Correspondence {
  // first2second[y][second_x] -> first_x list
  std::vector<std::vector<std::vector<int>>> first2second;

  void Update(const Image1f& disparity2, const int half_patch_size) {
    // Init
    if (first2second.size() != static_cast<size_t>(disparity2.rows)) {
      first2second.clear();
      first2second.resize(disparity2.rows);
      std::for_each(first2second.begin(), first2second.end(),
                    [&](std::vector<std::vector<int>>& col) {
                      col.resize(disparity2.cols);
                    });
    } else {
      for (int j = 0; j < disparity2.rows; j++) {
        for (int i = 0; i < disparity2.cols; i++) {
          first2second[j][i].clear();
        }
      }
    }

    for (int j = 0; j < disparity2.rows; j++) {
      for (int i = 0; i < disparity2.cols; i++) {
        float d = disparity2.at<float>(j, i);
#if 1
        float rx = std::round(i - d);
        rx = std::max(std::min(rx, static_cast<float>(disparity2.cols - 1 -
                                                      half_patch_size)),
                      static_cast<float>(half_patch_size));
        int rx_i = static_cast<int>(rx);

        first2second[j][rx_i].push_back(i);
#else
        {
          float rx = std::floor(i - d);
          rx = std::max(std::min(rx, static_cast<float>(disparity2.cols - 1 -
                                                        half_patch_size)),
                        static_cast<float>(half_patch_size));
          int rx_i = static_cast<int>(rx);

          first2second[j][rx_i].push_back(i);
        }

        {
          float rx = std::ceil(i - d);
          rx = std::max(std::min(rx, static_cast<float>(disparity2.cols - 1 -
                                                        half_patch_size)),
                        static_cast<float>(half_patch_size));
          int rx_i = static_cast<int>(rx);

          first2second[j][rx_i].push_back(i);
        }
#endif
      }
    }
  }

  void Get(int x, int y, std::vector<int>* match_x_list) const {
    match_x_list->clear();

    std::copy(first2second[y][x].begin(), first2second[y][x].end(),
              std::back_inserter(*match_x_list));
  }
};

inline float Rho(int lx, int y, float rx, const Image3b& right,
                 const Image1f& left_grad, const Image1f& right_grad,
                 float alpha, float tau_col, float tau_grad,
                 const Vec3b& left_y_lx) {
#if 1
  int rxmin = static_cast<int>(std::floor(rx));
  int rxmax = rxmin + 1;
  float w[2] = {rx - rxmin, rxmax - rx};

  float l1_color0 = L1(left_y_lx, right.at<Vec3b>(y, rxmin));
  float l1_color1 = L1(left_y_lx, right.at<Vec3b>(y, rxmax));
  float l1_color = w[0] * l1_color0 + w[1] * l1_color1;
  float color_term = (1.0f - alpha) * std::min(l1_color, tau_col);

  const float& lg = left_grad.at<float>(y, lx);
  float absdiff_grad0 = std::abs(lg - right_grad.at<float>(y, rxmin));
  float absdiff_grad1 = std::abs(lg - right_grad.at<float>(y, rxmax));
  float absdiff_grad = w[0] * absdiff_grad0 + w[1] * absdiff_grad1;

  float grad_term = alpha * std::min(absdiff_grad, tau_grad);

#else
  float rx_i = static_cast<int>(std::round(rx));
  float l1_color = L1(left_y_lx, right.at<Vec3b>(y, rx_i));
  float color_term = (1.0f - alpha) * std::min(l1_color, tau_col);

  const float& lg = left_grad.at<float>(y, lx);
  float absdiff_grad = std::abs(lg - right_grad.at<float>(y, rx_i));
  float grad_term = alpha * std::min(absdiff_grad, tau_grad);

#endif  // 0

  return color_term + grad_term;
}

inline float CalcPatchMatchCost(
    const Vec3f& plane, const Image3b& left, const Image3b& right,
    const Image1f& left_grad, const Image1f& right_grad, int posx, int posy,
    int minx, int maxx, int miny, int maxy, float gamma, float alpha,
    float tau_col, float tau_grad, int half_patch_size,
    float early_terminate_th = std::numeric_limits<float>::max()) {
  const float inverse_gamma = 1.0f / gamma;

  float cost_val = 0.0f;
  const Vec3b& lc = left.at<Vec3b>(posy, posx);
  const float width = static_cast<float>(left.cols - 1 - half_patch_size);

  // For l1 skip
  const float w_skip = 0.01f;
  const float l1_skip_th = -std::log(w_skip) * gamma;

  //#pragma omp parallel for reduction(+:cost_val)
  for (int j = miny; j <= maxy; j++) {
    for (int i = minx; i <= maxx; i++) {
      // Question: Take color of the same image. Is this right?
      const Vec3b& rc = left.at<Vec3b>(j, i);
      float l1 = L1(lc, rc);

      // Skip if l1 is too big -> w is small
      //  -> w * rho is zero -> cost_val doesn't change
      if (l1_skip_th < l1) {
        continue;
      }

      float w = std::exp(-l1 * inverse_gamma);

      float d = plane[0] * i + plane[1] * j + plane[2];
      float rx = static_cast<float>(i - d);

      // rx may be outside of image.
      // Because original plane is defined for (posx, posy)
      // but we are evaluating its surrounding pixels.
      // TODO: better guard
      rx = std::max(std::min(rx, width), static_cast<float>(half_patch_size));

      float rho_val = Rho(i, j, rx, right, left_grad, right_grad, alpha,
                          tau_col, tau_grad, rc);

      cost_val += (w * rho_val);
    }
    if (early_terminate_th < cost_val) {
      // ugu::LOGI("early stopped %f < %f (%d < %d < %d)\n", early_terminate_th,
      // cost_val, miny, j, maxy);
      return cost_val;
    }
  }

  return cost_val;
}

inline bool CalcPatchMatchCost(const Image3f& plane, const Image3b& left,
                               const Image3b& right, const Image1f& left_grad,
                               const Image1f& right_grad, Image1f* cost,
                               const PatchMatchStereoParam& param) {
  Timer<> timer;
  timer.Start();

  const int half_ps = param.patch_size / 2;

#ifdef UGU_USE_OPENMP
#pragma omp parallel for
#endif
  for (int j = half_ps; j < plane.rows - half_ps; j++) {
    for (int i = half_ps; i < plane.cols - half_ps; i++) {
      int minx = i - half_ps;
      int maxx = i + half_ps;
      int miny = j - half_ps;
      int maxy = j + half_ps;
      const Vec3f& p = plane.at<Vec3f>(j, i);
      float c = CalcPatchMatchCost(
          p, left, right, left_grad, right_grad, i, j, minx, maxx, miny, maxy,
          param.gamma, param.alpha, param.tau_col, param.tau_grad, half_ps);
      cost->at<float>(j, i) = c;
    }
  }

  timer.End();
  ugu::LOGI("CalcPatchMatchCost: %.2f ms\n", timer.elapsed_msec());

  return true;
}

inline bool InitPlaneRandom(Image3f* plane_image, Image1f* disparity,
                            std::default_random_engine& engine,
                            float initial_random_disparity_range,
                            bool fronto_parallel, int half_patch_size,
                            bool is_right, float theta_deg_min = 0.0f,
                            float theta_deg_max = 75.0f,
                            float phi_deg_min = 0.0f,
                            float phi_deg_max = 360.0f) {
  Timer<> timer;
  timer.Start();

  initial_random_disparity_range =
      initial_random_disparity_range > 0.0f
          ? initial_random_disparity_range
          : static_cast<float>(plane_image->rows - 1);

  std::uniform_real_distribution<float> theta_dist(radians(theta_deg_min),
                                                   radians(theta_deg_max));
  std::uniform_real_distribution<float> phi_dist(radians(phi_deg_min),
                                                 radians(phi_deg_max));

  std::vector<std::uniform_real_distribution<float>> disparity_dists(
      plane_image->cols);
  const float eps = 0.01f;
  for (int i = half_patch_size; i < plane_image->cols - half_patch_size; i++) {
    float disparity_max = 1.0f;
    // Maximaum disparity depends on left or right
    if (is_right) {
      disparity_max = std::min(initial_random_disparity_range,
                               static_cast<float>(plane_image->cols - i -
                                                  half_patch_size + 1 - eps));

    } else {
      disparity_max =
          std::min(initial_random_disparity_range,
                   static_cast<float>(i - half_patch_size + 1 - eps));
    }

    std::uniform_real_distribution<float> disparity_dist(0.000001f,
                                                         disparity_max);
    disparity_dists[i] = disparity_dist;
  }

  for (int j = half_patch_size; j < plane_image->rows - half_patch_size; j++) {
    for (int i = half_patch_size; i < plane_image->cols - half_patch_size;
         i++) {
      std::uniform_real_distribution<float>& disparity_dist =
          disparity_dists[i];

      float& d = disparity->at<float>(j, i);
      d = disparity_dist(engine);
      // Set disparity as negative for right image
      if (is_right) {
        d *= -1.0f;
      }

      Eigen::Vector3f n(0.0f, 0.0f, -1.0f);
      if (!fronto_parallel) {
        // z must be facing (negative), cos(theta) < 0
        float theta = theta_dist(engine);
        float phi = phi_dist(engine);
        n[0] = sin(theta) * cos(phi);
        n[1] = sin(theta) * sin(phi);
        n[2] = -cos(theta);
        n.normalize();
      }
      Vec3f& p = plane_image->at<Vec3f>(j, i);
      float inverse_normal_z = 1.0f / n[2];
      p[0] = -n[0] * inverse_normal_z;
      p[1] = -n[1] * inverse_normal_z;
      p[2] = (n[0] * i + n[1] * j + n[2] * d) * inverse_normal_z;
    }
  }

  AssertDisparity(*disparity, half_patch_size, is_right);
  AssertPlaneImage(*plane_image, half_patch_size, is_right);

  timer.End();
  ugu::LOGI("InitPlaneRandom: %.2f ms\n", timer.elapsed_msec());

  return true;
}

inline bool SpatialPropagation(int nowx, int nowy, int fromx, int fromy,
                               const Image3b& first, const Image3b& second,
                               const Image1f& grad1, const Image1f& grad2,
                               Image1f* disparity1, Image1f* cost1,
                               PlaneImage* plane1,
                               const PatchMatchStereoParam& param,
                               bool is_right) {
  float& now_cost = cost1->at<float>(nowy, nowx);

  if (now_cost < 0.0000001f) {
    return true;
  }

  const int half_ps = param.patch_size / 2;

  int minx = nowx - half_ps;
  int maxx = nowx + half_ps;
  int miny = nowy - half_ps;
  int maxy = nowy + half_ps;

  const Vec3f& p = plane1->at<Vec3f>(fromy, fromx);

  float d = p[0] * nowx + p[1] * nowy + p[2];
  if (!ValidateDisparity(nowx, d, half_ps, first.cols, is_right)) {
    return false;
  }

  // TODO: Integral image like accelation of original PatchMatch

  float from_cost =
      CalcPatchMatchCost(p, first, second, grad1, grad2, nowx, nowy, minx, maxx,
                         miny, maxy, param.gamma, param.alpha, param.tau_col,
                         param.tau_grad, half_ps, now_cost);

  if (from_cost < now_cost) {
    if (is_right) {
      // printf("cost %f -> %f\n", now_cost, from_cost);
      // printf("disp %f -> %f\n", disparity1->at<float>(nowy, nowx), d);
    }
    now_cost = from_cost;
    plane1->at<Vec3f>(nowy, nowx) = p;
    disparity1->at<float>(nowy, nowx) = d;
  }

  return true;
}

inline bool ViewPropagation(int nowx, int nowy, const Image3b& first,
                            const Image3b& second, const Image1f& grad1,
                            Image1f* disparity1, Image1f* cost1,
                            PlaneImage* plane1, const Image1f& grad2,
                            Image1f* disparity2, PlaneImage* plane2,
                            const PatchMatchStereoParam& param,
                            const Correspondence& first2second, bool is_right) {
  const int half_ps = param.patch_size / 2;

  float& now_cost = cost1->at<float>(nowy, nowx);
  Vec3f& now_p = plane1->at<Vec3f>(nowy, nowx);

  if (now_cost < 0.0000001f) {
    return true;
  }

  std::vector<int> match_x_list;
  first2second.Get(nowx, nowy, &match_x_list);

  int minx = nowx - half_ps;
  int maxx = nowx + half_ps;
  int miny = nowy - half_ps;
  int maxy = nowy + half_ps;

  for (const int x2 : match_x_list) {
    const Vec3f& p2 = plane2->at<Vec3f>(nowy, x2);
    const float org_d = disparity2->at<float>(nowy, x2);
    // Transform to first view
    Vec3f trans_p;
    trans_p[0] = p2[0];
    trans_p[1] = p2[1];
    trans_p[2] = static_cast<float>(static_cast<double>(-org_d) -
                                    static_cast<double>(trans_p[0]) * nowx -
                                    static_cast<double>(trans_p[1]) * nowy);
    float d =
        static_cast<float>(static_cast<double>(trans_p[0]) * nowx +
                           static_cast<double>(trans_p[1]) * nowy + trans_p[2]);
    assert(std::abs(d + org_d) < 0.01f);

    if (!ValidateDisparity(nowx, d, half_ps, first.cols, is_right)) {
      continue;
    }

    float trans_cost =
        CalcPatchMatchCost(trans_p, first, second, grad1, grad2, nowx, nowy,
                           minx, maxx, miny, maxy, param.gamma, param.alpha,
                           param.tau_col, param.tau_grad, half_ps, now_cost);

    if (trans_cost < now_cost) {
      now_cost = trans_cost;
      now_p = trans_p;
      disparity1->at<float>(nowy, nowx) = d;
    }
  }

#if 0 
				  float d = now_p[0] * nowx + now_p[1] * nowy + now_p[2];
  float rx = std::round(nowx - d);

  // rx may be outside of image.
  // Because original plane is defined for (posx, posy)
  // but we are evaluating its surrounding pixels.
  // TODO: better guard
  rx = std::max(std::min(rx, static_cast<float>(second.cols - 1)), 0.0f);
  int rx_i = static_cast<int>(rx);

  float& trans_cost = cost2->at<float>(nowy, rx_i);

  if (trans_cost < now_cost) {
  }
#endif  // 0

  return true;
}

inline bool LeftRightConsistencyCheck(Image1f* ldisparity, Image1f* rdisparity,
                                      Image1b* lvalid_mask,
                                      Image1b* rvalid_mask,
                                      float left_right_consistency_th,
                                      int half_patch_size) {
  Timer<> timer;
  timer.Start();
  if (lvalid_mask->rows != ldisparity->rows ||
      lvalid_mask->cols != ldisparity->cols) {
    *lvalid_mask = Image1b::zeros(ldisparity->rows, ldisparity->cols);
  }
  lvalid_mask->setTo(255);

  if (rvalid_mask->rows != rdisparity->rows ||
      rvalid_mask->cols != rdisparity->cols) {
    *rvalid_mask = Image1b::zeros(rdisparity->rows, rdisparity->cols);
  }
  rvalid_mask->setTo(255);

  // Check left to right
  for (int j = 0; j < ldisparity->rows; j++) {
    for (int i = 0; i < ldisparity->cols; i++) {
      float& ld = ldisparity->at<float>(j, i);
      float rx = std::round(i - ld);
      rx = std::max(std::min(rx, static_cast<float>(rdisparity->cols - 1 -
                                                    half_patch_size)),
                    static_cast<float>(half_patch_size));
      int rx_i = static_cast<int>(rx);

      float& rd = rdisparity->at<float>(j, rx_i);
      // Left disparity is plus, right one is negative
      float diff = std::abs(ld + rd);
      if (left_right_consistency_th < diff) {
        ld = std::numeric_limits<float>::max();
        rd = std::numeric_limits<float>::lowest();
        lvalid_mask->at<unsigned char>(j, i) = 0;
        // rvalid_mask->at<unsigned char>(j, rx_i) = 0;
      }
    }
  }

  // Check right to left
  for (int j = 0; j < rdisparity->rows; j++) {
    for (int i = 0; i < rdisparity->cols; i++) {
      float& rd = rdisparity->at<float>(j, i);
      float lx = std::round(i - rd);
      lx = std::max(std::min(lx, static_cast<float>(ldisparity->cols - 1 -
                                                    half_patch_size)),
                    static_cast<float>(half_patch_size));
      int lx_i = static_cast<int>(lx);

      float& ld = ldisparity->at<float>(j, lx_i);
      // Left disparity is plus, right one is negative
      float diff = std::abs(ld + rd);
      if (left_right_consistency_th < diff) {
        ld = std::numeric_limits<float>::max();
        rd = std::numeric_limits<float>::lowest();
        rvalid_mask->at<unsigned char>(j, i) = 0;
        // lvalid_mask->at<unsigned char>(j, lx_i) = 0;
      }
    }
  }

  timer.End();
  ugu::LOGI("LeftRightConsistencyCheck: %.2f ms\n", timer.elapsed_msec());
  return true;
}

inline bool FillHoleNn(Image1f* disparity, Image3f* plane,
                       const Image1b& valid_mask, int half_patch_size,
                       bool is_right) {
  // TODO: Faster alogrithm
  Timer<> timer;
  timer.Start();

  Image1b valid_mask_ = Image1b::zeros(valid_mask.rows, valid_mask.cols);
  valid_mask.copyTo(valid_mask_);
  for (int j = 0; j < disparity->rows; j++) {
    for (int i = 0; i < disparity->cols; i++) {
      unsigned char& v = valid_mask_.at<unsigned char>(j, i);
      if (v == 255) {
        continue;
      }

      bool updated = false;
      float& d = disparity->at<float>(j, i);
      Vec3f& p = plane->at<Vec3f>(j, i);
      float ld = std::numeric_limits<float>::max();
      Vec3f lp;
      lp[0] = 0.0f;
      lp[1] = 0.0f;
      lp[2] = 0.0f;
      for (int l = i - 1; 0 <= l; l--) {
        unsigned char lv = valid_mask_.at<unsigned char>(j, l);
        if (lv == 255) {
          const Vec3f& tmp_p = plane->at<Vec3f>(j, l);
          float tmp_d = tmp_p[0] * i + tmp_p[1] * j + tmp_p[2];
          if (!ValidateDisparity(l, tmp_d, half_patch_size, disparity->cols,
                                 is_right)) {
            continue;
          }
          lp = tmp_p;
          ld = tmp_d;
          v = 255;
          updated = true;
          break;
        }
      }

      float rd = std::numeric_limits<float>::max();
      Vec3f rp;
      rp[0] = 0.0f;
      rp[1] = 0.0f;
      rp[2] = 0.0f;
      for (int r = i + 1; r < disparity->rows; r++) {
        unsigned char rv = valid_mask_.at<unsigned char>(j, r);
        if (rv == 255) {
          const Vec3f& tmp_p = plane->at<Vec3f>(j, r);
          float tmp_d = tmp_p[0] * i + tmp_p[1] * j + tmp_p[2];
          if (!ValidateDisparity(r, tmp_d, half_patch_size, disparity->cols,
                                 is_right)) {
            continue;
          }
          rp = tmp_p;
          rd = tmp_d;
          v = 255;
          updated = true;
          break;
        }
      }

      // If no valid plane, give up filling hole
      if (!updated) {
        continue;
      }
      // Prefer to lower disparity
      if ((!is_right && rd < ld) || (is_right && rd > ld)) {
        p = rp;
        d = rd;
      } else {
        p = lp;
        d = ld;
      }
    }
  }

  timer.End();
  ugu::LOGI("FillHoleNn: %.2f ms\n", timer.elapsed_msec());

  return true;
}

inline bool WeightedMedianForFilled(Image1f* disparity, const Image3b& color,
                                    const Image1b& valid_mask,
                                    const PatchMatchStereoParam& param,
                                    bool is_right) {
  Timer<> timer;
  timer.Start();

  const float inverse_gamma = 1.0f / param.gamma;
  const int hk = param.patch_size / 2;
  Image1f org_disparity = Image1f::zeros(valid_mask.rows, valid_mask.cols);
  disparity->copyTo(org_disparity);

#ifdef UGU_USE_OPENMP
#pragma omp parallel for
#endif
  for (int j = 0; j < disparity->rows; j++) {
    int miny = std::max(j - hk, 0);
    int maxy = std::min(j + hk, disparity->rows - 1);

    for (int i = 0; i < disparity->cols; i++) {
      unsigned char v = valid_mask.at<unsigned char>(j, i);
      if (v == 255) {
        continue;
      }
      const Vec3b& c = color.at<Vec3b>(j, i);

      int minx = std::max(i - hk, 0);
      int maxx = std::min(i + hk, disparity->cols - 1);

      std::vector<float> data;
      std::vector<float> weights;
      for (int jj = miny; jj <= maxy; jj++) {
        for (int ii = minx; ii <= maxx; ii++) {
#if 0
          // Qestion: skip invalid pixels. Is this right?
          if (valid_mask.at<unsigned char>(j, i) == 0) {
            continue;
          }
#endif
          data.push_back(org_disparity.at<float>(jj, ii));

          const Vec3b& near_c = color.at<Vec3b>(jj, ii);
          float l1 = L1(c, near_c);
          float w = std::exp(-l1 * inverse_gamma);
          weights.push_back(w);
        }
      }

#if 0
      if (weights[0] < 1 && WeightedMedian(data, weights) > 0) {
        for (int k = 0; k < data.size(); k++) {
          ugu::LOGI("%f %f\n", data[k], weights[k]);
        }
        ugu::LOGI("%f -> %f\n", disparity->at<float>(j, i),
                  WeightedMedian(data, weights));
      }
#endif  // 0

      if (!data.empty()) {
        disparity->at<float>(j, i) = ugu::WeightedMedian(data, weights);
      } else {
        disparity->at<float>(j, i) = is_right
                                         ? std::numeric_limits<float>::lowest()
                                         : std::numeric_limits<float>::max();
      }
    }
  }
  timer.End();
  ugu::LOGI("WeightedMedianForFilled: %.2f ms\n", timer.elapsed_msec());

  return true;
}

inline bool RandomSearchPlaneRefinement(
    int nowx, int nowy, const Image3b& first, const Image3b& second,
    const Image1f& grad1, Image1f* disparity1, Image1f* cost1,
    PlaneImage* plane1, const Image1f& grad2,
    const PatchMatchStereoParam& param, std::default_random_engine& engine,
    std::uniform_real_distribution<float>& uniform_dist, float z_max,
    float theta_deg_max, float phi_deg_max, bool is_right) {
  const int half_ps = param.patch_size / 2;

  float& now_cost = cost1->at<float>(nowy, nowx);

  if (now_cost < 0.0000001f) {
    // return true;
  }

  int minx = nowx - half_ps;
  int maxx = nowx + half_ps;
  int miny = nowy - half_ps;
  int maxy = nowy + half_ps;

  float& now_d = disparity1->at<float>(nowy, nowx);
  float random_d = now_d;
  Vec3f& now_p = plane1->at<Vec3f>(nowy, nowx);
  Vec3f random_p = now_p;
  Vec3f normal_f;
  Vec3d normal;
  RecoverNormalFromPlane(random_p, &normal_f);
  normal[0] = normal_f[0];
  normal[1] = normal_f[1];
  normal[2] = normal_f[2];

  float random_d_max, random_d_min;
  if (is_right) {
    random_d_max = now_d + z_max <= 0.0f ? now_d + z_max : 0.0f;
    random_d_min = (now_d - z_max) > -(first.cols - nowx - half_ps + 1)
                       ? now_d - z_max
                       : -(first.cols - nowx - half_ps + 1);

  } else {
    random_d_max = (nowx - half_ps + 1) > (now_d + z_max)
                       ? (now_d + z_max)
                       : (nowx - half_ps + 1);
    random_d_min = now_d - z_max >= 0.0f ? now_d - z_max : 0.0f;
  }

  assert(random_d_max > random_d_min);
  float r = uniform_dist(engine);  // 0~1
  random_d = (random_d_max - random_d_min) * r + random_d_min;
  assert(int(nowx - random_d) >= 0.0f);
  assert(int(nowx - random_d) <= first.cols - 1);

  float theta = static_cast<float>(acos(-normal[2]));  // positive
  float phi = static_cast<float>(atan2(normal[1], normal[0]));

  float rtheta_min = theta - radians(theta_deg_max) >= 0.0f
                         ? theta - radians(theta_deg_max)
                         : 0.0f;
  const float theta_deg_max_internal = 89.0f;
  float rtheta_max =
      theta + radians(theta_deg_max) <= radians(theta_deg_max_internal)
          ? theta + radians(theta_deg_max)
          : radians(theta_deg_max_internal);
  r = uniform_dist(engine);
  theta = (rtheta_max - rtheta_min) * r + rtheta_min;
  assert(theta >= 0.0f && theta <= pi * 0.5);
  // theta += static_cast<float>(pi * 0.5f);

  r = uniform_dist(engine);
  r = 2.0f * r - 1.0f;
  phi = radians(phi_deg_max) * r + phi;

  normal[0] = sin(theta) * cos(phi);
  normal[1] = sin(theta) * sin(phi);
  normal[2] = -cos(theta);
  assert(normal[2] <= 0.0f && normal[2] >= -1.0f);

  double len =
      normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
  normal[0] /= len;
  normal[1] /= len;
  normal[2] /= len;

  double inverse_normal_z = 1.0 / normal[2];

  random_p[0] = static_cast<float>(-normal[0] * inverse_normal_z);
  random_p[1] = static_cast<float>(-normal[1] * inverse_normal_z);
  random_p[2] = static_cast<float>(
      (normal[0] * nowx + normal[1] * nowy + normal[2] * random_d) *
      inverse_normal_z);

  float random_cost =
      CalcPatchMatchCost(random_p, first, second, grad1, grad2, nowx, nowy,
                         minx, maxx, miny, maxy, param.gamma, param.alpha,
                         param.tau_col, param.tau_grad, half_ps, now_cost);

  if (random_cost < now_cost) {
    now_cost = random_cost;
    now_p = random_p;
    now_d = random_d;
  }

  return true;
}

inline bool DebugDump(const Image1f& disparity, const Image1f& cost,
                      const std::string& prefix, bool is_right) {
  Image1b vis_disp = Image1b::zeros(disparity.rows, disparity.cols);
  Image1f tmp_disp = Image1f::zeros(disparity.rows, disparity.cols);
  disparity.copyTo(tmp_disp);
  if (is_right) {
    for (int j = 0; j < tmp_disp.rows; j++) {
      for (int i = 0; i < tmp_disp.cols; i++) {
        tmp_disp.at<float>(j, i) *= -1.0f;
      }
    }
  }
  Depth2Gray(tmp_disp, &vis_disp, 0.0f, 255.0f / 3);
  ugu::imwrite("disp_" + prefix + ".png", vis_disp);

  Image3b vis_cost = Image3b::zeros(disparity.rows, disparity.cols);
  VisualizeCost(cost, &vis_cost, 0, 100);
  ugu::imwrite("cost_" + prefix + ".png", vis_cost);

  return true;
}

inline bool DebugDump(const Image1f& disparity, const Image1f& cost, int iter,
                      int stage, bool is_right) {
  std::string prefix = is_right ? "r_" : "l_";
  prefix = prefix + std::to_string(iter) + "_" + std::to_string(stage);

  DebugDump(disparity, cost, prefix, is_right);

  return true;
}

inline bool ComputePatchMatchStereoImplBodyFromUpperLeft(
    const Image3b& first, const Image3b& second, const Image1f& grad1,
    Image1f* disparity1, Image1f* cost1, PlaneImage* plane1,
    const Image1f& grad2, Image1f* disparity2, Image1f* cost2,
    PlaneImage* plane2, const PatchMatchStereoParam& param,
    const Correspondence& first2second, std::default_random_engine& engine,
    std::uniform_real_distribution<float>& uniform_dist, float z_max,
    float theta_deg_max, float phi_deg_max, bool is_right, int iter) {
  Timer<> timer;
  timer.Start();

  (void)cost2;
  const int half_ps = param.patch_size / 2;
  const int w = first.cols;
  const int h = first.rows;

  for (int j = half_ps; j < h - half_ps; j++) {
    if (param.debug && (j - half_ps) % param.debug_step == 0) {
      int stage = (j - half_ps) / param.debug_step;
      DebugDump(*disparity1, *cost1, iter, stage, is_right);
    }

    for (int i = half_ps; i < w - half_ps; i++) {
      // Spacial propagation
      SpatialPropagation(i, j, i - 1, j, first, second, grad1, grad2,
                         disparity1, cost1, plane1, param, is_right);
      SpatialPropagation(i, j, i, j - 1, first, second, grad1, grad2,
                         disparity1, cost1, plane1, param, is_right);

      // View propagation
      if (param.view_propagation) {
        ViewPropagation(i, j, first, second, grad1, disparity1, cost1, plane1,
                        grad2, disparity2, plane2, param, first2second,
                        is_right);
      }

      // Temporal propagation
      if (param.temporal_propagation) {
      }

      // Plane refinement (random re-initialization)
      if (param.plane_refinement) {
        RandomSearchPlaneRefinement(
            i, j, first, second, grad1, disparity1, cost1, plane1, grad2, param,
            engine, uniform_dist, z_max, theta_deg_max, phi_deg_max, is_right);
      }
    }
  }

  timer.End();
  ugu::LOGI("ComputePatchMatchStereoImplBodyFromUpperLeft: %.2f ms\n",
            timer.elapsed_msec());

  return true;
}

inline bool ComputePatchMatchStereoImplBodyFromLowerRight(
    const Image3b& first, const Image3b& second, const Image1f& grad1,
    Image1f* disparity1, Image1f* cost1, PlaneImage* plane1,
    const Image1f& grad2, Image1f* disparity2, Image1f* cost2,
    PlaneImage* plane2, const PatchMatchStereoParam& param,
    const Correspondence& first2second, std::default_random_engine& engine,
    std::uniform_real_distribution<float>& uniform_dist, float z_max,
    float theta_deg_max, float phi_deg_max, bool is_right, int iter) {
  Timer<> timer;
  timer.Start();

  (void)cost2;
  const int half_ps = param.patch_size / 2;
  const int w = first.cols;
  const int h = first.rows;

  for (int j = h - half_ps - 1; half_ps < j; j--) {
    if (param.debug && ((h - half_ps - 1) - j) % param.debug_step == 0) {
      int stage = ((h - half_ps - 1) - j) / param.debug_step;
      DebugDump(*disparity1, *cost1, iter, stage, is_right);
    }

    for (int i = w - half_ps - 1; half_ps < i; i--) {
      // Spacial propagation
      SpatialPropagation(i, j, i + 1, j, first, second, grad1, grad2,
                         disparity1, cost1, plane1, param, is_right);
      SpatialPropagation(i, j, i, j + 1, first, second, grad1, grad2,
                         disparity1, cost1, plane1, param, is_right);

      // View propagation
      if (param.view_propagation) {
        ViewPropagation(i, j, first, second, grad1, disparity1, cost1, plane1,
                        grad2, disparity2, plane2, param, first2second,
                        is_right);
      }

      // Temporal propagation
      if (param.temporal_propagation) {
      }

      // Plane refinement (random re-initialization)
      if (param.plane_refinement) {
        RandomSearchPlaneRefinement(
            i, j, first, second, grad1, disparity1, cost1, plane1, grad2, param,
            engine, uniform_dist, z_max, theta_deg_max, phi_deg_max, is_right);
      }
    }
  }

  timer.End();
  ugu::LOGI("ComputePatchMatchStereoImplBodyFromLowerRight: %.2f ms\n",
            timer.elapsed_msec());

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

  Image1f lgrad, rgrad;
  lgrad = Image1f::zeros(left.rows, left.cols);
  rgrad = Image1f::zeros(left.rows, left.cols);
  SobelX(left_gray, &lgrad);
  SobelX(right_gray, &rgrad);

  if (param.debug) {
    Image1b vis_lgrad, vis_rgrad;
    Depth2Gray(lgrad, &vis_lgrad, -200, 200);
    Depth2Gray(rgrad, &vis_rgrad, -200, 200);
    ugu::imwrite("lgrad.png", vis_lgrad);
    ugu::imwrite("rgrad.png", vis_rgrad);
  }

  // Random Initilalization
  InitPlaneRandom(&lplane, ldisparity, engine,
                  param.initial_random_disparity_range,
                  param.fronto_parallel_window, param.patch_size / 2, false,
                  param.theta_deg_min, param.theta_deg_max, param.phi_deg_min,
                  param.phi_deg_max);
  CalcPatchMatchCost(lplane, left, right, lgrad, rgrad, lcost, param);
  InitPlaneRandom(&rplane, rdisparity, engine,
                  param.initial_random_disparity_range,
                  param.fronto_parallel_window, param.patch_size / 2, true,
                  param.theta_deg_min, param.theta_deg_max, param.phi_deg_min,
                  param.phi_deg_max);
  CalcPatchMatchCost(rplane, right, left, rgrad, lgrad, rcost, param);

  Correspondence l2r, r2l;
  l2r.Update(*rdisparity, param.patch_size / 2);
  r2l.Update(*ldisparity, param.patch_size / 2);

  if (param.debug) {
    DebugDump(*ldisparity, *lcost, "l_init", false);
    DebugDump(*rdisparity, *rcost, "r_init", true);
  }

  float z_max = param.initial_random_disparity_range > 0.0f
                    ? param.initial_random_disparity_range * 0.5f
                    : static_cast<float>(left.rows - 1) * 0.5f;
  float theta_deg_max = param.theta_deg_max * 0.5f;
  // float theta_deg_min = param.theta_deg_min;
  float phi_deg_max = param.phi_deg_max * 0.5f;
  // float phi_deg_min = param.phi_deg_min;

  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (int i = 0; i < param.iter; i++) {
    ugu::LOGI("iter %d\n", i);

    if (param.alternately_reverse && i % 2 == 1) {
      // Left
      ugu::LOGI("reverse left\n");
      ComputePatchMatchStereoImplBodyFromLowerRight(
          left, right, lgrad, ldisparity, lcost, &lplane, rgrad, rdisparity,
          rcost, &rplane, param, l2r, engine, dist, z_max, theta_deg_max,
          phi_deg_max, false, i);
      r2l.Update(*ldisparity, param.patch_size / 2);

      // Right
      ugu::LOGI("reverse right\n");
      ComputePatchMatchStereoImplBodyFromLowerRight(
          right, left, rgrad, rdisparity, rcost, &rplane, lgrad, ldisparity,
          lcost, &lplane, param, r2l, engine, dist, z_max, theta_deg_max,
          phi_deg_max, true, i);
      l2r.Update(*rdisparity, param.patch_size / 2);

    } else {
      // Left
      ugu::LOGI("left\n");
      ComputePatchMatchStereoImplBodyFromUpperLeft(
          left, right, lgrad, ldisparity, lcost, &lplane, rgrad, rdisparity,
          rcost, &rplane, param, l2r, engine, dist, z_max, theta_deg_max,
          phi_deg_max, false, i);
      r2l.Update(*ldisparity, param.patch_size / 2);

      // Right
      ugu::LOGI("right\n");
      ComputePatchMatchStereoImplBodyFromUpperLeft(
          right, left, rgrad, rdisparity, rcost, &rplane, lgrad, ldisparity,
          lcost, &lplane, param, r2l, engine, dist, z_max, theta_deg_max,
          phi_deg_max, true, i);
      l2r.Update(*rdisparity, param.patch_size / 2);
    }

    z_max = z_max * 0.5f;
    if (z_max < 0.1f) {
      z_max = 0.1f;
    }

    theta_deg_max *= 0.5f;
    phi_deg_max *= 0.5f;

    AssertDisparity(*ldisparity, param.patch_size / 2, false);
    AssertDisparity(*rdisparity, param.patch_size / 2, true);

    AssertPlaneImage(lplane, param.patch_size / 2, false);
    AssertPlaneImage(rplane, param.patch_size / 2, true);
  }

  if (param.debug) {
    DebugDump(*ldisparity, *lcost, "l_iterend", false);
    DebugDump(*rdisparity, *rcost, "r_iterend", true);
  }

  // Post-processing
  if (param.base_param.left_right_consistency) {
    Image1b l_valid_mask, r_valid_mask;
    LeftRightConsistencyCheck(
        ldisparity, rdisparity, &l_valid_mask, &r_valid_mask,
        param.base_param.left_right_consistency_th, param.patch_size / 2);
    if (param.debug) {
      ugu::imwrite("l_valid_mask.png", l_valid_mask);
      ugu::imwrite("r_valid_mask.png", r_valid_mask);
      DebugDump(*ldisparity, *lcost, "l_lrconsistency", false);
      DebugDump(*rdisparity, *rcost, "r_lrconsistency", true);
    }

    // Hole filling for invalidated pixels
    if (param.fill_hole_nn) {
      FillHoleNn(ldisparity, &lplane, l_valid_mask, param.patch_size / 2,
                 false);
      FillHoleNn(rdisparity, &rplane, r_valid_mask, param.patch_size / 2, true);
      if (param.debug) {
        DebugDump(*ldisparity, *lcost, "l_fillholenn", false);
        DebugDump(*rdisparity, *rcost, "r_fillholenn", true);
      }

      // Weighted median filter for filled pixels
      if (param.weighted_median_for_filled) {
        WeightedMedianForFilled(ldisparity, left, l_valid_mask, param, false);
        WeightedMedianForFilled(rdisparity, right, r_valid_mask, param, true);
        if (param.debug) {
          DebugDump(*ldisparity, *lcost, "l_filled", false);
          DebugDump(*rdisparity, *rcost, "r_filled", true);
        }
      }
    }
  }

  Disparity2Depth(*ldisparity, depth, param.base_param.baseline,
                  param.base_param.fx, param.base_param.lcx,
                  param.base_param.rcx, param.base_param.mind,
                  param.base_param.maxd);

#if 0
  Disparity2Depth(*rdisparity, depth, -param.base_param.baseline,
                  param.base_param.fx, param.base_param.lcx,
                  param.base_param.rcx, param.base_param.mind,
                  param.base_param.maxd);
#endif

  if (param.debug) {
    DebugDump(*ldisparity, *lcost, "l_end", false);
    DebugDump(*rdisparity, *rcost, "r_end", true);
  }

  return true;
}

}  // namespace ugu
