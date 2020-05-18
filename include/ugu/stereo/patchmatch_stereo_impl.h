/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include <functional>
#include <random>

#include "ugu/stereo/base.h"
#include "ugu/util.h"

namespace ugu {

using PlaneImage = Image3f;

inline float L1(const Vec3b& a, const Vec3b& b) {
  float val = 0.0f;
  for (int k = 0; k < 3; k++) {
    val += std::abs(a[k] - b[k]);
  }
  return val;
}

inline void RecoverNormalFromPlane(const Vec3f& p, Vec3f* n) {
  (*n)[2] = -1.0f / std::sqrt((p[0] * p[0] + p[1] * p[1] + 1.0f));
  (*n)[0] = -p[0] * (*n)[2];
  (*n)[1] = -p[1] * (*n)[2];
}

struct Correspondence {
  // first->at<int>(j, i) -> i' of second
  // Image1i first2second;
  // Image1i second2first;
  // (i', j') of second -> [i1, i2, ...] list of first view correspondence

  // first->at<int>(j, i) -> list of corresponding x of sencond
  // std::map<std::pair<int, int>, std::vector<int>> second2frist;
  std::vector<std::vector<std::vector<int>>> first2second;

  void Update(const Image1f& disparity2) {
    // Init
    first2second.clear();
    first2second.resize(disparity2.rows);
    std::for_each(first2second.begin(), first2second.end(),
                  [&](std::vector<std::vector<int>>& col) {
                    col.resize(disparity2.cols);
                  });

    for (int j = 0; j < disparity2.rows; j++) {
      for (int i = 0; i < disparity2.cols; i++) {
        float d = disparity2.at<float>(j, i);

        float rx = std::round(i - d);
        rx = std::max(std::min(rx, static_cast<float>(disparity2.cols - 1)),
                      0.0f);
        int rx_i = static_cast<int>(rx);

        first2second[j][rx_i].push_back(i);
      }
    }
  }

  void Get(int x, int y, std::vector<int>* match_x_list) const {
    match_x_list->clear();

    std::copy(first2second[y][x].begin(), first2second[y][x].end(),
              std::back_inserter(*match_x_list));
  }
};

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
  const float width = static_cast<float>(left.cols - 1);
  for (int j = miny; j <= maxy; j++) {
    for (int i = minx; i <= maxx; i++) {
      // Question: Take color of the same image. Is this right?
      const Vec3b& rc = left.at<Vec3b>(j, i);
      float l1 = L1(lc, rc);
      double w = std::exp(static_cast<double>(-l1) *
                          static_cast<double>(inverse_gamma));

      float d = plane[0] * i + plane[1] * j + plane[2];
      float rx = static_cast<float>(i - d);

      // rx may be outside of image.
      // Because original plane is defined for (posx, posy)
      // but we are evaluating its surrounding pixels.
      // TODO: better guard
      rx = std::max(std::min(rx, width), 0.0f);

      float rho_val = rho(i, j, rx);

      cost_val += static_cast<float>(w * rho_val);
    }
  }

  return cost_val;
}

inline bool CalcPatchMatchCost(const Image3f& plane, const Image3b& left,
                               const Image3b& right, const Image1f& left_grad,
                               const Image1f& right_grad, Image1f* cost,
                               const PatchMatchStereoParam& param) {
  const int half_ps = param.patch_size / 2;

#pragma omp parallel
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
  const float normal_rad_th = radians(75.0f);
  std::uniform_real_distribution<float> theta_dist(
      static_cast<float>(pi - normal_rad_th), static_cast<float>(pi));
  std::uniform_real_distribution<float> phi_dist(0.0f,
                                                 static_cast<float>(2 * pi));

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
      float& d = disparity->at<float>(j, i);
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

  const Vec3f p = plane1->at<Vec3f>(fromy, fromx);

  float d = p[0] * nowx + p[1] * nowy + p[2];
  if ((!is_right && d < 0) || (is_right && 0 < d)) {
    return true;
  }

  float from_cost = CalcPatchMatchCost(
      p, first, second, grad1, grad2, nowx, nowy, minx, maxx, miny, maxy,
      param.gamma, param.alpha, param.tau_col, param.tau_grad);

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
    // Transform to first view
    Vec3f trans_p;
    trans_p[0] = -p2[0];
    trans_p[1] = p2[1];
    trans_p[2] = -disparity2->at<float>(nowy, x2) - trans_p[0] * nowx -
                 trans_p[1] * nowy;
    float d = trans_p[0] * nowx + trans_p[1] * nowy + trans_p[2];
    // printf("%f -> %f\n", disparity2->at<float>(nowy, x2), d);
    if ((!is_right && d < 0) || (is_right && 0 < d)) {
      continue;
    }

    float trans_cost = CalcPatchMatchCost(
        trans_p, first, second, grad1, grad2, nowx, nowy, minx, maxx, miny,
        maxy, param.gamma, param.alpha, param.tau_col, param.tau_grad);

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
                                      Image1b* valid_mask,
                                      float left_right_consistency_th) {
  if (valid_mask->rows != ldisparity->rows ||
      valid_mask->cols != ldisparity->cols) {
    *valid_mask = Image1b::zeros(ldisparity->rows, ldisparity->cols);
  }
  valid_mask->setTo(255);

  for (int j = 0; j < ldisparity->rows; j++) {
    for (int i = 0; i < ldisparity->cols; i++) {
      float& l = ldisparity->at<float>(j, i);
      float& r = rdisparity->at<float>(j, i);
      // Left disparity is plus, right one is negative
      float diff = std::abs(l + r);
      if (left_right_consistency_th < diff) {
        l = std::numeric_limits<float>::max();
        r = std::numeric_limits<float>::lowest();
        valid_mask->at<unsigned char>(j, i) = 0;
      }
    }
  }

  return true;
}

inline bool FillHoleNnLeft(Image1f* disparity, Image3f* plane,
                           const Image1b& valid_mask) {
  // TODO: Faster alogrithm

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
          if (tmp_d < 0) {
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
          if (tmp_d < 0) {
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
      if (rd < ld) {
        p = rp;
        d = rd;
      } else {
        p = lp;
        d = ld;
      }
    }
  }

  return true;
}

inline bool WeightedMedianForFilled(Image1f* disparity, const Image3b& color,
                                    const Image1b& valid_mask,
                                    const PatchMatchStereoParam& param) {
  const float inverse_gamma = 1.0f / param.gamma;
  const int hk = param.patch_size / 2;
  Image1f org_disparity = Image1f::zeros(valid_mask.rows, valid_mask.cols);
  disparity->copyTo(org_disparity);

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
          double w = std::exp(static_cast<double>(-l1) *
                              static_cast<double>(inverse_gamma));
          weights.push_back(static_cast<float>(w));
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
        disparity->at<float>(j, i) = WeightedMedian(data, weights);
      } else {
        disparity->at<float>(j, i) = std::numeric_limits<float>::max();
      }
    }
  }

  return true;
}

inline bool RandomSearchPlaneRefinement(
    int nowx, int nowy, const Image3b& first, const Image3b& second,
    const Image1f& grad1, Image1f* disparity1, Image1f* cost1,
    PlaneImage* plane1, const Image1f& grad2,
    const PatchMatchStereoParam& param, std::default_random_engine& engine,
    const std::uniform_real_distribution<float>& z_offset_dist,
    const std::uniform_real_distribution<float>& normal_offset_dist) {
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
  Vec3f normal;
  RecoverNormalFromPlane(random_p, &normal);

  random_d += z_offset_dist(engine);
  if ((now_d > 0 && random_d < 0) || (now_d < 0 && random_d > 0)) {
    random_d *= -1.0f;
  }

  for (int k = 0; k < 3; k++) {
    normal[k] += normal_offset_dist(engine);
    // Add small value to avoid zero division
    if (normal[k] < 0) {
      normal[k] -= 0.00001f;
    } else {
      normal[k] += 0.00001f;
    }
  }
  if (normal[2] > 0) {
    normal[2] *= -1.0f;
  }
  float len =
      normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
  normal[0] /= len;
  normal[1] /= len;
  normal[2] /= len;

  float inverse_normal_z = 1.0f / normal[2];

  random_p[0] = -normal[0] * inverse_normal_z;
  random_p[1] = -normal[1] * inverse_normal_z;
  random_p[2] = (normal[0] * nowx + normal[1] * nowy + normal[2] * random_d) *
                inverse_normal_z;

  float random_cost = CalcPatchMatchCost(
      random_p, first, second, grad1, grad2, nowx, nowy, minx, maxx, miny, maxy,
      param.gamma, param.alpha, param.tau_col, param.tau_grad);

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
  Depth2Gray(disparity, &vis_disp, is_right ? -100.0f : 0.0f,
             is_right ? 0.0f : 100.0f);
  ugu::imwrite("disp_" + prefix + ".png", vis_disp);

  Image3b vis_cost = Image3b::zeros(disparity.rows, disparity.cols);
  VisualizeCost(cost, &vis_cost, 0, 50);
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
    const std::uniform_real_distribution<float>& z_offset_dist,
    const std::uniform_real_distribution<float>& normal_offset_dist,
    bool is_right, int iter) {
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
        RandomSearchPlaneRefinement(i, j, first, second, grad1, disparity1,
                                    cost1, plane1, grad2, param, engine,
                                    z_offset_dist, normal_offset_dist);
      }
    }
  }

  return true;
}

inline bool ComputePatchMatchStereoImplBodyFromLowerRight(
    const Image3b& first, const Image3b& second, const Image1f& grad1,
    Image1f* disparity1, Image1f* cost1, PlaneImage* plane1,
    const Image1f& grad2, Image1f* disparity2, Image1f* cost2,
    PlaneImage* plane2, const PatchMatchStereoParam& param,
    const Correspondence& first2second, std::default_random_engine& engine,
    const std::uniform_real_distribution<float>& z_offset_dist,
    const std::uniform_real_distribution<float>& normal_offset_dist,
    bool is_right, int iter) {
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
        RandomSearchPlaneRefinement(i, j, first, second, grad1, disparity1,
                                    cost1, plane1, grad2, param, engine,
                                    z_offset_dist, normal_offset_dist);
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
                  param.fronto_parallel_window, param.patch_size / 2, false);
  CalcPatchMatchCost(lplane, left, right, lgrad, rgrad, lcost, param);
  InitPlaneRandom(&rplane, rdisparity, engine,
                  param.initial_random_disparity_range,
                  param.fronto_parallel_window, param.patch_size / 2, true);
  CalcPatchMatchCost(rplane, right, left, rgrad, lgrad, rcost, param);

  Correspondence l2r, r2l;
  l2r.Update(*rdisparity);
  r2l.Update(*ldisparity);

  if (param.debug) {
    DebugDump(*ldisparity, *lcost, "l_init", false);
    DebugDump(*rdisparity, *rcost, "r_init", true);
  }

  float z_max = param.initial_random_disparity_range > 0.0f
                    ? param.initial_random_disparity_range * 0.5f
                    : static_cast<float>(left.rows - 1) * 0.5f;
  float n_max = 1.0f;

  for (int i = 0; i < param.iter; i++) {
    ugu::LOGI("iter %d\n", i);
    std::uniform_real_distribution<float> z_offset_dist(-z_max, z_max);
    std::uniform_real_distribution<float> normal_offset_dist(-n_max, n_max);

    if (param.alternately_reverse && i % 2 == 1) {
      // Left
      ugu::LOGI("reverse left\n");
      ComputePatchMatchStereoImplBodyFromLowerRight(
          left, right, lgrad, ldisparity, lcost, &lplane, rgrad, rdisparity,
          rcost, &rplane, param, l2r, engine, z_offset_dist, normal_offset_dist,
          false, i);
      r2l.Update(*ldisparity);

      // Right
      ugu::LOGI("reverse right\n");
      ComputePatchMatchStereoImplBodyFromLowerRight(
          right, left, rgrad, rdisparity, rcost, &rplane, lgrad, ldisparity,
          lcost, &lplane, param, r2l, engine, z_offset_dist, normal_offset_dist,
          true, i);
      l2r.Update(*rdisparity);

    } else {
      // Left
      ugu::LOGI("left\n");
      ComputePatchMatchStereoImplBodyFromUpperLeft(
          left, right, lgrad, ldisparity, lcost, &lplane, rgrad, rdisparity,
          rcost, &rplane, param, l2r, engine, z_offset_dist, normal_offset_dist,
          false, i);
      r2l.Update(*ldisparity);

      // Right
      ugu::LOGI("right\n");
      ComputePatchMatchStereoImplBodyFromUpperLeft(
          right, left, rgrad, rdisparity, rcost, &rplane, lgrad, ldisparity,
          lcost, &lplane, param, r2l, engine, z_offset_dist, normal_offset_dist,
          true, i);
      l2r.Update(*rdisparity);
    }

    z_max = z_max * 0.5f;
    if (z_max < 0.1f) {
      z_max = 0.1f;
    }
    n_max = n_max * 0.5f;
  }

  if (param.debug) {
    DebugDump(*ldisparity, *lcost, "l_iterend", false);
    DebugDump(*rdisparity, *rcost, "r_iterend", true);
  }

  // Post-processing
  if (param.left_right_consistency) {
    Image1b valid_mask;
    LeftRightConsistencyCheck(ldisparity, rdisparity, &valid_mask,
                              param.left_right_consistency_th);
    if (param.debug) {
      ugu::imwrite("valid_mask.png", valid_mask);
      DebugDump(*ldisparity, *lcost, "l_lrconsistency", false);
    }

    // Hole filling for invalidated pixels
    if (param.fill_hole_nn) {
      FillHoleNnLeft(ldisparity, &lplane, valid_mask);
      if (param.debug) {
        DebugDump(*ldisparity, *lcost, "l_fillholenn", false);
      }

      // Weighted median filter for filled pixels
      if (param.weighted_median_for_filled) {
        WeightedMedianForFilled(ldisparity, left, valid_mask, param);
        if (param.debug) {
          DebugDump(*ldisparity, *lcost, "l_filled", false);
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
