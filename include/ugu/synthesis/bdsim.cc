/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include "bdsim.h"

#if __has_include("../third_party/nanopm/nanopm.h")
//#warning "nanopm.h" is found
#define UGU_HAS_NANOPM
#include "../third_party/nanopm/nanopm.h"
#else
//#warning "nanopm.h" not is found
#endif

#ifdef UGU_USE_OPENCV
#include "opencv2/imgproc.hpp"
#endif

namespace {

// https://qiita.com/hachisukansw/items/09caabe6bec46a2a0858
std::array<int, 3> RgbToLab(unsigned char r_, unsigned char g_,
                            unsigned char b_) {
  float var_R = r_ / 255.0;
  float var_G = g_ / 255.0;
  float var_B = b_ / 255.0;

  if (var_R > 0.04045)
    var_R = pow(((var_R + 0.055) / 1.055), 2.4);
  else
    var_R = var_R / 12.92;
  if (var_G > 0.04045)
    var_G = pow(((var_G + 0.055) / 1.055), 2.4);
  else
    var_G = var_G / 12.92;
  if (var_B > 0.04045)
    var_B = pow(((var_B + 0.055) / 1.055), 2.4);
  else
    var_B = var_B / 12.92;

  var_R = var_R * 100.;
  var_G = var_G * 100.;
  var_B = var_B * 100.;

  // Observer. = 2‹, Illuminant = D65
  float X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805;
  float Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722;
  float Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505;

  float var_X = X / 95.047;  // ref_X =  95.047   Observer= 2‹, Illuminant= D65
  float var_Y = Y / 100.000;  // ref_Y = 100.000
  float var_Z = Z / 108.883;  // ref_Z = 108.883

  if (var_X > 0.008856)
    var_X = pow(var_X, (1. / 3.));
  else
    var_X = (7.787 * var_X) + (16. / 116.);
  if (var_Y > 0.008856)
    var_Y = pow(var_Y, (1. / 3.));
  else
    var_Y = (7.787 * var_Y) + (16. / 116.);
  if (var_Z > 0.008856)
    var_Z = pow(var_Z, (1. / 3.));
  else
    var_Z = (7.787 * var_Z) + (16. / 116.);

  float l_s, a_s, b_s;
  l_s = (116. * var_Y) - 16.;
  a_s = 500. * (var_X - var_Y);
  b_s = 200. * (var_Y - var_Z);

  int L, a, b;
  L = l_s;  // * 255;
  a = a_s;  // * 255;
  b = b_s;  // * 255;

#if 0
				  double x, y, z;
  {
    // https://en.wikipedia.org/wiki/SRGB#The_reverse_transformation
    double r = r_ / 255.0;
    double g = g_ / 255.0;
    double b = b_ / 255.0;

    r = r > 0.04045 ? std::pow(((r + 0.055) / 1.055), 2.4) : (r / 12.92);
    g = g > 0.04045 ? std::pow(((g + 0.055) / 1.055), 2.4) : (g / 12.92);
    b = b > 0.04045 ? std::pow(((b + 0.055) / 1.055), 2.4) : (b / 12.92);

    x = (r * 0.4124) + (g * 0.3576) + (b * 0.1805);
    y = (r * 0.2126) + (g * 0.7152) + (b * 0.0722);
    z = (r * 0.0193) + (g * 0.1192) + (b * 0.9505);
  }

  // https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
  unsigned char L, a, b;

  x *= 100;
  y *= 100;
  z *= 100;

  x /= 95.047;
  y /= 100;
  z /= 108.883;

  x = x > 0.008856 ? std::pow(x, 1.0 / 3.0) : (7.787 * x) + (4.0 / 29.0);
  y = y > 0.008856 ? std::pow(y, 1.0 / 3.0) : (7.787 * y) + (4.0 / 29.0);
  z = z > 0.008856 ? std::pow(z, 1.0 / 3.0) : (7.787 * z) + (4.0 / 29.0);

  L = (116 * y) - 16;
  a = 500 * (x - y);
  b = 200 * (y - z);

#endif  // 0

  return {L, a, b};
}

ugu::Image3b RgbToLab(const ugu::Image3b& src) {
#ifdef UGU_USE_OPENCV

  cv::Mat3b lab;
  cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);

  return lab;

#else
  ugu::Image3b lab = ugu::Image3b::zeros(src.rows, src.cols);
  for (int h = 0; h < src.rows; h++) {
    for (int w = 0; w < src.cols; w++) {
      const auto& rgb = src.at<ugu::Vec3b>(h, w);
      // lab.at<ugu::Vec3b>(h, w) = RgbToLab(rgb[0], rgb[1], rgb[2]);
    }
  }
  return lab;
#endif
}

#ifdef UGU_USE_OPENCV
ugu::Image3b BgrToLab(const ugu::Image3b& bgr) {
  cv::Mat3b lab;
  cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);
  return lab;
}

ugu::Image3b LabToBgr(const ugu::Image3b& lab) {
  cv::Mat3b bgr;
  cv::cvtColor(lab, bgr, cv::COLOR_Lab2BGR);
  return bgr;
}

#endif

// https://stackoverflow.com/questions/7880264/convert-lab-color-to-rgb
void Lab2Rgb(int L, int a, int b, unsigned char& R, unsigned char& G,
             unsigned char& B) {
  float X, Y, Z, fX, fY, fZ;
  int RR, GG, BB;

  fY = pow((L + 16.0) / 116.0, 3.0);
  if (fY < 0.008856) fY = L / 903.3;
  Y = fY;

  if (fY > 0.008856)
    fY = powf(fY, 1.0 / 3.0);
  else
    fY = 7.787 * fY + 16.0 / 116.0;

  fX = a / 500.0 + fY;
  if (fX > 0.206893)
    X = powf(fX, 3.0);
  else
    X = (fX - 16.0 / 116.0) / 7.787;

  fZ = fY - b / 200.0;
  if (fZ > 0.206893)
    Z = powf(fZ, 3.0);
  else
    Z = (fZ - 16.0 / 116.0) / 7.787;

  X *= (0.950456 * 255);
  Y *= 255;
  Z *= (1.088754 * 255);

  RR = (int)(3.240479 * X - 1.537150 * Y - 0.498535 * Z + 0.5);
  GG = (int)(-0.969256 * X + 1.875992 * Y + 0.041556 * Z + 0.5);
  BB = (int)(0.055648 * X - 0.204043 * Y + 1.057311 * Z + 0.5);

  R = (unsigned char)(RR < 0 ? 0 : RR > 255 ? 255 : RR);
  G = (unsigned char)(GG < 0 ? 0 : GG > 255 ? 255 : GG);
  B = (unsigned char)(BB < 0 ? 0 : BB > 255 ? 255 : BB);

  // printf("Lab=(%f,%f,%f) ==> RGB(%f,%f,%f)\n",L,a,b,*R,*G,*B);
}

void DetermineCurrentSize(int src_w, int src_h, int cur_scale,
                          const ugu::BdsimParams& params, int& cur_w,
                          int& cur_h) {
  auto target_size = params.target_size;

  // Width
  float w_scale = 1.0f;
  cur_w = src_w;
  if (src_w < target_size.width) {
    // Width becomes bigger
    w_scale = pow((1.0f + params.rescale_ratio), cur_scale);
    cur_w = std::min(static_cast<int>(src_w * w_scale), target_size.width);
  } else {
    w_scale = pow((1.0f - params.rescale_ratio), cur_scale);
    cur_w = std::max(static_cast<int>(src_w * w_scale), target_size.width);
  }
  // Height
  float h_scale = 1.0f;
  cur_h = src_h;
  if (src_h < target_size.height) {
    // Height becomes bigger
    h_scale = pow((1.0f + params.rescale_ratio), cur_scale);
    cur_h = std::min(static_cast<int>(src_h * h_scale), target_size.height);
  } else {
    h_scale = pow((1.0f - params.rescale_ratio), cur_scale);
    cur_h = std::max(static_cast<int>(src_h * h_scale), target_size.height);
  }
}

}  // namespace

namespace ugu {

struct NnfInfo {
  Image2f nnf;
  Image1f dist;
  std::unordered_map<std::pair<int, int>, std::vector<ugu::Vec3b>> pixels;
};

inline bool SSD(const Image3b& A, int A_x, int A_y, const Image3b& B, int B_x,
                int B_y, int patch_size_x, int patch_size_y, float& val) {
  val = 0.0f;
  const float frac = 1.0f / 3.0f;
  for (int j = 0; j < patch_size_y; j++) {
    for (int i = 0; i < patch_size_x; i++) {
      auto& A_val = A.at<Vec3b>(A_y + j, A_x + i);
      auto& B_val = B.at<Vec3b>(B_y + j, B_x + i);

      float sum_diff{0.0f};
      for (int c = 0; c < 3; c++) {
        float diff = static_cast<float>(A_val[c] - B_val[c]);
        sum_diff += (diff * diff);
      }

      // average of 3 channels
      val += (sum_diff * frac);
    }
  }
  return true;
}

inline bool CalcDistance(const Image3b& A, int A_x, int A_y, const Image3b& B,
                         int B_x, int B_y, int patch_size_x, int patch_size_y,
                         BdsimPatchDistanceType distance_type,
                         float& distance) {
  if (distance_type == BdsimPatchDistanceType::SSD) {
    return SSD(A, A_x, A_y, B, B_x, B_y, patch_size_x, patch_size_y, distance);
  }

  return false;
}

inline bool SSD(const Image3b& A, int A_x, int A_y, const Image3b& B, int B_x,
                int B_y, int patch_size_x, int patch_size_y, float& val,
                float current_min) {
  val = 0.0f;
  const float frac = 1.0f / 3.0f;
  for (int j = 0; j < patch_size_y; j++) {
    for (int i = 0; i < patch_size_x; i++) {
      const Vec3b& A_val = A.at<Vec3b>(A_y + j, A_x + i);
      const Vec3b& B_val = B.at<Vec3b>(B_y + j, B_x + i);
      float sum_diff{0.0f};
      for (int c = 0; c < 3; c++) {
        float diff = static_cast<float>(A_val[c] - B_val[c]);
        sum_diff += (diff * diff);
      }

      // average of 3 channels
      val += (sum_diff * frac);
      if (val > current_min) {
        return false;
      }
    }
  }
  return true;
}

inline bool CalcDistance(const Image3b& A, int A_x, int A_y, const Image3b& B,
                         int B_x, int B_y, int patch_size_x, int patch_size_y,
                         BdsimPatchDistanceType distance_type, float& distance,
                         float current_min) {
  if (distance_type == BdsimPatchDistanceType::SSD) {
    return SSD(A, A_x, A_y, B, B_x, B_y, patch_size_x, patch_size_y, distance,
               current_min);
  }

  return false;
}

bool ComputeNnfBruteForceOwnImpl(const ugu::Image3b& A, const ugu::Image3b& B,
                                 ugu::Image2f& nnf, ugu::Image1f& distance,
                                 const BdsimParams& params) {
  // memory allocation
  nnf = Image2f::zeros(A.rows, A.cols);
  distance = Image1f::zeros(A.rows, A.cols);
  distance.setTo(std::numeric_limits<float>::max());
#ifdef UGU_USE_OPENMP
#pragma omp parallel for
#endif
  for (int j = 0; j < nnf.rows - params.patch_size; j++) {
    if (params.verbose && j % 10 == 0) {
      LOGI("current row %d\n", j);
    }

    for (int i = 0; i < nnf.cols - params.patch_size; i++) {
      // iterate all patches in B
      float& current_dist = distance.at<float>(j, i);
      ugu::Vec2f& current_nn = nnf.at<ugu::Vec2f>(j, i);
      for (int jj = 0; jj < B.rows - params.patch_size; jj++) {
        for (int ii = 0; ii < B.cols - params.patch_size; ii++) {
          float dist;
          bool ret = CalcDistance(A, i, j, B, ii, jj, params.patch_size,
                                  params.patch_size, params.distance_type, dist,
                                  current_dist);

          if (ret && dist < current_dist) {
            current_dist = dist;
            current_nn[0] = static_cast<float>(ii - i);
            current_nn[1] = static_cast<float>(jj - j);
          }
        }
      }
    }
  }

  return true;
}

#ifdef UGU_USE_OPENCV
bool ComputeNnfBruteForceOpencvImpl(const ugu::Image3b& A,
                                    const ugu::Image3b& B, ugu::Image2f& nnf,
                                    ugu::Image1f& distance,
                                    const BdsimParams& params) {
  // memory allocation
  nnf = Image2f::zeros(A.rows, A.cols);
  distance = Image1f::zeros(A.rows, A.cols);
  distance.setTo(std::numeric_limits<float>::max());
#ifdef UGU_USE_OPENMP
#pragma omp parallel for
#endif
  for (int j = 0; j < nnf.rows - params.patch_size; j++) {
    if (params.verbose && j % 10 == 0) {
      LOGI("current row %d\n", j);
    }
    for (int i = 0; i < nnf.cols - params.patch_size; i++) {
      cv::Rect roi(i, j, params.patch_size, params.patch_size);
      cv::Mat patch = A(roi);
      cv::Mat result;
      cv::matchTemplate(B, patch, result, cv::TemplateMatchModes::TM_SQDIFF);
      double minVal, maxVal;
      cv::Point minLoc, maxLoc;
      cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

      distance.at<float>(j, i) = static_cast<float>(minVal);
      ugu::Vec2f& current_nn = nnf.at<ugu::Vec2f>(j, i);
      current_nn[0] = minLoc.x;
      current_nn[1] = minLoc.y;
    }
  }

  return true;
}
#endif

bool ComputeNnfBruteForce(const ugu::Image3b& A, const ugu::Image3b& B,
                          ugu::Image2f& nnf, ugu::Image1f& distance,
                          const BdsimParams& params) {
#ifdef UGU_USE_OPENCV
  return ComputeNnfBruteForceOpencvImpl(A, B, nnf, distance, params);
#else
  return ComputeNnfBruteForceOwnImpl(A, B, nnf, distance, params);
#endif
}

#ifdef UGU_HAS_NANOPM
bool ComputeNnfPatchMatch(const ugu::Image3b& A, const ugu::Image3b& B,
                          ugu::Image2f& nnf, ugu::Image1f& distance,
                          const BdsimParams& params,
                          const ugu::Image2f& nnf_init) {
  nanopm::Option option;
  option.patch_size = params.patch_size;
  if (nnf_init.empty()) {
    option.init_type = nanopm::InitType::RANDOM;
  } else {
    option.init_type = nanopm::InitType::INITIAL_RANDOM;

    ugu::Image2f nnf_init_resized;
    // Resize to current size
    if (nnf_init_resized.rows != A.rows || nnf_init_resized.cols != A.cols) {
      ugu::resize(nnf_init, nnf_init_resized, ugu::Size(A.rows, A.cols));
    } else {
      nnf_init_resized = nnf_init;
    }

    option.initial =
        nanopm::Image2f::zeros(nnf_init_resized.rows, nnf_init_resized.cols);
    std::memcpy(
        option.initial.data, nnf_init_resized.data,
        sizeof(float) * 2 * nnf_init_resized.cols * nnf_init_resized.rows);
  }

  // todo: do not copy
  nanopm::Image3b A_, B_;
  A_ = nanopm::Image3b::zeros(A.rows, A.cols);
  std::memcpy(A_.data, A.data, sizeof(unsigned char) * 3 * A_.rows * A_.cols);
  B_ = nanopm::Image3b::zeros(B.rows, B.cols);
  std::memcpy(B_.data, B.data, sizeof(unsigned char) * 3 * B_.rows * B_.cols);
  nanopm::Image2f nnf_;
  nanopm::Image1f distance_;
  nanopm::Compute(A_, B_, nnf_, distance_, option);

  if (nnf.rows != nnf_.rows || nnf.cols != nnf_.cols) {
    nnf = ugu::Image2f::zeros(nnf_.rows, nnf_.cols);
  }
  std::memcpy(nnf.data, nnf_.data, sizeof(float) * 2 * nnf_.rows * nnf_.cols);

  if (distance.rows != distance_.rows || distance.cols != distance_.cols) {
    distance = ugu::Image1f::zeros(distance_.rows, distance_.cols);
  }
  std::memcpy(distance.data, distance_.data,
              sizeof(float) * 1 * distance_.rows * distance_.cols);

  return true;
}
#endif

bool ComputeNnf(const Image3b& S, const Image3b& T, NnfInfo& s2t_info,
                NnfInfo& t2s_info, const BdsimParams& params) {
  if (params.patch_search_method == BdsimPatchSearchMethod::BRUTE_FORCE) {
    ComputeNnfBruteForce(S, T, s2t_info.nnf, s2t_info.dist, params);
    ComputeNnfBruteForce(T, S, t2s_info.nnf, t2s_info.dist, params);
  } else if (params.patch_search_method ==
             BdsimPatchSearchMethod::PATCH_MATCH) {
#ifdef UGU_HAS_NANOPM
    ComputeNnfPatchMatch(S, T, s2t_info.nnf, s2t_info.dist, params,
                         s2t_info.nnf);
    ComputeNnfPatchMatch(T, S, t2s_info.nnf, t2s_info.dist, params,
                         t2s_info.nnf);
#else
    LOGW(
        "BdsimPatchSearchMethod::PATCH_MATCH was specified but we cannot use "
        "PatchMatch with this configuration. Use brute force\n");
    ComputeNnfBruteForce(S, T, s2t_info.nnf, s2t_info.dist, params);
    ComputeNnfBruteForce(T, S, t2s_info.nnf, t2s_info.dist, params);
#endif
  }
  return true;
}

bool Update(const Image3b& S, const Image3b& T, const BdsimParams& params,
            NnfInfo& s2t_info, NnfInfo& t2s_info) {
  // Make nnf
  ComputeNnf(S, T, s2t_info, t2s_info, params);

  // Calc coherence

  // Calc completeness

  // Generate image

  return true;
}

bool Synthesize(const Image3b& src, Image3b& dst, const BdsimParams& params) {
  const auto target_size = params.target_size;

#if 0
				  {
    unsigned char r, g, b;
    r = 100;
    g = 100;
    b = 100;
    auto lab = RgbToLab(r, g, b);
    ugu::LOGI("lab %d %d %d\n", lab[0], lab[1], lab[2]);

    ugu::Vec3b rgb;
    Lab2Rgb(lab[0], lab[1], lab[2], rgb[0], rgb[1], rgb[2]);
    ugu::LOGI("rgb %d %d %d\n", rgb[0], rgb[1], rgb[2]);

    return false;
  }
#endif  // 0

  Image3b lab;
#ifdef UGU_USE_OPENCV
  lab = BgrToLab(src);
#else
  lab = RgbToLab(src);
#endif
  Image3b S_lab, T_lab, T_lab_prev;
  lab.copyTo(S_lab);
  lab.copyTo(T_lab);
  lab.copyTo(T_lab_prev);

  int scale_level = 1;
  NnfInfo s2t_info, t2s_info;
  while (true) {
    // Resize image
    int cur_w, cur_h;
    DetermineCurrentSize(src.cols, src.rows, scale_level, params, cur_w, cur_h);
    ugu::resize(T_lab_prev, T_lab, Size(cur_w, cur_h));

    for (int iter = 0; iter < params.iteration_in_scale; iter++) {
      Update(S_lab, T_lab, params, s2t_info, t2s_info);
    }

    if (cur_w == target_size.width && cur_h == target_size.height) {
      break;
    }

    scale_level++;

    T_lab.copyTo(T_lab_prev);
  }

  return true;
}

}  // namespace ugu
