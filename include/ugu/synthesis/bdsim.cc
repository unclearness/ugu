/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include "bdsim.h"

#include <map>
#include <numeric>
#include <valarray>

#include "ugu/timer.h"

#ifdef UGU_USE_OPENCV
#include "opencv2/imgproc.hpp"
#define NANOPM_USE_OPENCV 1
#endif

#if __has_include("../third_party/nanopm/nanopm.h")
//#warning "nanopm.h" is found
#define UGU_HAS_NANOPM
#define NANOPM_USE_OPENMP

#ifdef UGU_USE_STB
#define NANOPM_USE_STB
#endif

#include "../third_party/nanopm/nanopm.h"
#else
//#warning "nanopm.h" not is found
#endif

namespace {

// https://stackoverflow.com/questions/7880264/convert-lab-color-to-rgb

// using http://www.easyrgb.com/index.php?X=MATH&H=01#text1
void rgb2lab(float R, float G, float B, float& l_s, float& a_s, float& b_s) {
  double var_R = R / 255.0;
  double var_G = G / 255.0;
  double var_B = B / 255.0;

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
  double X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805;
  double Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722;
  double Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505;

  double var_X = X / 95.047;  // ref_X =  95.047   Observer= 2‹, Illuminant= D65
  double var_Y = Y / 100.000;  // ref_Y = 100.000
  double var_Z = Z / 108.883;  // ref_Z = 108.883

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

  l_s = static_cast<float>((116. * var_Y) - 16.);
  a_s = static_cast<float>(500. * (var_X - var_Y));
  b_s = static_cast<float>(200. * (var_Y - var_Z));
}

void rgb2lab(unsigned char R, unsigned char G, unsigned char B,
             unsigned char& l_s, unsigned char& a_s, unsigned char& b_s) {
  float l_, a_, b_;
  rgb2lab(R, G, B, l_, a_, b_);

  l_ = std::clamp(l_, 0.f, 100.f);
  a_ = std::clamp(a_, -127.f, 127.f);  //  std::max(std::min(100.f, l_), 0.f);
  b_ = std::clamp(b_, -127.f, 127.f);

  l_s = static_cast<unsigned char>(std::round(l_ / 100.f * 255));
  a_s = static_cast<unsigned char>(std::round((a_ + 127.f) / 254.f * 255));
  b_s = static_cast<unsigned char>(std::round((b_ + 127.f) / 254.f * 255));
}

// http://www.easyrgb.com/index.php?X=MATH&H=01#text1
void lab2rgb(float l_s, float a_s, float b_s, float& R, float& G, float& B) {
  double var_Y = (l_s + 16.) / 116.;
  double var_X = a_s / 500. + var_Y;
  double var_Z = var_Y - b_s / 200.;

  if (pow(var_Y, 3) > 0.008856)
    var_Y = pow(var_Y, 3);
  else
    var_Y = (var_Y - 16. / 116.) / 7.787;
  if (pow(var_X, 3) > 0.008856)
    var_X = pow(var_X, 3);
  else
    var_X = (var_X - 16. / 116.) / 7.787;
  if (pow(var_Z, 3) > 0.008856)
    var_Z = pow(var_Z, 3);
  else
    var_Z = (var_Z - 16. / 116.) / 7.787;

  double X = 95.047 * var_X;   // ref_X =  95.047     Observer= 2‹, Illuminant=
                               // D65
  double Y = 100.000 * var_Y;  // ref_Y = 100.000
  double Z = 108.883 * var_Z;  // ref_Z = 108.883

  var_X =
      X / 100.;  // X from 0 to  95.047      (Observer = 2‹, Illuminant = D65)
  var_Y = Y / 100.;  // Y from 0 to 100.000
  var_Z = Z / 100.;  // Z from 0 to 108.883

  double var_R = var_X * 3.2406 + var_Y * -1.5372 + var_Z * -0.4986;
  double var_G = var_X * -0.9689 + var_Y * 1.8758 + var_Z * 0.0415;
  double var_B = var_X * 0.0557 + var_Y * -0.2040 + var_Z * 1.0570;

  if (var_R > 0.0031308)
    var_R = 1.055 * pow(var_R, (1 / 2.4)) - 0.055;
  else
    var_R = 12.92 * var_R;
  if (var_G > 0.0031308)
    var_G = 1.055 * pow(var_G, (1 / 2.4)) - 0.055;
  else
    var_G = 12.92 * var_G;
  if (var_B > 0.0031308)
    var_B = 1.055 * pow(var_B, (1 / 2.4)) - 0.055;
  else
    var_B = 12.92 * var_B;

  R = static_cast<float>(var_R * 255.);
  G = static_cast<float>(var_G * 255.);
  B = static_cast<float>(var_B * 255.);
}

void lab2rgb(unsigned char l_s, unsigned char a_s, unsigned char b_s,
             unsigned char& R, unsigned char& G, unsigned char& B) {
  float R_, G_, B_;

  float l_ = l_s / 255.f * 100.f;
  float a_ = a_s / 255.f * 254.f - 127.f;
  float b_ = b_s / 255.f * 254.f - 127.f;

  lab2rgb(l_, a_, b_, R_, G_, B_);

  R = static_cast<unsigned char>(std::round(R_));
  G = static_cast<unsigned char>(std::round(G_));
  B = static_cast<unsigned char>(std::round(B_));
}

ugu::Image3b RgbToLab(const ugu::Image3b& src) {
  ugu::Image3b lab = ugu::Image3b::zeros(src.rows, src.cols);
  for (int h = 0; h < src.rows; h++) {
    for (int w = 0; w < src.cols; w++) {
      const auto& rgb = src.at<ugu::Vec3b>(h, w);
      auto& lab_val = lab.at<ugu::Vec3b>(h, w);
      rgb2lab(rgb[0], rgb[1], rgb[2], lab_val[0], lab_val[1], lab_val[2]);
    }
  }
  return lab;
}

ugu::Image3b LabToRgb(const ugu::Image3b& src) {
  ugu::Image3b rgb = ugu::Image3b::zeros(src.rows, src.cols);
  for (int h = 0; h < src.rows; h++) {
    for (int w = 0; w < src.cols; w++) {
      auto& rgb_val = rgb.at<ugu::Vec3b>(h, w);
      const auto& lab = src.at<ugu::Vec3b>(h, w);
      lab2rgb(lab[0], lab[1], lab[2], rgb_val[0], rgb_val[1], rgb_val[2]);
    }
  }
  return rgb;
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

void DetermineCurrentSize(int src_w, int src_h, int cur_scale,
                          const ugu::BdsimParams& params, int& cur_w,
                          int& cur_h) {
  auto target_size = params.target_size;

  // Width
  float w_scale = 1.0f;
  cur_w = src_w;
  if (src_w < target_size.width) {
    // Width becomes bigger
    w_scale = static_cast<float>(pow((1.0f + params.rescale_ratio), cur_scale));
    cur_w = std::min(static_cast<int>(src_w * w_scale), target_size.width);
  } else {
    w_scale = static_cast<float>(pow((1.0f - params.rescale_ratio), cur_scale));
    cur_w = std::max(static_cast<int>(src_w * w_scale), target_size.width);
  }
  // Height
  float h_scale = 1.0f;
  cur_h = src_h;
  if (src_h < target_size.height) {
    // Height becomes bigger
    h_scale = static_cast<float>(pow((1.0f + params.rescale_ratio), cur_scale));
    cur_h = std::min(static_cast<int>(src_h * h_scale), target_size.height);
  } else {
    h_scale = static_cast<float>(pow((1.0f - params.rescale_ratio), cur_scale));
    cur_h = std::max(static_cast<int>(src_h * h_scale), target_size.height);
  }
}

}  // namespace

namespace ugu {

struct DirectionalInfo {
  // Size depends on direction
  Image2f nnf;
  Image1f dist;

  // Size is always same to T
  std::map<std::pair<int, int>, std::vector<ugu::Vec3b>> pixels;

  void InitPixels(int h, int w) {
    pixels.clear();
    for (int j = 0; j < h; j++) {
      for (int i = 0; i < w; i++) {
        pixels.insert({{i, j}, std::vector<ugu::Vec3b>()});
      }
    }
  }

  void ClearNNfAndDist() {
    nnf = Image2f();
    dist = Image1f();
  };
};

struct BidirectionalInfo {
  DirectionalInfo s2t;
  DirectionalInfo t2s;
  Image1f completeness;
  Image1f coherence;
  double completeness_total = 0.0;
  double coherence_total = 0.0;
  void DebugDump(const std::string& debug_dir) {
    LOGI("completeness_total %f\n", completeness_total);
    LOGI("coherence_total %f\n", coherence_total);
    if (!debug_dir.empty()) {
      // todo save images
    }
  }
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
      current_nn[0] = static_cast<float>(minLoc.x);
      current_nn[1] = static_cast<float>(minLoc.y);
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
                          const ugu::Image2f& nnf_init, int scale, int iter,
                          const std::string& prefix) {
  nanopm::Option option;
  option.patch_size = params.patch_size;
  // option.debug_dir = "./";
  option.verbose = false;
  if (nnf_init.empty()) {
    option.init_type = nanopm::InitType::RANDOM;
  } else {
    option.init_type = nanopm::InitType::INITIAL_RANDOM;

    ugu::Image2f nnf_init_resized;
    // Resize to current size
    if (nnf_init.rows != A.rows || nnf_init.cols != A.cols) {
      ugu::resize(nnf_init, nnf_init_resized, ugu::Size(A.cols, A.rows));
    } else {
      nnf_init.copyTo(nnf_init_resized);
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

  if (params.verbose && !params.debug_dir.empty()) {
    nanopm::Image3b vis_nnf, vis_distance;
    nanopm::ColorizeNnf(nnf_, vis_nnf);
    nanopm::imwrite(params.debug_dir + "/" + prefix + "_nnf_" +
                        std::to_string(scale) + "_" + std::to_string(iter) +
                        ".jpg",
                    vis_nnf);
    float mean, stddev;
    float max_d = 17000.0f;
    float min_d = 50.0f;
    nanopm::ColorizeDistance(distance_, vis_distance, max_d, min_d, mean,
                             stddev);
    printf("distance mean %f, stddev %f\n", mean, stddev);
    nanopm::imwrite(params.debug_dir + "/" + prefix + "_distance_" +
                        std::to_string(scale) + "_" + std::to_string(iter) +
                        ".jpg",
                    vis_distance);
  }

  return true;
}
#endif

bool ComputeNnf(const Image3b& S, const Image3b& T, DirectionalInfo& s2t_info,
                DirectionalInfo& t2s_info, const BdsimParams& params, int scale,
                int iter) {
  ugu::Timer timer;
  timer.Start();
  if (params.patch_search_method == BdsimPatchSearchMethod::BRUTE_FORCE) {
    ComputeNnfBruteForce(S, T, s2t_info.nnf, s2t_info.dist, params);
    ComputeNnfBruteForce(T, S, t2s_info.nnf, t2s_info.dist, params);
  } else if (params.patch_search_method ==
             BdsimPatchSearchMethod::PATCH_MATCH) {
#ifdef UGU_HAS_NANOPM
    ComputeNnfPatchMatch(S, T, s2t_info.nnf, s2t_info.dist, params,
                         s2t_info.nnf, scale, iter, "s2t");
    ComputeNnfPatchMatch(T, S, t2s_info.nnf, t2s_info.dist, params,
                         t2s_info.nnf, scale, iter, "t2s");
#else
    LOGW(
        "BdsimPatchSearchMethod::PATCH_MATCH was specified but we cannot use "
        "PatchMatch with this configuration. Use brute force\n");
    ComputeNnfBruteForce(S, T, s2t_info.nnf, s2t_info.dist, params);
    ComputeNnfBruteForce(T, S, t2s_info.nnf, t2s_info.dist, params);
#endif
  }
  timer.End();
  ugu::LOGI("ComputeNnf: %f ms\n", timer.elapsed_msec());
  return true;
}

double CalcCoherence(const Image3b& S, const Image3b& T,
                     const BdsimParams& params, DirectionalInfo& t2s_info,
                     Image1f& coherence) {
  ugu::Timer timer;
  timer.Start();
  double coherece_total = 0.0;
  t2s_info.InitPixels(T.rows, T.cols);

  for (int j = 0; j < t2s_info.nnf.rows - params.patch_size; j++) {
    for (int i = 0; i < t2s_info.nnf.cols - params.patch_size; i++) {
      for (int jj = 0; jj < params.patch_size; jj++) {
        for (int ii = 0; ii < params.patch_size; ii++) {
          int q_y = j + jj;
          int q_x = i + ii;
          const auto& q_nn_xy = t2s_info.nnf.at<ugu::Vec2f>(
              j, i);  // nnf value is same for pixels in (i, j) patch
          int q_nn_x = static_cast<int>(std::round(q_nn_xy[0]));
          int q_nn_y = static_cast<int>(std::round(q_nn_xy[1]));
          const auto& pix = S.at<ugu::Vec3b>(q_nn_y + q_y, q_nn_x + q_x);
          t2s_info.pixels[{q_x, q_y}].push_back(pix);

          const auto& q = T.at<ugu::Vec3b>(q_y, q_x);
          ugu::Vec3f diff_color;
          diff_color[0] = static_cast<float>(pix[0]) - static_cast<float>(q[0]);
          diff_color[1] = static_cast<float>(pix[1]) - static_cast<float>(q[1]);
          diff_color[2] = static_cast<float>(pix[2]) - static_cast<float>(q[2]);

          auto error = NormL2Squared(diff_color);
          coherence.at<float>(q_y, q_x) += error;
          coherece_total += error;
        }
      }
    }
  }
  int Nt = (t2s_info.nnf.rows - params.patch_size) *
           (t2s_info.nnf.cols - params.patch_size);
  double inv_Nt = 1.0 / Nt;
#pragma omp parallel for
  for (int j = 0; j < t2s_info.nnf.rows - params.patch_size; j++) {
    for (int i = 0; i < t2s_info.nnf.cols - params.patch_size; i++) {
      coherence.at<float>(j, i) =
          static_cast<float>(inv_Nt * coherence.at<float>(j, i));
    }
  }

  coherece_total *= inv_Nt;
  timer.End();
  ugu::LOGI("CalcCoherence: %f ms\n", timer.elapsed_msec());
  return coherece_total;
}

double CalcCompleteness(const Image3b& S, const Image3b& T,
                        const BdsimParams& params, DirectionalInfo& s2t_info,
                        Image1f& completeness) {
  ugu::Timer timer;
  timer.Start();
  double completeness_total = 0.0;
  s2t_info.InitPixels(T.rows, T.cols);
  for (int j = 0; j < s2t_info.nnf.rows - params.patch_size; j++) {
    for (int i = 0; i < s2t_info.nnf.cols - params.patch_size; i++) {
      for (int jj = 0; jj < params.patch_size; jj++) {
        for (int ii = 0; ii < params.patch_size; ii++) {
          int p_y = j + jj;
          int p_x = i + ii;
          const auto& p_nn_xy = s2t_info.nnf.at<ugu::Vec2f>(
              j, i);  // nnf value is same for pixels in (i, j) patch
          int p_nn_x = static_cast<int>(std::round(p_nn_xy[0]));
          int p_nn_y = static_cast<int>(std::round(p_nn_xy[1]));
          const auto& pix = S.at<ugu::Vec3b>(p_y, p_x);
          int q_x = p_nn_x + p_x;
          int q_y = p_nn_y + p_y;
          s2t_info.pixels[{q_x, q_y}].push_back(pix);

          const auto& q = T.at<ugu::Vec3b>(q_y, q_x);
          ugu::Vec3f diff_color;
          diff_color[0] = static_cast<float>(pix[0]) - static_cast<float>(q[0]);
          diff_color[1] = static_cast<float>(pix[1]) - static_cast<float>(q[1]);
          diff_color[2] = static_cast<float>(pix[2]) - static_cast<float>(q[2]);
          auto error = NormL2Squared(diff_color);
          completeness.at<float>(q_y, q_x) += error;
          completeness_total += error;
        }
      }
    }
  }
  int Ns = (s2t_info.nnf.rows - params.patch_size) *
           (s2t_info.nnf.cols - params.patch_size);
  double inv_Ns = 1.0 / Ns;
#pragma omp parallel for
  for (int j = 0; j < T.rows - params.patch_size; j++) {
    for (int i = 0; i < T.cols - params.patch_size; i++) {
      completeness.at<float>(j, i) =
          static_cast<float>(inv_Ns * completeness.at<float>(j, i));
    }
  }

  completeness_total *= inv_Ns;
  timer.End();
  ugu::LOGI("CalcCompleteness: %f ms\n", timer.elapsed_msec());
  return completeness_total;
}

bool GenerateUpdatedTarget(Image3b& T, const BdsimParams& params,
                           DirectionalInfo& s2t_info,
                           DirectionalInfo& t2s_info) {
  ugu::Timer timer;
  timer.Start();
  int Nt = (t2s_info.nnf.rows - params.patch_size) *
           (t2s_info.nnf.cols - params.patch_size);
  double inv_Nt = 1.0 / static_cast<double>(Nt);
  int Ns = (s2t_info.nnf.rows - params.patch_size) *
           (s2t_info.nnf.cols - params.patch_size);
  double inv_Ns = 1.0 / static_cast<double>(Ns);

  T.setTo(0);

#pragma omp parallel for
  for (int j = 0; j < T.rows; j++) {
    for (int i = 0; i < T.cols; i++) {
      auto& updated_p = T.at<Vec3b>(j, i);

      const auto& completeness_data = s2t_info.pixels[{i, j}];
      auto n = completeness_data.size();

      const auto& coherence_data = t2s_info.pixels[{i, j}];
      auto m = coherence_data.size();

      // LOGI("m %d,  n %d\n", m, n);

      double denom =
          static_cast<double>(n) * inv_Ns + static_cast<double>(m) * inv_Nt;

      if (denom < std::numeric_limits<double>::epsilon()) {
        // Handle no correspondence case
        updated_p[0] = 0;
        updated_p[1] = 0;
        updated_p[2] = 0;
        continue;
      }

      double inv_denom = 1.0 / denom;

      Vec3d comp_contrib{0, 0, 0};
      std::for_each(completeness_data.begin(), completeness_data.end(),
                    [&](const ugu::Vec3b& p) {
                      comp_contrib[0] += static_cast<double>(p[0]);
                      comp_contrib[1] += static_cast<double>(p[1]);
                      comp_contrib[2] += static_cast<double>(p[2]);
                    });
      comp_contrib[0] *= inv_Ns;
      comp_contrib[1] *= inv_Ns;
      comp_contrib[2] *= inv_Ns;

      Vec3d cohere_contrib{0, 0, 0};
      std::for_each(coherence_data.begin(), coherence_data.end(),
                    [&](const ugu::Vec3b& p) {
                      cohere_contrib[0] += static_cast<double>(p[0]);
                      cohere_contrib[1] += static_cast<double>(p[1]);
                      cohere_contrib[2] += static_cast<double>(p[2]);
                    });
      cohere_contrib[0] *= inv_Nt;
      cohere_contrib[1] *= inv_Nt;
      cohere_contrib[2] *= inv_Nt;

      Vec3d updated_p_d{0, 0, 0};
      for (int c = 0; c < 3; c++) {
        updated_p[c] = static_cast<unsigned char>(std::min(
            std::max(0.0, (comp_contrib[c] + cohere_contrib[c]) * inv_denom),
            255.0));
      }
    }
  }
  timer.End();
  ugu::LOGI("GenerateUpdatedTarget: %f ms\n", timer.elapsed_msec());
  return true;
}

bool Update(const Image3b& S, Image3b& T, const BdsimParams& params,
            BidirectionalInfo& bidir_info, int scale, int iter) {
  DirectionalInfo& s2t_info = bidir_info.s2t;
  DirectionalInfo& t2s_info = bidir_info.t2s;

  // Make nnf
  ComputeNnf(S, T, s2t_info, t2s_info, params, scale, iter);

  // Calc coherence
  bidir_info.coherence = Image1f::zeros(T.rows, T.cols);
  bidir_info.coherence_total =
      CalcCoherence(S, T, params, t2s_info, bidir_info.coherence);

  // Calc completeness
  bidir_info.completeness = Image1f::zeros(T.rows, T.cols);
  bidir_info.completeness_total =
      CalcCompleteness(S, T, params, s2t_info, bidir_info.completeness);

  // Generate new image
  GenerateUpdatedTarget(T, params, s2t_info, t2s_info);

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
  BidirectionalInfo bidir_info;
  while (true) {
    // Resize image
    int cur_w, cur_h;
    DetermineCurrentSize(src.cols, src.rows, scale_level, params, cur_w, cur_h);
    ugu::resize(T_lab_prev, T_lab, Size(cur_w, cur_h));

    // Clear nnf at each scale
    bidir_info.s2t.ClearNNfAndDist();
    bidir_info.t2s.ClearNNfAndDist();

    if (params.verbose) {
      LOGI("\nscale %d (%d, %d)\n", scale_level, cur_w, cur_h);
    }

    for (int iter = 0; iter < params.iteration_in_scale; iter++) {
      Update(S_lab, T_lab, params, bidir_info, scale_level, iter);
      if (params.verbose) {
        LOGI("%d th update\n", iter);
        bidir_info.DebugDump(params.debug_dir);
#ifdef UGU_USE_OPENCV
        dst = LabToBgr(T_lab);
#else
        dst = LabToRgb(T_lab);
#endif
        ugu::imwrite("out_" + std::to_string(scale_level) + "_" +
                         std::to_string(iter) + ".png",
                     dst);
      }
    }

    if (cur_w == target_size.width && cur_h == target_size.height) {
      break;
    }

    scale_level++;

    T_lab.copyTo(T_lab_prev);
  }

#ifdef UGU_USE_OPENCV
  dst = LabToBgr(T_lab);
#else
  dst = LabToRgb(T_lab);
#endif

  return true;
}

}  // namespace ugu
