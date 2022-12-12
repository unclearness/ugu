/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/image_proc.h"

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

  var_R = std::clamp(var_R, 0., 1.);
  var_G = std::clamp(var_G, 0., 1.);
  var_B = std::clamp(var_B, 0., 1.);

  R = static_cast<float>(var_R * 255.);
  G = static_cast<float>(var_G * 255.);
  B = static_cast<float>(var_B * 255.);
}

void lab2rgb(unsigned char l_s, unsigned char a_s, unsigned char b_s,
             unsigned char& R, unsigned char& G, unsigned char& B) {
  float R_, G_, B_;

  float l_ = l_s / 255.f * 100.f;
  float a_ = static_cast<float>(a_s) / 255.f * 254.f - 127.f;
  float b_ = static_cast<float>(b_s) / 255.f * 254.f - 127.f;

  lab2rgb(l_, a_, b_, R_, G_, B_);

  R = static_cast<unsigned char>(std::round(R_));
  G = static_cast<unsigned char>(std::round(G_));
  B = static_cast<unsigned char>(std::round(B_));
}

// TODO: These functions' outputs do not exactly match with OpenCV ones
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

}  // namespace

namespace ugu {

#ifndef UGU_USE_OPENCV
void cvtColor(InputArray src, OutputArray dst, int code, int dstCn) {
  (void)dstCn;
  if (code == ColorConversionCodes::COLOR_RGB2Lab) {
    dst = RgbToLab(src);
    return;
  }

  if (code == ColorConversionCodes::COLOR_Lab2RGB) {
    dst = LabToRgb(src);
    return;
  }

  throw std::runtime_error("Not implemented");
}

Scalar sum(InputArray src) {
  if (src.channels() > 4) {
    std::runtime_error("Too many channels.");
  }
  Scalar s = 0.0;

#define UGU_SUM(type, index)                                               \
  type* data = reinterpret_cast<type*>(src.data) + index * src.channels(); \
  for (int c = 0; c < src.channels(); c++) {                               \
    s[c] += static_cast<double>(data[c]);                                  \
  }

  auto sum_func = [&](size_t index_) {
    if (GetTypeidFromCvType(src.type()) == typeid(uint8_t)) {
      UGU_SUM(uint8_t, index_);
    } else if (GetTypeidFromCvType(src.type()) == typeid(int8_t)) {
      UGU_SUM(int8_t, index_);
    } else if (GetTypeidFromCvType(src.type()) == typeid(uint16_t)) {
      UGU_SUM(uint16_t, index_);
    } else if (GetTypeidFromCvType(src.type()) == typeid(int16_t)) {
      UGU_SUM(int16_t, index_);
    } else if (GetTypeidFromCvType(src.type()) == typeid(int32_t)) {
      UGU_SUM(int32_t, index_);
    } else if (GetTypeidFromCvType(src.type()) == typeid(float)) {
      UGU_SUM(float, index_);
    } else if (GetTypeidFromCvType(src.type()) == typeid(double)) {
      UGU_SUM(double, index_);
    } else {
      throw std::runtime_error("type error");
    }
  };
#undef UGU_SUM

  // TODO: Lock?
  parallel_for(size_t(0), src.total(), sum_func, 1);

  return s;
}

void subtract(InputArray src1, InputArray src2, OutputArray dst,
              InputArray mask, int dtype) {
  (void)dtype;
  (void)mask;

  if (src1.cols != src2.cols || src1.rows != src2.rows ||
      src1.type() != src2.type()) {
    throw std::runtime_error("Not supported");
  }

  dst = src1.clone();

#define UGU_SUB(type)                                         \
  for (size_t i = 0; i < dst.total() * dst.channels(); i++) { \
    *(reinterpret_cast<type*>(dst.data) + i) -=               \
        *(reinterpret_cast<type*>(src2.data) + i);            \
  }
  if (GetTypeidFromCvType(src1.type()) == typeid(uint8_t)) {
    UGU_SUB(uint8_t);
  } else if (GetTypeidFromCvType(src1.type()) == typeid(int8_t)) {
    UGU_SUB(int8_t);
  } else if (GetTypeidFromCvType(src1.type()) == typeid(uint16_t)) {
    UGU_SUB(uint16_t);
  } else if (GetTypeidFromCvType(src1.type()) == typeid(int16_t)) {
    UGU_SUB(int16_t);
  } else if (GetTypeidFromCvType(src1.type()) == typeid(int32_t)) {
    UGU_SUB(int32_t);
  } else if (GetTypeidFromCvType(src1.type()) == typeid(float)) {
    UGU_SUB(float);
  } else if (GetTypeidFromCvType(src1.type()) == typeid(double)) {
    UGU_SUB(double);
  } else {
    throw std::runtime_error("type error");
  }
#undef UGU_SUB
}

void meanStdDev(InputArray src, OutputArray mean, OutputArray stddev,
                InputArray mask) {
  // TODO: use mask
  (void)mask;

  // Mean
  Vec3d sum_ = sum(src);
  mean = ImageBase(sum_);
  mean = mean / static_cast<double>(src.total());

  // Stddev
  Vec3d sq_sum_vec = sum(src.mul(src));
  ImageBase sq_sum_ = ImageBase(sq_sum_vec) / src.total();
  subtract(sq_sum_, mean.mul(mean), stddev);
  stddev.forEach<double>([&](double& v, const int* yx) {
    (void)yx;
    assert(v >= 0);
    v = std::sqrt(v);
    return;
  });
}
#endif

Image3b ColorTransfer(const Image3b& refer, const Image3b& target,
                      const Image1b& mask, ColorTransferSpace color_space) {
  ImageBase result;

  if (color_space != ColorTransferSpace::CIE_LAB) {
    LOGE("Not implemented\n");
    return Image3b();
  }

  // Convert color space
  ImageBase refer_lab, target_lab;
#ifdef UGU_USE_OPENCV
  cvtColor(refer, refer_lab, ColorConversionCodes::COLOR_BGR2Lab);
#else
  cvtColor(refer, refer_lab, ColorConversionCodes::COLOR_RGB2Lab);
#endif

#ifdef UGU_USE_OPENCV
  cvtColor(target, target_lab, ColorConversionCodes::COLOR_BGR2Lab);
#else
  cvtColor(target, target_lab, ColorConversionCodes::COLOR_RGB2Lab);
#endif

  // Normalize to [0.0, 1.0]
  refer_lab.clone().convertTo(refer_lab, CV_64FC3, 1.0 / 255.0);
  target_lab.clone().convertTo(target_lab, CV_64FC3, 1.0 / 255.0);

  // Calc statistics
  Vec3d r_mean, r_stddev;
  Vec3d t_mean, t_stddev;
  meanStdDev(refer_lab, r_mean, r_stddev, mask);
  meanStdDev(target_lab, t_mean, t_stddev, mask);

  /** Color Transfer START **/
  // 1. Substract original mean
  result = target_lab - t_mean.t();

  // 2. Multiply ratio of standard deviation
  Vec3d scale_vec = Vec3d(ImageBase(r_stddev.div(t_stddev)));
  Image3d scale_mat = Image3d(target.rows, target.cols);
  scale_mat.setTo(scale_vec);
  result = result.mul(scale_mat);

  // 3. Add reference mean
  result = result + r_mean.t();
  /** Color Transfer END **/

  // Clamp values
  // TODO: Are these ok?
  result.setTo(0.0, result < 0.0);
  result.setTo(1.0, result > 1.0);

  // Recover pixel value range to [0, 255]
  result.clone().convertTo(result, CV_8UC3, 255.0);

  // Recover color space
#ifdef UGU_USE_OPENCV
  cvtColor(result.clone(), result, ColorConversionCodes::COLOR_Lab2BGR);
#else
  cvtColor(result.clone(), result, ColorConversionCodes::COLOR_Lab2RGB);
#endif

  return result;
}

}  // namespace ugu
