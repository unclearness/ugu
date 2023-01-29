/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/image.h"
#include "ugu/point.h"

#ifdef UGU_USE_OPENCV
#include "opencv2/imgproc.hpp"
#endif

namespace ugu {

// A variant of
// 'Color Transfer between Images'  by Erik Reinhard, Michael Ashikhmin, Bruce
// Gooch and Peter Shirley
// https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf
// Original paper uses  \ell \upalpha \upbeta color space.

enum class ColorTransferSpace { CIE_LAB };
Image3b ColorTransfer(
    const Image3b& refer, const Image3b& target,
    const Image1b& mask = Image1b(),
    ColorTransferSpace color_space = ColorTransferSpace::CIE_LAB);

Image3b PoissonBlend(const Image1b& mask, const Image3b& soure,
                     const Image3b& target, int32_t topx, int32_t topy);

#ifdef UGU_USE_OPENCV

using InterpolationFlags = cv::InterpolationFlags;
using cv::ColorConversionCodes;

using cv::cvtColor;

using cv::circle;
using cv::line;
using cv::meanStdDev;
using cv::addWeighted;

template <typename T>
void resize(const ugu::Image<T>& src, ugu::Image<T>& dst, Size dsize,
            double fx = 0, double fy = 0,
            int interpolation = InterpolationFlags::INTER_LINEAR) {
  cv::resize(src, dst, dsize, fx, fy, interpolation);
}

template <typename T, typename TT>
bool ConvertTo(const Image<T>& src, Image<TT>* dst, float scale = 1.0f) {
  src.convertTo(*dst, dst->type(), scale);

  return true;
}

inline void minMaxLoc(const cv::InputArray& src, double* minVal,
                      double* maxVal = 0, Point* minLoc = 0,
                      Point* maxLoc = 0) {
  cv::minMaxLoc(src, minVal, maxVal, minLoc, maxLoc);
}

#else

void subtract(InputArray src1, InputArray src2, OutputArray dst,
              InputArray mask = noArray(), int dtype = -1);
void cvtColor(InputArray src, OutputArray dst, int code, int dstCn = 0);
void meanStdDev(InputArray src, OutputArray mean, OutputArray stddev,
                InputArray mask = noArray());
template <int m>
void meanStdDev(InputArray src, Vec_<double, m>& mean, Vec_<double, m>& stddev,
                InputArray mask = noArray()) {
  assert(src.channels() == m);
  ImageBase mean_, stddev_;
  meanStdDev(src, mean_, stddev_, mask);
  std::memcpy(mean.val, mean_.data, sizeof(double) * mean.channels);
  std::memcpy(stddev.val, stddev_.data, sizeof(double) * stddev.channels);
}
void addWeighted(InputArray src1, double alpha, InputArray src2, double beta,
            double gamma, OutputArray dst, int dtype = -1);

enum ColorConversionCodes {
  COLOR_BGR2BGRA = 0,  //!< add alpha channel to RGB or BGR image
  COLOR_RGB2RGBA = COLOR_BGR2BGRA,

  COLOR_BGRA2BGR = 1,  //!< remove alpha channel from RGB or BGR image
  COLOR_RGBA2RGB = COLOR_BGRA2BGR,

  COLOR_BGR2RGBA = 2,  //!< convert between RGB and BGR color spaces (with or
                       //!< without alpha channel)
  COLOR_RGB2BGRA = COLOR_BGR2RGBA,

  COLOR_RGBA2BGR = 3,
  COLOR_BGRA2RGB = COLOR_RGBA2BGR,

  COLOR_BGR2RGB = 4,
  COLOR_RGB2BGR = COLOR_BGR2RGB,

  COLOR_BGRA2RGBA = 5,
  COLOR_RGBA2BGRA = COLOR_BGRA2RGBA,

  COLOR_BGR2GRAY = 6,  //!< convert between RGB/BGR and grayscale, @ref
                       //!< color_convert_rgb_gray "color conversions"
  COLOR_RGB2GRAY = 7,
  COLOR_GRAY2BGR = 8,
  COLOR_GRAY2RGB = COLOR_GRAY2BGR,
  COLOR_GRAY2BGRA = 9,
  COLOR_GRAY2RGBA = COLOR_GRAY2BGRA,
  COLOR_BGRA2GRAY = 10,
  COLOR_RGBA2GRAY = 11,

  COLOR_BGR2BGR565 =
      12,  //!< convert between RGB/BGR and BGR565 (16-bit images)
  COLOR_RGB2BGR565 = 13,
  COLOR_BGR5652BGR = 14,
  COLOR_BGR5652RGB = 15,
  COLOR_BGRA2BGR565 = 16,
  COLOR_RGBA2BGR565 = 17,
  COLOR_BGR5652BGRA = 18,
  COLOR_BGR5652RGBA = 19,

  COLOR_GRAY2BGR565 =
      20,  //!< convert between grayscale to BGR565 (16-bit images)
  COLOR_BGR5652GRAY = 21,

  COLOR_BGR2BGR555 =
      22,  //!< convert between RGB/BGR and BGR555 (16-bit images)
  COLOR_RGB2BGR555 = 23,
  COLOR_BGR5552BGR = 24,
  COLOR_BGR5552RGB = 25,
  COLOR_BGRA2BGR555 = 26,
  COLOR_RGBA2BGR555 = 27,
  COLOR_BGR5552BGRA = 28,
  COLOR_BGR5552RGBA = 29,

  COLOR_GRAY2BGR555 =
      30,  //!< convert between grayscale and BGR555 (16-bit images)
  COLOR_BGR5552GRAY = 31,

  COLOR_BGR2XYZ = 32,  //!< convert RGB/BGR to CIE XYZ, @ref
                       //!< color_convert_rgb_xyz "color conversions"
  COLOR_RGB2XYZ = 33,
  COLOR_XYZ2BGR = 34,
  COLOR_XYZ2RGB = 35,

  COLOR_BGR2YCrCb = 36,  //!< convert RGB/BGR to luma-chroma (aka YCC), @ref
                         //!< color_convert_rgb_ycrcb "color conversions"
  COLOR_RGB2YCrCb = 37,
  COLOR_YCrCb2BGR = 38,
  COLOR_YCrCb2RGB = 39,

  COLOR_BGR2HSV = 40,  //!< convert RGB/BGR to HSV (hue saturation value) with H
                       //!< range 0..180 if 8 bit image, @ref
                       //!< color_convert_rgb_hsv "color conversions"
  COLOR_RGB2HSV = 41,

  COLOR_BGR2Lab = 44,  //!< convert RGB/BGR to CIE Lab, @ref
                       //!< color_convert_rgb_lab "color conversions"
  COLOR_RGB2Lab = 45,

  COLOR_BGR2Luv = 50,  //!< convert RGB/BGR to CIE Luv, @ref
                       //!< color_convert_rgb_luv "color conversions"
  COLOR_RGB2Luv = 51,
  COLOR_BGR2HLS = 52,  //!< convert RGB/BGR to HLS (hue lightness saturation)
                       //!< with H range 0..180 if 8 bit image, @ref
                       //!< color_convert_rgb_hls "color conversions"
  COLOR_RGB2HLS = 53,

  COLOR_HSV2BGR = 54,  //!< backward conversions HSV to RGB/BGR with H range
                       //!< 0..180 if 8 bit image
  COLOR_HSV2RGB = 55,

  COLOR_Lab2BGR = 56,
  COLOR_Lab2RGB = 57,
  COLOR_Luv2BGR = 58,
  COLOR_Luv2RGB = 59,
  COLOR_HLS2BGR = 60,  //!< backward conversions HLS to RGB/BGR with H range
                       //!< 0..180 if 8 bit image
  COLOR_HLS2RGB = 61,

  COLOR_BGR2HSV_FULL = 66,  //!< convert RGB/BGR to HSV (hue saturation value)
                            //!< with H range 0..255 if 8 bit image, @ref
                            //!< color_convert_rgb_hsv "color conversions"
  COLOR_RGB2HSV_FULL = 67,
  COLOR_BGR2HLS_FULL = 68,  //!< convert RGB/BGR to HLS (hue lightness
                            //!< saturation) with H range 0..255 if 8 bit image,
                            //!< @ref color_convert_rgb_hls "color conversions"
  COLOR_RGB2HLS_FULL = 69,

  COLOR_HSV2BGR_FULL = 70,  //!< backward conversions HSV to RGB/BGR with H
                            //!< range 0..255 if 8 bit image
  COLOR_HSV2RGB_FULL = 71,
  COLOR_HLS2BGR_FULL = 72,  //!< backward conversions HLS to RGB/BGR with H
                            //!< range 0..255 if 8 bit image
  COLOR_HLS2RGB_FULL = 73,

  COLOR_LBGR2Lab = 74,
  COLOR_LRGB2Lab = 75,
  COLOR_LBGR2Luv = 76,
  COLOR_LRGB2Luv = 77,

  COLOR_Lab2LBGR = 78,
  COLOR_Lab2LRGB = 79,
  COLOR_Luv2LBGR = 80,
  COLOR_Luv2LRGB = 81,

  COLOR_BGR2YUV = 82,  //!< convert between RGB/BGR and YUV
  COLOR_RGB2YUV = 83,
  COLOR_YUV2BGR = 84,
  COLOR_YUV2RGB = 85,

  //! YUV 4:2:0 family to RGB
  COLOR_YUV2RGB_NV12 = 90,
  COLOR_YUV2BGR_NV12 = 91,
  COLOR_YUV2RGB_NV21 = 92,
  COLOR_YUV2BGR_NV21 = 93,
  COLOR_YUV420sp2RGB = COLOR_YUV2RGB_NV21,
  COLOR_YUV420sp2BGR = COLOR_YUV2BGR_NV21,

  COLOR_YUV2RGBA_NV12 = 94,
  COLOR_YUV2BGRA_NV12 = 95,
  COLOR_YUV2RGBA_NV21 = 96,
  COLOR_YUV2BGRA_NV21 = 97,
  COLOR_YUV420sp2RGBA = COLOR_YUV2RGBA_NV21,
  COLOR_YUV420sp2BGRA = COLOR_YUV2BGRA_NV21,

  COLOR_YUV2RGB_YV12 = 98,
  COLOR_YUV2BGR_YV12 = 99,
  COLOR_YUV2RGB_IYUV = 100,
  COLOR_YUV2BGR_IYUV = 101,
  COLOR_YUV2RGB_I420 = COLOR_YUV2RGB_IYUV,
  COLOR_YUV2BGR_I420 = COLOR_YUV2BGR_IYUV,
  COLOR_YUV420p2RGB = COLOR_YUV2RGB_YV12,
  COLOR_YUV420p2BGR = COLOR_YUV2BGR_YV12,

  COLOR_YUV2RGBA_YV12 = 102,
  COLOR_YUV2BGRA_YV12 = 103,
  COLOR_YUV2RGBA_IYUV = 104,
  COLOR_YUV2BGRA_IYUV = 105,
  COLOR_YUV2RGBA_I420 = COLOR_YUV2RGBA_IYUV,
  COLOR_YUV2BGRA_I420 = COLOR_YUV2BGRA_IYUV,
  COLOR_YUV420p2RGBA = COLOR_YUV2RGBA_YV12,
  COLOR_YUV420p2BGRA = COLOR_YUV2BGRA_YV12,

  COLOR_YUV2GRAY_420 = 106,
  COLOR_YUV2GRAY_NV21 = COLOR_YUV2GRAY_420,
  COLOR_YUV2GRAY_NV12 = COLOR_YUV2GRAY_420,
  COLOR_YUV2GRAY_YV12 = COLOR_YUV2GRAY_420,
  COLOR_YUV2GRAY_IYUV = COLOR_YUV2GRAY_420,
  COLOR_YUV2GRAY_I420 = COLOR_YUV2GRAY_420,
  COLOR_YUV420sp2GRAY = COLOR_YUV2GRAY_420,
  COLOR_YUV420p2GRAY = COLOR_YUV2GRAY_420,

  //! YUV 4:2:2 family to RGB
  COLOR_YUV2RGB_UYVY = 107,
  COLOR_YUV2BGR_UYVY = 108,
  // COLOR_YUV2RGB_VYUY = 109,
  // COLOR_YUV2BGR_VYUY = 110,
  COLOR_YUV2RGB_Y422 = COLOR_YUV2RGB_UYVY,
  COLOR_YUV2BGR_Y422 = COLOR_YUV2BGR_UYVY,
  COLOR_YUV2RGB_UYNV = COLOR_YUV2RGB_UYVY,
  COLOR_YUV2BGR_UYNV = COLOR_YUV2BGR_UYVY,

  COLOR_YUV2RGBA_UYVY = 111,
  COLOR_YUV2BGRA_UYVY = 112,
  // COLOR_YUV2RGBA_VYUY = 113,
  // COLOR_YUV2BGRA_VYUY = 114,
  COLOR_YUV2RGBA_Y422 = COLOR_YUV2RGBA_UYVY,
  COLOR_YUV2BGRA_Y422 = COLOR_YUV2BGRA_UYVY,
  COLOR_YUV2RGBA_UYNV = COLOR_YUV2RGBA_UYVY,
  COLOR_YUV2BGRA_UYNV = COLOR_YUV2BGRA_UYVY,

  COLOR_YUV2RGB_YUY2 = 115,
  COLOR_YUV2BGR_YUY2 = 116,
  COLOR_YUV2RGB_YVYU = 117,
  COLOR_YUV2BGR_YVYU = 118,
  COLOR_YUV2RGB_YUYV = COLOR_YUV2RGB_YUY2,
  COLOR_YUV2BGR_YUYV = COLOR_YUV2BGR_YUY2,
  COLOR_YUV2RGB_YUNV = COLOR_YUV2RGB_YUY2,
  COLOR_YUV2BGR_YUNV = COLOR_YUV2BGR_YUY2,

  COLOR_YUV2RGBA_YUY2 = 119,
  COLOR_YUV2BGRA_YUY2 = 120,
  COLOR_YUV2RGBA_YVYU = 121,
  COLOR_YUV2BGRA_YVYU = 122,
  COLOR_YUV2RGBA_YUYV = COLOR_YUV2RGBA_YUY2,
  COLOR_YUV2BGRA_YUYV = COLOR_YUV2BGRA_YUY2,
  COLOR_YUV2RGBA_YUNV = COLOR_YUV2RGBA_YUY2,
  COLOR_YUV2BGRA_YUNV = COLOR_YUV2BGRA_YUY2,

  COLOR_YUV2GRAY_UYVY = 123,
  COLOR_YUV2GRAY_YUY2 = 124,
  // CV_YUV2GRAY_VYUY    = CV_YUV2GRAY_UYVY,
  COLOR_YUV2GRAY_Y422 = COLOR_YUV2GRAY_UYVY,
  COLOR_YUV2GRAY_UYNV = COLOR_YUV2GRAY_UYVY,
  COLOR_YUV2GRAY_YVYU = COLOR_YUV2GRAY_YUY2,
  COLOR_YUV2GRAY_YUYV = COLOR_YUV2GRAY_YUY2,
  COLOR_YUV2GRAY_YUNV = COLOR_YUV2GRAY_YUY2,

  //! alpha premultiplication
  COLOR_RGBA2mRGBA = 125,
  COLOR_mRGBA2RGBA = 126,

  //! RGB to YUV 4:2:0 family
  COLOR_RGB2YUV_I420 = 127,
  COLOR_BGR2YUV_I420 = 128,
  COLOR_RGB2YUV_IYUV = COLOR_RGB2YUV_I420,
  COLOR_BGR2YUV_IYUV = COLOR_BGR2YUV_I420,

  COLOR_RGBA2YUV_I420 = 129,
  COLOR_BGRA2YUV_I420 = 130,
  COLOR_RGBA2YUV_IYUV = COLOR_RGBA2YUV_I420,
  COLOR_BGRA2YUV_IYUV = COLOR_BGRA2YUV_I420,
  COLOR_RGB2YUV_YV12 = 131,
  COLOR_BGR2YUV_YV12 = 132,
  COLOR_RGBA2YUV_YV12 = 133,
  COLOR_BGRA2YUV_YV12 = 134,

  //! Demosaicing, see @ref color_convert_bayer "color conversions" for
  //! additional information
  COLOR_BayerBG2BGR = 46,  //!< equivalent to RGGB Bayer pattern
  COLOR_BayerGB2BGR = 47,  //!< equivalent to GRBG Bayer pattern
  COLOR_BayerRG2BGR = 48,  //!< equivalent to BGGR Bayer pattern
  COLOR_BayerGR2BGR = 49,  //!< equivalent to GBRG Bayer pattern

  COLOR_BayerRGGB2BGR = COLOR_BayerBG2BGR,
  COLOR_BayerGRBG2BGR = COLOR_BayerGB2BGR,
  COLOR_BayerBGGR2BGR = COLOR_BayerRG2BGR,
  COLOR_BayerGBRG2BGR = COLOR_BayerGR2BGR,

  COLOR_BayerRGGB2RGB = COLOR_BayerBGGR2BGR,
  COLOR_BayerGRBG2RGB = COLOR_BayerGBRG2BGR,
  COLOR_BayerBGGR2RGB = COLOR_BayerRGGB2BGR,
  COLOR_BayerGBRG2RGB = COLOR_BayerGRBG2BGR,

  COLOR_BayerBG2RGB = COLOR_BayerRG2BGR,  //!< equivalent to RGGB Bayer pattern
  COLOR_BayerGB2RGB = COLOR_BayerGR2BGR,  //!< equivalent to GRBG Bayer pattern
  COLOR_BayerRG2RGB = COLOR_BayerBG2BGR,  //!< equivalent to BGGR Bayer pattern
  COLOR_BayerGR2RGB = COLOR_BayerGB2BGR,  //!< equivalent to GBRG Bayer pattern

  COLOR_BayerBG2GRAY = 86,  //!< equivalent to RGGB Bayer pattern
  COLOR_BayerGB2GRAY = 87,  //!< equivalent to GRBG Bayer pattern
  COLOR_BayerRG2GRAY = 88,  //!< equivalent to BGGR Bayer pattern
  COLOR_BayerGR2GRAY = 89,  //!< equivalent to GBRG Bayer pattern

  COLOR_BayerRGGB2GRAY = COLOR_BayerBG2GRAY,
  COLOR_BayerGRBG2GRAY = COLOR_BayerGB2GRAY,
  COLOR_BayerBGGR2GRAY = COLOR_BayerRG2GRAY,
  COLOR_BayerGBRG2GRAY = COLOR_BayerGR2GRAY,

  //! Demosaicing using Variable Number of Gradients
  COLOR_BayerBG2BGR_VNG = 62,  //!< equivalent to RGGB Bayer pattern
  COLOR_BayerGB2BGR_VNG = 63,  //!< equivalent to GRBG Bayer pattern
  COLOR_BayerRG2BGR_VNG = 64,  //!< equivalent to BGGR Bayer pattern
  COLOR_BayerGR2BGR_VNG = 65,  //!< equivalent to GBRG Bayer pattern

  COLOR_BayerRGGB2BGR_VNG = COLOR_BayerBG2BGR_VNG,
  COLOR_BayerGRBG2BGR_VNG = COLOR_BayerGB2BGR_VNG,
  COLOR_BayerBGGR2BGR_VNG = COLOR_BayerRG2BGR_VNG,
  COLOR_BayerGBRG2BGR_VNG = COLOR_BayerGR2BGR_VNG,

  COLOR_BayerRGGB2RGB_VNG = COLOR_BayerBGGR2BGR_VNG,
  COLOR_BayerGRBG2RGB_VNG = COLOR_BayerGBRG2BGR_VNG,
  COLOR_BayerBGGR2RGB_VNG = COLOR_BayerRGGB2BGR_VNG,
  COLOR_BayerGBRG2RGB_VNG = COLOR_BayerGRBG2BGR_VNG,

  COLOR_BayerBG2RGB_VNG =
      COLOR_BayerRG2BGR_VNG,  //!< equivalent to RGGB Bayer pattern
  COLOR_BayerGB2RGB_VNG =
      COLOR_BayerGR2BGR_VNG,  //!< equivalent to GRBG Bayer pattern
  COLOR_BayerRG2RGB_VNG =
      COLOR_BayerBG2BGR_VNG,  //!< equivalent to BGGR Bayer pattern
  COLOR_BayerGR2RGB_VNG =
      COLOR_BayerGB2BGR_VNG,  //!< equivalent to GBRG Bayer pattern

  //! Edge-Aware Demosaicing
  COLOR_BayerBG2BGR_EA = 135,  //!< equivalent to RGGB Bayer pattern
  COLOR_BayerGB2BGR_EA = 136,  //!< equivalent to GRBG Bayer pattern
  COLOR_BayerRG2BGR_EA = 137,  //!< equivalent to BGGR Bayer pattern
  COLOR_BayerGR2BGR_EA = 138,  //!< equivalent to GBRG Bayer pattern

  COLOR_BayerRGGB2BGR_EA = COLOR_BayerBG2BGR_EA,
  COLOR_BayerGRBG2BGR_EA = COLOR_BayerGB2BGR_EA,
  COLOR_BayerBGGR2BGR_EA = COLOR_BayerRG2BGR_EA,
  COLOR_BayerGBRG2BGR_EA = COLOR_BayerGR2BGR_EA,

  COLOR_BayerRGGB2RGB_EA = COLOR_BayerBGGR2BGR_EA,
  COLOR_BayerGRBG2RGB_EA = COLOR_BayerGBRG2BGR_EA,
  COLOR_BayerBGGR2RGB_EA = COLOR_BayerRGGB2BGR_EA,
  COLOR_BayerGBRG2RGB_EA = COLOR_BayerGRBG2BGR_EA,

  COLOR_BayerBG2RGB_EA =
      COLOR_BayerRG2BGR_EA,  //!< equivalent to RGGB Bayer pattern
  COLOR_BayerGB2RGB_EA =
      COLOR_BayerGR2BGR_EA,  //!< equivalent to GRBG Bayer pattern
  COLOR_BayerRG2RGB_EA =
      COLOR_BayerBG2BGR_EA,  //!< equivalent to BGGR Bayer pattern
  COLOR_BayerGR2RGB_EA =
      COLOR_BayerGB2BGR_EA,  //!< equivalent to GBRG Bayer pattern

  //! Demosaicing with alpha channel
  COLOR_BayerBG2BGRA = 139,  //!< equivalent to RGGB Bayer pattern
  COLOR_BayerGB2BGRA = 140,  //!< equivalent to GRBG Bayer pattern
  COLOR_BayerRG2BGRA = 141,  //!< equivalent to BGGR Bayer pattern
  COLOR_BayerGR2BGRA = 142,  //!< equivalent to GBRG Bayer pattern

  COLOR_BayerRGGB2BGRA = COLOR_BayerBG2BGRA,
  COLOR_BayerGRBG2BGRA = COLOR_BayerGB2BGRA,
  COLOR_BayerBGGR2BGRA = COLOR_BayerRG2BGRA,
  COLOR_BayerGBRG2BGRA = COLOR_BayerGR2BGRA,

  COLOR_BayerRGGB2RGBA = COLOR_BayerBGGR2BGRA,
  COLOR_BayerGRBG2RGBA = COLOR_BayerGBRG2BGRA,
  COLOR_BayerBGGR2RGBA = COLOR_BayerRGGB2BGRA,
  COLOR_BayerGBRG2RGBA = COLOR_BayerGRBG2BGRA,

  COLOR_BayerBG2RGBA =
      COLOR_BayerRG2BGRA,  //!< equivalent to RGGB Bayer pattern
  COLOR_BayerGB2RGBA =
      COLOR_BayerGR2BGRA,  //!< equivalent to GRBG Bayer pattern
  COLOR_BayerRG2RGBA =
      COLOR_BayerBG2BGRA,  //!< equivalent to BGGR Bayer pattern
  COLOR_BayerGR2RGBA =
      COLOR_BayerGB2BGRA,  //!< equivalent to GBRG Bayer pattern

  COLOR_COLORCVT_MAX = 143
};

enum InterpolationFlags {
  INTER_NEAREST = 0,
  INTER_LINEAR = 1,
  INTER_CUBIC = 2,
  INTER_AREA = 3,
  INTER_LANCZOS4 = 4,
  INTER_LINEAR_EXACT = 5,
  INTER_NEAREST_EXACT = 6,
  INTER_MAX = 7,
  WARP_FILL_OUTLIERS = 8,
  WARP_INVERSE_MAP = 16
};

enum LineTypes { FILLED = -1, LINE_4 = 4, LINE_8 = 8, LINE_AA = 16 };

template <typename T, typename TT>
bool ConvertTo(const Image<T>& src, Image<TT>* dst, float scale = 1.0f) {
  if (src.channels() != dst->channels()) {
    LOGE("ConvertTo failed src channel %d, dst channel %d\n", src.channels(),
         dst->channels());
    return false;
  }
  Init(dst, src.cols, src.rows);

#define UGU_CAST1(type) \
  (static_cast<double>(reinterpret_cast<const type*>(&src_at)[c]))
#define UGU_CAST2(type) \
  (reinterpret_cast<type*>(&dst_at)[c] = static_cast<type>(scale * val))

  const std::type_info* cpp_type_src = &GetTypeidFromCvType(src.type());
  const std::type_info* cpp_type_dst = &GetTypeidFromCvType(dst->type());

  for (int y = 0; y < src.rows; y++) {
    for (int x = 0; x < src.cols; x++) {
      TT& dst_at = dst->template at<TT>(y, x);
      const T& src_at = src.template at<T>(y, x);
      for (int c = 0; c < dst->channels(); c++) {
#if 1
        double val = 0.0;
        if (*cpp_type_src == typeid(uint8_t)) {
          val = UGU_CAST1(uint8_t);
        } else if (*cpp_type_src == typeid(int8_t)) {
          val = UGU_CAST1(int8_t);
        } else if (*cpp_type_src == typeid(uint16_t)) {
          val = UGU_CAST1(uint16_t);
        } else if (*cpp_type_src == typeid(int16_t)) {
          val = UGU_CAST1(int16_t);
        } else if (*cpp_type_src == typeid(int32_t)) {
          val = UGU_CAST1(int32_t);
        } else if (*cpp_type_src == typeid(float)) {
          val = UGU_CAST1(float);
        } else if (*cpp_type_src == typeid(double)) {
          val = UGU_CAST1(double);
        } else {
          throw std::runtime_error("type error");
        }

        if (*cpp_type_dst == typeid(uint8_t)) {
          UGU_CAST2(uint8_t);
        } else if (*cpp_type_dst == typeid(int8_t)) {
          UGU_CAST2(int8_t);
        } else if (*cpp_type_dst == typeid(uint16_t)) {
          UGU_CAST2(uint16_t);
        } else if (*cpp_type_dst == typeid(int16_t)) {
          UGU_CAST2(int16_t);
        } else if (*cpp_type_dst == typeid(int32_t)) {
          UGU_CAST2(int32_t);
        } else if (*cpp_type_dst == typeid(float)) {
          UGU_CAST2(float);
        } else if (*cpp_type_dst == typeid(double)) {
          UGU_CAST2(double);
        } else {
          throw std::runtime_error("type error");
        }

#endif
      }
    }
  }
#undef UGU_CAST1
#undef UGU_CAST2

  return true;
}

template <typename T>
void minMaxLoc(const Image<T>& src, double* minVal, double* maxVal = nullptr,
               Point* minLoc = nullptr, Point* maxLoc = nullptr) {
  if (src.channels() != 1 || minVal == nullptr) {
    return;
  }

  double minVal_ = std::numeric_limits<double>::max();
  double maxVal_ = std::numeric_limits<double>::lowest();
  Point minLoc_, maxLoc_;

  for (int y = 0; y < src.rows; y++) {
    for (int x = 0; x < src.cols; x++) {
      const T& val_c = src.template at<T>(y, x);
      for (int c = 0; c < src.channels(); c++) {
        const auto& val = (&val_c)[c];
        if (val < minVal_) {
          minVal_ = val;
          minLoc_.x = x;
          minLoc_.y = y;
        }
        if (val > maxVal_) {
          maxVal_ = val;
          maxLoc_.x = x;
          maxLoc_.y = y;
        }
      }
    }
  }

  if (minVal != nullptr) {
    *minVal = minVal_;
  }
  if (maxVal != nullptr) {
    *maxVal = maxVal_;
  }
  if (minLoc != nullptr) {
    *minLoc = minLoc_;
  }
  if (maxLoc != nullptr) {
    *maxLoc = maxLoc_;
  }
}

template <typename T>
void resize(const Image<T>& src, Image<T>& dst, Size dsize, double fx = 0.0,
            double fy = 0.0,
            int interpolation = InterpolationFlags::INTER_LINEAR) {
#ifdef UGU_USE_STB
  (void)interpolation;

  int w = src.cols;
  int h = src.rows;
  int n = src.channels();

  int out_w, out_h;
  if (dsize.height <= 0 || dsize.width <= 0) {
    out_w = static_cast<int>(w * fx);
    out_h = static_cast<int>(h * fy);
  } else {
    out_w = dsize.width;
    out_h = dsize.height;
  }

  if (w <= 0 || h <= 0 || out_w <= 0 || out_h <= 0) {
    LOGE("Wrong size\n");
    return;
  }

  dst = Image<T>::zeros(out_h, out_w);

  stbir_resize_uint8(src.data, w, h, 0, dst.data, out_w, out_h, 0, n);

  return;
#else
  (void)src;
  (void)dst;
  (void)dsize;
  (void)fx;
  (void)fy;
  (void)interpolation;
  LOGE("can't resize image with this configuration\n");
  return;
#endif
}

template <typename T>
void circle(Image<T>& img, Point center, int radius, const T& color,
            int thickness = 1, int lineType = LINE_8, int shift = 0) {
  (void)lineType;
  (void)shift;
  auto min_x = std::max(0, std::min({center.x - radius, img.cols - 1}));
  auto max_x = std::min(img.cols - 1, std::max({center.x + radius, 0}));
  auto min_y = std::max(0, std::min({center.y - radius, img.rows - 1}));
  auto max_y = std::min(img.rows - 1, std::max({center.y + radius, 0}));

  float radius_f = static_cast<float>(radius);
  float thickness_f = static_cast<float>(thickness);

  for (int y = min_y; y <= max_y; y++) {
    for (int x = min_x; x <= max_x; x++) {
      float dist = std::sqrt(static_cast<float>(
          (center.x - x) * (center.x - x) + (center.y - y) * (center.y - y)));
      if (thickness < 0) {
        if (dist <= radius_f) {
          img.template at<T>(y, x) = color;
        }
      } else {
        if (dist < radius_f && radius_f - dist <= thickness_f) {
          img.template at<T>(y, x) = color;
        }
      }
    }
  }
}

template <typename T>
void line(Image<T>& img, Point pt1, Point pt2, const T& color,
          int thickness = 1, int lineType = 8, int shift = 0) {
  (void)lineType;
  (void)shift;

  // Naive implementation of "All cases"
  // https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm

  thickness = std::max(1, thickness);

  auto plotLineLow = [&](int x0, int y0, int x1, int y1) {
    int dx = x1 - x0;
    int dy = y1 - y0;
    int yi = 1;
    if (dy < 0) {
      yi = -1;
      dy = -dy;
    }

    int D = (2 * dy) - dx;
    int y = y0;

    x0 = std::clamp(x0, 0, img.cols - 1);
    x1 = std::clamp(x1, 0, img.cols - 1);

    for (int x = x0; x <= x1; x++) {
      for (int t = 0; t < thickness; t++) {
        int y_ = std::clamp(t + y, 0, img.rows - 1);
        img.template at<T>(y_, x) = color;
      }
      if (D > 0) {
        y = y + yi;
        D = D + (2 * (dy - dx));
      } else {
        D = D + 2 * dy;
      }
    }
  };

  auto plotLineHigh = [&](int x0, int y0, int x1, int y1) {
    int dx = x1 - x0;
    int dy = y1 - y0;
    int xi = 1;
    if (dx < 0) {
      xi = -1;
      dx = -dx;
    }

    int D = (2 * dx) - dy;
    int x = x0;

    y0 = std::clamp(y0, 0, img.rows - 1);
    y1 = std::clamp(y1, 0, img.rows - 1);

    for (int y = y0; y <= y1; y++) {
      for (int t = 0; t < thickness; t++) {
        int x_ = std::clamp(t + x, 0, img.cols - 1);
        img.template at<T>(y, x_) = color;
      }
      if (D > 0) {
        x = x + xi;
        D = D + (2 * (dx - dy));
      } else {
        D = D + 2 * dx;
      }
    }
  };

  int x1 = pt1.x;
  int y1 = pt1.y;
  int x2 = pt2.x;
  int y2 = pt2.y;
  if (std::abs(y2 - y1) < std::abs(x2 - x1)) {
    if (x1 > x2) {
      plotLineLow(x2, y2, x1, y1);
    } else {
      plotLineLow(x1, y1, x2, y2);
    }
  } else {
    if (y1 > y2) {
      plotLineHigh(x2, y2, x1, y1);
    } else {
      plotLineHigh(x1, y1, x2, y2);
    }
  }
}

#endif

}  // namespace ugu
