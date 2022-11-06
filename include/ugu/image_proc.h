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

#ifdef UGU_USE_OPENCV

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

#undef UGU_CAST

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
