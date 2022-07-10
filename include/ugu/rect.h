/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/image.h"

namespace ugu {

#ifdef UGU_USE_OPENCV
using Rect = cv::Rect;
using Rect2f = cv::Rect2f;
#else
template <typename T>
struct Rect_ {
  T x, y;
  T width, height;
  T area() const { return width * height; }
  Point_<T, 2> br() const {
    Point_<T, 2> br;
    br[0] = x + width;
    br[1] = y + height;
    return br;
  }
  Point_<T, 2> tl() const {
    Point_<T, 2> tl;
    tl[0] = x;
    tl[1] = y;
    return tl;
  }
  Rect_(T _x, T _y, T _width, T _height)
      : x(_x), y(_y), width(_width), height(_height) {}
  Rect_() : x(0), y(0), width(0), height(0) {}
  ~Rect_() {}
};

using Rect2i = Rect_<int>;
using Rect = Rect2i;
using Rect2f = Rect_<float>;

#endif

}  // namespace ugu
