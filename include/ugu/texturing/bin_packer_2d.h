/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/image.h"

namespace ugu {

#ifdef UGU_USE_OPENCV
using Rect = cv::Rect;
#else
template <typename T>
struct Rect_ {
  T x, y;
  T width, height;
  T area() const { return width * height; }
  Point_<T, 2> br() const {
    Point_<T, 2> br;
    br[0] = x + width;
    br[1] = y + height return br;
  }
  Point_<T, 2> tl() const {
    Point_<T, 2> tl;
    tl[0] = x;
    tl[1] = y;
    return tl;
  }
  Rect_(T _x, T _y, int _width, int _height)
      : x(_x), y(_y), width(_width), height(_height) {}
  Rect_() : x(0), y(0), width(0), height(0) {}
};

using Rect = Rect_<int>;

#endif

// rects should be sorted
bool BinPacking2D(const std::vector<Rect>& rects, std::vector<Rect>* packed_pos,
                  std::vector<Rect>* available_rects,
                  int w, int h);

}  // namespace ugu
