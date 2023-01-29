/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#pragma once

#include "image.h"

#ifdef UGU_USE_OPENCV
#ifdef _WIN32
#pragma warning(push, UGU_OPENCV_WARNING_LEVEL)
#endif
#include "opencv2/core.hpp"
#ifdef _WIN32
#pragma warning(pop)
#endif
#endif

namespace ugu {

#ifdef UGU_USE_OPENCV

using Point = cv::Point;

#else

template <typename TT, int N>
using Point_ = Vec_<TT, N>;

class Point2i {
 public:
  int x = -1;
  int y = -1;
  Point2i(){};
  Point2i(int x_, int y_) : x(x_), y(y_){};
  ~Point2i(){};
};

using Point = Point2i;

#endif

}  // namespace ugu
