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

enum class PointOnFaceType {
  NAMED_POINT_ON_TRIANGLE,
  POINT_ON_TRIANGLE,
  THREED_POINT
};

struct PointOnFace {
  std::string name;
  uint32_t fid = ~0u;
  float u = -1.f;
  float v = -1.f;
  Eigen::Vector3f pos;
};

std::vector<PointOnFace> LoadPoints(const std::string& json_path,
                                    const PointOnFaceType& type);

void WritePoints(const std::string& json_path,
                 const std::vector<PointOnFace>& points,
                 const PointOnFaceType& type);

}  // namespace ugu
