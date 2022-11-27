/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/image.h"

#ifdef UGU_USE_OPENCV
#include "opencv2/imgcodecs.hpp"
#endif

namespace ugu {

#ifdef UGU_USE_OPENCV

using ImreadModes = cv::ImreadModes;

using cv::imread;
using cv::imwrite;

#else

bool WriteBinary(const ImageBase& img, const std::string& path);

enum ImreadModes {
  IMREAD_UNCHANGED = -1,
  IMREAD_GRAYSCALE = 0,
  IMREAD_COLOR = 1,
  IMREAD_ANYDEPTH = 2,
  IMREAD_ANYCOLOR = 4,
  IMREAD_LOAD_GDAL = 8,
  IMREAD_REDUCED_GRAYSCALE_2 = 16,
  IMREAD_REDUCED_COLOR_2 = 17,
  IMREAD_REDUCED_GRAYSCALE_4 = 32,
  IMREAD_REDUCED_COLOR_4 = 33,
  IMREAD_REDUCED_GRAYSCALE_8 = 64,
  IMREAD_REDUCED_COLOR_8 = 65,
  IMREAD_IGNORE_ORIENTATION = 128,
};

bool imwrite(const std::string& filename, const ImageBase& img,
             const std::vector<int>& params = std::vector<int>());
ImageBase imread(const std::string& filename,
                 int flags = ImreadModes::IMREAD_COLOR);
#endif

template <typename T>
T Imread(const std::string& filename, int flags = ImreadModes::IMREAD_COLOR) {
  ImageBase loaded = imread(filename, flags);

  if (loaded.channels() != T().channels() ||
      loaded.elemSize1() != T().elemSize1()) {
    LOGE("desired channel %d, actual %d\n", T().channels(), loaded.channels());
    return T();
  }

  return loaded;
}

}  // namespace ugu
