/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/superpixel/superpixel.h"

#ifdef UGU_USE_OPENCV
#ifdef _WIN32
#pragma warning(push, UGU_OPENCV_WARNING_LEVEL)
#endif
#include "opencv2/ximgproc.hpp"
#ifdef _WIN32
#pragma warning(pop)
#endif
#endif

namespace ugu {
void Slic(const ImageBase& img, Image1i& labels, Image1b& contour_mask,
          int region_size, float ruler, int min_element_size_percent,
          int num_iterations) {
#ifdef UGU_USE_OPENCV
  auto slic = cv::ximgproc::createSuperpixelSLIC(img, cv::ximgproc::SLIC,
                                                 region_size, ruler);

  slic->iterate(num_iterations);

  slic->enforceLabelConnectivity(min_element_size_percent);

  slic->getLabels(labels);

  slic->getLabelContourMask(contour_mask, false);
#else
  (void)img, (void)labels, (void)contour_mask, (void)region_size, (void)ruler,
      (void)min_element_size_percent, (void)num_iterations;

  LOGE("Not avairable with this configuration\n");
#endif
}

}  // namespace ugu
