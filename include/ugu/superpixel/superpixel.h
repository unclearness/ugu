/*
 * Copyright (C) 2023, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/image.h"

namespace ugu {

void Slic(const ImageBase& img, Image1i& labels, Image1b& contour_mask,
          int& sp_num, int region_size = 20, float ruler = 30.f,
          int min_element_size_percent = 10, int num_iterations = 4);

void SimilarColorClustering(const ImageBase& img, Image1i& labels,
                            int& labels_num, int region_size = 20,
                            float ruler = 30.f,
                            int min_element_size_percent = 10,
                            int num_iterations = 4, size_t min_clusters = 2,
                            double max_color_diff = 40.0,
                            double max_boundary_strengh = 80.0);

}  // namespace ugu
