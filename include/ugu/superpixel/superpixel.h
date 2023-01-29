/*
 * Copyright (C) 2023, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/image.h"

namespace ugu {

void Slic(const ImageBase& img, Image1i& labels, Image1b& contour_mask,
          int region_size = 20, float ruler = 30.f,
          int min_element_size_percent = 10, int num_iterations = 4);

}  // namespace ugu
