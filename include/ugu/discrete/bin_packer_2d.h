/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/rect.h"

namespace ugu {

// Reference: https://blackpawn.com/texts/lightmaps/default.html
// To get better result, rects should be sorted in advance
bool BinPacking2D(const std::vector<Rect>& rects, std::vector<Rect>* packed_pos,
                  std::vector<Rect>* available_rects, int x_min, int x_max,
                  int y_min, int y_max);
bool BinPacking2D(const std::vector<Rect2f>& rects,
                  std::vector<Rect2f>* packed_pos,
                  std::vector<Rect2f>* available_rects, float x_min,
                  float x_max, float y_min, float y_max);

Image3b DrawPackedRects(const std::vector<Rect>& packed_rects, int w, int h);
Image3b DrawPackedRects(const std::vector<Rect2f>& packed_rects, int w, int h);
}  // namespace ugu
