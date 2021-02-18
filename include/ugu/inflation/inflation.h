/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/image.h"

namespace ugu {

enum class InflationMethod {
  BARAN  // "Notes on Inflating Curves" [Baran and Lehtinen 2009].
         // http://alecjacobson.com/weblog/media/notes-on-inflating-curves-2009-baran.pdf
};

bool Inflation(const Image1b& mask, Image1f& height, bool inverse = false,
               InflationMethod method = InflationMethod::BARAN);

}  // namespace ugu
