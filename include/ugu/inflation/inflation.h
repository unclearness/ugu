/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/image.h"

namespace ugu {

enum class InflationMethod {
  BARAN // Notes on Inflating Curves" [Baran and Lehtinen 2009].
};


bool Inflation(const Image1b& mask, Image1f& height, InflationMethod method = InflationMethod::BARAN);


}  // namespace ugu
