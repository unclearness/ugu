/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 *
 */

#pragma once

#include "ugu/image.h"

namespace ugu {

// Implements Fast Marching Method (FMM) used in the following paper.
// Compared to Distance Transfrom, FMM is slower but more accurate and more
// smooth. Telea, Alexandru. "An image inpainting technique based on the fast
// marching method." Journal of graphics tools 9.1 (2004): 23-34.
void FastMarchingMethod(const Image1b& mask, Image1f* dist,
                        const float illegal_val = 0.f);

enum class InpaintMethod { NAIVE, TELEA };

void Inpaint(const Image1b& mask, Image3b& color, float inpaint_radius = 5.0f,
             InpaintMethod method = InpaintMethod::TELEA);

}  // namespace ugu

