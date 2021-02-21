/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/image.h"
#include "ugu/mesh.h"
#include "ugu/inpaint/inpaint.h"

namespace ugu {

enum class InflationMethod {
  BARAN  // "Notes on Inflating Curves" [Baran and Lehtinen 2009].
         // http://alecjacobson.com/weblog/media/notes-on-inflating-curves-2009-baran.pdf
};

enum class InflationBackTextureType { MIRRORED, INPAINT };

struct InflationParams {
  // Common
  InflationMethod method = InflationMethod::BARAN;
  bool inverse = false;

  // For mesh
  Image3b* texture = nullptr;
  bool generate_back = false;
  InflationBackTextureType back_texture_type =
      InflationBackTextureType::MIRRORED;
  bool centering = true;

  // For InflationBackTextureType::INPAINT
  int inpaint_kernel_size = 5;
  InpaintMethod inpaint_method = InpaintMethod::TELEA;
};

bool Inflation(const Image1b& mask, Image1f& height,
               const InflationParams& params = InflationParams());
bool Inflation(const Image1b& mask, Image1f& height, Mesh& mesh,
               const InflationParams& params = InflationParams());

}  // namespace ugu
