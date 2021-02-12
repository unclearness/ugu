/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/image.h"

namespace ugu {

enum class BdsimPatchSearchMethod { BRUTE_FORCE, PATCH_MATCH };
enum class BdsimPatchDistanceType { SSD };

struct BdsimParams {
  float alpha = 0.5f;
  int patch_size = 7;
  BdsimPatchSearchMethod patch_search_method =
      BdsimPatchSearchMethod::PATCH_MATCH;
  BdsimPatchDistanceType distance_type = BdsimPatchDistanceType::SSD;
  int iteration_in_scale = 5;

  float rescale_ratio = 0.05f;
  Size target_size;

  bool verbose;
  std::string debug_dir;

};

bool Synthesize(const Image3b& src, Image3b& dst, const BdsimParams& params);

}  // namespace ugu
