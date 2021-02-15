/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 *
 * Implementation of the following paper
 *
 * Simakov, Denis, et al. "Summarizing visual data using bidirectional
 * similarity." 2008 IEEE Conference on Computer Vision and Pattern Recognition.
 * IEEE, 2008.
 *
 */

#pragma once

#if __has_include("../third_party/nanopm/nanopm.h")

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
  int iteration_in_scale = 10;

  float rescale_ratio = 0.05f;
  Size target_size;

  bool verbose = false;
  std::string debug_dir = "";
};

bool Synthesize(const Image3b& src, Image3b& dst, const BdsimParams& params);

}  // namespace ugu

#endif