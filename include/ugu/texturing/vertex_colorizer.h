/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#ifdef UGU_USE_NANORT

#include "visibility_tester.h"

namespace ugu {

enum class ColorCriteria {
  kMinViewingAngle = 0,
  kMinDistance = 1,
  kMeanViewingAngle = 2,
  kMedianViewingAngle = 3,
  kMeanDistance = 4,
  kMedianDistance = 5
};


class VertexColorizer {
 public:
  VertexColorizer();
  ~VertexColorizer();
  bool Colorize(const VisibilityInfo& info, Mesh* mesh,
                ColorCriteria criteria = ColorCriteria::kMinViewingAngle) const;
};

}  // namespace ugu

#endif