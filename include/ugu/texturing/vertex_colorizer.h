/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include "visibility_tester.h"

namespace ugu {

class VertexColorizer {
 public:
  VertexColorizer();
  ~VertexColorizer();
  bool Colorize(const VisibilityInfo& info, Mesh* mesh,
                ViewSelectionCriteria criteria =
                    ViewSelectionCriteria::kMinViewingAngle) const;
};

}  // namespace ugu
