/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#ifdef UGU_USE_NANORT

#include "visibility_tester.h"

namespace ugu {

enum TextureMappingType { kSimpleProjection = 0 };

enum OutputUvType { kGenerateSimpleTile = 0, kUseOriginalMeshUv = 1 };

struct TextureMappingOption {
  ViewSelectionCriteria criteria = ViewSelectionCriteria::kMaxArea;
  TextureMappingType type = TextureMappingType::kSimpleProjection;
  OutputUvType uv_type = OutputUvType::kGenerateSimpleTile;
};

bool TextureMapping(const std::vector<std::shared_ptr<Keyframe>>& keyframes,
                    const VisibilityInfo& info, Mesh* mesh,
                    const TextureMappingOption& option);

}  // namespace ugu

#endif