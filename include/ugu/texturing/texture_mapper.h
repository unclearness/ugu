/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#ifdef UGU_USE_NANORT

#include "visibility_tester.h"

namespace ugu {

enum TextureMappingType { kSimpleProjection = 0 };

enum OutputUvType {
  kGenerateSimpleTile = 0,
  kUseOriginalMeshUv = 1,
  kGenerateSimpleTriangles = 2,
  kGenerateSimpleCharts = 3
};

struct TextureMappingOption {
  ViewSelectionCriteria criteria = ViewSelectionCriteria::kMaxArea;
  TextureMappingType type = TextureMappingType::kSimpleProjection;
  OutputUvType uv_type = OutputUvType::kGenerateSimpleTile;
  std::string texture_base_name = "ugutex";
  int tex_w = 1024;
  int tex_h = 1024;
  int padding_kernel = 3;
};

bool TextureMapping(const std::vector<std::shared_ptr<Keyframe>>& keyframes,
                    const VisibilityInfo& info, Mesh* mesh,
                    const TextureMappingOption& option);

bool Parameterize(Mesh& mesh, int tex_w = 1024, int tex_h = 1024,
                  OutputUvType type = OutputUvType::kGenerateSimpleTriangles);

bool Parameterize(const std::vector<Eigen::Vector3f>& vertices,
                  const std::vector<Eigen::Vector3i>& faces,
                  std::vector<Eigen::Vector2f>& uvs,
                  std::vector<Eigen::Vector3i>& uv_faces, int tex_w = 1024,
                  int tex_h = 1024,
                  OutputUvType type = OutputUvType::kGenerateSimpleTriangles);

}  // namespace ugu

#endif