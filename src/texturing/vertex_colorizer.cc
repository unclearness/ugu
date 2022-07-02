/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "ugu/texturing/vertex_colorizer.h"

namespace ugu {

VertexColorizer::VertexColorizer() {}
VertexColorizer::~VertexColorizer() {}
bool VertexColorizer::Colorize(const VisibilityInfo& info, Mesh* mesh,
                               ViewSelectionCriteria criteria) const {
  if (!info.has_vertex_stat) {
    LOGE("no vertex stat\n");
    return false;
  }

  assert(info.vertex_info_list.size() == mesh->vertices().size());

  std::vector<Eigen::Vector3f> vertex_colors;

  std::function<Eigen::Vector3f(const VertexInfo& info)> select =
      [&](const VertexInfo& info) -> Eigen::Vector3f {
    return info.min_viewing_angle_color;
  };
  if (criteria == ViewSelectionCriteria::kMinViewingAngle) {
    select = [&](const VertexInfo& info) {
      return info.min_viewing_angle_color;
    };
  } else if (criteria == ViewSelectionCriteria::kMinDistance) {
    select = [&](const VertexInfo& info) { return info.min_distance_color; };
  } else if (criteria == ViewSelectionCriteria::kMeanViewingAngle) {
    select = [&](const VertexInfo& info) {
      return info.mean_viewing_angle_color;
    };
  } else if (criteria == ViewSelectionCriteria::kMedianViewingAngle) {
    select = [&](const VertexInfo& info) {
      return info.median_viewing_angle_color;
    };
  } else if (criteria == ViewSelectionCriteria::kMeanDistance) {
    select = [&](const VertexInfo& info) { return info.mean_distance_color; };
  } else if (criteria == ViewSelectionCriteria::kMedianDistance) {
    select = [&](const VertexInfo& info) { return info.median_distance_color; };
  } else if (criteria == ViewSelectionCriteria::kMinIntensity) {
    select = [&](const VertexInfo& info) { return info.min_intensity_color; };
  } else if (criteria == ViewSelectionCriteria::kMedianIntensity) {
    select = [&](const VertexInfo& info) {
      return info.median_intensity_color;
    };
  } else if (criteria == ViewSelectionCriteria::kMode) {
    select = [&](const VertexInfo& info) {
      return info.mode;
      ;
    };
  } else if (criteria == ViewSelectionCriteria::kModeViewingAngle) {
    select = [&](const VertexInfo& info) {
      return info.mode_viewing_angle_color;
      ;
    };
  } else if (criteria == ViewSelectionCriteria::kCustom) {
    select = [&](const VertexInfo& info) {
      return info.custom_best;
      ;
    };
  }

  for (size_t i = 0; i < info.vertex_info_list.size(); i++) {
    vertex_colors.push_back(select(info.vertex_info_list[i]));
  }

  mesh->set_vertex_colors(vertex_colors);

  return true;
}

}  // namespace ugu
