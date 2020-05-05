/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "vertex_colorizer.h"

namespace ugu {

VertexColorizer::VertexColorizer() {}
VertexColorizer::~VertexColorizer() {}
bool VertexColorizer::Colorize(const VisibilityInfo& info, Mesh* mesh,
                               ColorCriteria criteria) const {
  if (!info.has_vertex_stat) {
    LOGE("no vertex stat\n");
    return false;
  }

  assert(info.vertex_info_list.size() == mesh->vertices().size());

  std::vector<Eigen::Vector3f> vertex_colors;

  for (size_t i = 0; i < info.vertex_info_list.size(); i++) {
    if (criteria == ColorCriteria::kMinViewingAngle) {
      vertex_colors.push_back(info.vertex_info_list[i].min_viewing_angle_color);
    } else if (criteria == ColorCriteria::kMinDistance) {
      vertex_colors.push_back(info.vertex_info_list[i].min_distance_color);
    } else if (criteria == ColorCriteria::kMeanViewingAngle) {
      vertex_colors.push_back(
          info.vertex_info_list[i].mean_viewing_angle_color);
    } else if (criteria == ColorCriteria::kMedianViewingAngle) {
      vertex_colors.push_back(
          info.vertex_info_list[i].median_viewing_angle_color);
    } else if (criteria == ColorCriteria::kMeanDistance) {
      vertex_colors.push_back(info.vertex_info_list[i].mean_distance_color);
    } else if (criteria == ColorCriteria::kMedianDistance) {
      vertex_colors.push_back(info.vertex_info_list[i].median_distance_color);
    }
  }

  mesh->set_vertex_colors(vertex_colors);

  return true;
}

}  // namespace ugu
