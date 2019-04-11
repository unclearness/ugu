/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include "currender/util.h"

namespace currender {

void Depth2PointCloud(const Image1f& depth, const Camera& camera,
                      Image3f* point_cloud) {
  point_cloud->Init(depth.width(), depth.height(), 0.0f);

#if defined(_OPENMP) && defined(CURRENDER_USE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int y = 0; y < camera.height(); y++) {
    for (int x = 0; x < camera.width(); x++) {
      const float& d = depth.at(x, y, 0);
      if (d < std::numeric_limits<float>::min()) {
        continue;
      }

      Eigen::Vector3f image_p(static_cast<float>(x), static_cast<float>(y), d);
      Eigen::Vector3f camera_p;
      camera.Unproject(image_p, &camera_p);
      point_cloud->at(x, y, 0) = camera_p[0];
      point_cloud->at(x, y, 1) = camera_p[1];
      point_cloud->at(x, y, 2) = camera_p[2];
    }
  }
}

void Depth2PointCloud(const Image1f& depth, const Camera& camera,
                      Mesh* point_cloud) {
  point_cloud->Clear();

  std::vector<Eigen::Vector3f> vertices;

  for (int y = 0; y < camera.height(); y++) {
    for (int x = 0; x < camera.width(); x++) {
      const float& d = depth.at(x, y, 0);
      if (d < std::numeric_limits<float>::min()) {
        continue;
      }
      Eigen::Vector3f image_p(static_cast<float>(x), static_cast<float>(y), d);
      Eigen::Vector3f camera_p;
      camera.Unproject(image_p, &camera_p);
      vertices.push_back(camera_p);
    }
  }

  point_cloud->set_vertices(vertices);
}

void Depth2Mesh(const Image1f& depth, const Camera& camera, Mesh* mesh,
                float max_connect_z_diff, int x_step, int y_step) {
  if (max_connect_z_diff < 0) {
    LOGE("Depth2Mesh max_connect_z_diff must be positive %f\n",
         max_connect_z_diff);
    return;
  }
  if (x_step < 1) {
    LOGE("Depth2Mesh x_step must be positive %d\n", x_step);
    return;
  }
  if (y_step < 1) {
    LOGE("Depth2Mesh y_step must be positive %d\n", y_step);
    return;
  }

  mesh->Clear();

  std::vector<Eigen::Vector3f> vertices;
  std::vector<Eigen::Vector3i> vertex_indices;

  std::vector<int> added_table(depth.data().size(), -1);
  int vertex_id{0};
  for (int y = y_step; y < camera.height(); y += y_step) {
    for (int x = x_step; x < camera.width(); x += x_step) {
      const float& d = depth.at(x, y, 0);
      if (d < std::numeric_limits<float>::min()) {
        continue;
      }

      Eigen::Vector3f image_p(static_cast<float>(x), static_cast<float>(y), d);
      Eigen::Vector3f camera_p;
      camera.Unproject(image_p, &camera_p);

      vertices.push_back(camera_p);

      added_table[y * camera.width() + x] = vertex_id;

      const int& current_index = vertex_id;
      const int& upper_left_index =
          added_table[(y - y_step) * camera.width() + (x - x_step)];
      const int& upper_index = added_table[(y - y_step) * camera.width() + x];
      const int& left_index = added_table[y * camera.width() + (x - x_step)];

      const float upper_left_diff =
          std::abs(depth.at(x - x_step, y - y_step, 0) - d);
      const float upper_diff = std::abs(depth.at(x, y - y_step, 0) - d);
      const float left_diff = std::abs(depth.at(x - x_step, y, 0) - d);

      if (upper_left_index > 0 && upper_index > 0 &&
          upper_left_diff < max_connect_z_diff &&
          upper_diff < max_connect_z_diff) {
        vertex_indices.push_back(
            Eigen::Vector3i(upper_left_index, current_index, upper_index));
      }

      if (upper_left_index > 0 && left_index > 0 &&
          upper_left_diff < max_connect_z_diff &&
          left_diff < max_connect_z_diff) {
        vertex_indices.push_back(
            Eigen::Vector3i(upper_left_index, left_index, current_index));
      }

      vertex_id++;
    }
  }

  mesh->set_vertices(vertices);
  mesh->set_vertex_indices(vertex_indices);
}

}  // namespace currender
