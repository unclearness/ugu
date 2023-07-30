/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <fstream>
#include <string>

#include "ugu/camera.h"
#include "ugu/image.h"
#include "ugu/mesh.h"

namespace ugu {

bool Depth2PointCloud(const Image1f& depth, const Camera& camera,
                      Image3f* point_cloud, bool gl_coord = false);

bool Depth2PointCloud(const Image1f& depth, const Camera& camera,
                      Mesh* point_cloud, bool gl_coord = false);

bool Depth2PointCloud(const Image1f& depth, const Image3b& color,
                      const Camera& camera, Mesh* point_cloud,
                      bool gl_coord = false);

bool Depth2Mesh(const Image1f& depth, const Camera& camera, Mesh* mesh,
                float max_connect_z_diff, int x_step = 1, int y_step = 1,
                bool gl_coord = false, Image3f* point_cloud = nullptr,
                Image3f* normal = nullptr);

bool Depth2Mesh(const Image1f& depth, const Image3b& color,
                const Camera& camera, Mesh* mesh, float max_connect_z_diff,
                int x_step = 1, int y_step = 1, bool gl_coord = false,
                const std::string& material_name = "Depth2Mesh_mat",
                bool with_vertex_color = false, Image3f* point_cloud = nullptr,
                Image3f* normal = nullptr);

bool Depth2Mesh(const Image1f& depth, const Image3b& color,
                const Camera& depth_camera, const Camera& color_camera,
                const Eigen::Affine3f depth2color, Mesh* mesh,
                float max_connect_z_diff, int x_step = 1, int y_step = 1,
                bool gl_coord = false,
                const std::string& material_name = "Depth2Mesh_mat",
                bool with_vertex_color = false, Image3f* point_cloud = nullptr,
                Image3f* normal = nullptr);

}  // namespace ugu
