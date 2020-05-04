/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <string>

#include "ugu/camera.h"
#include "ugu/common.h"
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
                bool gl_coord = false);

bool Depth2Mesh(const Image1f& depth, const Image3b& color,
                const Camera& camera, Mesh* mesh, float max_connect_z_diff,
                int x_step = 1, int y_step = 1, bool gl_coord = false,
                const std::string& material_name = "Depth2Mesh_mat");

void WriteFaceIdAsText(const Image1i& face_id, const std::string& path);

}  // namespace ugu
