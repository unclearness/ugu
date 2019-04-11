/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include "currender/camera.h"
#include "currender/common.h"
#include "currender/image.h"
#include "currender/mesh.h"

namespace currender {

void Depth2PointCloud(const Image1f& depth, const Camera& camera,
                      Image3f* point_cloud);

void Depth2PointCloud(const Image1f& depth, const Camera& camera,
                      Mesh* point_cloud);

void Depth2Mesh(const Image1f& depth, const Camera& camera, Mesh* mesh,
                float max_connect_z_diff, int x_step = 1, int y_step = 1);

}  // namespace currender
