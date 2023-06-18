/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

// original code is from
// http://paulbourke.net/geometry/polygonise/

#pragma once

#include "ugu/mesh.h"
#include "ugu/voxel/voxel.h"

namespace ugu {

void MarchingCubes(const VoxelGrid& voxel_grid, Mesh* mesh,
                   double iso_level = 0.0, bool with_color = false);

}  // namespace ugu
