/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

// original code is from
// http://paulbourke.net/geometry/polygonise/

#pragma once

#include "ugu/sfs/voxel_carver.h"

namespace ugu {

void MarchingCubes(const VoxelGrid& voxel_grid, Mesh* mesh,
                   double iso_level = 0.0);

}  // namespace ugu
