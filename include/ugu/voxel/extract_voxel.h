/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/mesh.h"
#include "ugu/voxel/voxel.h"

namespace ugu {

void ExtractVoxel(VoxelGrid* voxel_grid, Mesh* mesh, bool inside_empty);

}  // namespace ugu
