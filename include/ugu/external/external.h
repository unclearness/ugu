/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 *
 */

#pragma once

#include "ugu/mesh.h"

namespace ugu {

bool FastQuadricMeshSimplification(const Mesh& src, int target_face_num,
                                   Mesh* decimated);

}  // namespace ugu
