/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 *
 */

#pragma once

#include "ugu/mesh.h"
#include "ugu/texturing/texture_mapper.h"

namespace ugu {

bool FastQuadricMeshSimplification(const Mesh& src, int target_face_num,
                                   Mesh* decimated);

bool MvsTexturing(const std::vector<std::shared_ptr<ugu::Keyframe>>& keyframes,
                  ugu::Mesh* mesh, ugu::Mesh* debug_mesh = nullptr);

}  // namespace ugu
