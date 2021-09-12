/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 *
 */

#pragma once

#include "ugu/mesh.h"

namespace ugu {

enum class QSlimType { XYZ, XYZ_UV };

bool QSlim(MeshPtr mesh, QSlimType type, int32_t target_face_num,
           int32_t target_vertex_num, bool keep_geom_boundary,
           bool keep_uv_boundary, bool accept_non_edge, float non_edge_dist);

}  // namespace ugu