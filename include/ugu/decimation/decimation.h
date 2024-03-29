/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 *
 */

#pragma once

#include "ugu/mesh.h"

namespace ugu {

enum class QSlimType { XYZ, XYZ_UV };

bool RandomDecimation(MeshPtr mesh, QSlimType type, int32_t target_face_num,
                      int32_t target_vertex_num, bool keep_geom_boundary = true,
                      bool keep_uv_boundary = true);

// TODO: Only valid for keep_geom_boundary = true, keep_uv_boundary = true
bool QSlim(MeshPtr mesh, QSlimType type, int32_t target_face_num,
           int32_t target_vertex_num, bool keep_geom_boundary = true,
           bool keep_uv_boundary = true);

}  // namespace ugu