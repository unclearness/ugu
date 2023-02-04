/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <fstream>
#include <memory>

#include "ugu/renderer/cpu/renderer.h"

namespace ugu {

bool ValidateAndInitBeforeRender(bool mesh_initialized,
                                 std::shared_ptr<const Camera> camera,
                                 std::shared_ptr<const Mesh> mesh,
                                 const RendererCpuOption& option,
                                 Image3b* color, Image1f* depth,
                                 Image3f* normal, Image1b* mask,
                                 Image1i* face_id);

}  // namespace ugu
