/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "ugu/renderer/util_private.h"

#include <fstream>

namespace ugu {

bool ValidateAndInitBeforeRender(bool mesh_initialized,
                                 std::shared_ptr<const Camera> camera,
                                 std::shared_ptr<const Mesh> mesh,
                                 const RendererOption& option, Image3b* color,
                                 Image1f* depth, Image3f* normal, Image1b* mask,
                                 Image1i* face_id) {
  if (camera == nullptr) {
    LOGE("camera has not been set\n");
    return false;
  }
  if (!mesh_initialized) {
    LOGE("mesh has not been initialized\n");
    return false;
  }
  if (option.backface_culling && mesh->face_normals().empty()) {
    LOGE("specified back-face culling but face normal is empty.\n");
    return false;
  }
  if (option.diffuse_color == DiffuseColor::kTexture &&
      mesh->materials().empty()) {
    LOGE("specified texture as diffuse color but texture is empty.\n");
    return false;
  }
  if (option.diffuse_color == DiffuseColor::kTexture) {
    for (int i = 0; i < static_cast<int>(mesh->materials().size()); i++) {
      if (mesh->materials()[i].diffuse_tex.empty()) {
        LOGE("specified texture as diffuse color but %d th texture is empty.\n",
             i);
        return false;
      }
    }
  }
  if (option.diffuse_color == DiffuseColor::kVertex &&
      mesh->vertex_colors().empty()) {
    LOGE(
        "specified vertex color as diffuse color but vertex color is empty.\n");
    return false;
  }
  if (option.shading_normal == ShadingNormal::kFace &&
      mesh->face_normals().empty()) {
    LOGE("specified face normal as shading normal but face normal is empty.\n");
    return false;
  }
  if (option.shading_normal == ShadingNormal::kVertex &&
      mesh->normals().empty()) {
    LOGE(
        "specified vertex normal as shading normal but vertex normal is "
        "empty.\n");
    return false;
  }
  if (color == nullptr && depth == nullptr && normal == nullptr &&
      mask == nullptr && face_id == nullptr) {
    LOGE("all arguments are nullptr. nothing to do\n");
    return false;
  }

  int width = camera->width();
  int height = camera->height();

  if (color != nullptr) {
    Init(color, width, height, unsigned char(0));
  }
  if (depth != nullptr) {
    Init(depth, width, height, 0.0f);
  }
  if (normal != nullptr) {
    Init(normal, width, height, 0.0f);
  }
  if (mask != nullptr) {
    Init(mask, width, height, unsigned char(0));
  }
  if (face_id != nullptr) {
    // initialize with -1 (no hit)
    Init(face_id, width, height, -1);
  }

  return true;
}

}  // namespace ugu
