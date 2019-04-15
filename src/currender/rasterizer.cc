/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "currender/rasterizer.h"

#include <cassert>

#include "currender/timer.h"

namespace {}  // namespace

namespace currender {

// Rasterizer::Impl implementation
class Rasterizer::Impl {
  bool mesh_initialized_{false};
  std::shared_ptr<const Camera> camera_{nullptr};
  std::shared_ptr<const Mesh> mesh_{nullptr};
  RendererOption option_;

  bool ValidateAndInitBeforeRender(Image3b* color, Image1f* depth,
                                   Image3f* normal, Image1b* mask,
                                   Image1i* face_id) const;

 public:
  Impl();
  ~Impl();

  explicit Impl(const RendererOption& option);
  void set_option(const RendererOption& option);

  void set_mesh(std::shared_ptr<const Mesh> mesh);

  bool PrepareMesh();

  void set_camera(std::shared_ptr<const Camera> camera);

  bool Render(Image3b* color, Image1f* depth, Image3f* normal, Image1b* mask,
              Image1i* face_id) const;

  bool RenderColor(Image3b* color) const;
  bool RenderDepth(Image1f* depth) const;
  bool RenderNormal(Image3f* normal) const;
  bool RenderMask(Image1b* mask) const;
  bool RenderFaceId(Image1i* face_id) const;

  bool RenderW(Image3b* color, Image1w* depth, Image3f* normal, Image1b* mask,
               Image1i* face_id) const;
  bool RenderDepthW(Image1w* depth) const;
};

Rasterizer::Impl::Impl() {}
Rasterizer::Impl::~Impl() {}

Rasterizer::Impl::Impl(const RendererOption& option) { set_option(option); }

void Rasterizer::Impl::set_option(const RendererOption& option) {
  option.CopyTo(&option_);
}

void Rasterizer::Impl::set_mesh(std::shared_ptr<const Mesh> mesh) {
  mesh_initialized_ = false;
  mesh_ = mesh;

  if (mesh_->face_normals().empty()) {
    LOGW("face normal is empty. culling and shading may not work\n");
  }

  if (mesh_->normals().empty()) {
    LOGW("vertex normal is empty. shading may not work\n");
  }
}

bool Rasterizer::Impl::PrepareMesh() {
  if (mesh_ == nullptr) {
    LOGE("mesh has not been set\n");
    return false;
  }

  mesh_initialized_ = true;

  return true;
}

void Rasterizer::Impl::set_camera(std::shared_ptr<const Camera> camera) {
  camera_ = camera;
}

bool Rasterizer::Impl::ValidateAndInitBeforeRender(Image3b* color,
                                                   Image1f* depth,
                                                   Image3f* normal,
                                                   Image1b* mask,
                                                   Image1i* face_id) const {
  if (camera_ == nullptr) {
    LOGE("camera has not been set\n");
    return false;
  }
  if (!mesh_initialized_) {
    LOGE("mesh has not been initialized\n");
    return false;
  }
  if (option_.backface_culling && mesh_->face_normals().empty()) {
    LOGE("specified back-face culling but face normal is empty.\n");
    return false;
  }
  if (option_.diffuse_color == DiffuseColor::kTexture &&
      mesh_->diffuse_texs().empty()) {
    LOGE("specified texture as diffuse color but texture is empty.\n");
    return false;
  }
  if (option_.diffuse_color == DiffuseColor::kTexture) {
    for (int i = 0; i < static_cast<int>(mesh_->diffuse_texs().size()); i++) {
      if (mesh_->diffuse_texs()[i].empty()) {
        LOGE("specified texture as diffuse color but %d th texture is empty.\n",
             i);
        return false;
      }
    }
  }
  if (option_.diffuse_color == DiffuseColor::kVertex &&
      mesh_->vertex_colors().empty()) {
    LOGE(
        "specified vertex color as diffuse color but vertex color is empty.\n");
    return false;
  }
  if (option_.shading_normal == ShadingNormal::kFace &&
      mesh_->face_normals().empty()) {
    LOGE("specified face normal as shading normal but face normal is empty.\n");
    return false;
  }
  if (option_.shading_normal == ShadingNormal::kVertex &&
      mesh_->normals().empty()) {
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

  int width = camera_->width();
  int height = camera_->height();

  if (color != nullptr) {
    color->Init(width, height);
  }
  if (depth != nullptr) {
    depth->Init(width, height);
  }
  if (normal != nullptr) {
    normal->Init(width, height);
  }
  if (mask != nullptr) {
    mask->Init(width, height);
  }
  if (face_id != nullptr) {
    // initialize with -1 (no hit)
    face_id->Init(width, height, -1);
  }

  return true;
}

bool Rasterizer::Impl::Render(Image3b* color, Image1f* depth, Image3f* normal,
                              Image1b* mask, Image1i* face_id) const {
  if (!ValidateAndInitBeforeRender(color, depth, normal, mask, face_id)) {
    return false;
  }


  const Eigen::Matrix3f w2c_R = camera_->w2c().rotation().cast<float>();
  const Eigen::Vector3f w2c_t = camera_->w2c().translation().cast<float>();

  Timer<> timer;
  timer.Start();


  timer.End();
  LOGI("  Rendering main loop time: %.1f msecs\n", timer.elapsed_msec());

  return true;
}

bool Rasterizer::Impl::RenderColor(Image3b* color) const {
  return Render(color, nullptr, nullptr, nullptr, nullptr);
}

bool Rasterizer::Impl::RenderDepth(Image1f* depth) const {
  return Render(nullptr, depth, nullptr, nullptr, nullptr);
}

bool Rasterizer::Impl::RenderNormal(Image3f* normal) const {
  return Render(nullptr, nullptr, normal, nullptr, nullptr);
}

bool Rasterizer::Impl::RenderMask(Image1b* mask) const {
  return Render(nullptr, nullptr, nullptr, mask, nullptr);
}

bool Rasterizer::Impl::RenderFaceId(Image1i* face_id) const {
  return Render(nullptr, nullptr, nullptr, nullptr, face_id);
}

bool Rasterizer::Impl::RenderW(Image3b* color, Image1w* depth, Image3f* normal,
                               Image1b* mask, Image1i* face_id) const {
  if (depth == nullptr) {
    LOGE("depth is nullptr");
    return false;
  }

  Image1f f_depth;
  bool org_ret = Render(color, &f_depth, normal, mask, face_id);

  if (org_ret) {
    f_depth.ConvertTo(depth);
  }

  return org_ret;
}

bool Rasterizer::Impl::RenderDepthW(Image1w* depth) const {
  return RenderW(nullptr, depth, nullptr, nullptr, nullptr);
}

// Renderer implementation
Rasterizer::Rasterizer() : pimpl_(std::unique_ptr<Impl>(new Impl)) {}

Rasterizer::~Rasterizer() {}

Rasterizer::Rasterizer(const RendererOption& option)
    : pimpl_(std::unique_ptr<Impl>(new Impl(option))) {}

void Rasterizer::set_option(const RendererOption& option) {
  pimpl_->set_option(option);
}

void Rasterizer::set_mesh(std::shared_ptr<const Mesh> mesh) {
  pimpl_->set_mesh(mesh);
}

bool Rasterizer::PrepareMesh() { return pimpl_->PrepareMesh(); }

void Rasterizer::set_camera(std::shared_ptr<const Camera> camera) {
  pimpl_->set_camera(camera);
}

bool Rasterizer::Render(Image3b* color, Image1f* depth, Image3f* normal,
                        Image1b* mask, Image1i* face_id) const {
  return pimpl_->Render(color, depth, normal, mask, face_id);
}

bool Rasterizer::RenderColor(Image3b* color) const {
  return pimpl_->RenderColor(color);
}

bool Rasterizer::RenderDepth(Image1f* depth) const {
  return pimpl_->RenderDepth(depth);
}

bool Rasterizer::RenderNormal(Image3f* normal) const {
  return pimpl_->RenderNormal(normal);
}

bool Rasterizer::RenderMask(Image1b* mask) const {
  return pimpl_->RenderMask(mask);
}

bool Rasterizer::RenderFaceId(Image1i* face_id) const {
  return pimpl_->RenderFaceId(face_id);
}

bool Rasterizer::RenderW(Image3b* color, Image1w* depth, Image3f* normal,
                         Image1b* mask, Image1i* face_id) const {
  return pimpl_->RenderW(color, depth, normal, mask, face_id);
}

bool Rasterizer::RenderDepthW(Image1w* depth) const {
  return pimpl_->RenderDepthW(depth);
}

}  // namespace currender
