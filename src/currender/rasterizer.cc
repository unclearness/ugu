/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "currender/rasterizer.h"

#include <cassert>

#include "currender/pixel_shader.h"
#include "currender/timer.h"

namespace {

template <typename T>
void Argsort(const std::vector<T>& data, std::vector<size_t>* indices) {
  indices->resize(data.size());
  std::iota(indices->begin(), indices->end(), 0);
  std::sort(indices->begin(), indices->end(),
            [&data](size_t i1, size_t i2) { return data[i1] < data[i2]; });
}

inline float EdgeFunction(const Eigen::Vector3f& a, const Eigen::Vector3f& b,
                          const Eigen::Vector3f& c) {
  return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0]);
}

}  // namespace

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

  // make pixel shader
  std::unique_ptr<PixelShader> pixel_shader = PixelShaderFactory::Create(
      option_.diffuse_color, option_.interp, option_.diffuse_shading);

  OrenNayarParam oren_nayar_param(option_.oren_nayar_sigma);

  const Eigen::Matrix3f w2c_R = camera_->w2c().rotation().cast<float>();
  const Eigen::Vector3f w2c_t = camera_->w2c().translation().cast<float>();

  Timer<> timer;
  timer.Start();

  // project face to 2d (fully parallel)
  std::vector<Eigen::Vector3f> camera_vertices(mesh_->vertices().size());
  std::vector<Eigen::Vector3f> camera_normals(mesh_->vertices().size());
  std::vector<float> camera_depth_list(mesh_->vertices().size());
  std::vector<Eigen::Vector3f> image_vertices(mesh_->vertices().size());

#if defined(_OPENMP) && defined(CURRENDER_USE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int i = 0; i < static_cast<int>(mesh_->vertices().size()); i++) {
    camera_vertices[i] = w2c_R * mesh_->vertices()[i] + w2c_t;
    camera_normals[i] = w2c_R * mesh_->normals()[i];
    camera_depth_list[i] = camera_vertices[i].z();
    camera_->Project(camera_vertices[i], &image_vertices[i]);
  }

  // std::vector<size_t> indices;
  // Argsort(camera_depth_list, &indices);

  // grouping as tiles

  // z-buffer at each tile (parallel)
  // Bresenham's line algorithm
  Image1f depth_internal;
  Image1f* depth_{depth};
  if (depth_ == nullptr) {
    depth_ = &depth_internal;
  }
  depth_->Init(camera_->width(), camera_->height(), 0.0f);

  // 255: backface, 0:frontface
  Image1b backface_image(camera_->width(), camera_->height(), 0);

#if defined(_OPENMP) && defined(CURRENDER_USE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int i = 0; i < static_cast<int>(mesh_->vertex_indices().size()); i++) {
    const Eigen::Vector3i& face = mesh_->vertex_indices()[i];
    const Eigen::Vector3f& v0_i = image_vertices[face[0]];
    const Eigen::Vector3f& v1_i = image_vertices[face[1]];
    const Eigen::Vector3f& v2_i = image_vertices[face[2]];

    // skip if a vertex is back of the camera
    // todo: add near and far plane
    if (v0_i.z() < 0.0f || v1_i.z() < 0.0f || v2_i.z() < 0.0f) {
      continue;
    }

    float xmin = std::min({v0_i.x(), v1_i.x(), v2_i.x()});
    float ymin = std::min({v0_i.y(), v1_i.y(), v2_i.y()});
    float xmax = std::max({v0_i.x(), v1_i.x(), v2_i.x()});
    float ymax = std::max({v0_i.y(), v1_i.y(), v2_i.y()});

    // the triangle is out of screen
    if (xmin > camera_->width() - 1 || xmax < 0 ||
        ymin > camera_->height() - 1 || ymax < 0) {
      continue;
    }

    uint32_t x0 = std::max(int32_t(0), (int32_t)(std::floor(xmin)));
    uint32_t x1 = std::min(camera_->width() - 1, (int32_t)(std::floor(xmax)));
    uint32_t y0 = std::max(int32_t(0), (int32_t)(std::floor(ymin)));
    uint32_t y1 = std::min(camera_->height() - 1, (int32_t)(std::floor(ymax)));

    float area = EdgeFunction(v0_i, v1_i, v2_i);
    if (std::abs(area) < std::numeric_limits<float>::min()) {
      continue;
    }
    for (uint32_t y = y0; y <= y1; ++y) {
      for (uint32_t x = x0; x <= x1; ++x) {
        Eigen::Vector3f ray_w;
        camera_->ray_w(static_cast<float>(x), static_cast<float>(y), &ray_w);
        bool backface = mesh_->face_normals()[i].dot(ray_w) > 0;
        Eigen::Vector3f pixel_sample(x + 0.5f, y + 0.5f, 0.0f);
        float w0 = EdgeFunction(v1_i, v2_i, pixel_sample);
        float w1 = EdgeFunction(v2_i, v0_i, pixel_sample);
        float w2 = EdgeFunction(v0_i, v1_i, pixel_sample);
        if ((!backface && (w0 >= 0 && w1 >= 0 && w2 >= 0)) ||
            (backface && (w0 <= 0 && w1 <= 0 && w2 <= 0))) {
          w0 /= area;
          w1 /= area;
          w2 /= area;
          pixel_sample.z() = v0_i.z() * w0 + v1_i.z() * w1 + v2_i.z() * w2;

#pragma omp critical
          if (depth_->at(x, y, 0) < std::numeric_limits<float>::min() ||
              pixel_sample.z() < depth_->at(x, y, 0)) {
            depth_->at(x, y, 0) = pixel_sample.z();
            backface_image.at(x, y, 0) = backface ? 255 : 0;

            if (!backface) {
              // fill face id
              if (face_id != nullptr) {
                face_id->at(x, y, 0) = i;
              }

              // fill mask
              if (mask != nullptr) {
                mask->at(x, y, 0) = 255;
              }

              // calculate shading normal
              Eigen::Vector3f shading_normal_w{0.0f, 0.0f, 0.0f};
              if (option_.shading_normal == ShadingNormal::kFace) {
                shading_normal_w = mesh_->face_normals()[i];
              } else if (option_.shading_normal == ShadingNormal::kVertex) {
                // barycentric interpolation of normal
                const auto& normals = mesh_->normals();
                const auto& normal_indices = mesh_->normal_indices();
                shading_normal_w = w0 * normals[normal_indices[i][0]] +
                                   w1 * normals[normal_indices[i][1]] +
                                   w2 * normals[normal_indices[i][2]];
              }

              // set shading normal
              if (normal != nullptr) {
                Eigen::Vector3f shading_normal_c =
                    w2c_R * shading_normal_w;  // rotate to camera coordinate
                for (int k = 0; k < 3; k++) {
                  normal->at(x, y, k) = shading_normal_c[k];
                }
              }

              // delegate color calculation to pixel_shader
              if (color != nullptr) {
                Eigen::Vector3f light_dir = ray_w;  // emit light same as ray
                PixelShaderInput pixel_shader_input(
                    color, x, y, w1, w2, i, &ray_w, &light_dir,
                    &shading_normal_w, &oren_nayar_param, mesh_);
                pixel_shader->Process(pixel_shader_input);
              }
            }
          }
        }
      }
    }
  }

#if defined(_OPENMP) && defined(CURRENDER_USE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int y = 0; y < backface_image.height(); y++) {
    for (int x = 0; x < backface_image.width(); x++) {
      if (backface_image.at(x, y, 0) == 255) {
        depth_->at(x, y, 0) = 0.0f;

        if (face_id != nullptr) {
          face_id->at(x, y, 0) = -1;
        }
        if (mask != nullptr) {
          mask->at(x, y, 0) = 255;
        }

        if (normal != nullptr) {
          for (int k = 0; k < 3; k++) {
            normal->at(x, y, k) = 0.0f;
          }
        }
        if (color != nullptr) {
          for (int k = 0; k < 3; k++) {
            color->at(x, y, k) = 0;
          }
        }
      }
    }
  }

  // process faces at boundary of tiles (partially parallel)

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
