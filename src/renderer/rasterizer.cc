/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "ugu/renderer/rasterizer.h"

#include <cassert>

#include "ugu/image_proc.h"
#include "ugu/renderer/pixel_shader.h"
#include "ugu/renderer/util_private.h"
#include "ugu/timer.h"

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

namespace ugu {

// Rasterizer::Impl implementation
class Rasterizer::Impl {
  bool mesh_initialized_{false};
  std::shared_ptr<const Camera> camera_{nullptr};
  std::shared_ptr<const Mesh> mesh_{nullptr};
  RendererOption option_;

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

bool Rasterizer::Impl::Render(Image3b* color, Image1f* depth, Image3f* normal,
                              Image1b* mask, Image1i* face_id) const {
  if (!ValidateAndInitBeforeRender(mesh_initialized_, camera_, mesh_, option_,
                                   color, depth, normal, mask, face_id)) {
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

  // get projected vertex positions
  for (int i = 0; i < static_cast<int>(mesh_->vertices().size()); i++) {
    camera_vertices[i] = w2c_R * mesh_->vertices()[i] + w2c_t;
    camera_normals[i] = w2c_R * mesh_->normals()[i];
    camera_depth_list[i] = camera_vertices[i].z();
    camera_->Project(camera_vertices[i], &image_vertices[i]);
  }

  Image1f depth_internal;
  Image1f* depth_{depth};
  if (depth_ == nullptr) {
    depth_ = &depth_internal;
  }
  Init(depth_, camera_->width(), camera_->height(), 0.0f);

  Image1i face_id_internal;
  Image1i* face_id_{face_id};
  if (face_id_ == nullptr) {
    face_id_ = &face_id_internal;
  }
  Init(face_id_, camera_->width(), camera_->height(), -1);

  // 255: backface, 0:frontface
  Image1b backface_image;
  Init(&backface_image, camera_->width(), camera_->height(),
       static_cast<unsigned char>(0));

  // 0:(1 - u - v), 1:u, 2:v
  Image3f weight_image;
  Init(&weight_image, camera_->width(), camera_->height(), 0.0f);

  // make face id image by z-buffer method
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

    uint32_t x0 = std::max(int32_t(0), (int32_t)(std::ceil(xmin)));
    uint32_t x1 = std::min(camera_->width() - 1, (int32_t)(std::floor(xmax)));
    uint32_t y0 = std::max(int32_t(0), (int32_t)(std::ceil(ymin)));
    uint32_t y1 = std::min(camera_->height() - 1, (int32_t)(std::floor(ymax)));

    float area = EdgeFunction(v0_i, v1_i, v2_i);
    if (std::abs(area) < std::numeric_limits<float>::min()) {
      continue;
    }
    for (uint32_t y = y0; y <= y1; ++y) {
      for (uint32_t x = x0; x <= x1; ++x) {
        Eigen::Vector3f ray_w;
        camera_->ray_w(static_cast<int>(x), static_cast<int>(y), &ray_w);
        // even if back-face culling is enabled, dont' skip back-face
        // need to update z-buffer to handle front-face occluded by back-face
        bool backface = mesh_->face_normals()[i].dot(ray_w) > 0;
        Eigen::Vector3f pixel_sample(static_cast<float>(x),
                                     static_cast<float>(y), 0.0f);
        float w0 = EdgeFunction(v1_i, v2_i, pixel_sample);
        float w1 = EdgeFunction(v2_i, v0_i, pixel_sample);
        float w2 = EdgeFunction(v0_i, v1_i, pixel_sample);
        if ((!backface && (w0 >= 0 && w1 >= 0 && w2 >= 0)) ||
            (backface && (w0 <= 0 && w1 <= 0 && w2 <= 0))) {
          w0 /= area;
          w1 /= area;
          w2 /= area;
#if 0
         // original
          pixel_sample.z() = w0 * v0_i.z() + w1 * v1_i.z() + w2 * v2_i.z();
#else
          /** Perspective-Correct Interpolation **/
          w0 /= v0_i.z();
          w1 /= v1_i.z();
          w2 /= v2_i.z();

          pixel_sample.z() = 1.0f / (w0 + w1 + w2);

          w0 = w0 * pixel_sample.z();
          w1 = w1 * pixel_sample.z();
          w2 = w2 * pixel_sample.z();
          /** Perspective-Correct Interpolation **/
#endif

          float& d = depth_->at<float>(y, x);
          if (d < std::numeric_limits<float>::min() || pixel_sample.z() < d) {
            d = pixel_sample.z();
            face_id_->at<int>(y, x) = i;
            Vec3f& weight = weight_image.at<Vec3f>(y, x);
            weight[0] = w0;
            weight[1] = w1;
            weight[2] = w2;
            backface_image.at<unsigned char>(y, x) = backface ? 255 : 0;
          }
        }
      }
    }
  }

  // make images by referring to face id image
  for (int y = 0; y < backface_image.rows; y++) {
    for (int x = 0; x < backface_image.cols; x++) {
      const unsigned char& bf = backface_image.at<unsigned char>(y, x);
      int& fid = face_id_->at<int>(y, x);
      if (option_.backface_culling && bf == 255) {
        depth_->at<float>(y, x) = 0.0f;
        fid = -1;
        continue;
      }

      if (fid > 0) {
        Eigen::Vector3f ray_w;
        camera_->ray_w(x, y, &ray_w);

        Vec3f& weight = weight_image.at<Vec3f>(y, x);
        float w0 = weight[0];
        float w1 = weight[1];
        float w2 = weight[2];

        // fill mask
        if (mask != nullptr) {
          mask->at<unsigned char>(y, x) = 255;
        }

        // calculate shading normal
        Eigen::Vector3f shading_normal_w{0.0f, 0.0f, 0.0f};
        if (option_.shading_normal == ShadingNormal::kFace) {
          shading_normal_w = mesh_->face_normals()[fid];
        } else if (option_.shading_normal == ShadingNormal::kVertex) {
          // barycentric interpolation of normal
          const auto& normals = mesh_->normals();
          const auto& normal_indices = mesh_->normal_indices();
          shading_normal_w = w0 * normals[normal_indices[fid][0]] +
                             w1 * normals[normal_indices[fid][1]] +
                             w2 * normals[normal_indices[fid][2]];
        }

        // set shading normal
        if (normal != nullptr) {
          Eigen::Vector3f shading_normal_c =
              w2c_R * shading_normal_w;  // rotate to camera coordinate
          Vec3f& n = normal->at<Vec3f>(y, x);
          for (int k = 0; k < 3; k++) {
            n[k] = shading_normal_c[k];
          }
        }

        // delegate color calculation to pixel_shader
        if (color != nullptr) {
          Eigen::Vector3f light_dir = ray_w;  // emit light same as ray
          PixelShaderInput pixel_shader_input(color, x, y, w1, w2, fid, &ray_w,
                                              &light_dir, &shading_normal_w,
                                              &oren_nayar_param, mesh_);
          pixel_shader->Process(pixel_shader_input);
        }
      }
    }
  }

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
    ConvertTo(f_depth, depth);
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

}  // namespace ugu
