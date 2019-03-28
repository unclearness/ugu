/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "currender/pixel_shader.h"

#include <algorithm>
#include <utility>

namespace currender {

OrenNayarParam::OrenNayarParam() {}
OrenNayarParam::OrenNayarParam(float sigma) : sigma(sigma) {
  assert(0 <= sigma);
  float sigma_2 = sigma * sigma;
  A = 1.0f - (0.5f * sigma_2 / (sigma_2 + 0.33f));
  B = 0.45f * sigma_2 / (sigma_2 + 0.09f);
}

PixelShaderInput::PixelShaderInput() {}
PixelShaderInput::~PixelShaderInput() {}
PixelShaderInput::PixelShaderInput(Image3b* color, int x, int y, float u,
                                   float v, uint32_t face_index,
                                   const Eigen::Vector3f* ray_w,
                                   const Eigen::Vector3f* light_dir,
                                   const Eigen::Vector3f* shading_normal,
                                   const OrenNayarParam* oren_nayar_param,
                                   std::shared_ptr<const Mesh> mesh)
    : color(color),
      x(x),
      y(y),
      u(u),
      v(v),
      face_index(face_index),
      ray_w(ray_w),
      light_dir(light_dir),
      shading_normal(shading_normal),
      oren_nayar_param(oren_nayar_param),
      mesh(mesh) {}

PixelShaderFactory::PixelShaderFactory() {}

PixelShaderFactory::~PixelShaderFactory() {}

std::unique_ptr<PixelShader> PixelShaderFactory::Create(
    DiffuseColor diffuse_color, ColorInterpolation interp,
    DiffuseShading diffuse_shading) {
  std::unique_ptr<DiffuseColorizer> colorizer;
  std::unique_ptr<DiffuseShader> shader;

  if (diffuse_color == DiffuseColor::kVertex) {
    colorizer.reset(new DiffuseVertexColorColorizer);
  } else if (diffuse_color == DiffuseColor::kTexture) {
    if (interp == ColorInterpolation::kNn) {
      colorizer.reset(new DiffuseTextureNnColorizer);
    } else if (interp == ColorInterpolation::kBilinear) {
      colorizer.reset(new DiffuseTextureBilinearColorizer);
    }
  } else if (diffuse_color == DiffuseColor::kNone) {
    colorizer.reset(new DiffuseDefaultColorizer);
  }
  assert(colorizer);

  if (diffuse_shading == DiffuseShading::kNone) {
    shader.reset(new DiffuseDefaultShader);
  } else if (diffuse_shading == DiffuseShading::kLambertian) {
    shader.reset(new DiffuseLambertianShader);
  } else if (diffuse_shading == DiffuseShading::kOrenNayar) {
    shader.reset(new DiffuseOrenNayarShader);
  }
  assert(shader);

  return std::unique_ptr<PixelShader>(
      new PixelShader(std::move(colorizer), std::move(shader)));
}

PixelShader::PixelShader() {}
PixelShader::~PixelShader() {}

PixelShader::PixelShader(std::unique_ptr<DiffuseColorizer>&& diffuse_colorizer,
                         std::unique_ptr<DiffuseShader>&& diffuse_shader) {
  diffuse_colorizer_ = std::move(diffuse_colorizer);
  diffuse_shader_ = std::move(diffuse_shader);
}

void PixelShader::Process(const PixelShaderInput& input) const {
  diffuse_colorizer_->Process(input);
  diffuse_shader_->Process(input);
}

DiffuseColorizer::DiffuseColorizer() {}
DiffuseColorizer::~DiffuseColorizer() {}

DiffuseDefaultColorizer::DiffuseDefaultColorizer() {}
DiffuseDefaultColorizer::~DiffuseDefaultColorizer() {}
void DiffuseDefaultColorizer::Process(const PixelShaderInput& input) const {
  Image3b* color = input.color;
  int x = input.x;
  int y = input.y;

  color->at(x, y, 0) = 255;
  color->at(x, y, 1) = 255;
  color->at(x, y, 2) = 255;
}

DiffuseVertexColorColorizer::DiffuseVertexColorColorizer() {}
DiffuseVertexColorColorizer::~DiffuseVertexColorColorizer() {}
void DiffuseVertexColorColorizer::Process(const PixelShaderInput& input) const {
  Image3b* color = input.color;
  int x = input.x;
  int y = input.y;
  float u = input.u;
  float v = input.v;
  uint32_t face_index = input.face_index;
  std::shared_ptr<const Mesh> mesh = input.mesh;

  const auto& vertex_colors = mesh->vertex_colors();
  const auto& faces = mesh->vertex_indices();
  Eigen::Vector3f interp_color;
  // barycentric interpolation of vertex color
  interp_color = (1.0f - u - v) * vertex_colors[faces[face_index][0]] +
                 u * vertex_colors[faces[face_index][1]] +
                 v * vertex_colors[faces[face_index][2]];

  for (int k = 0; k < 3; k++) {
    color->at(x, y, k) = static_cast<unsigned char>(interp_color[k]);
  }
}

DiffuseTextureNnColorizer::DiffuseTextureNnColorizer() {}
DiffuseTextureNnColorizer::~DiffuseTextureNnColorizer() {}
void DiffuseTextureNnColorizer::Process(const PixelShaderInput& input) const {
  Image3b* color = input.color;
  int x = input.x;
  int y = input.y;
  float u = input.u;
  float v = input.v;
  uint32_t face_index = input.face_index;
  std::shared_ptr<const Mesh> mesh = input.mesh;

  const auto& uv = mesh->uv();
  const auto& uv_indices = mesh->uv_indices();
  const auto& diffuse_texture = mesh->diffuse_tex();

  Eigen::Vector3f interp_color;
  // barycentric interpolation of uv
  Eigen::Vector2f interp_uv = (1.0f - u - v) * uv[uv_indices[face_index][0]] +
                              u * uv[uv_indices[face_index][1]] +
                              v * uv[uv_indices[face_index][2]];
  float f_tex_pos[2];
  f_tex_pos[0] = interp_uv[0] * (diffuse_texture.width() - 1);
  f_tex_pos[1] = (1.0f - interp_uv[1]) * (diffuse_texture.height() - 1);

  int tex_pos[2] = {0, 0};
  // get nearest integer index by round
  tex_pos[0] = static_cast<int>(std::round(f_tex_pos[0]));
  tex_pos[1] = static_cast<int>(std::round(f_tex_pos[1]));
  for (int k = 0; k < 3; k++) {
    interp_color[k] = diffuse_texture.at(tex_pos[0], tex_pos[1], k);
  }

  for (int k = 0; k < 3; k++) {
    color->at(x, y, k) = static_cast<unsigned char>(interp_color[k]);
  }
}

DiffuseTextureBilinearColorizer::DiffuseTextureBilinearColorizer() {}
DiffuseTextureBilinearColorizer::~DiffuseTextureBilinearColorizer() {}
void DiffuseTextureBilinearColorizer::Process(
    const PixelShaderInput& input) const {
  Image3b* color = input.color;
  int x = input.x;
  int y = input.y;
  float u = input.u;
  float v = input.v;
  uint32_t face_index = input.face_index;
  std::shared_ptr<const Mesh> mesh = input.mesh;

  const auto& uv = mesh->uv();
  const auto& uv_indices = mesh->uv_indices();
  const auto& diffuse_texture = mesh->diffuse_tex();

  Eigen::Vector3f interp_color;

  // barycentric interpolation of uv
  Eigen::Vector2f interp_uv = (1.0f - u - v) * uv[uv_indices[face_index][0]] +
                              u * uv[uv_indices[face_index][1]] +
                              v * uv[uv_indices[face_index][2]];
  float f_tex_pos[2];
  f_tex_pos[0] = interp_uv[0] * (diffuse_texture.width() - 1);
  f_tex_pos[1] = (1.0f - interp_uv[1]) * (diffuse_texture.height() - 1);

  int tex_pos_min[2] = {0, 0};
  int tex_pos_max[2] = {0, 0};
  tex_pos_min[0] = static_cast<int>(std::floor(f_tex_pos[0]));
  tex_pos_min[1] = static_cast<int>(std::floor(f_tex_pos[1]));
  tex_pos_max[0] = tex_pos_min[0] + 1;
  tex_pos_max[1] = tex_pos_min[1] + 1;

  float local_u = f_tex_pos[0] - tex_pos_min[0];
  float local_v = f_tex_pos[1] - tex_pos_min[1];

  for (int k = 0; k < 3; k++) {
    // bilinear interpolation of pixel color
    interp_color[k] =
        (1.0f - local_u) * (1.0f - local_v) *
            diffuse_texture.at(tex_pos_min[0], tex_pos_min[1], k) +
        local_u * (1.0f - local_v) *
            diffuse_texture.at(tex_pos_max[0], tex_pos_min[1], k) +
        (1.0f - local_u) * local_v *
            diffuse_texture.at(tex_pos_min[0], tex_pos_max[1], k) +
        local_u * local_v *
            diffuse_texture.at(tex_pos_max[0], tex_pos_max[1], k);

    assert(0.0f <= interp_color[k] && interp_color[k] <= 255.0f);
  }

  for (int k = 0; k < 3; k++) {
    color->at(x, y, k) = static_cast<unsigned char>(interp_color[k]);
  }
}

DiffuseShader::DiffuseShader() {}
DiffuseShader::~DiffuseShader() {}

DiffuseDefaultShader::DiffuseDefaultShader() {}
DiffuseDefaultShader::~DiffuseDefaultShader() {}
void DiffuseDefaultShader::Process(const PixelShaderInput& input) const {
  // do nothing.
  (void)input;
}

DiffuseLambertianShader::DiffuseLambertianShader() {}
DiffuseLambertianShader::~DiffuseLambertianShader() {}
void DiffuseLambertianShader::Process(const PixelShaderInput& input) const {
  Image3b* color = input.color;
  int x = input.x;
  int y = input.y;

  // dot product of normal and inverse light direction
  float coeff = -input.light_dir->dot(*input.shading_normal);

  // if negative (may happen at back-face or occluding boundary), bound to 0
  if (coeff < 0.0f) {
    coeff = 0.0f;
  }

  for (int k = 0; k < 3; k++) {
    color->at(x, y, k) = static_cast<uint8_t>(coeff * color->at(x, y, k));
  }
}

DiffuseOrenNayarShader::DiffuseOrenNayarShader() {}
DiffuseOrenNayarShader::~DiffuseOrenNayarShader() {}
void DiffuseOrenNayarShader::Process(const PixelShaderInput& input) const {
  // angle against normal
  float dot_light = -input.light_dir->dot(*input.shading_normal);
  float theta_i = std::acos(dot_light);
  float dot_ray = -input.ray_w->dot(*input.shading_normal);
  float theta_r = std::acos(dot_ray);

  // angle against binormal (perpendicular to normal)
  Eigen::Vector3f binormal_light =
      -*input.shading_normal * dot_light - *input.light_dir;
  Eigen::Vector3f binormal_ray =
      -*input.shading_normal * dot_light - *input.ray_w;
  float phi_diff_cos = std::max(0.0f, binormal_light.dot(binormal_ray));

  float alpha = std::max(theta_i, theta_r);
  float beta = std::min(theta_i, theta_r);

  float A = input.oren_nayar_param->A;
  float B = input.oren_nayar_param->B;
  float coeff = std::max(0.0f, dot_light) *
                (A + (B * phi_diff_cos * std::sin(alpha) * std::tan(beta)));

  Image3b* color = input.color;
  int x = input.x;
  int y = input.y;
  for (int k = 0; k < 3; k++) {
    color->at(x, y, k) = static_cast<uint8_t>(coeff * color->at(x, y, k));
  }
}

}  // namespace currender