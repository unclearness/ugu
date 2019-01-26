/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "src/pixel_shader.h"

#include <utility>

namespace currender {

PixelShader::PixelShader() {}
PixelShader::~PixelShader() {}

PixelShaderFactory::PixelShaderFactory() {}

PixelShaderFactory::~PixelShaderFactory() {}

std::unique_ptr<PixelShader> PixelShaderFactory::Create(
    DiffuseColor diffuse_color, ColorInterpolation interp,
    DiffuseShading diffuse_shading) {
  std::unique_ptr<DiffuseColorizer> colorizer{nullptr};
  std::unique_ptr<DiffuseShader> shader{nullptr};

  if (diffuse_color == DiffuseColor::kVertex) {
    colorizer =
        std::unique_ptr<DiffuseColorizer>(new DiffuseVertexColorColorizer);

  } else if (diffuse_color == DiffuseColor::kTexture) {
    if (interp == ColorInterpolation::kNn) {
      colorizer =
          std::unique_ptr<DiffuseColorizer>(new DiffuseTextureNnColorizer);

    } else if (interp == ColorInterpolation::kBilinear) {
      colorizer = std::unique_ptr<DiffuseColorizer>(
          new DiffuseTextureBilinearColorizer);
    }
  } else if (diffuse_color == DiffuseColor::kNone) {
    colorizer = std::unique_ptr<DiffuseColorizer>(new DiffuseDefaultColorizer);
  }
  assert(colorizer != nullptr);

  if (diffuse_shading == DiffuseShading::kNone) {
    shader = std::unique_ptr<DiffuseShader>(new DiffuseDefaultShader);
  } else if (diffuse_shading == DiffuseShading::kLambertian) {
    shader = std::unique_ptr<DiffuseShader>(new DiffuseLambertianShader);
  } else if (diffuse_shading == DiffuseShading::kOrenNayar) {
    shader = std::unique_ptr<DiffuseShader>(new DiffuseOrenNayarShader);
  }
  assert(shader != nullptr);

  return std::unique_ptr<PixelShader>(
      new PixelShader(std::move(colorizer), std::move(shader)));
}

PixelShader::PixelShader(std::unique_ptr<DiffuseColorizer>&& diffuse_colorizer,
                         std::unique_ptr<DiffuseShader>&& diffuse_shader) {
  diffuse_colorizer_ = std::move(diffuse_colorizer);
  diffuse_shader_ = std::move(diffuse_shader);
}

void PixelShader::Process(Image3b* color, int x, int y, float u, float v,
                          uint32_t face_index, const Mesh& mesh) const {
  diffuse_colorizer_->Process(color, x, y, u, v, face_index, mesh);
  diffuse_shader_->Process(color, x, y, u, v, face_index, mesh);
}

DiffuseColorizer::DiffuseColorizer() {}
DiffuseColorizer::~DiffuseColorizer() {}

DiffuseDefaultColorizer::DiffuseDefaultColorizer() {}
DiffuseDefaultColorizer::~DiffuseDefaultColorizer() {}
void DiffuseDefaultColorizer::Process(Image3b* color, int x, int y, float u,
                                      float v, uint32_t face_index,
                                      const Mesh& mesh) const {
  (void)u;
  (void)v;
  (void)face_index;
  (void)mesh;
  color->at(x, y, 0) = 255;
  color->at(x, y, 1) = 255;
  color->at(x, y, 2) = 255;
}

DiffuseVertexColorColorizer::DiffuseVertexColorColorizer() {}
DiffuseVertexColorColorizer::~DiffuseVertexColorColorizer() {}
void DiffuseVertexColorColorizer::Process(Image3b* color, int x, int y, float u,
                                          float v, uint32_t face_index,
                                          const Mesh& mesh) const {
  const auto& vertex_colors = mesh.vertex_colors();
  const auto& faces = mesh.vertex_indices();
  glm::vec3 interp_color;
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
void DiffuseTextureNnColorizer::Process(Image3b* color, int x, int y, float u,
                                        float v, uint32_t face_index,
                                        const Mesh& mesh) const {
  const auto& uv = mesh.uv();
  const auto& uv_indices = mesh.uv_indices();
  const auto& diffuse_texture = mesh.diffuse_tex();

  glm::vec3 interp_color;
  // barycentric interpolation of uv
  glm::vec2 interp_uv = (1.0f - u - v) * uv[uv_indices[face_index][0]] +
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
void DiffuseTextureBilinearColorizer::Process(Image3b* color, int x, int y,
                                              float u, float v,
                                              uint32_t face_index,
                                              const Mesh& mesh) const {
  const auto& uv = mesh.uv();
  const auto& uv_indices = mesh.uv_indices();
  const auto& diffuse_texture = mesh.diffuse_tex();

  glm::vec3 interp_color;

  // barycentric interpolation of uv
  glm::vec2 interp_uv = (1.0f - u - v) * uv[uv_indices[face_index][0]] +
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
void DiffuseDefaultShader::Process(Image3b* color, int x, int y, float u,
                                   float v, uint32_t face_index,
                                   const Mesh& mesh) const {}

DiffuseLambertianShader::DiffuseLambertianShader() {}
DiffuseLambertianShader::~DiffuseLambertianShader() {}
void DiffuseLambertianShader::Process(Image3b* color, int x, int y, float u,
                                      float v, uint32_t face_index,
                                      const Mesh& mesh) const {}

DiffuseOrenNayarShader::DiffuseOrenNayarShader() {}
DiffuseOrenNayarShader::~DiffuseOrenNayarShader() {}
void DiffuseOrenNayarShader::Process(Image3b* color, int x, int y, float u,
                                     float v, uint32_t face_index,
                                     const Mesh& mesh) const {}

}  // namespace currender
