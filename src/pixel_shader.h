/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <memory>

#include "include/common.h"
#include "include/image.h"
#include "include/mesh.h"
#include "include/renderer.h"
#include "nanort/nanort.h"

namespace currender {

class DiffuseColorizer {
 public:
  DiffuseColorizer();
  virtual ~DiffuseColorizer();
  virtual void Process(Image3b* color, int x, int y, float u, float v,
                       uint32_t face_index, const Mesh& mesh) const = 0;
};

class DiffuseDefaultColorizer : public DiffuseColorizer {
 public:
  DiffuseDefaultColorizer();
  ~DiffuseDefaultColorizer();
  void Process(Image3b* color, int x, int y, float u, float v,
               uint32_t face_index, const Mesh& mesh) const override;
};

class DiffuseVertexColorColorizer : public DiffuseColorizer {
 public:
  DiffuseVertexColorColorizer();
  ~DiffuseVertexColorColorizer();
  void Process(Image3b* color, int x, int y, float u, float v,
               uint32_t face_index, const Mesh& mesh) const override;
};

class DiffuseTextureNnColorizer : public DiffuseColorizer {
 public:
  DiffuseTextureNnColorizer();
  ~DiffuseTextureNnColorizer();
  void Process(Image3b* color, int x, int y, float u, float v,
               uint32_t face_index, const Mesh& mesh) const override;
};

class DiffuseTextureBilinearColorizer : public DiffuseColorizer {
 public:
  DiffuseTextureBilinearColorizer();
  ~DiffuseTextureBilinearColorizer();
  void Process(Image3b* color, int x, int y, float u, float v,
               uint32_t face_index, const Mesh& mesh) const override;
};

class DiffuseShader {
 public:
  DiffuseShader();
  virtual ~DiffuseShader();
  virtual void Process(Image3b* color, int x, int y, float u, float v,
                       uint32_t face_index, const Mesh& mesh) const = 0;
};

class DiffuseDefaultShader : public DiffuseShader {
 public:
  DiffuseDefaultShader();
  ~DiffuseDefaultShader();
  void Process(Image3b* color, int x, int y, float u, float v,
               uint32_t face_index, const Mesh& mesh) const override;
};

class DiffuseLambertianShader : public DiffuseShader {
 public:
  DiffuseLambertianShader();
  ~DiffuseLambertianShader();
  void Process(Image3b* color, int x, int y, float u, float v,
               uint32_t face_index, const Mesh& mesh) const override;
};

class DiffuseOrenNayarShader : public DiffuseShader {
 public:
  DiffuseOrenNayarShader();
  ~DiffuseOrenNayarShader();
  void Process(Image3b* color, int x, int y, float u, float v,
               uint32_t face_index, const Mesh& mesh) const override;
};

class PixelShader {
  const DiffuseColorizer* diffuse_colorizer_{nullptr};
  const DiffuseShader* diffuse_shader_{nullptr};

  PixelShader(const PixelShader&) = delete;
  PixelShader& operator=(const PixelShader&) = delete;
  PixelShader(PixelShader&&) = delete;
  PixelShader& operator=(PixelShader&&) = delete;

  PixelShader(const DiffuseColorizer* diffuse_colorizer,
              const DiffuseShader* diffuse_shader);

 public:
  friend class PixelShaderFactory;
  PixelShader();
  ~PixelShader();
  void Process(Image3b* color, int x, int y, float u, float v,
               uint32_t face_index, const Mesh& mesh) const;
};

class PixelShaderFactory {
  PixelShaderFactory();
  ~PixelShaderFactory();

 public:
  static std::unique_ptr<PixelShader> Create(DiffuseColor diffuse_color,
                                             ColorInterpolation interp,
                                             DiffuseShading diffuse_shading);
};

}  // namespace currender
