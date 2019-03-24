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

struct OrenNayarParam {
 public:
  float sigma{0.0f};
  float A{0.0f};
  float B{0.0f};
  OrenNayarParam();
  explicit OrenNayarParam(float sigma);
};

struct PixelShaderInput {
 private:
  PixelShaderInput();

 public:
  Image3b* color{nullptr};
  int x;
  int y;
  float u;
  float v;
  uint32_t face_index;
  const Eigen::Vector3f* ray_w{nullptr};
  const Eigen::Vector3f* light_dir{nullptr};
  const Eigen::Vector3f* shading_normal{nullptr};
  const OrenNayarParam* oren_nayar_param{nullptr};
  std::shared_ptr<const Mesh> mesh{nullptr};

  PixelShaderInput(Image3b* color, int x, int y, float u, float v,
                   uint32_t face_index, const Eigen::Vector3f* ray_w,
                   const Eigen::Vector3f* light_dir,
                   const Eigen::Vector3f* shading_normal,
                   const OrenNayarParam* oren_nayar_param,
                   std::shared_ptr<const Mesh> mesh);
  ~PixelShaderInput();
};

class DiffuseColorizer {
 public:
  DiffuseColorizer();
  virtual ~DiffuseColorizer();
  virtual void Process(const PixelShaderInput& input) const = 0;
};

class DiffuseDefaultColorizer : public DiffuseColorizer {
 public:
  DiffuseDefaultColorizer();
  ~DiffuseDefaultColorizer();
  void Process(const PixelShaderInput& input) const override;
};

class DiffuseVertexColorColorizer : public DiffuseColorizer {
 public:
  DiffuseVertexColorColorizer();
  ~DiffuseVertexColorColorizer();
  void Process(const PixelShaderInput& input) const override;
};

class DiffuseTextureNnColorizer : public DiffuseColorizer {
 public:
  DiffuseTextureNnColorizer();
  ~DiffuseTextureNnColorizer();
  void Process(const PixelShaderInput& input) const override;
};

class DiffuseTextureBilinearColorizer : public DiffuseColorizer {
 public:
  DiffuseTextureBilinearColorizer();
  ~DiffuseTextureBilinearColorizer();
  void Process(const PixelShaderInput& input) const override;
};

class DiffuseShader {
 public:
  DiffuseShader();
  virtual ~DiffuseShader();
  virtual void Process(const PixelShaderInput& input) const = 0;
};

class DiffuseDefaultShader : public DiffuseShader {
 public:
  DiffuseDefaultShader();
  ~DiffuseDefaultShader();
  void Process(const PixelShaderInput& input) const override;
};

class DiffuseLambertianShader : public DiffuseShader {
 public:
  DiffuseLambertianShader();
  ~DiffuseLambertianShader();
  void Process(const PixelShaderInput& input) const override;
};

class DiffuseOrenNayarShader : public DiffuseShader {
 public:
  DiffuseOrenNayarShader();
  ~DiffuseOrenNayarShader();
  void Process(const PixelShaderInput& input) const override;
};

class PixelShader {
  std::unique_ptr<DiffuseColorizer> diffuse_colorizer_{nullptr};
  std::unique_ptr<DiffuseShader> diffuse_shader_{nullptr};

  PixelShader(const PixelShader&) = delete;
  PixelShader& operator=(const PixelShader&) = delete;
  PixelShader(PixelShader&&) = delete;
  PixelShader& operator=(PixelShader&&) = delete;

  PixelShader(std::unique_ptr<DiffuseColorizer>&& diffuse_colorizer,
              std::unique_ptr<DiffuseShader>&& diffuse_shader);

 public:
  friend class PixelShaderFactory;
  PixelShader();
  ~PixelShader();
  void Process(const PixelShaderInput& input) const;
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
