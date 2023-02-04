
/*
 * Copyright (C) 2023, unclearness
 * All rights reserved.
 */

#pragma once

#include <string>

#include "ugu/common.h"

namespace ugu {

enum class FragShaderType { WHITE, UNLIT, NORMAL, POS, UV, GBUF, DEFERRED };
enum class VertShaderType { DEFAULT, GBUF, DEFERRED };

class Shader {
 public:
  uint32_t ID = uint32_t(~0);
  FragShaderType frag_type = FragShaderType::UNLIT;
  VertShaderType vert_type = VertShaderType::DEFAULT;

  Shader();
  ~Shader();

  void SetFragType(const FragShaderType &frag_type);
  void SetVertType(const VertShaderType &frag_type);
  bool Prepare();

  bool LoadStr(const std::string &vertex_code, const std::string &fragment_code,
               const std::string &geometry_code = "");
  bool LoadFile(const std::string &vertex_path,
                const std::string &fragment_path,
                const std::string &geometry_path = "");
  void Use();
  void SetBool(const std::string &name, bool value) const;
  void SetInt(const std::string &name, int value) const;
  void SetFloat(const std::string &name, float value) const;
  void SetVec2(const std::string &name, const Eigen::Vector2f &value) const;
  void SetVec2(const std::string &name, float x, float y) const;
  void SetVec3(const std::string &name, const Eigen::Vector3f &value) const;
  void SetVec3(const std::string &name, float x, float y, float z) const;
  void SetVec4(const std::string &name, const Eigen::Vector4f &value) const;
  void SetVec4(const std::string &name, float x, float y, float z,
               float w) const;
  void SetMat2(const std::string &name, const Eigen::Matrix2f &mat) const;
  void SetMat3(const std::string &name, const Eigen::Matrix3f &mat) const;
  void SetMat4(const std::string &name, const Eigen::Matrix4f &mat) const;
};
}  // namespace ugu
