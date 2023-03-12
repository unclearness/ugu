/*
 * Copyright (C) 2023, unclearness
 * All rights reserved.
 */

#include "ugu/shader/shader.h"

#include "shader/frag.h"
#include "shader/geom.h"
#include "shader/vert.h"

#ifdef UGU_USE_GLFW
#include "glad/gl.h"
#endif

#include <fstream>

#ifdef UGU_USE_GLFW

namespace {

bool CheckCompileErrors(GLuint shader, std::string type) {
  GLint success;
  GLchar infoLog[1024];
  if (type != "PROGRAM") {
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
      glGetShaderInfoLog(shader, 1024, NULL, infoLog);
      std::cout
          << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n"
          << infoLog
          << "\n -- --------------------------------------------------- -- "
          << std::endl;
    }
  } else {
    glGetProgramiv(shader, GL_LINK_STATUS, &success);
    if (!success) {
      glGetProgramInfoLog(shader, 1024, NULL, infoLog);
      std::cout
          << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n"
          << infoLog
          << "\n -- --------------------------------------------------- -- "
          << std::endl;
    }
  }
  return success != 0;
}

}  // namespace

namespace ugu {

Shader::Shader() : ID(ID_counter++) {}
Shader::~Shader() {}

void Shader::SetFragType(const FragShaderType &frag_type_) {
  frag_type = frag_type_;
}

void Shader::SetVertType(const VertShaderType &vert_type_) {
  vert_type = vert_type_;
}

bool Shader::Prepare() {
  std::string vertex_code = "";
  std::string fragment_code = "";
  std::string geometry_code = "";

  vertex_code = vert_gbuf_code;
  if (vert_type == VertShaderType::GBUF) {
    vertex_code = vert_gbuf_code;
    geometry_code = geom_dummy_code;
  } else if (vert_type == VertShaderType::DEFERRED) {
    vertex_code = vert_deferred_code;
  } else if (vert_type == VertShaderType::TEXT) {
    vertex_code = vert_text_code;
  }

  if (frag_type == FragShaderType::WHITE) {
  } else if (frag_type == FragShaderType::UNLIT) {
    fragment_code = frag_deferred_code;
  } else if (frag_type == FragShaderType::NORMAL) {
  } else if (frag_type == FragShaderType::POS) {
  } else if (frag_type == FragShaderType::UV) {
  } else if (frag_type == FragShaderType::GBUF) {
    fragment_code = frag_gbuf_code;
  } else if (frag_type == FragShaderType::DEFERRED) {
    fragment_code = frag_deferred_code;
  } else if (frag_type == FragShaderType::TEXT) {
    fragment_code = frag_text_code;
  }

  return LoadStr(vertex_code, fragment_code, geometry_code);
}

bool Shader::LoadFile(const std::string &vertex_path,
                      const std::string &fragment_path,
                      const std::string &geometry_path) {
  // 1. retrieve the vertex/fragment source code from filePath
  std::string vertexCode;
  std::string fragmentCode;
  std::string geometryCode;
  std::ifstream vShaderFile;
  std::ifstream fShaderFile;
  std::ifstream gShaderFile;
  // ensure ifstream objects can throw exceptions:
  vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  gShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  try {
    // open files
    vShaderFile.open(vertex_path);
    fShaderFile.open(fragment_path);
    std::stringstream vShaderStream, fShaderStream;
    // read file's buffer contents into streams
    vShaderStream << vShaderFile.rdbuf();
    fShaderStream << fShaderFile.rdbuf();
    // close file handlers
    vShaderFile.close();
    fShaderFile.close();
    // convert stream into string
    vertexCode = vShaderStream.str();
    fragmentCode = fShaderStream.str();
    // if geometry shader path is present, also load a geometry shader
    if (!geometry_path.empty()) {
      gShaderFile.open(geometry_path);
      std::stringstream gShaderStream;
      gShaderStream << gShaderFile.rdbuf();
      gShaderFile.close();
      geometryCode = gShaderStream.str();
    }
  } catch (std::ifstream::failure &e) {
    std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ: " << e.what()
              << std::endl;
    return false;
  }
  return LoadStr(vertexCode, fragmentCode, geometryCode);
}

bool Shader::LoadStr(const std::string &vertex_code,
                     const std::string &fragment_code,
                     const std::string &geometry_code) {
  const char *vShaderCode = vertex_code.c_str();
  const char *fShaderCode = fragment_code.c_str();

  bool ret = true;

  // 2. compile shaders
  unsigned int vertex = ~0u, fragment = ~0u;
  // vertex shader
  vertex = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertex, 1, &vShaderCode, NULL);
  glCompileShader(vertex);
  ret &= CheckCompileErrors(vertex, "VERTEX");

  // fragment Shader
  fragment = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragment, 1, &fShaderCode, NULL);
  glCompileShader(fragment);
  ret &= CheckCompileErrors(fragment, "FRAGMENT");
  // if geometry shader is given, compile geometry shader
  unsigned int geometry = ~0u;
  if (!geometry_code.empty()) {
    const char *gShaderCode = geometry_code.c_str();
    geometry = glCreateShader(GL_GEOMETRY_SHADER);
    glShaderSource(geometry, 1, &gShaderCode, NULL);
    glCompileShader(geometry);
    ret &= CheckCompileErrors(geometry, "GEOMETRY");
  }
  // shader Program
  ID = glCreateProgram();
  glAttachShader(ID, vertex);
  glAttachShader(ID, fragment);
  if (!geometry_code.empty()) {
    glAttachShader(ID, geometry);
  }
  glLinkProgram(ID);
  ret &= CheckCompileErrors(ID, "PROGRAM");
  // delete the shaders as they're linked into our program now and no longer
  // necessery
  glDeleteShader(vertex);
  glDeleteShader(fragment);
  if (!geometry_code.empty()) {
    glDeleteShader(geometry);
  }

  return ret;
}

void Shader::Use() { glUseProgram(ID); }

void Shader::SetBool(const std::string &name, bool value) const {
  glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
}

void Shader::SetInt(const std::string &name, int value) const {
  glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
}
void Shader::SetFloat(const std::string &name, float value) const {
  glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
}
void Shader::SetVec2(const std::string &name,
                     const Eigen::Vector2f &value) const {
  glUniform2fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
}
void Shader::SetVec2(const std::string &name, float x, float y) const {
  glUniform2f(glGetUniformLocation(ID, name.c_str()), x, y);
}
void Shader::SetVec3(const std::string &name,
                     const Eigen::Vector3f &value) const {
  glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
}
void Shader::SetVec3(const std::string &name, float x, float y, float z) const {
  glUniform3f(glGetUniformLocation(ID, name.c_str()), x, y, z);
}
void Shader::SetVec4(const std::string &name,
                     const Eigen::Vector4f &value) const {
  glUniform4fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
}

void Shader::SetVec4(const std::string &name, float x, float y, float z,
                     float w) const {
  glUniform4f(glGetUniformLocation(ID, name.c_str()), x, y, z, w);
}

void Shader::SetMat2(const std::string &name,
                     const Eigen::Matrix2f &mat) const {
  glUniformMatrix2fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE,
                     mat.data());
}

void Shader::SetMat3(const std::string &name,
                     const Eigen::Matrix3f &mat) const {
  glUniformMatrix3fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE,
                     mat.data());
}

void Shader::SetMat4(const std::string &name,
                     const Eigen::Matrix4f &mat) const {
  glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE,
                     mat.data());
}

void Shader::SetVec2Array(const std::string &name,
                          const std::vector<Eigen::Vector2f> &values) const {
  glUniform2fv(glGetUniformLocation(ID, name.c_str()),
               static_cast<GLuint>(values.size()), values[0].data());
}

void Shader::SetVec3Array(const std::string &name,
                          const std::vector<Eigen::Vector3f> &values) const {
  glUniform3fv(glGetUniformLocation(ID, name.c_str()),
               static_cast<GLuint>(values.size()), values[0].data());
}

}  // namespace ugu

#else

namespace ugu {

Shader::Shader() : ID(uint32_t(~0)) {
  LOGE("Not supported with this configuration\n");
}
Shader::~Shader() {}

bool Shader::LoadFile(const std::string &vertex_path,
                      const std::string &fragment_path,
                      const std::string &geometry_path) {
  LOGE("Not supported with this configuration\n");
  return false;
}

bool Shader::LoadStr(const std::string &vertex_code,
                     const std::string &fragment_code,
                     const std::string &geometry_code) {
  LOGE("Not supported with this configuration\n");
  return false;
}

void Shader::Use() { LOGE("Not supported with this configuration\n"); }

void Shader::SetBool(const std::string &name, bool value) const {
  LOGE("Not supported with this configuration\n");
}

void Shader::SetInt(const std::string &name, int value) const {
  LOGE("Not supported with this configuration\n");
}
void Shader::SetFloat(const std::string &name, float value) const {
  LOGE("Not supported with this configuration\n");
}
void Shader::SetVec2(const std::string &name,
                     const Eigen::Vector2f &value) const {
  LOGE("Not supported with this configuration\n");
}
void Shader::SetVec2(const std::string &name, float x, float y) const {
  LOGE("Not supported with this configuration\n");
}
void Shader::SetVec3(const std::string &name,
                     const Eigen::Vector3f &value) const {
  LOGE("Not supported with this configuration\n");
}
void Shader::SetVec3(const std::string &name, float x, float y, float z) const {
  LOGE("Not supported with this configuration\n");
}
void Shader::SetVec4(const std::string &name,
                     const Eigen::Vector4f &value) const {
  LOGE("Not supported with this configuration\n");
}

void Shader::SetVec4(const std::string &name, float x, float y, float z,
                     float w) const {
  LOGE("Not supported with this configuration\n");
}

void Shader::SetMat2(const std::string &name,
                     const Eigen::Matrix2f &mat) const {
  LOGE("Not supported with this configuration\n");
}

void Shader::SetMat3(const std::string &name,
                     const Eigen::Matrix3f &mat) const {
  LOGE("Not supported with this configuration\n");
}

void Shader::SetMat4(const std::string &name,
                     const Eigen::Matrix4f &mat) const {
  LOGE("Not supported with this configuration\n");
}
}  // namespace ugu
#endif