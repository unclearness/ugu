#pragma once

#include <vector>
#include <string>

#include "include/common.h"
#include "include/image.h"

namespace unclearness {

class MeshStats {
 public:
  glm::vec3 center;
  glm::vec3 bb_min;
  glm::vec3 bb_max;
};

class Mesh {
  std::vector<glm::vec3> vertices_;
  std::vector<glm::vec3> vertex_colors_;    // optional, RGB order
  std::vector<glm::ivec3> vertex_indices_;  // face

  std::vector<glm::vec3> normals_;
  std::vector<glm::ivec3> normal_indices_;

  std::vector<glm::vec2> uv_;
  std::vector<glm::ivec3> uv_indices_;

  std::string diffuse_texname_;
  std::string diffuse_texpath_;
  Image3b diffuse_tex_;
  MeshStats stats_;

 public:
  Mesh();
  ~Mesh();
  void clear();
  void calc_normal();
  void calc_stats();
  void rotate(const glm::mat3& R);
  void translate(const glm::vec3& t);
  void transform(const glm::mat3& R, const glm::vec3& t); // NOLINT
  const std::vector<glm::vec3>& vertices() const;
  const std::vector<glm::vec3>& vertex_colors() const;
  const std::vector<glm::ivec3>& vertex_indices() const;
  const std::vector<glm::vec2>& uv() const;
  const std::vector<glm::ivec3>& uv_indices() const;
  const MeshStats& stats() const;
  const Image3b& diffuse_tex() const;
  bool load_obj(const std::string& obj_path, const std::string& mtl_dir);
  bool load_ply(const std::string& ply_path);
};

}  // namespace unclearness
