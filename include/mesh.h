#pragma once

#include <vector>

#include "common.h"
#include "image.h"

namespace unclearness {
class Mesh {
  std::vector<glm::vec3> vertices_;
  std::vector<glm::vec3> vertex_colors_; // optional, RGB order
  std::vector<glm::ivec3> vertex_indices_;  // face

  std::vector<glm::vec3> normals_;
  std::vector<glm::ivec3> normal_indices_;

  std::vector<glm::vec2> uv_;
  std::vector<glm::ivec3> uv_indices_;

  std::string diffuse_texname_;
  std::string diffuse_texpath_;
  Image3b diffuse_tex_;
 public:
  Mesh();
  ~Mesh();
  void clear();
  void calc_normal();
  const std::vector<glm::vec3>& vertices();
  const std::vector<glm::vec3>& vertex_colors();
  const std::vector<glm::ivec3>& vertex_indices();
  const std::vector<glm::vec2>& uv();
  const std::vector<glm::ivec3>& uv_indices();
  const Image3b& diffuse_tex();
  bool load_obj(const std::string& obj_path, const std::string& mtl_dir);
  bool load_ply(const std::string& ply_path);
};

}  // namespace unclearness