/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "include/mesh.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader/tiny_obj_loader.h"

namespace crender {

Mesh::Mesh() {}
Mesh::~Mesh() {}

const std::vector<glm::vec3>& Mesh::vertices() const { return vertices_; }
const std::vector<glm::vec3>& Mesh::vertex_colors() const {
  return vertex_colors_;
}
const std::vector<glm::ivec3>& Mesh::vertex_indices() const {
  return vertex_indices_;
}
const std::vector<glm::vec2>& Mesh::uv() const { return uv_; }
const std::vector<glm::ivec3>& Mesh::uv_indices() const { return uv_indices_; }

const MeshStats& Mesh::stats() const { return stats_; }

const Image3b& Mesh::diffuse_tex() const { return diffuse_tex_; }

void Mesh::calc_stats() {
  stats_.bb_min = glm::vec3(std::numeric_limits<float>::max());
  stats_.bb_max = glm::vec3(std::numeric_limits<float>::lowest());

  if (vertex_indices_.empty()) {
    return;
  }

  double sum[3] = {0.0, 0.0, 0.0};  // use double to avoid overflow
  for (const auto& v : vertices_) {
    for (int i = 0; i < 3; i++) {
      sum[i] += v[i];

      if (v[i] < stats_.bb_min[i]) {
        stats_.bb_min[i] = v[i];
      }

      if (stats_.bb_max[i] < v[i]) {
        stats_.bb_max[i] = v[i];
      }
    }
  }

  for (int i = 0; i < 3; i++) {
    stats_.center[i] = static_cast<float>(sum[i] / vertices_.size());
  }
}

void Mesh::rotate(const glm::mat3& R) {
  for (auto& v : vertices_) {
    v = R * v;
  }
  for (auto& n : normals_) {
    n = R * n;
  }
  calc_stats();
}
void Mesh::translate(const glm::vec3& t) {
  for (auto& v : vertices_) {
    v = v + t;
  }
  calc_stats();
}

void Mesh::transform(const glm::mat3& R, const glm::vec3& t) {
  rotate(R);
  translate(t);
}

void Mesh::clear() {
  vertices_.clear();
  vertex_colors_.clear();
  vertex_indices_.clear();  // face

  normals_.clear();
  normal_indices_.clear();

  uv_.clear();
  uv_indices_.clear();

  diffuse_tex_.clear();
}

void Mesh::calc_normal() {
  normals_.clear();
  normal_indices_.clear();

  std::copy(vertex_indices_.begin(), vertex_indices_.end(),
            std::back_inserter(normal_indices_));

  glm::vec3 zero{0.0f, 0.0f, 0.0f};
  normals_.resize(vertices_.size(), zero);

  std::vector<int> add_count(vertices_.size(), 0);

  for (const auto& f : vertex_indices_) {
    glm::vec3 v1 = glm::normalize(vertices_[f[1]] - vertices_[f[0]]);
    glm::vec3 v2 = glm::normalize(vertices_[f[2]] - vertices_[f[0]]);

    glm::vec3 face_normal = glm::cross(v1, v2);

    for (int i = 0; i < 3; i++) {
      int idx = f[i];
      normals_[idx] += face_normal;
      add_count[idx]++;
    }
  }

  for (size_t i = 0; i < vertices_.size(); i++) {
    normals_[i] /= static_cast<float>(add_count[i]);
    normals_[i] = glm::normalize(normals_[i]);
  }
}

bool Mesh::load_obj(const std::string& obj_path, const std::string& mtl_dir) {
  clear();

  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  tinyobj::attrib_t attrib;
  std::string err_str, warn_str;
  bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn_str, &err_str,
                              obj_path.c_str(), mtl_dir.c_str());

  if (!err_str.empty()) {  // `err` may contain warning message.
    LOGE("%s\n", err_str.c_str());
  }

  if (!ret) {
    return false;
  }

  if (materials.size() != 1) {
    LOGE("Doesn't support obj materials num %d. Must be 1\n",
         static_cast<int>(materials.size()));
    return false;
  }

  size_t face_num = 0;
  for (size_t s = 0; s < shapes.size(); s++) {
    face_num += shapes[s].mesh.num_face_vertices.size();
  }
  vertex_indices_.resize(face_num);  // face
  uv_indices_.resize(face_num);
  normal_indices_.resize(face_num);

  vertices_.resize(attrib.vertices.size());
  normals_.resize(attrib.normals.size());
  uv_.resize(attrib.texcoords.size());
  vertex_colors_.resize(attrib.colors.size());

  size_t face_offset = 0;
  // Loop over shapes
  for (size_t s = 0; s < shapes.size(); s++) {
    // Loop over faces(polygon)
    size_t index_offset = 0;

    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
      int fv = shapes[s].mesh.num_face_vertices[f];

      if (fv != 3) {
        LOGE("Doesn't support face num %d. Must be 3\n", fv);
        return false;
      }

      // Loop over vertices in the face.
      for (int v = 0; v < fv; v++) {
        // access to vertex
        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
        tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
        tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
        tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];

        vertex_indices_[face_offset][v] = idx.vertex_index;

        vertices_[idx.vertex_index][0] = vx;
        vertices_[idx.vertex_index][1] = vy;
        vertices_[idx.vertex_index][2] = vz;

        if (!attrib.normals.empty()) {
          tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
          tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
          tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];

          normal_indices_[face_offset][v] = idx.normal_index;
          normals_[idx.normal_index][0] = nx;
          normals_[idx.normal_index][1] = ny;
          normals_[idx.normal_index][2] = nz;
        }

        if (!attrib.texcoords.empty()) {
          tinyobj::real_t tx = attrib.texcoords[2 * idx.texcoord_index + 0];
          tinyobj::real_t ty = attrib.texcoords[2 * idx.texcoord_index + 1];

          uv_indices_[face_offset][v] = idx.texcoord_index;
          uv_[idx.texcoord_index][0] = tx;
          uv_[idx.texcoord_index][1] = ty;
        }
        // Optional: vertex colors
        if (!attrib.colors.empty()) {
          tinyobj::real_t red = attrib.colors[3 * idx.vertex_index + 0];
          tinyobj::real_t green = attrib.colors[3 * idx.vertex_index + 1];
          tinyobj::real_t blue = attrib.colors[3 * idx.vertex_index + 2];

          vertex_colors_[idx.vertex_index][0] = red;
          vertex_colors_[idx.vertex_index][1] = green;
          vertex_colors_[idx.vertex_index][2] = blue;
        }
      }
      index_offset += fv;
      face_offset++;

      // per-face material
      shapes[s].mesh.material_ids[f];
    }
  }

  if (normals_.empty()) {
    calc_normal();
  }

  calc_stats();

  diffuse_texname_ = materials[0].diffuse_texname;
  diffuse_texpath_ = mtl_dir + diffuse_texname_;

  diffuse_tex_.load(diffuse_texpath_);

  return true;
}
bool Mesh::load_ply(const std::string& ply_path) {
  (void)ply_path;
  LOGE("Haven't been implemented\n");
  return false;
}

};  // namespace crender
