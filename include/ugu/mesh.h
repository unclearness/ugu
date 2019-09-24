/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "ugu/common.h"
#include "ugu/image.h"

namespace ugu {

struct MeshStats {
  Eigen::Vector3f center;
  Eigen::Vector3f bb_min;
  Eigen::Vector3f bb_max;
};

// partial copy of tinyobj::material_t
struct ObjMaterial {
  std::string name;

  // same to bunny.mtl
  std::array<float, 3> ambient{0.117647f, 0.117647f, 0.117647f};   // Ka
  std::array<float, 3> diffuse{0.752941f, 0.752941f, 0.752941f};   // Kd
  std::array<float, 3> specular{0.752941f, 0.752941f, 0.752941f};  // Ks
  float shininess{8.0f};                                           // Ns
  float dissolve{
      1.0f};  // 1 == opaque; 0 == fully transparent, (inverted: Tr = 1 - d)
  // illumination model (see http://www.fileformat.info/format/material/)
  int illum{1};

  std::string diffuse_texname;
  std::string diffuse_texpath;
  Image3b diffuse_tex;

  std::string ToString() const;
};

class Mesh {
  std::vector<Eigen::Vector3f> vertices_;
  std::vector<Eigen::Vector3f> vertex_colors_;   // optional, RGB order
  std::vector<Eigen::Vector3i> vertex_indices_;  // face

  std::vector<Eigen::Vector3f> normals_;       // normal per vertex
  std::vector<Eigen::Vector3f> face_normals_;  // normal per face
  std::vector<Eigen::Vector3i> normal_indices_;

  std::vector<Eigen::Vector2f> uv_;
  std::vector<Eigen::Vector3i> uv_indices_;

  std::vector<ObjMaterial> materials_;

  // material_ids_[i]: face index i's material id.
  // This is used to access materials_.
  std::vector<int> material_ids_;

  // face_indices_per_material_[i]: the vector of material i's face indices.
  std::vector<std::vector<int>> face_indices_per_material_;
  MeshStats stats_;

 public:
  Mesh();
  ~Mesh();
  Mesh(const Mesh& src);
  void Clear();

  // get average normal per vertex from face normal
  // caution: this does not work for cube with 8 vertices unless vertices are
  // splitted (24 vertices)
  void CalcNormal();

  void CalcFaceNormal();
  void CalcStats();
  void Rotate(const Eigen::Matrix3f& R);
  void Translate(const Eigen::Vector3f& t);
  void Transform(const Eigen::Matrix3f& R, const Eigen::Vector3f& t);
  void Scale(float scale);
  void Scale(float x_scale, float y_scale, float z_scale);
  const std::vector<Eigen::Vector3f>& vertices() const;
  const std::vector<Eigen::Vector3f>& vertex_colors() const;
  const std::vector<Eigen::Vector3i>& vertex_indices() const;
  const std::vector<Eigen::Vector3f>& normals() const;
  const std::vector<Eigen::Vector3f>& face_normals() const;
  const std::vector<Eigen::Vector3i>& normal_indices() const;
  const std::vector<Eigen::Vector2f>& uv() const;
  const std::vector<Eigen::Vector3i>& uv_indices() const;
  const MeshStats& stats() const;
  const std::vector<int>& material_ids() const;
  const std::vector<ObjMaterial>& materials() const;

  bool set_vertices(const std::vector<Eigen::Vector3f>& vertices);
  bool set_vertex_colors(const std::vector<Eigen::Vector3f>& vertex_colors);
  bool set_vertex_indices(const std::vector<Eigen::Vector3i>& vertex_indices);
  bool set_normals(const std::vector<Eigen::Vector3f>& normals);
  bool set_face_normals(const std::vector<Eigen::Vector3f>& face_normals);
  bool set_normal_indices(const std::vector<Eigen::Vector3i>& normal_indices);
  bool set_uv(const std::vector<Eigen::Vector2f>& uv);
  bool set_uv_indices(const std::vector<Eigen::Vector3i>& uv_indices);
  bool set_material_ids(const std::vector<int>& material_ids);
  bool set_materials(const std::vector<ObjMaterial>& materials);

  bool LoadObj(const std::string& obj_path, const std::string& mtl_dir);
  bool LoadPly(const std::string& ply_path);
  bool WritePly(const std::string& ply_path) const;
  // not const since this will update texture name and path
  bool WriteObj(const std::string& obj_dir, const std::string& obj_basename,
                const std::string& mtl_basename = "", bool write_obj = true,
                bool write_mtl = true, bool write_texture = true);
};

// make cube with 24 vertices
std::shared_ptr<Mesh> MakeCube(const Eigen::Vector3f& length,
                               const Eigen::Matrix3f& R,
                               const Eigen::Vector3f& t);
std::shared_ptr<Mesh> MakeCube(const Eigen::Vector3f& length);
std::shared_ptr<Mesh> MakeCube(float length, const Eigen::Matrix3f& R,
                               const Eigen::Vector3f& t);
std::shared_ptr<Mesh> MakeCube(float length);

void SetRandomVertexColor(std::shared_ptr<Mesh> mesh, int seed = 0);

}  // namespace ugu
