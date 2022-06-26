/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <array>
#include <map>
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
  std::string name = "default_mat";

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
  std::vector<uint8_t> diffuse_compressed;

  std::string with_alpha_texname;
  std::string with_alpha_texpath;
  Image4b with_alpha_tex;
  std::vector<uint8_t> with_alpha_compressed;

  std::string ToString() const;
};

struct Blendshape {
  float max = 1.f;
  float min = 0.f;
  float weight = 0.f;
  std::string name;
  std::vector<Eigen::Vector3f> vertices;
  std::vector<Eigen::Vector3f> normals;
};

enum class AnimInterp { LINEAR, NN };

struct AnimKeyframe {
  bool R_valid = false;
  bool t_valid = false;
  bool s_valid = false;
  bool weights_valid = false;

  Eigen::Quaternionf q = Eigen::Quaternionf::Identity();
  Eigen::Vector3f t = Eigen::Vector3f::Zero();
  float s = 1.f;

  std::vector<float> weights;

  bool valid() const { return R_valid & t_valid & s_valid & weights_valid; }
};

class Mesh;
using MeshPtr = std::shared_ptr<Mesh>;


#if 0
				class AnimationSystem {
private:
  std::vector<MeshPtr> meshes_;
  std::unordered_map<MeshPtr, std::map<uint32_t, AnimKeyframe>> keyframes_;
  uint32_t fps_ = 30;
  AnimInterp anim_interp_ = AnimInterp::LINEAR;

public:
  
};
#endif  // 0



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

  std::vector<Blendshape> blendshapes_;

  // To keep key (sec.) order, use map
  std::map<float, AnimKeyframe> keyframes_;
  //uint32_t fps_ = 30;
  AnimInterp anim_interp_ = AnimInterp::LINEAR;

 public:
  Mesh();
  ~Mesh();
  Mesh(const Mesh& src);
  static MeshPtr Create();
  static MeshPtr Create(const Mesh& src);

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
  void Transform(const Eigen::Affine3f& T);
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
  const std::vector<std::vector<int>>& face_indices_per_material() const;
  const std::vector<Blendshape>& blendshapes() const;
  const std::map<float, AnimKeyframe>& keyframes() const;
  const AnimInterp& anim_interp() const;

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
  bool set_single_material(const ObjMaterial& material);
  bool set_default_material();
  bool set_face_indices_per_material(
      const std::vector<std::vector<int>>& face_indices_per_material);
  bool set_blendshapes(const std::vector<Blendshape>& blendshapes);
  bool set_keyframes(const std::map<float, AnimKeyframe>& keyframes);
  bool set_anim_interp(const AnimInterp& anim_interp);

  bool LoadObj(const std::string& obj_path, const std::string& mtl_dir);
  bool LoadPly(const std::string& ply_path);
  bool WritePly(const std::string& ply_path) const;
  // not const since this will update texture name and path
  bool WriteObj(const std::string& obj_dir, const std::string& obj_basename,
                const std::string& mtl_basename = "", bool write_obj = true,
                bool write_mtl = true, bool write_texture = true);
  bool WriteGltfSeparate(const std::string& gltf_dir,
                         const std::string& gltf_basename,
                         bool is_unlit = false);
  bool WriteGlb(const std::string& glb_dir, const std::string& glb_name,
                bool is_unlit = false);

  int RemoveVertices(const std::vector<bool>& valid_vertex_table);
  int RemoveUnreferencedVertices();
  int RemoveFaces();
  int RemoveFaces(const std::vector<bool>& valid_face_table);
  int RemoveDuplicateFaces();

  bool SplitMultipleUvVertices();

  bool FlipFaces();

  bool AnimatedShape(float frame, std::vector<Eigen::Vector3f>& anim_verts,
                     std::vector<Eigen::Vector3f>& anim_normals,
                     const AnimInterp& anim_interp = AnimInterp::LINEAR);
};

// TODO: Scene graph
using Scene = std::vector<MeshPtr>;
bool WriteGltfSeparate(Scene& scene, const std::string& gltf_dir,
                       const std::string& gltf_basename, bool is_unlit = false);
bool WriteGlb(Scene& scene, const std::string& glb_dir, const std::string& glb_name,
              bool is_unlit = false);

}  // namespace ugu
