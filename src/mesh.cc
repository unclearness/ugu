/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "ugu/mesh.h"

#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <unordered_set>

#include "gltf.h"
#include "ugu/face_adjacency.h"
#include "ugu/util/image_util.h"
#include "ugu/util/string_util.h"

#ifdef UGU_USE_TINYOBJLOADER
#include "tiny_obj_loader.h"
#endif

namespace {

template <typename T>
void CopyVec(const std::vector<T>& src, std::vector<T>* dst,
             bool clear_dst = true) {
  if (clear_dst) {
    dst->clear();
    dst->reserve(src.size());
  }
  std::copy(src.begin(), src.end(), std::back_inserter(*dst));
}

bool WriteMtl(const std::string& path,
              const std::vector<ugu::ObjMaterial>& materials) {
  std::ofstream ofs(path);
  if (ofs.fail()) {
    ugu::LOGE("couldn't open mtl path: %s\n", path.c_str());
    return false;
  }

  for (size_t i = 0; i < materials.size(); i++) {
    const ugu::ObjMaterial& material = materials[i];
    ofs << material.ToString();
    if (i != materials.size() - 1) {
      ofs << '\n';
    }
  }
  ofs.close();

  return true;
}

bool WriteTexture(const std::vector<ugu::ObjMaterial>& materials) {
  // write texture
  bool ret{true};
  for (size_t i = 0; i < materials.size(); i++) {
    const ugu::ObjMaterial& material = materials[i];
    bool ret_write = false;
    if (!material.diffuse_tex.empty() && !material.diffuse_texpath.empty()) {
      ret_write = imwrite(material.diffuse_texpath, material.diffuse_tex);
    }
    if (!material.with_alpha_tex.empty() &&
        !material.with_alpha_texpath.empty()) {
      ret_write = imwrite(material.with_alpha_texpath, material.with_alpha_tex);
    }
    if (ret) {
      ret = ret_write;
    }
  }

  return ret;
}
}  // namespace

namespace ugu {

std::string ObjMaterial::ToString() const {
  std::stringstream ss;

  ss << "newmtl " << name << '\n'
     << "Ka " << ambient[0] << " " << ambient[1] << " " << ambient[2] << '\n'
     << "Kd " << diffuse[0] << " " << diffuse[1] << " " << diffuse[2] << '\n'
     << "Ks " << specular[0] << " " << specular[1] << " " << specular[2] << '\n'
     << "Tr " << 1.0f - dissolve << '\n'
     << "illum " << illum << '\n'
     << "Ns " << shininess << '\n';
  if (!diffuse_texname.empty()) {
    ss << "map_Kd " << diffuse_texname << "\n";
  }

  ss.flush();

  return ss.str();
}

Mesh::Mesh() {}
Mesh::Mesh(const Mesh& src) {
  CopyVec(src.vertices_, &vertices_);
  CopyVec(src.vertex_colors_, &vertex_colors_);
  CopyVec(src.vertex_indices_, &vertex_indices_);

  CopyVec(src.normals_, &normals_);
  CopyVec(src.face_normals_, &face_normals_);
  CopyVec(src.normal_indices_, &normal_indices_);

  CopyVec(src.uv_, &uv_);
  CopyVec(src.uv_indices_, &uv_indices_);

  CopyVec(src.materials_, &materials_);
  CopyVec(src.material_ids_, &material_ids_);
  CopyVec(src.face_indices_per_material_, &face_indices_per_material_);

  stats_ = src.stats_;
}
Mesh::~Mesh() {}

MeshPtr Mesh::Create() { return std::make_shared<Mesh>(); }

MeshPtr Mesh::Create(const Mesh& src) { return std::make_shared<Mesh>(src); }

const std::vector<Eigen::Vector3f>& Mesh::vertices() const { return vertices_; }
const std::vector<Eigen::Vector3f>& Mesh::vertex_colors() const {
  return vertex_colors_;
}
const std::vector<Eigen::Vector3i>& Mesh::vertex_indices() const {
  return vertex_indices_;
}

const std::vector<Eigen::Vector3f>& Mesh::normals() const { return normals_; }
const std::vector<Eigen::Vector3f>& Mesh::face_normals() const {
  return face_normals_;
}
const std::vector<Eigen::Vector3i>& Mesh::normal_indices() const {
  return normal_indices_;
}

const std::vector<Eigen::Vector2f>& Mesh::uv() const { return uv_; }
const std::vector<Eigen::Vector3i>& Mesh::uv_indices() const {
  return uv_indices_;
}

const MeshStats& Mesh::stats() const { return stats_; }

const std::vector<int>& Mesh::material_ids() const { return material_ids_; }
const std::vector<ObjMaterial>& Mesh::materials() const { return materials_; }

const std::vector<std::vector<int>>& Mesh::face_indices_per_material() const {
  return face_indices_per_material_;
}

const std::vector<Blendshape>& Mesh::blendshapes() const {
  return blendshapes_;
};

const std::map<float, AnimKeyframe>& Mesh::keyframes() const {
  return keyframes_;
};

const AnimInterp& Mesh::anim_interp() const { return anim_interp_; };

void Mesh::CalcStats() {
  stats_.bb_min = Eigen::Vector3f(std::numeric_limits<float>::max(),
                                  std::numeric_limits<float>::max(),
                                  std::numeric_limits<float>::max());
  stats_.bb_max = Eigen::Vector3f(std::numeric_limits<float>::lowest(),
                                  std::numeric_limits<float>::lowest(),
                                  std::numeric_limits<float>::lowest());

  if (vertices_.empty()) {
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

void Mesh::Rotate(const Eigen::Matrix3f& R) {
  for (auto& v : vertices_) {
    v = R * v;
  }
  for (auto& n : normals_) {
    n = R * n;
  }
  for (auto& fn : face_normals_) {
    fn = R * fn;
  }
  CalcStats();
}

void Mesh::Translate(const Eigen::Vector3f& t) {
  for (auto& v : vertices_) {
    v = v + t;
  }
  CalcStats();
}

void Mesh::Transform(const Eigen::Matrix3f& R, const Eigen::Vector3f& t) {
  Rotate(R);
  Translate(t);
}

void Mesh::Transform(const Eigen::Affine3f& T) {
  Transform(T.rotation(), T.translation());
}

void Mesh::Scale(float scale) { Scale(scale, scale, scale); }

void Mesh::Scale(float x_scale, float y_scale, float z_scale) {
  for (auto& v : vertices_) {
    v[0] = v[0] * x_scale;
    v[1] = v[1] * y_scale;
    v[2] = v[2] * z_scale;
  }
}

void Mesh::Clear() {
  vertices_.clear();
  vertex_colors_.clear();
  vertex_indices_.clear();  // face

  normals_.clear();
  normal_indices_.clear();

  uv_.clear();
  uv_indices_.clear();

  materials_.clear();
  material_ids_.clear();
  face_indices_per_material_.clear();
}

void Mesh::CalcNormal() {
  if (vertex_indices_.empty()) {
    return;
  }

  CalcFaceNormal();

  normals_.clear();
  normal_indices_.clear();

  std::copy(vertex_indices_.begin(), vertex_indices_.end(),
            std::back_inserter(normal_indices_));

  Eigen::Vector3f zero{0.0f, 0.0f, 0.0f};
  normals_.resize(vertices_.size(), zero);

  std::vector<int> add_count(vertices_.size(), 0);

  for (size_t i = 0; i < vertex_indices_.size(); i++) {
    const auto& face = vertex_indices_[i];
    for (int j = 0; j < 3; j++) {
      int idx = face[j];
      normals_[idx] += face_normals_[i];
      add_count[idx]++;
    }
  }

  // get average normal
  // caution: this does not work for cube
  // https://answers.unity.com/questions/441722/splitting-up-verticies.html
  for (size_t i = 0; i < vertices_.size(); i++) {
    if (add_count[i] > 0) {
      normals_[i] /= static_cast<float>(add_count[i]);
      normals_[i].normalize();
    } else {
      // for unreferenced vertices, set (0, 0, 0)
      normals_[i].setZero();
    }
  }
}

void Mesh::CalcFaceNormal() {
  face_normals_.clear();
  face_normals_.resize(vertex_indices_.size());

  for (size_t i = 0; i < vertex_indices_.size(); i++) {
    const auto& f = vertex_indices_[i];
    Eigen::Vector3f v1 = (vertices_[f[1]] - vertices_[f[0]]).normalized();
    Eigen::Vector3f v2 = (vertices_[f[2]] - vertices_[f[0]]).normalized();
    face_normals_[i] = v1.cross(v2).normalized();
  }
}

bool Mesh::set_vertices(const std::vector<Eigen::Vector3f>& vertices) {
  if (vertices.size() > std::numeric_limits<int>::max()) {
    LOGE("The number of vertices exceeds the maximum: %d\n",
         std::numeric_limits<int>::max());
    return false;
  }
  CopyVec(vertices, &vertices_);
  return true;
}

bool Mesh::set_vertex_colors(
    const std::vector<Eigen::Vector3f>& vertex_colors) {
  if (vertex_colors.size() > std::numeric_limits<int>::max()) {
    LOGE("The number of vertices exceeds the maximum: %d\n",
         std::numeric_limits<int>::max());
    return false;
  }
  CopyVec(vertex_colors, &vertex_colors_);
  return true;
}

bool Mesh::set_vertex_indices(
    const std::vector<Eigen::Vector3i>& vertex_indices) {
  if (vertex_indices.size() > std::numeric_limits<int>::max()) {
    LOGE("The number of faces exceeds the maximum: %d\n",
         std::numeric_limits<int>::max());
    return false;
  }
  CopyVec(vertex_indices, &vertex_indices_);
  return true;
}

bool Mesh::set_normals(const std::vector<Eigen::Vector3f>& normals) {
  if (normals.size() > std::numeric_limits<int>::max()) {
    LOGE("The number of vertices exceeds the maximum: %d\n",
         std::numeric_limits<int>::max());
    return false;
  }
  CopyVec(normals, &normals_);
  return true;
}

bool Mesh::set_face_normals(const std::vector<Eigen::Vector3f>& face_normals) {
  if (face_normals.size() > std::numeric_limits<int>::max()) {
    LOGE("The number of faces exceeds the maximum: %d\n",
         std::numeric_limits<int>::max());
    return false;
  }
  CopyVec(face_normals, &face_normals_);
  return true;
}

bool Mesh::set_normal_indices(
    const std::vector<Eigen::Vector3i>& normal_indices) {
  if (normal_indices.size() > std::numeric_limits<int>::max()) {
    LOGE("The number of faces exceeds the maximum: %d\n",
         std::numeric_limits<int>::max());
    return false;
  }
  CopyVec(normal_indices, &normal_indices_);
  return true;
}

bool Mesh::set_uv(const std::vector<Eigen::Vector2f>& uv) {
  if (uv.size() > std::numeric_limits<int>::max()) {
    LOGE("The number of vertices exceeds the maximum: %d\n",
         std::numeric_limits<int>::max());
    return false;
  }
  CopyVec(uv, &uv_);
  return true;
}

bool Mesh::set_uv_indices(const std::vector<Eigen::Vector3i>& uv_indices) {
  if (uv_indices.size() > std::numeric_limits<int>::max()) {
    LOGE("The number of faces exceeds the maximum: %d\n",
         std::numeric_limits<int>::max());
    return false;
  }
  CopyVec(uv_indices, &uv_indices_);
  return true;
}

bool Mesh::set_material_ids(const std::vector<int>& material_ids) {
  if (material_ids.empty()) {
    LOGE("material id is empty\n");
    return false;
  }
  int max_id = *std::max_element(material_ids.begin(), material_ids.end());
  if (max_id < 0) {
    LOGE("material id must be positive\n");
    return false;
  }

  CopyVec(material_ids, &material_ids_);

  face_indices_per_material_.resize(max_id + 1);
  for (auto& fipm : face_indices_per_material_) {
    fipm.clear();
  }
  for (int i = 0; i < static_cast<int>(material_ids_.size()); i++) {
    face_indices_per_material_[material_ids_[i]].push_back(i);
  }

  return true;
}

bool Mesh::set_materials(const std::vector<ObjMaterial>& materials) {
  CopyVec(materials, &materials_);
  return true;
}

bool Mesh::set_face_indices_per_material(
    const std::vector<std::vector<int>>& face_indices_per_material) {
  face_indices_per_material_.resize(face_indices_per_material.size());
  for (size_t k = 0; k < face_indices_per_material.size(); k++) {
    CopyVec(face_indices_per_material[k], &face_indices_per_material_[k]);
  }
  return true;
}

bool Mesh::set_single_material(const ObjMaterial& material) {
  set_materials({material});
  std::vector<int> material_ids(vertex_indices_.size(), 0);
  set_material_ids(material_ids);
  return true;
}

bool Mesh::set_default_material() {
  const static auto default_mat = ObjMaterial();
  return set_single_material(default_mat);
}

bool Mesh::set_blendshapes(const std::vector<Blendshape>& blendshapes) {
  CopyVec(blendshapes, &blendshapes_);
  return true;
}

bool Mesh::set_keyframes(const std::map<float, AnimKeyframe>& keyframes) {
  keyframes_ = keyframes;
  return true;
}

bool Mesh::set_anim_interp(const AnimInterp& anim_interp) {
  anim_interp_ = anim_interp;
  return true;
}

#ifdef UGU_USE_TINYOBJLOADER
bool Mesh::LoadObj(const std::string& obj_path, const std::string& mtl_dir) {
  Clear();

  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  tinyobj::attrib_t attrib;
  std::string err_str, warn_str;
  bool return_default_vertex_color{false};
  bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn_str, &err_str,
                              obj_path.c_str(), mtl_dir.c_str(), true,
                              return_default_vertex_color);

  if (!err_str.empty()) {  // `err` may contain warning message.
    LOGE("%s\n", err_str.c_str());
  }

  if (!ret) {
    return false;
  }

  size_t face_num = 0;
  for (size_t s = 0; s < shapes.size(); s++) {
    face_num += shapes[s].mesh.num_face_vertices.size();
  }

  if (face_num > std::numeric_limits<int>::max()) {
    LOGE("The number of faces exceeds the maximum: %d\n",
         std::numeric_limits<int>::max());
    return false;
  }

  vertex_indices_.resize(face_num);  // face
  uv_indices_.resize(face_num);
  normal_indices_.resize(face_num);
  material_ids_.resize(face_num, 0);

  if (attrib.vertices.size() / 3 > std::numeric_limits<int>::max() ||
      attrib.normals.size() / 3 > std::numeric_limits<int>::max() ||
      attrib.texcoords.size() / 2 > std::numeric_limits<int>::max() ||
      attrib.colors.size() / 3 > std::numeric_limits<int>::max()) {
    LOGE("The number of vertices exceeds the maximum: %d\n",
         std::numeric_limits<int>::max());
    return false;
  }

  vertices_.resize(attrib.vertices.size() / 3);
  normals_.resize(attrib.normals.size() / 3);
  uv_.resize(attrib.texcoords.size() / 2);
  vertex_colors_.resize(attrib.colors.size() / 3);

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

      // per-face material
      material_ids_[face_offset] = shapes[s].mesh.material_ids[f];

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
          normals_[idx.normal_index].normalize();  // Fail safe
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
    }
  }

  CalcFaceNormal();

  if (normals_.empty()) {
    CalcNormal();
  }

  CalcStats();

  if (materials.empty()) {
    materials_.resize(1);
    materials_[0] = ObjMaterial();
    material_ids_.assign(face_num, 0);

    LOGW(
        "Default material was added because material did not find on input "
        "obj\n");

  } else {
    materials_.resize(materials.size());
    for (size_t i = 0; i < materials.size(); i++) {
      materials_[i].name = materials[i].name;
      std::copy(std::begin(materials[i].ambient),
                std::end(materials[i].ambient), materials_[i].ambient.begin());
      std::copy(std::begin(materials[i].diffuse),
                std::end(materials[i].diffuse), materials_[i].diffuse.begin());
      std::copy(std::begin(materials[i].specular),
                std::end(materials[i].specular),
                materials_[i].specular.begin());
      materials_[i].shininess = materials[i].shininess;
      materials_[i].dissolve = materials[i].dissolve;
      materials_[i].illum = materials[i].illum;

      materials_[i].diffuse_texname = materials[i].diffuse_texname;
      materials_[i].diffuse_texpath = mtl_dir + materials_[i].diffuse_texname;
      std::ifstream ifs(materials_[i].diffuse_texpath);
      if (ifs.is_open()) {
#if defined(UGU_USE_STB) || defined(UGU_USE_OPENCV)
        // todo: force convert to Image3b
        materials_[i].diffuse_tex =
            imread<Image3b>(materials_[i].diffuse_texpath);
        ret = !materials_[i].diffuse_tex.empty();
#else
        LOGW("define UGU_USE_STB to load diffuse texture.\n");
#endif
      } else {
        LOGW("diffuse texture doesn't exist %s\n",
             materials_[i].diffuse_texpath.c_str());
      }
    }
  }

  face_indices_per_material_.resize(materials_.size());
  for (int i = 0; i < static_cast<int>(material_ids_.size()); i++) {
    face_indices_per_material_[material_ids_[i]].push_back(i);
  }

  return ret;
}
#else
bool Mesh::LoadObj(const std::string& obj_path, const std::string& mtl_dir) {
  (void)obj_path;
  (void)mtl_dir;
  LOGE("can't load obj with this configuration\n");
  return false;
}
#endif

bool Mesh::LoadPly(const std::string& ply_path) {
  std::ifstream ifs(ply_path);
  std::string str;
  if (ifs.fail()) {
    LOGE("couldn't open ply: %s\n", ply_path.c_str());
    return false;
  }

  getline(ifs, str);
  if (str != "ply") {
    LOGE("ply first line is wrong: %s\n", str.c_str());
    return false;
  }
  getline(ifs, str);
  if (str.find("ascii") == std::string::npos) {
    LOGE("only ascii ply is supported: %s\n", str.c_str());
    return false;
  }

  bool ret = false;
  std::int64_t vertex_num = 0;
  while (getline(ifs, str)) {
    if (str.find("element vertex") != std::string::npos) {
      std::vector<std::string> splitted = Split(str, ' ');
      if (splitted.size() == 3) {
        vertex_num = std::atol(splitted[2].c_str());
        ret = true;
        break;
      }
    }
  }
  if (!ret) {
    LOGE("couldn't find element vertex\n");
    return false;
  }
  if (vertex_num > std::numeric_limits<int>::max()) {
    LOGE("The number of vertices exceeds the maximum: %d\n",
         std::numeric_limits<int>::max());
    return false;
  }

  ret = false;
  std::int64_t face_num = 0;
  while (getline(ifs, str)) {
    if (str.find("element face") != std::string::npos) {
      std::vector<std::string> splitted = Split(str, ' ');
      if (splitted.size() == 3) {
        face_num = std::atol(splitted[2].c_str());
        ret = true;
        break;
      }
    }
  }
  if (!ret) {
    LOGE("couldn't find element face\n");
    return false;
  }
  if (face_num > std::numeric_limits<int>::max()) {
    LOGE("The number of faces exceeds the maximum: %d\n",
         std::numeric_limits<int>::max());
    return false;
  }

  while (getline(ifs, str)) {
    if (str.find("end_header") != std::string::npos) {
      break;
    }
  }

  vertices_.resize(vertex_num);
  int vertex_count = 0;
  while (getline(ifs, str)) {
    std::vector<std::string> splitted = Split(str, ' ');
    vertices_[vertex_count][0] =
        static_cast<float>(std::atof(splitted[0].c_str()));
    vertices_[vertex_count][1] =
        static_cast<float>(std::atof(splitted[1].c_str()));
    vertices_[vertex_count][2] =
        static_cast<float>(std::atof(splitted[2].c_str()));

    // TODO
    if (splitted.size() >= 9) {
      auto vc = Eigen::Vector3f();
      vc[0] = static_cast<float>(std::atof(splitted[6].c_str()));
      vc[1] = static_cast<float>(std::atof(splitted[7].c_str()));
      vc[2] = static_cast<float>(std::atof(splitted[8].c_str()));
      vertex_colors_.push_back(vc);
    }
    vertex_count++;
    if (vertex_count >= vertex_num) {
      break;
    }
  }

  vertex_indices_.resize(face_num);
  int face_count = 0;
  while (getline(ifs, str)) {
    std::vector<std::string> splitted = Split(str, ' ');
    vertex_indices_[face_count][0] = std::atoi(splitted[1].c_str());
    vertex_indices_[face_count][1] = std::atoi(splitted[2].c_str());
    vertex_indices_[face_count][2] = std::atoi(splitted[3].c_str());

    face_count++;
    if (face_count >= face_num) {
      break;
    }
  }

  ifs.close();

  CalcNormal();

  CalcStats();

  return true;
}

bool Mesh::WritePly(const std::string& ply_path) const {
  std::ofstream ofs(ply_path);
  std::string str;
  if (ofs.fail()) {
    LOGE("couldn't open ply: %s\n", ply_path.c_str());
    return false;
  }

  bool has_vertex_normal =
      !vertices_.empty() && vertices_.size() == normals_.size();

  bool has_vertex_color =
      !vertices_.empty() && vertices_.size() == vertex_colors_.size();

  ofs << "ply"
      << "\n";
  ofs << "format ascii 1.0"
      << "\n";
  ofs << "element vertex " + std::to_string(vertices_.size()) << "\n";
  ofs << "property float x\n"
         "property float y\n"
         "property float z\n";
  if (has_vertex_normal) {
    ofs << "property float nx\n"
           "property float ny\n"
           "property float nz\n";
  }
  if (has_vertex_color) {
    ofs << "property uchar red\n"
           "property uchar green\n"
           "property uchar blue\n"
           "property uchar alpha\n";
  }
  ofs << "element face " + std::to_string(vertex_indices_.size()) << "\n";
  ofs << "property list uchar int vertex_indices"
      << "\n";
  ofs << "end_header"
      << "\n";

  for (size_t i = 0; i < vertices_.size(); i++) {
    ofs << vertices_[i][0] << " " << vertices_[i][1] << " " << vertices_[i][2]
        << " ";
    if (has_vertex_normal) {
      ofs << normals_[i][0] << " " << normals_[i][1] << " " << normals_[i][2]
          << " ";
    }
    if (has_vertex_color) {
      ofs << static_cast<int>(std::round(vertex_colors_[i][0])) << " "
          << static_cast<int>(std::round(vertex_colors_[i][1])) << " "
          << static_cast<int>(std::round(vertex_colors_[i][2])) << " 255 ";
    }
    ofs << "\n";
  }

  for (size_t i = 0; i < vertex_indices_.size(); i++) {
    ofs << "3 " << vertex_indices_[i][0] << " " << vertex_indices_[i][1] << " "
        << vertex_indices_[i][2] << " "
        << "\n";
  }

  ofs.close();

  return true;
}

bool Mesh::WriteObj(const std::string& obj_dir, const std::string& obj_basename,
                    const std::string& mtl_basename, bool write_obj,
                    bool write_mtl, bool write_texture) {
  bool ret{true};
  std::string mtl_name = mtl_basename + ".mtl";
  if (mtl_basename.empty()) {
    mtl_name = obj_basename + ".mtl";
  }
  std::string mtl_path = obj_dir + "/" + mtl_name;

  std::string obj_path = obj_dir + "/" + obj_basename + ".obj";

  // write obj
  if (write_obj) {
    std::ofstream ofs(obj_path);
    if (ofs.fail()) {
      LOGE("couldn't open obj path: %s\n", obj_path.c_str());
      return false;
    }

    ofs << "mtllib " << mtl_name << "\n"
        << "\n";

    // vertices
    if (!vertex_colors_.empty() && vertex_colors_.size() == vertices_.size()) {
      for (size_t i = 0; i < vertices_.size(); i++) {
        const auto& v = vertices_[i];
        const auto& vc = vertex_colors_[i] / 255.f;
        ofs << "v " << v.x() << " " << v.y() << " " << v.z() << " " << vc.x()
            << " " << vc.y() << " " << vc.z() << "\n";
      }
    } else {
      for (const auto& v : vertices_) {
        ofs << "v " << v.x() << " " << v.y() << " " << v.z() << " 1.0"
            << "\n";
      }
    }

    // uv
    for (const auto& vt : uv_) {
      ofs << "vt " << vt.x() << " " << vt.y() << " 0"
          << "\n";
    }

    // vertex normals
    for (const auto& vn : normals_) {
      ofs << "vn " << vn.x() << " " << vn.y() << " " << vn.z() << "\n";
    }

    // indices by material (group)
    // CAUTION: This breaks original face indices
    bool write_uv_indices = !uv_indices_.empty();
    bool write_normal_indices = !normal_indices_.empty();
#if 1
    for (size_t k = 0; k < face_indices_per_material_.size(); k++) {
      auto mat_name = materials_[k].name;
      if (!mat_name.empty()) {
        ofs << "usemtl " << materials_[k].name << "\n";
      }
      for (size_t i = 0; i < face_indices_per_material_[k].size(); i++) {
        int f_idx = face_indices_per_material_[k][i];
        ofs << "f";
        for (int j = 0; j < 3; j++) {
          ofs << " " << std::to_string(vertex_indices_[f_idx][j] + 1);
          if (!write_uv_indices && !write_normal_indices) {
            continue;
          }
          ofs << "/";
          if (write_uv_indices) {
            ofs << std::to_string(uv_indices_[f_idx][j] + 1);
          }
          if (write_normal_indices) {
            ofs << "/" << std::to_string(normal_indices_[f_idx][j] + 1);
          }
        }
        ofs << "\n";
      }
    }
#else
    // naive writing keeping original face indices
    for (size_t i = 0; i < vertex_indices_.size(); i++) {
      ofs << "f";
      for (int j = 0; j < 3; j++) {
        ofs << " " << std::to_string(vertex_indices_[i][j] + 1);
        if (!write_uv_indices && !write_normal_indices) {
          continue;
        }
        ofs << "/";
        if (write_uv_indices) {
          ofs << std::to_string(uv_indices_[i][j] + 1);
        }
        ofs << "/" << std::to_string(normal_indices_[i][j] + 1);
      }
      ofs << "\n";
    }
#endif

    ofs.close();
  }

  // update texture path
  for (auto& material : materials_) {
    if (material.diffuse_texname.empty() && !material.diffuse_tex.empty()) {
      // default name
      material.diffuse_texname = obj_basename + ".png";
    }

    // update path
    if (!material.diffuse_texname.empty()) {
      material.diffuse_texpath = obj_dir + "/" + material.diffuse_texname;
    }
  }

  // write mtl
  if (write_mtl) {
    ret = WriteMtl(mtl_path, materials_);
  }

  if (write_texture) {
    ret = WriteTexture(materials_);
  }

  return ret;
}

int Mesh::RemoveVertices(const std::vector<bool>& valid_vertex_table) {
  if (valid_vertex_table.size() != vertices_.size()) {
    LOGE("valid_vertex_table must be same size to vertices");
    return -1;
  }

#if 0
  int num_removed{0};
  std::vector<int> valid_table(vertices_.size(), -1);
  std::vector<Eigen::Vector3f> valid_vertices, valid_vertex_colors;
  std::vector<Eigen::Vector2f> valid_uv;
  std::vector<Eigen::Vector3i> valid_indices;
  bool with_uv = !uv_.empty() && !uv_indices_.empty();
  bool with_vertex_color = !vertex_colors_.empty();
  int valid_count = 0;
  for (size_t i = 0; i < vertices_.size(); i++) {
    if (valid_vertex_table[i]) {
      valid_table[i] = valid_count;
      valid_vertices.push_back(vertices_[i]);
      if (with_uv) {
        valid_uv.push_back(uv_[i]);
      }
      if (with_vertex_color) {
        valid_vertex_colors.push_back(vertex_colors_[i]);
      }
      valid_count++;
    } else {
      num_removed++;
    }
  }

  int valid_face_count{0};
  std::vector<int> valid_face_table(vertex_indices_.size(), -1);
  for (size_t i = 0; i < vertex_indices_.size(); i++) {
    Eigen::Vector3i face;
    bool valid{true};
    for (int j = 0; j < 3; j++) {
      int new_index = valid_table[vertex_indices_[i][j]];
      if (new_index < 0) {
        valid = false;
        break;
      }
      face[j] = new_index;
    }
    if (!valid) {
      continue;
    }
    valid_indices.push_back(face);
    valid_face_table[i] = valid_face_count;
    valid_face_count++;
  }
#endif

  auto [num_removed, valid_table, valid_vertices, valid_vertex_colors, valid_uv,
        valid_indices, valid_face_table, with_uv, with_vertex_color] =
      RemoveVerticesBase(vertices_, vertex_indices_, vertex_colors_, uv_,
                         uv_indices_, valid_vertex_table);

  set_vertices(valid_vertices);
  set_vertex_indices(valid_indices);
  if (with_uv) {
    set_uv(valid_uv);
    set_uv_indices(valid_indices);
  }
  if (with_vertex_color) {
    set_vertex_colors(valid_vertex_colors);
  }
  CalcNormal();

  std::vector<int> new_material_ids(valid_indices.size(), 0);
  const std::vector<int>& old_material_ids = material_ids_;

  for (size_t i = 0; i < old_material_ids.size(); i++) {
    int org_f_idx = static_cast<int>(i);
    int new_f_idx = valid_face_table[org_f_idx];
    if (new_f_idx < 0) {
      continue;
    }
    new_material_ids[new_f_idx] = old_material_ids[org_f_idx];
  }

  set_material_ids(new_material_ids);

  // no need to operate face_indices_per_material directly
  // set_material_ids() will update it
#if 0
  std::vector<std::vector<int>> updated_face_indices_per_material(
      face_indices_per_material_.size());
  for (size_t k = 0; k < face_indices_per_material_.size(); k++) {
    for (size_t i = 0; i < face_indices_per_material_[k].size(); i++) {
      int org_f_idx = face_indices_per_material_[k][i];
      int new_f_idx = valid_face_table[org_f_idx];
      if (new_f_idx < 0) {
        continue;
      }
      updated_face_indices_per_material[k].push_back(new_f_idx);
    }
  }
  set_face_indices_per_material(updated_face_indices_per_material);
#endif  // 0

  return num_removed;
}

bool Mesh::WriteGltfSeparate(const std::string& gltf_dir,
                             const std::string& gltf_basename, bool is_unlit) {
#ifdef UGU_USE_JSON
  gltf::Model model;
  // Make .bin and update model info
  this->CalcStats();  // ensure min/max
  model.meshes.resize(1);
  model.meshes[0].name = ExtractPathWithoutExt(gltf_basename);

  // Prepare blendshapes
  model.meshes[0].with_blendshapes = !this->blendshapes_.empty();
  for (auto& p : model.meshes[0].primitives) {
    p.with_blendshapes = model.meshes[0].with_blendshapes;
  }
  for (const auto& b : this->blendshapes_) {
    model.meshes[0].blendshape_names.push_back(b.name);
    model.meshes[0].blendshape_weights.push_back(b.weight);
  }
  model.meshes[0].primitives[0].blendshape_num =
      static_cast<std::uint32_t>(this->blendshapes_.size());

  std::string bin_name = gltf_basename + ".bin";
  std::vector<std::uint8_t> bin;
  model.accessors.clear();
  model.bufferViews.clear();
  MakeGltfBinAndUpdateModel(*this, bin_name, false, model, bin);

  // Write .bin
  std::ofstream bin_out(gltf_dir + bin_name,
                        std::ios::out | std::ios::binary | std::ios::trunc);
  bin_out.write(reinterpret_cast<char*>(bin.data()), bin.size());

  // Write texture
  // Update path
  for (auto& mat : this->materials_) {
    mat.diffuse_texpath = gltf_dir + mat.diffuse_texname;
    mat.with_alpha_texpath = gltf_dir + mat.with_alpha_texname;
  }

  // Update materials and textures of the model
  model.materials.resize(this->materials_.size());  // todo: update pbr params
  for (size_t i = 0; i < model.materials.size(); i++) {
    model.materials[i].name = this->materials_[i].name;
    model.materials[i].is_unlit = is_unlit;
  }

  model.images.clear();
  std::vector<ObjMaterial> to_write_mats;
  for (auto& mat : this->materials_) {
    to_write_mats.push_back(mat);
    std::string tex_name = mat.with_alpha_texname;
    std::string tex_path = mat.with_alpha_texpath;
    if (tex_name.empty()) {
      tex_name = mat.diffuse_texname;
      tex_path = mat.diffuse_texpath;
      // Kill with_alpha
      to_write_mats.back().with_alpha_texpath = "";
    } else {
      // Kill diffuse
      to_write_mats.back().diffuse_texpath = "";
    }
    if (tex_name.empty()) {
      continue;
    }

    gltf::Image image;
    image.uri = tex_name;
    image.name = ExtractPathWithoutExt(tex_name);
    model.images.push_back(image);
  }

  WriteTexture(to_write_mats);

  // Write .gltf (json)
  gltf::WriteGltfJsonToFile(model, gltf_dir + gltf_basename + ".gltf");

  return true;
#else

  return false;
#endif
}

bool Mesh::WriteGlb(const std::string& glb_dir, const std::string& glb_name,
                    bool is_unlit) {
#ifdef UGU_USE_JSON
  gltf::Model model;
  // Make .bin and update model info
  this->CalcStats();  // ensure min/max
  model.meshes.resize(1);
  model.meshes[0].name = ExtractPathWithoutExt(glb_name);

  // Prepare blendshapes
  model.meshes[0].with_blendshapes = !this->blendshapes_.empty();
  for (auto& p : model.meshes[0].primitives) {
    p.with_blendshapes = model.meshes[0].with_blendshapes;
  }
  for (const auto& b : this->blendshapes_) {
    model.meshes[0].blendshape_names.push_back(b.name);
    model.meshes[0].blendshape_weights.push_back(b.weight);
  }
  model.meshes[0].primitives[0].blendshape_num =
      static_cast<std::uint32_t>(this->blendshapes_.size());

  // Update materials and textures of the model
  model.materials.resize(this->materials_.size());  // todo: update pbr params
  for (size_t i = 0; i < model.materials.size(); i++) {
    model.materials[i].name = this->materials_[i].name;
    model.materials[i].is_unlit = is_unlit;
  }
  model.images.clear();
  for (auto& mat : this->materials_) {
    std::string tex_name = mat.with_alpha_texname;
    std::string tex_path = mat.with_alpha_texpath;
    std::vector<uint8_t> tex_data = mat.with_alpha_compressed;
    if (tex_name.empty()) {
      tex_name = mat.diffuse_texname;
      tex_path = mat.diffuse_texpath;
      tex_data = mat.diffuse_compressed;
    }
    if (tex_name.empty()) {
      continue;
    }
    gltf::Image image;
    image.is_glb = true;
    std::string ext = ExtractPathExt(tex_path);
    if (ext == "jpg" || ext == "jpeg") {
      image.mimeType = "image/jpeg";
    } else if (ext == "png") {
      image.mimeType = "image/png";
    } else {
      LOGE("ext %s is not supported\n", ext.c_str());
      continue;
    }

    image.name = ExtractPathWithoutExt(tex_name);
    if (tex_data.empty()) {
      // Read jpeg or png data
      std::ifstream ifs(tex_path, std::ios::in | std::ios::binary);
      // Get size
      ifs.seekg(0, std::ios::end);
      long long int size = ifs.tellg();
      ifs.seekg(0);
      // Read binary
      image.data.resize(size);
      ifs.read(reinterpret_cast<char*>(image.data.data()), size);
    } else {
      // Copy from memory
      image.data = std::move(tex_data);
    }

    model.images.push_back(image);
  }

  std::string bin_name = glb_name + ".bin";
  std::vector<std::uint8_t> bin;
  model.accessors.clear();
  model.bufferViews.clear();
  MakeGltfBinAndUpdateModel(*this, bin_name, true, model, bin);

  gltf::Chunk bin_chunk{0x004E4942, bin};  // 0x004E4942 -> "BIN" in ASCII
  std::vector<std::uint8_t> bin_bin = gltf::ChunkToBin(bin_chunk);

  // Make json
  std::string json_string = gltf::WriteGltfJsonToString(model);

  std::vector<std::uint8_t> json_bytes(json_string.size());
  std::memcpy(json_bytes.data(), json_string.c_str(), json_string.size());
  gltf::Chunk json_chunk{0x4E4F534A,
                         json_bytes};  // 0x4E4F534A	-> "JSON" in ASCII
  std::vector<std::uint8_t> json_bin = gltf::ChunkToBin(json_chunk);

  std::uint32_t magic = 0x46546C67;  //  0x46546C67 -> "glTF"in ASCII
  std::uint32_t version = 2;
  int header_size = 12;
  std::uint32_t length = static_cast<std::uint32_t>(
      header_size + json_bin.size() + bin_bin.size());
  std::vector<std::uint8_t> header(header_size);
  std::memcpy(header.data(), &magic, 4);
  std::memcpy(header.data() + 4, &version, 4);
  std::memcpy(header.data() + 8, &length, 4);

  std::vector<std::uint8_t> combined(length);
  std::memcpy(combined.data(), header.data(), header_size);
  std::memcpy(combined.data() + header_size, json_bin.data(), json_bin.size());
  std::memcpy(combined.data() + header_size + json_bin.size(), bin_bin.data(),
              bin_bin.size());

  std::ofstream glb_out(glb_dir + glb_name,
                        std::ios::out | std::ios::binary | std::ios::trunc);
  glb_out.write(reinterpret_cast<char*>(combined.data()), length);

  return true;
#else
  return false;
#endif
}

int Mesh::RemoveUnreferencedVertices() {
  std::vector<bool> reference_table(vertices_.size(), false);
  for (const auto& f : vertex_indices_) {
    for (int i = 0; i < 3; i++) {
      reference_table[f[i]] = true;
    }
  }

  return RemoveVertices(reference_table);
}

int Mesh::RemoveFaces() {
  std::vector<bool> valid_face_table(vertex_indices_.size(), false);

  return RemoveFaces(valid_face_table);
}

int Mesh::RemoveFaces(const std::vector<bool>& valid_face_table) {
  if (valid_face_table.size() != vertex_indices_.size()) {
    LOGE("valid_face_table must be same size to vertex_indices");
    return -1;
  }

  int num_removed = 0;
  std::vector<Eigen::Vector3i> removed_vertex_indices;
  std::vector<Eigen::Vector3f> removed_face_normals;
  std::vector<Eigen::Vector3i> removed_uv_indices;
  std::vector<int> removed_material_ids;
  bool remove_uv = vertex_indices_.size() == uv_indices_.size();
  bool remove_material_id = vertex_indices_.size() == material_ids_.size();
  for (int i = 0; i < static_cast<int>(valid_face_table.size()); i++) {
    if (valid_face_table[i]) {
      removed_vertex_indices.push_back(vertex_indices_[i]);
      removed_face_normals.push_back(face_normals_[i]);
      if (remove_uv) {
        removed_uv_indices.push_back(uv_indices_[i]);
      }
      if (remove_material_id) {
        removed_material_ids.push_back(material_ids_[i]);
      }
    } else {
      num_removed++;
    }
  }

  set_vertex_indices(removed_vertex_indices);
  set_face_normals(removed_face_normals);
  if (remove_uv) {
    set_uv_indices(removed_uv_indices);
  }

  // CalcNormal();

  set_material_ids(removed_material_ids);

  return num_removed;
}

int Mesh::RemoveDuplicateFaces() {
  std::map<std::pair<int, int>, int> edge2count;
  std::unordered_set<int> to_remove_faceids;
  for (int i = 0; i < static_cast<int>(vertex_indices_.size()); i++) {
    const auto& f = vertex_indices_[i];
    auto e0 = std::make_pair(f[0], f[1]);
    if (edge2count.find(e0) == edge2count.end()) {
      edge2count.insert(std::make_pair(e0, i));
    } else {
      ugu::LOGD("edge %d -> %d exisits at %d. remove %d\n", f[0], f[1],
                edge2count[e0], i);
      to_remove_faceids.insert(i);
      continue;
    }

    auto e1 = std::make_pair(f[1], f[2]);
    if (edge2count.find(e1) == edge2count.end()) {
      edge2count.insert(std::make_pair(e1, i));
    } else {
      ugu::LOGD("edge %d -> %d exisits at %d. remove %d\n", f[1], f[2],
                edge2count[e1], i);
      to_remove_faceids.insert(i);
      continue;
    }

    auto e2 = std::make_pair(f[2], f[0]);
    if (edge2count.find(e2) == edge2count.end()) {
      edge2count.insert(std::make_pair(e2, i));
    } else {
      ugu::LOGD("edge %d -> %d exisits at %d. remove %d\n", f[2], f[0],
                edge2count[e2], i);
      to_remove_faceids.insert(i);
      continue;
    }
  }

  if (to_remove_faceids.empty()) {
    return 0;
  }

  std::vector<Eigen::Vector3i> org_vertex_indices;
  std::vector<Eigen::Vector3i> org_uv_indices;

  bool keep_uv =
      (vertex_indices_.size() == uv_indices_.size()) && uv_indices_.size() > 0;

  CopyVec(vertex_indices_, &org_vertex_indices);
  CopyVec(uv_indices_, &org_uv_indices);
  vertex_indices_.clear();
  uv_indices_.clear();

  for (int i = 0; i < static_cast<int>(org_vertex_indices.size()); i++) {
    if (to_remove_faceids.count(i) == 0) {
      vertex_indices_.push_back(org_vertex_indices[i]);
      if (keep_uv) {
        uv_indices_.push_back(org_uv_indices[i]);
      }
    }
  }

  return static_cast<int>(to_remove_faceids.size());
}

bool Mesh::SplitMultipleUvVertices() {
  if (vertices_.size() == uv_.size()) {
    LOGI("No need to split\n");
    // return false;
  }
  if (vertex_indices_.size() != uv_indices_.size() ||
      vertex_indices_.size() != normal_indices_.size()) {
    LOGE("#indices is wrong\n");
    return false;
  }

  this->CalcNormal();
  std::vector<Eigen::Vector3f> vertices, normals;
  std::vector<Eigen::Vector2f> uv;
  std::vector<Eigen::Vector3i> indices;

  auto fnum = static_cast<int>(vertex_indices_.size());
  for (int i = 0; i < fnum; i++) {
    const auto& face = vertex_indices_[i];
    const auto& uvface = uv_indices_[i];
    for (int j = 0; j < 3; j++) {
      vertices.push_back(vertices_[face[j]]);
      normals.push_back(normals_[face[j]]);
      uv.push_back(uv_[uvface[j]]);
    }
    indices.push_back({i * 3, i * 3 + 1, i * 3 + 2});
  }
  set_vertices(vertices);
  set_normals(normals);
  set_uv(uv);

  set_vertex_indices(indices);
  set_normal_indices(indices);
  set_uv_indices(indices);

  return true;
}

bool Mesh::FlipFaces() {
  auto flip = [](Eigen::Vector3i& i) { std::swap(i[1], i[2]); };
  std::for_each(vertex_indices_.begin(), vertex_indices_.end(), flip);
  std::for_each(uv_indices_.begin(), uv_indices_.end(), flip);
  std::for_each(normal_indices_.begin(), normal_indices_.end(), flip);

  CalcNormal();

  return true;
}

bool Mesh::AnimatedShape(float frame, std::vector<Eigen::Vector3f>& anim_verts,
                         std::vector<Eigen::Vector3f>& anim_normals,
                         const AnimInterp& anim_interp) {
  if (keyframes_.empty()) {
    return false;
  }

  uint32_t lower_fn(uint32_t(~0)), upper_fn(uint32_t(~0));
  float lower_w(0.f), upper_w(0.f);
  decltype(keyframes_)::iterator lesseq_iter = keyframes_.lower_bound(frame);
  auto next_iter = std::next(lesseq_iter);

  // Find lower and upper keyframes
  if (lesseq_iter == keyframes_.end()) {
    lower_fn = upper_fn = (*(--lesseq_iter)).first;
  } else if (next_iter == keyframes_.end()) {
    lower_fn = upper_fn = (*(lesseq_iter)).first;
  } else {
    lower_fn = lesseq_iter->first;
    upper_fn = next_iter->first;
  }

  // Calc weights for the lower and upper keyframes
  if (anim_interp == AnimInterp::LINEAR) {
    if (lower_fn == upper_fn) {
      lower_w = 1.f;
      upper_w = 0.f;
    } else {
      uint32_t diff = upper_fn - lower_fn;
      lower_w = static_cast<float>(upper_fn - frame) / static_cast<float>(diff);
      upper_w = static_cast<float>(frame - lower_fn) / static_cast<float>(diff);
    }
  } else if (anim_interp == AnimInterp::NN) {
    if ((upper_fn - frame) < (frame - lower_fn)) {
      lower_w = 0.f;
      upper_w = 1.f;
    } else {
      lower_w = 1.f;
      upper_w = 0.f;
    }
  }

  const AnimKeyframe& lower_kf = keyframes_[lower_fn];
  const AnimKeyframe& upper_kf = keyframes_[upper_fn];

  // TODO
  if (!(lower_kf.valid() && upper_kf.valid())) {
    LOGE("should be all valid\n");
    return false;
  }

  // Apply weights
  anim_verts.clear();
  anim_normals.clear();
  Eigen::Quaternionf q = Eigen::Quaternionf::Identity();
  Eigen::Vector3f t(0.f, 0.f, 0.f);
  float s = 1.f;

  // TODO: correct?
  q = lower_kf.q.slerp(lower_w, upper_kf.q);

  t = lower_kf.t * lower_w + upper_kf.t * upper_w;
  s = lower_kf.s * lower_w + upper_kf.s * upper_w;

  Eigen::Affine3f trans = Eigen::Translation3f(t) * q * Eigen::Scaling(s);

  anim_verts = vertices_;
  anim_normals = normals_;
  if (!blendshapes_.empty() && lower_kf.weights.size() == blendshapes_.size() &&
      upper_kf.weights.size() == blendshapes_.size()) {
    bool with_normals =
        blendshapes_[0].vertices.size() == blendshapes_[0].normals.size();
    for (size_t i = 0; i < anim_verts.size(); i++) {
      for (size_t j = 0; j < blendshapes_.size(); j++) {
        float lower_w_bs = lower_w * lower_kf.weights[j];
        float upper_w_bs = upper_w * upper_kf.weights[j];
        anim_verts[i] += lower_w_bs * blendshapes_[j].vertices[i] +
                         upper_w_bs * blendshapes_[j].vertices[i];

        if (with_normals) {
          anim_normals[i] += lower_w_bs * blendshapes_[j].normals[i] +
                             upper_w_bs * blendshapes_[j].normals[i];
          anim_normals[i].normalize();
        }
      }
    }
  }

  auto apply_trans = [&](std::vector<Eigen::Vector3f>& vec) {
    for (auto& v : vec) {
      v = trans * v;
    }
  };
  auto apply_q = [&](std::vector<Eigen::Vector3f>& vec) {
    for (auto& v : vec) {
      v = q * v;
    }
  };

  apply_trans(anim_verts);
  apply_q(anim_normals);

#if 0
  anim_verts = vertices_;
  anim_normals = normals_;

  auto apply_trans = [&](std::vector<Eigen::Vector3f>& vec) {
    for (auto& v : vec) {
      v = trans * v;
    }
  };
  auto apply_q = [&](std::vector<Eigen::Vector3f>& vec) {
    for (auto& v : vec) {
      v = q * v;
    }
  };

  apply_trans(anim_verts);
  apply_q(anim_normals);

  if (!blendshapes_.empty() && lower_kf.weights.size() == blendshapes_.size() &&
      upper_kf.weights.size() == blendshapes_.size()) {
    std::vector<Blendshape> anim_blendshapes = blendshapes_;
    for (auto& bs : anim_blendshapes) {
      apply_trans(bs.vertices);
      apply_q(bs.normals);
    }

    for (size_t i = 0; i < anim_verts.size(); i++) {
      
    }

    for (size_t i = 0; i < anim_normals.size(); i++) {
      }

  }
#endif  // 0

  return true;
}

bool WriteGltfSeparate(Scene& scene, const std::string& gltf_dir,
                       const std::string& gltf_basename, bool is_unlit) {
#ifdef UGU_USE_JSON
  gltf::Model model;
  std::string bin_name = gltf_basename + ".bin";

  std::vector<std::uint8_t> bin_all;

  model.meshes.resize(scene.size());

  model.scenes[0].nodes.clear();
  model.nodes.clear();
  model.images.clear();
  model.materials.clear();
  model.accessors.clear();
  model.bufferViews.clear();
  model.buffers.resize(1);
  model.textures.clear();

  uint32_t total_mat_num = 0;
  uint32_t mat_count = 0;
  for (size_t msh_idx = 0; msh_idx < scene.size(); msh_idx++) {
    total_mat_num += static_cast<uint32_t>(scene[msh_idx]->materials().size());
  }

  model.materials.resize(total_mat_num);  // todo: update pbr params

  for (size_t msh_idx = 0; msh_idx < scene.size(); msh_idx++) {
    MeshPtr mesh = scene[msh_idx];

    model.scenes[0].nodes.push_back(static_cast<int>(msh_idx));
    ugu::gltf::Node node;
    node.mesh = static_cast<uint32_t>(msh_idx);
    node.name = std::to_string(msh_idx);  // TODO
    model.nodes.push_back(node);

    // Make .bin and update model info
    mesh->CalcStats();  // ensure min/max

    model.meshes[msh_idx].name = ExtractPathWithoutExt(gltf_basename);

    model.meshes[msh_idx].with_blendshapes = !mesh->blendshapes().empty();

    ugu::gltf::Primitive primitive;
    int primitive_offset = 4;  // model.meshes[msh_idx].with_blendshapes ? 5 :
                               // 4;
    if (!mesh->keyframes().empty()) {
      primitive_offset += 4;
    }
    primitive.indices = static_cast<uint32_t>(msh_idx * primitive_offset + 3);
    primitive.material = static_cast<uint32_t>(msh_idx);
    primitive.attributes["NORMAL"] =
        static_cast<uint32_t>(msh_idx * primitive_offset + 1);
    primitive.attributes["POSITION"] =
        static_cast<uint32_t>(msh_idx * primitive_offset + 0);
    primitive.attributes["TEXCOORD_0"] =
        static_cast<uint32_t>(msh_idx * primitive_offset + 2);
    model.meshes[msh_idx].primitives = {primitive};

    // Prepare blendshapes

    for (auto& p : model.meshes[msh_idx].primitives) {
      p.with_blendshapes = model.meshes[msh_idx].with_blendshapes;
    }
    for (const auto& b : mesh->blendshapes()) {
      model.meshes[msh_idx].blendshape_names.push_back(b.name);
      model.meshes[msh_idx].blendshape_weights.push_back(b.weight);
    }
    model.meshes[msh_idx].primitives[0].blendshape_num =
        static_cast<std::uint32_t>(mesh->blendshapes().size());

    // std::vector<std::uint8_t> bin =
    MakeGltfBinAndUpdateModel(*mesh, bin_name, false, model, bin_all);
    // std::copy(bin.begin(), bin.end(), std::back_inserter(bin_all));
    // bin_all = std::move(bin);

    // Write texture

    // Update materials and textures of the model

    for (size_t i = 0; i < mesh->materials().size(); i++) {
      model.materials[mat_count + i].name = mesh->materials()[i].name;
      model.materials[mat_count + i].is_unlit = is_unlit;
      model.materials[mat_count + i].with_alpha =
          !mesh->materials()[i].with_alpha_tex.empty();
      model.materials[mat_count + i]
          .pbrMetallicRoughness.baseColorTexture.index = mat_count;
      ugu::gltf::Texture texture;
      texture.source = mat_count;
      model.textures.push_back(texture);
      mat_count++;
    }

    std::vector<ObjMaterial> to_write_mats;

    for (auto mat : mesh->materials()) {
      mat.with_alpha_texpath = gltf_dir + mat.with_alpha_texname;
      mat.diffuse_texpath = gltf_dir + mat.diffuse_texname;
      to_write_mats.push_back(mat);
      std::string tex_name = mat.with_alpha_texname;
      std::string tex_path = mat.with_alpha_texpath;
      if (tex_name.empty()) {
        tex_name = mat.diffuse_texname;
        tex_path = mat.diffuse_texpath;
        // Kill with_alpha
        to_write_mats.back().with_alpha_texpath = "";
      } else {
        // Kill diffuse
        to_write_mats.back().diffuse_texpath = "";
      }
      if (tex_name.empty()) {
        continue;
      }

      gltf::Image image;
      image.uri = tex_name;
      image.name = ExtractPathWithoutExt(tex_name);
      model.images.push_back(image);
    }

    WriteTexture(to_write_mats);
  }

  // Write .bin
  std::ofstream bin_out(gltf_dir + bin_name,
                        std::ios::out | std::ios::binary | std::ios::trunc);
  bin_out.write(reinterpret_cast<char*>(bin_all.data()), bin_all.size());

  // Write .gltf (json)
  gltf::WriteGltfJsonToFile(model, gltf_dir + gltf_basename + ".gltf");

  return true;
#else

  return false;
#endif
}

bool WriteGlb(Scene& scene, const std::string& glb_dir,
              const std::string& glb_name, bool is_unlit) {
#ifdef UGU_USE_JSON
  gltf::Model model;
  std::string bin_name = glb_name + ".bin";

  std::vector<std::uint8_t> bin_all;

  model.meshes.resize(scene.size());

  model.scenes[0].nodes.clear();
  model.nodes.clear();
  model.images.clear();
  model.materials.clear();
  model.accessors.clear();
  model.bufferViews.clear();
  model.buffers.resize(1);
  model.textures.clear();

  uint32_t total_mat_num = 0;
  uint32_t mat_count = 0;
  for (size_t msh_idx = 0; msh_idx < scene.size(); msh_idx++) {
    total_mat_num += static_cast<uint32_t>(scene[msh_idx]->materials().size());
  }

  model.materials.resize(total_mat_num);  // todo: update pbr params

  for (size_t msh_idx = 0; msh_idx < scene.size(); msh_idx++) {
    MeshPtr mesh = scene[msh_idx];

    model.scenes[0].nodes.push_back(static_cast<int>(msh_idx));
    ugu::gltf::Node node;
    node.mesh = static_cast<uint32_t>(msh_idx);
    node.name = std::to_string(msh_idx);  // TODO
    model.nodes.push_back(node);

    // Make .bin and update model info
    mesh->CalcStats();  // ensure min/max

    model.meshes[msh_idx].name = ExtractPathWithoutExt(glb_name);

    model.meshes[msh_idx].with_blendshapes = !mesh->blendshapes().empty();

    ugu::gltf::Primitive primitive;
    int primitive_offset = 4;  // model.meshes[msh_idx].with_blendshapes ? 5 :
                               // 4;
    if (!mesh->keyframes().empty()) {
      primitive_offset += 4;
    }
    primitive.indices = static_cast<uint32_t>(msh_idx * primitive_offset + 3);
    primitive.material = static_cast<uint32_t>(msh_idx);
    primitive.attributes["NORMAL"] =
        static_cast<uint32_t>(msh_idx * primitive_offset + 1);
    primitive.attributes["POSITION"] =
        static_cast<uint32_t>(msh_idx * primitive_offset + 0);
    primitive.attributes["TEXCOORD_0"] =
        static_cast<uint32_t>(msh_idx * primitive_offset + 2);
    model.meshes[msh_idx].primitives = {primitive};

    // Prepare blendshapes

    for (auto& p : model.meshes[msh_idx].primitives) {
      p.with_blendshapes = model.meshes[msh_idx].with_blendshapes;
    }
    for (const auto& b : mesh->blendshapes()) {
      model.meshes[msh_idx].blendshape_names.push_back(b.name);
      model.meshes[msh_idx].blendshape_weights.push_back(b.weight);
    }
    model.meshes[msh_idx].primitives[0].blendshape_num =
        static_cast<std::uint32_t>(mesh->blendshapes().size());

    for (size_t i = 0; i < mesh->materials().size(); i++) {
      model.materials[mat_count + i].name = mesh->materials()[i].name;
      model.materials[mat_count + i].is_unlit = is_unlit;
      model.materials[mat_count + i].with_alpha =
          !mesh->materials()[i].with_alpha_tex.empty();
      model.materials[mat_count + i]
          .pbrMetallicRoughness.baseColorTexture.index = mat_count;
      ugu::gltf::Texture texture;
      texture.source = mat_count;
      model.textures.push_back(texture);
      mat_count++;
    }
#if 0
        
    std::vector<ObjMaterial> to_write_mats;

    for (auto mat : mesh->materials()) {
      mat.with_alpha_texpath =  + mat.with_alpha_texname;
      mat.diffuse_texpath = gltf_dir + mat.diffuse_texname;
      to_write_mats.push_back(mat);
      std::string tex_name = mat.with_alpha_texname;
      std::string tex_path = mat.with_alpha_texpath;
      if (tex_name.empty()) {
        tex_name = mat.diffuse_texname;
        tex_path = mat.diffuse_texpath;
        // Kill with_alpha
        to_write_mats.back().with_alpha_texpath = "";
      } else {
        // Kill diffuse
        to_write_mats.back().diffuse_texpath = "";
      }
      if (tex_name.empty()) {
        continue;
      }

      gltf::Image image;
      image.uri = tex_name;
      image.name = ExtractPathWithoutExt(tex_name);
      model.images.push_back(image);
    }

    WriteTexture(to_write_mats);
#endif  // 0

    for (auto mat : mesh->materials()) {
      std::string tex_name = mat.with_alpha_texname;
      std::string tex_path = mat.with_alpha_texpath;
      if (tex_path.empty()) {
        tex_path = tex_name;
      }
      if (mat.with_alpha_compressed.empty()) {
        std::string ext = ExtractPathExt(tex_path);
        if (ext == "png") {
          mat.with_alpha_compressed = PngData(mat.with_alpha_tex);
        }
      }
      std::vector<uint8_t> tex_data = mat.with_alpha_compressed;
      if (tex_name.empty()) {
        tex_name = mat.diffuse_texname;
        tex_path = mat.diffuse_texpath;
        if (tex_path.empty()) {
          tex_path = tex_name;
        }
        if (mat.diffuse_compressed.empty()) {
          std::string ext = ExtractPathExt(tex_path);
          if (ext == "jpg" || ext == "jpeg") {
            mat.diffuse_compressed = JpgData(mat.diffuse_tex);
          } else if (ext == "png") {
            mat.diffuse_compressed = PngData(mat.diffuse_tex);
          }
        }
        tex_data = mat.diffuse_compressed;
      }
      if (tex_name.empty()) {
        continue;
      }

      gltf::Image image;
      image.is_glb = true;
      std::string ext = ExtractPathExt(tex_path);
      if (ext == "jpg" || ext == "jpeg") {
        image.mimeType = "image/jpeg";
      } else if (ext == "png") {
        image.mimeType = "image/png";
      } else {
        LOGE("ext %s is not supported\n", ext.c_str());
        continue;
      }

      image.name = ExtractPathWithoutExt(tex_name);
      if (tex_data.empty()) {
        // Read jpeg or png data
        std::ifstream ifs(tex_path, std::ios::in | std::ios::binary);
        // Get size
        ifs.seekg(0, std::ios::end);
        long long int size = ifs.tellg();
        ifs.seekg(0);
        // Read binary
        image.data.resize(size);
        ifs.read(reinterpret_cast<char*>(image.data.data()), size);
      } else {
        // Copy from memory
        image.data = std::move(tex_data);
      }

      model.images.push_back(image);
    }

    std::string bin_name = glb_name + ".bin";
    MakeGltfBinAndUpdateModel(*mesh, bin_name, true, model, bin_all, false);
  }

  uint32_t num_bv = static_cast<uint32_t>(model.bufferViews.size());
  uint32_t total_size = static_cast<uint32_t>(bin_all.size());
  uint32_t org_total_size = total_size;
  std::vector<std::vector<std::uint8_t>> bytes_list;
  if (true) {
    model.buffers[0].is_glb = true;

    for (auto& image : model.images) {
      if (image.glb_processed) {
        continue;
      }

      image.is_glb = true;
      image.glb_processed = true;
      image.bufferView = num_bv;

      // Does not need accessor for image
      // BufferView and mimeType are enough for decoding
      gltf::BufferView bv;
      bv.buffer = 0;
      // Image data should be last of buffer
      // Otherwise you may get ""ACCESSOR_TOTAL_OFFSET_ALIGNMENT" Accessor's
      // total byteOffset XXXX isn't a multiple of componentType length 4." for
      // other (e.g. vertices) bufferViews
      bv.byteLength = static_cast<int>(image.data.size());
      bv.byteOffset = static_cast<std::uint32_t>(total_size);
      total_size += bv.byteLength;
      model.bufferViews.push_back(bv);

      bytes_list.push_back(image.data);

      num_bv++;
    }

  } else {
    model.buffers[0].is_glb = false;
    model.buffers[0].uri = bin_name;
  }

  model.buffers[0].byteLength = static_cast<std::uint32_t>(total_size);

  bin_all.resize(total_size);
  size_t offset = org_total_size;
  for (auto& bytes : bytes_list) {
    bin_all.insert(bin_all.begin() + offset, bytes.begin(), bytes.end());
    offset += bytes.size();
  }

  gltf::Chunk bin_chunk{0x004E4942, bin_all};  // 0x004E4942 -> "BIN" in ASCII
  std::vector<std::uint8_t> bin_bin = gltf::ChunkToBin(bin_chunk);

  // Make json
  std::string json_string = gltf::WriteGltfJsonToString(model);

  std::vector<std::uint8_t> json_bytes(json_string.size());
  std::memcpy(json_bytes.data(), json_string.c_str(), json_string.size());
  gltf::Chunk json_chunk{0x4E4F534A,
                         json_bytes};  // 0x4E4F534A	-> "JSON" in ASCII
  std::vector<std::uint8_t> json_bin = gltf::ChunkToBin(json_chunk);

  std::uint32_t magic = 0x46546C67;  //  0x46546C67 -> "glTF"in ASCII
  std::uint32_t version = 2;
  int header_size = 12;
  std::uint32_t length = static_cast<std::uint32_t>(
      header_size + json_bin.size() + bin_bin.size());
  std::vector<std::uint8_t> header(header_size);
  std::memcpy(header.data(), &magic, 4);
  std::memcpy(header.data() + 4, &version, 4);
  std::memcpy(header.data() + 8, &length, 4);

  std::vector<std::uint8_t> combined(length);
  std::memcpy(combined.data(), header.data(), header_size);
  std::memcpy(combined.data() + header_size, json_bin.data(), json_bin.size());
  std::memcpy(combined.data() + header_size + json_bin.size(), bin_bin.data(),
              bin_bin.size());

  std::ofstream glb_out(glb_dir + glb_name,
                        std::ios::out | std::ios::binary | std::ios::trunc);
  glb_out.write(reinterpret_cast<char*>(combined.data()), length);

  return true;
#else

  return false;
#endif
}

std::tuple<int, std::vector<int>, std::vector<Eigen::Vector3f>,
           std::vector<Eigen::Vector3f>, std::vector<Eigen::Vector2f>,
           std::vector<Eigen::Vector3i>, std::vector<int>, bool, bool>
RemoveVerticesBase(const std::vector<Eigen::Vector3f>& vertices,
                   const std::vector<Eigen::Vector3i>& vertex_indices,
                   const std::vector<Eigen::Vector3f>& vertex_colors,
                   const std::vector<Eigen::Vector2f>& uv,
                   const std::vector<Eigen::Vector3i>& uv_indices,
                   const std::vector<bool>& valid_vertex_table) {
  int num_removed{0};
  std::vector<int> valid_table(vertices.size(), -1);
  std::vector<Eigen::Vector3f> valid_vertices, valid_vertex_colors;
  std::vector<Eigen::Vector2f> valid_uv;
  std::vector<Eigen::Vector3i> valid_indices;
  std::vector<int> valid_face_table(vertex_indices.size(), -1);

  if (valid_vertex_table.size() != vertices.size()) {
    ugu::LOGE("valid_vertex_table must be same size to vertices");
    return {-1,       valid_table,   valid_vertices,   valid_vertex_colors,
            valid_uv, valid_indices, valid_face_table, false,
            false};
  }

  bool with_uv = !uv.empty() && !uv_indices.empty();
  bool with_vertex_color = !vertex_colors.empty();
  int valid_count = 0;
  for (size_t i = 0; i < vertices.size(); i++) {
    if (valid_vertex_table[i]) {
      valid_table[i] = valid_count;
      valid_vertices.push_back(vertices[i]);
      if (with_uv) {
        valid_uv.push_back(uv[i]);
      }
      if (with_vertex_color) {
        valid_vertex_colors.push_back(vertex_colors[i]);
      }
      valid_count++;
    } else {
      num_removed++;
    }
  }

  int valid_face_count{0};

  for (size_t i = 0; i < vertex_indices.size(); i++) {
    Eigen::Vector3i face;
    bool valid{true};
    for (int j = 0; j < 3; j++) {
      int new_index = valid_table[vertex_indices[i][j]];
      if (new_index < 0) {
        valid = false;
        break;
      }
      face[j] = new_index;
    }
    if (!valid) {
      continue;
    }
    valid_indices.push_back(face);
    valid_face_table[i] = valid_face_count;
    valid_face_count++;
  }

  return {num_removed,         valid_table, valid_vertices,
          valid_vertex_colors, valid_uv,    valid_indices,
          valid_face_table,    with_uv,     with_vertex_color};
}

std::tuple<int, std::vector<int>, std::vector<Eigen::Vector3f>,
           std::vector<Eigen::Vector3f>, std::vector<Eigen::Vector2f>,
           std::vector<Eigen::Vector3i>, std::vector<int>, bool, bool>
RemoveUnreferencedVerticesBase(
    const std::vector<Eigen::Vector3f>& vertices,
    const std::vector<Eigen::Vector3i>& vertex_indices,
    const std::vector<Eigen::Vector3f>& vertex_colors,
    const std::vector<Eigen::Vector2f>& uv,
    const std::vector<Eigen::Vector3i>& uv_indices) {
  std::vector<bool> reference_table(vertices.size(), false);
  for (const auto& f : vertex_indices) {
    for (int i = 0; i < 3; i++) {
      reference_table[f[i]] = true;
    }
  }

  return RemoveVerticesBase(vertices, vertex_indices, vertex_colors, uv,
                            uv_indices, reference_table);
}

}  // namespace ugu
