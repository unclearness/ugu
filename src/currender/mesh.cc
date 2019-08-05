/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "currender/mesh.h"

#include <fstream>
#include <random>
#include <sstream>

#ifdef CURRENDER_USE_TINYOBJLOADER
#include "tinyobjloader/tiny_obj_loader.h"
#endif

namespace {
template <typename T>
void CopyVec(const std::vector<T>& src, std::vector<T>* dst) {
  dst->clear();
  std::copy(src.begin(), src.end(), std::back_inserter(*dst));
}

std::vector<std::string> Split(const std::string& s, char delim) {
  std::vector<std::string> elems;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    if (!item.empty()) {
      elems.push_back(item);
    }
  }
  return elems;
}

inline std::string ExtractPathWithoutExt(const std::string& fn) {
  std::string::size_type pos;
  if ((pos = fn.find_last_of(".")) == std::string::npos) {
    return fn;
  }

  return fn.substr(0, pos);
}

inline std::string ReplaceExtention(const std::string& path,
                                    const std::string& ext) {
  return ExtractPathWithoutExt(path) + ext;
}

#ifdef CURRENDER_USE_STB
bool WriteMtl(const std::string& path,
              const std::vector<currender::ObjMaterial>& materials,
              bool write_texture) {
  std::ofstream ofs(path);
  if (ofs.fail()) {
    currender::LOGE("couldn't open mtl path: %s\n", path.c_str());
    return false;
  }

  for (size_t i = 0; i < materials.size(); i++) {
    const currender::ObjMaterial& material = materials[i];
    ofs << material.ToString();
    if (i != materials.size() - 1) {
      ofs << '\n';
    }
  }
  ofs.close();

  // write texture
  bool ret{true};
  if (write_texture) {
    for (size_t i = 0; i < materials.size(); i++) {
      const currender::ObjMaterial& material = materials[i];
      bool ret_write = material.diffuse_tex.WritePng(material.diffuse_texpath);
      if (ret) {
        ret = ret_write;
      }
    }
  }

  return ret;
}
#endif

}  // namespace

namespace currender {

std::string ObjMaterial::ToString() const {
  std::stringstream ss;

  ss << "newmtl " << name << '\n'
     << "Ka " << ambient[0] << " " << ambient[1] << " " << ambient[2] << '\n'
     << "Ka " << diffuse[0] << " " << diffuse[1] << " " << diffuse[2] << '\n'
     << "Ka " << specular[0] << " " << specular[1] << " " << specular[2] << '\n'
     << "Tr " << 1.0f - dissolve << '\n'
     << "illum " << illum << '\n'
     << "Ns " << shininess << '\n';
  if (!diffuse_texname.empty()) {
    ss << "map_Kd " << diffuse_texname << std::endl;
  }

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

void Mesh::CalcStats() {
  stats_.bb_min = Eigen::Vector3f(std::numeric_limits<float>::max(),
                                  std::numeric_limits<float>::max(),
                                  std::numeric_limits<float>::max());
  stats_.bb_max = Eigen::Vector3f(std::numeric_limits<float>::lowest(),
                                  std::numeric_limits<float>::lowest(),
                                  std::numeric_limits<float>::lowest());

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
    normals_[i] /= static_cast<float>(add_count[i]);
    normals_[i].normalize();
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
  int max_id = *std::max_element(material_ids.begin(), material_ids.end());
  if (max_id < 0) {
    LOGE("material id must be positive\n");
    return false;
  }

  CopyVec(material_ids, &material_ids_);

  face_indices_per_material_.resize(max_id + 1);
  for (int i = 0; i < static_cast<int>(material_ids_.size()); i++) {
    face_indices_per_material_[material_ids_[i]].push_back(i);
  }

  return true;
}

bool Mesh::set_materials(const std::vector<ObjMaterial>& materials) {
  CopyVec(materials, &materials_);
  return true;
}

#ifdef CURRENDER_USE_TINYOBJLOADER
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

  materials_.resize(materials.size());
  for (size_t i = 0; i < materials.size(); i++) {
    materials_[i].name = materials[i].name;
    std::copy(std::begin(materials[i].ambient), std::end(materials[i].ambient),
              materials_[i].ambient.begin());
    std::copy(std::begin(materials[i].diffuse), std::end(materials[i].diffuse),
              materials_[i].diffuse.begin());
    std::copy(std::begin(materials[i].specular),
              std::end(materials[i].specular), materials_[i].specular.begin());
    materials_[i].shininess = materials[i].shininess;
    materials_[i].dissolve = materials[i].dissolve;
    materials_[i].illum = materials[i].illum;

    materials_[i].diffuse_texname = materials[i].diffuse_texname;
    materials_[i].diffuse_texpath = mtl_dir + materials_[i].diffuse_texname;
    std::ifstream ifs(materials_[i].diffuse_texpath);
    if (ifs.is_open()) {
#ifdef CURRENDER_USE_STB
      ret = materials_[i].diffuse_tex.Load(materials_[i].diffuse_texpath);
#else
      LOGW("define CURRENDER_USE_STB to load diffuse texture.\n");
#endif
    } else {
      LOGW("diffuse texture doesn't exist %s\n",
           materials_[i].diffuse_texpath.c_str());
    }
  }

  face_indices_per_material_.resize(materials.size());
  for (int i = 0; i < static_cast<int>(material_ids_.size()); i++) {
    face_indices_per_material_[material_ids_[i]].push_back(i);
  }

  return ret;
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

  bool has_vertex_normal = !normals_.empty();
  if (has_vertex_normal) {
    assert(vertices_.size() == normals_.size());
  }
  bool has_vertex_color = !vertex_colors_.empty();
  if (has_vertex_color) {
    assert(vertices_.size() == vertex_colors_.size());
  }

  ofs << "ply" << std::endl;
  ofs << "format ascii 1.0" << std::endl;
  ofs << "element vertex " + std::to_string(vertices_.size()) << std::endl;
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
  ofs << "element face " + std::to_string(vertex_indices_.size()) << std::endl;
  ofs << "property list uchar int vertex_indices" << std::endl;
  ofs << "end_header" << std::endl;

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

#ifdef CURRENDER_USE_STB
bool Mesh::WriteObj(const std::string& obj_dir, const std::string& obj_basename,
                    const std::string& mtl_basename) {
  std::string mtl_name = mtl_basename + ".mtl";
  if (mtl_basename.empty()) {
    mtl_name = obj_basename + ".mtl";
  }
  std::string mtl_path = obj_dir + "/" + mtl_name;

  std::string obj_path = obj_dir + "/" + obj_basename + ".obj";

  // write obj
  {
    std::ofstream ofs(obj_path);
    if (ofs.fail()) {
      LOGE("couldn't open obj path: %s\n", obj_path.c_str());
      return false;
    }

    ofs << "mtllib " << mtl_name << std::endl << std::endl;

    // vertices
    for (const auto& v : vertices_) {
      ofs << "v " << v.x() << " " << v.y() << " " << v.z() << " 1.0"
          << std::endl;
    }

    // uv
    for (const auto& vt : uv_) {
      ofs << "vt " << vt.x() << " " << vt.y() << " 0" << std::endl;
    }

    // vertex normals
    for (const auto& vn : normals_) {
      ofs << "vn " << vn.x() << " " << vn.y() << " " << vn.z() << std::endl;
    }

    // indices by material (group)
    // CAUTION: This breaks original face indices
    bool write_uv_indices = !uv_indices_.empty();
    bool write_normal_indices = !normal_indices_.empty();
#if 1
    for (size_t k = 0; k < face_indices_per_material_.size(); k++) {
      ofs << "usemtl " << materials_[k].name << "\n";
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
          ofs << "/" << std::to_string(normal_indices_[f_idx][j] + 1);
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
      ofs << std::endl;
    }
#endif

    ofs.close();
  }

  // update texture path
  for (auto& material : materials_) {
    // replace extention with .png
    material.diffuse_texname =
        ReplaceExtention(material.diffuse_texname, ".png");

    // update path
    material.diffuse_texpath = obj_dir + "/" + material.diffuse_texname;
  }

  // write mtl
  WriteMtl(mtl_path, materials_, true);

  return true;
}
#endif

std::shared_ptr<Mesh> MakeCube(const Eigen::Vector3f& length,
                               const Eigen::Matrix3f& R,
                               const Eigen::Vector3f& t) {
  std::shared_ptr<Mesh> cube(new Mesh);
  std::vector<Eigen::Vector3f> vertices(24);
  std::vector<Eigen::Vector3i> vertex_indices(12);
  std::vector<Eigen::Vector3f> vertex_colors(24);

  const float h_x = length.x() / 2;
  const float h_y = length.y() / 2;
  const float h_z = length.z() / 2;

  vertices[0] = Eigen::Vector3f(-h_x, h_y, -h_z);
  vertices[1] = Eigen::Vector3f(h_x, h_y, -h_z);
  vertices[2] = Eigen::Vector3f(h_x, h_y, h_z);
  vertices[3] = Eigen::Vector3f(-h_x, h_y, h_z);
  vertex_indices[0] = Eigen::Vector3i(0, 2, 1);
  vertex_indices[1] = Eigen::Vector3i(0, 3, 2);

  vertices[4] = Eigen::Vector3f(-h_x, -h_y, -h_z);
  vertices[5] = Eigen::Vector3f(h_x, -h_y, -h_z);
  vertices[6] = Eigen::Vector3f(h_x, -h_y, h_z);
  vertices[7] = Eigen::Vector3f(-h_x, -h_y, h_z);
  vertex_indices[2] = Eigen::Vector3i(4, 5, 6);
  vertex_indices[3] = Eigen::Vector3i(4, 6, 7);

  vertices[8] = vertices[1];
  vertices[9] = vertices[2];
  vertices[10] = vertices[6];
  vertices[11] = vertices[5];
  vertex_indices[4] = Eigen::Vector3i(8, 9, 10);
  vertex_indices[5] = Eigen::Vector3i(8, 10, 11);

  vertices[12] = vertices[0];
  vertices[13] = vertices[3];
  vertices[14] = vertices[7];
  vertices[15] = vertices[4];
  vertex_indices[6] = Eigen::Vector3i(12, 14, 13);
  vertex_indices[7] = Eigen::Vector3i(12, 15, 14);

  vertices[16] = vertices[0];
  vertices[17] = vertices[1];
  vertices[18] = vertices[5];
  vertices[19] = vertices[4];
  vertex_indices[8] = Eigen::Vector3i(16, 17, 18);
  vertex_indices[9] = Eigen::Vector3i(16, 18, 19);

  vertices[20] = vertices[3];
  vertices[21] = vertices[2];
  vertices[22] = vertices[6];
  vertices[23] = vertices[7];
  vertex_indices[10] = Eigen::Vector3i(20, 22, 21);
  vertex_indices[11] = Eigen::Vector3i(20, 23, 22);

  // set default color
  for (int i = 0; i < 24; i++) {
    vertex_colors[i][0] = (-vertices[i][0] + h_x) / length.x() * 255;
    vertex_colors[i][1] = (-vertices[i][1] + h_y) / length.y() * 255;
    vertex_colors[i][2] = (-vertices[i][2] + h_z) / length.z() * 255;
  }

  cube->set_vertices(vertices);
  cube->set_vertex_indices(vertex_indices);
  cube->set_vertex_colors(vertex_colors);

  cube->Transform(R, t);

  cube->CalcNormal();

  return cube;
}

std::shared_ptr<Mesh> MakeCube(const Eigen::Vector3f& length) {
  const Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
  const Eigen::Vector3f t(0.0f, 0.0f, 0.0f);
  return MakeCube(length, R, t);
}

std::shared_ptr<Mesh> MakeCube(float length, const Eigen::Matrix3f& R,
                               const Eigen::Vector3f& t) {
  Eigen::Vector3f length_xyz{length, length, length};
  return MakeCube(length_xyz, R, t);
}

std::shared_ptr<Mesh> MakeCube(float length) {
  const Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
  const Eigen::Vector3f t(0.0f, 0.0f, 0.0f);
  return MakeCube(length, R, t);
}

void SetRandomVertexColor(std::shared_ptr<Mesh> mesh, int seed) {
  std::mt19937 mt(seed);
  std::uniform_int_distribution<int> random_color(0, 255);

  std::vector<Eigen::Vector3f> vertex_colors(mesh->vertices().size());
  for (auto& vc : vertex_colors) {
    vc[0] = static_cast<float>(random_color(mt));
    vc[1] = static_cast<float>(random_color(mt));
    vc[2] = static_cast<float>(random_color(mt));
  }

  mesh->set_vertex_colors(vertex_colors);
}

}  // namespace currender
