#ifndef UGU_SRC_GLTF_H
#define UGU_SRC_GLTF_H

#ifdef UGU_USE_JSON

#include "ugu/mesh.h"

#include <fstream>
#include <iomanip>
#include <unordered_map>

#include "nlohmann/json.hpp"

namespace ugu {
namespace gltf {

using namespace nlohmann;

struct Asset {
  std::string generator = "unknown";
  std::string version = "2.0";
};
void to_json(json& j, const Asset& obj) {
  j = json{{"generator", obj.generator}, {"version", obj.version}};
}

struct Scene {
  std::string name = "Scene";
  std::vector<int> nodes = {0};
};
void to_json(json& j, const Scene& obj) {
  j = json{{"name", obj.name}, {"nodes", obj.nodes}};
}

struct Node {
  int mesh = 0;
  std::string name = "unknown";
  // std::array<float, 4> rotation;
};
void to_json(json& j, const Node& obj) {
  j = json{{"mesh", obj.mesh}, {"name", obj.name}};
}

struct TextureInfo {
  int index = 0;
  int texCoord = 0;
};
void to_json(json& j, const TextureInfo& obj) {
  j = json{{"index", obj.index}, {"texCoord", obj.texCoord}};
}

struct PbrMetallicRoughness {
  TextureInfo baseColorTexture;
  float metallicFactor = 0.f;
  float roughnessFactor = 0.9057191014289856f;
};
void to_json(json& j, const PbrMetallicRoughness& obj) {
  j = json{{"baseColorTexture", obj.baseColorTexture},
           {"metallicFactor", obj.metallicFactor},
           {"roughnessFactor", obj.roughnessFactor}};
}

struct Material {
  bool doubleSided = true;
  std::array<float, 3> emissiveFactor = {0, 0, 0};
  std::string name = "material_000";
  PbrMetallicRoughness pbrMetallicRoughness;
};
void to_json(json& j, const Material& obj) {
  j = json{{"doubleSided", obj.doubleSided},
           {"emissiveFactor", obj.emissiveFactor},
           {"name", obj.name},
           {"pbrMetallicRoughness", obj.pbrMetallicRoughness}};
}

// using AttributeIndexPair = std::pair<std::string, int>;

#if 0
				struct PrimitiveAttribute {
  int
};
#endif  // 0

struct Primitive {
  std::unordered_map<std::string, int> attributes = {
      {"POSITION", 0}, {"NORMAL", 1}, {"TEXCOORD_0", 2}};
  int indices = 3;
  int material = 0;
};
void to_json(json& j, const Primitive& obj) {
  j = json{{"attributes", obj.attributes},
           {"indices", obj.indices},
           {"material", obj.material}};
}

struct Mesh {
  std::string name = "unknown";
  std::vector<Primitive> primitives = {Primitive()};
};
void to_json(json& j, const Mesh& obj) {
  j = json{{"name", obj.name}, {"primitives", obj.primitives}};
}

struct Texture {
  int source = 0;
};
void to_json(json& j, const Texture& obj) { j = json{{"source", obj.source}}; }

struct Image {
  // std::string mineType = "image/jpeg";
  std::string name = "untitled";
  std::string uri = "untitled.jpg";
};
void to_json(json& j, const Image& obj) {
  j = json{//{"mineType", obj.mineType},
           {"name", obj.name},
           {"uri", obj.uri}};
}

struct Accessor {
  int bufferView = 0;
  int componentType = 5126;
  int count = -1;
  std::string type;

  bool write_minmax = false;
  std::array<float, 3> max;
  std::array<float, 3> min;
};
void to_json(json& j, const Accessor& obj) {
  j = json{{"bufferView", obj.bufferView},
           {"componentType", obj.componentType},
           {"count", obj.count},
           {"type", obj.type}};
  if (obj.write_minmax) {
    j["max"] = obj.max;
    j["min"] = obj.min;
  }
}

std::vector<Accessor> GenDefaultAccessors() {
  std::vector<Accessor> accessors(4);
  // vertices
  accessors[0].write_minmax = true;
  accessors[0].type = "VEC3";

  // normal
  accessors[1].type = "VEC3";

  // texcoords
  accessors[2].type = "VEC2";

  // indices
  accessors[3].type = "SCALAR";

  return accessors;
}

struct BufferView {
  int buffer = 0;
  int byteLength = 0;
  int byteOffset = 0;
};
void to_json(json& j, const BufferView& obj) {
  j = json{{"buffer", obj.buffer},
           {"byteLength", obj.byteLength},
           {"byteOffset", obj.byteOffset}};
}

struct Buffer {
  int byteLength = 0;
  std::string uri = "untitled.bin";
};
void to_json(json& j, const Buffer& obj) {
  j = json{{"byteLength", obj.byteLength}, {"uri", obj.uri}};
}

struct Model {
  Asset asset;
  int scene = 0;
  std::vector<Scene> scenes = {Scene()};
  std::vector<Node> nodes = {Node()};
  std::vector<Material> materials = {Material()};
  std::vector<gltf::Mesh> meshes = {gltf::Mesh()};
  std::vector<Texture> textures = {Texture()};
  std::vector<Image> images = {Image()};
  std::vector<Accessor> accessors = GenDefaultAccessors();
  std::vector<BufferView> bufferViews = {BufferView(), BufferView(),
                                         BufferView(), BufferView()};
  std::vector<Buffer> buffers = {Buffer()};
};
void to_json(json& j, const Model& obj) {
  j = json{{"asset", obj.asset},         {"scene", obj.scene},
           {"scenes", obj.scenes},       {"nodes", obj.nodes},
           {"materials", obj.materials}, {"meshes", obj.meshes},
           {"textures", obj.textures},   {"images", obj.images},
           {"accessors", obj.accessors}, {"bufferViews", obj.bufferViews},
           {"buffers", obj.buffers}};
}

json MakeGltfJson(const Model& model) { return json(model); }

std::string WriteGltfJsonToString(const Model& model) {
  return MakeGltfJson(model).dump();
}

bool WriteGltfJsonToFile(const Model& model, const std::string& path) {
  std::ofstream o(path);
  o << std::setw(4) << MakeGltfJson(model) << std::endl;
  return true;
}

std::vector<std::uint8_t> MakeGltfBinAndUpdateModel(
    const std::vector<Eigen::Vector3f>& vertices,
    const Eigen::Vector3f& vert_max, const Eigen::Vector3f& vert_min,
    const std::vector<Eigen::Vector3f>& normals,
    const std::vector<Eigen::Vector2f>& uvs,
    const std::vector<Eigen::Vector3i>& indices, const std::string& bin_name,
    Model& model) {
  size_t total_size = 0;
  model.accessors.resize(4);
  model.bufferViews.resize(4);

  auto& vert_acc = model.accessors[0];
  vert_acc.bufferView = 0;
  vert_acc.componentType = 5126;
  vert_acc.count = static_cast<int>(vertices.size());
  vert_acc.write_minmax = true;
  vert_acc.max = {vert_max[0], vert_max[1], vert_max[2]};
  vert_acc.min = {vert_min[0], vert_min[1], vert_min[2]};
  vert_acc.type = "VEC3";
  size_t vert_size = vertices.size() * sizeof(float) * 3;
  std::vector<std::uint8_t> vert_bytes(vert_size);
  std::memcpy(vert_bytes.data(), vertices.data(), vert_size);
  auto& vert_bview = model.bufferViews[0];
  vert_bview.buffer = 0;
  vert_bview.byteLength = vert_size;
  vert_bview.byteOffset = total_size;
  total_size += vert_size;

  auto& nor_acc = model.accessors[1];
  nor_acc.bufferView = 1;
  nor_acc.componentType = 5126;
  nor_acc.count = static_cast<int>(normals.size());
  nor_acc.type = "VEC3";
  size_t nor_size = normals.size() * sizeof(float) * 3;
  std::vector<std::uint8_t> nor_bytes(nor_size);
  std::memcpy(nor_bytes.data(), normals.data(), nor_size);
  auto& nor_bview = model.bufferViews[1];
  nor_bview.buffer = 0;
  nor_bview.byteLength = nor_size;
  nor_bview.byteOffset = total_size;
  total_size += nor_size;

  auto& uv_acc = model.accessors[2];
  uv_acc.bufferView = 2;
  uv_acc.componentType = 5126;
  uv_acc.count = static_cast<int>(uvs.size());
  uv_acc.type = "VEC2";
  size_t uv_size = uvs.size() * sizeof(float) * 2;
  std::vector<std::uint8_t> uv_bytes(uv_size);
  std::memcpy(uv_bytes.data(), uvs.data(), uv_size);
  auto& uv_bview = model.bufferViews[2];
  uv_bview.buffer = 0;
  uv_bview.byteLength = uv_size;
  uv_bview.byteOffset = total_size;
  total_size += uv_size;

  auto& index_acc = model.accessors[3];
  index_acc.bufferView = 3;
  index_acc.componentType = 5125;
  index_acc.count = static_cast<int>(indices.size() * 3);
  index_acc.type = "SCALAR";
  size_t index_size = indices.size() * sizeof(int) * 3;
  std::vector<std::uint8_t> index_bytes(index_size);
  std::memcpy(index_bytes.data(), indices.data(), index_size);
  auto& index_bview = model.bufferViews[3];
  index_bview.buffer = 0;
  index_bview.byteLength = index_size;
  index_bview.byteOffset = total_size;
  total_size += index_size;

  model.buffers.resize(1);
  model.buffers[0].byteLength = total_size;
  model.buffers[0].uri = bin_name;

  std::vector<std::uint8_t> bytes;
  bytes.reserve(total_size);
  bytes.insert(bytes.end(), vert_bytes.begin(), vert_bytes.end());
  bytes.insert(bytes.end(), nor_bytes.begin(), nor_bytes.end());
  bytes.insert(bytes.end(), uv_bytes.begin(), uv_bytes.end());
  bytes.insert(bytes.end(), index_bytes.begin(), index_bytes.end());

  return bytes;
}

std::vector<std::uint8_t> MakeGltfBinAndUpdateModel(const ugu::Mesh& mesh,
                                                    const std::string& bin_name,
                                                    Model& model) {
  return MakeGltfBinAndUpdateModel(
      mesh.vertices(), mesh.stats().bb_max, mesh.stats().bb_min, mesh.normals(),
      mesh.uv(), mesh.vertex_indices(), bin_name, model);
}

}  // namespace gltf
}  // namespace ugu

#endif

#endif