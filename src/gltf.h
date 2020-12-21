#ifndef UGU_SRC_GLTF_H
#define UGU_SRC_GLTF_H

#ifdef UGU_USE_JSON
#include <fstream>
#include <iomanip>
#include <unordered_map>

#include "nlohmann/json.hpp"

namespace ugu {
namespace gltf {

using namespace nlohmann;

struct Asset {
  std::string generator = "unknown";
  std::string version = "1.0";
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
  std::string mineType = "image/jpeg";
  std::string name = "untitled";
  std::string uri = "untitled.jpg";
};
void to_json(json& j, const Image& obj) {
  j = json{{"mineType", obj.mineType}, {"name", obj.name}, {"uri", obj.uri}};
}

struct Accessor {
  int bufferView = 0;
  int componetType = 5126;
  int count = -1;
  std::string type;

  bool write_minmax = false;
  std::array<float, 3> max;
  std::array<float, 3> min;
};
void to_json(json& j, const Accessor& obj) {
  j = json{{"bufferView", obj.bufferView},
           {"componetType", obj.componetType},
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
  j = json{{"asset", obj.asset},
           {"scene", obj.scene},
           {"scenes", obj.scenes},
           {"materials", obj.materials},
           {"meshes", obj.meshes},
           {"textures", obj.textures},
           {"images", obj.images},
           {"accessors", obj.accessors},
           {"bufferViews", obj.bufferViews},
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

}  // namespace gltf
}  // namespace ugu

#endif

#endif