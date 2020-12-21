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

// https://stackoverflow.com/a/34571089/5155484
// typedef unsigned char uchar;
static const std::string b =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";  //=
static std::string base64_encode_str(const std::string& in) {
  std::string out;

  int val = 0, valb = -6;
  for (std::uint8_t c : in) {
    val = (val << 8) + c;
    valb += 8;
    while (valb >= 0) {
      out.push_back(b[(val >> valb) & 0x3F]);
      valb -= 6;
    }
  }
  if (valb > -6) out.push_back(b[((val << 8) >> (valb + 8)) & 0x3F]);
  while (out.size() % 4) out.push_back('=');
  return out;
}

static std::vector<std::uint8_t> base64_encode(const std::string& in) {
  std::vector<std::uint8_t> out;

  int val = 0, valb = -6;
  for (std::uint8_t c : in) {
    val = (val << 8) + c;
    valb += 8;
    while (valb >= 0) {
      out.push_back(b[(val >> valb) & 0x3F]);
      valb -= 6;
    }
  }
  if (valb > -6) out.push_back(b[((val << 8) >> (valb + 8)) & 0x3F]);
  while (out.size() % 4) out.push_back('=');
  return out;
}

static std::string base64_decode(const std::string& in) {
  std::string out;

  std::vector<int> T(256, -1);
  for (int i = 0; i < 64; i++) T[b[i]] = i;

  int val = 0, valb = -8;
  for (std::uint8_t c : in) {
    if (T[c] == -1) break;
    val = (val << 6) + T[c];
    valb += 6;
    if (valb >= 0) {
      out.push_back(char((val >> valb) & 0xFF));
      valb -= 8;
    }
  }
  return out;
}

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
  std::string name = "untitled";
  std::string uri = "untitled.jpg";

  bool is_glb = false;
  int bufferView = 0;
  std::string mimeType = "image/jpeg";
  std::vector<std::uint8_t> data;
  int w, h;
};
void to_json(json& j, const Image& obj) {
  if (obj.is_glb) {
    j = json{{"bufferView", obj.bufferView}, {"mimeType", obj.mimeType}};
  } else {
    j = json{{"name", obj.name}, {"uri", obj.uri}};
  }
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
  bool is_glb = false;
  std::string uri = "untitled.bin";
};
void to_json(json& j, const Buffer& obj) {
  if (obj.is_glb) {
    j = json{{"byteLength", obj.byteLength}};
  } else {
    j = json{{"byteLength", obj.byteLength}, {"uri", obj.uri}};
  }
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

struct Chunk {
  // std::uint32_t length;
  std::uint32_t type;
  std::vector<std::uint8_t> data;
};

std::vector<std::uint8_t> ChunkToBin(const Chunk& chunk) {
  size_t chunk_size = 4 + 4 + chunk.data.size();
  int padding_num = 0;
  if (chunk_size % 4 != 0) {
    padding_num = 4 - (chunk_size % 4);
  }
  std::vector<std::uint8_t> combined(chunk_size);
  // IMPORTANT: add padding_num
  std::uint32_t length =
      static_cast<std::uint32_t>(chunk.data.size() + padding_num);
  std::memcpy(combined.data(), &length, 4);
  std::memcpy(combined.data() + 4, &chunk.type, 4);
  std::memcpy(combined.data() + 8, chunk.data.data(), length);

  std::uint8_t padchar = 255;  // invalid
  if (chunk.type == 0x4E4F534A) {
    // json case
    padchar = 0x20;  //' ';
  } else if (chunk.type == 0x004E4942) {
    // bin case
    padchar = 0x00;
  }

  for (int i = 0; i < padding_num; i++) {
    combined.push_back(padchar);
  }

  return combined;
};

json MakeGltfJson(const Model& model) { return json(model); }

std::string WriteGltfJsonToString(const Model& model) {
  return MakeGltfJson(model).dump(-1, ' ', true);
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
    bool is_glb, Model& model) {
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

  std::vector<std::vector<std::uint8_t>> bytes_list = {vert_bytes, nor_bytes,
                                                       uv_bytes, index_bytes};

  model.buffers.resize(1);

  if (is_glb) {
    model.buffers[0].is_glb = true;

    int num_bv = 4;
    for (auto& image : model.images) {
      image.is_glb = true;
      image.bufferView = num_bv;

      // Does not need accessor for image
      // BufferView and mimeType are enough for decoding
      BufferView bv;
      bv.buffer = 0;
      bv.byteLength = static_cast<int>(image.data.size());
      bv.byteOffset = total_size;
      total_size += bv.byteLength;
      model.bufferViews.push_back(bv);

      bytes_list.push_back(image.data);

      num_bv++;
    }

  } else {
    model.buffers[0].is_glb = false;
    model.buffers[0].uri = bin_name;
  }

  model.buffers[0].byteLength = total_size;

  std::vector<std::uint8_t> combined_bytes;
  combined_bytes.reserve(total_size);
  for (auto& bytes : bytes_list) {
    combined_bytes.insert(combined_bytes.end(), bytes.begin(), bytes.end());
  }

  return combined_bytes;
}

std::vector<std::uint8_t> MakeGltfBinAndUpdateModel(const ugu::Mesh& mesh,
                                                    const std::string& bin_name,
                                                    bool is_glb, Model& model) {
  // Flip v
  auto gltf_uvs = mesh.uv();
  for (auto& uv : gltf_uvs) {
    uv[1] = 1.f - uv[1];
  }
  return MakeGltfBinAndUpdateModel(
      mesh.vertices(), mesh.stats().bb_max, mesh.stats().bb_min, mesh.normals(),
      gltf_uvs, mesh.vertex_indices(), bin_name, is_glb, model);
}

}  // namespace gltf
}  // namespace ugu

#endif

#endif