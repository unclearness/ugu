#ifndef UGU_SRC_GLTF_H
#define UGU_SRC_GLTF_H

#ifdef UGU_USE_JSON

#include <fstream>
#include <iomanip>
#include <unordered_map>

#include "nlohmann/json.hpp"
#include "ugu/mesh.h"

namespace ugu {
namespace gltf {

using namespace nlohmann;

void GetVertexMinMax(const std::vector<Eigen::Vector3f>& vertices,
                     Eigen::Vector3f& v_min, Eigen::Vector3f& v_max) {
  v_min = Eigen::Vector3f(std::numeric_limits<float>::max(),
                          std::numeric_limits<float>::max(),
                          std::numeric_limits<float>::max());
  v_max = Eigen::Vector3f(std::numeric_limits<float>::lowest(),
                          std::numeric_limits<float>::lowest(),
                          std::numeric_limits<float>::lowest());

  if (vertices.empty()) {
    return;
  }

  for (const auto& v : vertices) {
    for (int i = 0; i < 3; i++) {
      if (v[i] < v_min[i]) {
        v_min[i] = v[i];
      }

      if (v_max[i] < v[i]) {
        v_max[i] = v[i];
      }
    }
  }
}

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
  std::uint32_t mesh = 0;
  std::string name = "unknown";
  // std::array<float, 4> rotation;
};
void to_json(json& j, const Node& obj) {
  j = json{{"mesh", obj.mesh}, {"name", obj.name}};
}

struct TextureInfo {
  std::uint32_t index = 0;
  std::uint32_t texCoord = 0;
};
void to_json(json& j, const TextureInfo& obj) {
  j = json{{"index", obj.index}, {"texCoord", obj.texCoord}};
}

struct PbrMetallicRoughness {
  TextureInfo baseColorTexture;
  float metallicFactor = 0.f;
  float roughnessFactor = 1.f;
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

  bool is_unlit = false;

  bool with_alpha = false;
  float alphaCutoff = 0.1f;
  std::string alphaMode = "MASK";
};
void to_json(json& j, const Material& obj) {
  j = json{{"doubleSided", obj.doubleSided},
           {"emissiveFactor", obj.emissiveFactor},
           {"name", obj.name},
           {"pbrMetallicRoughness", obj.pbrMetallicRoughness}};
  if (obj.with_alpha) {
    j["alphaCutoff"] = obj.alphaCutoff;
    j["alphaMode"] = obj.alphaMode;
  }

  if (obj.is_unlit) {
    j["extensions"] = json();
    j["extensions"]["KHR_materials_unlit"] = json({});
  }
}

struct Primitive {
  std::unordered_map<std::string, std::uint32_t> attributes = {
      {"POSITION", 0}, {"NORMAL", 1}, {"TEXCOORD_0", 2}};
  std::uint32_t indices = 3;
  std::uint32_t material = 0;

  bool with_blendshapes = false;
  int blendshape_num = 0;
};
void to_json(json& j, const Primitive& obj) {
  j = json{{"attributes", obj.attributes},
           {"indices", obj.indices},
           {"material", obj.material}};

  if (obj.with_blendshapes) {
    std::vector<std::unordered_map<std::string, std::uint32_t>> targets;
    int offset = 4;
    for (int i = 0; i < obj.blendshape_num; i++) {
      std::unordered_map<std::string, std::uint32_t> target = {
          {"POSITION", offset + i * 2}, {"NORMAL", offset + i * 2 + 1}};
      targets.push_back(target);
    }
    j["targets"] = targets;
  }
}

struct Mesh {
  std::string name = "unknown";
  std::vector<Primitive> primitives = {Primitive()};

  bool with_blendshapes = false;
  std::vector<std::string> blendshape_names;
  std::vector<float> blendshape_weights;
};
void to_json(json& j, const Mesh& obj) {
  j = json{{"name", obj.name}, {"primitives", obj.primitives}};

  if (obj.with_blendshapes) {
    json targetNames = json{{"targetNames", obj.blendshape_names}};
    j["extras"] = targetNames;
    j["weights"] = obj.blendshape_weights;
  }
}

struct Texture {
  std::uint32_t source = 0;
};
void to_json(json& j, const Texture& obj) { j = json{{"source", obj.source}}; }

struct Image {
  std::string name = "untitled";
  std::string uri = "untitled.jpg";

  bool is_glb = false;
  bool glb_processed = false;
  std::uint32_t bufferView = 0;
  std::string mimeType = "image/jpeg";
  std::vector<std::uint8_t> data;
};
void to_json(json& j, const Image& obj) {
  if (obj.is_glb) {
    j = json{{"bufferView", obj.bufferView}, {"mimeType", obj.mimeType}};
  } else {
    j = json{{"name", obj.name}, {"uri", obj.uri}};
  }
}

struct Target {
  int node = -1;
  std::string path = "";
};
void to_json(json& j, const Target& obj) {
  j = json{{"node", obj.node}, {"path", obj.path}};
}

struct Channel {
  int sampler = -1;
  Target target;
};
void to_json(json& j, const Channel& obj) {
  j = json{{"sampler", obj.sampler}, {"target", obj.target}};
}

struct Sampler {
  int input = -1;
  std::string interpolation = "";
  int output = -1;
};
void to_json(json& j, const Sampler& obj) {
  j = json{{"input", obj.input},
           {"interpolation", obj.interpolation},
           {"output", obj.output}};
}

struct Animation {
  std::string name = "Animation";
  std::vector<Channel> channels;
  std::vector<Sampler> samplers;
};
void to_json(json& j, const Animation& obj) {
  j = json{{"name", obj.name},
           {"channels", obj.channels},
           {"samplers", obj.samplers}};
}

struct Accessor {
  std::uint32_t bufferView = 0;
  std::uint32_t componentType = 5126;
  std::uint32_t count = 1;
  std::string type;

  bool write_minmax = false;
  std::vector<float> max{std::numeric_limits<float>::lowest(),
                         std::numeric_limits<float>::lowest(),
                         std::numeric_limits<float>::lowest()};
  std::vector<float> min{std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::max()};
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
  std::uint32_t buffer = 0;
  std::uint32_t byteLength = 1;
  std::uint32_t byteOffset = 0;
};
void to_json(json& j, const BufferView& obj) {
  j = json{{"buffer", obj.buffer},
           {"byteLength", obj.byteLength},
           {"byteOffset", obj.byteOffset}};
}

struct Buffer {
  std::uint32_t byteLength = 1;
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
  std::vector<gltf::Animation> animations = {};
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

  if (obj.materials[0].is_unlit) {
    j["extensionsUsed"] = {"KHR_materials_unlit"};
  }

  if (!obj.animations.empty()) {
    j["animations"] = obj.animations;
  }
}

struct Chunk {
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

void MakeGltfBinAndUpdateModel(
    const std::vector<Eigen::Vector3f>& vertices,
    const Eigen::Vector3f& vert_max, const Eigen::Vector3f& vert_min,
    const std::vector<Eigen::Vector3f>& normals,
    const std::vector<Eigen::Vector2f>& uvs,
    const std::vector<Eigen::Vector3i>& indices, const std::string& bin_name,
    bool is_glb,
    const std::vector<std::vector<Eigen::Vector3f>>& blendshape_vertices,
    const std::vector<std::vector<Eigen::Vector3f>>& blendshape_normals,
    //  bool with_alpha,
    std::map<float, AnimKeyframe> keyframes, AnimInterp anim_interp,
    Model& model, std::vector<std::uint8_t>& combined_bytes,
    bool process_glb_images = true) {
  uint32_t org_total_size = static_cast<uint32_t>(combined_bytes.size());
  uint32_t total_size = org_total_size;
  uint32_t org_num_accs = static_cast<uint32_t>(model.accessors.size());
  uint32_t org_num_bv = static_cast<uint32_t>(model.bufferViews.size());
  model.accessors.resize(org_num_accs + 4);
  model.bufferViews.resize(org_num_bv + 4);

  auto& vert_acc = model.accessors[org_num_accs + 0];
  vert_acc.bufferView = org_num_bv + 0;
  vert_acc.componentType = 5126;
  vert_acc.count = static_cast<int>(vertices.size());
  vert_acc.write_minmax = true;
  vert_acc.max = {vert_max[0], vert_max[1], vert_max[2]};
  vert_acc.min = {vert_min[0], vert_min[1], vert_min[2]};
  vert_acc.type = "VEC3";
  size_t vert_size = vertices.size() * sizeof(float) * 3;
  std::vector<std::uint8_t> vert_bytes(vert_size);
  std::memcpy(vert_bytes.data(), vertices.data(), vert_size);
  auto& vert_bview = model.bufferViews[org_num_bv + 0];
  vert_bview.buffer = 0;
  vert_bview.byteLength = static_cast<std::uint32_t>(vert_size);
  vert_bview.byteOffset = static_cast<std::uint32_t>(total_size);
  total_size += static_cast<uint32_t>(vert_size);

  auto& nor_acc = model.accessors[org_num_accs + 1];
  nor_acc.bufferView = org_num_bv + 1;
  nor_acc.componentType = 5126;
  nor_acc.count = static_cast<int>(normals.size());
  nor_acc.type = "VEC3";
  size_t nor_size = normals.size() * sizeof(float) * 3;
  std::vector<std::uint8_t> nor_bytes(nor_size);
  std::memcpy(nor_bytes.data(), normals.data(), nor_size);
  auto& nor_bview = model.bufferViews[org_num_bv + 1];
  nor_bview.buffer = 0;
  nor_bview.byteLength = static_cast<std::uint32_t>(nor_size);
  nor_bview.byteOffset = static_cast<std::uint32_t>(total_size);
  total_size += static_cast<uint32_t>(nor_size);

  auto& uv_acc = model.accessors[org_num_accs + 2];
  uv_acc.bufferView = org_num_bv + 2;
  uv_acc.componentType = 5126;
  uv_acc.count = static_cast<int>(uvs.size());
  uv_acc.type = "VEC2";
  size_t uv_size = uvs.size() * sizeof(float) * 2;
  std::vector<std::uint8_t> uv_bytes(uv_size);
  std::memcpy(uv_bytes.data(), uvs.data(), uv_size);
  auto& uv_bview = model.bufferViews[org_num_bv + 2];
  uv_bview.buffer = 0;
  uv_bview.byteLength = static_cast<std::uint32_t>(uv_size);
  uv_bview.byteOffset = static_cast<std::uint32_t>(total_size);
  total_size += static_cast<uint32_t>(uv_size);

  auto& index_acc = model.accessors[org_num_accs + 3];
  index_acc.bufferView = org_num_bv + 3;
  index_acc.componentType = 5125;
  index_acc.count = static_cast<int>(indices.size() * 3);
  index_acc.type = "SCALAR";
  size_t index_size = indices.size() * sizeof(int) * 3;
  std::vector<std::uint8_t> index_bytes(index_size);
  std::memcpy(index_bytes.data(), indices.data(), index_size);
  auto& index_bview = model.bufferViews[org_num_bv + 3];
  index_bview.buffer = 0;
  index_bview.byteLength = static_cast<std::uint32_t>(index_size);
  index_bview.byteOffset = static_cast<std::uint32_t>(total_size);
  total_size += static_cast<uint32_t>(index_size);

  std::vector<std::vector<std::uint8_t>> bytes_list = {vert_bytes, nor_bytes,
                                                       uv_bytes, index_bytes};

  // model.buffers.resize(1);

  uint32_t num_bv = org_num_bv + 3 + 1;
  if (!blendshape_vertices.empty() &&
      blendshape_vertices.size() == blendshape_normals.size()) {
    for (size_t i = 0; i < blendshape_vertices.size(); i++) {
      const auto& v = blendshape_vertices[i];
      const auto& n = blendshape_normals[i];

      Accessor p_acc, n_acc;
      p_acc.componentType = 5126;
      p_acc.count = static_cast<std::uint32_t>(v.size());
      p_acc.type = "VEC3";
      p_acc.bufferView = num_bv;
      num_bv++;
      p_acc.write_minmax = true;
      Eigen::Vector3f v_min, v_max;
      GetVertexMinMax(v, v_min, v_max);
      for (int k = 0; k < 3; k++) {
        p_acc.min[k] = v_min[k];
        p_acc.max[k] = v_max[k];
      }

      n_acc.componentType = 5126;
      n_acc.count = static_cast<std::uint32_t>(n.size());
      n_acc.type = "VEC3";
      n_acc.bufferView = num_bv;
      num_bv++;

      model.accessors.push_back(p_acc);
      model.accessors.push_back(n_acc);

      BufferView v_bv, n_bv;
      v_bv.buffer = 0;
      size_t v_size = v.size() * sizeof(float) * 3;
      std::vector<std::uint8_t> v_bytes(v_size);
      std::memcpy(v_bytes.data(), v.data(), v_size);
      v_bv.byteLength = static_cast<std::uint32_t>(v_size);
      v_bv.byteOffset = static_cast<std::uint32_t>(total_size);
      total_size += v_bv.byteLength;
      model.bufferViews.push_back(v_bv);
      bytes_list.push_back(v_bytes);

      n_bv.buffer = 0;
      size_t n_size = n.size() * sizeof(float) * 3;
      std::vector<std::uint8_t> n_bytes(n_size);
      std::memcpy(n_bytes.data(), n.data(), n_size);
      n_bv.byteLength = static_cast<std::uint32_t>(n_size);
      n_bv.byteOffset = static_cast<std::uint32_t>(total_size);
      total_size += n_bv.byteLength;
      model.bufferViews.push_back(n_bv);
      bytes_list.push_back(n_bytes);
    }
  }

  if (!keyframes.empty()) {
    if (model.animations.empty()) {
      model.animations.resize(1);
    }

    std::vector<float> input;
    std::vector<Eigen::Vector3f> output;

    for (const auto& kf : keyframes) {
      input.push_back(kf.first);
      Eigen::Vector3f uni_scale(kf.second.s, kf.second.s, kf.second.s);
      output.push_back(uni_scale);
    }

    Accessor input_acc, output_acc;
    BufferView input_bv, output_bv;

    input_acc.componentType = 5126;
    input_acc.count = static_cast<std::uint32_t>(keyframes.size());
    input_acc.type = "SCALAR";
    input_acc.bufferView = num_bv;
    num_bv++;
    input_acc.write_minmax = true;
    auto min_max_t = std::minmax_element(input.begin(), input.end());
    input_acc.max.resize(1);
    input_acc.min.resize(1);
    input_acc.max[0] = *min_max_t.second;
    input_acc.min[0] = *min_max_t.first;
    model.accessors.push_back(input_acc);

    // TODO: scale case only
    output_acc.componentType = 5126;
    output_acc.count = static_cast<std::uint32_t>(keyframes.size());
    output_acc.type = "VEC3";
    output_acc.bufferView = num_bv;
    num_bv++;
    model.accessors.push_back(output_acc);

    input_bv.buffer = 0;
    size_t input_bv_size = input.size() * sizeof(float) * 1;
    std::vector<std::uint8_t> input_bv_bytes(input_bv_size);
    std::memcpy(input_bv_bytes.data(), input.data(), input_bv_size);
    input_bv.byteLength = static_cast<std::uint32_t>(input_bv_size);
    input_bv.byteOffset = static_cast<std::uint32_t>(total_size);
    total_size += input_bv.byteLength;
    model.bufferViews.push_back(input_bv);
    bytes_list.push_back(input_bv_bytes);

    output_bv.buffer = 0;
    size_t output_bv_size = output.size() * sizeof(float) * 3;
    std::vector<std::uint8_t> output_bv_bytes(output_bv_size);
    std::memcpy(output_bv_bytes.data(), output.data(), output_bv_size);
    output_bv.byteLength = static_cast<std::uint32_t>(output_bv_size);
    output_bv.byteOffset = static_cast<std::uint32_t>(total_size);
    total_size += output_bv.byteLength;
    model.bufferViews.push_back(output_bv);
    bytes_list.push_back(output_bv_bytes);

    gltf::Channel channel;
    channel.sampler = static_cast<int>(model.animations[0].samplers.size());

    // TODO
    /* DANGER static */
    static int anim_node_count = 0;
    channel.target.node = anim_node_count;
    anim_node_count++;
    channel.target.path = "scale";

    gltf::Sampler sampler;
    sampler.input = static_cast<int>(model.accessors.size() - 2);
    sampler.output = static_cast<int>(model.accessors.size() - 1);
    sampler.interpolation = "STEP";
    model.animations[0].channels.push_back(channel);
    model.animations[0].samplers.push_back(sampler);
  }

  if (is_glb) {
    model.buffers[0].is_glb = true;

    if (process_glb_images) {
      for (auto& image : model.images) {
        if (image.glb_processed) {
          continue;
        }

        image.is_glb = true;
        image.glb_processed = true;
        image.bufferView = num_bv;

        // Does not need accessor for image
        // BufferView and mimeType are enough for decoding
        BufferView bv;
        bv.buffer = 0;
        // Image data should be last of buffer
        // Otherwise you may get ""ACCESSOR_TOTAL_OFFSET_ALIGNMENT" Accessor's
        // total byteOffset XXXX isn't a multiple of componentType length 4."
        // for other (e.g. vertices) bufferViews
        bv.byteLength = static_cast<int>(image.data.size());
        bv.byteOffset = static_cast<std::uint32_t>(total_size);
        total_size += bv.byteLength;
        model.bufferViews.push_back(bv);

        bytes_list.push_back(image.data);

        num_bv++;
      }
    }

  } else {
    model.buffers[0].is_glb = false;
    model.buffers[0].uri = bin_name;
  }

  model.buffers[0].byteLength = static_cast<std::uint32_t>(total_size);

  combined_bytes.resize(total_size);
  size_t offset = org_total_size;
  for (auto& bytes : bytes_list) {
    combined_bytes.insert(combined_bytes.begin() + offset, bytes.begin(),
                          bytes.end());
    offset += bytes.size();
  }

  // model.materials.back().with_alpha = with_alpha;

  // return combined_bytes;
}

void MakeGltfBinAndUpdateModel(const ugu::Mesh& mesh,
                               const std::string& bin_name, bool is_glb,
                               Model& model,
                               std::vector<std::uint8_t>& combined_bytes,
                               bool process_glb_images = true) {
  // Flip v
  auto gltf_uvs = mesh.uv();
  for (auto& uv : gltf_uvs) {
    uv[1] = 1.f - uv[1];
  }

  std::vector<std::vector<Eigen::Vector3f>> blendshape_vertices,
      blendshape_normals;
  for (const auto& b : mesh.blendshapes()) {
    blendshape_vertices.push_back(b.vertices);
    blendshape_normals.push_back(b.normals);
  }

  /// bool with_alpha = !mesh.materials()[0].with_alpha_tex.empty();

  MakeGltfBinAndUpdateModel(
      mesh.vertices(), mesh.stats().bb_max, mesh.stats().bb_min, mesh.normals(),
      gltf_uvs, mesh.vertex_indices(), bin_name, is_glb, blendshape_vertices,
      blendshape_normals, mesh.keyframes(), mesh.anim_interp(), model,
      combined_bytes, process_glb_images);
}

}  // namespace gltf
}  // namespace ugu

#endif

#endif