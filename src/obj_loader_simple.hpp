// obj_loader_simple.hpp
#pragma once
#include <cctype>
#include <cstdlib>
// #include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace obj_loader_simple {

struct ObjMtl {
  std::string name = "default";
  std::vector<float> Ka;  // size 3
  std::vector<float> Kd;  // size 3
  std::vector<float> Ks;  // size 3
  float Tr = 0.0f;        // transparency (as in your code)
  int illum = 0;
  float Ns = 0.0f;
  std::string map_Kd;  // texture path
};

struct ObjData {
  std::vector<std::array<float, 3>> verts;
  std::vector<std::array<float, 2>> uvs;
  std::vector<std::array<float, 3>> normals;
  std::vector<std::array<float, 3>> vert_colors;

  // faces are stored as polygon indices (not triangulated)
  std::vector<std::vector<int>> indices;
  std::vector<std::vector<int>> uv_indices;
  std::vector<std::vector<int>> normal_indices;

  std::string mtl_path;  // empty if none
  std::vector<ObjMtl> mtls;

  // material name -> list of face IDs (0-based, corresponds to
  // indices[face_id])
  std::unordered_map<std::string, std::vector<int>> mtl_per_faces;
};

inline static std::string TrimCopy(const std::string& s) {
  size_t b = 0;
  while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  size_t e = s.size();
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

inline static bool starts_with(const std::string& s, const char* prefix) {
  const size_t n = std::char_traits<char>::length(prefix);
  return s.size() >= n && s.compare(0, n, prefix) == 0;
}

// Split by whitespace (like Python .split()).
inline static std::vector<std::string> SplitWs(const std::string& line) {
  std::istringstream iss(line);
  std::vector<std::string> out;
  std::string tok;
  while (iss >> tok) out.push_back(tok);
  return out;
}

inline static std::vector<float> ToFloats(
    const std::vector<std::string>& data) {
  std::vector<float> out;
  out.reserve(data.size());
  for (const auto& x : data) out.push_back(std::stof(x));
  return out;
}

inline static std::vector<ObjMtl> LoadMtls(const std::string& mtl_path) {
  std::vector<ObjMtl> mtls;

  std::ifstream fp(mtl_path);
  if (!fp) {
    // throw std::runtime_error("Failed to open mtl: " + mtl_path);
    return mtls;
  }

  std::string line;
  while (std::getline(fp, line)) {
    line = TrimCopy(line);
    if (line.empty()) continue;
    if (starts_with(line, "#")) continue;

    auto splitted = SplitWs(line);
    if (splitted.size() < 2) continue;

    const std::string& start = splitted[0];
    std::vector<std::string> data(splitted.begin() + 1, splitted.end());

    if (start == "newmtl") {
      ObjMtl mat;
      mat.name = data[0];
      mtls.push_back(std::move(mat));
    } else {
      if (mtls.empty()) {
        // Python版だと mtls[-1] で落ちるケース。C++では安全にする。
        // 必要なら throw に変えてOK。
        ObjMtl mat;
        mtls.push_back(std::move(mat));
      }
      ObjMtl& m = mtls.back();

      if (start == "Ka") {
        auto v = ToFloats(data);
        m.Ka = v;
      } else if (start == "Kd") {
        auto v = ToFloats(data);
        m.Kd = v;
      } else if (start == "Ks") {
        auto v = ToFloats(data);
        m.Ks = v;
      } else if (start == "Tr") {
        auto v = ToFloats(data);
        m.Tr = v.empty() ? 0.0f : v[0];
      } else if (start == "illum") {
        auto v = ToFloats(data);
        m.illum = v.empty() ? 0 : static_cast<int>(v[0]);
      } else if (start == "Ns") {
        auto v = ToFloats(data);
        m.Ns = v.empty() ? 0.0f : v[0];
      } else if (start == "map_Kd") {
        m.map_Kd = data[0];
      }
    }
  }

  return mtls;
}

// Parse an OBJ index token like:
//   "v"          -> v
//   "v/vt"       -> v, vt
//   "v//vn"      -> v, vn
//   "v/vt/vn"    -> v, vt, vn
// and returns 0-based indices or -1 if missing.
struct ObjIndexTriplet {
  int v = -1;
  int vt = -1;
  int vn = -1;
};

inline static ObjIndexTriplet ParseFaceToken(const std::string& tok) {
  ObjIndexTriplet t;

  // Split by '/'
  // Keep empty fields (for v//vn).
  std::vector<std::string> parts;
  parts.reserve(3);
  std::string cur;
  for (char c : tok) {
    if (c == '/') {
      parts.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  parts.push_back(cur);

  auto parse_one = [](const std::string& s) -> int {
    if (s.empty()) return -1;
    // Note: OBJ supports negative indices;
    return std::stoi(s) - 1;
  };

  if (!parts.empty()) t.v = parse_one(parts[0]);
  if (parts.size() >= 2) t.vt = parse_one(parts[1]);
  if (parts.size() >= 3) t.vn = parse_one(parts[2]);
  return t;
}

inline static ObjData LoadObjSimple(const std::string& obj_path,
                                    const std::string& mtl_dir) {
  ObjData out;

  int num_verts = 0;
  int num_uvs = 0;
  int num_normals = 0;
  int num_indices = 0;

  // const std::filesystem::path base_dir =
  //     std::filesystem::path(obj_path).parent_path();
  std::string mtl_file_name;
  bool has_mtl_path = false;

  std::ifstream f(obj_path);
  if (!f) {
    // throw std::runtime_error("Failed to open obj: " + obj_path);
    return out;
  }

  std::string current_mtl_name;

  std::string line;
  while (std::getline(f, line)) {
    auto vals = SplitWs(line);
    if (vals.empty()) continue;

    if (vals[0] == "v") {
      if (vals.size() < 4) continue;
      out.verts.push_back(
          {std::stof(vals[1]), std::stof(vals[2]), std::stof(vals[3])});
      if (vals.size() == 7) {
        out.vert_colors.push_back(
            {std::stof(vals[4]), std::stof(vals[5]), std::stof(vals[6])});
      }
      num_verts += 1;
    } else if (vals[0] == "vt") {
      if (vals.size() < 3) continue;
      out.uvs.push_back({std::stof(vals[1]), std::stof(vals[2])});
      num_uvs += 1;
    } else if (vals[0] == "vn") {
      if (vals.size() < 4) continue;
      out.normals.push_back(
          {std::stof(vals[1]), std::stof(vals[2]), std::stof(vals[3])});
      num_normals += 1;
    } else if (vals[0] == "f") {
      std::vector<int> v_index;
      std::vector<int> uv_index;
      std::vector<int> n_index;

      bool valid_face = false;

      for (size_t i = 1; i < vals.size(); ++i) {
        auto tri = ParseFaceToken(vals[i]);

        if (num_verts > 0 && tri.v >= 0) v_index.push_back(tri.v);
        if (num_uvs > 0 && tri.vt >= 0) uv_index.push_back(tri.vt);

        if (num_normals > 0) {
          if (tri.vn >= 0) {
            n_index.push_back(tri.vn);
          } else {
            std::cerr << "no normal index\n";
            if (tri.v >= 0) n_index.push_back(tri.v);
          }
        }
      }

      if (!v_index.empty()) {
        out.indices.push_back(std::move(v_index));
        valid_face = true;
      }
      if (!uv_index.empty()) {
        out.uv_indices.push_back(std::move(uv_index));
        valid_face = true;
      }
      if (!n_index.empty()) {
        out.normal_indices.push_back(std::move(n_index));
        valid_face = true;
      }

      if (valid_face) {
        num_indices += 1;
        if (!current_mtl_name.empty()) {
          // Python: mtl_per_faces[current].append(num_indices - 1)
          out.mtl_per_faces[current_mtl_name].push_back(num_indices - 1);
        }
      }
    } else if (vals[0] == "mtllib") {
      if (vals.size() >= 2) {
        mtl_file_name = vals[1];
        // std::filesystem::path p = base_dir / mtl_file_name;
        // out.mtl_path = std::filesystem::absolute(p).string();
        out.mtl_path = mtl_dir + "/" + mtl_file_name;
        has_mtl_path = true;
      }
    } else if (vals[0] == "usemtl") {
      if (vals.size() >= 2) {
        current_mtl_name = vals[1];
        if (out.mtl_per_faces.find(current_mtl_name) ==
            out.mtl_per_faces.end()) {
          out.mtl_per_faces[current_mtl_name] = {};
        }
      }
    }
  }

  if (has_mtl_path) {
    out.mtls = LoadMtls(out.mtl_path);
  } else {
    // Python: mtls = [ObjMtl()]; mtl_per_faces[mtls[0].name] =
    // list(range(len(indices)))
    out.mtls = {ObjMtl()};
    out.mtl_per_faces[out.mtls[0].name].clear();
    out.mtl_per_faces[out.mtls[0].name].reserve(
        static_cast<size_t>(out.indices.size()));
    for (int i = 0; i < static_cast<int>(out.indices.size()); ++i) {
      out.mtl_per_faces[out.mtls[0].name].push_back(i);
    }
  }

  return out;
}
}  // namespace obj_loader_simple