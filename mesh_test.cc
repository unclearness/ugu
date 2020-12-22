/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "ugu/mesh.h"

#include <filesystem>
#include <iostream>

inline std::vector<std::string> Split(const std::string& s, char delim) {
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

inline bool GetFileNames(std::string folderPath,
                         std::vector<std::string>& file_names) {
  using namespace std::filesystem;
  directory_iterator iter(folderPath), end;
  std::error_code err;

  for (; iter != end && !err; iter.increment(err)) {
    const directory_entry entry = *iter;
    std::string name = *(Split(entry.path().string(), '/').end() - 1);
    file_names.push_back(name);
  }

  if (err) {
    std::cout << err.value() << std::endl;
    std::cout << err.message() << std::endl;
    return false;
  }
  return true;
}

inline std::string ExtractPathExt(const std::string& fn) {
  std::string::size_type pos;
  if ((pos = fn.find_last_of(".")) == std::string::npos) {
    return "";
  }
  return fn.substr(pos + 1, fn.size());
}

inline std::string ExtractPathWithoutExt(const std::string& fn) {
  std::string::size_type pos;
  if ((pos = fn.find_last_of(".")) == std::string::npos) {
    return fn;
  }

  return fn.substr(0, pos);
}

void TestBlendshapes() {
  std::string data_dir = "../data/blendshape/";
  std::vector<std::string> file_names;
  GetFileNames(data_dir, file_names);
  std::string base_name = "cube.obj";
  ugu::Mesh base_mesh;
  base_mesh.LoadObj(data_dir + base_name, data_dir);
  base_mesh.SplitMultipleUvVertices();

  std::vector<std::string> obj_names;
  for (auto& name : file_names) {
    auto ext = ExtractPathExt(name);
    if (name == base_name || ext != "obj") {
      continue;
    }
    obj_names.push_back(name);
  }

  std::vector<ugu::Mesh> blendshape_meshes;
  for (auto& name : obj_names) {
    ugu::Mesh mesh;
    mesh.LoadObj(data_dir + name, data_dir);
    mesh.SplitMultipleUvVertices();
    blendshape_meshes.push_back(mesh);
  }

  std::vector<ugu::Blendshape> blendshapes;
  for (int i = 0; i < blendshape_meshes.size(); i++) {
    auto& blendshape_mesh = blendshape_meshes[i];
    auto name = ExtractPathWithoutExt(obj_names[i]);

    ugu::Blendshape blendshape;
    std::vector<Eigen::Vector3f> displacement = base_mesh.vertices();
    for (size_t j = 0; j < displacement.size(); j++) {
      displacement[j] = blendshape_mesh.vertices()[j] - displacement[j];
    }
    blendshape.vertices = displacement;
    blendshape.normals = blendshape_mesh.normals();
    blendshape.name = name;

    blendshapes.push_back(blendshape);
  }
  base_mesh.set_blendshapes(blendshapes);

  base_mesh.WriteGltfSeparate(data_dir, "blendshape");
  base_mesh.WriteGlb(data_dir, "blendshape.glb");
}

void TestIO() {
  std::string data_dir = "../data/bunny/";
  std::string in_obj_path = data_dir + "bunny.obj";

  ugu::Mesh src, dst;

  src.LoadObj(in_obj_path, data_dir);

  src.SplitMultipleUvVertices();
  src.WriteGltfSeparate(data_dir, "bunny");

  src.WriteGlb(data_dir, "bunny.glb");

  dst = ugu::Mesh(src);

  dst.FlipFaces();

  dst.WriteObj(data_dir, "bunny2");
}

void TestMerge() {
  std::string data1_dir = "../data/bunny/";
  std::string in_obj_path1 = data1_dir + "bunny.obj";
  ugu::Mesh bunny, bunny_moved, dst;
  bunny.LoadObj(in_obj_path1, data1_dir);
  bunny_moved = ugu::Mesh(bunny);
  bunny_moved.Translate(bunny.stats().bb_max);
  bunny_moved.FlipFaces();  // Flip face for moved bunny

  std::string data2_dir = "../data/buddha/";
  std::string in_obj_path2 = data2_dir + "buddha.obj";
  ugu::Mesh buddha;
  buddha.LoadObj(in_obj_path2, data2_dir);

  buddha.SplitMultipleUvVertices();
  buddha.WriteGlb(data2_dir, "buddha.glb");

  ugu::MergeMeshes(bunny, buddha, &dst);
  dst.WriteObj(data1_dir, "bunny_and_buddha");

  ugu::MergeMeshes(bunny, bunny_moved, &dst, true);
  dst.WriteObj(data1_dir, "bunny_twin");

  ugu::MergeMeshes(bunny, bunny_moved, &dst);
  dst.WriteObj(data1_dir, "bunny_twin_2materials");
}

void TestRemove() {
  std::string data1_dir = "../data/bunny/";
  std::string in_obj_path1 = data1_dir + "bunny.obj";
  ugu::Mesh bunny;
  bunny.LoadObj(in_obj_path1, data1_dir);

  std::vector<bool> valid_face_table;

  Eigen::Vector3f direc(0.0, 0.0, 1.0f);
  for (const auto& fn : bunny.face_normals()) {
    if (direc.dot(fn) > 0.0) {
      valid_face_table.push_back(true);
    } else {
      valid_face_table.push_back(false);
    }
  }

  bunny.RemoveFaces(valid_face_table);

  bunny.WriteObj(data1_dir, "bunny_removed_back");
}

int main() {
  TestBlendshapes();

  TestIO();

  TestMerge();

  TestRemove();

  return 0;
}