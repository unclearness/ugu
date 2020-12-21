/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "ugu/mesh.h"

void TestIO() {
  std::string data_dir = "../data/bunny/";
  std::string in_obj_path = data_dir + "bunny.obj";

  ugu::Mesh src, dst;

  src.LoadObj(in_obj_path, data_dir);

  src.SplitMultipleUvVertices();
  src.WriteGltfSeparate(data_dir, "bunny");

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
  TestIO();

  TestMerge();

  TestRemove();

  return 0;
}