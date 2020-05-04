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

  dst = ugu::Mesh(src);

  dst.WriteObj(data_dir, "bunny2");
}

void TestMerge() {
  std::string data1_dir = "../data/bunny/";
  std::string in_obj_path1 = data1_dir + "bunny.obj";
  ugu::Mesh bunny, bunny_moved, dst;
  bunny.LoadObj(in_obj_path1, data1_dir);
  bunny_moved = ugu::Mesh(bunny);
  bunny_moved.Translate(bunny.stats().bb_max);

  std::string data2_dir = "../data/buddha/";
  std::string in_obj_path2 = data2_dir + "buddha.obj";
  ugu::Mesh buddha;
  buddha.LoadObj(in_obj_path2, data2_dir);

  ugu::MergeMeshes(bunny, buddha, &dst);
  dst.WriteObj(data1_dir, "bunny_and_buddha");

  ugu::MergeMeshes(bunny, bunny_moved, &dst, true);
  dst.WriteObj(data1_dir, "bunny_twin");
}

int main() {
  TestIO();

  TestMerge();

  return 0;
}