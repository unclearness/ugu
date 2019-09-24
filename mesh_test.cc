/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "ugu/mesh.h"

int main() {
  
  std::string data_dir = "../data/bunny/";
  std::string in_obj_path = data_dir + "bunny.obj";

  ugu::Mesh src, dst;

  src.LoadObj(in_obj_path, data_dir);

  dst = ugu::Mesh(src);

  dst.WriteObj(data_dir, "bunny2");

  return 0;
}