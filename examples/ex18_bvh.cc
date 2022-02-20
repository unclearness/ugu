/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include <iostream>
#include <random>
#include <unordered_set>

#include "ugu/accel/bvh.h"
#include "ugu/image.h"
#include "ugu/mesh.h"
#include "ugu/timer.h"
#include "ugu/util/geom_util.h"

namespace {}  // namespace

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  ugu::MeshPtr mesh = ugu::Mesh::Create();
  mesh->LoadObj("../data/bunny/bunny.obj", "../data/bunny/");

  ugu::Bvh<Eigen::Vector3f, Eigen::Vector3i> bvh;
  bvh.SetAxisNum(3);
  bvh.SetMaxLeafDataNum(5);
  bvh.SetData(mesh->vertices(), mesh->vertex_indices());

  bvh.Build();
  auto meshes = bvh.Visualize(8);
  ugu::Mesh merged;
  ugu::MergeMeshes(meshes, &merged);
  merged.WriteObj("../data/bunny/", "bunny_bvh");

  return 0;
}
