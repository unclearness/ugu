/*
 * Copyright (C) 2023, unclearness
 * All rights reserved.
 */

#include "ugu/editing/poisson_mesh_editing.h"
#include "ugu/mesh.h"
#include "ugu/timer.h"
#include "ugu/util/io_util.h"

using namespace ugu;

int main() {
  Timer<> timer;

  MeshPtr pinned = Mesh::Create();
  pinned->LoadObj("../data/cylinder/cylinder.obj");

  std::vector<int> pinned_boundary_vids =
      LoadTxtAsVector<int>("../data/cylinder/boundary.txt");

  MeshPtr floating = Mesh::Create();
  floating->LoadObj("../data/face/mediapipe_face.obj");

  std::vector<int> floating_boundary_vids =
      LoadTxtAsVector<int>("../data/face/boundary.txt");

  timer.Start();
  MeshPtr merged = PoissonMeshMerging(pinned, pinned_boundary_vids, floating,
                                      floating_boundary_vids);
  timer.End();
  std::cout << "PoissonMeshMerging: " << timer.elapsed_msec() << " ms."
            << std::endl;
  merged->WriteObj("../data_out/ex29.obj");

  return 0;
}
