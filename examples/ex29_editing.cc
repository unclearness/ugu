/*
 * Copyright (C) 2023, unclearness
 * All rights reserved.
 */

#include "example_utils.h"
#include "ugu/editing/poisson_mesh_editing.h"
#include "ugu/mesh.h"
#include "ugu/timer.h"
#include "ugu/util/io_util.h"

using namespace ugu;

int main() {
  Timer<> timer;

  std::string cylinder_dir = ugu_example::GetDataDir("cylinder");
  std::string face_dir = ugu_example::GetDataDir("face");
  std::string out_dir = ugu_example::GetOutDir("ex29_editing");

  MeshPtr pinned = Mesh::Create();
  pinned->LoadObj(cylinder_dir + "cylinder.obj");

  std::vector<int> pinned_boundary_vids =
      LoadTxtAsVector<int>(cylinder_dir + "boundary.txt");

  MeshPtr floating = Mesh::Create();
  floating->LoadObj(face_dir + "mediapipe_face.obj");

  std::vector<int> floating_boundary_vids =
      LoadTxtAsVector<int>(face_dir + "boundary.txt");

  timer.Start();
  MeshPtr merged = PoissonMeshMerging(pinned, pinned_boundary_vids, floating,
                                      floating_boundary_vids);
  timer.End();
  std::cout << "PoissonMeshMerging: " << timer.elapsed_msec() << " ms."
            << std::endl;
  merged->WriteObj(out_dir + "ex29.obj");

  return 0;
}
