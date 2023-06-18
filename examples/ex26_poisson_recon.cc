/*
 * Copyright (C) 2023, unclearness
 * All rights reserved.
 */

#include "ugu/external/external.h"
#include "ugu/timer.h"
#include "ugu/util/geom_util.h"
#include "ugu/util/raster_util.h"
#include "ugu/voxel/marching_cubes.h"
#include "ugu/voxel/voxel.h"

int main() {
  using namespace ugu;
  Timer<> timer;

  MeshPtr src = Mesh::Create();
  src->LoadObj("../data/bunny/bunny.obj");

  std::vector<Eigen::Vector3f> colors;
  src->SplitMultipleUvVertices();
  FetchVertexAttributeFromTexture(src->materials()[0].diffuse_tex, src->uv(),
                                  colors);
  src->set_vertex_colors(colors);

  timer.Start();
  MeshPtr recon = PoissonRecon(src);
  timer.End();
  ugu::LOGI("PoissonRecon %f ms\n", timer.elapsed_msec());
  if (recon != nullptr) {
    recon->WriteObj("../data/bunny/", "bunny_spr");
  }

  {
    ugu::VoxelGrid voxel_grid;
    float resolution =
        (src->stats().bb_max - src->stats().bb_min).maxCoeff() / 32;
    Eigen::Vector3f offset = Eigen::Vector3f::Ones() * resolution * 2;
    voxel_grid.Init(src->stats().bb_max + offset, src->stats().bb_min - offset,
                    resolution);
    ugu::VoxelUpdateOption option = ugu::GenFuseDepthDefaultOption(resolution);
    // src->set_normals({});
    timer.Start();
    EstimateNormalsFromPoints(src.get());
    timer.End();
    ugu::LOGI("Normal estimation %f ms\n", timer.elapsed_msec());
    // src->WriteObj("../data/bunny/bunny_tmp.obj");

    timer.Start();
    ugu::FusePoints(src->vertices(), src->normals(), option, voxel_grid,
                    src->vertex_colors());
    timer.End();
    ugu::LOGI("PointCloud -> SDF %f ms\n", timer.elapsed_msec());

    auto pc_fused = ugu::Mesh::Create();
    timer.Start();
    ugu::MarchingCubes(voxel_grid, pc_fused.get(), 0.0, true);
    timer.End();
    ugu::LOGI("Marching Cubes %f ms\n", timer.elapsed_msec());

    pc_fused->set_default_material();
    pc_fused->CalcNormal();
    pc_fused->WriteObj("../data/bunny/bunny_mc.obj");
  }

  return 0;
}
