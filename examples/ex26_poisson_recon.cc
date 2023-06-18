/*
 * Copyright (C) 2023, unclearness
 * All rights reserved.
 */

#include "ugu/external/external.h"
#include "ugu/inpaint/inpaint.h"
#include "ugu/parameterize/parameterize.h"
#include "ugu/textrans/texture_transfer.h"
#include "ugu/timer.h"
#include "ugu/util/geom_util.h"
#include "ugu/util/raster_util.h"
#include "ugu/voxel/marching_cubes.h"
#include "ugu/voxel/voxel.h"
using namespace ugu;

void Textrans(MeshPtr src, MeshPtr recon) {
  Timer<> timer;
  timer.Start();
  recon->CalcFaceNormal();
  timer.End();
  ugu::LOGI("CalcFaceNormal %f ms\n", timer.elapsed_msec());

  timer.Start();
  Parameterize(*recon.get(), 512, 512, ugu::ParameterizeUvType::kSmartUv);
  timer.End();
  ugu::LOGI("Parameterize %f ms\n", timer.elapsed_msec());

  timer.Start();
  Image3f trans_tex;
  Image1b trans_mask;
  TexTransFromColoredPoints(*src.get(), *recon.get(), 512, 512, trans_tex,
                            trans_mask);
  timer.End();
  ugu::LOGI("TexTransFromColoredPoints %f ms\n", timer.elapsed_msec());

  recon->set_default_material();
  auto mats = recon->materials();
  trans_tex.convertTo(mats[0].diffuse_tex, CV_8UC3);

  Not(trans_mask.clone(), &trans_mask);
  timer.Start();
  Inpaint(trans_mask, mats[0].diffuse_tex);
  timer.End();
  ugu::LOGI("Inpaint %f ms\n", timer.elapsed_msec());

  recon->set_materials(mats);
}

int main() {
  Timer<> timer;

  MeshPtr src = Mesh::Create();
  src->LoadObj("../data/bunny/bunny.obj");

  std::vector<Eigen::Vector3f> colors;
  src->SplitMultipleUvVertices();
  FetchVertexAttributeFromTexture(src->materials()[0].diffuse_tex, src->uv(),
                                  colors);
  src->set_vertex_colors(colors);

  timer.Start();
  MeshPtr recon = PoissonRecon(src, 7);
  timer.End();
  ugu::LOGI("PoissonRecon %f ms\n", timer.elapsed_msec());

  Textrans(src, recon);

  if (recon != nullptr) {
    recon->WriteObj("../data/bunny/", "bunny_spr");
  }

  {
    timer.Start();
    ugu::VoxelGrid voxel_grid;
    float resolution =
        (src->stats().bb_max - src->stats().bb_min).maxCoeff() / 32;
    Eigen::Vector3f offset = Eigen::Vector3f::Ones() * resolution * 2;
    voxel_grid.Init(src->stats().bb_max + offset, src->stats().bb_min - offset,
                    resolution);
    ugu::VoxelUpdateOption option = ugu::GenFuseDepthDefaultOption(resolution);
    timer.End();
    ugu::LOGI("Init %f ms\n", timer.elapsed_msec());
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

    Textrans(src, pc_fused);

    pc_fused->WriteObj("../data/bunny/bunny_mc.obj");
  }

  return 0;
}
