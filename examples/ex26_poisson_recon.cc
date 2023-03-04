/*
 * Copyright (C) 2023, unclearness
 * All rights reserved.
 */

#include "ugu/external/external.h"
#include "ugu/util/raster_util.h"

int main() {
  using namespace ugu;
  MeshPtr src = Mesh::Create();
  src->LoadObj("../data/bunny/bunny.obj");

  std::vector<Eigen::Vector3f> colors;
  src->SplitMultipleUvVertices();
  FetchVertexAttributeFromTexture(src->materials()[0].diffuse_tex, src->uv(),
                                  colors);
  src->set_vertex_colors(colors);

  MeshPtr recon = PoissonRecon(src);
  if (recon != nullptr) {
    recon->WriteObj("../data/bunny/", "bunny_spr");
  }

  return 0;
}
