/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/face_adjacency.h"
#include "ugu/util/geom_util.h"

namespace {}  // namespace

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  std::string data_dir = "../data/";

  auto nonmanifold_mesh = ugu::MakePlane(1.f);
  std::vector<Eigen::Vector3f> vertices = nonmanifold_mesh->vertices();
  std::vector<Eigen::Vector3i> faces = nonmanifold_mesh->vertex_indices();
  // std::vector<Eigen::Vector3f> vertices = nonmanifold_mesh->vertices();

  vertices.push_back({1.f, 1.f, 1.f});
  vertices.push_back({1.f, 0.f, 1.f});
  vertices.push_back({1.f, -1.f, 2.f});
  vertices.push_back({-1.f, -1.f, 1.f});
  faces.push_back({0, 1, 4});
  faces.push_back({0, 1, 5});
  faces.push_back({0, 2, 6});
  faces.push_back({2, 3, 7});

  nonmanifold_mesh->set_vertices(vertices);
  nonmanifold_mesh->set_vertex_indices(faces);
  nonmanifold_mesh->set_uv({});
  nonmanifold_mesh->set_uv_indices({});
  nonmanifold_mesh->set_default_material();

  nonmanifold_mesh->CalcNormal();

  nonmanifold_mesh->CalcStats();

  nonmanifold_mesh->WriteObj(data_dir, "nonmanifold");

  ugu::FaceAdjacency fa;
  fa.Init(static_cast<int>(nonmanifold_mesh->vertices().size()),
          nonmanifold_mesh->vertex_indices());
  auto nonmanifold_vtx = fa.GetNonManifoldVertices();
  for (const auto& v : nonmanifold_vtx) {
    ugu::LOGI("nonmanifold vid %d\n", v);
  }

  std::vector<Eigen::Vector3f> clean_vertices;
  std::vector<Eigen::Vector3i> clean_faces;
  ugu::CleanGeom(nonmanifold_mesh->vertices(),
                 nonmanifold_mesh->vertex_indices(), clean_vertices,
                 clean_faces);

  auto clean_mesh = ugu::Mesh::Create();
  clean_mesh->set_vertices(clean_vertices);
  clean_mesh->set_vertex_indices(clean_faces);
  clean_mesh->set_default_material();

  clean_mesh->CalcNormal();

  clean_mesh->CalcStats();

  fa.Init(static_cast<int>(clean_mesh->vertices().size()),
          clean_mesh->vertex_indices());
  nonmanifold_vtx = fa.GetNonManifoldVertices();
  for (const auto& v : nonmanifold_vtx) {
    ugu::LOGI("nonmanifold vid %d\n", v);
  }

  clean_mesh->WriteObj(data_dir, "nonmanifold_cleaned");

  return 0;
}
