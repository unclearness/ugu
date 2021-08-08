/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include <fstream>

#include "ugu/face_adjacency.h"
#include "ugu/mesh.h"
#include "ugu/util/geom_util.h"

// test by bunny data with 6 views
int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  std::string data_dir = "../data/bunny/";
  std::string obj_path = data_dir + "bunny.obj";

  {
    std::ifstream ifs(obj_path);
    if (!ifs.is_open()) {
      printf("Please put %s\n", obj_path.c_str());
      return -1;
    }
  }

  // load mesh
  std::shared_ptr<ugu::Mesh> mesh = std::make_shared<ugu::Mesh>();
  mesh->LoadObj(obj_path, data_dir);

  auto [boundary_edges_list, boundary_vertex_ids_list] =
      ugu::FindBoundaryLoops(*mesh);

  for (auto i = 0; i < boundary_edges_list.size(); i++) {
    ugu::LOGI("%d th boundary\n", i);
    std::vector<Eigen::Vector3f> boundary_vertices;
    for (auto j = 0; j < boundary_edges_list[i].size(); j++) {
      ugu::LOGI("%d th vertex id %d\n", j, boundary_vertex_ids_list[i][j]);
      boundary_vertices.push_back(
          mesh->vertices()[boundary_vertex_ids_list[i][j]]);
    }
    ugu::Mesh tmp;
    tmp.set_vertices(boundary_vertices);
    tmp.WritePly(data_dir + "bunny_boundary_" + std::to_string(i) + ".ply");
  }

  ugu::FaceAdjacency face_adjacency;
  face_adjacency.Init(static_cast<int>(mesh->vertices().size()),
                      mesh->vertex_indices());

  auto [boundary_edges, boundary_vertex_ids] =
      face_adjacency.GetBoundaryEdges();

  ugu::LOGI("boundary_edges\n");
  for (const auto& p : boundary_edges) {
    ugu::LOGI("%d -> %d\n", p.first, p.second);
  }

  ugu::LOGI("\nboundary_vertex_ids\n");
  for (const auto& vid : boundary_vertex_ids) {
    ugu::LOGI("%d\n", vid);
  }

  std::vector<bool> valid_vertex_table(mesh->vertices().size(), true);
  for (const auto& vid : boundary_vertex_ids) {
    valid_vertex_table[vid] = false;
  }

  mesh->RemoveVertices(valid_vertex_table);
  mesh->WritePly(data_dir + "bunny_boundary_removed.ply");

  return 0;
}
