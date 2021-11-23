/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include <fstream>

#include "ugu/face_adjacency.h"
#include "ugu/mesh.h"
#include "ugu/util/geom_util.h"
#include "ugu/util/image_util.h"

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
  ugu::MeshPtr mesh = ugu::Mesh::Create();
  mesh->LoadObj(obj_path, data_dir);

  auto [clusters, non_orphans, orphans, clusters_f] =
      ugu::ClusterByConnectivity(mesh->uv_indices(),
                                 static_cast<int32_t>(mesh->uv().size()));

  ugu::Mesh cluster_colored = *mesh;
  auto random_colors =
      ugu::GenRandomColors(static_cast<int32_t>(clusters.size()));
  auto [vid2uvid, uvid2vid] =
      ugu::GenerateVertex2UvMap(mesh->vertex_indices(), mesh->vertices().size(),
                                mesh->uv_indices(), mesh->uv().size());
  std::vector<Eigen::Vector3f> colors(mesh->vertices().size());
  for (size_t i = 0; i < clusters.size(); i++) {
    const auto& cluster = clusters[i];
    for (const auto& uvid : cluster) {
      auto vid = uvid2vid[uvid];
      colors[vid] = random_colors[i];
    }
  }

  cluster_colored.set_vertex_colors(colors);
  cluster_colored.WritePly(data_dir + "bunny_uv_cluster.ply");

  auto [boundary_edges_list, boundary_vertex_ids_list] = ugu::FindBoundaryLoops(
      mesh->vertex_indices(), static_cast<int32_t>(mesh->vertices().size()));

  for (size_t i = 0; i < boundary_edges_list.size(); i++) {
    ugu::LOGI("%d th boundary\n", i);
    std::vector<Eigen::Vector3f> boundary_vertices;
    for (size_t j = 0; j < boundary_edges_list[i].size(); j++) {
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
