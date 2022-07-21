/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 *
 */

#include "ugu/external/external.h"
#include "ugu/parameterize/parameterize.h"
#include "ugu/util/math_util.h"

#ifdef UGU_USE_LIBGIL

//#include "igl/boundary_loop.h"
#include "igl/lscm.h"

#endif

namespace ugu {

bool LibiglLscm(const std::vector<Eigen::Vector3f>& vertices,
                const std::vector<Eigen::Vector3i>& vertex_indices,
                const std::vector<int>& boundary,
                std::vector<Eigen::Vector2f>& uvs) {
#ifdef UGU_USE_LIBIGL

  Eigen::MatrixXd V(vertices.size(), 3);
  Eigen::MatrixXi F(vertex_indices.size(), 3);

  for (size_t i = 0; i < vertices.size(); i++) {
    V.row(i) = vertices[i].cast<double>();
  }

  for (size_t i = 0; i < vertex_indices.size(); i++) {
    F.row(i) = vertex_indices[i];
  }

  Eigen::MatrixXd V_uv;

  Eigen::VectorXi b(2, 1);
  // igl::boundary_loop(F, bnd);
  b(0) = boundary[0];
  b(1) = boundary[boundary.size() / 2];
  Eigen::MatrixXd bc(2, 2);
  bc << 0, 0, 1, 0;

  // LSCM parameterization
  igl::lscm(V, F, b, bc, V_uv);

  uvs.clear();
  Eigen::Vector2f invalid;
  invalid.Constant(std::numeric_limits<float>::lowest());
  uvs.resize(vertices.size(), invalid);

  Eigen::Vector2d min_uv = V_uv.colwise().minCoeff();
  Eigen::Vector2d max_uv = V_uv.colwise().maxCoeff();
  Eigen::Vector2d diff = max_uv - min_uv;

  for (size_t i = 0; i < uvs.size(); i++) {
    uvs[i] = (V_uv.row(i) - min_uv.transpose())
                 .cwiseProduct(diff.cwiseInverse().transpose())
                 .cast<float>();
  }

  return true;
#else
  (void)vertices;
  (void)vertex_indices;
  (void)boundary;
  (void)uvs;
  ugu::LOGE("Not available in current configuration\n");
  return false;
#endif
}

bool LibiglLscm(const std::vector<Eigen::Vector3f>& vertices,
                const std::vector<Eigen::Vector3i>& vertex_indices, int tex_w,
                int tex_h, std::vector<Eigen::Vector2f>& uvs,
                std::vector<Eigen::Vector3i>& uv_indices) {
#ifdef UGU_USE_LIBIGL

  // Step 1: Clustering by connectivity
  std::vector<float> cluster_areas;
  std::vector<std::vector<Eigen::Vector3i>> clusters;
  std::vector<std::vector<uint32_t>> cluster_fids;
  auto [clusters_v, non_orphans, orphans, clusters_f] =
      ClusterByConnectivity(vertex_indices, vertices.size(), false);

  for (const auto& cluster_f : clusters_f) {
    std::vector<uint32_t> cluster_fid;
    std::vector<Eigen::Vector3i> cluster;
    double cluster_area = 0.0;

    for (const auto& f : cluster_f) {
      cluster_fid.push_back(static_cast<uint32_t>(f));
      cluster.push_back(vertex_indices[f]);

      const auto& v0 = vertices[vertex_indices[f][0]];
      const auto& v1 = vertices[vertex_indices[f][1]];
      const auto& v2 = vertices[vertex_indices[f][2]];
      const double area = std::abs(((v2 - v0).cross(v1 - v0)).norm()) * 0.5;
      cluster_area += area;
    }

    cluster_areas.push_back(static_cast<float>(cluster_area));
    clusters.push_back(std::move(cluster));
    cluster_fids.push_back(std::move(cluster_fid));
  }
  // Sort by cluster_areas
  auto indices = ugu::argsort(cluster_areas, true);
  cluster_areas = ApplyIndices(cluster_areas, indices);
  clusters = ApplyIndices(clusters, indices);
  cluster_fids = ApplyIndices(cluster_fids, indices);

  // Step 2: Parameterize per segment
  std::vector<std::vector<Eigen::Vector2f>> cluster_uvs(cluster_fids.size());
  std::vector<std::vector<Eigen::Vector3i>> cluster_sub_faces;
  for (size_t cid = 0; cid < clusters.size(); ++cid) {
    auto [cluster_vtx, cluster_face] =
        ExtractSubGeom(vertices, vertex_indices, cluster_fids[cid]);

    cluster_sub_faces.push_back(cluster_face);

    auto [boundary_edges_list, boundary_vertex_ids_list] = FindBoundaryLoops(
        cluster_face, static_cast<int32_t>(cluster_vtx.size()));

    if (boundary_vertex_ids_list.empty()) {
      LOGE("Failed to find boundary for LSCM. Possibly water-tight?\n");
      continue;
    }

    LibiglLscm(cluster_vtx, cluster_face, boundary_vertex_ids_list[0],
               cluster_uvs[cid]);
  }

  // Step 3: Pack segments
  PackUvIslands(cluster_areas, clusters, cluster_uvs, cluster_sub_faces,
                cluster_fids, vertex_indices.size(), tex_w, tex_h, uvs,
                uv_indices, false);

  return true;
#else
  (void)vertices;
  (void)vertex_indices;
  (void)tex_w;
  (void)tex_h;
  (void)uvs;
  (void)uv_indices;
  ugu::LOGE("Not available in current configuration\n");
  return false;
#endif
}

}  // namespace ugu
