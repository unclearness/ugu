/*
 * Copyright (C) 2024, unclearness
 * All rights reserved.
 */

/* This is a partial implementation of the following paper
 *
 * Yu, Y., Zhou, K., Xu, D., Shi, X., Bao, H., Guo, B., & Shum, H. Y. (2004).
 * Mesh editing with poisson-based gradient field manipulation.
 * In ACM SIGGRAPH 2004 Papers (pp. 644-651).
 *
 */

#include "ugu/editing/poisson_mesh_editing.h"

#include <unordered_map>
#include <unordered_set>

#include "ugu/face_adjacency.h"
#include "ugu/registration/rigid.h"
#include "ugu/util/geom_util.h"

#ifdef _WIN32
#pragma warning(push, UGU_EIGEN_WARNING_LEVEL)
#endif
#include "Eigen/SparseCholesky"
#include "Eigen/SparseCore"
#ifdef _WIN32
#pragma warning(pop)
#endif

namespace {

using namespace ugu;
#if 0
  struct BoundaryCondition {
    Eigen::Vector3f p;
    Eigen::Matrix3f f;
    float s;
    float r;
  };

  void SolvePossion(const std::vector<Eigen::Vector3f> verts,
                    const std::vector<BoundaryCondition> vert_bcs,
                    const std::vector<Eigen::Vector3i> indices,
                    const std::vector<int> floating_boundary_vids) {
    
    // "Therefore, the new vector fields are not likely to be
    // gradient fields of any scalar functions.To reconstruct a mesh from these
    // vector fields, we need to consider them as the guidance fields in the Poisson equation."
  

    // Compute prerequisites
    std::vector<double> tri_areas;
    std::vector<std::vector<Eigen::Vector3d>> grad_Bs;


    // Compute index_bcs



    // XYZ channal-wise solve
    for (int ch = 0; ch < 3; ch++) {
      // Construct equations

      // Solve
    
    
    }
  }
#endif

std::vector<Eigen::Vector3f> SolvePossionNaive(
    const std::vector<Eigen::Vector3f>& verts,
    const std::vector<Eigen::Vector3i>& indices,
    const std::vector<int>& boundary_vids,
    const std::vector<Eigen::Vector3f>& boundary_verts_new) {
  std::unordered_set<int> boundary_vids_set(boundary_vids.begin(),
                                            boundary_vids.end());

  auto va = GenerateVertexAdjacency(indices, verts.size());

  // Laplacian on the original vertices
  std::vector<Eigen::Vector3f> laps = ComputeMeshLaplacian(verts, indices, va);

  std::vector<Eigen::Vector3f> verts_updated = verts;
  for (size_t i = 0; i < boundary_vids.size(); i++) {
    const int& vid = boundary_vids[i];
    verts_updated[vid] = boundary_verts_new[i];
  }

  std::unordered_map<int, int> org2prm_map;
  std::unordered_map<int, int> prm2org_map;
  int prm_idx = 0;
  for (size_t ovidx = 0; ovidx < verts.size(); ovidx++) {
    if (boundary_vids_set.find(static_cast<int>(ovidx)) !=
        boundary_vids_set.end()) {
      continue;
    }
    org2prm_map[static_cast<int>(ovidx)] = prm_idx;
    prm2org_map[prm_idx] = static_cast<int>(ovidx);
    prm_idx++;
  }

  std::vector<Eigen::Vector3d> b_offset(verts.size() - boundary_vids.size(),
                                        Eigen::Vector3d::Zero());
  std::vector<Eigen::Triplet<double>> triplets;
  const Eigen::Index num_param =
      static_cast<Eigen::Index>(verts.size() - boundary_vids.size());
  int cur_row = 0;
  for (size_t ovidx = 0; ovidx < verts.size(); ovidx++) {
    if (boundary_vids_set.find(static_cast<int>(ovidx)) !=
        boundary_vids_set.end()) {
      continue;
    }

    const auto& vadj = va[ovidx];
    triplets.push_back({cur_row, org2prm_map[static_cast<int>(ovidx)],
                        static_cast<double>(vadj.size())});
    for (const int& vid : vadj) {
      if (boundary_vids_set.find(static_cast<int>(vid)) !=
          boundary_vids_set.end()) {
        // Offset to b
        // Originally - on the rhs. It becomes + on the lfs.
        b_offset[cur_row] += verts_updated[vid].cast<double>();
        continue;
      }
      triplets.push_back({cur_row, org2prm_map[vid], -1.0});
    }

    cur_row++;
  }

  Eigen::SparseMatrix<double> A(num_param, num_param);
  A.setFromTriplets(triplets.begin(), triplets.end());
  // TODO: Is this solver the best?
  Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> solver;
  // Prepare linear system
  solver.compute(A);

  Eigen::VectorXd b(num_param);

  std::array<Eigen::VectorXd, 3> solution_channels;

  for (int ic = 0; ic < 3; ++ic) {
    b.setZero();
    for (size_t pidx = 0; pidx < b_offset.size(); pidx++) {
      b[pidx] =
          static_cast<double>(laps[prm2org_map[static_cast<int>(pidx)]][ic]) +
          b_offset[pidx][ic];
    }
    // Solve for this channel
    solution_channels[ic] = solver.solve(b);
  }

  std::vector<Eigen::Vector3f> verts_poisson = verts_updated;
  for (size_t pidx = 0; pidx < b_offset.size(); pidx++) {
    for (int c = 0; c < 3; c++) {
      verts_poisson[prm2org_map[static_cast<int>(pidx)]][c] =
          static_cast<float>(solution_channels[c][pidx]);
    }
  }

  return verts_poisson;
}

}  // namespace

namespace ugu {

MeshPtr PoissonMeshMerging(const MeshPtr pinned,
                           const std::vector<int> pinned_boundary_vids,
                           const MeshPtr floating,
                           const std::vector<int> floating_boundary_vids) {
  std::vector<Eigen::Vector3f> merged_verts;
  std::vector<Eigen::Vector3i> merged_indices;

  bool ret = PoissonMeshMerging(
      pinned->vertices(), pinned->vertex_indices(), pinned_boundary_vids,
      floating->vertices(), floating->vertex_indices(), floating_boundary_vids,
      merged_verts, merged_indices);

  if (!ret) {
    return nullptr;
  }

  MeshPtr merged = Mesh::Create();

  merged->set_vertices(merged_verts);
  merged->set_vertex_indices(merged_indices);
  merged->set_default_material();

  return merged;
}

bool PoissonMeshMerging(const std::vector<Eigen::Vector3f> pinned_verts,
                        const std::vector<Eigen::Vector3i> pinned_indices,
                        const std::vector<int> pinned_boundary_vids,
                        const std::vector<Eigen::Vector3f> floating_verts,
                        const std::vector<Eigen::Vector3i> floating_indices,
                        const std::vector<int> floating_boundary_vids,
                        std::vector<Eigen::Vector3f>& merged_verts,
                        std::vector<Eigen::Vector3i>& merged_indices) {
  if (pinned_boundary_vids.size() < 3 ||
      pinned_boundary_vids.size() != floating_boundary_vids.size()) {
    return false;
  }

  // Similarity registration
  std::vector<Eigen::Vector3f> pinned_boundary_verts(
      pinned_boundary_vids.size());
  for (size_t i = 0; i < pinned_boundary_verts.size(); i++) {
    pinned_boundary_verts[i] = pinned_verts[pinned_boundary_vids[i]];
  }
  std::vector<Eigen::Vector3f> floating_boundary_verts(
      floating_boundary_vids.size());
  for (size_t i = 0; i < floating_boundary_verts.size(); i++) {
    floating_boundary_verts[i] = floating_verts[floating_boundary_vids[i]];
  }
  Eigen::Affine3d floating2pinned_d =
      FindSimilarityTransformFrom3dCorrespondences(floating_boundary_verts,
                                                   pinned_boundary_verts);
  Eigen::Affine3f floating2pinned = floating2pinned_d.cast<float>();
  std::vector<Eigen::Vector3f> floating_boundary_transed_verts(
      floating_boundary_vids.size());
  for (size_t i = 0; i < floating_boundary_transed_verts.size(); i++) {
    floating_boundary_transed_verts[i] =
        floating2pinned * floating_boundary_verts[i];
  }

  std::vector<Eigen::Vector3f> floating_verts_transed = floating_verts;
  for (size_t i = 0; i < floating_verts_transed.size(); i++) {
    floating_verts_transed[i] = floating2pinned * floating_verts[i];
  }

  std::unordered_set<int> floating_boundary_vids_org_set(
      floating_boundary_vids.begin(), floating_boundary_vids.end());
  // std::vector<int> floating_vids;
  std::unordered_set<int> floating_vids_set;
  std::unordered_map<int, int> old2new_vidmap;
  std::vector<Eigen::Vector3f> floating_verts_no_boundary;
  for (size_t i = 0; i < floating_boundary_verts.size(); i++) {
    int vid = static_cast<int>(i);
    if (floating_boundary_vids_org_set.find(vid) ==
        floating_boundary_vids_org_set.end()) {
      // Not found, add
      floating_vids_set.insert(vid);
      floating_verts_no_boundary.push_back(floating_verts[i]);
      old2new_vidmap.insert(
          {vid, static_cast<int>(floating_vids_set.size() - 1)});
    }
  }

  std::vector<Eigen::Vector3i> floating_indices_no_boundary;
  for (size_t fidx = 0; fidx < floating_indices.size(); fidx++) {
    bool all_found = true;
    Eigen::Vector3i new_face;
    for (int j = 0; j < 3; j++) {
      int vid = floating_indices[fidx][j];
      if (floating_vids_set.find(vid) == floating_vids_set.end()) {
        all_found = false;
        break;
      } else {
        new_face[j] = old2new_vidmap.at(vid);
      }
    }

    if (all_found) {
      floating_indices_no_boundary.push_back(new_face);
    }
  }

  // TODO: Consider scale and rotation
#if 0
  // TODO: Use geodesic distance on the original floating geometry
  std::vector<size_t> vmin_indices;
  std::vector<float> vmin_dists;
  // Nearest scheme
  for (size_t fvid = 0; fvid < floating_verts_no_boundary.size(); fvid++) {
    size_t min_idx(~0);
    float min_dist = std::numeric_limits<float>::max();
    for (size_t pvid = 0; pvid < pinned_boundary_verts.size(); pvid++) {
      // Euclid distance
      float dist =
          (pinned_boundary_verts[pvid] - floating_verts_no_boundary[fvid])
              .norm();
      if (dist < min_dist) {
        min_dist = dist;
        min_idx = pvid;
      }
    }
  }

  // Compute BC (floating_boundary_transed_verts)
  std::vector<BoundaryCondition> bc;

  // Compute BC' (pinned_boundary_verts)
  std::vector<BoundaryCondition> bc_updated;

  // Assign transformations vertex-wise

  // Apply transformations triangle-wise 

  std::vector<Eigen::Vector3f> guidance_vectors;
#endif

  // Current implementation is Naive Poisson, Figure 7 middle
  // Solve Poisson
  std::vector<Eigen::Vector3f> verts_poisson =
      SolvePossionNaive(floating_verts_transed, floating_indices,
                        floating_boundary_vids, pinned_boundary_verts);

  ConnectMeshes(pinned_verts, pinned_indices, pinned_boundary_vids,
                verts_poisson, floating_indices, floating_boundary_vids,
                merged_verts, merged_indices);

  return true;
}

}  // namespace ugu
