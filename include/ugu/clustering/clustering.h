/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 *
 */

#pragma once

#include <vector>

#include "ugu/mesh.h"
#include "ugu/plane.h"

namespace ugu {

void CalcCentroids(const std::vector<Eigen::VectorXf>& points,
                   const std::vector<size_t>& labels,
                   std::vector<Eigen::VectorXf>& centroids, size_t nc);

bool KMeans(const std::vector<Eigen::VectorXf>& points, int num_clusters,
            std::vector<size_t>& labels,
            std::vector<Eigen::VectorXf>& centroids, std::vector<float>& dists,
            std::vector<std::vector<Eigen::VectorXf>>& clustered_points,
            int term_max_iter = 100, float term_unchanged_ratio = 1.f,
            bool init_plus_plus = true, int random_seed = -1);

enum class MeanShiftKernel { GAUSSIAN };

bool MeanShift(const std::vector<Eigen::VectorXf>& points,
               const Eigen::VectorXf& init, float band_width,
               float term_min_threshold, int term_max_iter,
               Eigen::VectorXf& node,
               MeanShiftKernel kernel = MeanShiftKernel::GAUSSIAN);

bool MeanShiftClustering(
    const std::vector<Eigen::VectorXf>& points, int& num_clusters,
    std::vector<size_t>& labels, std::vector<Eigen::VectorXf>& nodes,
    std::vector<std::vector<Eigen::VectorXf>>& clustered_points,
    float band_width, float term_min_threshold, int term_max_iter,
    float cluster_theshold, MeanShiftKernel kernel = MeanShiftKernel::GAUSSIAN);

bool DBSCAN(const std::vector<Eigen::VectorXf>& points, int32_t& num_clusters,
            std::vector<int32_t>& labels,
            std::vector<std::vector<Eigen::VectorXf>>& clustered_points,
            std::vector<Eigen::VectorXf>& noise_points, float epsilon,
            size_t min_nn_points, bool use_kdtree = true);

struct SegmentMeshResult {
  std::vector<uint32_t> cluster_ids;
  std::vector<float> cluster_areas;
  std::vector<std::vector<Eigen::Vector3i>> clusters;
  std::vector<std::vector<Eigen::Vector3f>> cluster_normals;
  std::vector<std::vector<uint32_t>> cluster_fids;
  std::vector<Eigen::Vector3f> cluster_representative_normals;
};

bool SegmentMesh(const std::vector<Eigen::Vector3f>& vertices,
                 const std::vector<Eigen::Vector3i>& faces,
                 const std::vector<Eigen::Vector3f>& face_normals,
                 SegmentMeshResult& res, float angle_limit_deg = 66.4f,
                 float area_weight = 0.f, bool consider_connectiviy = true,
                 bool use_vertex_based_connectivity = false);

bool DisconnectPlaneAndOthers(const std::vector<Eigen::Vector3f>& points,
                              const Planef& plane, float dist_th,
                              std::vector<size_t>& plane_ids,
                              std::vector<size_t>& others_ids,
                              const std::vector<Eigen::Vector3f>& normals =
                                  std::vector<Eigen::Vector3f>(),
                              float angle_th = radians(45.f));

bool DisconnectPlaneAndOthers(const std::vector<Eigen::Vector3f>& vertices,
                              const std::vector<Eigen::Vector3i>& indices,
                              const Planef& plane, float dist_th,
                              std::vector<size_t>& plane_vids,
                              std::vector<size_t>& others_vids,
                              std::vector<size_t>& boundary_vids,
                              std::vector<Eigen::Vector3f>& plane_vertices,
                              std::vector<Eigen::Vector3i>& plane_indices,
                              std::vector<Eigen::Vector3f>& others_vertices,
                              std::vector<Eigen::Vector3i>& others_indices,
                              const std::vector<Eigen::Vector3f>& normals =
                                  std::vector<Eigen::Vector3f>(),
                              float angle_th = radians(45.f),
                              bool keep_boundary_both = true);

bool DisconnectPlaneAndOthers(const Mesh& mesh, const Planef& plane,
                              float dist_th, std::vector<size_t>& plane_vids,
                              std::vector<size_t>& others_vids,
                              std::vector<size_t>& boundary_vids,
                              Mesh& plane_mesh, Mesh& others_mesh,
                              float angle_th = radians(45.f),
                              bool keep_boundary_both = true);

}  // namespace ugu
