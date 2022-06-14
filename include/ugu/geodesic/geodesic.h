/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 *
 */

#pragma once

#include "Eigen/SparseCore"
#include "ugu/mesh.h"

namespace ugu {

enum class GeodesicComputeMethod { DIJKSTRA, FAST_MARCHING_METHOD };

bool ComputeGeodesicDistance(
    const Mesh& mesh, int src_vid, Eigen::SparseMatrix<float>& edge_dists,
    std::vector<double>& dists, std::vector<int>& min_path_edges,
    GeodesicComputeMethod method = GeodesicComputeMethod::FAST_MARCHING_METHOD);

bool ComputeGeodesicDistance(
    const Mesh& mesh, std::vector<int> src_vids,
    Eigen::SparseMatrix<float>& edge_dists, std::vector<double>& dists,
    std::vector<int>& min_path_edges,
    GeodesicComputeMethod method = GeodesicComputeMethod::FAST_MARCHING_METHOD);

}  // namespace ugu
