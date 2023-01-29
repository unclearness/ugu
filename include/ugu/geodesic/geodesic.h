/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 *
 */

#pragma once

#include "ugu/mesh.h"

#ifdef _WIN32
#pragma warning(push, UGU_EIGEN_WARNING_LEVEL)
#endif
#include "Eigen/SparseCore"
#ifdef _WIN32
#pragma warning(pop)
#endif

namespace ugu {

enum class GeodesicComputeMethod { DIJKSTRA, FAST_MARCHING_METHOD };

bool ComputeGeodesicDistance(
    const Mesh& mesh, int src_vid, Eigen::SparseMatrix<float>& edge_dists,
    std::vector<double>& dists, std::vector<int>& min_path_edges,
    GeodesicComputeMethod method = GeodesicComputeMethod::FAST_MARCHING_METHOD);

bool ComputeGeodesicDistance(
    const Mesh& mesh, const std::vector<int>& src_vids,
    Eigen::SparseMatrix<float>& edge_dists, std::vector<double>& dists,
    std::vector<int>& min_path_edges,
    GeodesicComputeMethod method = GeodesicComputeMethod::FAST_MARCHING_METHOD);

}  // namespace ugu
