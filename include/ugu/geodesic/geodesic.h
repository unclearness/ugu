/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 *
 */

#pragma once

#include "Eigen/SparseCore"
#include "ugu/mesh.h"

namespace ugu {

enum class GeodesicComputeMethod { DIJKSTRA };

bool ComputeGeodesicDistance(
    const Mesh& mesh, int src_vid, Eigen::SparseMatrix<float>& edge_dists,
    std::vector<double>& dists, std::vector<int>& min_path_edges,
    GeodesicComputeMethod method = GeodesicComputeMethod::DIJKSTRA);

}  // namespace ugu
