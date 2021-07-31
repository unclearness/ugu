/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 *
 */

#pragma once

#include "Eigen/Core"

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
            size_t min_nn_points);

}  // namespace ugu
