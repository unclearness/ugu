/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 *
 */

#include "ugu/clustering/clustering.h"

#include <random>
#include <unordered_set>

#include "ugu/common.h"

namespace {
void CalcCentroids(const std::vector<Eigen::VectorXf>& points,
                   const std::vector<size_t>& labels,
                   std::vector<Eigen::VectorXf>& centroids, size_t nc) {
  centroids.resize(nc);
  Eigen::VectorXf zero = Eigen::VectorXf(centroids[0].rows());
  zero.setZero();
  std::fill(centroids.begin(), centroids.end(), zero);
  std::vector<size_t> centroid_counts(nc, 0);

  for (size_t i = 0; i < points.size(); i++) {
    size_t l = labels[i];
    centroids[l] += points[i];
    centroid_counts[l] += 1;
  }

  Eigen::VectorXf inf = Eigen::VectorXf(centroids[0].rows());
  inf.setConstant(std::numeric_limits<float>::infinity());
  for (size_t i = 0; i < nc; i++) {
    if (centroid_counts[i] < 1) {
      // Set inf to illegal point
      centroids[i] = inf;
      continue;
    }
    centroids[i] /= static_cast<float>(centroid_counts[i]);
  }

  // for (size_t i = 0; i < nc; i++) {
  //  ugu::LOGI("%d %d (%f, %f, %f)\n", i, centroid_counts[i], centroids[i][0],
  //            centroids[i][1], centroids[i][2]);
  //}
}

void AssignLabelForInvalidClusters(const std::vector<Eigen::VectorXf>& points,
                                   std::vector<size_t>& labels,
                                   std::vector<Eigen::VectorXf>& centroids) {
  std::uniform_int_distribution<int> dstr(0,
                                          static_cast<int>(labels.size() - 1));
  std::default_random_engine engine(0);
  for (size_t i = 0; i < centroids.size(); i++) {
    if (std::isnormal(centroids[i][0])) {
      continue;
    }
    // Set this label to a random point
    int index = dstr(engine);
    labels[index] = i;
    centroids[i] = points[index];
  }
}

void InitKMeansPlusPlus(const std::vector<Eigen::VectorXf>& points,
                        std::vector<Eigen::VectorXf>& centroids,
                        std::default_random_engine& engine) {
  std::uniform_int_distribution<int> point_dstr(
      0, static_cast<int>(points.size()) - 1);
  std::uniform_real_distribution<double> sample_dstr(0.0, 1.0);
  // https://en.wikipedia.org/wiki/K-means%2B%2B
  size_t new_index = static_cast<size_t>(point_dstr(engine));
  centroids[0] = points[new_index];
  std::vector<double> init_dists(points.size(), 0.0);
  std::vector<double> dists_dstr(points.size(), 0.0);
  double init_dists_sum = 0.0;
  std::unordered_set<size_t> selected;
  selected.insert(new_index);
  for (size_t i = 1; i < centroids.size(); i++) {
    init_dists_sum = 0.0;
    std::fill(init_dists.begin(), init_dists.end(), 0.0);
    std::fill(dists_dstr.begin(), dists_dstr.end(), 0.0);
    // Calculate distances
    for (size_t j = 0; j < points.size(); j++) {
      // Skip if already selected
      if (selected.find(j) != selected.end()) {
        continue;
      }
      float min_dist = (centroids[0] - points[j]).norm();
      for (size_t ii = 1; ii < i; ii++) {
        float dist = (centroids[ii] - points[j]).norm();
        if (dist < min_dist) {
          min_dist = dist;
        }
      }
      double min_dist_d = static_cast<double>(min_dist);
      double squared_min_dist = min_dist_d * min_dist_d;
      init_dists[j] = squared_min_dist;
      init_dists_sum += squared_min_dist;
    }

    // Make distribution
    dists_dstr[0] = init_dists[0] / init_dists_sum;
    for (size_t j = 1; j < points.size(); j++) {
      dists_dstr[j] = dists_dstr[j - 1] + init_dists[j] / init_dists_sum;
    }

    // Sample new cluster centroid
    while (true) {
      // https://en.wikipedia.org/wiki/Inverse_transform_sampling
      double sampled_val = sample_dstr(engine);
      auto it =
          std::lower_bound(dists_dstr.begin(), dists_dstr.end(), sampled_val);
      new_index = std::distance(dists_dstr.begin(), it);
      // ugu::LOGI("new_index %d %f %f\n", new_index, sampled_val, *it);
      // If selects exisiting index, sample again
      if (selected.find(new_index) != selected.end()) {
        continue;
      }

      selected.insert(new_index);
      centroids[i] = points[new_index];
      break;
    }
  }
}

}  // namespace

namespace ugu {

bool KMeans(const std::vector<Eigen::VectorXf>& points, int num_clusters,
            std::vector<size_t>& labels,
            std::vector<Eigen::VectorXf>& centroids, std::vector<float>& dists,
            std::vector<std::vector<Eigen::VectorXf>>& clustered_points,
            int term_max_iter, float term_unchanged_ratio, bool init_plus_plus,
            int random_seed) {
  if (num_clusters <= 1 || points.size() < 2 || points.size() < num_clusters) {
    return false;
  }

  std::uniform_int_distribution<int> cluster_dstr(0, num_clusters - 1);

  std::random_device seed_gen;
  std::default_random_engine engine(
      random_seed < 0 ? seed_gen() : static_cast<size_t>(random_seed));

  // Init
  size_t nc = static_cast<size_t>(num_clusters);
  labels.resize(points.size(), size_t(~0));
  Eigen::VectorXf zero = Eigen::VectorXf(points[0].rows());
  zero.setZero();
  centroids.resize(nc, zero);
  dists.resize(points.size());
  clustered_points.resize(nc);
  if (init_plus_plus) {
    InitKMeansPlusPlus(points, centroids, engine);
  } else {
    // At least one sample for each cluster
    for (size_t i = 0; i < nc; i++) {
      labels[i] = i;
    }
    for (size_t i = nc; i < points.size(); i++) {
      labels[i] = static_cast<size_t>(cluster_dstr(engine));
    }

    CalcCentroids(points, labels, centroids, nc);
    AssignLabelForInvalidClusters(points, labels, centroids);
  }

  bool is_term_max_iter = term_max_iter > 0;
  bool is_term_unchanged_ratio =
      0.f < term_unchanged_ratio && term_unchanged_ratio <= 1.f;
  const int default_term_max_iter = 100;

  int iter = 0;
  float unchanged_ratio = 0.0f;

  auto terminated = [&] {
    if (!is_term_max_iter && !is_term_unchanged_ratio) {
      if (default_term_max_iter <= iter) {
        return true;
      } else {
        return false;
      }
    }

    if (is_term_max_iter && term_max_iter <= iter) {
      return true;
    }

    if (is_term_unchanged_ratio && term_unchanged_ratio <= unchanged_ratio) {
      return true;
    }

    return false;
  };

  while (!terminated()) {
    size_t unchanged_num = 0;

    // https://en.wikipedia.org/wiki/K-means_clustering

    // Assignment step
    for (size_t i = 0; i < points.size(); i++) {
      float min_dist = (centroids[0] - points[i]).norm();
      size_t min_c = 0;
      for (size_t c = 1; c < nc; c++) {
        float dist = (centroids[c] - points[i]).norm();
        if (dist < min_dist) {
          min_dist = dist;
          min_c = c;
        }
      }
      dists[i] = min_dist;
      if (labels[i] == min_c) {
        unchanged_num++;
      }
      labels[i] = min_c;
    }

    // Update step
    CalcCentroids(points, labels, centroids, nc);
    AssignLabelForInvalidClusters(points, labels, centroids);
    iter++;
    unchanged_ratio =
        static_cast<float>(unchanged_num) / static_cast<float>(points.size());
    // ugu::LOGI("%d %f \n", iter, unchanged_ratio);
  }

  //  clustered_points
  for (size_t i = 0; i < points.size(); i++) {
    clustered_points[labels[i]].push_back(points[i]);
  }

  return true;
}

}  // namespace ugu
