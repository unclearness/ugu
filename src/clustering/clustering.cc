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

#if 0

std::function<Eigen::VectorXf (const Eigen::VectorXf&)> GetKernelGrad(
    ugu::MeanShiftKernel kernel) {
  if (kernel == ugu::MeanShiftKernel::GAUSSIAN) {
    return [&](const Eigen::VectorXf& x) {
      return -x * std::exp(-x.squaredNorm() * 0.5f);
    };
  }

  return [&](const Eigen::VectorXf& x) {
    return x;
  };
}
#endif  // 0

#if 0
std::function<double(const Eigen::VectorXf&)> GetKernelGrad(
    ugu::MeanShiftKernel kernel) {
  if (kernel == ugu::MeanShiftKernel::GAUSSIAN) {
    return [&](const Eigen::VectorXf& x) {
      return -x .norm() * std::exp(-x.squaredNorm() * 0.5f);
    };
  }

  return [&](const Eigen::VectorXf& x) { return x.norm(); };
}
#endif

std::function<double(double)> GetKernelGrad(ugu::MeanShiftKernel kernel) {
  if (kernel == ugu::MeanShiftKernel::GAUSSIAN) {
    return [&](double x) { return -x * std::exp(-x * x * 0.5); };
  }

  ugu::LOGE("This kernel is not supported\n");
  return [&](double x) { return x; };
}

// TODO: kd-tree
std::unordered_set<int32_t> RangeQuery(
    const std::vector<Eigen::VectorXf>& points, int32_t q, float epsilon,
    bool remove_q = true) {
  std::unordered_set<int32_t> nn_set;
  // size_t q_ = static_cast<size_t>(q);
  for (size_t i = 0; i < points.size(); i++) {
    float dist = (points[i] - points[q]).norm();
    if (dist < epsilon) {
      nn_set.insert(static_cast<int32_t>(i));
    }
  }

  if (remove_q) {
    nn_set.erase(q);
  }

  return nn_set;
}

}  // namespace

namespace ugu {

void CalcCentroids(const std::vector<Eigen::VectorXf>& points,
                   const std::vector<size_t>& labels,
                   std::vector<Eigen::VectorXf>& centroids, size_t nc) {
  centroids.resize(nc);
  Eigen::VectorXf zero = Eigen::VectorXf(points[0].rows());
  zero.setZero();
  std::fill(centroids.begin(), centroids.end(), zero);
  std::vector<size_t> centroid_counts(nc, 0);

  for (size_t i = 0; i < points.size(); i++) {
    size_t l = labels[i];
    if (nc < l) {
      continue;
    }
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

// Implementation reference:
// http://takashiijiri.com/study/ImgProc/MeanShift.htm
bool MeanShift(const std::vector<Eigen::VectorXf>& points,
               const Eigen::VectorXf& init, float band_width,
               float term_min_threshold, int term_max_iter,
               Eigen::VectorXf& node, MeanShiftKernel kernel) {
  node = init;

  auto grad_func = GetKernelGrad(kernel);

  int iter = 0;
  float diff = std::numeric_limits<float>::max();
  bool is_term_max_iter = term_max_iter > 0;
  bool is_term_min_threshold = 0.f < term_min_threshold;
  const int default_term_max_iter = 100;
  auto terminated = [&] {
    if (!is_term_max_iter && !is_term_min_threshold) {
      if (default_term_max_iter <= iter) {
        return true;
      } else {
        return false;
      }
    }

    if (is_term_max_iter && term_max_iter <= iter) {
      return true;
    }

    if (is_term_min_threshold && diff < term_min_threshold) {
      return true;
    }

    return false;
  };

  Eigen::VectorXf prev_node(node);

  std::vector<double> coeffs(points.size());
  const double inv_sq_h = 1 / (band_width * band_width);
  while (!terminated()) {
    prev_node = node;

    // ugu::LOGI("prev (%f %f %f)\n", prev_node[0], prev_node[1], prev_node[2]);
    // ugu::LOGI("node (%f %f %f)\n", node[0], node[1], node[2]);

    double denom = 0;
    node.setZero();
    for (size_t i = 0; i < points.size(); i++) {
      double t = (prev_node - points[i]).squaredNorm() * inv_sq_h;
      coeffs[i] = -grad_func(t);
      node += points[i] * coeffs[i];
      denom += coeffs[i];
    }

    node /= static_cast<float>(denom);

    iter++;
    diff = (prev_node - node).norm();
    // ugu::LOGI("%d %f \n", iter, diff);
    // ugu::LOGI("prev (%f %f %f)\n", prev_node[0], prev_node[1], prev_node[2]);
    // ugu::LOGI("node (%f %f %f)\n", node[0], node[1], node[2]);
  }

  return true;
}

bool MeanShiftClustering(
    const std::vector<Eigen::VectorXf>& points, int& num_clusters,
    std::vector<size_t>& labels, std::vector<Eigen::VectorXf>& nodes,
    std::vector<std::vector<Eigen::VectorXf>>& clustered_points,
    float band_width, float term_min_threshold, int term_max_iter,
    float cluster_theshold, MeanShiftKernel kernel) {
  labels.resize(points.size(), size_t(~0));
  nodes.resize(points.size());

  // Run MeanShift for each point
  // TODO: Acceleration by kd-tree. Distance query targets are always "points"
#ifdef UGU_USE_OPENMP
#pragma omp parallel for
#endif
  for (long int i = 0; i < static_cast<long int>(points.size()); i++) {
    const auto& p = points[i];
    MeanShift(points, p, band_width, term_min_threshold, term_max_iter,
              nodes[i], kernel);
  }

  // Clustering extreme points
  num_clusters = 1;
  labels[0] = 0;
  clustered_points.push_back({points[0]});
  for (size_t i = 1; i < nodes.size(); i++) {
    // Check distance
    bool merged = false;
    for (size_t ii = 0; ii < i; ii++) {
      // If find near cluster, merge
      float dist = (nodes[i] - nodes[ii]).norm();
      // ugu::LOGI("%f \n", dist);
      if (dist < cluster_theshold) {
        // merge
        labels[i] = labels[ii];
        clustered_points[labels[i]].push_back(points[i]);

        merged = true;
        break;
      }
    }

    if (merged) {
      continue;
    }

    // Add new cluster
    num_clusters++;
    labels[i] = static_cast<size_t>(num_clusters) - 1;
    clustered_points.push_back({points[i]});
  }

  return true;
}

// A naive implementatio of DBSCAN
// https://en.wikipedia.org/wiki/DBSCAN
bool DBSCAN(const std::vector<Eigen::VectorXf>& points, int32_t& num_clusters,
            std::vector<int32_t>& labels,
            std::vector<std::vector<Eigen::VectorXf>>& clustered_points,
            std::vector<Eigen::VectorXf>& noise_points, float epsilon,
            size_t min_nn_points) {
  constexpr size_t DBSCAN_MAX_SIZE = std::numeric_limits<int32_t>::max();

  if (DBSCAN_MAX_SIZE < points.size()) {
    return false;
  }

  num_clusters = 0;
  constexpr int32_t illegal = std::numeric_limits<int32_t>::lowest();
  constexpr int32_t noise = -1;
  labels.resize(points.size(), illegal);
  for (size_t i = 0; i < points.size(); i++) {
    if (labels[i] != illegal) {
      // Already processed
      continue;
    }

    const std::unordered_set<int32_t> nn_ids_seed =
        RangeQuery(points, static_cast<int32_t>(i), epsilon);

    if (nn_ids_seed.size() < min_nn_points) {
      // Skip noise
      labels[i] = noise;
      continue;
    }

    // Current cluster label
    int32_t c = num_clusters;
    num_clusters += 1;
    labels[i] = c;

    std::unordered_set<int32_t> seed_set = nn_ids_seed;
    seed_set.insert(static_cast<int32_t>(i));

    std::unordered_set<int32_t> to_remove;
    while (!seed_set.empty()) {
      std::unordered_set<int32_t> expanded;
      for (const int32_t& q : seed_set) {
        if (labels[q] == noise) {
          labels[q] = c;
        } else if (labels[q] != illegal) {
          // Skip already processed points
          to_remove.insert(q);
          continue;
        }

        labels[q] = c;

        const std::unordered_set<int32_t> nn_ids =
            RangeQuery(points, q, epsilon);

        if (min_nn_points <= nn_ids.size()) {
          expanded.insert(nn_ids.begin(), nn_ids.end());
        }
      }

      // Merge
      seed_set.insert(expanded.begin(), expanded.end());

      // Remove
      for (const auto& r : to_remove) {
        seed_set.erase(r);
      }

      // expanded.clear();
    }
  }

  noise_points.clear();
  clustered_points.resize(num_clusters);
  for (size_t i = 0; i < points.size(); i++) {
    const auto& l = labels[i];
    const auto& p = points[i];
    if (l < 0) {
      noise_points.push_back(p);
    } else {
      clustered_points[l].push_back(p);
    }
  }

  return true;
}
}  // namespace ugu
