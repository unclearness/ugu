/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 *
 */

#include "ugu/clustering/clustering.h"

#include <random>
#include <unordered_set>

#include "ugu/accel/kdtree.h"
#include "ugu/common.h"
#include "ugu/face_adjacency.h"
#include "ugu/util/geom_util.h"
#include "ugu/util/math_util.h"

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

std::function<double(double)> GetKernelGrad(ugu::MeanShiftKernel kernel) {
  if (kernel == ugu::MeanShiftKernel::GAUSSIAN) {
    return [&](double x) { return -x * std::exp(-x * x * 0.5); };
  }

  ugu::LOGE("This kernel is not supported\n");
  return [&](double x) { return x; };
}

std::unordered_set<int32_t> RangeQueryNaive(
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

struct SegmentFace {
  float area;
  size_t org_id;
  Eigen::Vector3i efa;
  Eigen::Vector3f no;
  bool valid;
  bool tag;
};

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
  if (num_clusters <= 1 || points.size() < 2 ||
      points.size() < static_cast<size_t>(num_clusters)) {
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
            size_t min_nn_points, bool use_kdtree) {
  constexpr size_t DBSCAN_MAX_SIZE = std::numeric_limits<int32_t>::max();

  if (points.empty() || DBSCAN_MAX_SIZE < points.size()) {
    return false;
  }

  KdTreePtr<float, Eigen::Dynamic> kdtree;
  if (use_kdtree) {
    kdtree = GetDefaultKdTreeDynamic<float>();

    kdtree->SetData(points);
    if (!kdtree->Build()) {
      return false;
    }
  }

  auto nn_search_func = [&](size_t i) {
    if (use_kdtree) {
      auto results =
          kdtree->SearchRadius(points[i], static_cast<double>(epsilon));

      std::unordered_set<int32_t> nn_set;
      std::transform(results.begin(), results.end(),
                     std::inserter(nn_set, nn_set.begin()),
                     [&](const KdTreeSearchResult& res) {
                       return static_cast<int32_t>(res.index);
                     });

      nn_set.erase(i);
      return nn_set;
    } else {
      std::unordered_set<int32_t> nn_set;
      nn_set = RangeQueryNaive(points, static_cast<int32_t>(i), epsilon);
      return nn_set;
    }
  };

  num_clusters = 0;
  constexpr int32_t illegal = std::numeric_limits<int32_t>::lowest();
  constexpr int32_t noise = -1;
  labels.resize(points.size(), illegal);
  for (size_t i = 0; i < points.size(); i++) {
    if (labels[i] != illegal) {
      // Already processed
      continue;
    }

    std::unordered_set<int32_t> nn_ids_seed = nn_search_func(i);

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

        const std::unordered_set<int32_t> nn_ids = nn_search_func(q);

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

bool SegmentMesh(const std::vector<Eigen::Vector3f>& vertices,
                 const std::vector<Eigen::Vector3i>& faces,
                 const std::vector<Eigen::Vector3f>& face_normals,
                 SegmentMeshResult& res, float angle_limit_deg,
                 float area_weight, bool consider_connectiviy,
                 bool use_vertex_based_connectivity) {
  const float angle_limit_rad = std::clamp(radians(angle_limit_deg), 1e-4f,
                                           static_cast<float>(ugu::pi * 0.5));

  const float project_angle_limit_cos = std::cos(angle_limit_rad);
  const float project_angle_limit_half_cos = std::cos(angle_limit_rad / 2);

  std::vector<SegmentFace> thick_faces(faces.size());
  // Sort by face area
  std::vector<float> face_areas;
  std::transform(faces.begin(), faces.end(), std::back_inserter(face_areas),
                 [&](const Eigen::Vector3i& f) {
                   const auto& v0 = vertices[f[0]];
                   const auto& v1 = vertices[f[1]];
                   const auto& v2 = vertices[f[2]];
                   const float area =
                       std::abs(((v2 - v0).cross(v1 - v0)).norm()) * 0.5f;
                   return area;
                 });
  constexpr float area_ignore = 1e-12f;
  for (size_t i = 0; i < thick_faces.size(); i++) {
    thick_faces[i].area = face_areas[i];
    thick_faces[i].org_id = i;
    thick_faces[i].efa = faces[i];
    thick_faces[i].no = face_normals[i];
    thick_faces[i].valid = thick_faces[i].area > area_ignore;
    thick_faces[i].tag = false;
  }

  std::sort(thick_faces.begin(), thick_faces.end(),
            [&](const SegmentFace& a, const SegmentFace& b) {
              if (!((a.area > area_ignore) || (b.area > area_ignore))) {
                return false;
              }
              if (a.area < b.area) {
                return true;
              }
              if (a.area > b.area) {
                return false;
              }
              return false;
            });

  // Step 1: Normal clustering
  std::vector<Eigen::Vector3f> project_normal_array;
  {
    Eigen::Vector3f project_normal = thick_faces[0].no;

    std::vector<SegmentFace> project_thick_faces;
    while (true) {
      for (int64_t f_index = thick_faces.size() - 1; f_index >= 0; f_index--) {
        if (!thick_faces[f_index].valid || thick_faces[f_index].tag) {
          continue;
        }

        if (thick_faces[f_index].no.dot(project_normal) >
            project_angle_limit_half_cos) {
          project_thick_faces.push_back(thick_faces[f_index]);
          thick_faces[f_index].tag = true;
        }
      }

      Eigen::Vector3f average_normal = Eigen::Vector3f::Zero();

      if (area_weight <= 0.0f) {
        for (size_t f_proj_index = 0; f_proj_index < project_thick_faces.size();
             f_proj_index++) {
          const auto& tf = project_thick_faces[f_proj_index];
          average_normal += tf.no;
        }
      } else if (area_weight >= 1.0f) {
        for (size_t f_proj_index = 0; f_proj_index < project_thick_faces.size();
             f_proj_index++) {
          const auto& tf = project_thick_faces[f_proj_index];
          average_normal += (tf.no * tf.area);
        }
      } else {
        for (size_t f_proj_index = 0; f_proj_index < project_thick_faces.size();
             f_proj_index++) {
          const auto& tf = project_thick_faces[f_proj_index];
          const float area_blend =
              (tf.area * area_weight) + (1.0f - area_weight);
          average_normal += (tf.no * area_blend);
        }
      }

      if (average_normal.norm() > 0.f) {
        average_normal.normalize();
        project_normal_array.push_back(average_normal);
      }

      float angle_best = 1.0f;
      size_t angle_best_index = 0;

      for (int64_t f_index = thick_faces.size() - 1; f_index >= 0; f_index--) {
        if (!thick_faces[f_index].valid || thick_faces[f_index].tag) {
          continue;
        }

        float angle_test = -1.0f;
        for (size_t p_index = 0; p_index < project_normal_array.size();
             p_index++) {
          angle_test = std::max(angle_test, project_normal_array[p_index].dot(
                                                thick_faces[f_index].no));
        }

        if (angle_test < angle_best) {
          angle_best = angle_test;
          angle_best_index = f_index;
        }
      }

      if (angle_best < project_angle_limit_cos) {
        project_normal = thick_faces[angle_best_index].no;
        project_thick_faces.clear();
        project_thick_faces.push_back(thick_faces[angle_best_index]);
        thick_faces[angle_best_index].tag = true;
      } else {
        if (project_normal_array.size() >= 1) {
          break;
        }
      }
    }
  }

  // Step 2: Face clustering by the clustered normals
  std::vector<std::vector<SegmentFace>> thickface_project_groups(
      project_normal_array.size());
  for (int64_t f_index = thick_faces.size() - 1; f_index >= 0; f_index--) {
    const auto& f_normal = thick_faces[f_index].no;

    float angle_best = f_normal.dot(project_normal_array[0]);
    size_t angle_best_index = 0;

    for (size_t p_index = 1; p_index < project_normal_array.size(); p_index++) {
      const float angle_test = f_normal.dot(project_normal_array[p_index]);
      if (angle_test > angle_best) {
        angle_best = angle_test;
        angle_best_index = p_index;
      }
    }

    thickface_project_groups[angle_best_index].push_back(thick_faces[f_index]);
  }

  res.cluster_ids.clear();
  res.cluster_areas.clear();
  res.clusters.clear();
  res.cluster_normals.clear();
  res.cluster_fids.clear();
  res.cluster_representative_normals.clear();
  res.cluster_ids.resize(faces.size(), uint32_t(~0));

  uint32_t cid = 0;

  if (!consider_connectiviy) {
    // Step 2.5: If don't consider connectivity, return the face clusters
    for (size_t pid = 0; pid < thickface_project_groups.size(); pid++) {
      std::vector<Eigen::Vector3i> cluster;
      std::vector<Eigen::Vector3f> cluster_normal;
      std::vector<uint32_t> cluster_fid;

      for (const auto& tmp : thickface_project_groups[pid]) {
        cluster.push_back(tmp.efa);
        cluster_normal.push_back(tmp.no);
        cluster_fid.push_back(tmp.org_id);
        res.cluster_ids[tmp.org_id] = static_cast<uint32_t>(pid);
      }

      res.cluster_fids.push_back(cluster_fid);
      res.clusters.push_back(cluster);
      res.cluster_normals.push_back(cluster_normal);
      res.cluster_representative_normals.push_back(project_normal_array[pid]);

      double cluster_area = 0.0;
      for (const auto& fid : cluster_fid) {
        cluster_area += face_areas[fid];
      }
      res.cluster_areas.push_back(static_cast<float>(cluster_area));
    }
    return true;
  }

  // Step 3: Find connected components in the face clusters
  for (size_t pid = 0; pid < thickface_project_groups.size(); pid++) {
    std::vector<uint32_t> sub_face_ids;
    std::unordered_set<uint32_t> to_process_fids;
    std::unordered_map<uint32_t, uint32_t>
        org2sub;  // Original face id to face id in the cluster
    for (size_t i = 0; i < thickface_project_groups[pid].size(); i++) {
      auto fid = static_cast<uint32_t>(thickface_project_groups[pid][i].org_id);
      to_process_fids.insert(fid);
      sub_face_ids.push_back(fid);
      org2sub.insert({fid, i});
    }

    auto [sub_vertices, sub_faces] =
        ExtractSubGeom(vertices, faces, sub_face_ids);

    auto [geo_clusters, geo_non_orphans, geo_orphans, geo_clusters_f] =
        ClusterByConnectivity(sub_faces, sub_vertices.size(),
                              use_vertex_based_connectivity);

    // Face id in the cluster to the connected component id
    std::vector<uint32_t> fid2geocluster(sub_faces.size(), uint32_t(~0));
    for (size_t i = 0; i < geo_clusters_f.size(); i++) {
      uint32_t geo_cid = static_cast<uint32_t>(i);
      for (const int& fid : geo_clusters_f[geo_cid]) {
        fid2geocluster[fid] = geo_cid;
      }
    }

    uint32_t cur_fid =
        static_cast<uint32_t>(thickface_project_groups[pid][0].org_id);
    to_process_fids.erase(cur_fid);
    res.cluster_ids[cur_fid] = cid;

    while (true) {
      std::vector<Eigen::Vector3i> cluster{faces[cur_fid]};
      std::vector<Eigen::Vector3f> cluster_normal{face_normals[cur_fid]};
      std::vector<uint32_t> cluster_fid{cur_fid};
      res.cluster_ids[cur_fid] = cid;

      uint32_t geo_cid = fid2geocluster[org2sub[cur_fid]];

      std::unordered_set<uint32_t> processed;
      auto update_func = [&](uint32_t fid) {
        processed.insert(fid);
        cluster_fid.push_back(fid);
        cluster.push_back(faces[fid]);
        cluster_normal.push_back(face_normals[fid]);
        res.cluster_ids[fid] = cid;
      };

      update_func(cur_fid);

      for (const auto& fid : to_process_fids) {
        if (geo_cid == fid2geocluster[org2sub[fid]]) {
          update_func(fid);
        }
      }

      for (const auto& fid : processed) {
        to_process_fids.erase(fid);
      }

      res.cluster_fids.push_back(cluster_fid);
      res.clusters.push_back(cluster);
      res.cluster_normals.push_back(cluster_normal);
      res.cluster_representative_normals.push_back(project_normal_array[pid]);
      double cluster_area = 0.0;
      for (const auto& fid : cluster_fid) {
        cluster_area += face_areas[fid];
      }
      res.cluster_areas.push_back(static_cast<float>(cluster_area));

      cid++;

      if (to_process_fids.empty()) {
        break;
      }

      // Find remaining fid
      cur_fid = *to_process_fids.begin();
    }
  }

  return true;
}

bool DisconnectPlaneAndOthers(const std::vector<Eigen::Vector3f>& points,
                              const Planef& plane, float dist_th,
                              std::vector<size_t>& plane_ids,
                              std::vector<size_t>& others_ids,
                              const std::vector<Eigen::Vector3f>& normals,
                              float angle_th) {
  plane_ids.clear();
  others_ids.clear();

  const bool with_normal = points.size() == normals.size();

  const float dot_th = std::cos(angle_th);

  for (size_t i = 0; i < points.size(); i++) {
    const auto& p = points[i];
    const float dist = plane.SignedDist(p);

    if (std::abs(dist) <= dist_th &&
        (!with_normal || (with_normal && dot_th <= normals[i].dot(plane.n)))) {
      plane_ids.push_back(i);
    } else {
      others_ids.push_back(i);
    }
  }

  return true;
}

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
                              const std::vector<Eigen::Vector3f>& normals,
                              float angle_th, bool keep_boundary_both) {
  DisconnectPlaneAndOthers(vertices, plane, dist_th, plane_vids, others_vids,
                           normals, angle_th);

  if (keep_boundary_both) {
    FaceAdjacency fa;
    fa.Init(static_cast<int>(vertices.size()), indices);
    Adjacency va = fa.GenerateVertexAdjacency();
    std::set<size_t> others_vid_set, org_others_vid_set;
    for (const auto& other_vid : others_vids) {
      others_vid_set.insert(other_vid);
      org_others_vid_set.insert(other_vid);
      // Add one ring neighbor vid to include boundary vid in both
      for (const int& neighbor_vid : va[other_vid]) {
        others_vid_set.insert(static_cast<size_t>(neighbor_vid));
      }
    }

    others_vids.clear();
    std::copy(others_vid_set.begin(), others_vid_set.end(),
              std::back_inserter(others_vids));

    boundary_vids.clear();
    std::set_difference(others_vid_set.begin(), others_vid_set.end(),
                        org_others_vid_set.begin(), org_others_vid_set.end(),
                        std::back_inserter(boundary_vids));
  }

  {
    std::vector<bool> plane_vertices_table(vertices.size(), false);
    for (const auto& vid : plane_vids) {
      plane_vertices_table[vid] = true;
    }
    auto [num_removed, valid_table, valid_vertices, valid_vertex_colors,
          valid_uv, valid_indices, valid_face_table, with_uv,
          with_vertex_color] =
        RemoveVerticesBase(vertices, indices, {}, {}, {}, plane_vertices_table);

    plane_vertices = std::move(valid_vertices);
    plane_indices = std::move(valid_indices);
  }

  {
    std::vector<bool> others_vertices_table(vertices.size(), false);
    for (const auto& vid : others_vids) {
      others_vertices_table[vid] = true;
    }
    auto [num_removed, valid_table, valid_vertices, valid_vertex_colors,
          valid_uv, valid_indices, valid_face_table, with_uv,
          with_vertex_color] = RemoveVerticesBase(vertices, indices, {}, {}, {},
                                                  others_vertices_table);

    others_vertices = std::move(valid_vertices);
    others_indices = std::move(valid_indices);
  }

  return true;
}

bool DisconnectPlaneAndOthers(const Mesh& mesh, const Planef& plane,
                              float dist_th, std::vector<size_t>& plane_vids,
                              std::vector<size_t>& others_vids,
                              std::vector<size_t>& boundary_vids,
                              Mesh& plane_mesh, Mesh& others_mesh,
                              float angle_th, bool keep_boundary_both) {
  std::vector<Eigen::Vector3f> plane_vertices;
  std::vector<Eigen::Vector3i> plane_indices;
  std::vector<Eigen::Vector3f> others_vertices;
  std::vector<Eigen::Vector3i> others_indices;

  DisconnectPlaneAndOthers(mesh.vertices(), mesh.vertex_indices(), plane,
                           dist_th, plane_vids, others_vids, boundary_vids,
                           plane_vertices, plane_indices, others_vertices,
                           others_indices, mesh.normals(), angle_th,
                           keep_boundary_both);

  plane_mesh.Clear();
  plane_mesh.set_vertices(plane_vertices);
  plane_mesh.set_vertex_indices(plane_indices);
  plane_mesh.set_default_material();

  others_mesh.Clear();
  others_mesh.set_vertices(others_vertices);
  others_mesh.set_vertex_indices(others_indices);
  others_mesh.set_default_material();

  return true;
}

}  // namespace ugu
