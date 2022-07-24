/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/plane.h"

#include <random>

#include "ugu/parameterize/parameterize.h"
#include "ugu/util/geom_util.h"
#include "ugu/util/math_util.h"

namespace {

void UpdateCandidates(std::vector<ugu::PlaneEstimationResult>& candidates,
                      const ugu::PlaneEstimationResult& new_candidate,
                      const ugu::EstimateGroundPlaneRansacParam& param) {
  // if (candidates.size() < param.candidates_num) {
  candidates.push_back(new_candidate);
  //} else {

  //}
  using per = ugu::PlaneEstimationResult;
  std::sort(candidates.begin(), candidates.end(),
            [&](const per& a, const per& b) {
              return a.stat.inlier_ratio > b.stat.inlier_ratio;
            });

  while (candidates.size() > param.candidates_num) {
    candidates.pop_back();
  }
}

float ComputeProjectedArea(const std::vector<Eigen::Vector3f>& ps_3d,
                           const Eigen::Vector3f& n) {
  std::vector<Eigen::Vector2f> ps_2d;
  ugu::OrthoProjectToXY(n, ps_3d, ps_2d, true, false, false, false);
  Eigen::Vector2f bb_len =
      ugu::ComputeMaxBound(ps_2d) - ugu::ComputeMinBound(ps_2d);
  return bb_len[0] * bb_len[1];
}

}  // namespace

namespace ugu {

bool EstimatePlaneLeastSquares(const std::vector<Eigen::Vector3f>& points,
                               Planef& plane,
                               const Eigen::Vector3f& normal_hint) {
  if (points.size() < 3) {
    return false;
  }

  std::array<Eigen::Vector3f, 3> axes;
  std::array<float, 3> weights;
  Eigen::Vector3f centroid;
  ComputeAxisForPoints(points, axes, weights, centroid);

  Eigen::Vector3f n = axes[2];
  if (normal_hint.norm() > 0.001f) {
    if (n.dot(normal_hint) < 0.f) {
      n = -1.f * n;
    }
  }

  float d = -n.dot(centroid);

  plane = Planef(n, d);

  return true;
}

bool EstimateGroundPlaneRansac(const std::vector<Eigen::Vector3f>& points,
                               const std::vector<Eigen::Vector3f>& normals,
                               const EstimateGroundPlaneRansacParam& param,
                               std::vector<PlaneEstimationResult>& candidates) {
  EstimateGroundPlaneRansacParam param_ = param;
  // param_.inlier_ratio_th = std::clamp(param_.inlier_ratio_th, 0.f, 1.f);
  // At least 90 deg
  param_.inlier_angle_th =
      std::clamp(param_.inlier_angle_th, 0.f, radians(90.f));
#if 0
  if (param_.inliner_dist_th < 0.f) {
    // Set automatically
    constexpr float dist_r = 0.01f;
    param_.inliner_dist_th =
        (ComputeMaxBound(points) - ComputeMinBound(points)).norm() * dist_r;
  }
#endif
  if (param_.inliner_dist_th < 0.f) {
    LOGW("inliner_dist_th will be ignored\n");
  }

  if (points.empty() || points.size() != normals.size()) {
    return false;
  }

  std::default_random_engine engine;
  if (param_.seed < 0) {
    static std::random_device seed_gen;
    engine = std::default_random_engine(seed_gen());
  } else {
    engine = std::default_random_engine(static_cast<uint32_t>(param_.seed));
  }
  std::uniform_int_distribution<int64_t> dist(
      0, static_cast<int64_t>(points.size()));

  candidates.clear();

  uint32_t iter = 0;
  const float inlier_angle_cos = std::cos(param_.inlier_angle_th);
  const float p_size_inverse = 1.f / static_cast<float>(points.size());

  auto generate_hypothesis_3_points = [&]() {
    int64_t idx0 = dist(engine);
    int64_t idx1 = dist(engine);
    int64_t idx2 = dist(engine);

    Eigen::Vector3f ave_n = normals[idx0] + normals[idx1] + normals[idx2];
    if (0.00001f < ave_n.norm()) {
      ave_n.normalize();
    }
    Planef hypothesis(points[idx0], points[idx1], points[idx2], ave_n);
    return hypothesis;
  };

  auto generate_hypothesis_normal_hint = [&]() {
    int64_t idx0 = dist(engine);
    const auto& p = points[idx0];
    float d = -param_.normal_hint.dot(p);
    Planef hypothesis(param_.normal_hint, d);
    return hypothesis;
  };

  std::function<Planef()> generate_hypothesis = generate_hypothesis_3_points;
  bool use_normal_hint =
      param_.use_normal_hint && param_.normal_hint.norm() > 0.001f;
  Eigen::Vector3f normal_hint = Eigen::Vector3f::Zero();
  if (use_normal_hint) {
    generate_hypothesis = generate_hypothesis_normal_hint;
    normal_hint = param_.normal_hint;
  }

  // Estimate planes with many inliners as possible
  while (iter < param_.max_iter) {
    Planef hypothesis = generate_hypothesis();

    PlaneEstimationStat cur_stat;
    for (size_t i = 0; i < points.size(); i++) {
      const auto& p = points[i];
      const auto& n = normals[i];

      // Similar angle
      const float dot = hypothesis.n.dot(n);
      const float d = hypothesis.SignedDist(p);
      bool is_outlier = false;
      if (dot < inlier_angle_cos || (0.f <= param_.inliner_dist_th &&
                                     param_.inliner_dist_th < std::abs(d))) {
        cur_stat.outliers.push_back(i);
        is_outlier = true;
      }

      if (0.f >= d) {
        cur_stat.uppers.push_back(i);
      }

      if (!is_outlier) {
        cur_stat.inliers.push_back(i);
      }
    }

    cur_stat.inlier_ratio =
        static_cast<float>(cur_stat.inliers.size()) * p_size_inverse;

    cur_stat.upper_ratio =
        static_cast<float>(cur_stat.uppers.size()) * p_size_inverse;

    Planef refined_least_squares;  // Not computed here
    PlaneEstimationResult new_candidate{hypothesis, refined_least_squares,
                                        cur_stat};

    UpdateCandidates(candidates, new_candidate, param);

    iter++;
  }

  // Sort by projected area
  for (auto& c : candidates) {
    std::vector<Eigen::Vector3f> inliers;
    for (const auto& i : c.stat.inliers) {
      inliers.push_back(points[i]);
    }
    c.stat.area = ComputeProjectedArea(inliers, c.estimation.n);
    float all_area = ComputeProjectedArea(points, c.estimation.n);
    c.stat.area_ratio = all_area > 0 ? c.stat.area / all_area : -1.f;
  }

  using per = ugu::PlaneEstimationResult;
  std::sort(candidates.begin(), candidates.end(),
            [&](const per& a, const per& b) {
              return std::abs(a.stat.area_ratio - 1.f) <
                     std::abs(b.stat.area_ratio - 1.f);
            });

  if (param_.refine_least_squares) {
    for (auto& c : candidates) {
      std::vector<Eigen::Vector3f> inlier_points;
      for (const auto& i : c.stat.inliers) {
        inlier_points.push_back(points[i]);
      }
      Eigen::Vector3f n_hint = c.estimation.n;
      if (use_normal_hint) {
        n_hint = normal_hint;
      }
      EstimatePlaneLeastSquares(inlier_points, c.refined_least_squares, n_hint);
    }
  }

  return true;
}

}  // namespace ugu
