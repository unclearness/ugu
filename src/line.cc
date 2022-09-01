/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/line.h"

#include "ugu/accel/kdtree.h"
#include "ugu/accel/kdtree_nanoflann.h"
#include "ugu/plane.h"

namespace {

using namespace ugu;

template <typename T>
Line3<T> LocalMean(
    const Line3<T>& seed, const std::vector<Line3<T>>& unclean,
    const std::shared_ptr<ugu::KdTree<Eigen::Vector<float, 3>>> kdtree,
    double r_nei, double sigma_p, double sigma_d) {
  KdTreeSearchResults neighbors =
      kdtree->SearchRadius(seed.a.cast<float>(), static_cast<double>(r_nei));

  if (neighbors.empty()) {
    return seed;
  }

  Plane<T> plane(seed.d, -seed.a.dot(seed.d));

  std::vector<Eigen::Vector<T, 3>> intersections;
  for (const auto& n : neighbors) {
    Eigen::Vector<T, 3> intersection;
    T t;
    plane.CalcIntersctionPoint(unclean[n.index], t, intersection);
    intersections.push_back(intersection);
  }

  std::vector<double> bilateral_weights;
  double denom = 0.0;
  for (size_t i = 0; i < neighbors.size(); i++) {
    double pos_term = -((seed.a - intersections[i]).squaredNorm()) /
                      (2.0 * sigma_p * sigma_p);
    double numerator = std::acos(
        std::clamp(seed.d.dot(unclean[neighbors[i].index].d), -1.0, 1.0));
    double dir_term = -numerator * numerator / (2.0 * sigma_d * sigma_d);
    double w = std::exp(pos_term + dir_term);

    // ugu::LOGI("%f %f %f\n", pos_term, dir_term, w);
    assert(std::isnormal(w));
    bilateral_weights.push_back(w);
    denom += w;
  }

  assert(denom > 0);

  const double inv_denom = 1.0 / denom;
  Line3<T> moved;
  moved.a = Eigen::Vector<T, 3>::Zero();
  moved.d = Eigen::Vector<T, 3>::Zero();
  for (size_t i = 0; i < neighbors.size(); i++) {
    moved.a += unclean[neighbors[i].index].a * bilateral_weights[i] * inv_denom;
    moved.d += unclean[neighbors[i].index].d * bilateral_weights[i] * inv_denom;
  }
  moved.d = moved.d.normalized();

  return moved;
}

template <typename T>
bool LineClusteringImpl(const std::vector<Line3<T>>& unclean,
                        std::vector<Line3<T>>& fused, double tau_s,
                        double r_nei, double sigma_p, double sigma_d) {
  if (unclean.size() < 2) {
    return false;
  }
  if (tau_s < T(0)) {
    return false;
  }

  std::vector<Eigen::Vector3f> pos_list;
  std::transform(unclean.begin(), unclean.end(), std::back_inserter(pos_list),
                 [&](const Line3<T>& l) { return l.a.cast<float>(); });

  std::shared_ptr<ugu::KdTree<Eigen::Vector<float, 3>>> kdtree;
#ifdef UGU_USE_NANOFLANN
  kdtree =
      std::make_shared<ugu::KdTreeNanoflannVector<Eigen::Vector<float, 3>>>();
#else
  kdtree = std::make_shared<ugu::KdTreeNaive<Eigen::Vector<float, 3>>>;
  ;
  std::dynamic_pointer_cast<KdTreeNaive<Eigen::VectorXf>>(kdtree)->SetAxisNum(
      pos_list[0].rows());
#endif

  kdtree->SetData(pos_list);
  kdtree->Build();

  fused.clear();
  fused.resize(unclean.size());
#pragma omp parallel for
  for (int64_t i = 0; i < static_cast<int64_t>(unclean.size()); i++) {
    const Line3<T>& P = unclean[i];
    Line3<T> Q_prev = P;
    Line3<T> Q_next = Q_prev;
    T d = std::numeric_limits<T>::max();
    while (d > tau_s) {
      Q_next = LocalMean(Q_prev, unclean, kdtree, r_nei, sigma_p, sigma_d);
      d = (Q_next.a - Q_prev.a).norm();
      Q_prev = Q_next;
    }
    fused[i] = Q_next;
  }

  return true;
}

}  // namespace

namespace ugu {

bool LineClustering(const std::vector<Line3d>& unclean,
                    std::vector<Line3d>& fused, double tau_s, double r_nei,
                    double sigma_p, double sigma_d) {
  return LineClusteringImpl(unclean, fused, tau_s, r_nei, sigma_p, sigma_d);
}

}  // namespace ugu
