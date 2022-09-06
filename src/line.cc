/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/line.h"

#include <fstream>
#include <random>
#include <sstream>
#include <unordered_set>

#include "ugu/accel/kdtree.h"
#include "ugu/plane.h"
#include "ugu/timer.h"
#include "ugu/util/thread_util.h"

namespace {

using namespace ugu;

template <typename T>
std::pair<double, double> MinAngle(const T& a, const T& b) {
  double inv_mag = 1.0 / (a.norm() * b.norm());
  double v0 = std::acos(std::clamp(a.dot(b) * inv_mag, -1.0, 1.0));
  double v1 = ugu::pi - v0;
  if (v0 < v1) {
    return {v0, 1.0};
  }

  return {v1, -1.0};
}

template <typename T>
Line3<T> LocalMean(const Line3<T>& seed, const Line3<T>& X0,
                   const std::vector<Line3<T>>& unclean,
                   const std::shared_ptr<ugu::KdTree<T, 3>> kdtree,
                   double r_nei, double sigma_p, double sigma_d) {
  KdTreeSearchResults neighbors = kdtree->SearchRadius(seed.a, r_nei);

  if (neighbors.empty()) {
    return seed;
  }

  Plane<T> plane(seed.d, -seed.a.dot(seed.d));

  std::vector<Eigen::Vector<T, 3>> intersections;
  std::vector<bool> succeeded;
  for (const auto& n : neighbors) {
    Eigen::Vector<T, 3> intersection;
    T t;
    bool ret = plane.CalcIntersctionPoint(unclean[n.index], t, intersection);
    intersections.push_back(intersection);
    succeeded.push_back(ret);
  }

  std::vector<double> bilateral_weights;
  std::vector<double> direction_weights;
  double denom = 0.0;
  for (size_t i = 0; i < neighbors.size(); i++) {
    if (!succeeded[i]) {
      bilateral_weights.push_back(0.0);
      direction_weights.push_back(0.0);
      continue;
    }
    double pos_term =
        -((X0.a - intersections[i]).squaredNorm()) / (2.0 * sigma_p * sigma_p);
    auto [numerator, direction_w] =
        MinAngle(X0.d, unclean[neighbors[i].index].d);
    double dir_term = -numerator * numerator / (2.0 * sigma_d * sigma_d);
    double terms = std::clamp(pos_term + dir_term, -100.0,
                              100.0);  // clamp to avoid illegal values by exp()
    double w = std::exp(terms);

    assert(std::isnormal(w));
    bilateral_weights.push_back(w);
    direction_weights.push_back(direction_w);
    denom += w;
  }

  if (denom <= 0.0) {
    return seed;
  }

  assert(denom > 0.0);

  const double inv_denom = 1.0 / denom;
  Line3<T> moved;
  moved.a = Eigen::Vector<T, 3>::Zero();
  moved.d = Eigen::Vector<T, 3>::Zero();
  for (size_t i = 0; i < neighbors.size(); i++) {
    moved.a += unclean[neighbors[i].index].a * bilateral_weights[i] * inv_denom;
    moved.d += unclean[neighbors[i].index].d * bilateral_weights[i] *
               inv_denom * direction_weights[i];
  }
  moved.d = moved.d.normalized();

  return moved;
}

template <typename T>
auto GetKdtree(const std::vector<Eigen::Vector3<T>>& data) {
  ugu::KdTreePtr<T, 3> kdtree = GetDefaultKdTree<T, 3>();
  kdtree->SetData(data);
  kdtree->Build();
  return kdtree;
}

template <typename T>
bool LineClusteringImpl(const std::vector<Line3<T>>& unclean,
                        std::vector<Line3<T>>& fused, double tau_s,
                        double r_nei, double sigma_p, double sigma_d,
                        int max_iter) {
  if (unclean.size() < 2) {
    return false;
  }
  if (tau_s < T(0)) {
    return false;
  }

  std::vector<Eigen::Vector3<T>> pos_list;
  std::transform(unclean.begin(), unclean.end(), std::back_inserter(pos_list),
                 [&](const Line3<T>& l) { return l.a; });

  auto kdtree = GetKdtree(pos_list);

  fused.clear();
  fused.resize(unclean.size());

#if 0
  std::vector<int> iters;
  std::vector<double> times;
#endif
  auto func = [&](size_t i) {
    const Line3<T>& P = unclean[i];
    Line3<T> Q_prev = P;
    Line3<T> Q_next = Q_prev;
    int iter = 0;
    T d = std::numeric_limits<T>::max();

#if 0
    Timer timer;
    timer.Start();
#endif
    while (d > tau_s && iter < max_iter) {
      Q_next = LocalMean(Q_prev, P, unclean, kdtree, r_nei, sigma_p, sigma_d);
      d = (Q_next.a - Q_prev.a).norm();
      Q_prev = Q_next;
      iter++;
    }

#if 0
    timer.End();
    iters.push_back(iter);
    times.push_back(timer.elapsed_msec());
#endif
    fused[i] = Q_next;
  };

  ugu::parallel_for(size_t(0), unclean.size(), func);

#if 0
  std::ofstream ofs("times.csv");
  for (size_t i = 0; i < iters.size(); i++) {
    ofs << iters[i] << "," << times[i] << std::endl;
  }
#endif

  return true;
}

template <typename T>
bool WriteObjLineImpl(const std::vector<Eigen::Vector3<T>>& points,
                      const std::vector<std::vector<int>>& indices,
                      const std::string& path,
                      const std::vector<Eigen::Vector3<T>>& colors,
                      const std::vector<Eigen::Vector3<T>>& normals) {
  std::stringstream ss;

  bool with_colors = points.size() == colors.size();
  bool with_normals = points.size() == normals.size();

  for (size_t i = 0; i < points.size(); i++) {
    const auto& p = points[i];
    ss << "v " << p[0] << " " << p[1] << " " << p[2];
    if (with_colors) {
      const auto& c = colors[i];
      ss << " " << c[0] / 255 << " " << c[1] / 255 << " " << c[2] / 255;
    }
    ss << "\n";
  }

  if (with_normals) {
    for (size_t i = 0; i < normals.size(); i++) {
      const auto& n = normals[i];
      ss << "vn " << n[0] << " " << n[1] << " " << n[2] << "\n";
    }
  }

  for (const auto& i : indices) {
    ss << "l";
    for (size_t lidx = 0; lidx < i.size(); lidx++) {
      ss << " " << i[lidx];
    }
    ss << "\n";
  }

  std::ofstream ofs(path.c_str());

  ofs << ss.rdbuf();

  return true;
}

template <typename T>
bool WriteObjLineImpl(
    const std::vector<std::vector<Eigen::Vector3<T>>>& points,
    const std::string& path,
    const std::vector<std::vector<Eigen::Vector3<T>>>& colors,
    const std::vector<std::vector<Eigen::Vector3<T>>>& normals) {
  std::stringstream ss;

  std::vector<std::vector<int>> indices;
  int index = 1;

  std::vector<Eigen::Vector3<T>> points_flat;
  std::vector<Eigen::Vector3<T>> colors_flat;
  std::vector<Eigen::Vector3<T>> normals_flat;

  for (size_t i = 0; i < points.size(); i++) {
    std::vector<int> single_index;
    for (size_t j = 0; j < points[i].size(); j++) {
      const auto& p = points[i][j];
      points_flat.push_back(p);
      single_index.push_back(index);
      index++;
    }
    indices.push_back(single_index);
  }

  for (size_t i = 0; i < colors.size(); i++) {
    for (size_t j = 0; j < colors[i].size(); j++) {
      const auto& x = colors[i][j];
      colors_flat.push_back(x);
    }
  }

  for (size_t i = 0; i < normals.size(); i++) {
    for (size_t j = 0; j < normals[i].size(); j++) {
      const auto& x = normals[i][j];
      normals_flat.push_back(x);
    }
  }

  return WriteObjLineImpl(points_flat, indices, path, colors_flat,
                          normals_flat);
}

template <typename T>
bool WriteObjLineImpl(const std::vector<std::vector<Line3<T>>>& lines,
                      const std::string& path,
                      std::vector<std::vector<Eigen::Vector3<T>>> colors) {
  std::vector<std::vector<Eigen::Vector3<T>>> points;
  std::vector<std::vector<Eigen::Vector3<T>>> normals;
  for (const auto& line : lines) {
    std::vector<Eigen::Vector3<T>> single_l;
    std::vector<Eigen::Vector3<T>> single_n;
    for (const auto& p : line) {
      single_l.push_back(p.a);
      single_n.push_back(p.d);
    }
    points.push_back(single_l);
    normals.push_back(single_n);
  }
  return WriteObjLineImpl(points, path, colors, normals);
}

template <typename T>
bool GenerateStrandsImpl(const std::vector<Line3<T>>& lines,
                         std::vector<std::vector<Line3<T>>>& strands, double s,
                         double tau_r, double tau_a, int max_iter) {
  strands.clear();
  std::unordered_set<size_t> P;
  for (size_t i = 0; i < lines.size(); i++) {
    P.insert(i);
  }

  std::vector<Eigen::Vector3<T>> pos_list;
  std::transform(lines.begin(), lines.end(), std::back_inserter(pos_list),
                 [&](const Line3<T>& l) { return l.a; });
  auto kdtree = GetKdtree(pos_list);

  constexpr uint32_t rand_seed = 0;
  std::uniform_real_distribution<double> disr;
  std::default_random_engine engine(rand_seed);

  size_t P_seed_index = size_t(disr(engine) * P.size());
  while (!P.empty()) {
    size_t to_erase = P_seed_index;
    P.erase(to_erase);
    std::vector<Line3<T>> strand{lines[P_seed_index]};

    const auto& P_seed_forward = lines[P_seed_index];
    Line3<T> P_seed_backward = P_seed_forward;
    P_seed_backward.d *= -1.0;

    std::array<Line3<T>, 2> P_seeds = {P_seed_forward, P_seed_backward};
    for (size_t i = 0; i < 2; i++) {
      const Line3<T>& P_seed = P_seeds[i];
      Line3<T> P_cur = P_seed;
      int iter = 0;
      while (iter < max_iter) {
        iter++;
        Eigen::Vector3<T> stepped = P_cur.a + s * P_cur.d;
        KdTreeSearchResults neighbors_ = kdtree->SearchRadius(stepped, tau_r);
        KdTreeSearchResults neighbors;
        std::vector<double> weights;
        for (const auto& n : neighbors_) {
          if (P.find(n.index) == P.end()) {
            continue;
          }
          auto [rad, w] = MinAngle(lines[n.index].d, P_cur.d);
          if (rad > tau_a) {
            continue;
          }
          weights.push_back(w);
          neighbors.push_back(n);
        }
        if (neighbors.empty()) {
          break;
        }

        P_cur.a = Eigen::Vector3<T>::Zero();
        P_cur.d = Eigen::Vector3<T>::Zero();
        for (size_t j = 0; j < neighbors.size(); j++) {
          const auto& n = neighbors[j];
          const auto& w = weights[j];
          P_cur.a += lines[n.index].a;
          P_cur.d += lines[n.index].d * w;
        }
        P_cur.a /= static_cast<double>(neighbors.size());
        P_cur.d /= static_cast<double>(neighbors.size());
        P_cur.d.normalize();

        if (!strand.empty()) {
          // No description in the paper. My original method.
          // Sometimes the algorithm stucks at the same position.
          // To avoid the stuck, check movement is small or not
          auto latest = strand.back().a;
          if (i == 1) {
            latest = strand.front().a;
          }
          double movement = (P_cur.a - latest).norm();
          const double stop_th = tau_r * 0.01;
          if (movement < stop_th) {
            break;
          }
        }
        if (i == 0) {
          strand.push_back(P_cur);
        } else {
          strand.insert(strand.begin(), P_cur);
        }
      }
    }

    // Erase near points
    for (const auto& st : strand) {
      KdTreeSearchResults neighbors = kdtree->SearchRadius(st.a, tau_r);
      for (const auto& n : neighbors) {
        P.erase(n.index);
      }
    }

    strands.push_back(strand);

    if (P.empty()) {
      break;
    }

    // TODO: Randomly select valid index
    P_seed_index = *P.begin();
  }

  return true;
}

}  // namespace

namespace ugu {

bool LineClustering(const std::vector<Line3d>& unclean,
                    std::vector<Line3d>& fused, double tau_s, double r_nei,
                    double sigma_p, double sigma_d, int max_iter) {
  return LineClusteringImpl(unclean, fused, tau_s, r_nei, sigma_p, sigma_d,
                            max_iter);
}

bool WriteObjLine(const std::vector<std::vector<Line3f>>& lines,
                  const std::string& path,
                  const std::vector<std::vector<Eigen::Vector3f>>& colors) {
  return WriteObjLineImpl(lines, path, colors);
}
bool WriteObjLine(const std::vector<std::vector<Line3d>>& lines,
                  const std::string& path,
                  const std::vector<std::vector<Eigen::Vector3d>>& colors) {
  return WriteObjLineImpl(lines, path, colors);
}
bool WriteObjLine(const std::vector<std::vector<Eigen::Vector3f>>& lines,
                  const std::string& path,
                  const std::vector<std::vector<Eigen::Vector3f>>& colors,
                  const std::vector<std::vector<Eigen::Vector3f>>& normals) {
  return WriteObjLineImpl(lines, path, colors, normals);
}
bool WriteObjLine(const std::vector<std::vector<Eigen::Vector3d>>& lines,
                  const std::string& path,
                  const std::vector<std::vector<Eigen::Vector3d>>& colors,
                  const std::vector<std::vector<Eigen::Vector3d>>& normals) {
  return WriteObjLineImpl(lines, path, colors, normals);
}

bool GenerateStrands(const std::vector<Line3d>& lines,
                     std::vector<std::vector<Line3d>>& strands, double s,
                     double tau_r, double tau_a, int max_iter) {
  return GenerateStrandsImpl(lines, strands, s, tau_r, tau_a, max_iter);
}

}  // namespace ugu
