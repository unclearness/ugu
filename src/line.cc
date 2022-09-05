/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/line.h"

#include <fstream>
#include <random>
#include <set>
#include <sstream>
#include <unordered_set>

#include "ugu/accel/kdtree.h"
#include "ugu/plane.h"

namespace {

using namespace ugu;

template <typename T>
Line3<T> LocalMean(const Line3<T>& seed, const Line3<T>& X0,
                   const std::vector<Line3<T>>& unclean,
                   const std::shared_ptr<ugu::KdTree<float, 3>> kdtree,
                   double r_nei, double sigma_p, double sigma_d) {
  KdTreeSearchResults neighbors = kdtree->SearchRadius(
      seed.a.template cast<float>(), static_cast<double>(r_nei));

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
  double denom = 0.0;
  for (size_t i = 0; i < neighbors.size(); i++) {
    if (!succeeded[i]) {
      bilateral_weights.push_back(0.0);
      continue;
    }
    double pos_term =
        -((X0.a - intersections[i]).squaredNorm()) / (2.0 * sigma_p * sigma_p);
    double numerator = std::acos(
        std::clamp(X0.d.dot(unclean[neighbors[i].index].d), -1.0, 1.0));
    double dir_term = -numerator * numerator / (2.0 * sigma_d * sigma_d);
    double terms = std::clamp(pos_term + dir_term, -100.0,
                              100.0);  // clamp to avoid illegal values by exp()
    double w = std::exp(terms);

    assert(std::isnormal(w));
    bilateral_weights.push_back(w);
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
    moved.d += unclean[neighbors[i].index].d * bilateral_weights[i] * inv_denom;
  }
  moved.d = moved.d.normalized();

  return moved;
}

auto GetKdtree(const std::vector<Eigen::Vector3f>& data) {
  ugu::KdTreePtr<float, 3> kdtree = GetDefaultKdTree<float, 3>();
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

  std::vector<Eigen::Vector3f> pos_list;
  std::transform(unclean.begin(), unclean.end(), std::back_inserter(pos_list),
                 [&](const Line3<T>& l) { return l.a.template cast<float>(); });

  auto kdtree = GetKdtree(pos_list);

  fused.clear();
  fused.resize(unclean.size());
#pragma omp parallel for
  for (int64_t i = 0; i < static_cast<int64_t>(unclean.size()); i++) {
    const Line3<T>& P = unclean[i];
    Line3<T> Q_prev = P;
    Line3<T> Q_next = Q_prev;
    int iter = 0;
    T d = std::numeric_limits<T>::max();
    while (d > tau_s && iter < max_iter) {
      Q_next = LocalMean(Q_prev, P, unclean, kdtree, r_nei, sigma_p, sigma_d);
      d = (Q_next.a - Q_prev.a).norm();
      Q_prev = Q_next;
      iter++;
    }
    fused[i] = Q_next;
  }

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
      ss << " " << c[0] << " " << c[1] << " " << c[2];
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

#if 0
  bool with_colors = points.size() == colors.size();
  bool with_normals = points.size() == normals.size();

  for (size_t i = 0; i < points.size(); i++) {
    std::vector<int> single_index;
    for (size_t j = 0; j < points[i].size(); j++) {
      const auto& p = points[i][j];
      ss << "v " << p[0] << " " << p[1] << " " << p[2];
      if (with_colors) {
        const auto& c = colors[i][j];
        ss << " " << c[0] << " " << c[1] << " " << c[2];
      }
      ss << "\n";
      single_index.push_back(index);
      index++;
    }
  }

#if 0
  if (with_colors) {
    for (const auto& l : colors) {
      for (const auto& c : l) {
        ss << "v " << c[0] << " " << c[1] << " " << c[2] << "\n";
      }
    }
  }
#endif

  if (with_normals) {
    for (size_t i = 0; i < normals.size(); i++) {
      for (size_t j = 0; j < normals[i].size(); j++) {
        const auto& n = normals[i][j];
        ss << "vn " << n[0] << " " << n[1] << " " << n[2] << "\n";
      }
    }
  }

  for (const auto& i : indices) {
    ss << "l";
    for (size_t lidx = 0; lidx < i.size(); i++) {
      ss << " " << i[lidx];
    }
    ss << "\n";
  }

  std::ofstream ofs(path, "w");
  ofs << ss;
#endif
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

  std::vector<Eigen::Vector3f> pos_list;
  std::transform(lines.begin(), lines.end(), std::back_inserter(pos_list),
                 [&](const Line3<T>& l) { return l.a.template cast<float>(); });
  auto kdtree = GetKdtree(pos_list);

  // constexpr uint32_t rand_seed = 0;
  // std::uniform_real_distribution<double> disr;
  // std::default_random_engine engine(rand_seed);

  size_t P_seed_index = size_t(disr(engine) * P.size());
  while (!P.empty()) {
    size_t to_erase = P_seed_index;
    P.erase(to_erase);
    std::vector<Line3<T>> strand{lines[P_seed_index]};

    const auto& P_seed_forward = lines[P_seed_index];
    Line3<T> P_seed_backward = P_seed_forward;
    P_seed_backward.d *= -1.0;

    for (const Line3<T>& P_seed : {P_seed_forward, P_seed_backward}) {
      Line3<T> P_cur = P_seed;
      int iter = 0;
      while (iter < max_iter) {
        iter++;
        Eigen::Vector3<T> stepped = P_cur.a + s * P_cur.d;
        KdTreeSearchResults neighbors_ =
            kdtree->SearchRadius(stepped.template cast<float>(), tau_r);
        KdTreeSearchResults neighbors;
        for (const auto& n : neighbors_) {
          if (P.find(n.index) == P.end()) {
            continue;
          }
          if (std::acos(std::clamp(lines[n.index].d.dot(P_cur.d), -1.0, 1.0)) >
              tau_a) {
            continue;
          }

          neighbors.push_back(n);
        }
        if (neighbors.empty()) {
          break;
        }

        P_cur.a = Eigen::Vector3<T>::Zero();
        P_cur.d = Eigen::Vector3<T>::Zero();
        for (const auto& n : neighbors) {
          P_cur.a += lines[n.index].a;
          P_cur.d += lines[n.index].d;
        }
        P_cur.a /= neighbors.size();
        P_cur.d /= neighbors.size();
        P_cur.d.normalize();

        if (!strand.empty()) {
          // No description in the paper. My original method.
          // Sometimes the algorithm stucks at the same position.
          // To avoid the stuck, check movement is small or not
          double movement = (P_cur.a - strand.back().a).norm();
          const double stop_th = tau_r * 0.01;
          if (movement < stop_th) {
            break;
          }
        }

        strand.push_back(P_cur);
      }
    }

    // Erase near points
    for (const auto& s : strand) {
      KdTreeSearchResults neighbors =
          kdtree->SearchRadius(s.a.template cast<float>(), tau_r);
      for (const auto& n : neighbors) {
        P.erase(n.index);
      }
    }

    strands.push_back(strand);

    if (P.empty()) {
      break;
    }

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
