/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 *
 */

#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include "ugu/mesh.h"
#include "ugu/util/geom_util.h"

namespace ugu {

template <typename T>
struct Aabb {
  T max_v;
  T min_v;
  T length;
  T center;
  std::vector<Eigen::Vector3f> vertices;
  std::vector<Eigen::Vector3i> indices;

  void Init(const T& max_v_, const T& min_v_) {
    this->max_v = max_v_;
    this->min_v = min_v_;
    length = max_v - min_v;
    center = (max_v + min_v) * 0.5;

    vertices.resize(24);
    indices.resize(12);
    vertices[0] = Eigen::Vector3d(min_v[0], max_v[1], min_v[2]).cast<float>();
    vertices[1] = Eigen::Vector3d(max_v[0], max_v[1], min_v[2]).cast<float>();
    vertices[2] = Eigen::Vector3d(max_v[0], max_v[1], max_v[2]).cast<float>();
    vertices[3] = Eigen::Vector3d(min_v[0], max_v[1], max_v[2]).cast<float>();
    indices[0] = Eigen::Vector3i(0, 2, 1);
    indices[1] = Eigen::Vector3i(0, 3, 2);

    vertices[4] = Eigen::Vector3d(min_v[0], min_v[1], min_v[2]).cast<float>();
    vertices[5] = Eigen::Vector3d(max_v[0], min_v[1], min_v[2]).cast<float>();
    vertices[6] = Eigen::Vector3d(max_v[0], min_v[1], max_v[2]).cast<float>();
    vertices[7] = Eigen::Vector3d(min_v[0], min_v[1], max_v[2]).cast<float>();
    indices[2] = Eigen::Vector3i(4, 5, 6);
    indices[3] = Eigen::Vector3i(4, 6, 7);

    vertices[8] = vertices[1];
    vertices[9] = vertices[2];
    vertices[10] = vertices[6];
    vertices[11] = vertices[5];
    indices[4] = Eigen::Vector3i(8, 9, 10);
    indices[5] = Eigen::Vector3i(8, 10, 11);

    vertices[12] = vertices[0];
    vertices[13] = vertices[3];
    vertices[14] = vertices[7];
    vertices[15] = vertices[4];
    indices[6] = Eigen::Vector3i(12, 14, 13);
    indices[7] = Eigen::Vector3i(12, 15, 14);

    vertices[16] = vertices[0];
    vertices[17] = vertices[1];
    vertices[18] = vertices[5];
    vertices[19] = vertices[4];
    indices[8] = Eigen::Vector3i(16, 17, 18);
    indices[9] = Eigen::Vector3i(16, 18, 19);

    vertices[20] = vertices[3];
    vertices[21] = vertices[2];
    vertices[22] = vertices[6];
    vertices[23] = vertices[7];
    indices[10] = Eigen::Vector3i(20, 22, 21);
    indices[11] = Eigen::Vector3i(20, 23, 22);
  };

  Aabb(){};
  ~Aabb(){};
  Aabb(const T& max_v, const T& min_v) { Init(max_v, min_v); }
  Aabb(const std::vector<T>& points) {
    max_v = ComputeMaxBound(points);
    min_v = ComputeMinBound(points);
    Init(max_v, min_v);
  }

#if 0
  template <typename TT>
  Aabb(const std::vector<T>& vertices, const std::vector<TT>& indices) {
    std::vector<T> points;
    for (const auto& i : indices) {
      for (int j = 0; j < i.rows(); j++) {
        points.push_back(vertices[i[j]]);
      }
    }
    max_v = ComputeMaxBound(points);
    min_v = ComputeMinBound(points);
    Init(max_v, min_v);
  }
#endif

  static double SurfaceArea(const Aabb& x) {
    double sum = 0.0;
    for (int i = 0; i < x.rows(); i++) {
      for (int j = i + 1; j < x.rows(); j++) {
        sum = static_cast<double>(x.length[i] * x.length[j]);
      }
    }
    return 2.0 * sum;
  }
  double SurfaceArea() const { return SurfaceArea(*this); }
};

class BvhBuildStatistics {
 public:
  unsigned int max_tree_depth;
  unsigned int num_leaf_nodes;
  unsigned int num_branch_nodes;
  float build_secs;
  Aabb<Eigen::Vector3d> bb;

  // Set default value: Taabb = 0.2
  BvhBuildStatistics()
      : max_tree_depth(0),
        num_leaf_nodes(0),
        num_branch_nodes(0),
        build_secs(0.0f) {}
};

struct Ray {
  Eigen::Vector3f org, dir;
};

template <typename T, typename TT>
class Bvh {
 public:
  virtual ~Bvh(){};

  virtual void SetData(const std::vector<T>& vertices,
                       const std::vector<TT>& indices) = 0;
  virtual bool Build() = 0;
  virtual const BvhBuildStatistics& GetBuildStatistics() const = 0;
  virtual std::vector<IntersectResult> Intersect(
      const Ray& ray, bool test_all = true) const = 0;
};

enum kBvhCostType { NAIVE, SAH };  // Surface Area Heuristics

template <typename T, typename TT>
using BvhPtr = std::shared_ptr<Bvh<T, TT>>;

}  // namespace ugu
