/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 *
 */

#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include "Eigen/Core"
#include "ugu/mesh.h"
#include "ugu/util/geom_util.h"

namespace ugu {

template <typename T>
struct Aabb {
  T max_v;
  T min_v;
  T length;
  T center;

  void Init(const T& max_v, const T& min_v) {
    this->max_v = max_v;
    this->min_v = min_v;
    length = max_v - min_v;
    center = (max_v + min_v) * 0.5;
  };

  Aabb(){};
  ~Aabb(){};
  Aabb(const T& max_v, const T& min_v) { Init(max_v, min_v); }
  Aabb(const std::vector<T>& points) {
    max_v = ComputeMaxBound(points);
    min_v = ComputeMinBound(points);
    Init(max_v, min_v);
  }

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

  static double SurfaceArea(const Aabb& x) {
    double sum = 0.0;
    for (int i = 0; i < x.rows(); i++) {
      for (int j = i + 1; j < x.rows(); j++) {
        sum = static_cast<double>(x.length[i] * x.length[j]);
      }
    }
    return 2.0 * sum;
  }
  double SurfaceArea() const { return ::SurfaceArea(*this); }
};

using Aabb3f = Aabb<Eigen::Vector3f>;

template <typename T, typename TT>
class Bvh {
 public:
  Bvh(){};
  ~Bvh(){};

  enum kCostType { NAIVE, SAH };  // Surface Area Heuristics

  void SetData(const std::vector<T>& vertices, const std::vector<TT>& indices) {
    m_vertices = vertices;
    m_indices = indices;
  }

  void SetAxisNum(int axis_num) { m_axis_num = axis_num; }

  void SetMaxLeafDataNum(int max_leaf_data_num) {
    m_max_leaf_data_num = max_leaf_data_num;
  }

  void SetCostType(kCostType cost_type) { m_cost_type = cost_type; }

  bool Build() {
    if (m_axis_num < 0 || m_vertices.empty() || m_indices.empty()) {
      return false;
    }
    root = BuildImpl(m_indices, 0);
    return true;
  }

 private:
  std::vector<T> m_vertices;
  std::vector<TT> m_indices;
  int m_axis_num = -1;
  int m_max_leaf_data_num = -1;
  kCostType m_cost_type = kCostType::NAIVE;

  struct Node;
  using NodePtr = std::shared_ptr<Node>;
  struct Node {
    Node(){};
    ~Node(){};
    Aabb<T> bbox;
    std::array<NodePtr, 2> children{nullptr, nullptr};  // TODO
    std::vector<TT> indices;
    int depth = -1;
    int axis = -1;
  };
  NodePtr root = nullptr;

  NodePtr BuildImpl(const std::vector<TT>& indices, int depth) {
    const size_t poly_num = indices.size();
    if (m_max_leaf_data_num <= 0) {
      if (poly_num <= 0) {
        return nullptr;
      }
    } else {
      if (static_cast<int>(poly_num) <= m_max_leaf_data_num) {
        NodePtr node = std::make_shared<Node>();
        node->indices = indices;
        node->depth = depth;
        return node;
      }
    }

    auto [left_indices, right_indices, axis] = Split(indices, depth);

    NodePtr node = std::make_shared<Node>();
    node->axis = axis;
    node->depth = depth;
    node->bbox = Aabb(m_vertices, indices);
    node->children[0] = BuildImpl(left_indices, depth + 1);
    node->children[1] = BuildImpl(right_indices, depth + 1);

    return node;
  }

  std::tuple<std::vector<TT>, std::vector<TT>, int> Split(
      const std::vector<TT>& indices, int depth) {
    std::vector<TT> left_indices, right_indices;
    int axis = -1;
    if (m_cost_type == kCostType::NAIVE) {
      auto calc_center = [&](const TT& index) {
#if 0
        auto center = std::accumulate(
                          index.data(), index.data() + index.rows(), T::Zero(),
                          [&](const TT::Scalar& lhs, const TT::Scalar& rhs) {
                            return m_vertices[lhs] + m_vertices[rhs];
                          }) /
                      index.rows();
#endif
        T center = T::Zero();
        for (int i = 0; i < index.rows(); i++) {
          center += m_vertices[index[i]];
        }
        center /= index.rows();

        return center;
      };

      const size_t poly_num = indices.size();
      axis = depth % m_axis_num;

      const size_t mid_index = poly_num / 2;

      left_indices = indices;

      std::sort(left_indices.begin(), left_indices.end(),
                [&](const TT& lhs, const TT& rhs) {
                  auto center_lhs = calc_center(lhs);
                  auto center_rhs = calc_center(rhs);
                  return center_lhs[axis] < center_rhs[axis];
                });

      std::copy(left_indices.begin() + mid_index, left_indices.end(),
                std::back_inserter(right_indices));
      left_indices.resize(left_indices.size() - mid_index);

    } else {
      throw std::exception();
    }
    return {left_indices, right_indices, axis};
  }
};

}  // namespace ugu
