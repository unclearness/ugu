/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 *
 */

#pragma once

#include <algorithm>
#include <memory>
#include <vector>

namespace ugu {

using KdTreeSearchResult = std::pair<size_t, double>;
using KdTreeSearchResults = std::vector<KdTreeSearchResult>;

void UpdateKdTreeSearchResult(KdTreeSearchResults& res, const size_t& index,
                              const double& dist, const size_t& max_res_num) {
  res.push_back({index, dist});
  std::sort(res.begin(), res.end(),
            [](const KdTreeSearchResult& lfs, const KdTreeSearchResult& rfs) {
              return lfs.second < rfs.second;
            });

  if (max_res_num == 0 || res.size() < max_res_num) {
    return;
  }

  res.resize(max_res_num);

  return;
}

template <typename Point>
class KdTree {
 public:
  KdTree() {}
  ~KdTree() {}
  void SetAxisNum(int axis_num);
  void SetData(const std::vector<Point>& data);
  void SetMaxLeafDataNum(int max_leaf_data_num);
  const std::vector<Point>& Data() const;
  void Build();
  void Clear();

  KdTreeSearchResults SearchNn(const Point& query) const;
  KdTreeSearchResults SearchKnn(const Point& query, const size_t& k) const;
  KdTreeSearchResults SearchRadius(const Point& query, const double& r) const;
  double EuclidDist(const Point& p1, const Point& p2) const;

 private:
  std::vector<Point> m_data;
  std::vector<int> m_indices;
  int m_axis_num = -1;
  int m_max_leaf_data_num = -1;

  struct Node;
  using NodePtr = std::shared_ptr<Node>;
  struct Node {
    std::vector<size_t> indices;
    int depth = -1;
    int axis = -1;
    NodePtr left = nullptr;
    NodePtr right = nullptr;
    bool IsLeaf() const {
      if (axis < 0 || left == nullptr || right == nullptr) {
        return true;
      }
      return false;
    }
  };
  NodePtr root = nullptr;

  NodePtr BuildImpl(size_t start, size_t end, int depth);
  void SearchKnnImpl(const Point& query, const size_t& k, const NodePtr node,
                     KdTreeSearchResults& result) const;
  void SearchRadiusImpl(const Point& query, const double& r, const NodePtr node,
                        KdTreeSearchResults& result) const;
};

template <typename Point>
void KdTree<Point>::SetAxisNum(int axis_num) {
  m_axis_num = axis_num;
}

template <typename Point>
void KdTree<Point>::SetData(const std::vector<Point>& data) {
  m_data = data;
}

template <typename Point>
void KdTree<Point>::SetMaxLeafDataNum(int max_leaf_data_num) {
  m_max_leaf_data_num = max_leaf_data_num;
}

template <typename Point>
const std::vector<Point>& KdTree<Point>::Data() const {
  return m_data;
}

template <typename Point>
void KdTree<Point>::Build() {
  if (m_axis_num < 0 || m_data.empty()) {
    return;
  }
  m_indices.clear();
  m_indices.resize(m_data.size());
  std::iota(m_indices.begin(), m_indices.end(), 0);
  root = BuildImpl(0, m_indices.size(), 0);
}

template <typename Point>
void KdTree<Point>::Clear() {
  m_data.clear();
  m_indices.clear();
}

template <typename Point>
KdTreeSearchResults KdTree<Point>::SearchNn(const Point& query) const {
  return SearchKnn(query, 1);
}

template <typename Point>
KdTreeSearchResults KdTree<Point>::SearchKnn(const Point& query,
                                             const size_t& k) const {
  KdTreeSearchResults res;
  SearchKnnImpl(query, k, root, res);
  return res;
}

template <typename Point>
KdTreeSearchResults KdTree<Point>::SearchRadius(const Point& query,
                                                const double& r) const {
  KdTreeSearchResults res;
  SearchRadiusImpl(query, r, root, res);
  return res;
}

template <typename Point>
double KdTree<Point>::EuclidDist(const Point& p1, const Point& p2) const {
  double d = 0.0;
  for (int i = 0; i < m_axis_num; i++) {
    double diff = static_cast<double>(p1[i] - p2[i]);
    d += diff * diff;
  }
  return std::sqrt(d);
}

template <typename Point>
typename KdTree<Point>::NodePtr KdTree<Point>::BuildImpl(size_t start,
                                                         size_t end,
                                                         int depth) {
  const size_t points_num = end - start + 1;
  const int axis = depth % m_axis_num;

  if (m_max_leaf_data_num <= 0) {
    if (points_num <= 0) {
      return nullptr;
    }
  } else {
    if (static_cast<int>(points_num) <= m_max_leaf_data_num) {
      NodePtr node = std::make_shared<Node>();
      auto end_ = std::min(m_data.size() - 1, end);
      std::sort(m_indices.begin() + start, m_indices.begin() + end_ + 1,
                [&](size_t lhs, size_t rhs) {
                  return m_data[lhs][axis] < m_data[rhs][axis];
                });
      for (size_t i = start; i <= end_; i++) {
        node->indices.push_back(m_indices[i]);
      }
      node->depth = depth;
      node->axis = axis;
      return node;
    }
  }

  const size_t mid_index = (points_num - 1) / 2 + start;

  std::nth_element(m_indices.begin() + start, m_indices.begin() + mid_index,
                   m_indices.begin() + end, [&](size_t lhs, size_t rhs) {
                     return m_data[lhs][axis] < m_data[rhs][axis];
                   });

  NodePtr node = std::make_shared<Node>();
  node->indices.push_back(m_indices[mid_index]);
  node->axis = axis;
  node->depth = depth;

  node->left = BuildImpl(start, mid_index, depth + 1);
  node->right = BuildImpl(mid_index + 1, end, depth + 1);

  return node;
}

template <typename Point>
void KdTree<Point>::SearchKnnImpl(const Point& query, const size_t& k,
                                  const NodePtr node,
                                  KdTreeSearchResults& result) const {
  if (node == nullptr) {
    return;
  }
  for (const auto& i : node->indices) {
    const double dist = EuclidDist(query, m_data[i]);
    UpdateKdTreeSearchResult(result, i, dist, k);
  }
  const int axis = node->axis;
  const auto& pivot = m_data[node->indices[0]][axis];
  NodePtr next = query[axis] < pivot ? node->left : node->right;
  SearchKnnImpl(query, k, next, result);

  // If (1) updated minimum distance is larger than difference of the axis OR
  // (2) result contains less than k, search another child node
  const double diff = std::abs(query[axis] - pivot);
  if (result.size() < k || diff < result.back().second) {
    NodePtr next2 = (next == node->right) ? node->left : node->right;
    SearchKnnImpl(query, k, next2, result);
  }
}

template <typename Point>
void KdTree<Point>::SearchRadiusImpl(const Point& query, const double& r,
                                     const NodePtr node,
                                     KdTreeSearchResults& result) const {
  if (node == nullptr) {
    return;
  }
  for (const auto& i : node->indices) {
    const double dist = EuclidDist(query, m_data[i]);
    if (dist <= r) {
      UpdateKdTreeSearchResult(result, i, dist, 0);
    }
  }
  const int axis = node->axis;
  const auto& pivot = m_data[node->indices[0]][axis];
  NodePtr next = query[axis] < pivot ? node->left : node->right;
  SearchRadiusImpl(query, r, next, result);

  const double diff = std::abs(query[axis] - pivot);
  if (diff <= r) {
    NodePtr next2 = (next == node->right) ? node->left : node->right;
    SearchRadiusImpl(query, r, next2, result);
  }
}

}  // namespace ugu
