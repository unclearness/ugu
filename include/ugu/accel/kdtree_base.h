/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 *
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

#include "ugu/common.h"

namespace ugu {

struct KdTreeSearchResult {
  size_t index;
  double dist;
};

using KdTreeSearchResults = std::vector<KdTreeSearchResult>;

template <typename Scalar, int Rows>
class KdTree {
 public:
  using KdPoint = Eigen::Matrix<Scalar, Rows, 1>;
  virtual ~KdTree() {}
  virtual bool Build() = 0;
  virtual void Clear() = 0;
  virtual void SetData(const std::vector<KdPoint>& data) = 0;
  virtual void SetMaxLeafDataNum(int max_leaf_data_num) = 0;

  virtual KdTreeSearchResults SearchNn(const KdPoint& query) const = 0;
  virtual KdTreeSearchResults SearchKnn(const KdPoint& query,
                                        const size_t& k) const = 0;
  virtual KdTreeSearchResults SearchRadius(const KdPoint& query,
                                           const double& r) const = 0;
};

template <typename Scalar, int Rows>
using KdTreePtr = std::shared_ptr<KdTree<Scalar, Rows>>;

template <typename Scalar, int Rows>
using KdTreeUniquePtr = std::unique_ptr<KdTree<Scalar, Rows>>;

}  // namespace ugu
