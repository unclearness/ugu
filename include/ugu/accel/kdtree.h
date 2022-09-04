/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 *
 */

#pragma once

#include "ugu/accel/kdtree_base.h"
#include "ugu/accel/kdtree_naive.h"
#include "ugu/accel/kdtree_nanoflann.h"

namespace ugu {

template <typename Scalar>
KdTreePtr<Scalar, Eigen::Dynamic> GetDefaultKdTreeDynamic() {
  KdTreePtr<Scalar, Eigen::Dynamic> kdtree;
#ifdef UGU_USE_NANOFLANN
  kdtree = std::make_shared<KdTreeNanoflannEigenX<float>>();
#else
  kdtree = std::make_shared<KdTreeNaive<float, Eigen::Dynamic>>();
#endif
  return kdtree;
}

template <typename Scalar, int Rows>
KdTreePtr<Scalar, Rows> GetDefaultKdTree() {
  KdTreePtr<Scalar, Rows> kdtree;
#ifdef UGU_USE_NANOFLANN
  kdtree = std::make_shared<KdTreeNanoflannVector<float, Rows>>();
#else
  kdtree = std::make_shared<KdTreeNaive<float, Rows>>();
#endif
  return kdtree;
}

template <typename Scalar>
KdTreeUniquePtr<Scalar, Eigen::Dynamic> GetDefaultUniqueKdTreeDynamic() {
  KdTreeUniquePtr<Scalar, Eigen::Dynamic> kdtree;
#ifdef UGU_USE_NANOFLANN
  kdtree = std::make_unique<KdTreeNanoflannEigenX<float>>();
#else
  kdtree = std::make_unique<KdTreeNaive<float, Eigen::Dynamic>>();
#endif
  return kdtree;
}

template <typename Scalar, int Rows>
KdTreeUniquePtr<Scalar, Rows> GetDefaultUniqueKdTree() {
  KdTreeUniquePtr<Scalar, Rows> kdtree;
#ifdef UGU_USE_NANOFLANN
  kdtree = std::make_unique<KdTreeNanoflannVector<float, Rows>>();
#else
  kdtree = std::make_unique<KdTreeNaive<float, Rows>>();
#endif
  return kdtree;
}
}  // namespace ugu
