/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#pragma once

#include <Eigen/Core>
#include <functional>

static_assert(3 <= EIGEN_WORLD_VERSION);

// From 3.3, Eigen::Index is provided
// http://eigen.tuxfamily.org/index.php?title=3.3
#if (3 == EIGEN_WORLD_VERSION) && (EIGEN_MAJOR_VERSION < 3)
namespace Eigen {
typedef std::ptrdiff_t Index;
}
#endif

namespace std {

template <typename Scalar, int Rows, int Cols>
struct hash<Eigen::Matrix<Scalar, Rows, Cols>> {
  // https://wjngkoh.wordpress.com/2015/03/04/c-hash-function-for-eigen-matrix-and-vector/
  size_t operator()(const Eigen::Matrix<Scalar, Rows, Cols>& matrix) const {
    size_t seed = 0;
    for (Eigen::Index i = 0; i < matrix.size(); ++i) {
      Scalar elem = *(matrix.data() + i);
      seed ^=
          std::hash<Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

}  // namespace std