/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#pragma once

#include <iostream>

#include "ugu/inflation/inflation.h"

#include "Eigen/Sparse"

namespace {

bool InflationBaran(const ugu::Image1b& mask, ugu::Image1f& height) {
  constexpr double f = -4.0;

  height = ugu::Image1f::zeros(mask.rows, mask.cols);

  const size_t elem_num = mask.rows * mask.cols;

  // 4 neighbor laplacian
  std::vector<Eigen::Triplet<double>> triplets;
  for ( int j = 1; j < mask.rows - 1; j++) {
    for ( int i = 1; i < mask.cols - 1; i++) {
#if 0
      if (mask.at<unsigned char>(j, i) != 0) {
        triplets.push_back({i, j, -4.0});
      }
      if (mask.at<unsigned char>(j, i - 1) != 0) {
        triplets.push_back({i - 1, j, 1.0});
      }
      if (mask.at<unsigned char>(j, i + 1) != 0) {
        triplets.push_back({i + 1, j, 1.0});
      }
      if (mask.at<unsigned char>(j - 1, i) != 0) {
        triplets.push_back({i, j - 1, 1.0});
      }
      if (mask.at<unsigned char>(j + 1, i) != 0) {
        triplets.push_back({i, j + 1, 1.0});
      }

#else
      if (mask.at<unsigned char>(j, i) != 0) {
        triplets.push_back({i + 1, j + 1, -4.0});
      }
      if (mask.at<unsigned char>(j, i - 1) != 0) {
        triplets.push_back({i, j + 1, 1.0});
      }
      if (mask.at<unsigned char>(j, i + 1) != 0) {
        triplets.push_back({i + 2, j + 1, 1.0});
      }
      if (mask.at<unsigned char>(j - 1, i) != 0) {
        triplets.push_back({i + 1, j, 1.0});
      }
      if (mask.at<unsigned char>(j + 1, i) != 0) {
        triplets.push_back({i + 1, j + 2, 1.0});
      }
#endif  // 0
    }
  }

  Eigen::SparseMatrix<double> A(elem_num, elem_num);
  A.setFromTriplets(triplets.begin(), triplets.end());
  // TODO: Is this solver the best?
  Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> solver;
  solver.compute(A);

  Eigen::VectorXd b(elem_num);
  b.setZero();
  for (unsigned int j = 1; j < mask.rows - 1; j++) {
    for (unsigned int i = 1; i < mask.cols - 1; i++) {
      if (mask.at<unsigned char>(j, i) != 0) {
        b[j * mask.rows + i] = f;
      }
// else {
      //  b[j * mask.rows + i] = 0.0;
     // }
    }
  }

  Eigen::VectorXd solution = solver.solve(b);

#if 0
				  {
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
    std::cout << "#iterations:     " << cg.iterations() << std::endl;
    cg.compute(A);
    std::cout << "#iterations:     " << cg.iterations() << std::endl;
    Eigen::VectorXd x = cg.solve(b);
    std::cout << "#iterations:     " << cg.iterations() << std::endl;
    std::cout << "estimated error: " << cg.error() << std::endl;
    // update b, and solve again
    x = cg.solve(b);
  }
#endif  // 0


  for (unsigned int j = 1; j < mask.rows - 1; j++) {
    for (unsigned int i = 1; i < mask.cols - 1; i++) {
      height.at<float>(j, i) =
          static_cast<float>(std::sqrt(solution[j * mask.rows + i]));
      if (solution[j * mask.rows + i] < 0) {
        ugu::LOGI("%f\n", solution[j * mask.rows + i]);
      }
    }
  }

  return true;
}

}  // namespace

namespace ugu {

bool Inflation(const Image1b& mask, Image1f& height, InflationMethod method) {
  if (method == InflationMethod::BARAN) {
    return InflationBaran(mask, height);
  }

  return false;
}

}  // namespace ugu
