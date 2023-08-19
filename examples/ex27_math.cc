/*
 * Copyright (C) 2023, unclearness
 * All rights reserved.
 */

#include "ugu/timer.h"
#include "ugu/util/math_util.h"

using namespace ugu;

int main() {
  Timer<> timer;

  Pca<Eigen::MatrixXd> pca;

  Eigen::MatrixXd data;
  data.resize(3, 100);
  data.setRandom();

  // eigen values must match
  // eigen vectors must match except sign

  std::cout << "data_num (" << data.cols() << ") > data_dim (" << data.rows()
            << ")" << std::endl;
  // this case eigendecomposition is faster
  timer.Start();
  pca.Compute(data);
  timer.End();
  std::cout << "SVD: " << timer.elapsed_msec() << " ms" << std::endl;
  std::cout << pca.coeffs << std::endl;
  std::cout << pca.vecs << std::endl;

  timer.Start();
  pca.Compute(data, true, true, false);
  timer.End();
  std::cout << "Eigendecomposition: " << timer.elapsed_msec() << " ms"
            << std::endl;
  std::cout << pca.coeffs << std::endl;
  std::cout << pca.vecs << std::endl;

  std::cout << std::endl;

  data.transposeInPlace();

  std::cout << "data_num (" << data.cols() << ") < data_dim (" << data.rows()
            << ")" << std::endl;
  // this case svd is faster
  timer.Start();
  pca.Compute(data);
  timer.End();
  std::cout << "SVD: " << timer.elapsed_msec() << " ms" << std::endl;
  std::cout << pca.coeffs << std::endl;
  // std::cout << pca.vecs << std::endl;

  timer.Start();
  pca.Compute(data, true, true, false);
  timer.End();
  std::cout << "Eigendecomposition: " << timer.elapsed_msec() << " ms"
            << std::endl;
  std::cout << pca.coeffs << std::endl;
  // std::cout << pca.vecs << std::endl;

  return 0;
}
