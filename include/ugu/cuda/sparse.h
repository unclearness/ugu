#pragma once

#include "Eigen/SparseCore"

namespace ugu {

bool SolveSparse(const Eigen::SparseMatrix<double> &mat,
                 const Eigen::MatrixXd &b, Eigen::MatrixXd &x,int devID = -1);
;
bool SolveSparse(int rowsA, int colsA,
                 const std::vector<Eigen::Triplet<double>> &triplets,
                 Eigen::VectorXd b, Eigen::VectorXd &x, int devID = -1);

bool SolveSparse(int rowsA, int colsA, int nnzA, const double *h_csrValA,
                 const int *h_csrRowPtrA, const int *h_csrColIndA,
                 const double *h_b, double *h_x, int out_col, int devID = -1);

}  // namespace ugu