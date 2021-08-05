/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#pragma once

#include <functional>
#include <stdexcept>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Sparse"

namespace ugu {

using OptParams = Eigen::VectorXd;
using GradVec = Eigen::VectorXd;
using Hessian = Eigen::MatrixXd;
using OptIndex = Eigen::Index;

using LossFunc = std::function<double(const OptParams&)>;
using GradFunc = std::function<GradVec(const OptParams&)>;
using HessianFunc = std::function<Hessian(const OptParams&)>;
// using JacobFunc
// using HessianFunc

struct OptimizerTerminateCriteria {
  int max_iter = 100;
  double eps = 1e-10;
  OptimizerTerminateCriteria(int max_iter = 10000, double eps = 1e-6)
      : max_iter(max_iter), eps(eps) {}

  bool isTerminated(int iter, double diff) const {
    diff = std::abs(diff);
    if (max_iter < 0 && 0 <= eps) {
      return diff < eps;
    } else if (0 <= max_iter && eps < 0) {
      return max_iter <= iter;
    } else if (0 <= max_iter && 0 <= eps) {
      return (diff < eps) || (max_iter <= iter);
    }
    throw std::invalid_argument("max_iter or eps must be positive");
  }
};

struct OptimizerInput {
  OptParams init_param;
  LossFunc loss_func;
  GradFunc grad_func;
  double lr = 1.0;
  OptimizerTerminateCriteria terminate_criteria;
  HessianFunc hessian_func;
  OptIndex LBFGS_memoery_num = 10;
  Eigen::SparseMatrix<double> LBFGS_init;

  OptimizerInput() = delete;
  OptimizerInput(
      const OptParams& init_param, LossFunc loss_func, GradFunc grad_func,
      double lr = 0.001,
      OptimizerTerminateCriteria terminate_criteria =
          OptimizerTerminateCriteria(),
      HessianFunc hessian_func = HessianFunc(), int LBFGS_memoery_num = 10,
      Eigen::SparseMatrix<double> LBFGS_init = Eigen::SparseMatrix<double>())
      : init_param(init_param),
        loss_func(loss_func),
        grad_func(grad_func),
        lr(lr),
        terminate_criteria(terminate_criteria),
        hessian_func(hessian_func),
        LBFGS_memoery_num(LBFGS_memoery_num),
        LBFGS_init(LBFGS_init) {
    if (this->LBFGS_memoery_num < 1) {
      LBFGS_memoery_num = 10;
    }

    if (LBFGS_init.rows() != init_param.rows()) {
      this->LBFGS_init =
          Eigen::SparseMatrix<double>(init_param.rows(), init_param.rows());
      this->LBFGS_init.setIdentity();
    }
  };
};

struct OptimizerOutput {
  double best = std::numeric_limits<double>::max();
  OptParams best_param;
  GradVec best_grad;
  int best_iter = 0;

  std::vector<double> val_history;
  std::vector<OptParams> param_history;
  std::vector<GradVec> grad_history;

  void Clear() {
    best = std::numeric_limits<double>::max();
    best_grad.setZero();
    best_param.setZero();
    best_iter = 0;
    val_history.clear();
    param_history.clear();
    grad_history.clear();
  }
};

GradFunc GenNumericalGrad(LossFunc loss_func, double h);
HessianFunc GenNumericalHessian(LossFunc loss_func, double h);

void GradientDescent(const OptimizerInput& input, OptimizerOutput& output);
void Newton(const OptimizerInput& input, OptimizerOutput& output);

enum class QuasiNewtonMethod { LBFGS };

void QuasiNewton(const OptimizerInput& input, OptimizerOutput& output,
                 QuasiNewtonMethod method = QuasiNewtonMethod::LBFGS);

}  // namespace ugu
