/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#pragma once


#include "Eigen/Core"
#include <functional>
#include <stdexcept>
#include <vector>

namespace ugu {

using OptParams = Eigen::VectorXd;
using GradVec = Eigen::VectorXd;
using Hessian = Eigen::MatrixXd;

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

  OptimizerInput() = delete;
  OptimizerInput(const OptParams& init_param, LossFunc loss_func,
                 GradFunc grad_func, double lr = 0.001,
                 OptimizerTerminateCriteria terminate_criteria =
                     OptimizerTerminateCriteria(),
                 HessianFunc hessian_func = HessianFunc())
      : init_param(init_param),
        loss_func(loss_func),
        grad_func(grad_func),
        lr(lr),
        terminate_criteria(terminate_criteria),
        hessian_func(hessian_func) {};
};

struct OptimizerOutput {
  double best = std::numeric_limits<double>::max();
  OptParams best_param;
  int best_iter = 0;

  std::vector<double> val_history;
  std::vector<OptParams> param_history;

  void Clear() {
    best = std::numeric_limits<double>::max();
    best_param.setZero();
    best_iter = 0;
    val_history.clear();
    param_history.clear();
  }
};

GradFunc GenNumericalDifferentiation(LossFunc loss_func, double h);

void GradientDescent(const OptimizerInput& input, OptimizerOutput& output);
void Newton(const OptimizerInput& input, OptimizerOutput& output);

}  // namespace ugu
