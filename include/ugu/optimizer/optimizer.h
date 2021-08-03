/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#pragma once

#include <functional>
#include <stdexcept>
#include <vector>

namespace ugu {

using LossFunc = std::function<double(const std::vector<double>&)>;
using GradFunc = std::function<std::vector<double>(const std::vector<double>&)>;
// using JacobFunc
// using HessianFunc

struct OptimizerTerminateCriteria {
  int max_iter = 100;
  double eps = 1e-10;
  OptimizerTerminateCriteria(int max_iter = 2000, double eps = 1e-6)
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
  std::vector<double> init_param;
  LossFunc loss_func;
  GradFunc grad_func;
  double lr = 1.0;
  OptimizerTerminateCriteria terminate_criteria;

  OptimizerInput() = delete;
  OptimizerInput(std::vector<double> init_param, LossFunc loss_func,
                 GradFunc grad_func, double lr = 0.01,
                 OptimizerTerminateCriteria terminate_criteria =
                     OptimizerTerminateCriteria())
      : init_param(init_param),
        loss_func(loss_func),
        grad_func(grad_func),
        lr(lr),
        terminate_criteria(terminate_criteria){};
};

struct OptimizerOutput {
  double best = std::numeric_limits<double>::max();
  std::vector<double> best_param;
  int best_iter = 0;
  ;

  std::vector<double> val_history;
  std::vector<std::vector<double>> param_history;

  void Clear() {
    best = std::numeric_limits<double>::max();
    best_param.clear();
    best_iter = 0;
    val_history.clear();
    param_history.clear();
  }
};

GradFunc GenNumericalDifferentiation(LossFunc loss_func, double h);

void GradientDescent(const OptimizerInput& input, OptimizerOutput& output);

}  // namespace ugu
