/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include "ugu/optimizer/optimizer.h"

namespace ugu {

GradFunc GenNumericalDifferentiation(LossFunc loss_func, double h) {
  return [loss_func, h](const std::vector<double>& params) {
    std::vector<double> grad(params.size());
    const double inv_2h = 1.0 / (2 * h);
    for (size_t i = 0; i < params.size(); i++) {
      auto params_plus = params;
      params_plus[i] += h;
      auto params_minus = params;
      params_minus[i] -= h;
      grad[i] = (loss_func(params_plus) - loss_func(params_minus)) * inv_2h;
    }
    return grad;
  };
}

void GradientDescent(const OptimizerInput& input, OptimizerOutput& output) {
  output.Clear();
  output.best_param = input.init_param;
  output.best = input.loss_func(output.best_param);
  double prev = output.best;
  double diff = std::numeric_limits<double>::max();
  while (!input.terminate_criteria.isTerminated(output.best_iter, diff)) {
    output.val_history.push_back(output.best);
    output.param_history.push_back(output.best_param);

    // Eval grad
    auto grad = input.grad_func(output.best_param);

    for (size_t i = 0; i < output.best_param.size(); i++) {
      output.best_param[i] -= (input.lr * grad[i]);
    }

    output.best = input.loss_func(output.best_param);
    diff = prev - output.best;
    prev = output.best;
    output.best_iter++;
  }
}

}  // namespace ugu
