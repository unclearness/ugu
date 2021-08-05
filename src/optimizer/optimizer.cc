/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include "ugu/optimizer/optimizer.h"

#include <Eigen/LU>
#include <iostream>

namespace {

using UpdateFunc = std::function<void(const ugu::OptimizerInput& in,
                                      ugu::OptimizerOutput& out)>;

void LoopBody(const ugu::OptimizerInput& input, ugu::OptimizerOutput& output,
              UpdateFunc update_func) {
  output.Clear();
  output.best_param = input.init_param;
  output.best = input.loss_func(output.best_param);
  double prev = output.best;
  double diff = std::numeric_limits<double>::max();
  while (!input.terminate_criteria.isTerminated(output.best_iter, diff)) {
    output.val_history.push_back(output.best);
    output.param_history.push_back(output.best_param);

    update_func(input, output);

    output.best = input.loss_func(output.best_param);
    diff = prev - output.best;
    prev = output.best;
    output.best_iter++;
    // std::cout << prev << std::endl;
  }
}

void NewtonUpdateFunc(const ugu::OptimizerInput& in,
                      ugu::OptimizerOutput& out) {
  ugu::GradVec grad = in.grad_func(out.best_param);
  ugu::Hessian hessian = in.hessian_func(out.best_param);
  ugu::OptParams delta = (-hessian.inverse() * grad);
  // std::cout << hessian << " "
  //           << hessian.inverse() << std::endl;
  // for (size_t i = 0; i < out.best_param.size(); i++) {
  //  out.best_param[i] += (in.lr * delta[i]);
  //}
  out.best_param += (in.lr * delta);
}

}  // namespace

namespace ugu {

GradFunc GenNumericalDifferentiation(LossFunc loss_func, double h) {
  return [loss_func, h](const OptParams& params) {
    GradVec grad(params.size());
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

#if 0
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
#endif  // 0

void GradientDescent(const OptimizerInput& input, OptimizerOutput& output) {
  LoopBody(input, output,
           [&](const ugu::OptimizerInput& in, ugu::OptimizerOutput& out) {
             // Eval grad
             auto grad = in.grad_func(out.best_param);
             // for (size_t i = 0; i < out.best_param.size(); i++) {
             //  out.best_param[i] -= (in.lr * grad[i]);
             //}
             out.best_param -= (in.lr * grad);
           });
}

void Newton(const OptimizerInput& input, OptimizerOutput& output) {
  LoopBody(input, output, NewtonUpdateFunc);
}

}  // namespace ugu
