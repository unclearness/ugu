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

  out.best_param += (in.lr * delta);
}

}  // namespace

namespace ugu {

GradFunc GenNumericalGrad(LossFunc loss_func, double h) {
  return [loss_func, h](const OptParams& params) {
    GradVec grad(params.size());
    const double inv_2h = 1.0 / (2 * h);
    for (OptIndex i = 0; i < params.size(); i++) {
      auto params_plus = params;
      params_plus[i] += h;
      auto params_minus = params;
      params_minus[i] -= h;
      grad[i] = (loss_func(params_plus) - loss_func(params_minus)) * inv_2h;
    }
    return grad;
  };
}

HessianFunc GenNumericalHessian(LossFunc loss_func, double h) {
  return [loss_func, h](const OptParams& params) {
    Hessian hessian(params.rows(), params.rows());
    const double inv_2h = 1.0 / (2 * h);
    for (OptIndex i = 0; i < params.size(); i++) {
      auto params_i_p = params;
      params_i_p[i] += h;
      auto params_i_n = params;
      params_i_n[i] -= h;
      for (OptIndex j = 0; j < params.size(); j++) {
        auto params_i_p_j_p = params_i_p;
        params_i_p_j_p[j] += h;
        auto params_i_p_j_n = params_i_p;
        params_i_p_j_n[j] -= h;

        auto grad_ij_p =
            (loss_func(params_i_p_j_p) - loss_func(params_i_p_j_n)) * inv_2h;

        auto params_i_n_j_p = params_i_n;
        params_i_n_j_p[j] += h;
        auto params_i_n_j_n = params_i_n;
        params_i_n_j_n[j] -= h;

        auto grad_ij_n =
            (loss_func(params_i_n_j_p) - loss_func(params_i_n_j_n)) * inv_2h;

        hessian(i, j) = (grad_ij_p - grad_ij_n) * inv_2h;
        // std::cout << grad_ij_p << " " << grad_ij_n << " " << hessian(i, j)
        //          << std::endl;
      }
    }
    return hessian;
  };
}

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
