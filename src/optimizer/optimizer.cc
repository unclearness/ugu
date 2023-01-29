/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include "ugu/optimizer/optimizer.h"

#ifdef _WIN32
#pragma warning(push, UGU_EIGEN_WARNING_LEVEL)
#endif
#include <Eigen/LU>
#ifdef _WIN32
#pragma warning(pop)
#endif

#include <iostream>

namespace {

using UpdateFunc = std::function<void(const ugu::OptimizerInput& in,
                                      ugu::OptimizerOutput& out)>;

double LineSearchBacktracking(const ugu::GradVec& update_direc,
                              const ugu::OptimizerInput& in,
                              ugu::OptimizerOutput& out) {
  // BACK_TRACKING
  double a = in.line_search_max;
  double r = 0.5;
  while (true) {
    // Armijo rule
    double c1 = 1e-4;
    auto new_param = out.best_param + a * update_direc;
    double new_val = in.loss_func(new_param);

    double constraint = out.best + c1 * a * out.best_grad.dot(update_direc);

    if (new_val <= constraint) {
      return a;
    }

    a *= r;
  }
}

double LineSearch(const ugu::GradVec& update_direc,
                  const ugu::OptimizerInput& in, ugu::OptimizerOutput& out) {
  if (in.line_search_method == ugu::LineSearchMethod::BACK_TRACKING) {
    return LineSearchBacktracking(update_direc, in, out);
  }

  throw std::invalid_argument("This type is not implemented");

  // return 0.0;
}

void LoopBody(const ugu::OptimizerInput& input, ugu::OptimizerOutput& output,
              UpdateFunc update_func) {
  output.Clear();
  output.best_param = input.init_param;
  output.best = input.loss_func(output.best_param);
  output.best_grad = input.grad_func(output.best_param);

  output.val_history.push_back(output.best);
  output.param_history.push_back(output.best_param);
  output.grad_history.push_back(output.best_grad);

  double prev = output.best;
  double diff = std::numeric_limits<double>::max();
  size_t max_size = static_cast<size_t>(input.LBFGS_memoery_num);
  while (!input.terminate_criteria.isTerminated(output.best_iter, diff)) {
    prev = output.best;

    update_func(input, output);

    output.best = input.loss_func(output.best_param);
    output.best_grad = input.grad_func(output.best_param);

    output.val_history.push_back(output.best);
    output.param_history.push_back(output.best_param);
    output.grad_history.push_back(output.best_grad);
    while (max_size < output.val_history.size()) {
      output.val_history.erase(output.val_history.begin());
      output.param_history.erase(output.param_history.begin());
      output.grad_history.erase(output.grad_history.begin());
    }

    // output.best = input.loss_func(output.best_param);
    diff = prev - output.best;
    output.best_iter++;
    // std::cout << prev << std::endl;
  }
}

void GradientDescentUdpateFunc(const ugu::OptimizerInput& in,
                               ugu::OptimizerOutput& out) {
  // Eval grad
  auto grad = in.grad_func(out.best_param);

  double lr = in.lr;

  if (in.use_line_search) {
    lr = LineSearch(-grad, in, out);
  }

  out.best_param += (lr * -grad);
}

// http://www.dais.is.tohoku.ac.jp/~shioura/teaching/mp13/mp13-13.pdf
void NewtonUpdateFunc(const ugu::OptimizerInput& in,
                      ugu::OptimizerOutput& out) {
  ugu::GradVec grad = in.grad_func(out.best_param);
  // out.best_grad = grad;
  ugu::Hessian hessian = in.hessian_func(out.best_param);
  ugu::OptParams delta = (-hessian.inverse() * grad);

  double lr = in.lr;

  if (in.use_line_search) {
    lr = LineSearch(delta, in, out);
  }

  out.best_param += (lr * delta);
}

// https://en.wikipedia.org/wiki/Limited-memory_BFGS
// https://abicky.net/2010/06/22/114613/
void LBFGSUpdateFunc(const ugu::OptimizerInput& in, ugu::OptimizerOutput& out) {
  ugu::GradVec grad = in.grad_func(out.best_param);
  // out.best_grad = grad;
  ugu::GradVec q = grad;

  if (out.grad_history.size() < 2) {
    // Initialize by GradientDescent
    GradientDescentUdpateFunc(in, out);
    return;
  }

  ugu::OptIndex k = static_cast<ugu::OptIndex>(out.grad_history.size() - 1);
  std::vector<ugu::GradVec> y_list;
  std::vector<ugu::GradVec> s_list;
  std::vector<double> rho_list;

  for (ugu::OptIndex i = k; i >= 1; i--) {
    ugu::GradVec y = out.grad_history[i] - out.grad_history[i - 1];
    ugu::GradVec s = out.param_history[i] - out.param_history[i - 1];
    double dot = y.dot(s);

#if 0
    // Guard
    if (std::abs(dot) < 1e-10) {
      if (dot > 0) {
        dot = 1e-10;
      } else {
        dot = -1e-10;
      }
    }
    if (std::abs(dot) < 1e10) {
      if (dot > 0) {
        dot = 1e10;
      } else {
        dot = -1e10;
      }
    }
#endif

    double rho = 1.0 / dot;
#if 0
    if (std::isnan(rho)) {
        rho = 0;
    }
    std::cout << y << std::endl;
    std::cout << s << std::endl;
    std::cout << rho << std::endl;
    std::cout << std::endl;

    if (std::abs(rho) < 1e-10) {
      if (rho > 0) {
        rho = 1e-10;
      } else {
        rho = -1e-10;
      }
    }
#endif
    y_list.push_back(y);
    s_list.push_back(s);
    rho_list.push_back(rho);
  }

  std::vector<double> alpha_list;

  for (ugu::OptIndex i = 0; i < k; i++) {
    double alpha = rho_list[i] * s_list[i].dot(q);
    q = q - alpha * y_list[i];
    alpha_list.push_back(alpha);
  }

  double gamma = s_list[0].dot(y_list[0]) / y_list[0].dot(y_list[0]);
  Eigen::SparseMatrix<double> I(in.init_param.rows(), in.init_param.rows());
  I.setIdentity();
  Eigen::SparseMatrix<double> H = gamma * I;

  // std::cout << H << std::endl;

  ugu::GradVec z = H * q;

  for (ugu::OptIndex i = k - 1; i >= 0; i--) {
    double beta = rho_list[i] * y_list[i].dot(z);
    z = z + s_list[i] * (alpha_list[i] - beta);
  }

  z = -z;

  double lr = in.lr;

  if (in.use_line_search) {
    lr = LineSearch(z, in, out);
  }

  out.best_param += (lr * z);
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
  LoopBody(input, output, GradientDescentUdpateFunc);
}

void Newton(const OptimizerInput& input, OptimizerOutput& output) {
  LoopBody(input, output, NewtonUpdateFunc);
}

void QuasiNewton(const OptimizerInput& input, OptimizerOutput& output,
                 QuasiNewtonMethod method) {
  if (method == QuasiNewtonMethod::LBFGS) {
    LoopBody(input, output, LBFGSUpdateFunc);
  } else {
    throw std::invalid_argument("This method is not supported");
  }
}

}  // namespace ugu
