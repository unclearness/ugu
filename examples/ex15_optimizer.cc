/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include <iostream>

#include "ugu/log.h"
#include "ugu/optimizer/optimizer.h"
#include "ugu/timer.h"

namespace {

double f1(const ugu::OptParams& params) {
  double sum = 0;
  for (ugu::OptIndex i = 0; i < params.size(); i++) {
    const auto& p = params[i];
    sum += p * p;
  }
  return sum;
}

ugu::GradVec f1_grad(const ugu::OptParams& params) {
  ugu::GradVec grad(params.size());
  for (ugu::OptIndex i = 0; i < params.size(); i++) {
    const auto& p = params[i];
    grad[i] = 2 * p;
  }
  return grad;
}

ugu::Hessian f1_hessian(const ugu::OptParams& params) {
  ugu::Hessian h(params.rows(), params.rows());
  (void)params;
  h.setIdentity();
  h = h * 2.0;
  return h;
}

double f2(const std::vector<double>& params) {
  double sum = 0;
  for (const auto& p : params) {
    sum += p * p * p;
  }
  return sum;
}

// https://en.wikipedia.org/wiki/Rosenbrock_function
double Rosenbrock2(const ugu::OptParams& params) {
  double a = 1.0;
  double b = 100.0;
  double a_diff = (a - params[0]);
  double y_diff = (params[1] - params[0] * params[0]);
  return a_diff * a_diff + b * y_diff * y_diff;
}

ugu::GradVec Rosenbrock2_grad(const ugu::OptParams& params) {
  ugu::GradVec grad(params.size());
  double a = 1.0;
  double b = 100.0;

  grad[0] = 4 * b * params[0] * params[0] * params[0] +
            2 * (1 - 2 * b * params[1]) * params[0] - 2 * a;
  grad[1] = 2 * b * params[1] - 2 * b * params[0] * params[0];
  return grad;
}

ugu::Hessian Rosenbrock2_hessian(const ugu::OptParams& params) {
  ugu::Hessian h(params.rows(), params.rows());
  (void)params;
  // double a = 1.0;
  double b = 100.0;
  h(0, 0) = 12 * b * params[0] * params[0] + 2 * (1 - 2 * b * params[1]);
  h(0, 1) = -4 * b * params[0];
  h(1, 0) = -4 * b * params[0];
  h(1, 1) = 2 * b;
  return h;
}

}  // namespace

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  ugu::Timer<> timer;
  Eigen::VectorXd f1_init(2);
  f1_init[0] = 2.5;
  f1_init[1] = 3.1;
  ugu::OptimizerOutput f1_out;

#if 1
  timer.Start();
  ugu::GradientDescent({f1_init, f1, f1_grad}, f1_out);
  timer.End();
  ugu::LOGI("f1 iter %d : %lf (%lf, %lf) %f ms\n", f1_out.best_iter,
            f1_out.best, f1_out.best_param[0], f1_out.best_param[1],
            timer.elapsed_msec());
  timer.Start();
  ugu::GradientDescent({f1_init, f1, ugu::GenNumericalGrad(f1, 0.001)}, f1_out);
  timer.End();
  ugu::LOGI("f1 numerical iter %d : %lf (%lf, %lf) %f ms\n", f1_out.best_iter,
            f1_out.best, f1_out.best_param[0], f1_out.best_param[1],
            timer.elapsed_msec());

  timer.Start();
  ugu::Newton(
      {f1_init, f1, f1_grad, 1, ugu::OptimizerTerminateCriteria(), f1_hessian},
      f1_out);
  timer.End();
  ugu::LOGI("f1 newton iter %d : %lf (%lf, %lf) %f ms\n", f1_out.best_iter,
            f1_out.best, f1_out.best_param[0], f1_out.best_param[1],
            timer.elapsed_msec());

  timer.Start();
  ugu::QuasiNewton({f1_init, f1, f1_grad, 0.01,
                    ugu::OptimizerTerminateCriteria(), f1_hessian},
                   f1_out);
  timer.End();
  ugu::LOGI("f1 LBFGS iter %d : %lf (%lf, %lf) %f ms\n", f1_out.best_iter,
            f1_out.best, f1_out.best_param[0], f1_out.best_param[1],
            timer.elapsed_msec());

  timer.Start();
  ugu::GradientDescent({f1_init, Rosenbrock2, Rosenbrock2_grad, 0.001}, f1_out);
  timer.End();
  ugu::LOGI("Rosenbrock2 iter %d : %lf (%lf, %lf) %f ms\n", f1_out.best_iter,
            f1_out.best, f1_out.best_param[0], f1_out.best_param[1],
            timer.elapsed_msec());
  timer.Start();
  ugu::GradientDescent(
      {f1_init, Rosenbrock2, ugu::GenNumericalGrad(Rosenbrock2, 0.001), 0.001},
      f1_out);
  timer.End();
  ugu::LOGI("Rosenbrock2 numerical iter %d : %lf (%lf, %lf) %f ms\n",
            f1_out.best_iter, f1_out.best, f1_out.best_param[0],
            f1_out.best_param[1], timer.elapsed_msec());

  timer.Start();
  ugu::Newton({f1_init, Rosenbrock2, Rosenbrock2_grad, 0.01,
               ugu::OptimizerTerminateCriteria(), Rosenbrock2_hessian},
              f1_out);
  timer.End();
  ugu::LOGI("Rosenbrock2 newton iter %d : %lf (%lf, %lf) %f ms\n",
            f1_out.best_iter, f1_out.best, f1_out.best_param[0],
            f1_out.best_param[1], timer.elapsed_msec());

  timer.Start();
  ugu::Newton({f1_init, Rosenbrock2, Rosenbrock2_grad, 0.01,
               ugu::OptimizerTerminateCriteria(),
               ugu::GenNumericalHessian(Rosenbrock2, 0.01)},
              f1_out);
  timer.End();
  ugu::LOGI("Rosenbrock2 newton numerical iter %d : %lf (%lf, %lf) %f ms\n",
            f1_out.best_iter, f1_out.best, f1_out.best_param[0],
            f1_out.best_param[1], timer.elapsed_msec());

#endif
  timer.Start();
  ugu::QuasiNewton({f1_init, Rosenbrock2, Rosenbrock2_grad, 0.01,
                    ugu::OptimizerTerminateCriteria()},
                   f1_out);
  timer.End();
  ugu::LOGI("Rosenbrock2 LBFGS iter %d : %lf (%lf, %lf) %f ms\n",
            f1_out.best_iter, f1_out.best, f1_out.best_param[0],
            f1_out.best_param[1], timer.elapsed_msec());

  timer.Start();
  ugu::QuasiNewton({f1_init, Rosenbrock2, Rosenbrock2_grad, 0.001,
                    ugu::OptimizerTerminateCriteria(), ugu::HessianFunc(), 10,
                    Eigen::SparseMatrix<double>(), true,
                    ugu::LineSearchMethod::BACK_TRACKING},
                   f1_out);
  timer.End();
  ugu::LOGI("Rosenbrock2 LBFGS line search iter %d : %lf (%lf, %lf) %f ms\n",
            f1_out.best_iter, f1_out.best, f1_out.best_param[0],
            f1_out.best_param[1], timer.elapsed_msec());

  return 0;
}
