/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include "ugu/log.h"
#include "ugu/optimizer/optimizer.h"
#include "ugu/timer.h"

namespace {

double f1(const std::vector<double>& params) {
  double sum = 0;
  for (const auto& p : params) {
    sum += p * p;
  }
  return sum;
}

std::vector<double> f1_grad(const std::vector<double>& params) {
  std::vector<double> grad;  //(params.size());
  for (const auto& p : params) {
    grad.push_back(2 * p);
  }
  return grad;
}

double f2(const std::vector<double>& params) {
  double sum = 0;
  for (const auto& p : params) {
    sum += p * p * p;
  }
  return sum;
}

}  // namespace

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  ugu::Timer<> timer;
  std::vector<double> f1_init = {100.0, 50.0};
  ugu::OptimizerOutput f1_out;
  timer.Start();
  ugu::GradientDescent({f1_init, f1, f1_grad}, f1_out);
  timer.End();
  ugu::LOGI("f1 iter %d : %lf (%lf, %lf) %f ms\n", f1_out.best_iter,
            f1_out.best, f1_out.best_param[0], f1_out.best_param[1],
            timer.elapsed_msec());
  timer.Start();
  ugu::GradientDescent(
      {f1_init, f1, ugu::GenNumericalDifferentiation(f1, 0.001)}, f1_out);
  timer.End();
  ugu::LOGI("f1 numerical iter %d : %lf (%lf, %lf) %f ms\n", f1_out.best_iter,
            f1_out.best, f1_out.best_param[0], f1_out.best_param[1],
            timer.elapsed_msec());
  return 0;
}
