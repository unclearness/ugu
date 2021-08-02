/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include "ugu/image.h"
#include "ugu/timer.h"
#include "ugu/util/thread_util.h"

namespace {

void ForSpeedTest() {
  const ugu::Image3b color_org =
      ugu::imread<ugu::Image3b>("../data/inpaint/fruits.jpg");
  ugu::Image3b color;
  const int ksize = 31;
  const int hksize = ksize / 2;
  ugu::Timer<> timer;

  // Simple box blur
  auto loop_body = [&](int j) {
    for (int i = 0; i < color.cols; i++) {
      double ave[3] = {0, 0, 0};
      int count = 0;
      for (int jj = -hksize; jj <= hksize; jj++) {
        for (int ii = -hksize; ii <= hksize; ii++) {
          if (j + jj < 0 || color.rows - 1 < j + jj || i + ii < 0 ||
              color.cols - 1 < i + ii) {
            continue;
          }
          const auto& pix_org = color_org.at<ugu::Vec3b>(j + jj, i + ii);
          ave[0] += pix_org[0];
          ave[1] += pix_org[1];
          ave[2] += pix_org[2];
          count++;
        }
      }
      ave[0] /= count;
      ave[1] /= count;
      ave[2] /= count;

      ave[0] = std::max(std::min(255.0, ave[0]), 0.0);
      ave[1] = std::max(std::min(255.0, ave[1]), 0.0);
      ave[2] = std::max(std::min(255.0, ave[2]), 0.0);

      auto& pix = color.at<ugu::Vec3b>(j, i);
      pix[0] = static_cast<unsigned char>(ave[0]);
      pix[1] = static_cast<unsigned char>(ave[1]);
      pix[2] = static_cast<unsigned char>(ave[2]);
    }
  };

  // Naive
  color = color_org.clone();
  timer.Start();
  for (int j = 0; j < color.rows; j++) {
    loop_body(j);
  }
  timer.End();
  ugu::LOGI("Naive: %f ms\n", timer.elapsed_msec());
  ugu::imwrite("fruits_blur_naive.jpg", color);

  // OpenMP
  color = color_org.clone();
  timer.Start();
#pragma omp parallel for
  for (int j = 0; j < color.rows; j++) {
    loop_body(j);
  }
  timer.End();
  ugu::LOGI("OpenMP: %f ms\n", timer.elapsed_msec());
  ugu::imwrite("fruits_blur_omp.jpg", color);

  // ugu::parallel_for
  color = color_org.clone();
  timer.Start();
  ugu::parallel_for(0, color.rows, loop_body);
  timer.End();
  ugu::LOGI("ugu::parallel_for: %f ms\n", timer.elapsed_msec());
  ugu::imwrite("fruits_blur_ugu.jpg", color);
}

}  // namespace

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  ForSpeedTest();

  return 0;
}
