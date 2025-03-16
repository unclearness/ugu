/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include <stdio.h>

#include <fstream>

#include "ugu/cuda/image.h"
#include "ugu/image.h"
#include "ugu/image_io.h"
#include "ugu/image_proc.h"
#include "ugu/timer.h"
#include "ugu/util/image_util.h"
#include "ugu/util/rgbd_util.h"

#ifdef UGU_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace {
void TestNormal() {
  std ::vector<ugu::Image1f> depths;
  std::vector<ugu::Image3f> normals;
  std::vector<ugu::PinholeCameraPtr> cameras;

  ugu::Image1w imgw = ugu::imread("../data/bunny/00000_depth.png");
  int target_width = 640;
  float r = static_cast<float>(target_width) / static_cast<float>(imgw.cols);
  int target_height = static_cast<int>(imgw.rows * r);

  // Make PinholeCamera
  // borrow KinectV1 intrinsics of Freiburg 1 RGB
  // https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
  // float r = 0.5f;  // scale to smaller size from VGA
  float r2 = static_cast<float>(target_width) / 640.f;
  int width = static_cast<int>(640 * r2);
  int height = static_cast<int>(480 * r2);
  Eigen::Vector2f principal_point(318.6f * r2, 255.3f * r2);
  Eigen::Vector2f focal_length(517.3f * r2, 516.5f * r2);
  std::shared_ptr<ugu::PinholeCamera> camera =
      std::make_shared<ugu::PinholeCamera>(width, height,
                                           Eigen::Affine3d::Identity(),
                                           principal_point, focal_length);

  for (int i = 0; i < 6; i++) {
    ugu::Image1w imgw =
        ugu::imread("../data/bunny/0000" + std::to_string(i) + "_depth.png");
    if (imgw.empty()) {
      std::cerr << "Failed to load image" << std::endl;
      return;
    }
    ugu::Image1f imgf;
    imgw.convertTo(imgf, CV_32FC1);

    imgf = ugu::ResizeNearest(imgf, target_width, target_height);

    depths.push_back(imgf);

    ugu::Image3f normal = ugu::Image3f::zeros(imgf.rows, imgf.cols);
    normals.push_back(normal);

    cameras.push_back(camera);
  }

  ugu::Timer timer;
  int n_trials = 100;
  timer.Start();
  for (int i = 0; i < n_trials; i++) {
    ugu::ComputeNormalsCuda(depths, cameras, normals);
  }
  timer.End();
  std::cout << "ComputeNormalsCuda (ugu::Image) : " << timer.elapsed_msec()
            << " / " << timer.elapsed_msec() / n_trials << std::endl;
  for (int i = 0; i < 6; i++) {
    ugu::Image3b vis;
    ugu::Normal2Color(normals[i], &vis, true);
    ugu::imwrite("0000" + std::to_string(i) + "_normal_cudaugu.png", vis);
  }

  const int num_images = static_cast<int>(depths.size());
  // const int width = depths[0].cols;
  // const int height = depths[0].rows;

  std::vector<float> h_depths(num_images * width * height);

  // #pragma omp parallel for
  for (int i = 0; i < num_images; ++i) {
    std::memcpy(h_depths.data() + i * width * height, depths[i].data,
                sizeof(float) * width * height);
  }

  std::vector<float> h_fx(num_images);
  std::vector<float> h_fy(num_images);
  std::vector<float> h_cx(num_images);
  std::vector<float> h_cy(num_images);

  for (int i = 0; i < num_images; ++i) {
    h_fx[i] = cameras[i]->focal_length().x();
    h_fy[i] = cameras[i]->focal_length().y();
    h_cx[i] = cameras[i]->principal_point().x();
    h_cy[i] = cameras[i]->principal_point().y();
  }

  std::vector<float> h_normals(num_images * width * height * 3);
  std::memset(h_normals.data(), 0, sizeof(float) * width * height * 3);
  timer.Start();
  for (int i = 0; i < n_trials; i++) {
    ugu::ComputeNormalsCuda(num_images, width, height, h_depths, h_normals,
                            h_fx, h_fy, h_cx, h_cy);
  }
  timer.End();
  std::cout << "ComputeNormalsCuda (Raw) : " << timer.elapsed_msec() << " / "
            << timer.elapsed_msec() / n_trials << std::endl;
  if (normals.size() != num_images) {
    normals.resize(num_images);
  }
  // #pragma omp parallel for
  for (int i = 0; i < num_images; ++i) {
    if (normals[i].cols != width || normals[i].rows != height) {
      normals[i] = ugu::Image1f::zeros(height, width);
    }
    std::memcpy(normals[i].data, h_normals.data() + i * width * height * 3,
                sizeof(float) * width * height * 3);
  }
  for (int i = 0; i < 6; i++) {
    ugu::Image3b vis;
    ugu::Normal2Color(normals[i], &vis, true);
    ugu::imwrite("0000" + std::to_string(i) + "_normal_cuda.png", vis);
  }

  {
    ugu::NormalComputerCuda normal_computer(width, height, num_images,
                                            h_fx.data(), h_fy.data(),
                                            h_cx.data(), h_cy.data());
    std::memset(h_normals.data(), 0, sizeof(float) * width * height * 3);
    timer.Start();
    for (int i = 0; i < n_trials; i++) {
      normal_computer.ComputeNormals(h_depths.data(), h_normals.data());
    }
    timer.End();
    std::cout << "NormalComputerCuda: " << timer.elapsed_msec() << " / "
              << timer.elapsed_msec() / n_trials << std::endl;
    for (int i = 0; i < num_images; ++i) {
      if (normals[i].cols != width || normals[i].rows != height) {
        normals[i] = ugu::Image1f::zeros(height, width);
      }
      std::memcpy(normals[i].data, h_normals.data() + i * width * height * 3,
                  sizeof(float) * width * height * 3);
    }
    for (int i = 0; i < 6; i++) {
      ugu::Image3b vis;
      ugu::Normal2Color(normals[i], &vis, true);
      ugu::imwrite("0000" + std::to_string(i) + "_normal_cudaclass.png", vis);
    }
  }

#ifdef UGU_USE_CUDA
  {
    ugu::NormalComputerCuda normal_computer(width, height, num_images,
                                            h_fx.data(), h_fy.data(),
                                            h_cx.data(), h_cy.data());

    float *h_depths_pinned, *h_normals_pinned;
    cudaMallocHost(&h_depths_pinned,
                   sizeof(float) * num_images * width * height);
    cudaMallocHost(&h_normals_pinned,
                   sizeof(float) * num_images * width * height * 3);
    std::memcpy(h_depths_pinned, h_depths.data(),
                sizeof(float) * num_images * width * height);
    timer.Start();
    for (int i = 0; i < n_trials; i++) {
      normal_computer.ComputeNormals(h_depths_pinned, h_normals_pinned);
    }
    timer.End();
    std::cout << "NormalComputerCuda (pinned memory): " << timer.elapsed_msec()
              << " / " << timer.elapsed_msec() / n_trials << std::endl;
    for (int i = 0; i < num_images; ++i) {
      if (normals[i].cols != width || normals[i].rows != height) {
        normals[i] = ugu::Image1f::zeros(height, width);
      }
      std::memcpy(normals[i].data, h_normals_pinned + i * width * height * 3,
                  sizeof(float) * width * height * 3);
    }
    for (int i = 0; i < 6; i++) {
      ugu::Image3b vis;
      ugu::Normal2Color(normals[i], &vis, true);
      ugu::imwrite("0000" + std::to_string(i) + "_normal_cudaclass_pinned.png",
                   vis);
    }
    cudaFreeHost(h_depths_pinned);
    cudaFreeHost(h_normals_pinned);
  }
#endif

  timer.Start();
  for (int i = 0; i < n_trials; i++) {
#pragma omp parallel for
    for (int j = 0; j < 6; j++) {
      ugu::ComputeNormal(depths[j], *cameras[j].get(), &normals[j], 1e8f);
    }
  }
  timer.End();
  std::cout << "ComputeNormals: " << timer.elapsed_msec() << " / "
            << timer.elapsed_msec() / n_trials << std::endl;
  for (int i = 0; i < 6; i++) {
    ugu::Image3b vis;
    ugu::Normal2Color(normals[i], &vis, true);
    ugu::imwrite("0000" + std::to_string(i) + "_normal_cpu.png", vis);
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  TestNormal();

  {
    ugu::Image3b img = ugu::imread("../data/color_transfer/reference_00.jpg");
    ugu::Image3b img_org = img.clone();
    ugu::Timer timer;
    int kernel = 51;
    ugu::BoxFilterCuda(img, kernel);
    ugu::imwrite("box_blur_cuda.jpg", img);
    img = img_org.clone();
    timer.Start();
    for (size_t i = 0; i < 1000; i++) {
      ugu::BoxFilterCuda(img, kernel);
    }
    timer.End();
    std::cout << "BoxFilterCuda: " << timer.elapsed_msec() << " / "
              << timer.elapsed_msec() / 1000 << std::endl;

    img = img_org.clone();
    ugu::BoxFilter(img.clone(), &img, kernel);
    ugu::imwrite("box_blur_cpu.jpg", img);
    img = img_org.clone();
    timer.Start();
    for (size_t i = 0; i < 1000; i++) {
      ugu::BoxFilter(img.clone(), &img, kernel);
    }
    timer.End();
    std::cout << "BoxFilter: " << timer.elapsed_msec() << " / "
              << timer.elapsed_msec() / 1000 << std::endl;
  }

  std::string data_dir = "../data/bunny/";
  std::string mask_path = data_dir + "00000_mask.png";

  ugu::Image1b mask = ugu::Imread<ugu::Image1b>(mask_path, -1);

  // 2D SDF
  ugu::Image1f sdf;
  ugu::MakeSignedDistanceField(mask, &sdf, true, false, -1.0f);
  ugu::Image3b vis_sdf;
  ugu::SignedDistance2Color(sdf, &vis_sdf, -1.0f, 1.0f);
  ugu::imwrite(data_dir + "00000_sdf.png", vis_sdf);

  ugu::circle(vis_sdf, {200, 200}, 20, {255, 0, 255}, 3);
  ugu::circle(vis_sdf, {100, 100}, 10, {0, 0, 0}, -1);
  ugu::line(vis_sdf, {0, 0}, {50, 50}, {255, 0, 0}, 1);
  ugu::line(vis_sdf, {10, 200}, {100, 200}, {0, 0, 255}, 5);
  ugu::imwrite(data_dir + "00000_sdf_circle.png", vis_sdf);

  // GIF load
  auto [images, delays] = ugu::LoadGif("../data/gif/dancing.gif");
  for (size_t i = 0; i < images.size(); i++) {
    ugu::imwrite("../data/gif/" + std::to_string(i) + "_" +
                     std::to_string(delays[i]) + "ms.png",
                 images[i]);
  }

  {
    ugu::ImageBase refer =
        ugu::imread("../data/color_transfer/reference_00.jpg");
    ugu::ImageBase target = ugu::imread("../data/color_transfer/target_00.jpg");
    ugu::Image3b res = ugu::ColorTransfer(refer, target);
    ugu::imwrite("../data/color_transfer/result_00.jpg", res);
  }

  {
    ugu::ImageBase source = ugu::imread("../data/poisson_blending/source.png");
    ugu::ImageBase target = ugu::imread("../data/poisson_blending/target.png");
    ugu::ImageBase mask_ = ugu::imread("../data/poisson_blending/mask.png", 0);
    ugu::Image3b res = ugu::PoissonBlend(mask_, source, target, -35, 35);
    ugu::imwrite("../data/poisson_blending/result.png", res);
  }

  return 0;
}
