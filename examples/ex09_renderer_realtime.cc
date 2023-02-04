/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#ifdef UGU_USE_OPENCV

#include <fstream>
#include <iomanip>
#include <memory>
#include <thread>

#include "ugu/common.h"

#ifdef _WIN32
#pragma warning(push, UGU_OPENCV_WARNING_LEVEL)
#endif
#include "opencv2/highgui.hpp"
#ifdef _WIN32
#pragma warning(pop)
#endif

#include "ugu/renderer/rasterizer.h"
#include "ugu/renderer/raytracer.h"
#include "ugu/timer.h"
#include "ugu/util/image_util.h"

// #define USE_RASTERIZER

namespace {

std::string msec2fpsstr(float msec) {
  float fps = 1000.f / msec;
  std::string fps_str;
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(1) << fps;
  fps_str = ss.str();
  return fps_str;
}
}  // namespace

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  std::string data_dir = "../data/bunny/";
  std::string obj_path = data_dir + "bunny.obj";

  std::ifstream ifs(obj_path);
  if (!ifs.is_open()) {
    printf("Please put %s\n", obj_path.c_str());
    return -1;
  }

  ugu::set_log_level(ugu::LogLevel::kNone);

  // load mesh
  std::shared_ptr<ugu::Mesh> mesh = std::make_shared<ugu::Mesh>();
  mesh->LoadObj(obj_path, data_dir);

  // original mesh with z:backward, y:up, x:right, like OpenGL
  // align z:forward, y:down, x:right
  // AlignMesh(mesh);

  // initialize renderer with diffuse texture color and lambertian shading
  ugu::RendererOption option;
  option.diffuse_color = ugu::DiffuseColor::kTexture;
  option.diffuse_shading = ugu::DiffuseShading::kLambertian;
#ifdef USE_RASTERIZER
  std::unique_ptr<ugu::Renderer> renderer =
      std::make_unique<ugu::Rasterizer>(option);
#else
  std::unique_ptr<ugu::Renderer> renderer =
      std::make_unique<ugu::Raytracer>(option);
#endif

  // set mesh
  renderer->set_mesh(mesh);

  // prepare mesh for rendering (e.g. make BVH)
  renderer->PrepareMesh();

  // Make PinholeCamera
  // borrow KinectV1 intrinsics of Freiburg 1 RGB
  // https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
  float r = 0.5f;  // scale to smaller size from VGA
  int width = static_cast<int>(640 * r);
  int height = static_cast<int>(480 * r);
  Eigen::Vector2f principal_point(318.6f * r, 255.3f * r);
  Eigen::Vector2f focal_length(517.3f * r, 516.5f * r);
  std::shared_ptr<ugu::Camera> camera = std::make_shared<ugu::PinholeCamera>(
      width, height, Eigen::Affine3d::Identity(), principal_point,
      focal_length);

  // set camera
  renderer->set_camera(camera);

  ugu::MeshStats stats = mesh->stats();
  Eigen::Vector3f center = stats.center;
  Eigen::Vector3f eye;
  Eigen::Matrix4f c2w_mat;

  // translation offset is the largest edge length of bounding box * 1.5
  Eigen::Vector3f diff = stats.bb_max - stats.bb_min;
  float offset = std::max(diff[0], std::max(diff[1], diff[2])) * 1.5f;

  ugu::Timer<> timer_render, timer_all;
  float angular_velocity_msec = ugu::radians(45.f) / 1000.f;
  float current_angle = 0.f;
  ugu::Image3b color;
  ugu::Image1f depth;
  ugu::Image3f normal;
  ugu::Image1b mask;
  ugu::Image1i face_id;
  ugu::Image3b show_img = ugu::Image3b::zeros(height * 2, width * 2);
  ugu::Image1b vis_depth_1b;
  ugu::Image3b vis_depth;
  ugu::Image3b vis_normal;
  ugu::Image3b vis_face_id;
  unsigned int n_threads = std::thread::hardware_concurrency();
  std::string n_threads_str = std::to_string(n_threads) + " threads";
  std::string image_size_str =
      "(" + std::to_string(width) + ", " + std::to_string(height) + ")";

  while (true) {
    timer_all.Start();
    float elapsed_msec_render = static_cast<float>(timer_render.average_msec());
    std::string render_fps = msec2fpsstr(elapsed_msec_render);
    float elapsed_msec_all = static_cast<float>(timer_all.average_msec());
    std::string all_fps = msec2fpsstr(elapsed_msec_all);

    current_angle += elapsed_msec_all * angular_velocity_msec;
    if (ugu::radians(360.f) < current_angle) {
      current_angle -= ugu::radians(360.f);
    }

    // Rotate on xz plane
    eye = center;
    eye[0] += offset * std::sin(current_angle);
    eye[2] += offset * std::cos(current_angle);
    ugu::c2w(eye, center, Eigen::Vector3f(0, 1, 0), &c2w_mat);
    camera->set_c2w(Eigen::Affine3d(c2w_mat.cast<double>()));
    timer_render.Start();
    renderer->Render(&color, &depth, &normal, &mask, &face_id);
    timer_render.End();

    cv::putText(color, "shaded color", {10, 15}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255));
    color.copyTo(show_img(cv::Rect(0, 0, width, height)));

    ugu::Depth2Gray(depth, &vis_depth_1b);
    cv::cvtColor(vis_depth_1b, vis_depth, cv::COLOR_GRAY2BGR);
    cv::putText(vis_depth, "depth", {10, 15}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255));
    vis_depth.copyTo(show_img(cv::Rect(width, 0, width, height)));

    ugu::Normal2Color(normal, &vis_normal);
    cv::putText(vis_normal, "normal in camera coord.", {10, 15},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255));
    vis_normal.copyTo(show_img(cv::Rect(0, height, width, height)));

    ugu::FaceId2Color(face_id, &vis_face_id, 0,
                      static_cast<int>(mesh->vertex_indices().size()));
    cv::putText(vis_face_id, "face id", {10, 15}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255));
    vis_face_id.copyTo(show_img(cv::Rect(width, height, width, height)));

    cv::putText(show_img, image_size_str + "  " + n_threads_str, {10, 40},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
    cv::putText(show_img, "rendering: " + render_fps + " fps", {10, 60},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
    cv::putText(show_img, "total    : " + all_fps + " fps", {10, 80},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));

    // cv::imshow and cv::waitKey take more than 10ms on Windows10 with
    // OpenCV 4.5.1. So, total fps may become much slower than rendering only
    // fps.
    cv::imshow("cpu rendering", show_img);
    cv::waitKey(1);

    timer_all.End();
  }

  return 0;
}
#else
int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;
  return 0;
}
#endif
