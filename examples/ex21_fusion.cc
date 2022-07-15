/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include <random>

#include "ugu/plane.h"
#include "ugu/renderer/rasterizer.h"
#include "ugu/renderer/raytracer.h"
#include "ugu/timer.h"
#include "ugu/util/geom_util.h"
#include "ugu/util/rgbd_util.h"

namespace {

void AddDepthNoise(ugu::Image1f& depth, float mu, float sigma, float lack_ratio,
                   size_t seed = 0) {
  std::default_random_engine engine(seed);
  std::normal_distribution<float> noise_dist(mu, sigma);
  std::uniform_real_distribution<float> lack_dist;

  depth.forEach<float>([&](float& val, const int* pos) {
    if (val < std::numeric_limits<float>::epsilon()) {
      return;
    }

    if (lack_dist(engine) < lack_ratio) {
      val = 0.f;
      return;
    }

    val += noise_dist(engine);
  });
}
}  // namespace

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  std::string data_dir = "../data/";

  auto object = ugu::Mesh::Create();
  object->LoadObj(data_dir + "/bunny/bunny.obj", data_dir + "bunny/");

  const float bb_len_max =
      (object->stats().bb_max - object->stats().bb_min).maxCoeff();
  const float plane_len = bb_len_max * 3;
  ugu::Image3b plane_texture =
      ugu::imread<ugu::Image3b>("../data/inpaint/fruits.jpg");
  auto plane = ugu::MakeTexturedPlane(plane_texture, plane_len);
  plane->Rotate(
      Eigen::AngleAxisf(ugu::radians(-90.f), Eigen::Vector3f(1.f, 0.f, 0.f))
          .matrix());
  plane->Translate({0.f, object->stats().bb_min.y(), 0.f});

  auto combined = ugu::Mesh::Create();
  ugu::MergeMeshes({object, plane}, combined.get());

  combined->WriteObj(data_dir, "object_and_plane");

  size_t view_num = 12;
  ugu::RendererOption renderer_option;
  renderer_option.diffuse_color = ugu::DiffuseColor::kTexture;
  renderer_option.backface_culling = false;
  // TODO rasterizer is buggy
  // ugu::RendererPtr renderer =
  //    std::make_shared<ugu::Rasterizer>(renderer_option);
  ugu::RendererPtr renderer = std::make_shared<ugu::Raytracer>(renderer_option);

  constexpr float fov_y_deg = 30.f;
  ugu::CameraPtr camera =
      std::make_shared<ugu::PinholeCamera>(160, 120, fov_y_deg);

  renderer->set_camera(camera);
  renderer->set_mesh(combined);
  renderer->PrepareMesh();

  const float cam_radius = bb_len_max * 3;
  const float cam_height = bb_len_max * 2;
  const Eigen::Vector3f up{0, 1.f, 0.f};
  ugu::Image1f depth;
  ugu::Image3b color;
  std::vector<ugu::MeshPtr> depth_meshes;
  ugu::MeshPtr depth_merged = ugu::Mesh::Create();
  const float mu = 0.f;
  const float sigma = bb_len_max * 0.01f;
  for (size_t i = 0; i < view_num; i++) {
    Eigen::Matrix3f R;
    Eigen::Vector3f pos;
    const float rad =
        static_cast<float>(i) / static_cast<float>(view_num) * ugu::pi * 2;
    pos.x() = cam_radius * std::cos(rad);
    pos.y() = cam_height;
    pos.z() = cam_radius * std::sin(rad);
    ugu::c2w(pos, object->stats().center, up, &R);

    Eigen::Affine3d c2w = (Eigen::Translation3f(pos) * R).cast<double>();
    camera->set_c2w(c2w);

    renderer->Render(&color, &depth, nullptr, nullptr, nullptr);

    AddDepthNoise(depth, mu, sigma, 0.01f, i);

    ugu::MeshPtr depth_mesh = ugu::Mesh::Create();
    ugu::Depth2Mesh(depth, color, *camera, depth_mesh.get(), 100.f, 1, 1, false,
                    "Depth2Mesh_mat", true);

    depth_mesh->Transform(R, pos);

    depth_meshes.push_back(depth_mesh);

    depth_mesh->WriteObj(data_dir, "depthmesh" + std::to_string(i));
  }

  ugu::MergeMeshes(depth_meshes, depth_merged.get());
  depth_merged->WriteObj(data_dir, "depthmesh");

  ugu::EstimateGroundPlaneRansacParam param;
  std::vector<ugu::PlaneEstimationResult> candidates;
  param.inliner_dist_th = 10.f;
  ugu::Timer timer;
  timer.Start();
  ugu::EstimateGroundPlaneRansac(depth_merged->vertices(),
                                 depth_merged->normals(), param, candidates);
  timer.End();
  ugu::LOGI("EstimateGroundPlaneRansac %f ms\n", timer.elapsed_msec());
  for (const auto& c : candidates) {
    ugu::LOGI("Ground plane candidate (%f %f,%f) %f\n", c.estimation.n.x(),
              c.estimation.n.y(), c.estimation.n.z(), c.estimation.d);
    ugu::LOGI("inlier_ratio %f, upper_ratio %f, area %f\n\n",
              c.stat.inlier_ratio, c.stat.upper_ratio, c.stat.area);
  }

  std::vector<bool> keep_vertices(depth_merged->vertices().size(), false);
  for (size_t i = 0; i < candidates[0].stat.outliers.size(); i++) {
    keep_vertices[candidates[0].stat.outliers[i]] = true;
  }

  depth_merged->RemoveVertices(keep_vertices);
  depth_merged->WriteObj(data_dir, "depthmesh_remove_plane");

  return 0;
}
