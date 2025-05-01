/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include <random>

#include "ugu/clustering/clustering.h"
#include "ugu/cuda/image.h"
#include "ugu/cuda/voxel.h"
#include "ugu/image_io.h"
#include "ugu/image_proc.h"
#include "ugu/plane.h"
#include "ugu/renderer/cpu/rasterizer.h"
#include "ugu/renderer/cpu/raytracer.h"
#include "ugu/timer.h"
#include "ugu/util/geom_util.h"
#include "ugu/util/image_util.h"
#include "ugu/util/rgbd_util.h"
#include "ugu/voxel/extract_voxel.h"
#include "ugu/voxel/marching_cubes.h"
#include "ugu/voxel/voxel.h"

namespace {

void AddDepthNoise(ugu::Image1f& depth, float mu, float sigma, float lack_ratio,
                   uint32_t seed = 0) {
  std::default_random_engine engine(seed);
  std::normal_distribution<float> noise_dist(mu, sigma);
  std::uniform_real_distribution<float> lack_dist;

  depth.forEach([&](float& val, const int* pos) {
    (void)pos;
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

  ugu::Timer timer;
  auto object = ugu::Mesh::Create();
  object->LoadObj(data_dir + "/bunny/bunny.obj", data_dir + "bunny/");

  const float bb_len_max =
      (object->stats().bb_max - object->stats().bb_min).maxCoeff();
  const float plane_len = bb_len_max * 1.3f;
  ugu::Image3b plane_texture =
      ugu::Imread<ugu::Image3b>("../data/inpaint/fruits.jpg");
  auto plane = ugu::MakeTexturedPlane(plane_texture, plane_len);
  plane->Rotate(
      Eigen::AngleAxisf(ugu::radians(-90.f), Eigen::Vector3f(1.f, 0.f, 0.f))
          .matrix());
  plane->Translate({0.f, object->stats().bb_min.y(), 0.f});

  auto combined = ugu::Mesh::Create();
  ugu::MergeMeshes({object, plane}, combined.get());

  combined->WriteObj(data_dir, "object_and_plane");

  size_t view_num = 12;
  ugu::RendererCpuOption renderer_option;
  renderer_option.diffuse_color = ugu::DiffuseColor::kTexture;
  renderer_option.backface_culling = false;
  // TODO rasterizer is buggy
  // ugu::RendererPtr renderer =
  //    std::make_shared<ugu::Rasterizer>(renderer_option);
  ugu::RendererCpuPtr renderer =
      std::make_shared<ugu::Raytracer>(renderer_option);

  constexpr float fov_y_deg = 20.f;

  renderer->set_mesh(combined);
  renderer->PrepareMesh();

  const float cam_radius = bb_len_max * 3;
  const float cam_height = bb_len_max * 2;
  const Eigen::Vector3f up{0, 1.f, 0.f};

  std::vector<ugu::Image1f> depths;
  std::vector<ugu::Image3b> colors;
  std::vector<ugu::PinholeCameraPtr> cameras;
  std::vector<ugu::MeshPtr> depth_meshes;
  ugu::MeshPtr depth_merged = ugu::Mesh::Create();
  ugu::MeshPtr depth_fused = ugu::Mesh::Create();
  const float mu = 0.f;
  const float sigma = bb_len_max * 0.02f;

  for (size_t i = 0; i < view_num; i++) {
    Eigen::Matrix3f R;
    Eigen::Vector3f pos;
    ugu::Image1f depth;
    ugu::Image3b color;
    const float rad = static_cast<float>(i) / static_cast<float>(view_num) *
                      static_cast<float>(ugu::pi * 2);
    pos.x() = cam_radius * std::cos(rad);
    pos.y() = cam_height;
    pos.z() = cam_radius * std::sin(rad);
    ugu::c2w(pos, object->stats().center, up, &R);

    Eigen::Affine3d c2w = (Eigen::Translation3f(pos) * R).cast<double>();
    ugu::PinholeCameraPtr camera =
        std::make_shared<ugu::PinholeCamera>(640, 480, fov_y_deg);
    renderer->set_camera(camera);
    camera->set_c2w(c2w);

    renderer->Render(&color, &depth, nullptr, nullptr, nullptr);

    AddDepthNoise(depth, mu, sigma, 0.02f, static_cast<uint32_t>(i));

    depths.push_back(depth.clone());
    colors.push_back(color.clone());
    cameras.push_back(camera);
  }

#ifdef UGU_USE_CUDA
  {
    ugu::Timer timer;

    ugu::NormalComputerCuda normal_computer;
    std::vector<float> h_fx_vec, h_fy_vec, h_cx_vec, h_cy_vec, h_R_vec, h_t_vec;

    for (size_t i = 0; i < cameras.size(); i++) {
      h_fx_vec.push_back(cameras[i]->focal_length().x());
      h_fy_vec.push_back(cameras[i]->focal_length().y());
      h_cx_vec.push_back(cameras[i]->principal_point().x());
      h_cy_vec.push_back(cameras[i]->principal_point().y());

      Eigen::Matrix3f R = cameras[i]->c2w().rotation().cast<float>();
      Eigen::Vector3f t = cameras[i]->c2w().translation().cast<float>();
      // Eigen is col-major on memory as default
      // Use accesor to assume row-major
      for (int k = 0; k < 3; k++) {
        for (int j = 0; j < 3; j++) {
          h_R_vec.push_back(R(k, j));
        }
      }
      for (int j = 0; j < 3; j++) {
        h_t_vec.push_back(t.data()[j]);
      }
    }

    normal_computer.Init(depths[0].cols, depths[0].rows, depths.size(),
                         h_fx_vec.data(), h_fy_vec.data(), h_cx_vec.data(),
                         h_cy_vec.data(), 1e6f, 1, false, h_R_vec.data(),
                         h_t_vec.data());
    const int num_images = static_cast<int>(depths.size());
    const int width = depths[0].cols;
    const int height = depths[0].rows;
    std::vector<float> h_depths(num_images * width * height);

    for (int i = 0; i < num_images; ++i) {
      std::memcpy(h_depths.data() + i * width * height, depths[i].data,
                  sizeof(float) * width * height);
    }

    float *h_depths_pinned, *h_normals_pinned, *h_points_pinned;
    cudaMallocHost(&h_depths_pinned, sizeof(float) * depths[0].cols *
                                         depths[0].rows * depths.size());
    cudaMallocHost(&h_normals_pinned, sizeof(float) * depths[0].cols * 3 *
                                          depths[0].rows * depths.size());
    cudaMallocHost(&h_points_pinned, sizeof(float) * depths[0].cols * 3 *
                                         depths[0].rows * depths.size());
    std::memcpy(h_depths_pinned, h_depths.data(),
                sizeof(float) * num_images * width * height);
    timer.Start();
    normal_computer.ComputeNormals(h_depths_pinned);
    timer.End();
    std::cout << "ComputeNormals  " << timer.elapsed_msec() << " ms"
              << std::endl;
    timer.Start();
    normal_computer.GetNormalsCpu(h_normals_pinned);
    timer.End();
    std::cout << "GetNormalsCpu  " << timer.elapsed_msec() << " ms"
              << std::endl;
    timer.Start();
    normal_computer.GetPointsCpu(h_points_pinned);
    timer.End();
    std::cout << "GetPointsCpu  " << timer.elapsed_msec() << " ms" << std::endl;

    {
      std::vector<ugu::Image3f> normals(depths.size());
      std::vector<ugu::Image3f> points(depths.size());
      for (int i = 0; i < num_images; ++i) {
        if (normals[i].cols != width || normals[i].rows != height) {
          normals[i] = ugu::Image3f::zeros(height, width);
        }
        std::memcpy(normals[i].data, h_normals_pinned + i * width * height * 3,
                    sizeof(float) * width * height * 3);
      }
      for (int i = 0; i < num_images; i++) {
        ugu::Image3b vis;
        ugu::Normal2Color(normals[i], &vis, true);
        ugu::imwrite(
            "0000" + std::to_string(i) + "_normal_cudaclass_pinned_fuse.png",
            vis);
      }
      for (int i = 0; i < num_images; ++i) {
        if (points[i].cols != width || points[i].rows != height) {
          points[i] = ugu::Image3f::zeros(height, width);
        }
        std::memcpy(points[i].data, h_points_pinned + i * width * height * 3,
                    sizeof(float) * width * height * 3);
      }
      for (int i = 0; i < num_images; i++) {
        ugu::Image3b vis = ugu::ColorizePosMap(points[i]);
        ugu::imwrite(
            "0000" + std::to_string(i) + "_points_cudaclass_pinned_fuse.png",
            vis);
      }
    }

    ugu::VoxelGridCudaNaive voxel_grid_naive;
    Eigen::Vector3f resolution(10.f, 10.f, 10.f);
    Eigen::Vector3f offset = resolution * 2;
    ugu::VoxelUpdateOption option =
        ugu::GenFuseDepthDefaultOption(resolution.minCoeff());

    voxel_grid_naive.Init(combined->stats().bb_max + offset,
                          combined->stats().bb_min - offset, resolution);

    ugu::VoxelGridCudaNaiveFuseOption fusion_option;
    fusion_option.set_default_truncation_band_from_resolution(
        resolution.minCoeff());
    fusion_option.sample_num = 3;
    fusion_option.nn_range = 1;
    timer.Start();
    voxel_grid_naive.FusePointCloudMulti(
        normal_computer.GetPointsGpu(), normal_computer.GetNormalsGpu(), width,
        height, num_images, fusion_option, true);
    timer.End();
    std::cout << "FusePointCloudMulti  " << timer.elapsed_msec() << " ms"
              << std::endl;

    ugu::Mesh out_mesh;
    std::vector<Eigen::Vector3f> vertices;
    std::vector<Eigen::Vector3i> faces;
    timer.Start();
    voxel_grid_naive.ExtractMesh();
    timer.End();
    std::cout << "ExtractMesh  " << timer.elapsed_msec() << " ms" << std::endl;
    timer.Start();
    voxel_grid_naive.GetExtractMeshCpu(vertices, faces);
    timer.End();
    std::cout << "GetExtractMeshCpu  " << timer.elapsed_msec() << " ms"
              << std::endl;
    out_mesh.set_vertices(vertices);
    out_mesh.set_vertex_indices(faces);
    out_mesh.set_default_material();
    out_mesh.CalcNormal();
    out_mesh.WriteObj("cuda_mc.obj");

    ugu::VoxelGrid voxel_grid_cpu;
    voxel_grid_cpu.Init(combined->stats().bb_max + offset,
                        combined->stats().bb_min - offset, resolution);
    voxel_grid_naive.GetVoxelGridCpu(voxel_grid_cpu);
    ugu::MarchingCubes(voxel_grid_cpu, &out_mesh);
    out_mesh.set_default_material();
    out_mesh.CalcNormal();
    out_mesh.WriteObj("cpu_mc.obj");

    {
      ugu::VoxelGridCudaNaive voxel_grid_naive2;
      voxel_grid_naive2.Init(combined->stats().bb_max + offset,
                             combined->stats().bb_min - offset, resolution);

      timer.Start();
      voxel_grid_naive2.FuseDepthMulti(
          h_depths_pinned, width, height, num_images, h_fx_vec.data(),
          h_fy_vec.data(), h_cx_vec.data(), h_cy_vec.data(), h_R_vec.data(),
          h_t_vec.data(), fusion_option, true);
      timer.End();
      std::cout << "FuseDepthMulti  " << timer.elapsed_msec() << " ms"
                << std::endl;

      ugu::Mesh out_mesh;
      std::vector<Eigen::Vector3f> vertices;
      std::vector<Eigen::Vector3i> faces;
      timer.Start();
      voxel_grid_naive2.ExtractMesh();
      timer.End();
      std::cout << "ExtractMesh  " << timer.elapsed_msec() << " ms"
                << std::endl;
      timer.Start();
      voxel_grid_naive2.GetExtractMeshCpu(vertices, faces);
      timer.End();
      std::cout << "GetExtractMeshCpu  " << timer.elapsed_msec() << " ms"
                << std::endl;
      out_mesh.set_vertices(vertices);
      out_mesh.set_vertex_indices(faces);
      out_mesh.set_default_material();
      out_mesh.CalcNormal();
      out_mesh.WriteObj("cuda_mc2.obj");

      ugu::VoxelGrid voxel_grid_cpu;
      voxel_grid_cpu.Init(combined->stats().bb_max + offset,
                          combined->stats().bb_min - offset, resolution);
      voxel_grid_naive2.GetVoxelGridCpu(voxel_grid_cpu);
      ugu::MarchingCubes(voxel_grid_cpu, &out_mesh);
      out_mesh.set_default_material();
      out_mesh.CalcNormal();
      out_mesh.WriteObj("cpu_mc2.obj");
    }
  }
#endif

  {
    ugu::VoxelGrid voxel_grid;
    Eigen::Vector3f resolution(10.f, 10.f, 10.f);
    Eigen::Vector3f offset = resolution * 2;
    voxel_grid.Init(combined->stats().bb_max + offset,
                    combined->stats().bb_min - offset, resolution);
    ugu::VoxelUpdateOption option =
        ugu::GenFuseDepthDefaultOption(resolution.minCoeff());
    for (size_t i = 0; i < view_num; i++) {
#if 0
      ugu::Mesh pc;
      ugu::Depth2PointCloud(depths[i], *cameras[i], &pc);
      pc.Transform(cameras[i]->c2w().cast<float>());
      pc.WritePly(data_dir + "pc" + std::to_string(i) + ".ply");
#endif

      ugu::FuseDepth(*cameras[i], depths[i], option, voxel_grid);
    }

    ugu::MarchingCubes(voxel_grid, depth_fused.get());
    depth_fused->set_default_material();
    depth_fused->CalcNormal();
    depth_fused->WriteObj(data_dir, "depthfuse");
  }

#if 0
  // Merge naive
  {
    for (size_t i = 0; i < view_num; i++) {
      ugu::MeshPtr depth_mesh = ugu::Mesh::Create();
      ugu::Depth2Mesh(depths[i], colors[i], *cameras[i], depth_mesh.get(),
                      100.f, 1, 1, false, "Depth2Mesh_mat", true);

      depth_mesh->Transform(cameras[i]->c2w().rotation().cast<float>(),
                            cameras[i]->c2w().translation().cast<float>());

      depth_meshes.push_back(depth_mesh);

      depth_mesh->WriteObj(data_dir, "depthmesh" + std::to_string(i));
    }
    ugu::MergeMeshes(depth_meshes, depth_merged.get());
    depth_merged->WriteObj(data_dir, "depthmesh");
  }
#endif

  ugu::EstimateGroundPlaneRansacParam param;
  std::vector<ugu::PlaneEstimationResult> candidates;
  param.inliner_dist_th = 10.f;
  param.max_iter = 1000;
  param.candidates_num = 10;
  // param.use_normal_hint = true;
  // param.normal_hint = Eigen::Vector3f(0.f, 1.f, 0.f);

  timer.Start();
  ugu::EstimateGroundPlaneRansac(depth_fused->vertices(),
                                 depth_fused->normals(), param, candidates);
  timer.End();
  ugu::LOGI("EstimateGroundPlaneRansac %f ms\n", timer.elapsed_msec());
  for (const auto& c : candidates) {
    ugu::LOGI("Ground plane candidate (%f %f,%f) %f\n", c.estimation.n.x(),
              c.estimation.n.y(), c.estimation.n.z(), c.estimation.d);
    ugu::LOGI("Refined by least squares (%f %f,%f) %f\n",
              c.refined_least_squares.n.x(), c.refined_least_squares.n.y(),
              c.refined_least_squares.n.z(), c.refined_least_squares.d);
    ugu::LOGI("inlier_ratio %f, upper_ratio %f, area %f area_ratio %f\n\n",
              c.stat.inlier_ratio, c.stat.upper_ratio, c.stat.area,
              c.stat.area_ratio);
  }

  {
    ugu::Mesh tmp = *depth_fused;
    ugu::RemoveSmallConnectedComponents(tmp, 100, true);

    std::vector<size_t> plane_vids;
    std::vector<size_t> others_vids;
    std::vector<size_t> boundary_vids;
    ugu::Mesh plane_mesh, others_mesh;
    ugu::DisconnectPlaneAndOthers(tmp, candidates[0].estimation,
                                  param.inliner_dist_th * 2, plane_vids,
                                  others_vids, boundary_vids, plane_mesh,
                                  others_mesh, param.inlier_angle_th * 2, true);

    plane_mesh.WriteObj(data_dir, "depthfuse_plane");
    others_mesh.WriteObj(data_dir, "depthfuse_others");

#ifdef UGU_USE_LIBIGL
    {
      ugu::Mesh merged;
      ugu::MergeMeshes(others_mesh, plane_mesh, &merged);

      ugu::LibiglLscm(merged, 512, 512);

      auto mats = merged.materials();
      // mats[0].diffuse_tex

      merged.set_materials(object->materials());

      merged.WriteObj(data_dir, "merged_plane_disconnect");
      auto uv_img = ugu::DrawUv(merged.uv(), merged.uv_indices(),
                                {255, 255, 255}, {0, 0, 0});
      ugu::imwrite(data_dir + "merged_plane_disconnect.jpg", uv_img);
    }
#endif
  }

  std::vector<bool> keep_vertices(depth_fused->vertices().size(), false);
  for (size_t i = 0; i < candidates[0].stat.outliers.size(); i++) {
    keep_vertices[candidates[0].stat.outliers[i]] = true;
  }

  depth_fused->RemoveVertices(keep_vertices);

  ugu::RemoveSmallConnectedComponents(*depth_fused, 100, true);

  depth_fused->WriteObj(data_dir, "depthmesh_remove_plane");

  return 0;
}
