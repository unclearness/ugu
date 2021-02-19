/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/inflation/inflation.h"

#include <unordered_map>

#include "Eigen/Sparse"
#include "ugu/camera.h"
#include "ugu/util.h"

namespace {

bool InflationBaran(const ugu::Image1b& mask, ugu::Image1f& height,
                    bool inverse) {
  constexpr double f = -4.0;  // from reference

  height = ugu::Image1f::zeros(mask.rows, mask.cols);

  auto get_idx = [&](int x, int y) { return y * mask.rows + x; };

  // Map from image index to parameter index
  std::unordered_map<int, int> idx_map;
  {
    int idx = 0;
    for (int y = 0; y < mask.rows; ++y) {
      for (int x = 0; x < mask.cols; ++x) {
        if (mask.at<unsigned char>(y, x) != 0) {
          idx_map[get_idx(x, y)] = idx;
          ++idx;
        }
      }
    }
  }
  const unsigned int num_param = (unsigned int)idx_map.size();

  std::vector<Eigen::Triplet<double>> triplets;
  {
    int cur_row = 0;
    // 4 neighbor laplacian
    for (int j = 1; j < mask.rows - 1; j++) {
      for (int i = 1; i < mask.cols - 1; i++) {
        if (mask.at<unsigned char>(j, i) != 0) {
          triplets.push_back({cur_row, idx_map[get_idx(i, j)], -4.0});
          if (mask.at<unsigned char>(j, i - 1) != 0) {
            triplets.push_back({cur_row, idx_map[get_idx(i - 1, j)], 1.0});
          }
          if (mask.at<unsigned char>(j, i + 1) != 0) {
            triplets.push_back({cur_row, idx_map[get_idx(i + 1, j)], 1.0});
          }
          if (mask.at<unsigned char>(j - 1, i) != 0) {
            triplets.push_back({cur_row, idx_map[get_idx(i, j - 1)], 1.0});
          }
          if (mask.at<unsigned char>(j + 1, i) != 0) {
            triplets.push_back({cur_row, idx_map[get_idx(i, j + 1)], 1.0});
          }
          cur_row++;  // Go to the next equation
        }
      }
    }
  }

  Eigen::SparseMatrix<double> A(num_param, num_param);
  A.setFromTriplets(triplets.begin(), triplets.end());
  // TODO: Is this solver the best?
  Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> solver;
  // Prepare linear system
  solver.compute(A);

  Eigen::VectorXd b(num_param);
  b.setZero();
  {
    unsigned int cur_row = 0;
    for (int j = 1; j < mask.rows - 1; j++) {
      for (int i = 1; i < mask.cols - 1; i++) {
        if (mask.at<unsigned char>(j, i) != 0) {
          b[cur_row] = f;  // Set condition

          cur_row++;
        }
      }
    }
  }

  // Solve linear system
  Eigen::VectorXd x = solver.solve(b);

  float max_h = -1.0f;
  for (int j = 1; j < mask.rows - 1; j++) {
    for (int i = 1; i < mask.cols - 1; i++) {
      if (mask.at<unsigned char>(j, i) != 0) {
        auto idx = idx_map[get_idx(i, j)];
        auto& h = height.at<float>(j, i);
        h = static_cast<float>(std::sqrt(x[idx]));
        if (h > max_h) {
          max_h = h;
        }
      }
    }
  }

  // Inverse if specified
  if (inverse) {
    max_h += 0.00001f;
    for (int j = 1; j < mask.rows - 1; j++) {
      for (int i = 1; i < mask.cols - 1; i++) {
        if (mask.at<unsigned char>(j, i) != 0) {
          auto idx = idx_map[get_idx(i, j)];
          auto& h = height.at<float>(j, i);
          h = max_h - static_cast<float>(std::sqrt(x[idx]));
        }
      }
    }
  }

  return true;
}

}  // namespace

namespace ugu {

bool Inflation(const Image1b& mask, Image1f& height,
               const InflationParams& params) {
  if (params.method == InflationMethod::BARAN) {
    return InflationBaran(mask, height, params.inverse);
  }

  return false;
}

bool Inflation(const Image1b& mask, Image1f& height, Mesh& mesh,
               const InflationParams& params) {
  InflationParams params_ = params;
  params_.inverse = true;

  bool ret = Inflation(mask, height, params_);
  if (!ret) {
    return false;
  }

  // Make mesh with ortho camera
  std::shared_ptr<ugu::OrthoCamera> camera =
      std::make_shared<ugu::OrthoCamera>();
  camera->set_size(mask.cols, mask.rows);
  camera->set_c2w(Eigen::Affine3d::Identity());

  bool with_texture = params_.texture != nullptr;

  mesh.Clear();
  if (with_texture) {
    ugu::Depth2Mesh(height, *params_.texture, *camera, &mesh, 999999.9f);
  } else {
    ugu::Depth2Mesh(height, *camera, &mesh, 999999.9f);
  }

  mesh.CalcStats();
  MeshStats stats = mesh.stats();

  if (params_.centering) {
    mesh.Translate(-stats.center);
  }

  if (!params_.generate_back) {
    return true;
  }

  // Find boundary loops to connect
  auto [boundary_edges_list, boundary_vertex_ids_list] =
      ugu::FindBoundaryLoops(mesh);

  ugu::Mesh front = ugu::Mesh(mesh);
  ugu::Mesh back = ugu::Mesh(mesh);

  // Invert back vertex position
  stats = mesh.stats();
  auto back_vertices = back.vertices();
  auto z_diff = stats.bb_max.z() - stats.bb_min.z();
  auto inv_z_diff = 1.0f / z_diff;
  for (auto& v : back_vertices) {
    auto nz = (v.z() - stats.bb_min.z()) * inv_z_diff;
    v.z() = (1.0f - nz) * z_diff + stats.bb_max.z();
  }
  back.set_vertices(back_vertices);

  // Flip faces of the back mesh
  back.FlipFaces();

  bool use_same_material =
      (params_.back_texture == InflationBackTexture::MIRRORED);
  ugu::MergeMeshes(front, back, &mesh, use_same_material);

  return true;
}

}  // namespace ugu
