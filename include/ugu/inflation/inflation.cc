/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/inflation/inflation.h"

#include <set>
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

  // Merge front and back meshes
  bool use_same_material =
      (params_.back_texture_type == InflationBackTextureType::MIRRORED);
  // Make inpainted texture for back
  if (with_texture &&
      params_.back_texture_type == InflationBackTextureType::INPAINT) {
    // Mask boundary pixels are original ones
    ugu::Image1b eroded = ugu::Image1b::zeros(mask.rows, mask.cols);
    ugu::Erode(mask, &eroded, 5);
    ugu::Image1b org_pix_mask = ugu::Image1b::zeros(mask.rows, mask.cols);
    ugu::Diff(mask, eroded, &org_pix_mask);  // Make diff with eroded to
                                             // identify mask boundary pixels;

    // ugu::imwrite("hoge.png", org_pix_mask);
    // printf("%d\n", mesh.materials().size());
    auto inpainted_mat = back.materials()[0];
    inpainted_mat.diffuse_tex = ugu::Image3b::zeros(
        mask.rows, mask.cols);  // Inpainted texture is same size to mask
    auto& inpainted_tex = inpainted_mat.diffuse_tex;
    inpainted_mat.diffuse_texname = inpainted_mat.name + "_inpainted.png";

    // auto get_idx = [&](int x, int y) { return y * mask.rows + x; };

    ugu::Image3b resized_org_tex = *params_.texture;
    if (resized_org_tex.rows != mask.rows ||
        resized_org_tex.cols != mask.cols) {
      ugu::resize(*params_.texture, resized_org_tex,
                  Size(mask.cols, mask.rows));
    }

    // Initialize not_visited pixels within inside mask and not on org_pix_mask
    std::set<std::pair<int, int>> not_visited;
    ugu::Image1b not_visited_mask = ugu::Image1b::zeros(mask.rows, mask.cols);
    for (int j = 1; j < mask.rows - 1; j++) {
      for (int i = 1; i < mask.cols - 1; i++) {
        if (mask.at<unsigned char>(j, i) != 0 &&
            org_pix_mask.at<unsigned char>(j, i) == 0) {
          not_visited.insert({i, j});
          not_visited_mask.at<unsigned char>(j, i) = 255;
        }

        if (org_pix_mask.at<unsigned char>(j, i) != 0) {
          // Copy from original texture
          inpainted_tex.at<Vec3b>(j, i) = resized_org_tex.at<Vec3b>(j, i);
        }
      }
    }

    auto inpaint_process = [&](int i, int j) {
      auto& m = not_visited_mask.at<unsigned char>(j, i);
      if (m != 0) {
        // Check at least one inpainted pixel
        std::vector<Vec3b> sources;
        for (int jj = -1; jj <= 1; jj++) {
          for (int ii = -1; ii <= 1; ii++) {
            if (ii == 0 && jj == 0) {
              continue;
            }
            if (not_visited_mask.at<unsigned char>(j + jj, i + ii) == 0) {
              sources.push_back(inpainted_tex.at<Vec3b>(j + jj, i + ii));
            }
          }
        }
        if (sources.empty()) {
          return;
        }

        // Make inpainted color
        auto& inpainted_pix = inpainted_tex.at<Vec3b>(j, i);
        for (int c = 0; c < 3; c++) {
          double ave_color = 0.0;
          for (const auto& s : sources) {
            ave_color += s[c];
          }
          ave_color /= sources.size();
          inpainted_pix[c] = static_cast<unsigned char>(ave_color);
        }

        // Update not_visited flags
        m = 0;
        not_visited.erase({i, j});
      }
      return;
    };

    int iter = 0;
    while (!not_visited.empty()) {
      // Update with two directions alternately to avoid artifact
      if (iter % 2 == 0) {
        for (int j = 1; j < mask.rows - 1; j++) {
          for (int i = 1; i < mask.cols - 1; i++) {
            inpaint_process(i, j);
          }
        }
      } else {
        for (int j = mask.rows - 2; j > 0; j--) {
          for (int i = mask.cols - 2; i > 0; i--) {
            inpaint_process(i, j);
          }
        }
      }
      iter++;
    }

    back.set_materials({inpainted_mat});
  }

  ugu::MergeMeshes(front, back, &mesh, use_same_material);

  // Generate side faces to make watertight surface
  std::vector<Eigen::Vector3i> side_faces;
  const auto f2b_voffset = front.vertices().size();
  for (const auto& boundary_edges : boundary_edges_list) {
    for (auto i = 0; i < boundary_edges.size(); i++) {
      const auto& front_edge = boundary_edges[i];
      const auto back_edge = std::make_pair<int, int>(
          front_edge.first + f2b_voffset, front_edge.second + f2b_voffset);
      Eigen::Vector3i f0 = {back_edge.first, front_edge.first,
                            front_edge.second};
      Eigen::Vector3i f1 = {back_edge.first, front_edge.second,
                            back_edge.second};
      side_faces.emplace_back(f0);
      side_faces.emplace_back(f1);
    }
  }

  // Add side faces to merged mesh
  std::vector<Eigen::Vector3i> vertex_indices_with_side = mesh.vertex_indices();
  std::copy(side_faces.begin(), side_faces.end(),
            std::back_inserter(vertex_indices_with_side));
  mesh.set_vertex_indices(vertex_indices_with_side);
  mesh.set_uv_indices(vertex_indices_with_side);
  mesh.set_normal_indices(vertex_indices_with_side);
  mesh.CalcNormal();

  // Update material_ids for side faces
  std::vector<int> material_ids_with_side = mesh.material_ids();
  std::vector<int> side_material_ids(side_faces.size(),
                                     0);  // side material id is 0, front's one
  std::copy(side_material_ids.begin(), side_material_ids.end(),
            std::back_inserter(material_ids_with_side));
  mesh.set_material_ids(material_ids_with_side);

  return true;
}

}  // namespace ugu
