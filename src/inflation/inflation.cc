/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include "ugu/inflation/inflation.h"

#include <set>
#include <unordered_map>

#include "Eigen/Sparse"
#include "ugu/camera.h"
#include "ugu/image_proc.h"
#include "ugu/util/geom_util.h"
#include "ugu/util/image_util.h"
#include "ugu/util/rgbd_util.h"

namespace {

bool InflationBaran(const ugu::Image1b& mask, ugu::Image1f& height,
                    bool inverse) {
  // '''
  // We have observed that solving the Poisson equation \nabla2h(x, y) = -4,
  // subject to h(\partial\Omega) = 0 and setting z = \sqrt{h} produces very
  // nice results(Figure 1) and is relatively fast and easy to compute.
  // '''

  height = ugu::Image1f::zeros(mask.rows, mask.cols);

  auto get_idx = [&](int x, int y) { return y * mask.rows + x; };

  // Map from image index to parameter index
  std::unordered_map<int, int> img2prm_idx;
  {
    int prm_idx = 0;
    for (int y = 0; y < mask.rows; ++y) {
      for (int x = 0; x < mask.cols; ++x) {
        if (mask.at<unsigned char>(y, x) != 0) {
          img2prm_idx[get_idx(x, y)] = prm_idx;
          ++prm_idx;
        }
      }
    }
  }
  const unsigned int num_param = (unsigned int)img2prm_idx.size();

  std::vector<Eigen::Triplet<double>> triplets;
  {
    int cur_row = 0;
    // 4 neighbor laplacian
    for (int j = 1; j < mask.rows - 1; j++) {
      for (int i = 1; i < mask.cols - 1; i++) {
        if (mask.at<unsigned char>(j, i) != 0) {
          triplets.push_back({cur_row, img2prm_idx[get_idx(i, j)], -4.0});

          // Skip if a neighbor is not on mask (not an unknown parameter to
          // estimate). This means setting 0 for the neighbor as known boundary
          // condition. So that this skipping formualtes h(\partinal\Omega) = 0
          if (mask.at<unsigned char>(j, i - 1) != 0) {
            triplets.push_back({cur_row, img2prm_idx[get_idx(i - 1, j)], 1.0});
          }
          if (mask.at<unsigned char>(j, i + 1) != 0) {
            triplets.push_back({cur_row, img2prm_idx[get_idx(i + 1, j)], 1.0});
          }
          if (mask.at<unsigned char>(j - 1, i) != 0) {
            triplets.push_back({cur_row, img2prm_idx[get_idx(i, j - 1)], 1.0});
          }
          if (mask.at<unsigned char>(j + 1, i) != 0) {
            triplets.push_back({cur_row, img2prm_idx[get_idx(i, j + 1)], 1.0});
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

  constexpr double rhs = -4.0;  //  \nabla2h(x, y) = -4
  {
    int cur_row = 0;
    for (int j = 1; j < mask.rows - 1; j++) {
      for (int i = 1; i < mask.cols - 1; i++) {
        if (mask.at<unsigned char>(j, i) != 0) {
          b[cur_row] = rhs;  // Set condition
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
        auto idx = img2prm_idx[get_idx(i, j)];
        auto& h = height.at<float>(j, i);
        h = static_cast<float>(std::sqrt(x[idx]));  // setting z = \sqrt{h}
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
          auto idx = img2prm_idx[get_idx(i, j)];
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
  auto [boundary_edges_list, boundary_vertex_ids_list] = ugu::FindBoundaryLoops(
      mesh.vertex_indices(), static_cast<int32_t>(mesh.vertices().size()));

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
    auto blend_color = [&](Vec3b& blended_pix,
                           const std::vector<Vec3b>& sources) {
      if (sources.empty()) {
        return;
      }
      for (int c = 0; c < 3; c++) {
        double ave_color = 0.0;
        for (const auto& s : sources) {
          ave_color += s[c];
        }
        ave_color /= sources.size();
        ave_color = std::clamp(ave_color, 0.0, 255.0);
        blended_pix[c] = static_cast<unsigned char>(ave_color);
      }
      return;
    };

    auto inpainted_mat = back.materials()[0];
    inpainted_mat.diffuse_tex = ugu::Image3b::zeros(
        mask.rows, mask.cols);  // Inpainted texture is same size to mask
    auto& inpainted_tex = inpainted_mat.diffuse_tex;
    inpainted_mat.diffuse_texname = inpainted_mat.name + "_inpainted.png";

    ugu::Image3b resized_org_color = *params_.texture;
    if (resized_org_color.rows != mask.rows ||
        resized_org_color.cols != mask.cols) {
      ugu::resize(*params_.texture, resized_org_color,
                  Size(mask.cols, mask.rows));
    }

    int ks = std::max(3, params_.inpaint_kernel_size);
    int hks = ks / 2;
    int erode_kernel_size = hks * 2 + 1;

    // Mask boundary pixels are original ones
    ugu::Image1b eroded = ugu::Image1b::zeros(mask.rows, mask.cols);
    ugu::Erode(mask, &eroded, erode_kernel_size);
    ugu::Image1b org_pix_mask = ugu::Image1b::zeros(mask.rows, mask.cols);
    ugu::Diff(mask, eroded, &org_pix_mask);  // Make diff with eroded to
                                             // identify mask boundary pixels;

    // Copy and blend some boundary pixels from front
    for (int j = hks; j < mask.rows - hks; j++) {
      for (int i = hks; i < mask.cols - hks; i++) {
        if (org_pix_mask.at<unsigned char>(j, i) != 0) {
          // blend from original texture
          std::vector<Vec3b> sources;
          for (int jj = -hks; jj <= hks; jj++) {
            for (int ii = -hks; ii <= hks; ii++) {
              if (ii == 0 && jj == 0) {
                continue;
              }
              if (mask.at<unsigned char>(j + jj, i + ii) != 0) {
                sources.push_back(resized_org_color.at<Vec3b>(j + jj, i + ii));
              }
            }
          }
          auto& inpainted_pix = inpainted_tex.at<Vec3b>(j, i);
          blend_color(inpainted_pix, sources);
        }
      }
    }

    Inpaint(eroded, inpainted_tex,
            static_cast<float>(params_.inpaint_kernel_size),
            params_.inpaint_method);

    back.set_materials({inpainted_mat});
  }

  ugu::MergeMeshes(front, back, &mesh, use_same_material);

  // Generate side faces to make watertight surface
  std::vector<Eigen::Vector3i> side_faces;
  const auto f2b_voffset = front.vertices().size();
  for (const auto& boundary_edges : boundary_edges_list) {
    for (size_t i = 0; i < boundary_edges.size(); i++) {
      const auto& front_edge = boundary_edges[i];
      const auto back_edge = std::make_pair<int, int>(
          static_cast<int>(front_edge.first + f2b_voffset),
          static_cast<int>(front_edge.second + f2b_voffset));
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
