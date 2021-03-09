/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "ugu/util.h"

#include <fstream>

namespace {

bool Depth2PointCloudImpl(const ugu::Image1f& depth, const ugu::Image3b& color,
                          const ugu::Camera& camera, ugu::Mesh* point_cloud,
                          bool with_texture, bool gl_coord) {
  if (depth.cols != camera.width() || depth.rows != camera.height()) {
    ugu::LOGE(
        "Depth2PointCloud depth size (%d, %d) and camera size (%d, %d) are "
        "different\n",
        depth.cols, depth.rows, camera.width(), camera.height());
    return false;
  }

  if (with_texture) {
    float depth_aspect_ratio =
        static_cast<float>(depth.cols) / static_cast<float>(depth.rows);
    float color_aspect_ratio =
        static_cast<float>(color.cols) / static_cast<float>(color.rows);
    const float aspect_ratio_diff_th{0.01f};  // 1%
    const float aspect_ratio_diff =
        std::abs(depth_aspect_ratio - color_aspect_ratio);
    if (aspect_ratio_diff > aspect_ratio_diff_th) {
      ugu::LOGE(
          "Depth2PointCloud depth aspect ratio %f and color aspect ratio %f "
          "are very "
          "different\n",
          depth_aspect_ratio, color_aspect_ratio);
      return false;
    }
  }

  point_cloud->Clear();

  std::vector<Eigen::Vector3f> vertices;
  std::vector<Eigen::Vector3f> vertex_colors;

  for (int y = 0; y < camera.height(); y++) {
    for (int x = 0; x < camera.width(); x++) {
      const float& d = depth.at<float>(y, x);
      if (d < std::numeric_limits<float>::min()) {
        continue;
      }
      Eigen::Vector3f image_p(static_cast<float>(x), static_cast<float>(y), d);
      Eigen::Vector3f camera_p;
      camera.Unproject(image_p, &camera_p);
      if (gl_coord) {
        // flip y and z to align with OpenGL coordinate
        camera_p.y() = -camera_p.y();
        camera_p.z() = -camera_p.z();
      }
      vertices.push_back(camera_p);

      if (with_texture) {
        // +0.5f comes from mapping 0~1 to -0.5~width(or height)+0.5
        // since uv 0 and 1 is pixel boundary at ends while pixel position is
        // the center of pixel
        Eigen::Vector2f uv(
            static_cast<float>(x + 0.5f) / static_cast<float>(depth.cols),
            static_cast<float>(y + 0.5f) / static_cast<float>(depth.rows));

        // nearest neighbor
        // todo: bilinear
        Eigen::Vector2i pixel_pos(
            static_cast<int>(std::round(uv.x() * color.cols)),
            static_cast<int>(std::round(uv.y() * color.rows)));

        Eigen::Vector3f pixel_color;
        const ugu::Vec3b& tmp_color =
            color.at<ugu::Vec3b>(pixel_pos.y(), pixel_pos.x());
        pixel_color.x() = tmp_color[0];
        pixel_color.y() = tmp_color[1];
        pixel_color.z() = tmp_color[2];

        vertex_colors.push_back(pixel_color);
      }
    }
  }

  // todo: add normal

  point_cloud->set_vertices(vertices);
  if (with_texture) {
    point_cloud->set_vertex_colors(vertex_colors);
  }

  return true;
}

bool Depth2MeshImpl(const ugu::Image1f& depth, const ugu::Image3b& color,
                    const ugu::Camera& camera, ugu::Mesh* mesh,
                    bool with_texture, bool with_vertex_color,
                    float max_connect_z_diff, int x_step, int y_step,
                    bool gl_coord, const std::string& material_name,
                    ugu::Image3f* point_cloud, ugu::Image3f* normal) {
  if (max_connect_z_diff < 0) {
    ugu::LOGE("Depth2Mesh max_connect_z_diff must be positive %f\n",
              max_connect_z_diff);
    return false;
  }
  if (x_step < 1) {
    ugu::LOGE("Depth2Mesh x_step must be positive %d\n", x_step);
    return false;
  }
  if (y_step < 1) {
    ugu::LOGE("Depth2Mesh y_step must be positive %d\n", y_step);
    return false;
  }
  if (depth.cols != camera.width() || depth.rows != camera.height()) {
    ugu::LOGE(
        "Depth2Mesh depth size (%d, %d) and camera size (%d, %d) are "
        "different\n",
        depth.cols, depth.rows, camera.width(), camera.height());
    return false;
  }

  if (with_texture) {
    float depth_aspect_ratio =
        static_cast<float>(depth.cols) / static_cast<float>(depth.rows);
    float color_aspect_ratio =
        static_cast<float>(color.cols) / static_cast<float>(color.rows);
    const float aspect_ratio_diff_th{0.01f};  // 1%
    const float aspect_ratio_diff =
        std::abs(depth_aspect_ratio - color_aspect_ratio);
    if (aspect_ratio_diff > aspect_ratio_diff_th) {
      ugu::LOGE(
          "Depth2Mesh depth aspect ratio %f and color aspect ratio %f are very "
          "different\n",
          depth_aspect_ratio, color_aspect_ratio);
      return false;
    }
  }

  mesh->Clear();

  std::vector<Eigen::Vector2f> uvs;
  std::vector<Eigen::Vector3f> vertices;
  std::vector<Eigen::Vector3i> vertex_indices;
  std::vector<Eigen::Vector3f> vertex_colors;

  std::vector<std::pair<int, int>> vid2xy;

  std::vector<int> added_table(depth.cols * depth.rows, -1);
  int vertex_id{0};
  for (int y = y_step; y < camera.height(); y += y_step) {
    for (int x = x_step; x < camera.width(); x += x_step) {
      const float& d = depth.at<float>(y, x);
      if (d < std::numeric_limits<float>::min()) {
        continue;
      }

      Eigen::Vector3f image_p(static_cast<float>(x), static_cast<float>(y), d);
      Eigen::Vector3f camera_p;
      camera.Unproject(image_p, &camera_p);

      if (gl_coord) {
        // flip y and z to align with OpenGL coordinate
        camera_p.y() = -camera_p.y();
        camera_p.z() = -camera_p.z();
      }

      vertices.push_back(camera_p);

      vid2xy.push_back(std::make_pair(x, y));

      Eigen::Vector2f uv(
          static_cast<float>(x + 0.5f) / static_cast<float>(depth.cols),
          static_cast<float>(y + 0.5f) / static_cast<float>(depth.rows));

      if (with_vertex_color) {
        // nearest neighbor
        // todo: bilinear
        Eigen::Vector2i pixel_pos(
            static_cast<int>(std::round(uv.x() * color.cols)),
            static_cast<int>(std::round(uv.y() * color.rows)));

        Eigen::Vector3f pixel_color;
        const ugu::Vec3b& tmp_color =
            color.at<ugu::Vec3b>(pixel_pos.y(), pixel_pos.x());
        pixel_color.x() = tmp_color[0];
        pixel_color.y() = tmp_color[1];
        pixel_color.z() = tmp_color[2];

        vertex_colors.push_back(pixel_color);
      }

      if (with_texture) {
        // +0.5f comes from mapping 0~1 to -0.5~width(or height)+0.5
        // since uv 0 and 1 is pixel boundary at ends while pixel position is
        // the center of pixel
        uv.y() = 1.0f - uv.y();
        uvs.emplace_back(uv);
      }

      added_table[y * camera.width() + x] = vertex_id;

      const int& current_index = vertex_id;
      const int& upper_left_index =
          added_table[(y - y_step) * camera.width() + (x - x_step)];
      const int& upper_index = added_table[(y - y_step) * camera.width() + x];
      const int& left_index = added_table[y * camera.width() + (x - x_step)];

      const float upper_left_diff =
          std::abs(depth.at<float>(y - y_step, x - x_step) - d);
      const float upper_diff = std::abs(depth.at<float>(y - y_step, x) - d);
      const float left_diff = std::abs(depth.at<float>(y, x - x_step) - d);

      if (upper_left_index > 0 && upper_index > 0 &&
          upper_left_diff < max_connect_z_diff &&
          upper_diff < max_connect_z_diff) {
        vertex_indices.push_back(
            Eigen::Vector3i(upper_left_index, current_index, upper_index));
      }

      if (upper_left_index > 0 && left_index > 0 &&
          upper_left_diff < max_connect_z_diff &&
          left_diff < max_connect_z_diff) {
        vertex_indices.push_back(
            Eigen::Vector3i(upper_left_index, left_index, current_index));
      }

      vertex_id++;
    }
  }

  mesh->set_vertices(vertices);
  mesh->set_vertex_indices(vertex_indices);
  mesh->CalcNormal();

  if (with_texture) {
    mesh->set_uv(uvs);
    mesh->set_uv_indices(vertex_indices);

    ugu::ObjMaterial material;
    color.copyTo(material.diffuse_tex);
    material.name = material_name;
    std::vector<ugu::ObjMaterial> materials;
    materials.push_back(material);
    mesh->set_materials(materials);
    std::vector<int> material_ids(vertex_indices.size(), 0);
    mesh->set_material_ids(material_ids);
  }

  if (with_vertex_color) {
    mesh->set_vertex_colors(vertex_colors);
  }

  if (point_cloud != nullptr) {
    ugu::Init(point_cloud, depth.cols, depth.rows, 0.0f);
    for (int i = 0; i < static_cast<int>(vid2xy.size()); i++) {
      const auto& xy = vid2xy[i];
      auto& p = point_cloud->at<ugu::Vec3f>(xy.second, xy.first);
      p[0] = mesh->vertices()[i][0];
      p[1] = mesh->vertices()[i][1];
      p[2] = mesh->vertices()[i][2];
    }
  }

  if (normal != nullptr) {
    ugu::Init(normal, depth.cols, depth.rows, 0.0f);
    for (int i = 0; i < static_cast<int>(vid2xy.size()); i++) {
      const auto& xy = vid2xy[i];
      auto& n = normal->at<ugu::Vec3f>(xy.second, xy.first);
      n[0] = mesh->normals()[i][0];
      n[1] = mesh->normals()[i][1];
      n[2] = mesh->normals()[i][2];
    }
  }

  return true;
}
}  // namespace

namespace ugu {

bool Depth2PointCloud(const Image1f& depth, const Camera& camera,
                      Image3f* point_cloud, bool gl_coord) {
  if (depth.cols != camera.width() || depth.rows != camera.height()) {
    ugu::LOGE(
        "Depth2PointCloud depth size (%d, %d) and camera size (%d, %d) are "
        "different\n",
        depth.cols, depth.rows, camera.width(), camera.height());
    return false;
  }

  Init(point_cloud, depth.cols, depth.rows, 0.0f);

#if defined(_OPENMP) && defined(UGU_USE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int y = 0; y < camera.height(); y++) {
    for (int x = 0; x < camera.width(); x++) {
      const float& d = depth.at<float>(y, x);
      if (d < std::numeric_limits<float>::min()) {
        continue;
      }

      Eigen::Vector3f image_p(static_cast<float>(x), static_cast<float>(y), d);
      Eigen::Vector3f camera_p;
      camera.Unproject(image_p, &camera_p);

      if (gl_coord) {
        // flip y and z to align with OpenGL coordinate
        camera_p.y() = -camera_p.y();
        camera_p.z() = -camera_p.z();
      }
      Vec3f& pc = point_cloud->at<Vec3f>(y, x);
      pc[0] = camera_p[0];
      pc[1] = camera_p[1];
      pc[2] = camera_p[2];
    }
  }

  return true;
}

bool Depth2PointCloud(const Image1f& depth, const Camera& camera,
                      Mesh* point_cloud, bool gl_coord) {
  Image3b stub_color;
  return Depth2PointCloudImpl(depth, stub_color, camera, point_cloud, false,
                              gl_coord);
}

bool Depth2PointCloud(const Image1f& depth, const Image3b& color,
                      const Camera& camera, Mesh* point_cloud, bool gl_coord) {
  return Depth2PointCloudImpl(depth, color, camera, point_cloud, true,
                              gl_coord);
}

bool Depth2Mesh(const Image1f& depth, const Camera& camera, Mesh* mesh,
                float max_connect_z_diff, int x_step, int y_step, bool gl_coord,
                ugu::Image3f* point_cloud, ugu::Image3f* normal) {
  Image3b stub_color;
  return Depth2MeshImpl(depth, stub_color, camera, mesh, false, false,
                        max_connect_z_diff, x_step, y_step, gl_coord,
                        "illegal_material", point_cloud, normal);
}

bool Depth2Mesh(const Image1f& depth, const Image3b& color,
                const Camera& camera, Mesh* mesh, float max_connect_z_diff,
                int x_step, int y_step, bool gl_coord,
                const std::string& material_name, bool with_vertex_color,
                ugu::Image3f* point_cloud, ugu::Image3f* normal) {
  return Depth2MeshImpl(depth, color, camera, mesh, true, with_vertex_color,
                        max_connect_z_diff, x_step, y_step, gl_coord,
                        material_name, point_cloud, normal);
}

void WriteFaceIdAsText(const Image1i& face_id, const std::string& path) {
  std::ofstream ofs;
  ofs.open(path, std::ios::out);

  for (int y = 0; y < face_id.rows; y++) {
    for (int x = 0; x < face_id.cols; x++) {
      ofs << face_id.at<int>(y, x) << "\n";
    }
  }

  ofs.flush();
}

// FINDING OPTIMAL ROTATION AND TRANSLATION BETWEEN CORRESPONDING 3D POINTS
// http://nghiaho.com/?page_id=671
Eigen::Affine3d FindRigidTransformFrom3dCorrespondences(
    const std::vector<Eigen::Vector3d>& src,
    const std::vector<Eigen::Vector3d>& dst) {
  if (src.size() < 3 || src.size() != dst.size()) {
    return Eigen::Affine3d::Identity();
  }

  Eigen::Vector3d src_centroid;
  src_centroid.setZero();
  for (const auto& p : src) {
    src_centroid += p;
  }
  src_centroid /= static_cast<double>(src.size());

  Eigen::Vector3d dst_centroid;
  dst_centroid.setZero();
  for (const auto& p : dst) {
    dst_centroid += p;
  }
  dst_centroid /= static_cast<double>(dst.size());

  Eigen::MatrixXd normed_src(3, src.size());
  for (auto i = 0; i < src.size(); i++) {
    normed_src.col(i) = src[i] - src_centroid;
  }

  Eigen::MatrixXd normed_dst(3, dst.size());
  for (auto i = 0; i < dst.size(); i++) {
    normed_dst.col(i) = dst[i] - dst_centroid;
  }

  Eigen::MatrixXd normed_dst_T = normed_dst.transpose();
  Eigen::Matrix3d H = normed_src * normed_dst_T;

  // TODO: rank check

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d R = svd.matrixV() * svd.matrixU().transpose();
  double det = R.determinant();

  constexpr double assert_eps = 0.001;
  assert(std::abs(std::abs(det) - 1.0) < assert_eps);

  if (det < 0) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd2(
        R, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Matrix3d V = svd2.matrixV();

    V.coeffRef(0, 2) *= -1.0;
    V.coeffRef(1, 2) *= -1.0;
    V.coeffRef(2, 2) *= -1.0;

    R = V * svd2.matrixU().transpose();
  }
  assert(std::abs(det - 1.0) < assert_eps);

  Eigen::Vector3d t = dst_centroid - R * src_centroid;

  Eigen::Affine3d T = Eigen::Translation3d(t) * R;

  return T;
}

Eigen::Affine3d FindRigidTransformFrom3dCorrespondences(
    const std::vector<Eigen::Vector3f>& src,
    const std::vector<Eigen::Vector3f>& dst) {
  std::vector<Eigen::Vector3d> src_d, dst_d;
  auto to_double = [](const std::vector<Eigen::Vector3f>& fvec,
                      std::vector<Eigen::Vector3d>& dvec) {
    std::transform(fvec.begin(), fvec.end(), std::back_inserter(dvec),
                   [](const Eigen::Vector3f& f) { return f.cast<double>(); });
  };

  to_double(src, src_d);
  to_double(dst, dst_d);
  return FindRigidTransformFrom3dCorrespondences(src_d, dst_d);
}

Eigen::Affine3d FindSimilarityTransformFrom3dCorrespondences(
    const std::vector<Eigen::Vector3d>& src,
    const std::vector<Eigen::Vector3d>& dst) {
  Eigen::MatrixXd src_(src.size(), 3);
  for (auto i = 0; i < src.size(); i++) {
    src_.row(i) = src[i];
  }
  Eigen::MatrixXd dst_(dst.size(), 3);
  for (auto i = 0; i < dst.size(); i++) {
    dst_.row(i) = dst[i];
  }
  return FindSimilarityTransformFrom3dCorrespondences(src_, dst_);
}

Eigen::Affine3d FindSimilarityTransformFrom3dCorrespondences(
    const std::vector<Eigen::Vector3f>& src,
    const std::vector<Eigen::Vector3f>& dst) {
  Eigen::MatrixXd src_(src.size(), 3);
  for (auto i = 0; i < src.size(); i++) {
    src_.row(i) = src[i].cast<double>();
  }
  Eigen::MatrixXd dst_(dst.size(), 3);
  for (auto i = 0; i < dst.size(); i++) {
    dst_.row(i) = dst[i].cast<double>();
  }
  return FindSimilarityTransformFrom3dCorrespondences(src_, dst_);
}

Eigen::Affine3d FindSimilarityTransformFrom3dCorrespondences(
    const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst) {
  Eigen::MatrixXd R;
  Eigen::MatrixXd t;
  Eigen::MatrixXd scale;
  Eigen::MatrixXd T;
  bool ret =
      FindSimilarityTransformFromPointCorrespondences(src, dst, R, t, scale, T);
  assert(R.rows() == 3 && R.cols() == 3);
  assert(t.rows() == 3 && t.cols() == 1);
  assert(T.rows() == 4 && T.cols() == 4);
  Eigen::Affine3d T_3d = Eigen::Affine3d::Identity();
  if (ret) {
    T_3d = Eigen::Translation3d(t) * R * Eigen::Scaling(scale.diagonal());
  }

  return T_3d;
}

bool FindSimilarityTransformFromPointCorrespondences(
    const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst, Eigen::MatrixXd& R,
    Eigen::MatrixXd& t, Eigen::MatrixXd& scale, Eigen::MatrixXd& T) {
  const size_t n_data = src.rows();
  const size_t n_dim = src.cols();
  if (n_data < 1 || n_dim < 1 || n_data < n_dim || src.rows() != dst.rows() ||
      src.cols() != dst.cols()) {
    return false;
  }

  Eigen::VectorXd src_mean = src.colwise().mean();
  Eigen::VectorXd dst_mean = dst.colwise().mean();

  Eigen::MatrixXd src_demean = src.rowwise() - src_mean.transpose();
  Eigen::MatrixXd dst_demean = dst.rowwise() - dst_mean.transpose();

  Eigen::MatrixXd A =
      dst_demean.transpose() * src_demean / static_cast<double>(n_data);

  Eigen::VectorXd d = Eigen::VectorXd::Ones(n_dim);
  double det_A = A.determinant();
  if (det_A < 0) {
    d.coeffRef(n_dim - 1, 0) = -1;
  }

  T = Eigen::MatrixXd::Identity(n_dim + 1, n_dim + 1);

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::MatrixXd U = svd.matrixU();
  Eigen::MatrixXd S = svd.singularValues().asDiagonal();
  Eigen::MatrixXd V = svd.matrixV();

  double det_U = U.determinant();
  double det_V = V.determinant();
  double det_orgR = det_U * det_V;
  constexpr double assert_eps = 0.001;
  assert(std::abs(std::abs(det_orgR) - 1.0) < assert_eps);

  int rank_A =  static_cast<int>(svd.rank());
  if (rank_A == 0) {
    // null matrix case
    return false;
  } else if (rank_A == n_dim - 1) {
    if (det_orgR > 0) {
      // Valid rotation case
      R = U * V.transpose();
    } else {
      // Mirror (reflection) case
      double s = d.coeff(n_dim - 1, 0);
      d.coeffRef(n_dim - 1, 0) = -1;
      R = U * d.asDiagonal() * V.transpose();
      d.coeffRef(n_dim - 1, 0) = s;
    }
  } else {
    // degenerate case
    R = U * d.asDiagonal() * V.transpose();
  }
  assert(std::abs(R.determinant() - 1.0) < assert_eps);

  // Eigen::MatrixXd src_demean_cov =
  //    (src_demean.adjoint() * src_demean) / double(n_data);
  double src_var =
      src_demean.rowwise().squaredNorm().sum() / double(n_data) + 1e-30;
  double uniform_scale = 1.0 / src_var * (S * d.asDiagonal()).trace();
  // Question: Is it possible to estimate non-uniform scale?
  scale = Eigen::MatrixXd::Identity(R.rows(), R.cols());
  scale *= uniform_scale;

  t = dst_mean - scale * R * src_mean;

  T.block(0, 0, n_dim, n_dim) = scale * R;
  T.block(0, n_dim, n_dim, 1) = t;

  return true;
}

}  // namespace ugu
