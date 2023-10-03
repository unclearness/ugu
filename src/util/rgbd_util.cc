/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/util/rgbd_util.h"

#include "ugu/util/raster_util.h"

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
        Eigen::Vector2f uv(ugu::X2U(static_cast<float>(x), depth.cols),
                           ugu::Y2V(static_cast<float>(y), depth.rows, false));

        // nearest neighbor
        // todo: bilinear
        Eigen::Vector2i pixel_pos(
            static_cast<int>(std::round(ugu::U2X(uv.x(), color.cols))),
            static_cast<int>(std::round(ugu::V2Y(uv.y(), color.rows, false))));

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

  std::vector<int> added_table(static_cast<size_t>(depth.cols * depth.rows),
                               -1);
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

      Eigen::Vector2f uv(ugu::X2U(static_cast<float>(x), depth.cols),
                         ugu::Y2V(static_cast<float>(y), depth.rows, false));

      if (with_vertex_color) {
        // nearest neighbor
        // todo: bilinear
        Eigen::Vector2i pixel_pos(
            static_cast<int>(std::round(ugu::U2X(uv.x(), color.cols))),
            static_cast<int>(std::round(ugu::V2Y(uv.y(), color.rows, false))));

        Eigen::Vector3f pixel_color;
        const ugu::Vec3b& tmp_color =
            color.at<ugu::Vec3b>(pixel_pos.y(), pixel_pos.x());
        pixel_color.x() = tmp_color[0];
        pixel_color.y() = tmp_color[1];
        pixel_color.z() = tmp_color[2];

        vertex_colors.push_back(pixel_color);
      }

      if (with_texture) {
        uv.y() = 1.0f - uv.y();
        uvs.emplace_back(uv);
      }

      added_table[static_cast<size_t>(y) * static_cast<size_t>(camera.width()) +
                  static_cast<size_t>(x)] = vertex_id;

      const int& current_index = vertex_id;
      const int& upper_left_index =
          added_table[(static_cast<size_t>(y) - static_cast<size_t>(y_step)) *
                          static_cast<size_t>(camera.width()) +
                      (static_cast<size_t>(x) - static_cast<size_t>(x_step))];
      const int& upper_index =
          added_table[(static_cast<size_t>(y) - static_cast<size_t>(y_step)) *
                          static_cast<size_t>(camera.width()) +
                      static_cast<size_t>(x)];
      const int& left_index =
          added_table[static_cast<size_t>(y) *
                          static_cast<size_t>(camera.width()) +
                      (static_cast<size_t>(x) - static_cast<size_t>(x_step))];

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

bool Depth2MeshImpl(const ugu::Image1f& depth, const ugu::Image3b& color,
                    const ugu::Camera& depth_camera,
                    const ugu::Camera& color_camera,
                    const Eigen::Affine3f& depth2color, ugu::Mesh* mesh,
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
  if (depth.cols != depth_camera.width() ||
      depth.rows != depth_camera.height()) {
    ugu::LOGE(
        "Depth2Mesh depth size (%d, %d) and camera size (%d, %d) are "
        "different\n",
        depth.cols, depth.rows, depth_camera.width(), depth_camera.height());
    return false;
  }

  mesh->Clear();

  std::vector<Eigen::Vector2f> uvs;
  std::vector<Eigen::Vector3f> vertices;
  std::vector<Eigen::Vector3i> vertex_indices;
  std::vector<Eigen::Vector3f> vertex_colors;

  std::vector<std::pair<int, int>> vid2xy;

  std::vector<int> added_table(static_cast<size_t>(depth.cols * depth.rows),
                               -1);
  int vertex_id{0};
  for (int y = y_step; y < depth_camera.height(); y += y_step) {
    for (int x = x_step; x < depth_camera.width(); x += x_step) {
      const float& d = depth.at<float>(y, x);
      if (d < std::numeric_limits<float>::min()) {
        continue;
      }

      Eigen::Vector3f depth_image_p(static_cast<float>(x),
                                    static_cast<float>(y), d);
      Eigen::Vector3f depth_camera_p;
      depth_camera.Unproject(depth_image_p, &depth_camera_p);

      if (gl_coord) {
        // flip y and z to align with OpenGL coordinate
        depth_camera_p.y() = -depth_camera_p.y();
        depth_camera_p.z() = -depth_camera_p.z();
      }

      vertices.push_back(depth_camera_p);

      vid2xy.push_back(std::make_pair(x, y));

      Eigen::Vector3f color_camera_p = depth2color * depth_camera_p;
      Eigen::Vector3f color_image_p;
      color_camera.Project(color_camera_p, &color_image_p);

      Eigen::Vector2f uv(ugu::X2U(color_image_p.x(), color.cols),
                         ugu::Y2V(color_image_p.y(), color.rows, false));

      if (with_vertex_color) {
        // nearest neighbor
        // todo: bilinear
        Eigen::Vector2i pixel_pos(
            static_cast<int>(std::round(ugu::U2X(uv.x(), color.cols))),
            static_cast<int>(std::round(ugu::V2Y(uv.y(), color.rows, false))));

        Eigen::Vector3f pixel_color;
        const ugu::Vec3b& tmp_color =
            color.at<ugu::Vec3b>(pixel_pos.y(), pixel_pos.x());
        pixel_color.x() = tmp_color[0];
        pixel_color.y() = tmp_color[1];
        pixel_color.z() = tmp_color[2];

        vertex_colors.push_back(pixel_color);
      }

      if (with_texture) {
        uv.y() = 1.0f - uv.y();
        uvs.emplace_back(uv);
      }

      added_table[static_cast<size_t>(y) *
                      static_cast<size_t>(depth_camera.width()) +
                  static_cast<size_t>(x)] = vertex_id;

      const int& current_index = vertex_id;
      const int& upper_left_index =
          added_table[(static_cast<size_t>(y) - static_cast<size_t>(y_step)) *
                          static_cast<size_t>(depth_camera.width()) +
                      (static_cast<size_t>(x) - static_cast<size_t>(x_step))];
      const int& upper_index =
          added_table[(static_cast<size_t>(y) - static_cast<size_t>(y_step)) *
                          static_cast<size_t>(depth_camera.width()) +
                      static_cast<size_t>(x)];
      const int& left_index =
          added_table[static_cast<size_t>(y) *
                          static_cast<size_t>(depth_camera.width()) +
                      (static_cast<size_t>(x) - static_cast<size_t>(x_step))];

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
    if (point_cloud->cols != depth.cols || point_cloud->rows != depth.rows) {
      ugu::Init(point_cloud, depth.cols, depth.rows, 0.0f);
    }
    for (int i = 0; i < static_cast<int>(vid2xy.size()); i++) {
      const auto& xy = vid2xy[i];
      auto& p = point_cloud->at<ugu::Vec3f>(xy.second, xy.first);
      p[0] = mesh->vertices()[i][0];
      p[1] = mesh->vertices()[i][1];
      p[2] = mesh->vertices()[i][2];
    }
  }

  if (normal != nullptr) {
    if (normal->cols != depth.cols || normal->rows != depth.rows) {
      ugu::Init(normal, depth.cols, depth.rows, 0.0f);
    }
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

  if (point_cloud->cols != depth.cols || point_cloud->rows != depth.rows) {
    Init(point_cloud, depth.cols, depth.rows, 0.0f);
  }

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
                Image3f* point_cloud, Image3f* normal) {
  Image3b stub_color;
  return Depth2MeshImpl(depth, stub_color, camera, mesh, false, false,
                        max_connect_z_diff, x_step, y_step, gl_coord,
                        "illegal_material", point_cloud, normal);
}

bool Depth2Mesh(const Image1f& depth, const Image3b& color,
                const Camera& camera, Mesh* mesh, float max_connect_z_diff,
                int x_step, int y_step, bool gl_coord,
                const std::string& material_name, bool with_vertex_color,
                Image3f* point_cloud, Image3f* normal) {
  return Depth2MeshImpl(depth, color, camera, mesh, true, with_vertex_color,
                        max_connect_z_diff, x_step, y_step, gl_coord,
                        material_name, point_cloud, normal);
}

bool Depth2Mesh(const Image1f& depth, const Image3b& color,
                const Camera& depth_camera, const Camera& color_camera,
                const Eigen::Affine3f depth2color, Mesh* mesh,
                float max_connect_z_diff, int x_step, int y_step, bool gl_coord,
                const std::string& material_name, bool with_vertex_color,
                Image3f* point_cloud, Image3f* normal) {
  return Depth2MeshImpl(depth, color, depth_camera, color_camera, depth2color,
                        mesh, true, with_vertex_color, max_connect_z_diff,
                        x_step, y_step, gl_coord, material_name, point_cloud,
                        normal);
}

bool ComputeNormal(const Image1f& depth, const Camera& camera, Image3f* normal,
                   float max_connect_z_diff, int x_step, int y_step,
                   bool gl_coord, bool to_world, Image3f* organized_pc,
                   Image1b* valid_mask, uint32_t num_threads) {
  if (organized_pc == nullptr) {
    ugu::Init(organized_pc, depth.cols, depth.rows, 0.0f);
  }

  if (valid_mask == nullptr) {
    ugu::Init(valid_mask, depth.cols, depth.rows, uint8_t(0));
  }

  Eigen::Affine3f c2w = camera.c2w().cast<float>();

  auto process_body = [&](int x, int y, float d) {
    Eigen::Vector3f image_p(static_cast<float>(x), static_cast<float>(y), d);
    Eigen::Vector3f camera_p;
    camera.Unproject(image_p, &camera_p);
    if (gl_coord) {
      // flip y and z to align with OpenGL coordinate
      camera_p.y() = -camera_p.y();
      camera_p.z() = -camera_p.z();
    }

    if (to_world) {
      camera_p = c2w * camera_p;
    }

    auto& pc = organized_pc->at<Vec3f>(y, x);
    pc[0] = camera_p[0];
    pc[1] = camera_p[1];
    pc[2] = camera_p[2];

    valid_mask->at<uint8_t>(y, x) = 255;
  };

  if (num_threads == 1u) {
    for (int y = 0; y < camera.height(); y++) {
      for (int x = 0; x < camera.width(); x++) {
        const float& d = depth.at<float>(y, x);
        if (d < std::numeric_limits<float>::min()) {
          continue;
        }
        process_body(x, y, d);
      }
    }
  } else {
    auto pix_func = [&](int index) {
      int y = index / camera.width();
      int x = index % camera.width();
      const float& d = depth.at<float>(y, x);
      if (d < std::numeric_limits<float>::min()) {
        return;
      }
      process_body(x, y, d);
    };
    parallel_for(0, camera.height() * camera.width(), pix_func, num_threads);
  }

  return ComputeNormal(*organized_pc, normal, max_connect_z_diff, x_step,
                       y_step, valid_mask, num_threads);
}

bool ComputeNormal(const Image3f& organized_pc, Image3f* normal,
                   float max_connect_z_diff, int x_step, int y_step,
                   Image1b* valid_mask, uint32_t num_threads) {
  if (normal == nullptr) {
    ugu::Init(normal, organized_pc.cols, organized_pc.rows, 0.0f);
  }

  if (valid_mask == nullptr) {
    ugu::Init(valid_mask, organized_pc.cols, organized_pc.rows, uint8_t(255));
  }

  if (num_threads == 1u) {
    for (int y = 0; y < organized_pc.rows; y++) {
      for (int x = 0; x < organized_pc.cols; x++) {
        if (x < x_step || organized_pc.cols - x_step <= x || y < y_step ||
            organized_pc.rows - y_step <= y) {
          continue;
        }

        if (valid_mask->at<uint8_t>(y, x) < 255 ||
            valid_mask->at<uint8_t>(y, x - x_step) < 255 ||
            valid_mask->at<uint8_t>(y, x + x_step) < 255 ||
            valid_mask->at<uint8_t>(y - y_step, x) < 255 ||
            valid_mask->at<uint8_t>(y + y_step, x) < 255) {
          continue;
        }

        const Vec3f& pos = organized_pc.at<Vec3f>(y, x);
        const Vec3f& left_pos = organized_pc.at<Vec3f>(y, x - x_step);
        const Vec3f& right_pos = organized_pc.at<Vec3f>(y, x + x_step);

        if (max_connect_z_diff > 0.f &&
            (max_connect_z_diff < std::abs(pos[2] - left_pos[2]) ||
             max_connect_z_diff < std::abs(pos[2] - right_pos[2]))) {
          continue;
        }

        const Vec3f& up_pos = organized_pc.at<Vec3f>(y - y_step, x);
        const Vec3f& down_pos = organized_pc.at<Vec3f>(y + y_step, x);

        if (max_connect_z_diff > 0.f &&
            (max_connect_z_diff < std::abs(pos[2] - up_pos[2]) ||
             max_connect_z_diff < std::abs(pos[2] - down_pos[2]))) {
          continue;
        }

        Eigen::Vector3f left2right(right_pos[0] - left_pos[0],
                                   right_pos[1] - left_pos[1],
                                   right_pos[2] - left_pos[2]);
        Eigen::Vector3f up2down(down_pos[0] - up_pos[0],
                                down_pos[1] - up_pos[1],
                                down_pos[2] - up_pos[2]);

        Eigen::Vector3f n_ =
            up2down.normalized().cross(left2right.normalized()).normalized();

        auto& n = normal->at<Vec3f>(y, x);
        n[0] = n_[0];
        n[1] = n_[1];
        n[2] = n_[2];
      }
    }
  } else {
    auto pix_func = [&](int index) {
      int y = index / organized_pc.cols;
      int x = index % organized_pc.cols;

      if (x < x_step || organized_pc.cols - x_step <= x || y < y_step ||
          organized_pc.rows - y_step <= y) {
        return;
      }

      if (valid_mask->at<uint8_t>(y, x) < 255 ||
          valid_mask->at<uint8_t>(y, x - x_step) < 255 ||
          valid_mask->at<uint8_t>(y, x + x_step) < 255 ||
          valid_mask->at<uint8_t>(y - y_step, x) < 255 ||
          valid_mask->at<uint8_t>(y + y_step, x) < 255) {
        return;
      }

      const Vec3f& pos = organized_pc.at<Vec3f>(y, x);
      const Vec3f& left_pos = organized_pc.at<Vec3f>(y, x - x_step);
      const Vec3f& right_pos = organized_pc.at<Vec3f>(y, x + x_step);

      if (max_connect_z_diff > 0.f &&
          (max_connect_z_diff < std::abs(pos[2] - left_pos[2]) ||
           max_connect_z_diff < std::abs(pos[2] - right_pos[2]))) {
        return;
      }

      const Vec3f& up_pos = organized_pc.at<Vec3f>(y - y_step, x);
      const Vec3f& down_pos = organized_pc.at<Vec3f>(y + y_step, x);

      if (max_connect_z_diff > 0.f &&
          (max_connect_z_diff < std::abs(pos[2] - up_pos[2]) ||
           max_connect_z_diff < std::abs(pos[2] - down_pos[2]))) {
        return;
      }

      Eigen::Vector3f left2right(right_pos[0] - left_pos[0],
                                 right_pos[1] - left_pos[1],
                                 right_pos[2] - left_pos[2]);
      Eigen::Vector3f up2down(down_pos[0] - up_pos[0], down_pos[1] - up_pos[1],
                              down_pos[2] - up_pos[2]);

      Eigen::Vector3f n_ =
          up2down.normalized().cross(left2right.normalized()).normalized();

      auto& n = normal->at<Vec3f>(y, x);
      n[0] = n_[0];
      n[1] = n_[1];
      n[2] = n_[2];
    };

    parallel_for(0, organized_pc.cols * organized_pc.rows, pix_func,
                 num_threads);
  }

  return true;
}

}  // namespace ugu
