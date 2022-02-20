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
        // +0.5f comes from mapping 0~1 to -0.5~width(or height)+0.5
        // since uv 0 and 1 is pixel boundary at ends while pixel position is
        // the center of pixel
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

}  // namespace ugu
