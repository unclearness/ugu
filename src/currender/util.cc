/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "currender/util.h"

#include <fstream>

namespace {

bool Depth2PointCloudImpl(const currender::Image1f& depth,
                          const currender::Image3b& color,
                          const currender::Camera& camera,
                          currender::Mesh* point_cloud, bool with_texture,
                          bool gl_coord) {
  if (depth.width() != camera.width() || depth.height() != camera.height()) {
    currender::LOGE(
        "Depth2PointCloud depth size (%d, %d) and camera size (%d, %d) are "
        "different\n",
        depth.width(), depth.height(), camera.width(), camera.height());
    return false;
  }

  if (with_texture) {
    float depth_aspect_ratio =
        static_cast<float>(depth.width()) / static_cast<float>(depth.height());
    float color_aspect_ratio =
        static_cast<float>(color.width()) / static_cast<float>(color.height());
    const float aspect_ratio_diff_th{0.01f};  // 1%
    const float aspect_ratio_diff =
        std::abs(depth_aspect_ratio - color_aspect_ratio);
    if (aspect_ratio_diff > aspect_ratio_diff_th) {
      currender::LOGE(
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
      const float& d = depth.at(x, y, 0);
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
            static_cast<float>(x + 0.5f) / static_cast<float>(depth.width()),
            static_cast<float>(y + 0.5f) / static_cast<float>(depth.height()));

        // nearest neighbor
        // todo: bilinear
        Eigen::Vector2i pixel_pos(
            static_cast<int>(std::round(uv.x() * color.width())),
            static_cast<int>(std::round(uv.y() * color.height())));

        Eigen::Vector3f pixel_color;
        pixel_color.x() = color.at(pixel_pos.x(), pixel_pos.y(), 0);
        pixel_color.y() = color.at(pixel_pos.x(), pixel_pos.y(), 1);
        pixel_color.z() = color.at(pixel_pos.x(), pixel_pos.y(), 2);

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

bool Depth2MeshImpl(const currender::Image1f& depth,
                    const currender::Image3b& color,
                    const currender::Camera& camera, currender::Mesh* mesh,
                    bool with_texture, float max_connect_z_diff, int x_step,
                    int y_step, bool gl_coord,
                    const std::string material_name) {
  if (max_connect_z_diff < 0) {
    currender::LOGE("Depth2Mesh max_connect_z_diff must be positive %f\n",
                    max_connect_z_diff);
    return false;
  }
  if (x_step < 1) {
    currender::LOGE("Depth2Mesh x_step must be positive %d\n", x_step);
    return false;
  }
  if (y_step < 1) {
    currender::LOGE("Depth2Mesh y_step must be positive %d\n", y_step);
    return false;
  }

  if (depth.width() != camera.width() || depth.height() != camera.height()) {
    currender::LOGE(
        "Depth2Mesh depth size (%d, %d) and camera size (%d, %d) are "
        "different\n",
        depth.width(), depth.height(), camera.width(), camera.height());
    return false;
  }

  if (with_texture) {
    float depth_aspect_ratio =
        static_cast<float>(depth.width()) / static_cast<float>(depth.height());
    float color_aspect_ratio =
        static_cast<float>(color.width()) / static_cast<float>(color.height());
    const float aspect_ratio_diff_th{0.01f};  // 1%
    const float aspect_ratio_diff =
        std::abs(depth_aspect_ratio - color_aspect_ratio);
    if (aspect_ratio_diff > aspect_ratio_diff_th) {
      currender::LOGE(
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

  std::vector<int> added_table(depth.data().size(), -1);
  int vertex_id{0};
  for (int y = y_step; y < camera.height(); y += y_step) {
    for (int x = x_step; x < camera.width(); x += x_step) {
      const float& d = depth.at(x, y, 0);
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
        uvs.emplace_back(
            static_cast<float>(x + 0.5f) / static_cast<float>(depth.width()),
            1.0f - static_cast<float>(y + 0.5f) /
                       static_cast<float>(depth.height()));
      }

      added_table[y * camera.width() + x] = vertex_id;

      const int& current_index = vertex_id;
      const int& upper_left_index =
          added_table[(y - y_step) * camera.width() + (x - x_step)];
      const int& upper_index = added_table[(y - y_step) * camera.width() + x];
      const int& left_index = added_table[y * camera.width() + (x - x_step)];

      const float upper_left_diff =
          std::abs(depth.at(x - x_step, y - y_step, 0) - d);
      const float upper_diff = std::abs(depth.at(x, y - y_step, 0) - d);
      const float left_diff = std::abs(depth.at(x - x_step, y, 0) - d);

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
    currender::ObjMaterial material;
    color.CopyTo(&material.diffuse_tex);
    material.name = material_name;
    std::vector<currender::ObjMaterial> materials;
    materials.push_back(material);
    mesh->set_materials(materials);
    std::vector<int> material_ids(vertex_indices.size(), 0);
    mesh->set_material_ids(material_ids);
  }

  return true;
}
}  // namespace

namespace currender {

bool Depth2PointCloud(const Image1f& depth, const Camera& camera,
                      Image3f* point_cloud, bool gl_coord) {
  if (depth.width() != camera.width() || depth.height() != camera.height()) {
    currender::LOGE(
        "Depth2PointCloud depth size (%d, %d) and camera size (%d, %d) are "
        "different\n",
        depth.width(), depth.height(), camera.width(), camera.height());
    return false;
  }

  point_cloud->Init(depth.width(), depth.height(), 0.0f);

#if defined(_OPENMP) && defined(CURRENDER_USE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int y = 0; y < camera.height(); y++) {
    for (int x = 0; x < camera.width(); x++) {
      const float& d = depth.at(x, y, 0);
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
      point_cloud->at(x, y, 0) = camera_p[0];
      point_cloud->at(x, y, 1) = camera_p[1];
      point_cloud->at(x, y, 2) = camera_p[2];
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
                float max_connect_z_diff, int x_step, int y_step,
                bool gl_coord) {
  Image3b stub_color;
  return Depth2MeshImpl(depth, stub_color, camera, mesh, false,
                        max_connect_z_diff, x_step, y_step, gl_coord,
                        "illegal_material");
}

bool Depth2Mesh(const Image1f& depth, const Image3b& color,
                const Camera& camera, Mesh* mesh, float max_connect_z_diff,
                int x_step, int y_step, bool gl_coord,
                const std::string material_name) {
  return Depth2MeshImpl(depth, color, camera, mesh, true, max_connect_z_diff,
                        x_step, y_step, gl_coord, material_name);
}

void WriteFaceIdAsText(const Image1i& face_id, const std::string& path) {
  std::ofstream ofs;
  ofs.open(path, std::ios::out);

  for (int y = 0; y < face_id.height(); y++) {
    for (int x = 0; x < face_id.width(); x++) {
      ofs << face_id.at(x, y, 0) << std::endl;
    }
  }
}

}  // namespace currender
