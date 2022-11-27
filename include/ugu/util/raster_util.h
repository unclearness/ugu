/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/image.h"
#include "ugu/util/image_util.h"
#include "ugu/util/thread_util.h"

namespace ugu {

// +0.5f comes from mapping 0~1 to -0.5~width(or height)+0.5
// since uv 0 and 1 is pixel boundary at ends while pixel position is
// the center of pixel

inline float U2X(float u, int w) { return u * w - 0.5f; }
inline float V2Y(float v, int h, bool flip = true) {
  return flip ? (1.f - v) * h - 0.5f : v * h - 0.5f;
}
inline float X2U(float x, int w) {
  return static_cast<float>(x + 0.5f) / static_cast<float>(w);
}
inline float Y2V(float y, int h, bool flip = true) {
  float v = static_cast<float>(y + 0.5f) / static_cast<float>(h);
  return flip ? 1.f - v : v;
}

template <typename T>
float EdgeFunction(const T& a, const T& b, const T& c) {
  return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0]);
}

template <typename T>
float TriArea(const T& a, const T& b, const T& c) {
  return 0.5f * ((b - a).cross(c - a)).norm();
}

inline std::tuple<bool, Eigen::Vector2f> IsPoint3dInsideTriangle(
    const Eigen::Vector3f& p, const Eigen::Vector3f& v0,
    const Eigen::Vector3f& v1, const Eigen::Vector3f& v2, float eps = 0.01f) {
  float area = ugu::TriArea(v0, v1, v2);
  float inv_area = 1.f / area;
  float w0 = ugu::TriArea(v1, v2, p) * inv_area;
  float w1 = ugu::TriArea(v2, v0, p) * inv_area;
  float w2 = ugu::TriArea(v0, v1, p) * inv_area;
  Eigen::Vector2f bary(w0, w1);
  if (w0 < 0 || w1 < 0 || w2 < 0 || 1 < w0 || 1 < w1 || 1 < w2) {
    return std::make_tuple(false, bary);
  }

  if (std::abs(w0 + w1 + w2 - 1.f) > eps || w0 + w1 > 1.f) {
    return std::make_tuple(false, bary);
  }
  return std::make_tuple(true, bary);
}

template <typename T>
std::tuple<double, double, double> Barycentric(const T& p, const T& a,
                                               const T& b, const T& c) {
  T v0 = b - a;
  T v1 = c - a;
  T v2 = p - a;
  double d00 = v0.dot(v0);
  double d01 = v0.dot(v1);
  double d11 = v1.dot(v1);
  double d20 = v2.dot(v0);
  double d21 = v2.dot(v1);
  double denom = d00 * d11 - d01 * d01;
  double v = (d11 * d20 - d01 * d21) / denom;
  double w = (d00 * d21 - d01 * d20) / denom;
  double u = 1.0 - v - w;
  return {u, v, w};
}

template <typename T>
bool RasterizeTriangle(const std::array<Eigen::Vector3f, 3>& src_vetex_color,
                       const std::array<Eigen::Vector2f, 3>& target_tri,
                       ugu::Image<T>* target, ugu::Image1b* mask,
                       float min_val = 1.f, float max_val = -1.f) {
  // Area could be negative
  float area = EdgeFunction(target_tri[0], target_tri[1], target_tri[2]);
  if (std::abs(area) < std::numeric_limits<float>::min()) {
    area = area > 0 ? std::numeric_limits<float>::min()
                    : -std::numeric_limits<float>::min();
  }
  float inv_area = 1.0f / area;

  // Loop for bounding box of the target triangle
  int xmin = static_cast<int>(
      std::min({target_tri[0].x(), target_tri[1].x(), target_tri[2].x()}) - 1);
  xmin = std::max(0, std::min(xmin, target->cols - 1));
  int xmax = static_cast<int>(
      std::max({target_tri[0].x(), target_tri[1].x(), target_tri[2].x()}) + 1);
  xmax = std::max(0, std::min(xmax, target->cols - 1));

  int ymin = static_cast<int>(
      std::min({target_tri[0].y(), target_tri[1].y(), target_tri[2].y()}) - 1);
  ymin = std::max(0, std::min(ymin, target->rows - 1));
  int ymax = static_cast<int>(
      std::max({target_tri[0].y(), target_tri[1].y(), target_tri[2].y()}) + 1);
  ymax = std::max(0, std::min(ymax, target->rows - 1));

  auto saturate_cast = [=](float x) {
    constexpr bool is_integer =
        std::numeric_limits<typename T::value_type>::is_integer;
    if (min_val < max_val) {
      x = std::clamp(x, min_val, max_val);
    }
    if (is_integer) {
      return std::round(x);
    }
    return x;
  };

  for (int y = ymin; y <= ymax; y++) {
    for (int x = xmin; x <= xmax; x++) {
      Eigen::Vector2f pixel_sample(static_cast<float>(x),
                                   static_cast<float>(y));
      float w0 = EdgeFunction(target_tri[1], target_tri[2], pixel_sample);
      float w1 = EdgeFunction(target_tri[2], target_tri[0], pixel_sample);
      float w2 = EdgeFunction(target_tri[0], target_tri[1], pixel_sample);
      // Barycentric in the target triangle
      w0 *= inv_area;
      w1 *= inv_area;
      w2 *= inv_area;

      // Barycentric coordinate should be positive inside of the triangle
      // Skip outside of the target triangle
      if (w0 < 0 || w1 < 0 || w2 < 0) {
        continue;
      }

      // Barycentric to interpolate color
      Eigen::Vector3f color = w0 * src_vetex_color[0] +
                              w1 * src_vetex_color[1] + w2 * src_vetex_color[2];

      target->template at<T>(y, x) =
          T({static_cast<typename T::value_type>(saturate_cast(color[0])),
             static_cast<typename T::value_type>(saturate_cast(color[1])),
             static_cast<typename T::value_type>(saturate_cast(color[2]))});

      if (mask != nullptr) {
        mask->at<unsigned char>(y, x) = 255;
      }
    }
  }

  return true;
}

template <typename T>
bool RasterizeVertexAttributeToTexture(
    const std::vector<Eigen::Vector3f>& vertex_attrs,
    const std::vector<Eigen::Vector3i>& vertex_attr_indices,
    const std::vector<Eigen::Vector2f>& uvs,
    const std::vector<Eigen::Vector3i>& uv_indices, Image<T>& texture,
    int width = 512, int height = 512, ugu::Image1b* mask = nullptr,
    int num_threads = -1) {
  if (vertex_attr_indices.empty() ||
      vertex_attr_indices.size() != uv_indices.size()) {
    return false;
  }

  if (width > 0 && height > 0) {
    texture = Image<T>::zeros(height, width);
  } else if (texture.empty()) {
    return false;
  }

  const auto& face_num = vertex_attr_indices.size();

  auto face_func = [&](size_t i) {
    const auto& vc_face = vertex_attr_indices[i];
    const auto& uv_face = uv_indices[i];

    std::array<Eigen::Vector3f, 3> src_vetex_color{vertex_attrs[vc_face[0]],
                                                   vertex_attrs[vc_face[1]],
                                                   vertex_attrs[vc_face[2]]};
    std::array<Eigen::Vector2f, 3> target_tri{uvs[uv_face[0]], uvs[uv_face[1]],
                                              uvs[uv_face[2]]};

    for (auto& tri : target_tri) {
      tri.x() = ugu::U2X(tri.x(), texture.cols);
      tri.y() = ugu::V2Y(tri.y(), texture.rows);
    }

    RasterizeTriangle(src_vetex_color, target_tri, &texture, mask);
  };

  parallel_for(size_t(0), face_num, face_func, num_threads);

  return true;
}

template <typename T>
bool RasterizeFaceAttributeToTexture(
    const std::vector<Eigen::Vector3f>& face_attrs,
    const std::vector<Eigen::Vector2f>& uvs,
    const std::vector<Eigen::Vector3i>& uv_indices, Image<T>& texture,
    int width = 512, int height = 512, ugu::Image1b* mask = nullptr,
    int num_threads = -1) {
  if (face_attrs.empty() || face_attrs.size() != uv_indices.size()) {
    return false;
  }

  if (width > 0 && height > 0) {
    texture = Image<T>::zeros(height, width);
  } else if (texture.empty()) {
    return false;
  }

  const auto& face_num = face_attrs.size();
  auto face_func = [&](size_t i) {
    const auto& face_attr = face_attrs[i];
    const auto& uv_face = uv_indices[i];

    std::array<Eigen::Vector3f, 3> src_vetex_color{face_attr, face_attr,
                                                   face_attr};
    std::array<Eigen::Vector2f, 3> target_tri{uvs[uv_face[0]], uvs[uv_face[1]],
                                              uvs[uv_face[2]]};

    for (auto& tri : target_tri) {
      tri.x() = ugu::U2X(tri.x(), texture.cols);
      tri.y() = ugu::V2Y(tri.y(), texture.rows);
    }

    RasterizeTriangle(src_vetex_color, target_tri, &texture, mask);
  };

  parallel_for(size_t(0), face_num, face_func, num_threads);

  return true;
}

template <typename T>
bool GenerateUvMask(const std::vector<Eigen::Vector2f>& uvs,
                    const std::vector<Eigen::Vector3i>& uv_indices,
                    Image<T>& mask, const Eigen::Vector3f& color,
                    int width = 512, int height = 512, int num_threads = -1) {
  std::vector<Eigen::Vector3f> vertex_colors{color};
  std::vector<Eigen::Vector3i> vertex_color_indices(uv_indices.size(),
                                                    Eigen::Vector3i::Zero());
  return RasterizeVertexAttributeToTexture(vertex_colors, vertex_color_indices,
                                           uvs, uv_indices, mask, width, height,
                                           nullptr, num_threads);
}

bool GenerateUvMask(const std::vector<Eigen::Vector2f>& uvs,
                    const std::vector<Eigen::Vector3i>& uv_indices,
                    Image1b& mask, uint8_t color, int width = 512,
                    int height = 512, int num_threads = -1);

template <typename T>
bool FetchVertexAttributeFromTexture(const Image<T>& texture,
                                     const std::vector<Eigen::Vector2f>& uvs,
                                     std::vector<Eigen::Vector3f>& vertex_attrs,
                                     bool use_bilinear = true) {
  vertex_attrs.clear();

  const int w = texture.cols;
  const int h = texture.rows;
  for (const auto& uv : uvs) {
    float x = U2X(uv.x(), w);
    float y = V2Y(uv.y(), h);
    T attr;
    if (use_bilinear) {
      attr = BilinearInterpolation(x, y, texture);
    } else {
      int x_i = static_cast<int>(
          std::clamp(std::round(x), 0.f, static_cast<float>(w - 1)));
      int y_i = static_cast<int>(
          std::clamp(std::round(y), 0.f, static_cast<float>(h - 1)));
      attr = texture.template at<T>(y_i, x_i);
    }
    vertex_attrs.emplace_back(static_cast<float>(attr[0]),
                              static_cast<float>(attr[1]),
                              static_cast<float>(attr[2]));
  }

  return true;
}

}  // namespace ugu
