/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "currender/image.h"

#include <algorithm>
#include <array>
#include <random>
#include <unordered_map>

#ifdef CURRENDER_USE_STB
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4100)
#endif
#include "stb/stb_image.h"
#ifdef _WIN32
#pragma warning(pop)
#endif

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
#include "stb/stb_image_write.h"
#ifdef _WIN32
#pragma warning(pop)
#endif
#endif

namespace {

template <typename T>
void BoxFilterCpuIntegral(int width, int height, int channel, int kernel,
                          const T* in_data, double* work_data, T* out_data) {
  const int hk = kernel / 2;
  const int stride = width * channel;

  // initialize the top left pixel
  for (int k = 0; k < channel; k++) {
    std::int64_t base_index = (0 * stride + 0 * channel) + k;
    work_data[base_index] = in_data[base_index];
  }

  // initialize the most left column
  for (int j = 1; j < height; j++) {
    for (int k = 0; k < channel; k++) {
      std::int64_t prev_index = ((j - 1) * stride + 0 * channel) + k;
      std::int64_t base_index = (j * stride + 0 * channel) + k;
      work_data[base_index] = in_data[base_index] + work_data[prev_index];
    }
  }

  // initialize the most top row
  for (int i = 1; i < width; i++) {
    for (int k = 0; k < channel; k++) {
      std::int64_t prev_index = (0 * stride + (i - 1) * channel) + k;
      std::int64_t base_index = (0 * stride + i * channel) + k;
      work_data[base_index] = in_data[base_index] + work_data[prev_index];
    }
  }

  // 1st path: calc integral image
  for (int j = 1; j < height; j++) {
    for (int i = 1; i < width; i++) {
      for (int k = 0; k < channel; k++) {
        std::int64_t upleft_index = ((j - 1) * stride + (i - 1) * channel) + k;
        std::int64_t up_index = ((j - 1) * stride + i * channel) + k;
        std::int64_t left_index = (j * stride + (i - 1) * channel) + k;
        std::int64_t base_index = (j * stride + i * channel) + k;
        work_data[base_index] = -work_data[upleft_index] + work_data[up_index] +
                                work_data[left_index] + in_data[base_index];
      }
    }
  }

  // 2nd path: get average
#if defined(_OPENMP) && defined(CURRENDER_USE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int j = 0; j < height; j++) {
    int up = j > hk ? j - hk - 1 : 0;
    int down = j < height - hk ? j + hk : height - 1;
    int h_len = down - up;
    for (int i = 0; i < width; i++) {
      int left = i > hk ? i - hk - 1 : 0;
      int right = i < width - hk ? i + hk : width - 1;
      int w_len = right - left;
      int adjusted_block_size = h_len * w_len;
      for (int k = 0; k < channel; k++) {
        std::int64_t upleft_index = (up * stride + left * channel) + k;
        std::int64_t upright_index = (up * stride + right * channel) + k;
        std::int64_t downleft_index = (down * stride + left * channel) + k;
        std::int64_t downright_index = (down * stride + right * channel) + k;
        std::int64_t base_index = (j * stride + i * channel) + k;
        out_data[base_index] = static_cast<T>(
            (work_data[downright_index] - work_data[upright_index] -
             work_data[downleft_index] + work_data[upleft_index]) /
            adjusted_block_size);
      }
    }
  }
}

template <typename T>
void BoxFilterCpuIntegral(int width, int height, int channel, int kernel,
                          const T* in_data, T* out_data) {
  std::vector<double> work_data(width * height * channel, 0.0);
  BoxFilterCpuIntegral(width, height, channel, kernel, in_data, &work_data[0],
                       out_data);
}

template <typename T, int N>
void BoxFilterCpuIntegral(const currender::Image<T, N>& src,
                          currender::Image<T, N>* dst, int kernel) {
  assert(src.rows == dst->rows);
  assert(src.cols == dst->cols);
  assert(src.channel() == dst->channel());

  BoxFilterCpuIntegral(src.cols, src.rows, src.channel(), kernel,
                       reinterpret_cast<T*>(src.data),
                       reinterpret_cast<T*>(dst->data));
}

}  // namespace

namespace currender {

void Depth2Gray(const Image1f& depth, Image1b* vis_depth, float min_d,
                float max_d) {
  assert(min_d < max_d);
  assert(vis_depth != nullptr);

  Init(vis_depth, depth.cols, depth.rows);

  float inv_denom = 1.0f / (max_d - min_d);
  for (int y = 0; y < vis_depth->rows; y++) {
    for (int x = 0; x < vis_depth->cols; x++) {
      auto d = at(depth, x, y, 0);

      float norm_color = (d - min_d) * inv_denom;
      norm_color = std::min(std::max(norm_color, 0.0f), 1.0f);

      at(vis_depth, x, y, 0) = static_cast<uint8_t>(norm_color * 255);
    }
  }
}

void Normal2Color(const Image3f& normal, Image3b* vis_normal) {
  assert(vis_normal != nullptr);

  Init(vis_normal, normal.cols, normal.rows);

  // Followed https://en.wikipedia.org/wiki/Normal_mapping
  // X: -1 to +1 :  Red: 0 to 255
  // Y: -1 to +1 :  Green: 0 to 255
  // Z: 0 to -1 :  Blue: 128 to 255
  for (int y = 0; y < vis_normal->rows; y++) {
    for (int x = 0; x < vis_normal->cols; x++) {
      at(vis_normal, x, y, 0) = static_cast<uint8_t>(
          std::round((at(normal, x, y, 0) + 1.0) * 0.5 * 255));
      at(vis_normal, x, y, 1) = static_cast<uint8_t>(
          std::round((at(normal, x, y, 1) + 1.0) * 0.5 * 255));
      at(vis_normal, x, y, 2) =
          static_cast<uint8_t>(std::round(-at(normal, x, y, 2) * 127.0) + 128);
    }
  }
}

void FaceId2RandomColor(const Image1i& face_id, Image3b* vis_face_id) {
  assert(vis_face_id != nullptr);

  Init(vis_face_id, face_id.cols, face_id.rows,
       static_cast<unsigned char>(0));

  std::unordered_map<int, std::array<uint8_t, 3>> id2color;

  for (int y = 0; y < vis_face_id->rows; y++) {
    for (int x = 0; x < vis_face_id->cols; x++) {
      int fid = at(face_id, x, y, 0);
      if (fid < 0) {
        continue;
      }

      std::array<uint8_t, 3> color;
      auto iter = id2color.find(fid);
      if (iter != id2color.end()) {
        color = iter->second;
      } else {
        std::mt19937 mt(fid);
        // stl distribution depends on environment while mt19937 is independent.
        // so simply mod mt19937 value for random color reproducing the same
        // color in different environment.
        color[0] = static_cast<uint8_t>(mt() % 256);
        color[1] = static_cast<uint8_t>(mt() % 256);
        color[2] = static_cast<uint8_t>(mt() % 256);
        id2color[fid] = color;
      }

      at(vis_face_id, x, y, 0) = color[0];
      at(vis_face_id, x, y, 1) = color[1];
      at(vis_face_id, x, y, 2) = color[2];
    }
  }
}

void BoxFilter(const Image1b& src, Image1b* dst, int kernel) {
  BoxFilterCpuIntegral(src, dst, kernel);
}
void BoxFilter(const Image1f& src, Image1f* dst, int kernel) {
  BoxFilterCpuIntegral(src, dst, kernel);
}
void BoxFilter(const Image3b& src, Image3b* dst, int kernel) {
  BoxFilterCpuIntegral(src, dst, kernel);
}
void BoxFilter(const Image3f& src, Image3f* dst, int kernel) {
  BoxFilterCpuIntegral(src, dst, kernel);
}

#ifdef CURRENDER_USE_TINYCOLORMAP
void Depth2Color(const Image1f& depth, Image3b* vis_depth, float min_d,
                 float max_d, tinycolormap::ColormapType type) {
  assert(min_d < max_d);
  assert(vis_depth != nullptr);

  Init(vis_depth, depth.cols, depth.rows);

  float inv_denom = 1.0f / (max_d - min_d);
  for (int y = 0; y < vis_depth->rows; y++) {
    for (int x = 0; x < vis_depth->cols; x++) {
      auto d = at(depth, x, y, 0);

      float norm_color = (d - min_d) * inv_denom;
      norm_color = std::min(std::max(norm_color, 0.0f), 1.0f);

      const tinycolormap::Color& color =
          tinycolormap::GetColor(norm_color, type);

      at(vis_depth, x, y, 0) = static_cast<uint8_t>(color.r() * 255);
      at(vis_depth, x, y, 1) = static_cast<uint8_t>(color.g() * 255);
      at(vis_depth, x, y, 2) = static_cast<uint8_t>(color.b() * 255);
    }
  }
}
void FaceId2Color(const Image1i& face_id, Image3b* vis_face_id, int min_id,
                  int max_id, tinycolormap::ColormapType type) {
  assert(vis_face_id != nullptr);

  Init(vis_face_id, face_id.cols, face_id.rows,
       static_cast<unsigned char>(0));

  if (min_id < 0 || max_id < 0) {
    std::vector<int> valid_ids;
    int* face_id_data = reinterpret_cast<int*>(face_id.data);
    for (int i = 0; i < face_id.cols * face_id.rows; i++) {
      if (i >= 0) {
        valid_ids.push_back(face_id_data[i]);
      }
    }
    if (min_id < 0) {
      min_id = *std::min_element(valid_ids.begin(), valid_ids.end());
    }
    if (max_id < 0) {
      max_id = *std::max_element(valid_ids.begin(), valid_ids.end());
    }
  }

  assert(min_id < max_id);

  float inv_denom = 1.0f / (max_id - min_id);
  for (int y = 0; y < vis_face_id->rows; y++) {
    for (int x = 0; x < vis_face_id->cols; x++) {
      int fid = at(face_id, x, y, 0);
      if (fid < 0) {
        continue;
      }

      float norm_id = (fid - min_id) * inv_denom;
      norm_id = std::min(std::max(norm_id, 0.0f), 1.0f);

      const tinycolormap::Color& color = tinycolormap::GetColor(norm_id, type);

      at(vis_face_id, x, y, 0) = static_cast<uint8_t>(color.r() * 255);
      at(vis_face_id, x, y, 1) = static_cast<uint8_t>(color.g() * 255);
      at(vis_face_id, x, y, 2) = static_cast<uint8_t>(color.b() * 255);
    }
  }
}

#endif

}  // namespace currender
