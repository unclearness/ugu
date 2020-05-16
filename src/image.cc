/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "ugu/image.h"

#include <algorithm>
#include <array>
#include <random>
#include <unordered_map>

#if defined(UGU_USE_STB) && !defined(UGU_USE_OPENCV)
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4100)
#endif
#include "stb_image.h"
#ifdef _WIN32
#pragma warning(pop)
#endif

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
#include "stb_image_write.h"
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
#if defined(_OPENMP) && defined(UGU_USE_OPENMP)
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

template <typename TT, typename T>
inline void BoxFilterCpuIntegral(const T& src, T* dst, int kernel) {
  assert(src.rows == dst->rows);
  assert(src.cols == dst->cols);
  assert(src.channels() == dst->channels());

  BoxFilterCpuIntegral(src.cols, src.rows, src.channels(), kernel,
                       reinterpret_cast<TT*>(src.data),
                       reinterpret_cast<TT*>(dst->data));
}

template <typename T>
void ErodeDilateBase(int width, int height, int channel, int kernel, T from,
                     T to, const T* in_data, T* out_data) {
  const int hk = kernel / 2;
  const int stride = width * channel;

  std::memcpy(out_data, in_data, sizeof(T) * width * height * channel);

#if defined(_OPENMP) && defined(UGU_USE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int j = hk; j < height - hk; j++) {
    for (int i = hk; i < width - hk; i++) {
      // expect only 1 channel
      std::int64_t base_index = (j * stride + i * channel);
      T v0 = in_data[base_index];
      if (v0 != from) {
        continue;
      }
      bool to_update{false};
      for (int jj = -hk; jj < hk + 1; jj++) {
        for (int ii = -hk; ii < hk + 1; ii++) {
          if (ii == 0 && jj == 0) {
            continue;
          }
          std::int64_t index = ((j + jj) * stride + (i + ii) * channel);
          T v1 = in_data[index];
          if (v1 == to) {
            to_update = true;
            break;
          }
        }
        if (to_update) {
          break;
        }
      }

      if (to_update) {
        out_data[base_index] = to;
      }
    }
  }
}

template <typename TT, typename T>
inline void Erode(const T& src, T* dst, int kernel) {
  assert(src.rows == dst->rows);
  assert(src.cols == dst->cols);
  assert(src.channels() == dst->channels());
  assert(src.channels() == 1);

  ErodeDilateBase(src.cols, src.rows, src.channels(), kernel, TT(255), TT(0),
                  reinterpret_cast<TT*>(src.data),
                  reinterpret_cast<TT*>(dst->data));
}

template <typename TT, typename T>
inline void Dilate(const T& src, T* dst, int kernel) {
  assert(src.rows == dst->rows);
  assert(src.cols == dst->cols);
  assert(src.channels() == dst->channels());
  assert(src.channels() == 1);

  ErodeDilateBase(src.cols, src.rows, src.channels(), kernel, TT(0), TT(255),
                  reinterpret_cast<TT*>(src.data),
                  reinterpret_cast<TT*>(dst->data));
}

inline void Diff(int width, int height, int channel, const unsigned char* src1,
                 const unsigned char* src2, unsigned char* dst) {
  std::memcpy(dst, src1, sizeof(unsigned char) * width * height * channel);

  const int stride = width * channel;
#if defined(_OPENMP) && defined(UGU_USE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      for (int k = 0; k < channel; k++) {
        std::int64_t index = (j * stride + i * channel) + k;
        dst[index] = dst[index] - src2[index];
      }
    }
  }
}

}  // namespace

namespace ugu {

void Depth2Gray(const Image1f& depth, Image1b* vis_depth, float min_d,
                float max_d) {
  assert(min_d < max_d);
  assert(vis_depth != nullptr);

  Init(vis_depth, depth.cols, depth.rows, static_cast<unsigned char>(0));

  float inv_denom = 1.0f / (max_d - min_d);
  for (int y = 0; y < vis_depth->rows; y++) {
    for (int x = 0; x < vis_depth->cols; x++) {
      const float& d = depth.at<float>(y, x);

      float norm_color = (d - min_d) * inv_denom;
      norm_color = std::min(std::max(norm_color, 0.0f), 1.0f);

      vis_depth->at<unsigned char>(y, x) =
          static_cast<uint8_t>(norm_color * 255);
    }
  }
}

void Normal2Color(const Image3f& normal, Image3b* vis_normal) {
  assert(vis_normal != nullptr);

  Init(vis_normal, normal.cols, normal.rows, static_cast<unsigned char>(0));

  // Followed https://en.wikipedia.org/wiki/Normal_mapping
  // X: -1 to +1 :  Red: 0 to 255
  // Y: -1 to +1 :  Green: 0 to 255
  // Z: 0 to -1 :  Blue: 128 to 255
  for (int y = 0; y < vis_normal->rows; y++) {
    for (int x = 0; x < vis_normal->cols; x++) {
      Vec3b& vis = vis_normal->at<Vec3b>(y, x);
      const Vec3f& n = normal.at<Vec3f>(y, x);

#ifdef UGU_USE_OPENCV
      // BGR
      vis[2] = static_cast<uint8_t>(std::round((n[0] + 1.0) * 0.5 * 255));
      vis[1] = static_cast<uint8_t>(std::round((n[1] + 1.0) * 0.5 * 255));
      vis[0] = static_cast<uint8_t>(std::round(-n[2] * 127.0) + 128);
#else
      // RGB
      vis[0] = static_cast<uint8_t>(std::round((n[0] + 1.0) * 0.5 * 255));
      vis[1] = static_cast<uint8_t>(std::round((n[1] + 1.0) * 0.5 * 255));
      vis[2] = static_cast<uint8_t>(std::round(-n[2] * 127.0) + 128);
#endif
    }
  }
}

void FaceId2RandomColor(const Image1i& face_id, Image3b* vis_face_id) {
  assert(vis_face_id != nullptr);

  Init(vis_face_id, face_id.cols, face_id.rows, static_cast<unsigned char>(0));

  std::unordered_map<int, std::array<uint8_t, 3>> id2color;

  for (int y = 0; y < vis_face_id->rows; y++) {
    for (int x = 0; x < vis_face_id->cols; x++) {
      int fid = face_id.at<int>(y, x);
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

      Vec3b& vis = vis_face_id->at<Vec3b>(y, x);
#ifdef UGU_USE_OPENCV
      // BGR
      vis[2] = color[0];
      vis[1] = color[1];
      vis[0] = color[2];
#else
      // RGB
      vis = color;
#endif
    }
  }
}

void Color2Gray(const Image3b& color, Image1b* gray) {
#ifdef UGU_USE_OPENCV
  // BGR
  cv::cvtColor(color, *gray, cv::COLOR_BGR2GRAY);
#else
  // RGB
  if (color.rows != gray->rows || color.cols != gray->cols) {
    *gray = Image1b::zeros(color.rows, color.cols);
  }
  for (int y = 0; y < color.rows; y++) {
    for (int x = 0; x < color.cols; x++) {
      const Vec3b& c = color.at<Vec3b>(y, x);
      gray->at<unsigned char>(y, x) = static_cast<unsigned char>(
          0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]);
    }
  }
#endif
}

void Conv(const Image1b& src, Image1f* dst, float* filter, int kernel_size) {
  const int hk = kernel_size / 2;

  // unsigned char* src_data = src.data;
  // float* dst_data = reinterpret_cast<float*>(dst->data);
  dst->setTo(0.0f);
  for (int y = hk; y < src.rows - hk; y++) {
    for (int x = hk; x < src.cols - hk; x++) {
      float& dst_val = dst->at<float>(y, x);
      for (int yy = -hk; yy <= hk; yy++) {
        for (int xx = -hk; xx <= hk; xx++) {
          dst_val += filter[(yy+hk) * kernel_size + (xx+hk)] *
                     src.at<unsigned char>(y + yy, x + xx);
        }
      }
    }
  }
}

void SobelX(const Image1b& gray, Image1f* gradx, bool scharr) {
  float sobelx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  float scharrx[9] = {-3, 0, 3, -10, 0, 10, -3, 0, 3};

  if (scharr) {
    Conv(gray, gradx, scharrx, 3);
  } else {
    Conv(gray, gradx, sobelx, 3);
  }
}

void SobelY(const Image1b& gray, Image1f* grady, bool scharr) {
  float sobely[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
  float scharry[9] = {-3, -10, -3, 0, 0, 0, 3, 10, 3};

  if (scharr) {
    Conv(gray, grady, scharry, 3);
  } else {
    Conv(gray, grady, sobely, 3);
  }
}

void Laplacian(const Image1b& gray, Image1f* laplacian) {
  float l[9] = {0, 1, 0, 1, -4, 1, 0, 1, 0};

  Conv(gray, laplacian, l, 3);
}

void BoxFilter(const Image1b& src, Image1b* dst, int kernel) {
  BoxFilterCpuIntegral<unsigned char>(src, dst, kernel);
}
void BoxFilter(const Image1f& src, Image1f* dst, int kernel) {
  BoxFilterCpuIntegral<float>(src, dst, kernel);
}
void BoxFilter(const Image3b& src, Image3b* dst, int kernel) {
  BoxFilterCpuIntegral<unsigned char>(src, dst, kernel);
}
void BoxFilter(const Image3f& src, Image3f* dst, int kernel) {
  BoxFilterCpuIntegral<float>(src, dst, kernel);
}

void Erode(const Image1b& src, Image1b* dst, int kernel) {
  ::Erode<unsigned char>(src, dst, kernel);
}

void Dilate(const Image1b& src, Image1b* dst, int kernel) {
  ::Dilate<unsigned char>(src, dst, kernel);
}

void Diff(const Image1b& src1, const Image1b& src2, Image1b* dst) {
  assert(src1.rows == dst->rows);
  assert(src1.cols == dst->cols);
  assert(src1.channels() == dst->channels());
  assert(src1.rows == src2.rows);
  assert(src1.cols == src2.cols);
  assert(src1.channels() == src2.channels());
  assert(src1.channels() == 1);

  ::Diff(src1.cols, src1.rows, src1.channels(), src1.data, src2.data,
         dst->data);
}

#ifdef UGU_USE_TINYCOLORMAP
void Depth2Color(const Image1f& depth, Image3b* vis_depth, float min_d,
                 float max_d, tinycolormap::ColormapType type) {
  assert(min_d < max_d);
  assert(vis_depth != nullptr);

  Init(vis_depth, depth.cols, depth.rows, static_cast<unsigned char>(0));

  float inv_denom = 1.0f / (max_d - min_d);
  for (int y = 0; y < vis_depth->rows; y++) {
    for (int x = 0; x < vis_depth->cols; x++) {
      const float& d = depth.at<float>(y, x);

      float norm_color = (d - min_d) * inv_denom;
      norm_color = std::min(std::max(norm_color, 0.0f), 1.0f);

      const tinycolormap::Color& color =
          tinycolormap::GetColor(norm_color, type);

      Vec3b& vis = vis_depth->at<Vec3b>(y, x);
#ifdef UGU_USE_OPENCV
      // BGR
      vis[2] = static_cast<uint8_t>(color.r() * 255);
      vis[1] = static_cast<uint8_t>(color.g() * 255);
      vis[0] = static_cast<uint8_t>(color.b() * 255);
#else
      // RGB
      vis[0] = static_cast<uint8_t>(color.r() * 255);
      vis[1] = static_cast<uint8_t>(color.g() * 255);
      vis[2] = static_cast<uint8_t>(color.b() * 255);
#endif
    }
  }
}
void FaceId2Color(const Image1i& face_id, Image3b* vis_face_id, int min_id,
                  int max_id, tinycolormap::ColormapType type) {
  assert(vis_face_id != nullptr);

  Init(vis_face_id, face_id.cols, face_id.rows, static_cast<unsigned char>(0));

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
      int fid = face_id.at<int>(y, x);
      if (fid < 0) {
        continue;
      }

      float norm_id = (fid - min_id) * inv_denom;
      norm_id = std::min(std::max(norm_id, 0.0f), 1.0f);

      const tinycolormap::Color& color = tinycolormap::GetColor(norm_id, type);

      Vec3b& vis = vis_face_id->at<Vec3b>(y, x);
#ifdef UGU_USE_OPENCV
      // BGR
      vis[2] = static_cast<uint8_t>(color.r() * 255);
      vis[1] = static_cast<uint8_t>(color.g() * 255);
      vis[0] = static_cast<uint8_t>(color.b() * 255);
#else
      // RGB
      vis[0] = static_cast<uint8_t>(color.r() * 255);
      vis[1] = static_cast<uint8_t>(color.g() * 255);
      vis[2] = static_cast<uint8_t>(color.b() * 255);
#endif
    }
  }
}

#endif

}  // namespace ugu
