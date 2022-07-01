/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "ugu/util/image_util.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <random>
#include <unordered_map>

#include "ugu/util/thread_util.h"

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

#include "ugu_stb.h"

#ifdef UGU_USE_OPENCV
#include "opencv2/imgproc.hpp"
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

template <typename VT, typename VT2>
void SplitImpl(ugu::Image<VT>& src, std::vector<ugu::Image<VT2>>& planes) {
  // static_assert(VT::value_type == VT2::value_type);
  planes.resize(src.channels());
  for (auto& p : planes) {
    p = ugu::Image<VT2>::zeros(src.rows, src.cols);
  }

  auto copy_pix = [&](VT& val, const int* index) {
    (void) val;
    VT& src_val = src.template at<VT>(index[1], index[0]);
    for (int i = 0; i < src.channels(); i++) {
      ugu::Image<VT2>& p = planes[i];
      p.template at<VT2>(index[1], index[0])[0] = src_val[i];
    }
  };

  src.template forEach<VT>(copy_pix);
}

template <typename T>
ugu::Image<T> MergeByteImpl(const std::vector<ugu::Image1b>& planes) {
  ugu::Image<T> merged = ugu::Image<T>::zeros(planes[0].rows, planes[0].cols);
  auto f = [=](T& val, const int* index) {
    for (int i = 0; i < merged.channels(); i++) {
      const auto& p = planes[i];
      val[i] = p.at<uint8_t>(index[1], index[0]);
    }
  };
  merged.template forEach<T>(f);
  return merged;
}

typedef struct {
  int last_pos;
  void* context;
} custom_stbi_mem_context;

static void custom_stbi_write_mem(void* context, void* data, int size) {
  custom_stbi_mem_context* c = (custom_stbi_mem_context*)context;
  char* dst = (char*)c->context;
  char* src = (char*)data;
  int cur_pos = c->last_pos;
  for (int i = 0; i < size; i++) {
    dst[cur_pos++] = src[i];
  }
  c->last_pos = cur_pos;
}

bool CompressedDataImpl(
    int width, int height, int channels, uint8_t* data,
    std::vector<uint8_t>& compressed_data,
    std::function<int(int, int, int, uint8_t*, custom_stbi_mem_context&)> cb) {
#ifdef UGU_USE_STB
  size_t bitmap_bytes = static_cast<size_t>(width) *
                        static_cast<size_t>(height) *
                        static_cast<size_t>(channels) * sizeof(uint8_t);

  compressed_data.resize(bitmap_bytes);  // Safe resize
  // https://github.com/nothings/stb/issues/1132#issuecomment-850628100
  custom_stbi_mem_context context;
  context.last_pos = 0;
  context.context = (void*)compressed_data.data();

  int result = cb(width, height, channels, data, context);

  if (result != 1) {
    ugu::LOGE("failed to write data to memory\n");
    compressed_data.clear();
    return false;
  }
  compressed_data.resize(context.last_pos);
  return true;
#else
  ugu::LOGE("not suppored in this configration\n");
  return false;
#endif
}

bool JpgDataImpl(int width, int height, int channels, uint8_t* data,
                 std::vector<uint8_t>& jpg_data) {
#ifdef UGU_USE_STB
  auto cb = [](int width, int height, int channels, uint8_t* data,
               custom_stbi_mem_context& context) {
    const int max_quality{100};
    return stbi_write_jpg_to_func(custom_stbi_write_mem, &context, width,
                                  height, channels, data, max_quality);
  };
  return CompressedDataImpl(width, height, channels, data, jpg_data, cb);
#else
  ugu::LOGE("not suppored in this configration\n");
  return false;
#endif
}

bool PngDataImpl(int width, int height, int channels, uint8_t* data,
                 std::vector<uint8_t>& png_data) {
#ifdef UGU_USE_STB
  auto cb = [](int width, int height, int channels, uint8_t* data,
               custom_stbi_mem_context& context) {
    int stride = width * channels * sizeof(uint8_t);
    return stbi_write_png_to_func(custom_stbi_write_mem, &context, width,
                                  height, channels, data, stride);
  };
  return CompressedDataImpl(width, height, channels, data, png_data, cb);
#else
  ugu::LOGE("not suppored in this configration\n");
  return false;
#endif
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

  const unsigned char zero = static_cast<unsigned char>(0);
  Init(vis_normal, normal.cols, normal.rows, zero);

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

const float InvalidSdf::kVal = std::numeric_limits<float>::lowest();

void DistanceTransformL1(const Image1b& mask, Image1f* dist) {
  return DistanceTransformL1(mask, Eigen::Vector2i(0, 0),
                             Eigen::Vector2i(mask.cols - 1, mask.rows - 1),
                             dist);
}

void DistanceTransformL1(const Image1b& mask, const Eigen::Vector2i& roi_min,
                         const Eigen::Vector2i& roi_max, Image1f* dist) {
  *dist = Image1f::zeros(mask.rows, mask.cols);

  // init inifinite inside mask
  for (int y = roi_min.y(); y <= roi_max.y(); y++) {
    for (int x = roi_min.x(); x <= roi_max.x(); x++) {
      if (mask.at<unsigned char>(y, x) != 255) {
        continue;
      }
      dist->at<float>(y, x) = std::numeric_limits<float>::max();
    }
  }

  // forward path
  for (int y = roi_min.y() + 1; y <= roi_max.y(); y++) {
    float up = dist->at<float>(y - 1, roi_min.x());
    if (up < std::numeric_limits<float>::max()) {
      dist->at<float>(y, roi_min.x()) =
          std::min(up + 1.0f, dist->at<float>(y, roi_min.x()));
    }
  }
  for (int x = roi_min.x() + 1; x <= roi_max.x(); x++) {
    float left = dist->at<float>(roi_min.y(), x - 1);
    if (left < std::numeric_limits<float>::max()) {
      dist->at<float>(roi_min.y(), x) =
          std::min(left + 1.0f, dist->at<float>(roi_min.y(), x));
    }
  }
  for (int y = roi_min.y() + 1; y <= roi_max.y(); y++) {
    for (int x = roi_min.x() + 1; x <= roi_max.x(); x++) {
      float up = dist->at<float>(y - 1, x);
      float left = dist->at<float>(y, x - 1);
      float min_dist = std::min(up, left);
      if (min_dist < std::numeric_limits<float>::max()) {
        dist->at<float>(y, x) =
            std::min(min_dist + 1.0f, dist->at<float>(y, x));
      }
    }
  }

  // backward path
  for (int y = roi_max.y() - 1; roi_min.y() <= y; y--) {
    float down = dist->at<float>(y + 1, roi_max.x());
    if (down < std::numeric_limits<float>::max()) {
      dist->at<float>(y, roi_max.x()) =
          std::min(down + 1.0f, dist->at<float>(y, roi_max.x()));
    }
  }
  for (int x = roi_max.x() - 1; roi_min.x() <= x; x--) {
    float right = dist->at<float>(roi_max.y(), x + 1);
    if (right < std::numeric_limits<float>::max()) {
      dist->at<float>(roi_max.y(), x) =
          std::min(right + 1.0f, dist->at<float>(roi_max.y(), x));
    }
  }
  for (int y = roi_max.y() - 1; roi_min.y() <= y; y--) {
    for (int x = roi_max.x() - 1; roi_min.x() <= x; x--) {
      float down = dist->at<float>(y + 1, x);
      float right = dist->at<float>(y, x + 1);
      float min_dist = std::min(down, right);
      if (min_dist < std::numeric_limits<float>::max()) {
        dist->at<float>(y, x) =
            std::min(min_dist + 1.0f, dist->at<float>(y, x));
      }
    }
  }
}

void MakeSignedDistanceField(const Image1b& mask, Image1f* dist,
                             bool minmax_normalize, bool use_truncation,
                             float truncation_band) {
  return MakeSignedDistanceField(mask, Eigen::Vector2i(0, 0),
                                 Eigen::Vector2i(mask.cols - 1, mask.rows - 1),
                                 dist, minmax_normalize, use_truncation,
                                 truncation_band);
}

void MakeSignedDistanceField(const Image1b& mask,
                             const Eigen::Vector2i& roi_min,
                             const Eigen::Vector2i& roi_max, Image1f* dist,
                             bool minmax_normalize, bool use_truncation,
                             float truncation_band) {
  Image1f* negative_dist = dist;
  DistanceTransformL1(mask, roi_min, roi_max, negative_dist);
  for (int y = roi_min.y(); y <= roi_max.y(); y++) {
    for (int x = roi_min.x(); x <= roi_max.x(); x++) {
      if (negative_dist->at<float>(y, x) > 0) {
        negative_dist->at<float>(y, x) *= -1;
      }
    }
  }

  Image1b inv_mask = Image1b::zeros(mask.rows, mask.cols);
  mask.copyTo(inv_mask);
  for (int y = roi_min.y(); y <= roi_max.y(); y++) {
    for (int x = roi_min.x(); x <= roi_max.x(); x++) {
      auto& i = inv_mask.at<unsigned char>(y, x);
      if (i == 255) {
        i = 0;
      } else {
        i = 255;
      }
    }
  }

  Image1f positive_dist;
  DistanceTransformL1(inv_mask, roi_min, roi_max, &positive_dist);
  for (int y = roi_min.y(); y <= roi_max.y(); y++) {
    for (int x = roi_min.x(); x <= roi_max.x(); x++) {
      if (inv_mask.at<unsigned char>(y, x) == 255) {
        dist->at<float>(y, x) = positive_dist.at<float>(y, x);
      }
    }
  }

  if (minmax_normalize) {
    // Outside of roi is set to 0, so does not affect min/max
    double max_dist, min_dist;
    ugu::minMaxLoc(*dist, &min_dist, &max_dist);

    float abs_max =
        static_cast<float>(std::max(std::abs(max_dist), std::abs(min_dist)));

    if (abs_max > std::numeric_limits<float>::min()) {
      float norm_factor = 1.0f / abs_max;

      for (int y = roi_min.y(); y <= roi_max.y(); y++) {
        for (int x = roi_min.x(); x <= roi_max.x(); x++) {
          dist->at<float>(y, x) *= norm_factor;
        }
      }
    }
  }

  // truncation process same to KinectFusion
  if (use_truncation) {
    for (int y = roi_min.y(); y <= roi_max.y(); y++) {
      for (int x = roi_min.x(); x <= roi_max.x(); x++) {
        float& d = dist->at<float>(y, x);
        if (-truncation_band >= d) {
          d = InvalidSdf::kVal;
        } else {
          d = std::min(1.0f, d / truncation_band);
        }
      }
    }
  }
}

void SignedDistance2Color(const Image1f& sdf, Image3b* vis_sdf,
                          float min_negative_d, float max_positive_d) {
  assert(min_negative_d < 0);
  assert(0 < max_positive_d);
  assert(vis_sdf != nullptr);

  *vis_sdf = Image3b::zeros(sdf.rows, sdf.cols);

  for (int y = 0; y < vis_sdf->rows; y++) {
    for (int x = 0; x < vis_sdf->cols; x++) {
      const float& d = sdf.at<float>(y, x);
      auto& c = vis_sdf->at<Vec3b>(y, x);
      if (d > 0) {
        float norm_inv_dist = (max_positive_d - d) / max_positive_d;
        norm_inv_dist = std::min(std::max(norm_inv_dist, 0.0f), 1.0f);
        c[0] = static_cast<uint8_t>(255);
        c[1] = static_cast<uint8_t>(255 * norm_inv_dist);
        c[2] = static_cast<uint8_t>(255 * norm_inv_dist);

      } else {
        float norm_inv_dist = (d - min_negative_d) / (-min_negative_d);
        norm_inv_dist = std::min(std::max(norm_inv_dist, 0.0f), 1.0f);
        c[0] = static_cast<uint8_t>(255 * norm_inv_dist);
        c[1] = static_cast<uint8_t>(255 * norm_inv_dist);
        c[2] = static_cast<uint8_t>(255);
      }
    }
  }
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
          dst_val += filter[(yy + hk) * kernel_size + (xx + hk)] *
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

void Not(const Image1b& src, Image1b* dst) {
  if (src.rows != dst->rows || src.cols != dst->cols) {
    *dst = Image1b::zeros(src.rows, src.cols);
  }
  for (int y = 0; y < src.rows; y++) {
    for (int x = 0; x < src.cols; x++) {
      const auto& s = src.at<unsigned char>(y, x);
      dst->at<unsigned char>(y, x) = ~s;
    }
  }
}

std::vector<Eigen::Vector3f> GenRandomColors(int32_t num, float min_val,
                                             float max_val, size_t seed) {
  std::vector<Eigen::Vector3f> colors;
  std::uniform_real_distribution<float> dist(min_val, max_val);
  std::default_random_engine engine(static_cast<unsigned int>(seed));

  for (int32_t i = 0; i < num; i++) {
    colors.emplace_back(dist(engine), dist(engine), dist(engine));
  }
  return colors;
}

bool Remap(const Image3f& src, const Image3f& map, const Image1b& mask,
           Image3f& dst, int32_t interp, const ugu::Vec3f& bkg_val) {
  if (interp != InterpolationFlags::INTER_LINEAR &&
      interp != InterpolationFlags::INTER_NEAREST) {
    ugu::LOGE("interp is not supported\n");
    return false;
  }

  if (map.rows != dst.rows || map.cols != dst.cols) {
    dst = Image3f::zeros(map.rows, map.cols);
    for (int j = 0; j < dst.rows; j++) {
      for (int i = 0; i < dst.cols; i++) {
        dst.at<ugu::Vec3f>(j, i) = bkg_val;
      }
    }
  }

  auto loop_body = [&](int j) {
    for (int i = 0; i < map.cols; i++) {
      if (mask.at<uint8_t>(j, i) == 0) {
        continue;
      }

      const Vec3f& spos = map.at<Vec3f>(j, i);

      Vec3f& src_color = dst.at<Vec3f>(j, i);
      if (interp == InterpolationFlags::INTER_LINEAR) {
        src_color = ugu::BilinearInterpolation(spos[0], spos[1], src);
      } else if (interp == InterpolationFlags::INTER_NEAREST) {
        src_color =
            src.at<ugu::Vec3f>(static_cast<int32_t>(std::round(spos[1])),
                               static_cast<int32_t>(std::round(spos[0])));
      } else {
        ugu::LOGE("interp is not supported\n");
      }
    }
  };
  ugu::parallel_for(0, map.rows, loop_body);

  return true;
}

ugu::Image3b ColorizeImagePosMap(const ugu::Image3f& srcpos_tex, int32_t src_w,
                                 int32_t src_h) {
  ugu::Image3b srcpos_tex_vis =
      ugu::Image3b::zeros(srcpos_tex.rows, srcpos_tex.cols);
  for (int y = 0; y < srcpos_tex_vis.rows; y++) {
    for (int x = 0; x < srcpos_tex_vis.cols; x++) {
      auto& color = srcpos_tex_vis.at<ugu::Vec3b>(y, x);
      const auto& spos = srcpos_tex.at<ugu::Vec3f>(y, x);

      color[0] = static_cast<uint8_t>(spos[0] / src_w * 255);

      color[1] = static_cast<uint8_t>(spos[1] / src_h * 255);
    }
  }
  return srcpos_tex_vis;
}

ugu::Image3b ColorizePosMap(const ugu::Image3f& pos_tex,
                            Eigen::Vector3f pos_min, Eigen::Vector3f pos_max) {
  ugu::Image3b pos_tex_vis = ugu::Image3b::zeros(pos_tex.rows, pos_tex.cols);

  if (pos_max[0] <= pos_min[0] || pos_max[1] <= pos_min[1] ||
      pos_max[2] <= pos_min[2]) {
    for (int y = 0; y < pos_tex_vis.rows; y++) {
      for (int x = 0; x < pos_tex_vis.cols; x++) {
        const auto& pos = pos_tex.at<ugu::Vec3f>(y, x);
        for (int c = 0; c < 3; c++) {
          if (pos[c] < pos_min[c]) {
            pos_min[c] = pos[c];
          }
          if (pos_max[c] < pos[c]) {
            pos_max[c] = pos[c];
          }
        }
      }
    }
  }

  for (int y = 0; y < pos_tex_vis.rows; y++) {
    for (int x = 0; x < pos_tex_vis.cols; x++) {
      auto& color = pos_tex_vis.at<ugu::Vec3b>(y, x);
      const auto& pos = pos_tex.at<ugu::Vec3f>(y, x);
      for (int c = 0; c < 3; c++) {
        color[c] = static_cast<uint8_t>((pos[c] - pos_min[c]) /
                                        (pos_max[c] - pos_min[c]) * 255);
      }
    }
  }
  return pos_tex_vis;
}

ugu::Image3b ColorizeBarycentric(const ugu::Image3f& bary_tex) {
  ugu::Image3b bary_tex_vis = ugu::Image3b::zeros(bary_tex.rows, bary_tex.cols);
  ugu::ConvertTo(bary_tex, &bary_tex_vis, 255);
  for (int y = 0; y < bary_tex_vis.rows; y++) {
    for (int x = 0; x < bary_tex_vis.cols; x++) {
      bary_tex_vis.at<ugu::Vec3b>(y, x)[2] = 0;
    }
  }
  return bary_tex_vis;
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

bool AlignChannels(const Image4b& src, Image3b& dst) {
  dst = Image3b::zeros(src.rows, src.cols);
  auto f = [=](Vec3b& val, const int* index) {
    const auto& src_val = src.at<Vec4b>(index[1], index[0]);
    val[0] = src_val[0];
    val[1] = src_val[1];
    val[2] = src_val[2];
  };
  dst.forEach<Vec3b>(f);
  return true;
}

void Split(const Image3b& src, std::vector<Image1b>& planes) {
  SplitImpl(const_cast<Image3b&>(src), planes);
}

void Split(const Image4b& src, std::vector<Image1b>& planes) {
  SplitImpl(const_cast<Image4b&>(src), planes);
}

void Split(const Image4b& src, Image3b& color, Image1b& mask) {
  std::vector<Image1b> planes;
  SplitImpl(const_cast<Image4b&>(src), planes);
  mask = planes[3];
  color = Merge(planes[0], planes[1], planes[2]);
}

Image4b Merge(const Image3b& color, const Image1b& alpha) {
  Image4b with_alpha = Image4b::zeros(color.rows, color.cols);
  std::vector<Image1b> planes;
  Split(color, planes);
  planes.push_back(alpha);
  return MergeByteImpl<Vec4b>(planes);
}

Image3b Merge(const Image1b& a, const Image1b& b, const Image1b& c) {
  std::vector<Image1b> planes = {a, b, c};
  return MergeByteImpl<Vec3b>(planes);
}

std::vector<uint8_t> JpgData(const Image3b& color) {
  std::vector<uint8_t> compressed_data;
  JpgDataImpl(color.cols, color.rows, color.channels(), color.data,
              compressed_data);
  return compressed_data;
}

std::vector<uint8_t> PngData(const Image3b& color) {
  std::vector<uint8_t> compressed_data;
  PngDataImpl(color.cols, color.rows, color.channels(), color.data,
              compressed_data);
  return compressed_data;
}

std::vector<uint8_t> PngData(const Image4b& color) {
  std::vector<uint8_t> compressed_data;
  PngDataImpl(color.cols, color.rows, color.channels(), color.data,
              compressed_data);
  return compressed_data;
}

std::pair<std::vector<Image4b>, std::vector<int>> LoadGif(
    const std::string& path) {
  std::vector<Image4b> images;
  std::vector<int> delays;
#ifdef UGU_USE_STB
  int x = -1;
  int y = -1;
  int frames = -1;

  std::vector<unsigned char> data = LoadGif(path, x, y, frames, delays);

  if (data.empty() || x <= 0 || y <= 0 || frames <= 0 || delays.empty()) {
    return {{}, {}};
  }

  assert(frames == delays.size());

  images.resize(frames);

  const size_t pix_per_image =
      static_cast<size_t>(x) * static_cast<size_t>(y) * 4;
  const size_t bytes_per_image = pix_per_image * sizeof(unsigned char);

  for (int i = 0; i < frames; i++) {
    images[i] = Image4b::zeros(y, x);
    std::memcpy(images[i].data, data.data() + i * bytes_per_image,
                bytes_per_image);
  }
#endif
  return {images, delays};
}

}  // namespace ugu
