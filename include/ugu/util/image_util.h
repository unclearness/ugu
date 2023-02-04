/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/image.h"
#include "ugu/image_proc.h"
#include "ugu/util/camera_util.h"

#ifdef UGU_USE_TINYCOLORMAP
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4067)
#endif
#include "tinycolormap.hpp"
#ifdef _WIN32
#pragma warning(pop)
#endif
#endif

namespace ugu {

template <typename T>
T BilinearInterpolation(float x, float y, const ugu::Image<T>& image);

template <typename T>
double BilinearInterpolation(float x, float y, int channel,
                             const ugu::Image<T>& image);
template <typename T>
void UndistortImageOpencv(const ugu::Image<T>& src, ugu::Image<T>* dst,
                          float fx, float fy, float cx, float cy, float k1,
                          float k2, float p1, float p2, float k3 = 0.0f,
                          float k4 = 0.0f, float k5 = 0.0f, float k6 = 0.0f);

void Depth2Gray(const Image1f& depth, Image1b* vis_depth, float min_d = 200.0f,
                float max_d = 1500.0f);

void Normal2Color(const Image3f& normal, Image3b* vis_normal,
                  bool gl_coord = false);

void FaceId2RandomColor(const Image1i& face_id, Image3b* vis_face_id);

void Color2Gray(const Image3b& color, Image1b* gray);

template <typename T>
T Color2Gray(const T& r, const T& g, const T& b);
template <typename T>
typename T::Scalar Color2Gray(const T& color);

void Conv(const Image1b& src, Image1f* dst, float* filter, int kernel_size);
void SobelX(const Image1b& gray, Image1f* gradx, bool scharr = false);
void SobelY(const Image1b& gray, Image1f* grady, bool scharr = false);
void Laplacian(const Image1b& gray, Image1f* laplacian);

struct InvalidSdf {
  static const float kVal;
};

void DistanceTransformL1(const Image1b& mask, Image1f* dist);
void DistanceTransformL1(const Image1b& mask, const Eigen::Vector2i& roi_min,
                         const Eigen::Vector2i& roi_max, Image1f* dist);
void MakeSignedDistanceField(const Image1b& mask, Image1f* dist,
                             bool minmax_normalize, bool use_truncation,
                             float truncation_band);
void MakeSignedDistanceField(const Image1b& mask,
                             const Eigen::Vector2i& roi_min,
                             const Eigen::Vector2i& roi_max, Image1f* dist,
                             bool minmax_normalize, bool use_truncation,
                             float truncation_band);
void SignedDistance2Color(const Image1f& sdf, Image3b* vis_sdf,
                          float min_negative_d, float max_positive_d);

Image3b ColorizeImagePosMap(const ugu::Image3f& srcpos_tex, int32_t src_w,
                            int32_t src_h);

Image3b ColorizePosMap(const ugu::Image3f& pos_tex,
                       Eigen::Vector3f pos_min = Eigen::Vector3f::Constant(
                           std::numeric_limits<float>::max()),
                       Eigen::Vector3f pos_max = Eigen::Vector3f::Constant(
                           std::numeric_limits<float>::lowest()));

Image3b ColorizeBarycentric(const ugu::Image3f& bary_tex);

#ifdef UGU_USE_TINYCOLORMAP
void Depth2Color(
    const Image1f& depth, Image3b* vis_depth, float min_d = 200.0f,
    float max_d = 1500.0f,
    tinycolormap::ColormapType type = tinycolormap::ColormapType::Viridis);
void FaceId2Color(
    const Image1i& face_id, Image3b* vis_face_id, int min_id = 0,
    int max_id = -1,
    tinycolormap::ColormapType type = tinycolormap::ColormapType::Viridis);
#endif

void BoxFilter(const Image1b& src, Image1b* dst, int kernel);
void BoxFilter(const Image1f& src, Image1f* dst, int kernel);
void BoxFilter(const Image3b& src, Image3b* dst, int kernel);
void BoxFilter(const Image3f& src, Image3f* dst, int kernel);

void Erode(const Image1b& src, Image1b* dst, int kernel);
void Dilate(const Image1b& src, Image1b* dst, int kernel);
void Diff(const Image1b& src1, const Image1b& src2, Image1b* dst);
void Not(const Image1b& src, Image1b* dst);

std::vector<Eigen::Vector3f> GenRandomColors(int32_t num, float min_val = 0.f,
                                             float max_val = 255.f,
                                             size_t seed = 0);

template <typename T>
float NormL2(const T& src);

template <typename T>
float NormL2Squared(const T& src);

bool Remap(const Image3f& src, const Image3f& map, const Image1b& mask,
           Image3f& dst, int32_t interp = InterpolationFlags::INTER_LINEAR,
           const Vec3f& bkg_val = Vec3f({0.f, 0.f, 0.f}));

bool AlignChannels(const Image4b& src, Image3b& dst);

void Split(const Image3b& src, std::vector<Image1b>& planes);
void Split(const Image4b& src, std::vector<Image1b>& planes);
void Split(const Image4b& src, Image3b& color, Image1b& mask);

Image4b Merge(const Image3b& color, const Image1b& alpha);
Image3b Merge(const Image1b& a, const Image1b& b, const Image1b& c);

std::vector<uint8_t> JpgData(const Image3b& color);
std::vector<uint8_t> PngData(const Image3b& color);
std::vector<uint8_t> PngData(const Image4b& color);

std::pair<std::vector<Image4b>, std::vector<int>> LoadGif(
    const std::string& path);

Image3b DrawUv(const std::vector<Eigen::Vector2f>& uvs,
               const std::vector<Eigen::Vector3i>& uv_faces,
               const Vec3b& line_col, const Vec3b& bkg_col,
               const Image3b& bkg_img = Image3b(), int32_t tex_w = 512,
               int32_t tex_h = 512, int32_t thickness = 1);

template <typename T>
T BilinearInterpolation(float x, float y, const ugu::Image<T>& image) {
  std::array<int, 2> pos_min = {{0, 0}};
  std::array<int, 2> pos_max = {{0, 0}};
  pos_min[0] = static_cast<int>(std::floor(x));
  pos_min[1] = static_cast<int>(std::floor(y));
  pos_max[0] = pos_min[0] + 1;
  pos_max[1] = pos_min[1] + 1;

  // really need these?
  if (pos_min[0] < 0.0f) {
    pos_min[0] = 0;
  }
  if (pos_min[1] < 0.0f) {
    pos_min[1] = 0;
  }
  if (image.cols <= pos_max[0]) {
    pos_max[0] = image.cols - 1;
  }
  if (image.rows <= pos_max[1]) {
    pos_max[1] = image.rows - 1;
  }

  float local_u = x - pos_min[0];
  float local_v = y - pos_min[1];

  // bilinear interpolation
  typename T::value_type zero(0);
  T color(zero);
  for (int i = 0; i < image.channels(); i++) {
    float colorf =
        (1.0f - local_u) * (1.0f - local_v) *
            image.template at<T>(pos_min[1], pos_min[0])[i] +
        local_u * (1.0f - local_v) *
            image.template at<T>(pos_min[1], pos_max[0])[i] +
        (1.0f - local_u) * local_v *
            image.template at<T>(pos_max[1], pos_min[0])[i] +
        local_u * local_v * image.template at<T>(pos_max[1], pos_max[0])[i];
    color[i] = static_cast<typename T::value_type>(colorf);
  }

  return color;
}

template <typename T>
void UndistortImageOpencv(const ugu::Image<T>& src, ugu::Image<T>* dst,
                          float fx, float fy, float cx, float cy, float k1,
                          float k2, float p1, float p2, float k3, float k4,
                          float k5, float k6) {
  if (dst->rows != src.rows || dst->cols != src.cols) {
    *dst = ugu::Image<T>::zeros(src.rows, src.cols);
  }
  for (int y = 0; y < src.rows; y++) {
    for (int x = 0; x < src.cols; x++) {
      float xf = static_cast<float>(x);
      float yf = static_cast<float>(y);

      // TODO: denser and other interpolation methods.
      UndistortPixelOpencv(&xf, &yf, fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5,
                           k6);
      int nn_x = static_cast<int>(std::round(xf));
      int nn_y = static_cast<int>(std::round(yf));

      if (nn_x < 0 || src.cols <= nn_x || nn_y < 0 || src.rows <= nn_y) {
        continue;
      }

      dst->template at<T>(nn_y, nn_x) = src.template at<T>(y, x);
    }
  }
}

template <typename T>
T Color2Gray(const T& r, const T& g, const T& b) {
  return saturate_cast<T>(0.2989 * r + 0.5870 * g + 0.1140 * b);
}

template <typename T>
typename T::Scalar Color2Gray(const T& color) {
#ifdef UGU_USE_OPENCV
  typename T::Scalar r = color[2];
  typename T::Scalar g = color[1];
  typename T::Scalar b = color[0];
#else
  typename T::Scalar r = color[0];
  typename T::Scalar g = color[1];
  typename T::Scalar b = color[2];
#endif
  return Color2Gray(r, g, b);
}

template <typename T>
float NormL2(const T& src) {
  return static_cast<float>(
      std::sqrt(src[0] * src[0] + src[1] * src[1] + src[2] * src[2]));
}

template <typename T>
float NormL2Squared(const T& src) {
  return static_cast<float>(src[0] * src[0] + src[1] * src[1] +
                            src[2] * src[2]);
}

}  // namespace ugu
