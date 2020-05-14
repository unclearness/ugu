/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <cstdint>
#include <cstring>

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "ugu/common.h"

#if defined(UGU_USE_STB) && !defined(UGU_USE_OPENCV)
#include "stb_image.h"
#include "stb_image_write.h"
#endif

#if defined(UGU_USE_LODEPNG) && !defined(UGU_USE_OPENCV)
#include "lodepng.h"
#endif

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

#ifdef UGU_USE_OPENCV
#include "opencv2/imgcodecs.hpp"
#endif

namespace ugu {

#ifdef UGU_USE_OPENCV

template <typename T>
using Image = cv::Mat_<T>;

using Image1b = cv::Mat1b;
using Image3b = cv::Mat3b;
using Image1w = cv::Mat1w;
using Image1i = cv::Mat1i;
using Image1f = cv::Mat1f;
using Image3f = cv::Mat3f;

using Vec1v = unsigned char;
using Vec1f = float;
using Vec1i = int;
using Vec1w = std::uint16_t;
using Vec3f = cv::Vec3f;
using Vec3b = cv::Vec3b;

using Point = cv::Point;

using ImreadModes = cv::ImreadModes;

template <typename T>
inline bool imwrite(const std::string& filename, const T& img,
                    const std::vector<int>& params = std::vector<int>()) {
  return cv::imwrite(filename, img, params);
}

template <typename T>
inline T imread(const std::string& filename,
                int flags = ImreadModes::IMREAD_COLOR) {
  return cv::imread(filename, flags);
}

template <typename T, typename TT>
inline void Init(Image<T>* image, int width, int height, TT val) {
  if (image->cols == width && image->rows == height) {
    image->setTo(val);
  } else {
    if (val == TT(0)) {
      *image = Image<T>::zeros(height, width);
    } else {
      *image = Image<T>::ones(height, width) * val;
    }
  }
}

template <typename T, typename TT>
bool ConvertTo(const Image<T>& src, Image<TT>* dst, float scale = 1.0f) {
  src.convertTo(*dst, dst->type(), scale);

  return true;
}

inline void minMaxLoc(const cv::InputArray& src, double* minVal,
                      double* maxVal = 0, Point* minLoc = 0,
                      Point* maxLoc = 0) {
  cv::minMaxLoc(src, minVal, maxVal, minLoc, maxLoc);
}

#else

template <typename TT, int N>
using Vec_ = std::array<TT, N>;

using Vec1f = Vec_<float, 1>;
using Vec1i = Vec_<int, 1>;
using Vec1w = Vec_<std::uint16_t, 1>;
using Vec1b = Vec_<unsigned char, 1>;
using Vec3b = Vec_<unsigned char, 3>;
using Vec3f = Vec_<float, 3>;

template <typename TT, int N>
using Point_ = Vec_<TT, N>;

template <typename T>
class Image {
 private:
  int bit_depth_{sizeof(typename T::value_type)};
  int channels_{std::tuple_size<T>::value};
  int width_{-1};
  int height_{-1};
  std::shared_ptr<std::vector<T> > data_{nullptr};

  void Init(int width, int height) {
    if (width < 1 || height < 1) {
      LOGE("wrong width or height\n");
      return;
    }

    width_ = width;
    height_ = height;
    data_->resize(height_ * width_);
    data = reinterpret_cast<unsigned char*>(data_->data());
    rows = height;
    cols = width;

    channels_ = static_cast<int>((*data_)[0].size());
  }

  void Init(int width, int height, typename T::value_type val) {
    if (width < 1 || height < 1) {
      LOGE("wrong width or height\n");
      return;
    }

    Init(width, height);

    this->setTo(val);
  }

 public:
  Image() : data_(new std::vector<T>) {}
  ~Image() {}
  int channels() const { return channels_; }

  int rows;
  int cols;
  unsigned char* data;

  bool empty() const {
    if (width_ < 0 || height_ < 0 || data_->empty()) {
      return true;
    }
    return false;
  }

  template <typename TT>
  TT& at(int y, int x) {
    return *(reinterpret_cast<TT*>(data_->data()) + (y * cols + x));
  }
  template <typename TT>
  const TT& at(int y, int x) const {
    return *(reinterpret_cast<TT*>(data_->data()) + (y * cols + x));
  }

  void setTo(typename T::value_type val) {
    for (auto& v : *data_) {
      for (auto& vv : v) {
        vv = val;
      }
    }
  }

  static Image<T> zeros(int height, int width) {
    Image<T> tmp;
    tmp.Init(width, height, static_cast<typename T::value_type>(0));
    return tmp;
  }

#ifdef UGU_USE_STB
  bool Load(const std::string& path) {
    unsigned char* in_pixels_tmp;
    int width;
    int height;
    int bpp;

    if (bit_depth_ == 2) {
      in_pixels_tmp = reinterpret_cast<unsigned char*>(
          stbi_load_16(path.c_str(), &width, &height, &bpp, channels_));
    } else if (bit_depth_ == 1) {
      in_pixels_tmp = stbi_load(path.c_str(), &width, &height, &bpp, channels_);
    } else {
      LOGE("Load() for bit_depth %d and channel %d is not supported\n",
           bit_depth_, channels_);
      return false;
    }

    if (bpp != channels_) {
      delete in_pixels_tmp;
      LOGE("desired channel %d, actual %d\n", channels_, bpp);
      return false;
    }

    Init(width, height);

    std::memcpy(data_->data(), in_pixels_tmp, sizeof(T) * width_ * height_);
    delete in_pixels_tmp;

    return true;
  }

#ifdef UGU_USE_LODEPNG
  // https://github.com/lvandeve/lodepng/issues/74#issuecomment-405049566
  bool WritePng16Bit1Channel(const std::string& path) const {
    if (bit_depth_ != 2 || channels_ != 1) {
      LOGE("WritePng16Bit1Channel invalid bit_depth %d or channel %d\n",
           bit_depth_, channels_);
      return false;
    }
    std::vector<unsigned char> data_8bit;
    data_8bit.resize(width_ * height_ * 2);  // 2 bytes per pixel
    const int kMostMask = 0b1111111100000000;
    const int kLeastMask = ~kMostMask;
    for (int y = 0; y < height_; y++) {
      for (int x = 0; x < width_; x++) {
        std::uint16_t d = this->at<std::uint16_t>(y, x);  // At(*this, x, y, 0);
        data_8bit[2 * width_ * y + 2 * x + 0] = static_cast<unsigned char>(
            (d & kMostMask) >> 8);  // most significant
        data_8bit[2 * width_ * y + 2 * x + 1] =
            static_cast<unsigned char>(d & kLeastMask);  // least significant
      }
    }
    unsigned error = lodepng::encode(
        path, data_8bit, width_, height_, LCT_GREY,
        16);  // note that the LCT_GREY and 16 parameters are of the std::vector
              // we filled in, lodepng will choose its output format itself
              // based on the colors it gets, it will choose 16-bit greyscale in
              // this case though because of the pixel data we feed it
    if (error != 0) {
      LOGE("lodepng::encode errorcode: %d\n", error);
      return false;
    }
    return true;
  }
#endif

  bool WritePng(const std::string& path) const {
#ifdef UGU_USE_LODEPNG
    if (bit_depth_ == 2 && channels_ == 1) {
      return WritePng16Bit1Channel(path);
    }
#endif

    if (bit_depth_ != 1) {
      LOGE("1 byte per channel is required to save by stb_image: actual %d\n",
           bit_depth_);
      return false;
    }

    if (width_ < 0 || height_ < 0) {
      LOGE("image is empty\n");
      return false;
    }

    int ret = stbi_write_png(path.c_str(), width_, height_, channels_,
                             data_->data(), width_ * sizeof(T));
    return ret != 0;
  }

  bool WriteJpg(const std::string& path) const {
    if (bit_depth_ != 1) {
      LOGE("1 byte per channel is required to save by stb_image: actual %d\n",
           bit_depth_);
      return false;
    }

    if (width_ < 0 || height_ < 0) {
      LOGE("image is empty\n");
      return false;
    }

    if (channels_ > 3) {
      LOGW("alpha channel is ignored to save as .jpg. channels(): %d\n",
           channels_);
    }

    // JPEG does ignore alpha channels in input data; quality is between 1
    // and 100. Higher quality looks better but results in a bigger image.
    const int max_quality{100};

    int ret = stbi_write_jpg(path.c_str(), width_, height_, channels_,
                             data_->data(), max_quality);
    return ret != 0;
  }
#else
  bool Load(const std::string& path) {
    (void)path;
    LOGE("can't load image with this configuration\n");
    return false;
  }

  bool WritePng(const std::string& path) const {
    (void)path;
    LOGE("can't write image with this configuration\n");
    return false;
  }

  bool WriteJpg(const std::string& path) const {
    (void)path;
    LOGE("can't write image with this configuration\n");
    return false;
  }
#endif

  void copyTo(Image<T>& dst) const {  // NOLINT
    if (dst.cols != cols || dst.rows != rows) {
      dst = Image<T>::zeros(rows, cols);
    }
    std::memcpy(dst.data_->data(), data_->data(), sizeof(T) * rows * cols);
  }
};

using Image1b = Image<Vec1b>;  // For gray image.
using Image3b = Image<Vec3b>;  // For color image. RGB order.
using Image1w = Image<Vec1w>;  // For depth image with 16 bit (unsigned
                               // short) mm-scale format
using Image1i = Image<Vec1i>;  // For face visibility. face id is within int32_t
using Image1f = Image<Vec1f>;  // For depth image with any scale
using Image3f = Image<Vec3f>;  // For normal or point cloud. XYZ order.

enum ImreadModes {
  IMREAD_UNCHANGED = -1,
  IMREAD_GRAYSCALE = 0,
  IMREAD_COLOR = 1,
  IMREAD_ANYDEPTH = 2,
  IMREAD_ANYCOLOR = 4,
  IMREAD_LOAD_GDAL = 8,
  IMREAD_REDUCED_GRAYSCALE_2 = 16,
  IMREAD_REDUCED_COLOR_2 = 17,
  IMREAD_REDUCED_GRAYSCALE_4 = 32,
  IMREAD_REDUCED_COLOR_4 = 33,
  IMREAD_REDUCED_GRAYSCALE_8 = 64,
  IMREAD_REDUCED_COLOR_8 = 65,
  IMREAD_IGNORE_ORIENTATION = 128,
};

template <typename T>
inline void Init(Image<T>* image, int width, int height) {
  if (image->cols != width || image->rows != height) {
    *image = Image<T>::zeros(height, width);
  }
}

template <typename T>
inline void Init(Image<T>* image, int width, int height,
                 typename T::value_type val) {
  if (image->cols != width || image->rows != height) {
    *image = Image<T>::zeros(height, width);
  }
  image->setTo(val);
}

template <typename T>
inline bool imwrite(const std::string& filename, const T& img,
                    const std::vector<int>& params = std::vector<int>()) {
  (void)params;
  if (filename.size() < 4) {
    return false;
  }

  size_t ext_i = filename.find_last_of(".");
  std::string extname = filename.substr(ext_i, filename.size() - ext_i);
  if (extname == ".png" || extname == ".PNG") {
    return img.WritePng(filename);
  } else if (extname == ".jpg" || extname == ".jpeg" || extname == ".JPG" ||
             extname == ".JPEG") {
    return img.WriteJpg(filename);
  }

  LOGE(
      "acceptable extention is .png, .jpg or .jpeg. this extention is not "
      "supported: %s\n",
      filename.c_str());
  return false;
}

template <typename T>
inline T imread(const std::string& filename,
                int flags = ImreadModes::IMREAD_COLOR) {
  (void)flags;
  T loaded;
  loaded.Load(filename);
  return loaded;
}

template <typename T, typename TT>
bool ConvertTo(const Image<T>& src, Image<TT>* dst, float scale = 1.0f) {
  if (src.channels() != dst->channels()) {
    LOGE("ConvertTo failed src channel %d, dst channel %d\n", src.channels(),
         dst->channels());
    return false;
  }

  Init(dst, src.cols, src.rows);

  for (int y = 0; y < src.rows; y++) {
    for (int x = 0; x < src.cols; x++) {
      for (int c = 0; c < dst->channels(); c++) {
        dst->template at<TT>(y, x)[c] = static_cast<typename TT::value_type>(
            scale * src.template at<T>(y, x)[c]);
      }
    }
  }

  return true;
}

class Point2i {
 public:
  int x = -1;
  int y = -1;
};

using Point = Point2i;

template <typename T>
void minMaxLoc(const Image<T>& src, double* minVal, double* maxVal = nullptr,
               Point* minLoc = nullptr, Point* maxLoc = nullptr) {
  if (src.channels() != 1 || minVal == nullptr) {
    return;
  }

  double minVal_ = std::numeric_limits<double>::max();
  double maxVal_ = std::numeric_limits<double>::lowest();
  Point minLoc_, maxLoc_;

  for (int y = 0; y < src.rows; y++) {
    for (int x = 0; x < src.cols; x++) {
      for (int c = 0; c < src.channels(); c++) {
        const auto& val = src.template at<T>(y, x)[c];
        if (val < minVal_) {
          minVal_ = val;
          minLoc_.x = x;
          minLoc_.y = y;
        }
        if (val > maxVal_) {
          maxVal_ = val;
          maxLoc_.x = x;
          maxLoc_.y = y;
        }
      }
    }
  }

  if (minVal != nullptr) {
    *minVal = minVal_;
  }
  if (maxVal != nullptr) {
    *maxVal = maxVal_;
  }
  if (minLoc != nullptr) {
    *minLoc = minLoc_;
  }
  if (maxLoc != nullptr) {
    *maxLoc = maxLoc_;
  }
}

#endif

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
  T color;
  for (int i = 0; i < image.channels(); i++) {
    float colorf =
        (1.0f - local_u) * (1.0f - local_v) *
            image.at<T>(pos_min[1], pos_min[0])[i] +
        local_u * (1.0f - local_v) * image.at<T>(pos_max[1], pos_min[0])[i] +
        (1.0f - local_u) * local_v * image.at<T>(pos_min[1], pos_max[0])[i] +
        local_u * local_v * image.at<T>(pos_max[1], pos_max[0])[i];
    color[i] = static_cast<typename T::value_type>(colorf);
  }

  return color;
}

void Depth2Gray(const Image1f& depth, Image1b* vis_depth, float min_d = 200.0f,
                float max_d = 1500.0f);

void Normal2Color(const Image3f& normal, Image3b* vis_normal);

void FaceId2RandomColor(const Image1i& face_id, Image3b* vis_face_id);

void Color2Gray(const Image3b& color, Image1b* gray);

void Conv(const Image1b& src, Image1f* dst, float* filter, int kernel_size);
void SobelX(const Image1b& gray, Image1f* gradx, bool scharr = false);
void SobelY(const Image1b& gray, Image1f* grady, bool scharr = false);
void Laplacian(const Image1b& gray, Image1f* laplacian);

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

}  // namespace ugu
