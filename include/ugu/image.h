/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

#include "ugu/common.h"
#include "ugu/util/thread_util.h"

#if defined(UGU_USE_STB) && !defined(UGU_USE_OPENCV)
#include "stb_image.h"
#include "stb_image_resize.h"
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
#include "opencv2/imgproc.hpp"
#endif

namespace ugu {

inline bool WriteBinary(const std::string& path, void* data, size_t size) {
  std::ofstream ofs(path, std::ios::binary);
  ofs.write(reinterpret_cast<char*>(data), size);

  if (ofs.bad()) {
    return false;
  }
  return true;
}

#ifdef UGU_USE_OPENCV

template <typename T>
using Image = cv::Mat_<T>;

using Image1w = cv::Mat1w;
using Image1i = cv::Mat1i;
using Image1f = cv::Mat1f;
using Image1b = cv::Mat1b;
using Image2f = cv::Mat2f;
using Image3f = cv::Mat3f;
using Image3b = cv::Mat3b;
using Image4b = cv::Mat4b;

using Vec1b = uint8_t;
using Vec1f = float;
using Vec1i = int;
using Vec1w = std::uint16_t;
using Vec2f = cv::Vec2f;
using Vec2d = cv::Vec2d;
using Vec3f = cv::Vec3f;
using Vec3b = cv::Vec3b;
using Vec3d = cv::Vec3d;
using Vec4b = cv::Vec4b;

using Point = cv::Point;

using ImreadModes = cv::ImreadModes;
using InterpolationFlags = cv::InterpolationFlags;

using Size = cv::Size;

template <typename T>
inline bool imwrite(const std::string& filename, const T& img,
                    const std::vector<int>& params = std::vector<int>()) {
  return cv::imwrite(filename, img, params);
}

template <typename T>
void resize(const ugu::Image<T>& src, ugu::Image<T>& dst, Size dsize,
            double fx = 0, double fy = 0,
            int interpolation = InterpolationFlags::INTER_LINEAR) {
  cv::resize(src, dst, dsize, fx, fy, interpolation);
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
using Vec1b = Vec_<uint8_t, 1>;
using Vec1w = Vec_<std::uint16_t, 1>;
using Vec2f = Vec_<float, 2>;
using Vec2d = Vec_<double, 2>;
using Vec3b = Vec_<uint8_t, 3>;
using Vec3f = Vec_<float, 3>;
using Vec3d = Vec_<double, 3>;
using Vec4b = Vec_<uint8_t, 4>;

template <typename TT, int N>
using Point_ = Vec_<TT, N>;

#define CV_CN_MAX 512
#define CV_CN_SHIFT 3
#define CV_DEPTH_MAX (1 << CV_CN_SHIFT)

#define CV_8U 0
#define CV_8S 1
#define CV_16U 2
#define CV_16S 3
#define CV_32S 4
#define CV_32F 5
#define CV_64F 6
#define CV_16F 7

#define CV_MAT_DEPTH_MASK (CV_DEPTH_MAX - 1)
#define CV_MAT_DEPTH(flags) ((flags)&CV_MAT_DEPTH_MASK)

#define CV_MAKETYPE(depth, cn) (CV_MAT_DEPTH(depth) + (((cn)-1) << CV_CN_SHIFT))
#define CV_MAKE_TYPE CV_MAKETYPE

#define CV_8UC1 CV_MAKETYPE(CV_8U, 1)
#define CV_8UC2 CV_MAKETYPE(CV_8U, 2)
#define CV_8UC3 CV_MAKETYPE(CV_8U, 3)
#define CV_8UC4 CV_MAKETYPE(CV_8U, 4)
#define CV_8UC(n) CV_MAKETYPE(CV_8U, (n))

#define CV_8SC1 CV_MAKETYPE(CV_8S, 1)
#define CV_8SC2 CV_MAKETYPE(CV_8S, 2)
#define CV_8SC3 CV_MAKETYPE(CV_8S, 3)
#define CV_8SC4 CV_MAKETYPE(CV_8S, 4)
#define CV_8SC(n) CV_MAKETYPE(CV_8S, (n))

#define CV_16UC1 CV_MAKETYPE(CV_16U, 1)
#define CV_16UC2 CV_MAKETYPE(CV_16U, 2)
#define CV_16UC3 CV_MAKETYPE(CV_16U, 3)
#define CV_16UC4 CV_MAKETYPE(CV_16U, 4)
#define CV_16UC(n) CV_MAKETYPE(CV_16U, (n))

#define CV_16SC1 CV_MAKETYPE(CV_16S, 1)
#define CV_16SC2 CV_MAKETYPE(CV_16S, 2)
#define CV_16SC3 CV_MAKETYPE(CV_16S, 3)
#define CV_16SC4 CV_MAKETYPE(CV_16S, 4)
#define CV_16SC(n) CV_MAKETYPE(CV_16S, (n))

#define CV_32SC1 CV_MAKETYPE(CV_32S, 1)
#define CV_32SC2 CV_MAKETYPE(CV_32S, 2)
#define CV_32SC3 CV_MAKETYPE(CV_32S, 3)
#define CV_32SC4 CV_MAKETYPE(CV_32S, 4)
#define CV_32SC(n) CV_MAKETYPE(CV_32S, (n))

#define CV_32FC1 CV_MAKETYPE(CV_32F, 1)
#define CV_32FC2 CV_MAKETYPE(CV_32F, 2)
#define CV_32FC3 CV_MAKETYPE(CV_32F, 3)
#define CV_32FC4 CV_MAKETYPE(CV_32F, 4)
#define CV_32FC(n) CV_MAKETYPE(CV_32F, (n))

#define CV_64FC1 CV_MAKETYPE(CV_64F, 1)
#define CV_64FC2 CV_MAKETYPE(CV_64F, 2)
#define CV_64FC3 CV_MAKETYPE(CV_64F, 3)
#define CV_64FC4 CV_MAKETYPE(CV_64F, 4)
#define CV_64FC(n) CV_MAKETYPE(CV_64F, (n))

#define CV_16FC1 CV_MAKETYPE(CV_16F, 1)
#define CV_16FC2 CV_MAKETYPE(CV_16F, 2)
#define CV_16FC3 CV_MAKETYPE(CV_16F, 3)
#define CV_16FC4 CV_MAKETYPE(CV_16F, 4)
#define CV_16FC(n) CV_MAKETYPE(CV_16F, (n))

#define CV_GETCN(type) ((type >> CV_CN_SHIFT) + 1)
//#define CV_GETCN(type) (type)

//#define GET_ACTUAL_TYPE(type) ((type == ))

static int GetBitsFromCvType(int cv_type) {
  int cv_depth = CV_MAT_DEPTH(cv_type);

  if (cv_depth < 0) {
    throw std::runtime_error("");
  }

  if (cv_depth <= CV_8S) {
    return 8;
  } else if (cv_depth <= CV_16S) {
    return 16;
  } else if (cv_depth <= CV_32F) {
    return 32;
  } else if (cv_depth <= CV_64F) {
    return 64;
  } else if (cv_depth <= CV_16F) {
    return 16;
  }

  throw std::runtime_error("");
}

static const std::type_info& GetTypeidFromCvType(int cv_type) {
  // int cv_depth = CV_MAT_DEPTH(cv_type);

  // typeid(uint8_t) == typeid(int);

  if (cv_type < CV_8S) {
    return typeid(uint8_t);
  } else if (cv_type < CV_16U) {
    return typeid(int8_t);
  } else if (cv_type < CV_16S) {
    return typeid(uint16_t);
  } else if (cv_type < CV_32F) {
    return typeid(int16_t);
  } else if (cv_type < CV_64F) {
    return typeid(float);
  } else if (cv_type < CV_16F) {
    return typeid(double);
  }

  throw std::runtime_error("");
}

class ImageBase;

typedef ImageBase _InputArray;
typedef ImageBase _OutputArray;
typedef ImageBase _InputOutputArray;

typedef const _InputArray& InputArray;
typedef InputArray InputArrayOfArrays;
typedef const _OutputArray& OutputArray;
typedef OutputArray OutputArrayOfArrays;
typedef const _InputOutputArray& InputOutputArray;
typedef InputOutputArray InputOutputArrayOfArrays;

static inline InputOutputArray noArray();
static inline size_t SizeInBytes(const ImageBase& mat);

class ImageBase {
 protected:
  int cv_type = CV_8UC1;
  int cv_depth = -1;
  int cv_ch = -1;
  int bit_depth_ = -1;
  const std::type_info* cpp_type;
  // std::type_info type = typeid(int);
  // int bit_depth_{sizeof(typename T::value_type)};
  // int channels_{std::tuple_size<T>::value};
  // int width_{-1};
  // int height_{-1};
  std::shared_ptr<std::vector<uint8_t> > data_{nullptr};

  void Init(int rows_, int cols_, int type) {
    rows = rows_;
    cols = cols_;
    cv_type = type;

    cv_depth = CV_MAT_DEPTH(type);
    cv_ch = CV_GETCN(type);
    bit_depth_ = GetBitsFromCvType(cv_type) / 8;
    cpp_type = &GetTypeidFromCvType(cv_type);
    data_->resize(static_cast<size_t>(rows) * static_cast<size_t>(cols) *
                  cv_ch * bit_depth_);
    data = reinterpret_cast<uint8_t*>(data_->data());

    step[0] = size_t(cols * bit_depth_ * cv_ch);
    step[1] = 1;
  }

 public:
  ImageBase(int rows, int cols, int type) { Init(rows, cols, type); }
  ImageBase() { Init(0, 0, 0); };

  ImageBase(const ImageBase& src) = default;
  virtual ~ImageBase() {}

  int channels() const { return cv_ch; }
  int rows{-1};
  int cols{-1};
  std::array<size_t, 2> step;  // Not compatible
  uint8_t* data{nullptr};
  int type() const { return cv_type; };

  bool empty() const {
    if (rows < 1 || cols < 1 || data_->empty()) {
      return true;
    }
    return false;
  }

  size_t elemSize() const { return size_t(bit_depth_ * channels()); }

  size_t elemSize1() const { return size_t(bit_depth_); }

  size_t total() const { return rows * cols; }

  bool isContinuous() const {
    // Always true for this implementation
    return true;
  }

  template <typename TT>
  TT& at(int y, int x) {
    return *(reinterpret_cast<TT*>(data_->data()) +
             ((static_cast<size_t>(y) * static_cast<size_t>(cols)) + x));
  }

  template <typename TT>
  const TT& at(int y, int x) const {
    return *(reinterpret_cast<TT*>(data_->data()) +
             ((static_cast<size_t>(y) * static_cast<size_t>(cols)) + x));
  }

  void setTo(InputArray value, InputArray mask = noArray()) {
    if (mask.empty()) {
      *this = value;
      return;
    }

    if (this->cols != mask.cols || this->rows != mask.rows ||
        mask.cv_type != CV_8U) {
      throw std::runtime_error("Type error");
    }

    if (cv_type != value.cv_type) {
      throw std::runtime_error("Type error");
    }

    auto copy_func = [&](int index_) {
      int x = index_ % cols;
      int y = index_ / cols;
      if (mask.at<uint8_t>(y, x) != 255) {
        return;
      }

      size_t index = (x + y * cols) * this->channels();
      size_t offset = index * bit_depth_;
      size_t size = channels() * bit_depth_;
      std::memcpy(this->data + offset, value.data + offset, size);
    };

    parallel_for(0, cols * rows, copy_func);
  }

  void setTo(double value, InputArray mask = noArray()) {
    if (mask.empty()) {
      *this = value;
      return;
    }

    if (this->cols != mask.cols || this->rows != mask.rows ||
        mask.cv_type != CV_8U) {
      throw std::runtime_error("Type error");
    }

#define UGU_SET(type) (this->at<type>(y, x) = static_cast<type>(value))

    auto copy_func = [&](int index_) {
      int x = index_ % cols;
      int y = index_ / cols;
      if (mask.at<uint8_t>(y, x) != 255) {
        return;
      }

      if (*cpp_type == typeid(uint8_t)) {
        UGU_SET(uint8_t);
      } else if (*cpp_type == typeid(int8_t)) {
        UGU_SET(int8_t);
      } else if (*cpp_type == typeid(uint16_t)) {
        UGU_SET(uint16_t);
      } else if (*cpp_type == typeid(int16_t)) {
        UGU_SET(int16_t);
      } else if (*cpp_type == typeid(float)) {
        UGU_SET(float);
      } else if (*cpp_type == typeid(double)) {
        UGU_SET(double);
      } else {
        throw std::runtime_error("type error");
      }
    };
#undef UGU_SET

    parallel_for(0, cols * rows, copy_func);
  }

  static ImageBase zeros(int height, int width, int type) {
    ImageBase zero(width, height, type);
    zero = 0;
    return zero;
  }

  void copyTo(ImageBase& dst) const {  // NOLINT
    if (dst.cols != cols || dst.rows != rows) {
      dst = zeros(rows, cols, cv_type);
    }
    std::memcpy(dst.data_->data(), data_->data(), SizeInBytes(*this));
  }

  ImageBase clone() const {
    ImageBase dst;
    this->copyTo(dst);
    return dst;
  }

  template <typename TT>
  void forEach(std::function<void(TT&, const int[2])> f) {
    if (empty()) {
      return;
    }
    size_t st(0);
    size_t ed = static_cast<size_t>(cols * rows * sizeof(T) / sizeof(TT));
    auto f2 = [&](const size_t& i) {
      const int xy[2] = {static_cast<int32_t>(i) % cols,
                         static_cast<int32_t>(i) / cols};
      f(reinterpret_cast<TT*>(data)[i], xy);
    };
    parallel_for(st, ed, f2);
  }

  ImageBase& operator=(const ImageBase& rhs) {
    *this = rhs;
    return *this;
  }

  ImageBase& operator=(const double& rhs) {
#define UGU_FILL_CAST(type)                                                \
  (std::fill(reinterpret_cast<std::vector<type>*>(data_->data())->begin(), \
             reinterpret_cast<std::vector<type>*>(data_->data())->end(),   \
             static_cast<type>(rhs)));

    if (*cpp_type == typeid(uint8_t)) {
      std::memset(data, static_cast<uint8_t>(rhs), SizeInBytes(*this));
    } else if (*cpp_type == typeid(int8_t)) {
      //     std::fill(reinterpret_cast<std::vector<int8_t>*>(data_.get())->begin(),
      //                reinterpret_cast<std::vector<int8_t>*>(data_.get())->end(),
      //                static_cast<int8_t>(rhs));
      UGU_FILL_CAST(int8_t);
    } else if (*cpp_type == typeid(uint16_t)) {
      UGU_FILL_CAST(uint16_t);
    } else if (*cpp_type == typeid(int16_t)) {
      UGU_FILL_CAST(int16_t);
    } else if (*cpp_type == typeid(float)) {
      UGU_FILL_CAST(float);
    } else if (*cpp_type == typeid(double)) {
      UGU_FILL_CAST(double);
    } else {
      throw std::runtime_error("type error");
    }

#undef UGU_FILL_CAST

    return *this;
  }
};

size_t SizeInBytes(const ImageBase& mat) {
  // https://stackoverflow.com/questions/26441072/finding-the-size-in-bytes-of-cvmat
  return size_t(mat.step[0] * mat.rows);
}

InputOutputArray noArray() { return _InputOutputArray(); }

#if 0
template <typename T,
          std::enable_if_t<std::is_scalar_v<T>, std::nullptr_t> = nullptr>
int ParseVec() {
  int ch = 0;
  std::type_info* info;
  if (std::is_scalar<T>()) {
    ch = 1;
    info = const_cast<std::type_info*>(&typeid(T));
  } else {
    ch = T().size();
    info = const_cast<std::type_info*>(&typeid(T::value_type));
  }

  int code = -1;

  if (info == &typeid(uint8_t)) {
    code = CV_MAKETYPE(CV_8U, ch);
  } else if (info == &typeid(int8_t)) {
    code = CV_MAKETYPE(CV_8S, ch);
  } else if (info == &typeid(uint16_t)) {
    code = CV_MAKETYPE(CV_16U, ch);
  } else if (info == &typeid(int16_t)) {
    code = CV_MAKETYPE(CV_16S, ch);
  } else if (info == &typeid(float)) {
    code = CV_MAKETYPE(CV_32F, ch);
  } else if (info == &typeid(double)) {
    code = CV_MAKETYPE(CV_64F, ch);
  } else {
    throw std::runtime_error("type error");
  }

  return code;
}
#endif

inline int MakeCvType(const std::type_info* info, int ch) {
  int code = -1;

  if (info == &typeid(uint8_t)) {
    code = CV_MAKETYPE(CV_8U, ch);
  } else if (info == &typeid(int8_t)) {
    code = CV_MAKETYPE(CV_8S, ch);
  } else if (info == &typeid(uint16_t)) {
    code = CV_MAKETYPE(CV_16U, ch);
  } else if (info == &typeid(int16_t)) {
    code = CV_MAKETYPE(CV_16S, ch);
  } else if (info == &typeid(float)) {
    code = CV_MAKETYPE(CV_32F, ch);
  } else if (info == &typeid(double)) {
    code = CV_MAKETYPE(CV_64F, ch);
  } else {
    throw std::runtime_error("type error");
  }

  return code;
}

template <typename T,
          std::enable_if_t<std::is_scalar_v<T>, std::nullptr_t> = nullptr>
int ParseVec() {
  const int ch = 1;
  const std::type_info* info = &typeid(T);
  return MakeCvType(info, ch);
}

template <typename T,
          std::enable_if_t<std::is_compound_v<T>, std::nullptr_t> = nullptr>
int ParseVec() {
  const int ch = T().size();
  const std::type_info* info = &typeid(T::value_type);
  return MakeCvType(info, ch);
}

template <typename T>
class Image : public ImageBase {
 private:
 public:
  Image(int rows, int cols, int type) = delete;
  Image(int rows, int cols) {
    int code = ParseVec<T>();
    Init(rows, cols, code);
  };
  Image() { Init(0, 0, 0); }

  ~Image(){};

  static Image<T> zeros(int height, int width) {
    Image<T> zero(width, height);
    zero = 0.0;
    return zero;
  }

  Image<T> clone() const {
    return *dynamic_cast<Image<T>*>(&ImageBase::clone());
  }

  using ImageBase::setTo;
  void setTo(T value, InputArray mask = noArray()) {
    if (mask.empty()) {
      *this = value;
      return;
    }
    LOGE("Please implement!\n");
  }

#ifndef UGU_USE_STB
  bool Load(const std::string& path) {
    uint8_t* in_pixels_tmp;
    int width;
    int height;
    int bpp;

    if (bit_depth_ == 2) {
      in_pixels_tmp = reinterpret_cast<uint8_t*>(
          stbi_load_16(path.c_str(), &width, &height, &bpp, channels_));
    } else if (bit_depth_ == 1) {
      in_pixels_tmp = stbi_load(path.c_str(), &width, &height, &bpp, channels_);
    } else {
      LOGE("Load() for bit_depth %d and channel %d is not supported\n",
           bit_depth_, channels_);
      return false;
    }

    if (bpp != channels_) {
      stbi_image_free(in_pixels_tmp);
      LOGE("desired channel %d, actual %d\n", channels_, bpp);
      return false;
    }

    Init(width, height);

    std::memcpy(data_->data(), in_pixels_tmp, sizeof(T) * width_ * height_);
    stbi_image_free(in_pixels_tmp);

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
    std::vector<uint8_t> data_8bit;
    data_8bit.resize(width_ * height_ * 2);  // 2 bytes per pixel
    const int kMostMask = 0b1111111100000000;
    const int kLeastMask = ~kMostMask;
    for (int y = 0; y < height_; y++) {
      for (int x = 0; x < width_; x++) {
        std::uint16_t d = this->at<std::uint16_t>(y, x);  // At(*this, x, y, 0);
        data_8bit[2 * width_ * y + 2 * x + 0] =
            static_cast<uint8_t>((d & kMostMask) >> 8);  // most significant
        data_8bit[2 * width_ * y + 2 * x + 1] =
            static_cast<uint8_t>(d & kLeastMask);  // least significant
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

  bool WriteBinary(const std::string& path) const {
    return ugu::WriteBinary(path, data, bit_depth_ * cols * rows);
  }

#if 0
  void copyTo(Image<T>& dst) const {  // NOLINT
    if (dst.cols != cols || dst.rows != rows) {
      // dst = Image<T>::zeros(rows, cols);
      dst.data_ = std::make_shared<std::vector<T> >();
      dst.Init(cols, rows, cv_type);
    }
    std::memcpy(dst.data_->data(), data_->data(), sizeof(T) * rows * cols);
  }

  Image<T> clone() const {
    Image<T> dst;
    this->copyTo(dst);
    return dst;
  }
#endif

  void forEach(std::function<void(T&, const int[2])> f) {
    if (empty()) {
      return;
    }
    size_t st(0);
    size_t ed = static_cast<size_t>(cols * rows * sizeof(T));
    auto f2 = [&](const size_t& i) {
      const int xy[2] = {static_cast<int32_t>(i) % cols,
                         static_cast<int32_t>(i) / cols};
      f(reinterpret_cast<T*>(data)[i], xy);
    };
    ugu::parallel_for(st, ed, f2);
  }

  Image<T>& operator=(const T& rhs);
  Image<T>& operator=(const ImageBase& rhs);
  Image<T>& operator=(const double& rhs);
};

template <typename T>
Image<T>& Image<T>::operator=(const T& rhs) {
  std::fill(reinterpret_cast<std::vector<T>*>(data_->data())->begin(),
            reinterpret_cast<std::vector<T>*>(data_->data())->end(), rhs);
  return *this;
}

template <typename T>
Image<T>& Image<T>::operator=(const ImageBase& rhs) {
  *this = rhs;
  return *this;
}

template <typename T>
Image<T>& Image<T>::operator=(const double& rhs) {
  ImageBase::operator=(rhs);
  return *this;
}

using Image1b = Image<uint8_t>;   // For gray image.
using Image1w = Image<uint16_t>;  // For depth image with 16 bit (unsigned
                                  // short) mm-scale format
using Image1i =
    Image<int32_t>;            // For face visibility. face id is within int32_t
using Image1f = Image<float>;  // For depth image with any scale
using Image2f = Image<Vec2f>;
using Image3f = Image<Vec3f>;  // For normal or point cloud. XYZ order.
using Image3b = Image<Vec3b>;  // For color image. RGB order.
using Image4b = Image<Vec4b>;  // For color image. RGBA order.

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

enum InterpolationFlags {
  INTER_NEAREST = 0,
  INTER_LINEAR = 1,
  INTER_CUBIC = 2,
  INTER_AREA = 3,
  INTER_LANCZOS4 = 4,
  INTER_LINEAR_EXACT = 5,
  INTER_NEAREST_EXACT = 6,
  INTER_MAX = 7,
  WARP_FILL_OUTLIERS = 8,
  WARP_INVERSE_MAP = 16
};

enum LineTypes { FILLED = -1, LINE_4 = 4, LINE_8 = 8, LINE_AA = 16 };

template <typename T>
struct Size_ {
  T height = T(-1);
  T width = T(-1);
  Size_() {}
  Size_(T width_, T height_) {
    width = width_;
    height = height_;
  }
  T area() const { return height * width; }
};

using Size = Size_<int>;

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
  image->setTo(static_cast<double>(val));
}

template <typename T>
inline void Init(Image<T>* image, int width, int height, typename T val) {
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
  } else if (extname == ".bin" || extname == ".BIN") {
    return img.WriteBinary(filename);
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

#define UGU_CAST1(type) \
  (static_cast<double>(reinterpret_cast<const type*>(&src_at)[c]))
#define UGU_CAST2(type) \
  (reinterpret_cast<type*>(&dst_at)[c] = static_cast<type>(scale * val))

  const std::type_info* cpp_type_src = &GetTypeidFromCvType(src.type());
  const std::type_info* cpp_type_dst = &GetTypeidFromCvType(dst->type());

  for (int y = 0; y < src.rows; y++) {
    for (int x = 0; x < src.cols; x++) {
      TT& dst_at = dst->template at<TT>(y, x);
      const T& src_at = src.template at<T>(y, x);
      for (int c = 0; c < dst->channels(); c++) {
        //(&dst_at)[c] =
        //    static_cast<typename TT::value_type>(scale * (&src_at)[c]);

// UGU_CAST(uint8_t);
// reinterpret_cast<uint8_t*>(&dst_at)[c] = 255;//static_cast<uint8_t>(scale *
// ((&src_at)[c]));
#if 1
        double val = 0.0;
        if (*cpp_type_src == typeid(uint8_t)) {
          val = UGU_CAST1(uint8_t);
        } else if (*cpp_type_src == typeid(int8_t)) {
          val = UGU_CAST1(int8_t);
        } else if (*cpp_type_src == typeid(uint16_t)) {
          val = UGU_CAST1(uint16_t);
        } else if (*cpp_type_src == typeid(int16_t)) {
          val = UGU_CAST1(int16_t);
        } else if (*cpp_type_src == typeid(float)) {
          val = UGU_CAST1(float);
        } else if (*cpp_type_src == typeid(double)) {
          val = UGU_CAST1(double);
        } else {
          throw std::runtime_error("type error");
        }

        if (*cpp_type_dst == typeid(uint8_t)) {
          UGU_CAST2(uint8_t);
        } else if (*cpp_type_dst == typeid(int8_t)) {
          UGU_CAST2(int8_t);
        } else if (*cpp_type_dst == typeid(uint16_t)) {
          UGU_CAST2(uint16_t);
        } else if (*cpp_type_dst == typeid(int16_t)) {
          UGU_CAST2(int16_t);
        } else if (*cpp_type_dst == typeid(float)) {
          UGU_CAST2(float);
        } else if (*cpp_type_dst == typeid(double)) {
          UGU_CAST2(double);
        } else {
          throw std::runtime_error("type error");
        }

#endif
      }
    }
  }

#undef UGU_CAST

  return true;
}

class Point2i {
 public:
  int x = -1;
  int y = -1;
  Point2i(){};
  Point2i(int x_, int y_) : x(x_), y(y_){};
  ~Point2i(){};
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
      const T& val_c = src.template at<T>(y, x);
      for (int c = 0; c < src.channels(); c++) {
        const auto& val = (&val_c)[c];
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

template <typename T>
void resize(const Image<T>& src, Image<T>& dst, Size dsize, double fx = 0.0,
            double fy = 0.0,
            int interpolation = InterpolationFlags::INTER_LINEAR) {
#ifdef UGU_USE_STB
  (void)interpolation;

  int w = src.cols;
  int h = src.rows;
  int n = src.channels();

  int out_w, out_h;
  if (dsize.height <= 0 || dsize.width <= 0) {
    out_w = static_cast<int>(w * fx);
    out_h = static_cast<int>(h * fy);
  } else {
    out_w = dsize.width;
    out_h = dsize.height;
  }

  if (w <= 0 || h <= 0 || out_w <= 0 || out_h <= 0) {
    LOGE("Wrong size\n");
    return;
  }

  dst = Image<T>::zeros(out_h, out_w);

  stbir_resize_uint8(src.data, w, h, 0, dst.data, out_w, out_h, 0, n);

  return;
#else
  (void)src;
  (void)dst;
  (void)dsize;
  (void)fx;
  (void)fy;
  (void)interpolation;
  LOGE("can't resize image with this configuration\n");
  return;
#endif
}

template <typename T>
void circle(Image<T>& img, Point center, int radius, const T& color,
            int thickness = 1, int lineType = LINE_8, int shift = 0) {
  (void)lineType;
  (void)shift;
  auto min_x = std::max(0, std::min({center.x - radius, img.cols - 1}));
  auto max_x = std::min(img.cols - 1, std::max({center.x + radius, 0}));
  auto min_y = std::max(0, std::min({center.y - radius, img.rows - 1}));
  auto max_y = std::min(img.rows - 1, std::max({center.y + radius, 0}));

  float radius_f = static_cast<float>(radius);
  float thickness_f = static_cast<float>(thickness);

  for (int y = min_y; y <= max_y; y++) {
    for (int x = min_x; x <= max_x; x++) {
      float dist = std::sqrt(static_cast<float>(
          (center.x - x) * (center.x - x) + (center.y - y) * (center.y - y)));
      if (thickness < 0) {
        if (dist <= radius_f) {
          img.template at<T>(y, x) = color;
        }
      } else {
        if (dist < radius_f && radius_f - dist <= thickness_f) {
          img.template at<T>(y, x) = color;
        }
      }
    }
  }
}

template <typename T>
void line(Image<T>& img, Point pt1, Point pt2, const T& color,
          int thickness = 1, int lineType = 8, int shift = 0) {
  (void)lineType;
  (void)shift;

  // Naive implementation of "All cases"
  // https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm

  thickness = std::max(1, thickness);

  auto plotLineLow = [&](int x0, int y0, int x1, int y1) {
    int dx = x1 - x0;
    int dy = y1 - y0;
    int yi = 1;
    if (dy < 0) {
      yi = -1;
      dy = -dy;
    }

    int D = (2 * dy) - dx;
    int y = y0;

    x0 = std::clamp(x0, 0, img.cols - 1);
    x1 = std::clamp(x1, 0, img.cols - 1);

    for (int x = x0; x <= x1; x++) {
      for (int t = 0; t < thickness; t++) {
        int y_ = std::clamp(t + y, 0, img.rows - 1);
        img.template at<T>(y_, x) = color;
      }
      if (D > 0) {
        y = y + yi;
        D = D + (2 * (dy - dx));
      } else {
        D = D + 2 * dy;
      }
    }
  };

  auto plotLineHigh = [&](int x0, int y0, int x1, int y1) {
    int dx = x1 - x0;
    int dy = y1 - y0;
    int xi = 1;
    if (dx < 0) {
      xi = -1;
      dx = -dx;
    }

    int D = (2 * dx) - dy;
    int x = x0;

    y0 = std::clamp(y0, 0, img.rows - 1);
    y1 = std::clamp(y1, 0, img.rows - 1);

    for (int y = y0; y <= y1; y++) {
      for (int t = 0; t < thickness; t++) {
        int x_ = std::clamp(t + x, 0, img.cols - 1);
        img.template at<T>(y, x_) = color;
      }
      if (D > 0) {
        x = x + xi;
        D = D + (2 * (dx - dy));
      } else {
        D = D + 2 * dx;
      }
    }
  };

  int x1 = pt1.x;
  int y1 = pt1.y;
  int x2 = pt2.x;
  int y2 = pt2.y;
  if (std::abs(y2 - y1) < std::abs(x2 - x1)) {
    if (x1 > x2) {
      plotLineLow(x2, y2, x1, y1);
    } else {
      plotLineLow(x1, y1, x2, y2);
    }
  } else {
    if (y1 > y2) {
      plotLineHigh(x2, y2, x1, y1);
    } else {
      plotLineHigh(x1, y1, x2, y2);
    }
  }
}

#endif

}  // namespace ugu
