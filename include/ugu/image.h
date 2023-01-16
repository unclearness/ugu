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

#ifdef UGU_USE_OPENCV
#include "opencv2/core.hpp"
#endif

namespace ugu {

#ifdef UGU_USE_OPENCV

using ImageBase = cv::Mat;

template <typename T>
using Image = cv::Mat_<T>;

using Image1b = cv::Mat1b;
using Image1w = cv::Mat1w;
using Image1i = cv::Mat1i;
using Image1f = cv::Mat1f;
using Image2f = cv::Mat2f;
using Image3b = cv::Mat3b;
using Image3f = cv::Mat3f;
using Image3d = cv::Mat3d;
using Image4b = cv::Mat4b;

using Vec2f = cv::Vec2f;
using Vec2d = cv::Vec2d;
using Vec3b = cv::Vec3b;
using Vec3f = cv::Vec3f;
using Vec3d = cv::Vec3d;
using Vec4b = cv::Vec4b;

using Size = cv::Size;

using cv::noArray;

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

#else

class ImageBase;

template <typename _Tp, int m, int n>
class Matx {
 public:
  using value_type = _Tp;
  enum { rows = m, cols = n, channels = rows * cols };
  Matx() {  // std::fill(val.begin(), val.end(), 0);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        val[i * n + j] = static_cast<_Tp>(0);
      }
    }
  };
  Matx(const double& v) {
    // std::fill(val.begin(), val.end(), static_cast<_Tp>(v));

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        val[i * n + j] = static_cast<_Tp>(v);
      }
    }
  };
  Matx(const Matx<_Tp, n, m>& a) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        val[i * n + j] = a.val[i * n + j];
      }
    }
  }
  template <int m_>
  Matx(const Matx<_Tp, m_, n>& rhs) {
    *this = rhs;
  }

  Matx(std::initializer_list<_Tp> list) {
    assert(list.size() == channels);
    int i = 0;
    for (const auto& elem : list) {
      val[i++] = elem;
    }
  }

  Matx(const ImageBase& a);

  ~Matx(){};

  inline _Tp operator[](size_t index) const { return val[index]; }

  inline _Tp& operator[](size_t index) { return val[index]; }

  Matx<_Tp, m, n>& operator=(const double& rhs) {
    for (int i = 0; i < channels; i++) {
      val[i] = static_cast<_Tp>(rhs);
    }
    return *this;
  }

  template <int m_>
  Matx<_Tp, m, n>& operator=(const Matx<_Tp, m_, n>& rhs) {
    int r_min = std::min(m, m_);  // TODO
    for (int r = 0; r < r_min; r++) {
      for (int c = 0; c < cols; c++) {
        val[r * cols + c] = rhs.val[r * cols + c];
      }
    }
    return *this;
  }

  Matx<_Tp, n, m> t() const { return Matx<_Tp, n, m>(*this); }

  Matx<_Tp, m, n> div(const Matx<_Tp, m, n>& a) const {
    Matx<_Tp, m, n> out = *this;
    for (int i = 0; i < channels; i++) {
      out.val[i] /= a.val[i];
    }
    return out;
  }

  Matx<_Tp, m, n> operator-() const {
    Matx<_Tp, m, n> out = *this;
    for (int i = 0; i < channels; i++) {
      out.val[i] *= -1;
    }
    return out;
  }

  // std::array<_Tp, m * n> val;
  _Tp val[m * n];
};

template <typename TT, int N>
using Vec_ = Matx<TT, N, 1>;

using Vec2f = Vec_<float, 2>;
using Vec2d = Vec_<double, 2>;
using Vec3b = Vec_<uint8_t, 3>;
using Vec3f = Vec_<float, 3>;
using Vec3d = Vec_<double, 3>;
using Vec4d = Vec_<double, 4>;
using Vec4b = Vec_<uint8_t, 4>;
using Vec4f = Vec_<float, 4>;
using Vec4i = Vec_<int32_t, 4>;

template <typename TT>
using Scalar_ = Vec_<TT, 4>;

using Scalar = Scalar_<double>;

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

int GetBitsFromCvType(int cv_type);
const std::type_info& GetTypeidFromCvType(int cv_type);
template <typename T>
const int GetDepth() {
  int depth = -1;
  const std::type_info& cpp_type = typeid(T);
  if (cpp_type == typeid(uint8_t)) {
    depth = CV_8U;
  } else if (cpp_type == typeid(int8_t)) {
    depth = CV_8S;
  } else if (cpp_type == typeid(uint16_t)) {
    depth = CV_16U;
  } else if (cpp_type == typeid(int16_t)) {
    depth = CV_16S;
  } else if (cpp_type == typeid(int32_t)) {
    depth = CV_32S;
  } else if (cpp_type == typeid(float)) {
    depth = CV_32F;
  } else if (cpp_type == typeid(double)) {
    depth = CV_64F;
  } else {
    throw std::runtime_error("Not supported");
  }
  return depth;
}

class ImageBase;

// TODO
// Not strictly follow to OpenCV because the imitation of xxxxArray would take
// time...
typedef ImageBase _InputArray;
typedef ImageBase _OutputArray;
typedef ImageBase _InputOutputArray;

typedef const _InputArray& InputArray;
typedef InputArray InputArrayOfArrays;
typedef _OutputArray& OutputArray;
typedef OutputArray OutputArrayOfArrays;
typedef const _InputOutputArray& InputOutputArray;
typedef InputOutputArray InputOutputArrayOfArrays;

size_t SizeInBytes(const ImageBase& mat);
InputOutputArray noArray();
int MakeCvType(const std::type_info* info, int ch);

class ImageBase {
 protected:
  int cv_type = CV_8UC1;
  int cv_depth = -1;
  int cv_ch = -1;
  int bit_depth_ = -1;
  const std::type_info* cpp_type;
  std::shared_ptr<std::vector<uint8_t> > data_{nullptr};

  void Init(int rows_, int cols_, int type) {
    if (rows_ < 0 || cols_ < 0) {
      throw std::runtime_error("Type error");
    }

    rows = rows_;
    cols = cols_;
    cv_type = type;

    cv_depth = CV_MAT_DEPTH(type);
    cv_ch = CV_GETCN(type);
    bit_depth_ = GetBitsFromCvType(cv_type) / 8;
    cpp_type = &GetTypeidFromCvType(cv_type);
    data_ = std::make_shared<std::vector<uint8_t> >();
    data_->resize(static_cast<size_t>(rows) * static_cast<size_t>(cols) *
                  cv_ch * bit_depth_);
    data = reinterpret_cast<uint8_t*>(data_->data());

    step[0] = size_t(cols * bit_depth_ * cv_ch);
    step[1] = 1;
  }

 public:
  int rows{-1};
  int cols{-1};
  std::array<size_t, 2> step;  // Not compatible
  uint8_t* data{nullptr};

  ImageBase(int rows, int cols, int type) { Init(rows, cols, type); }
  ImageBase() { Init(0, 0, 0); };
  template <typename _Tp, int m, int n>
  ImageBase(const Matx<_Tp, m, n>& rhs) {
    Init(m, 1, CV_MAKETYPE(GetDepth<_Tp>(), n));
    std::memcpy(data, rhs.val, sizeof(_Tp) * rhs.channels);
  };
  ImageBase(const ImageBase& src) = default;
  virtual ~ImageBase() {}

  int channels() const { return cv_ch; }

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
    // return *(reinterpret_cast<TT*>(data_->data()) +
    //          ((static_cast<size_t>(y) * static_cast<size_t>(cols)) + x));
    return ((TT*)(data + step[0] * y))[x];
  }

  template <typename TT>
  const TT& at(int y, int x) const {
    // return *(reinterpret_cast<TT*>(data_->data()) +
    //          ((static_cast<size_t>(y) * static_cast<size_t>(cols)) + x));
    return ((TT*)(data + step[0] * y))[x];
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
      } else if (*cpp_type == typeid(int32_t)) {
        UGU_SET(int32_t);
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
    ImageBase zero(height, width, type);
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

  ImageBase mul(const ImageBase& a) const {
    assert(rows == a.rows && cols == a.cols && cv_type == a.cv_type);
    ImageBase dst = clone();

// #define UGU_SET(type) (dst.at<type>(y, x) *= a.at<type>(y, x))
#define UGU_SET(type)                                                       \
  type* data = reinterpret_cast<type*>(dst.data) + index_ * dst.channels(); \
  type* src = reinterpret_cast<type*>(a.data) + index_ * dst.channels();    \
  for (int c = 0; c < dst.channels(); c++) {                                \
    data[c] = data[c] * src[c];                                             \
  }
    auto copy_func = [&](int index_) {
      if (*cpp_type == typeid(uint8_t)) {
        UGU_SET(uint8_t);
      } else if (*cpp_type == typeid(int8_t)) {
        UGU_SET(int8_t);
      } else if (*cpp_type == typeid(uint16_t)) {
        UGU_SET(uint16_t);
      } else if (*cpp_type == typeid(int16_t)) {
        UGU_SET(int16_t);
      } else if (*cpp_type == typeid(int32_t)) {
        UGU_SET(int32_t);
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

    return dst;
  }

  void convertTo(OutputArray m, int rtype, double alpha = 1,
                 double beta = 0) const {
    m = ImageBase(rows, cols, rtype);
    if (channels() != m.channels()) {
      throw std::runtime_error("#channels must be same.");
    }

    //  std::cout << GetBitsFromCvType() << std::endl;

    ImageBase tmp_double =
        ImageBase(rows, cols, MakeCvType(&typeid(double), channels()));
    // Convert to double
#define UGU_COPY_DOUBLE(type)                                                \
  for (size_t i = 0; i < m.total() * m.channels(); i++) {                    \
    *(reinterpret_cast<double*>(tmp_double.data_->data()) + i) =             \
        static_cast<double>(*(reinterpret_cast<type*>(data_->data()) + i)) * \
            alpha +                                                          \
        beta;                                                                \
  }
    auto copy_to_double_func = [&]() {
      if (*cpp_type == typeid(uint8_t)) {
        UGU_COPY_DOUBLE(uint8_t);
      } else if (*cpp_type == typeid(int8_t)) {
        UGU_COPY_DOUBLE(int8_t);
      } else if (*cpp_type == typeid(uint16_t)) {
        UGU_COPY_DOUBLE(uint16_t);
      } else if (*cpp_type == typeid(int16_t)) {
        UGU_COPY_DOUBLE(int16_t);
      } else if (*cpp_type == typeid(int32_t)) {
        UGU_COPY_DOUBLE(int32_t);
      } else if (*cpp_type == typeid(float)) {
        UGU_COPY_DOUBLE(float);
      } else if (*cpp_type == typeid(double)) {
        UGU_COPY_DOUBLE(double);
      } else {
        throw std::runtime_error("type error");
      }
    };
#undef UGU_COPY_DOUBLE
    copy_to_double_func();

    // Double to target
#define UGU_COPY_TARGET(type)                                              \
  for (size_t i = 0; i < m.total() * m.channels(); i++) {                  \
    *(reinterpret_cast<type*>(m.data_->data()) + i) = saturate_cast<type>( \
        *(reinterpret_cast<double*>(tmp_double.data_->data()) + i));       \
  }
    auto copy_to_target_func = [&]() {
      if (GetTypeidFromCvType(m.type()) == typeid(uint8_t)) {
        UGU_COPY_TARGET(uint8_t);
      } else if (GetTypeidFromCvType(m.type()) == typeid(int8_t)) {
        UGU_COPY_TARGET(int8_t);
      } else if (GetTypeidFromCvType(m.type()) == typeid(uint16_t)) {
        UGU_COPY_TARGET(uint16_t);
      } else if (GetTypeidFromCvType(m.type()) == typeid(int16_t)) {
        UGU_COPY_TARGET(int16_t);
      } else if (GetTypeidFromCvType(m.type()) == typeid(int32_t)) {
        UGU_COPY_TARGET(int32_t);
      } else if (GetTypeidFromCvType(m.type()) == typeid(float)) {
        UGU_COPY_TARGET(float);
      } else if (GetTypeidFromCvType(m.type()) == typeid(double)) {
        UGU_COPY_TARGET(double);
      } else {
        throw std::runtime_error("type error");
      }
    };
#undef UGU_COPY_TARGET
    copy_to_target_func();
  }

  template <typename TT>
  void forEach(std::function<void(const TT&, const int[2])> f) const {
    if (empty()) {
      return;
    }
    size_t st(0);
    size_t ed = static_cast<size_t>(cols * rows * bit_depth_ / sizeof(TT));
    auto f2 = [&](const size_t& i) {
      const int xy[2] = {static_cast<int32_t>(i) % cols,
                         static_cast<int32_t>(i) / cols};
      f(reinterpret_cast<TT*>(data)[i], xy);
    };
    parallel_for(st, ed, f2);
  }

  template <typename TT>
  void forEach(std::function<void(TT&, const int[2])> f) {
    if (empty()) {
      return;
    }
    size_t st(0);
    size_t ed = static_cast<size_t>(cols * rows * bit_depth_ / sizeof(TT));
    auto f2 = [&](const size_t& i) {
      const int xy[2] = {static_cast<int32_t>(i) % cols,
                         static_cast<int32_t>(i) / cols};
      f(reinterpret_cast<TT*>(data)[i], xy);
    };
    parallel_for(st, ed, f2);
  }

  ImageBase& operator=(const double& rhs) {

#if 0
#define UGU_FILL_CAST(type)                                                 \
  for (size_t i = 0; i < total() * channels(); i++) {                       \
    *(reinterpret_cast<type*>(data_->data()) + i) = static_cast<type>(rhs); \
  }
#else
#define UGU_FILL_CAST(type) \
  std::fill(data_->begin(), data_->end(), static_cast<type>(rhs));
#endif

    if (*cpp_type == typeid(uint8_t)) {
      UGU_FILL_CAST(uint8_t);
    } else if (*cpp_type == typeid(int8_t)) {
      UGU_FILL_CAST(int8_t);
    } else if (*cpp_type == typeid(uint16_t)) {
      UGU_FILL_CAST(uint16_t);
    } else if (*cpp_type == typeid(int16_t)) {
      UGU_FILL_CAST(int16_t);
    } else if (*cpp_type == typeid(int32_t)) {
      UGU_FILL_CAST(int32_t);
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

template <typename _Tp, int m, int n>
Matx<_Tp, m, n>::Matx(const ImageBase& a) {
  assert(channels == a.total() * a.channels());
  std::memcpy(val, a.data, sizeof(_Tp) * channels);
}

inline ImageBase operator*(const ImageBase& lhs, const double& rhs) {
  ImageBase ret = lhs.clone();

#define UGU_MUL(type)                                         \
  for (size_t i = 0; i < lhs.total() * lhs.channels(); i++) { \
    type& v = *(reinterpret_cast<type*>(ret.data) + i);       \
    v = static_cast<type>(v * rhs);                           \
  }
  if (GetTypeidFromCvType(lhs.type()) == typeid(uint8_t)) {
    UGU_MUL(uint8_t);
  } else if (GetTypeidFromCvType(lhs.type()) == typeid(int8_t)) {
    UGU_MUL(int8_t);
  } else if (GetTypeidFromCvType(lhs.type()) == typeid(uint16_t)) {
    UGU_MUL(uint16_t);
  } else if (GetTypeidFromCvType(lhs.type()) == typeid(int16_t)) {
    UGU_MUL(int16_t);
  } else if (GetTypeidFromCvType(lhs.type()) == typeid(int32_t)) {
    UGU_MUL(int32_t);
  } else if (GetTypeidFromCvType(lhs.type()) == typeid(float)) {
    UGU_MUL(float);
  } else if (GetTypeidFromCvType(lhs.type()) == typeid(double)) {
    UGU_MUL(double);
  } else {
    throw std::runtime_error("type error");
  }
#undef UGU_MUL

  return ret;
}

inline ImageBase operator/(const ImageBase& lhs, const double& rhs) {
  return lhs * (1.0 / rhs);
}

inline ImageBase operator<(const ImageBase& lhs, const double& rhs) {
  ImageBase mask(lhs.rows, lhs.cols, CV_8UC1);
  mask.setTo(0);

#define UGU_SMALLER(type, index)                                           \
  type* data = reinterpret_cast<type*>(lhs.data) + index * lhs.channels(); \
  uint8_t& val = *(mask.data + index);                                     \
  val = 255;                                                               \
  for (int c = 0; c < lhs.channels(); c++) {                               \
    if (static_cast<double>(data[c]) >= rhs) {                             \
      val = 0;                                                             \
      break;                                                               \
    }                                                                      \
  }

  auto smaller_func = [&](size_t index_) {
    if (GetTypeidFromCvType(lhs.type()) == typeid(uint8_t)) {
      UGU_SMALLER(uint8_t, index_);
    } else if (GetTypeidFromCvType(lhs.type()) == typeid(int8_t)) {
      UGU_SMALLER(int8_t, index_);
    } else if (GetTypeidFromCvType(lhs.type()) == typeid(uint16_t)) {
      UGU_SMALLER(uint16_t, index_);
    } else if (GetTypeidFromCvType(lhs.type()) == typeid(int16_t)) {
      UGU_SMALLER(int16_t, index_);
    } else if (GetTypeidFromCvType(lhs.type()) == typeid(int32_t)) {
      UGU_SMALLER(int32_t, index_);
    } else if (GetTypeidFromCvType(lhs.type()) == typeid(float)) {
      UGU_SMALLER(float, index_);
    } else if (GetTypeidFromCvType(lhs.type()) == typeid(double)) {
      UGU_SMALLER(double, index_);
    } else {
      throw std::runtime_error("type error");
    }
  };
#undef UGU_SMALLER

  parallel_for(size_t(0), lhs.total(), smaller_func);

  return mask;
}

inline ImageBase operator>(const ImageBase& lhs, const double& rhs) {
  return lhs < (-rhs);
}

template <typename _Tp, int m, int n>
inline ImageBase& operator+(ImageBase& lhs, const Matx<_Tp, m, n>& rhs) {
  assert(lhs.channels() == n);
  assert(m == 1);

#define UGU_ADD(type, index)                                               \
  type* data = reinterpret_cast<type*>(lhs.data) + index * lhs.channels(); \
  for (int c = 0; c < lhs.channels(); c++) {                               \
    data[c] += static_cast<type>(rhs[c]);                                  \
  }

  auto sum_func = [&](size_t index_) {
    if (GetTypeidFromCvType(lhs.type()) == typeid(uint8_t)) {
      UGU_ADD(uint8_t, index_);
    } else if (GetTypeidFromCvType(lhs.type()) == typeid(int8_t)) {
      UGU_ADD(int8_t, index_);
    } else if (GetTypeidFromCvType(lhs.type()) == typeid(uint16_t)) {
      UGU_ADD(uint16_t, index_);
    } else if (GetTypeidFromCvType(lhs.type()) == typeid(int16_t)) {
      UGU_ADD(int16_t, index_);
    } else if (GetTypeidFromCvType(lhs.type()) == typeid(int32_t)) {
      UGU_ADD(int32_t, index_);
    } else if (GetTypeidFromCvType(lhs.type()) == typeid(float)) {
      UGU_ADD(float, index_);
    } else if (GetTypeidFromCvType(lhs.type()) == typeid(double)) {
      UGU_ADD(double, index_);
    } else {
      throw std::runtime_error("type error");
    }
  };
#undef UGU_ADD
  parallel_for(size_t(0), lhs.total(), sum_func);

  return lhs;
}

template <typename _Tp, int m, int n>
inline ImageBase& operator-(ImageBase& lhs, const Matx<_Tp, m, n>& rhs) {
  return lhs + (-rhs);
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
  const int ch = static_cast<int>(T().channels);
  const std::type_info* info = &typeid(typename T::value_type);
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
  Image() {
    int code = ParseVec<T>();
    Init(0, 0, code);
  }
  Image(const Image& other) = default;
  Image(const ImageBase& other) : ImageBase(other) {}

  ~Image(){};

  static Image<T> zeros(int height, int width) {
    Image<T> zero(height, width);
    zero = 0.0;
    return zero;
  }

  Image<T> clone() const {
    Image<T> dst;
    copyTo(dst);
    return dst;
  }

  using ImageBase::setTo;
  void setTo(T value, InputArray mask = noArray()) {
    if (mask.empty()) {
      *this = value;
      return;
    }
    LOGE("Please implement!\n");
  }

  void forEach(std::function<void(const T&, const int[2])> f) const {
    if (empty()) {
      return;
    }
    size_t st(0);
    size_t ed = static_cast<size_t>(cols * rows);
    auto f2 = [&](const size_t& i) {
      const int xy[2] = {static_cast<int32_t>(i) % cols,
                         static_cast<int32_t>(i) / cols};
      f(reinterpret_cast<T*>(data)[i], xy);
    };
    parallel_for(st, ed, f2);
  }

  void forEach(std::function<void(T&, const int[2])> f) {
    if (empty()) {
      return;
    }
    size_t st(0);
    size_t ed = static_cast<size_t>(cols * rows);
    auto f2 = [&](const size_t& i) {
      const int xy[2] = {static_cast<int32_t>(i) % cols,
                         static_cast<int32_t>(i) / cols};
      f(reinterpret_cast<T*>(data)[i], xy);
    };
    parallel_for(st, ed, f2);
  }

  Image<T>& operator=(const T& rhs);
  Image<T>& operator=(const ImageBase& rhs);
  Image<T>& operator=(const double& rhs);
};

template <typename T>
inline Image<T>& Image<T>::operator=(const T& rhs) {
  for (size_t i = 0; i < total(); i++) {
    *(reinterpret_cast<T*>(data_->data()) + i) = rhs;
  }

  return *this;
}

template <typename T>
inline Image<T>& Image<T>::operator=(const ImageBase& rhs) {
  *this = rhs;
  return *this;
}

template <typename T>
inline Image<T>& Image<T>::operator=(const double& rhs) {
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
using Image3d = Image<Vec3d>;
using Image3b = Image<Vec3b>;  // For color image. RGB order.
using Image4b = Image<Vec4b>;  // For color image. RGBA order.
using Image4f = Image<Vec4f>;
using Image4i = Image<Vec4i>;

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
inline void Init(Image<T>* image, int width, int height, T val) {
  if (image->cols != width || image->rows != height) {
    *image = Image<T>::zeros(height, width);
  }
  image->setTo(val);
}

#endif

inline Vec3f operator*(const Eigen::Matrix3f& a, const Vec3f& b) {
  Vec3f ret;

  for (int i = 0; i < 3; i++) {
    ret[i] = a(i, 0) * b[0] + a(i, 1) * b[1] + a(i, 2) * b[2];
  }

  return ret;
}

inline Vec3f operator+(const Eigen::Vector3f& a, const Vec3f& b) {
  Vec3f ret;

  for (int i = 0; i < 3; i++) {
    ret[i] = a[i] + b[i];
  }

  return ret;
}

}  // namespace ugu
