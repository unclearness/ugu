/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "currender/common.h"

#ifdef CURRENDER_USE_STB
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#endif

#ifdef CURRENDER_USE_LODEPNG
#include "lodepng/lodepng.h"
#endif

#ifdef CURRENDER_USE_TINYCOLORMAP
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4067)
#endif
#include "tinycolormap/include/tinycolormap.hpp"
#ifdef _WIN32
#pragma warning(pop)
#endif
#endif

namespace currender {

template <typename T, int N>
class Image {
  std::vector<T> data_;
  int width_{-1};
  int height_{-1};
  const int bit_depth_{sizeof(T)};
  const int channel_{N};

 public:
  Image() {}
  ~Image() {}
  Image(const Image<T, N>& src) { src.copyTo(*this); }
  Image(int width, int height) { Init(width, height); }
  Image(int width, int height, T val) { Init(width, height, val); }
  int width() const { return width_; }
  int height() const { return height_; }
  int channel() const { return channel_; }
  unsigned char* data;
  void Clear() {
    data_.clear();
    width_ = -1;
    height_ = -1;
  }
  bool empty() const {
    if (width_ < 0 || height_ < 0 || data_.empty()) {
      return true;
    }
    return false;
  }
  void Init(int width, int height, T val = 0) {
    Clear();
    width_ = width;
    height_ = height;
    data_.resize(height_ * width_ * channel_, val);
    data = reinterpret_cast<unsigned char*>(&data_[0]);
  }

#ifdef CURRENDER_USE_STB
  bool Load(const std::string& path) {
    unsigned char* in_pixels_tmp;
    int width;
    int height;
    int bpp;

    if (bit_depth_ == 2) {
      in_pixels_tmp = reinterpret_cast<unsigned char*>(
          stbi_load_16(path.c_str(), &width, &height, &bpp, channel_));
    } else if (bit_depth_ == 1) {
      in_pixels_tmp = stbi_load(path.c_str(), &width, &height, &bpp, channel_);
    } else {
      LOGE("Load() for bit_depth %d and channel %d is not supported\n",
           bit_depth_, channel_);
      return false;
    }

    if (bpp != channel_) {
      delete in_pixels_tmp;
      LOGE("desired channel %d, actual %d\n", channel_, bpp);
      return false;
    }

    width_ = width;
    height_ = height;
    data_.resize(height_ * width_ * channel_);
    data = &data_[0];

    std::memcpy(&data_[0], in_pixels_tmp,
                sizeof(T) * channel_ * width_ * height_);
    delete in_pixels_tmp;

    return true;
  }

#ifdef CURRENDER_USE_LODEPNG
  // https://github.com/lvandeve/lodepng/issues/74#issuecomment-405049566
  bool WritePng16Bit1Channel(const std::string& path) const {
    if (bit_depth_ != 2 || channel_ != 1) {
      LOGE("WritePng16Bit1Channel invalid bit_depth %d or channel %d\n",
           bit_depth_, channel_);
      return false;
    }
    std::vector<unsigned char> data_8bit;
    data_8bit.resize(width_ * height_ * 2);  // 2 bytes per pixel
    const int kMostMask = 0b1111111100000000;
    const int kLeastMask = ~kMostMask;
    for (int y = 0; y < height_; y++) {
      for (int x = 0; x < width_; x++) {
        std::uint16_t d = at(*this, x, y, 0);
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
#ifdef CURRENDER_USE_LODEPNG
    if (bit_depth_ == 2 && channel_ == 1) {
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

    stbi_write_png(path.c_str(), width_, height_, channel_, &data_[0],
                   width_ * channel_ * sizeof(T));
    return true;
  }
#endif

  bool copyTo(Image<T, N>& dst) const {
    if (channel_ != dst.channel()) {
      LOGE("ConvertTo failed src channel %d, dst channel %d\n", channel_,
           dst.channel());
      return false;
    }

    dst.Init(width_, height_);

    std::memcpy(dst.data, data, sizeof(T) * height_ * width_ * channel_);

    return true;
  }
};

template <typename T, int N>
T& at(Image<T, N>* image, int x, int y, int c) {
  assert(0 <= x && x < image->width() && 0 <= y && y < image->height() &&
         0 <= c && c < image->channel());
  return reinterpret_cast<T*>(
      image->data)[image->width() * image->channel() * y +
                   x * image->channel() + c];
}

template <typename T, int N>
const T& at(const Image<T, N>& image, int x, int y, int c) {
  assert(0 <= x && x < image.width() && 0 <= y && y < image.height() &&
         0 <= c && c < image.channel());
  return reinterpret_cast<T*>(image.data)[image.width() * image.channel() * y +
                                          x * image.channel() + c];
}

template <typename T, int N, typename TT, int NN>
bool ConvertTo(const Image<T, N>& src, Image<TT, NN>* dst, float scale = 1.0f) {
  if (src.channel() != dst->channel()) {
    LOGE("ConvertTo failed src channel %d, dst channel %d\n", src.channel(),
         dst->channel());
    return false;
  }

  dst->Init(src.width(), src.height());

  for (int y = 0; y < src.height(); y++) {
    for (int x = 0; x < src.width(); x++) {
      for (int c = 0; c < N; c++) {
        at(dst, x, y, c) = static_cast<TT>(scale * at(src, x, y, c));
      }
    }
  }

  return true;
}

using Image1b = Image<uint8_t, 1>;   // For gray image.
using Image3b = Image<uint8_t, 3>;   // For color image. RGB order.
using Image1w = Image<uint16_t, 1>;  // For depth image with 16 bit (unsigned
                                     // short) mm-scale format
using Image1i =
    Image<int32_t, 1>;  // For face visibility. face id is within int32_t
using Image1f = Image<float, 1>;  // For depth image with any scale
using Image3f = Image<float, 3>;  // For normal or point cloud. XYZ order.

void Depth2Gray(const Image1f& depth, Image1b* vis_depth, float min_d = 200.0f,
                float max_d = 1500.0f);

void Normal2Color(const Image3f& normal, Image3b* vis_normal);

void FaceId2RandomColor(const Image1i& face_id, Image3b* vis_face_id);

#ifdef CURRENDER_USE_TINYCOLORMAP
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

}  // namespace currender
