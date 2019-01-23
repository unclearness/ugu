/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <stdint.h>
#include <cstring>
#include <string>
#include <vector>

#include "include/common.h"
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

namespace crender {

template <typename T, int N>
class Image {
  std::vector<T> data_;
  int width_{-1};
  int height_{-1};
  const int channel_{N};

 public:
  Image() {}
  ~Image() {}
  Image(int width, int height) { Init(width, height); }
  int width() const { return width_; }
  int height() const { return height_; }
  int channel() const { return channel_; }
  void Clear() {
    data_.clear();
    width_ = -1;
    height_ = -1;
  }
  void Init(int width, int height) {
    Clear();
    width_ = width;
    height_ = height;
    data_.resize(height_ * width_ * channel_, 0);
  }
  T* at(int x, int y) {
    return &data_[0] + (width_ * channel_ * y + x * channel_);
  }
  const T* at(int x, int y) const {
    return &data_[0] + (width_ * channel_ * y + x * channel_);
  }
  T& at(int x, int y, int c) {
    return data_[width_ * channel_ * y + x * channel_ + c];
  }
  const T& at(int x, int y, int c) const {
    return data_[width_ * channel_ * y + x * channel_ + c];
  }

  bool Load(const std::string& path) {
    unsigned char* in_pixels_tmp;
    int width;
    int height;
    int bpp;

    in_pixels_tmp = stbi_load(path.c_str(), &width, &height, &bpp, 0);

    if (bpp != channel_) {
      delete in_pixels_tmp;
      LOGE("desired channel %d, actual %d\n", channel_, bpp);
      return false;
    }

    width_ = width;
    height_ = height;

    data_.resize(height_ * width_ * channel_);
    std::memcpy(&data_[0], in_pixels_tmp,
                sizeof(T) * channel_ * width_ * height_);

    delete in_pixels_tmp;
    return true;
  }

  bool WritePng(const std::string& path) const {
    stbi_write_png(path.c_str(), width_, height_, channel_, &data_[0],
                   width_ * channel_ * sizeof(T));
    return true;
  }

  template <typename TT, int NN>
  bool ConvertTo(Image<TT, NN>* dst, float scale = 1.0f) const {
    if (channel_ != dst->channel()) {
      LOGE("ConvertTo failed src channel %d, dst channel %d\n", channel_,
           dst->channel());
      return false;
    }

    dst->Init(width_, height_);

    for (int y = 0; y < height_; y++) {
      for (int x = 0; x < width_; x++) {
        for (int c = 0; c < N; c++) {
          dst->at(x, y, c) = static_cast<TT>(scale * at(x, y, c));
        }
      }
    }
    return true;
  }
};

using Image1b = Image<uint8_t, 1>;
using Image3b = Image<uint8_t, 3>;
using Image1w = Image<uint16_t, 1>;
using Image1f = Image<float, 1>;

}  // namespace crender
