#pragma once

#include <cstring>
#include <string>
#include <vector>

#include "common.h"

#include "stb_image.h"
#include "stb_image_write.h"

namespace unclearness {

template <typename T, int N>
class Image {
  std::vector<T> data;
  int width_{-1};
  int height_{-1};
  const int channel_{N};

 public:
  int width() const { return width_;}
  int height() const { return height_;}
  void clear() {
    data.clear();
    width_ = -1;
    height_ = -1;
  }
  void init(int width, int height) {
    clear();
    width_ = width;
    height_ = height;
    data.resize(height_ * width_ * channel_, 0);
  }
  T* at(int x, int y) { 
      return &data[0] + (width_ * channel_ * y + x * channel_);
  }
  const T* at(int x, int y) const {
    return &data[0] + (width_ * channel_ * y + x * channel_);
  }
  T& at(int x, int y, int c) {
    return data[width_ * channel_ * y + x * channel_ + c];
  }
  const T& at(int x, int y, int c) const {
    return data[width_ * channel_ * y + x * channel_ + c];
  }

  bool load(const std::string& path) {
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

    data.resize(height_ * width_ * channel_);
    std::memcpy(&data[0], in_pixels_tmp,
                sizeof(T) * channel_ * width_ * height_);

    delete in_pixels_tmp;
    return true;
  };
  
  bool write_png(const std::string& path) const {
    stbi_write_png(path.c_str(), width_, height_, channel_, &data[0],
                   width_ * channel_ * sizeof(T));
    return true;
  }

};

using Image1b = Image<unsigned char, 1>;
using Image3b = Image<unsigned char, 3>;
using Image1w = Image<unsigned short, 1>;

}  // namespace unclearness