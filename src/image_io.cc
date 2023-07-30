/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/image_io.h"

#include "ugu/util/io_util.h"

#ifdef UGU_USE_OPENCV
#else

namespace {
using namespace ugu;

#ifdef UGU_USE_STB
bool LoadByStb(ImageBase& img, const std::string& path) {
  int bit_depth = 0;
  // https://stackoverflow.com/questions/6278159/find-out-if-png-is-8-or-24
  {
    std::ifstream fin(path, std::ios::in | std::ios::binary);
    fin.seekg(24);
    bit_depth = fin.get();
  }

  uint8_t* in_pixels_tmp = NULL;
  int width = -1;
  int height = -1;
  int bpp = -1;

  const int desired_ch = 0;  // Not desire
  int cv_type = -1;

  if (bit_depth == 8) {
    in_pixels_tmp = stbi_load(path.c_str(), &width, &height, &bpp, desired_ch);
    cv_type = MakeCvType(&typeid(uint8_t), bpp);
  } else if (bit_depth == 16 || in_pixels_tmp == NULL) {
    in_pixels_tmp = reinterpret_cast<uint8_t*>(
        stbi_load_16(path.c_str(), &width, &height, &bpp, desired_ch));
    cv_type = MakeCvType(&typeid(uint16_t), bpp);
  } else {
    LOGE("Load failed. bit depth is %d\n", bit_depth);
    return false;
  }

  if (in_pixels_tmp == NULL) {
    LOGE("Load failed\n");
    return false;
  }

  img = ImageBase(height, width, cv_type);
  std::memcpy(img.data, in_pixels_tmp, SizeInBytes(img));
  stbi_image_free(in_pixels_tmp);

  return true;
}

#ifdef UGU_USE_LODEPNG
// https://github.com/lvandeve/lodepng/issues/74#issuecomment-405049566
bool WritePng16Bit1Channel(const ImageBase& img, const std::string& path) {
  const int type = img.type();
  // const int cv_depth = CV_MAT_DEPTH(type);
  const int cv_ch = CV_GETCN(type);
  const int bit_depth_ = GetBitsFromCvType(type) / 8;

  if (bit_depth_ != 2 || cv_ch != 1) {
    LOGE("WritePng16Bit1Channel invalid bit_depth %d or channel %d\n",
         bit_depth_, cv_ch);
    return false;
  }
  std::vector<uint8_t> data_8bit;
  data_8bit.resize(img.cols * img.rows * 2);  // 2 bytes per pixel
  const int kMostMask = 0b1111111100000000;
  const int kLeastMask = ~kMostMask;
  for (int y = 0; y < img.rows; y++) {
    for (int x = 0; x < img.cols; x++) {
      std::uint16_t d = img.at<std::uint16_t>(y, x);  // At(*this, x, y, 0);
      data_8bit[2 * img.cols * y + 2 * x + 0] =
          static_cast<uint8_t>((d & kMostMask) >> 8);  // most significant
      data_8bit[2 * img.cols * y + 2 * x + 1] =
          static_cast<uint8_t>(d & kLeastMask);  // least significant
    }
  }
  unsigned error = lodepng::encode(
      path, data_8bit, img.cols, img.rows, LCT_GREY,
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

bool WritePng(const ImageBase& img, const std::string& path) {
  const int type = img.type();
  // const int cv_depth = CV_MAT_DEPTH(type);
  const int cv_ch = CV_GETCN(type);
  const int bit_depth_ = GetBitsFromCvType(type) / 8;

#ifdef UGU_USE_LODEPNG
  if (bit_depth_ == 2 && cv_ch == 1) {
    return WritePng16Bit1Channel(img, path);
  }
#endif

  if (bit_depth_ != 1) {
    LOGE("1 byte per channel is required to save by stb_image: actual %d\n",
         bit_depth_);
    return false;
  }

  if (img.cols < 0 || img.rows < 0) {
    LOGE("image is empty\n");
    return false;
  }

  int ret = stbi_write_png(path.c_str(), img.cols, img.rows, cv_ch, img.data,
                           static_cast<int>(img.cols * img.elemSize()));
  return ret != 0;
}

bool WriteJpg(const ImageBase& img, const std::string& path) {
  const int type = img.type();
  //  const int cv_depth = CV_MAT_DEPTH(type);
  const int cv_ch = CV_GETCN(type);
  const int bit_depth_ = GetBitsFromCvType(type) / 8;

  if (bit_depth_ != 1) {
    LOGE("1 byte per channel is required to save by stb_image: actual %d\n",
         bit_depth_);
    return false;
  }

  if (img.cols < 0 || img.rows < 0) {
    LOGE("image is empty\n");
    return false;
  }

  if (cv_ch > 3) {
    LOGW("alpha channel is ignored to save as .jpg. channels(): %d\n", cv_ch);
  }

  // JPEG does ignore alpha channels in input data; quality is between 1
  // and 100. Higher quality looks better but results in a bigger image.
  const int max_quality{100};

  int ret = stbi_write_jpg(path.c_str(), img.cols, img.rows, cv_ch, img.data,
                           max_quality);
  return ret != 0;
}
#else
bool LoadByStb(ImageBase& img, const std::string& path) {
  (void)path;
  LOGE("can't load image with this configuration\n");
  return false;
}

bool WritePng(const ImageBase& img, const std::string& path) const {
  (void)path;
  LOGE("can't write image with this configuration\n");
  return false;
}

bool WriteJpg(const ImageBase& img, const std::string& path) const {
  (void)path;
  LOGE("can't write image with this configuration\n");
  return false;
}
#endif

}  // namespace

namespace ugu {

bool WriteBinary(const ImageBase& img, const std::string& path) {
  return WriteBinary(path, img.data, SizeInBytes(img));
}

bool imwrite(const std::string& filename, const ImageBase& img,
             const std::vector<int>& params) {
  (void)params;
  if (filename.size() < 4) {
    return false;
  }

  size_t ext_i = filename.find_last_of(".");
  std::string extname = filename.substr(ext_i, filename.size() - ext_i);
  if (extname == ".png" || extname == ".PNG") {
    return WritePng(img, filename);
  } else if (extname == ".jpg" || extname == ".jpeg" || extname == ".JPG" ||
             extname == ".JPEG") {
    return WriteJpg(img, filename);
  } else if (extname == ".bin" || extname == ".BIN") {
    return WriteBinary(img, filename);
  }

  LOGE(
      "acceptable extention is .png, .jpg or .jpeg. this extention is not "
      "supported: %s\n",
      filename.c_str());
  return false;
}

ImageBase imread(const std::string& filename, int flags) {
  (void)flags;
  ImageBase loaded;
  LoadByStb(loaded, filename);
  return loaded;
}

}  // namespace ugu
#endif
