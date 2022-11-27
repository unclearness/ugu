/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/image.h"

namespace ugu {
#ifdef UGU_USE_OPENCV
#else
int GetBitsFromCvType(int cv_type) {
  int cv_depth = CV_MAT_DEPTH(cv_type);

  if (cv_depth < 0) {
    throw std::runtime_error("");
  }

  if (cv_depth <= CV_8S) {
    return 8;
  } else if (cv_depth <= CV_16S) {
    return 16;
  } else if (cv_depth <= CV_32S) {
    return 32;
  } else if (cv_depth <= CV_32F) {
    return 32;
  } else if (cv_depth <= CV_64F) {
    return 64;
  } else if (cv_depth <= CV_16F) {
    return 16;
  }

  throw std::runtime_error("");
}

const std::type_info& GetTypeidFromCvType(int cv_type) {
  int cv_depth = CV_MAT_DEPTH(cv_type);

  if (cv_depth < CV_8S) {
    return typeid(uint8_t);
  } else if (cv_depth < CV_16U) {
    return typeid(int8_t);
  } else if (cv_depth < CV_16S) {
    return typeid(uint16_t);
  } else if (cv_depth < CV_32S) {
    return typeid(int16_t);
  } else if (cv_depth < CV_32F) {
    return typeid(int32_t);
  } else if (cv_depth < CV_64F) {
    return typeid(float);
  } else if (cv_depth < CV_16F) {
    return typeid(double);
  }

  throw std::runtime_error("");
}

size_t SizeInBytes(const ImageBase& mat) {
  // https://stackoverflow.com/questions/26441072/finding-the-size-in-bytes-of-cvmat
  return size_t(mat.step[0] * mat.rows);
}

InputOutputArray noArray() {
  static _InputOutputArray no_array;
  return no_array;
}

int MakeCvType(const std::type_info* info, int ch) {
  int code = -1;

  if (info == &typeid(uint8_t)) {
    code = CV_MAKETYPE(CV_8U, ch);
  } else if (info == &typeid(int8_t)) {
    code = CV_MAKETYPE(CV_8S, ch);
  } else if (info == &typeid(uint16_t)) {
    code = CV_MAKETYPE(CV_16U, ch);
  } else if (info == &typeid(int16_t)) {
    code = CV_MAKETYPE(CV_16S, ch);
  } else if (info == &typeid(int32_t)) {
    code = CV_MAKETYPE(CV_32S, ch);
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

}  // namespace ugu
