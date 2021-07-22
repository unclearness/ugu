/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <fstream>

#include "ugu/image.h"

namespace ugu {

void WriteFaceIdAsText(const Image1i& face_id, const std::string& path);

inline bool WriteBinary(const std::string& path, void* data, size_t size) {
  std::ofstream ofs(path, std::ios::binary);
  ofs.write(reinterpret_cast<char*>(data), size);

  if (ofs.bad()) {
    return false;
  }
  return true;
}

inline bool LoadBinaryBase(const std::string& path, std::vector<char>* data) {
  std::ifstream ifs(path, std::ios::binary);

  ifs.seekg(0, std::ios::end);
  long long int size = ifs.tellg();
  ifs.seekg(0);

  data->resize(size);
  ifs.read(data->data(), size);

  return true;
}

inline bool LoadBinaryBase(const std::string& path, char* data) {
  std::ifstream ifs(path, std::ios::binary);

  ifs.seekg(0, std::ios::end);
  long long int size = ifs.tellg();
  ifs.seekg(0);

  ifs.read(data, size);

  return true;
}

template <typename T>
bool LoadBinary(const std::string& path, std::vector<T>* data) {
  std::vector<char> internal_data;
  LoadBinaryBase(path, &internal_data);
  size_t elem_num = internal_data.size() / sizeof(T);
  data->resize(elem_num);

  std::memcpy(data->data(), internal_data.data(), internal_data.size());

  return true;
}

#ifdef UGU_USE_OPENCV
template <typename T>
bool WriteBinary(const std::string& path, ugu::Image<T>& image) {
  size_t size_in_bytes = image.total() * image.elemSize();
  return WriteBinary(path, image.data, size_in_bytes);
}

template <typename T>
bool LoadBinary(const std::string& path, ugu::Image<T>& image) {
  std::vector<char> internal_data;
  LoadBinaryBase(path, &internal_data);

  size_t size_in_bytes = image.total() * image.elemSize();

  if (size_in_bytes != internal_data.size()) {
    return false;
  }

  std::memcpy(image.data, internal_data.data(), size_in_bytes);

  return true;
}

#endif

}  // namespace ugu
