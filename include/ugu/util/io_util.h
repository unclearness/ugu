/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <fstream>

#include "ugu/image.h"
#include "ugu/util/string_util.h"

namespace ugu {

void WriteFaceIdAsText(const Image1i& face_id, const std::string& path);

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

template <typename T>
std::vector<T> LoadTxt(const std::string& path) {
  std::ifstream ifs(path);
  std::vector<T> data;

  std::string line;
  while (std::getline(ifs, line)) {
    data.push_back(static_cast<T>(std::atof(line.c_str())));
  }
  return data;
}

template <typename T>
std::vector<std::vector<T>> LoadTxt(const std::string& path,
                                    const char sep = ' ') {
  std::ifstream ifs(path);
  std::vector<std::vector<T>> data_list;

  std::string line;
  while (std::getline(ifs, line)) {
    std::vector<std::string> splited = Split(line, sep);

    std::vector<T> data;
    for (const auto& s : splited) {
      data.push_back(static_cast<T>(std::atof(s.c_str())));
    }

    data_list.push_back(data);
  }

  return data_list;
}

template <typename T>
std::vector<T> LoadTxtAsEigenVec(const std::string& path,
                                 const char sep = ' ') {
  std::ifstream ifs(path);
  std::vector<T> data;

  std::string line;
  while (std::getline(ifs, line)) {
    if (line.substr(0, 1) == "#") {
      continue;
    }

    std::vector<std::string> splited = Split(line, sep);

    if (splited.size() != T::RowsAtCompileTime) {
      LOGW("load error\n");
      continue;
    }

    std::vector<double> v;
    for (const auto& s : splited) {
      v.push_back(std::atof(s.c_str()));
    }

    Eigen::VectorXd v2 =
        Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(v.data(), v.size());

    data.push_back(T(v2.cast<T::Scalar>()));
  }
  return data;
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
