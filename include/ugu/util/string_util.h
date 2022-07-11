/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#pragma once

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace ugu {

std::string ExtractDir(const std::string& path);
std::string ExtractExt(const std::string& path, bool no_dot = true);
std::string ExtractPathWithoutExt(const std::string& fn);
std::string ExtractPathExt(const std::string& fn);
std::string ReplaceExtention(const std::string& path, const std::string& ext);

std::vector<std::string> Split(const std::string& input, char delimiter);

template <typename T>
std::string zfill(const T& val, int num = 5) {
  std::ostringstream sout;
  sout << std::setfill('0') << std::setw(num) << val;
  return sout.str();
}

// Optimized C++ 11.1.6
std::streamoff stream_size(std::istream& f);
bool stream_read_string(std::istream& f, std::string& result);

}  // namespace ugu
