/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/util/string_util.h"

#include <algorithm>

namespace {
std::string::size_type FindLastSeparatorPos(const std::string& path) {
  std::string::size_type pos = std::string::npos;
  const std::string::size_type unix_pos = path.find_last_of('/');
  const std::string::size_type windows_pos = path.find_last_of('\\');
  if (unix_pos != std::string::npos) {
    if (pos == std::string::npos) {
      pos = unix_pos;
    } else {
      pos = std::max(pos, unix_pos);
    }
  }
  if (windows_pos != std::string::npos) {
    if (pos == std::string::npos) {
      pos = windows_pos;
    } else {
      pos = std::max(pos, unix_pos);
    }
  }
  return pos;
}

}  // namespace

namespace ugu {

std::string ExtractFilename(const std::string& path, bool without_ext) {
  std::string::size_type pos = FindLastSeparatorPos(path);
  std::string fn =
      (pos == std::string::npos) ? path : path.substr(pos + 1, path.size());
  if (without_ext) {
    return ExtractPathWithoutExt(fn);
  }
  return fn;
}

std::string ExtractDir(const std::string& path) {
  std::string::size_type pos = FindLastSeparatorPos(path);
  return (pos == std::string::npos) ? "./" : path.substr(0, pos + 1);
}

std::string ExtractExt(const std::string& path, bool no_dot) {
  size_t ext_i = path.find_last_of(".");
  if (ext_i == std::string::npos) {
    return "";
  }
  std::string extname = no_dot ? path.substr(ext_i + 1, path.size() - ext_i)
                               : path.substr(ext_i, path.size() - ext_i);
  return extname;
}

std::vector<std::string> Split(const std::string& input, char delimiter) {
  std::istringstream stream(input);
  std::string field;
  std::vector<std::string> result;
  while (std::getline(stream, field, delimiter)) {
    result.push_back(field);
  }
  return result;
}

std::streamoff stream_size(std::istream& f) {
  std::istream::pos_type current_pos = f.tellg();
  if (-1 == current_pos) {
    return -1;
  }
  f.seekg(0, std::istream::end);
  std::istream::pos_type end_pos = f.tellg();
  f.seekg(current_pos);
  return end_pos - current_pos;
}

bool stream_read_string(std::istream& f,
                        std::string& result) {  // NOLINT
  std::streamoff len = stream_size(f);
  if (len == -1) {
    return false;
  }

  result.resize(static_cast<std::string::size_type>(len));

  f.read(&result[0], result.length());
  return true;
}

std::string ExtractPathWithoutExt(const std::string& fn) {
  std::string::size_type pos;
  if ((pos = fn.find_last_of(".")) == std::string::npos) {
    return fn;
  }

  return fn.substr(0, pos);
}

std::string ReplaceExtention(const std::string& path, const std::string& ext) {
  return ExtractPathWithoutExt(path) + ext;
}

}  // namespace ugu
