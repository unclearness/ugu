/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/util/string_util.h"

#include <algorithm>

namespace ugu {

std::string ExtractDir(const std::string& path) {
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
  return (pos == std::string::npos) ? "./" : path.substr(0, pos + 1);
}

std::string ExtractExt(const std::string& path, bool no_dot) {
  size_t ext_i = path.find_last_of(".");
  if (ext_i == std::string::npos) {
    // TODO
  }
  std::string extname = no_dot ? path.substr(ext_i + 1, path.size() - ext_i)
                               : path.substr(ext_i, path.size() - ext_i);
  return extname;
}

}  // namespace ugu
