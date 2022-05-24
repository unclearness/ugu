/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/util/path_util.h"

#include <fstream>

#if defined(_WIN32)
#include <direct.h>
#include <io.h>
#include <shlobj.h>
#include <sys/stat.h>
#include <sys/types.h>
#else
#include <dirent.h>
#include <pwd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace ugu {

bool Exists(const std::string& path) {
#ifdef _WIN32
  struct _stat statbuf;
  if (::_stat(path.c_str(), &statbuf) < 0) {
    return false;
  }
#else   // _WIN32
  struct stat statbuf;
  if (::stat(path.c_str(), &statbuf) < 0) {
    return false;
  }
#endif  // _WIN32

  return true;
}

bool DirExists(const std::string& path) {
#ifdef _WIN32
  struct _stat statbuf;
  if (::_stat(path.c_str(), &statbuf) < 0) {
    return false;
  }

  if (!(statbuf.st_mode & _S_IFDIR)) {
    return false;
  }
#else   // _WIN32
  struct stat statbuf;
  if (::stat(path.c_str(), &statbuf) < 0) {
    return false;
  }

  if (!S_ISDIR(statbuf.st_mode)) {
    return false;
  }
#endif  // _WIN32

  return true;
}

bool FileExists(const std::string& path) {
#if 0
#ifdef _WIN32
  struct _stat statbuf;
  if (::_stat(path.c_str(), &statbuf) < 0) {
    return false;
  }

  if (!(statbuf.st_mode & _S_IFREG)) {
    return false;
  }
#else   // _WIN32
  struct stat statbuf;
  if (::stat(path.c_str(), &statbuf) < 0) {
    return false;
  }

  if (!S_ISREG(statbuf.st_mode)) {
    return false;
  }
#endif  // _WIN32

  return true;
#else
  std::ifstream ifs(path);
  return ifs.is_open();
#endif
}

bool EnsureDirExists(const std::string& path) {
  if (DirExists(path)) {
    return true;
  }
  return MkDir(path);
}

bool MkDir(const std::string& path) {
#ifdef _WIN32
  if (::_mkdir(path.c_str()) < 0) {
    return false;
  }
#else   // _WIN32
  if (::mkdir(path.c_str(), S_IRWXU | S_IRGRP | S_IXGRP) < 0) {
    return false;
  }
#endif  // _WIN32
  return true;
}

bool RmDir(const std::string& path) {
#ifdef _WIN32
  return ::_rmdir(path.c_str()) >= 0;
#else   // _WIN32
  return ::rmdir(path.c_str()) >= 0;
#endif  // _WIN32
}

bool RmFile(const std::string& path, bool check_exists) {
  if (check_exists) {
    if (!FileExists(path)) {
      // Already removed
      return true;
    }
  }

#ifdef _WIN32
  return ::_unlink(path.c_str()) >= 0;
#else   // _WIN32
  return ::unlink(path.c_str()) >= 0;
#endif  // _WIN32
}

bool Rename(const std::string& from, const std::string& to) {
  if (std::rename(from.c_str(), to.c_str()) < 0) {
    return false;
  }

  return true;
}

bool CpFile(const std::string& src, const std::string& dst) {
  std::ifstream src_stream(src, std::ios::binary);
  if (!src_stream.good()) {
    return false;
  }
  std::ofstream dst_stream(dst, std::ios::binary);
  if (!dst_stream.good()) {
    return false;
  }

  constexpr int32_t BUFFER_SIZE = 4096;
  char buffer[BUFFER_SIZE];
  while (!src_stream.eof()) {
    src_stream.read(buffer, BUFFER_SIZE);
    if (src_stream.bad()) {
      return false;
    }
    dst_stream.write(buffer, src_stream.gcount());
    if (!dst_stream.good()) {
      return false;
    }
  }

  src_stream.close();
  if (src_stream.bad()) {
    return false;
  }
  dst_stream.close();
  if (!dst_stream.good()) {
    return false;
  };

  return true;
}

}  // namespace ugu
