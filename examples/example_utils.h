/*
 * Copyright (C) 2026, unclearness
 * All rights reserved.
 */

#pragma once

#include <string>

#include "ugu/util/path_util.h"

// Helpers shared by the exNN demos. All examples assume CWD = bin/:
// inputs are read from ../data/, outputs go to ../out/<example_name>/.
namespace ugu_example {

// "../data/" or "../data/<subdir>/" (trailing slash, matching the
// `dir + "file.ext"` concatenation style used by the examples).
inline std::string GetDataDir(const std::string& subdir = "") {
  std::string dir = "../data/";
  if (!subdir.empty()) {
    dir += subdir + "/";
  }
  return dir;
}

// "../out/<ex_name>/[<subdir>/]" with trailing slash. Directories are
// created level by level because ugu::EnsureDirExists does not create
// parents.
inline std::string GetOutDir(const std::string& ex_name,
                             const std::string& subdir = "") {
  std::string dir = "../out/";
  ugu::EnsureDirExists(dir);
  dir += ex_name + "/";
  ugu::EnsureDirExists(dir);
  if (!subdir.empty()) {
    dir += subdir + "/";
    ugu::EnsureDirExists(dir);
  }
  return dir;
}

// ex02_renderer renders the bunny (tumpose.txt, NNNNN_color.png,
// NNNNN_depth.png, NNNNN_mask.png, r_NNNNN_color.png, ...) into this
// directory. ex01, ex06, ex10 and ex12 read from here: run ex02_renderer
// first. Does not create the directory.
inline std::string GetEx02RenderedBunnyDir() { return "../out/ex02_renderer/"; }

}  // namespace ugu_example
