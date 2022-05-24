/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#pragma once

#include <string>

namespace ugu {

bool Exists(const std::string& path);

bool DirExists(const std::string& path);

bool FileExists(const std::string& path);

bool EnsureDirExists(const std::string& path);

bool MkDir(const std::string& path);

bool RmDir(const std::string& path);

bool RmFile(const std::string& path, bool check_exists = true);

bool Rename(const std::string& from, const std::string& to);

bool CpFile(const std::string& src, const std::string& dst);

}  // namespace ugu
