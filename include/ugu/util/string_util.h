/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#pragma once

#include <string>

namespace ugu {

std::string ExtractDir(const std::string& path);
std::string ExtractExt(const std::string& path, bool no_dot = true);

}  // namespace ugu
