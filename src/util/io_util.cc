/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/util/io_util.h"

#include <fstream>

namespace ugu {

std::string LoadTxt(const std::string& path) {
  std::ifstream ifs(path);
  if (ifs.fail()) {
    return "";
  }
  std::string str((std::istreambuf_iterator<char>(ifs)),
                  std::istreambuf_iterator<char>());
  return str;
}

void WriteFaceIdAsText(const Image1i& face_id, const std::string& path) {
  std::ofstream ofs;
  ofs.open(path, std::ios::out);

  for (int y = 0; y < face_id.rows; y++) {
    for (int x = 0; x < face_id.cols; x++) {
      ofs << face_id.at<int>(y, x) << "\n";
    }
  }

  ofs.flush();
}

}  // namespace ugu
