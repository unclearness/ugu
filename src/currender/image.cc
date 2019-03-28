/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "currender/image.h"

#include <random>

#ifdef CURRENDER_USE_STB
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4100)
#endif
#include "stb/stb_image.h"
#ifdef _WIN32
#pragma warning(pop)
#endif

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
#include "stb/stb_image_write.h"
#ifdef _WIN32
#pragma warning(pop)
#endif
#endif

namespace currender {

void Depth2Gray(const Image1f& depth, Image1b* vis_depth, float min_d,
                float max_d) {
  assert(min_d < max_d);
  assert(vis_depth != nullptr);

  vis_depth->Init(depth.width(), depth.height());

  for (int y = 0; y < vis_depth->height(); y++) {
    for (int x = 0; x < vis_depth->width(); x++) {
      auto d = depth.at(x, y, 0);
      if (d < 1) {
        continue;
      }

      int color = static_cast<int>((d - min_d) / (max_d - min_d) * 255.0);

      if (color < 0) {
        color = 0;
      }
      if (255 < color) {
        color = 255;
      }

      vis_depth->at(x, y, 0) = static_cast<uint8_t>(color);
    }
  }
}

void Normal2Color(const Image3f& normal, Image3b* vis_normal) {
  assert(vis_normal != nullptr);

  vis_normal->Init(normal.width(), normal.height());

  // Followed https://en.wikipedia.org/wiki/Normal_mapping
  // X: -1 to +1 :  Red: 0 to 255
  // Y: -1 to +1 :  Green: 0 to 255
  // Z: 0 to -1 :  Blue: 128 to 255
  for (int y = 0; y < vis_normal->height(); y++) {
    for (int x = 0; x < vis_normal->width(); x++) {
      vis_normal->at(x, y, 0) = static_cast<uint8_t>(
          std::round((normal.at(x, y, 0) + 1.0) * 0.5 * 255));
      vis_normal->at(x, y, 1) = static_cast<uint8_t>(
          std::round((normal.at(x, y, 1) + 1.0) * 0.5 * 255));
      vis_normal->at(x, y, 2) =
          static_cast<uint8_t>(std::round(-normal.at(x, y, 2) * 127.0) + 128);
    }
  }
}

void FaceId2RandomColor(const Image1i& face_id, Image3b* vis_face_id) {
  assert(vis_face_id != nullptr);

  vis_face_id->Init(face_id.width(), face_id.height(), 0);

  for (int y = 0; y < vis_face_id->height(); y++) {
    for (int x = 0; x < vis_face_id->width(); x++) {
      int fid = face_id.at(x, y, 0);
      if (fid < 0) {
        continue;
      }

      // todo: cache for speed up
      std::mt19937 mt(fid);
      // stl distribution depends on environment while mt19937 is independent.
      // so simply mod mt19937 value for random color reproducing the same
      // color in different environment.
      vis_face_id->at(x, y, 0) = static_cast<uint8_t>(mt() % 256);
      vis_face_id->at(x, y, 1) = static_cast<uint8_t>(mt() % 256);
      vis_face_id->at(x, y, 2) = static_cast<uint8_t>(mt() % 256);
    }
  }
}

}  // namespace currender
