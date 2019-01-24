/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "include/image.h"

#ifdef CURRENDER_USE_STB
#pragma warning(push)
#pragma warning(disable : 4100)
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#pragma warning(pop)

#pragma warning(push)
#pragma warning(disable : 4996)
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#pragma warning(pop)
#endif

namespace currender {

void GrayFromDepth(const Image1f& depth, Image1b* vis_depth, float min_d,
                    float max_d) {
  assert(min_d < max_d);

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

}  // namespace currender
