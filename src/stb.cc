/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "ugu_stb.h"

#ifdef UGU_USE_STB
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4100)
#endif
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#ifdef _WIN32
#pragma warning(pop)
#endif

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#ifdef _WIN32
#pragma warning(pop)
#endif

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"
#ifdef _WIN32
#pragma warning(pop)
#endif

#include <cstring>

namespace {
STBIDEF unsigned char *stbi_xload(stbi__context *s, int *x, int *y, int *frames,
                                  int **delays) {
  int comp;
  unsigned char *result = 0;

  if (stbi__gif_test(s))
    return reinterpret_cast<unsigned char *>(
        stbi__load_gif_main(s, delays, x, y, frames, &comp, 4));

  stbi__result_info ri;
  result = reinterpret_cast<unsigned char *>(
      stbi__load_main(s, x, y, &comp, 4, &ri, 8));
  *frames = !!result;

  if (ri.bits_per_channel != 8) {
    STBI_ASSERT(ri.bits_per_channel == 16);
    result = stbi__convert_16_to_8((stbi__uint16 *)result, *x, *y, 4);
    ri.bits_per_channel = 8;
  }

  return result;
}
}  // namespace

namespace ugu {

STBIDEF unsigned char *stbi_xload_file(char const *filename, int *x, int *y,
                                       int *frames, int **delays) {
  FILE *f;
  stbi__context s;
  unsigned char *result = 0;
  f = stbi__fopen(filename, "rb");
  if (!f) {
    return stbi__errpuc("can't fopen", "Unable to open file");
  }

  stbi__start_file(&s, f);
  result = stbi_xload(&s, x, y, frames, delays);
  fclose(f);

  return result;
}

std::vector<unsigned char> LoadGif(const std::string &filename, int &x, int &y,
                                   int &frames, std::vector<int> &delays) {
  std::vector<unsigned char> data;

  delays.clear();

  int *tmp_delays = NULL;
  unsigned char *tmp_data =
      stbi_xload_file(filename.c_str(), &x, &y, &frames, &tmp_delays);

  if (tmp_data == NULL) {
    return data;
  }

  for (int i = 0; i < frames; i++) {
    delays.push_back(tmp_delays[i]);
  }

  // Copy data
  const size_t num_pix = static_cast<size_t>(x) * static_cast<size_t>(y) * 4 *
                         static_cast<size_t>(frames);
  data.resize(num_pix, 0);
  std::memcpy(data.data(), tmp_data, sizeof(unsigned char) * num_pix);

  // Free tmp data allocated by stb
  STBI_FREE(tmp_delays);
  STBI_FREE(tmp_data);

  return data;
}

}  // namespace ugu

#endif