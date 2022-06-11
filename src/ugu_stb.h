#pragma once

#ifdef UGU_USE_STB
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4100)
#endif
#include "stb_image.h"
#ifdef _WIN32
#pragma warning(pop)
#endif

#include <string>
#include <vector>

namespace ugu {
STBIDEF unsigned char* stbi_xload_file(char const* filename, int* x, int* y,
                                       int* frames, int** delays);

std::vector<unsigned char> LoadGif(const std::string& filename, int& x, int& y,
                                   int& frames, std::vector<int>& delays);
}  // namespace ugu

#endif
