/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

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
#include "stb_image_resize.h"
#ifdef _WIN32
#pragma warning(pop)
#endif
#endif
