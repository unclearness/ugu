/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#pragma once

#include <stdint.h>

#include <functional>

namespace ugu {

// Used only if > 0
inline uint32_t UGU_THREADS_NUM = 0;

void parallel_for(int st, int ed, std::function<void(int)> func,
                  int num_theads = -1, int inc = 1);
void parallel_for(size_t st, size_t ed, std::function<void(size_t)> func,
                  int num_theads = -1, size_t inc = 1);

}  // namespace ugu
