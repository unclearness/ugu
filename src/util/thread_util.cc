/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/util/thread_util.h"

#include <cassert>
#include <mutex>
#include <thread>
#include <vector>

namespace {

template <typename Index, class Func>
void parallel_for_impl(Index st, Index ed, Func func, int num_theads = -1,
                       Index inc = Index(1)) {
  uint32_t num_cpu = num_theads > 0
                         ? std::min(static_cast<unsigned int>(num_theads),
                                    std::thread::hardware_concurrency())
                         : std::thread::hardware_concurrency();

  // Prefer to UGU_THREADS_NUM
  if (ugu::UGU_THREADS_NUM > 0) {
    num_cpu = ugu::UGU_THREADS_NUM;
  }

  assert(num_cpu > 0);

  if (num_cpu == 1) {
    for (Index j = st; j < ed; j += inc) {
      func(j);
    }
    return;
  }

  const Index jobs_per_thread =
      static_cast<Index>((ed - st + num_cpu - 1) / num_cpu);

  std::vector<std::thread> threads;

  for (Index i = st; i < ed; i += jobs_per_thread) {
    Index end_this_thread = std::min(i + jobs_per_thread, ed);
    threads.emplace_back([i, end_this_thread, inc, &func]() {
      for (Index j = i; j < end_this_thread; j += inc) {
        func(j);
      }
    });
  }

  // Wait all threads...
  for (auto& t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }
}
}  // namespace
namespace ugu {

void parallel_for(int st, int ed, std::function<void(int)> func, int num_theads,
                  int inc) {
  parallel_for_impl(st, ed, func, num_theads, inc);
}
void parallel_for(size_t st, size_t ed, std::function<void(size_t)> func,
                  int num_theads, size_t inc) {
  parallel_for_impl(st, ed, func, num_theads, inc);
}

}  // namespace ugu
