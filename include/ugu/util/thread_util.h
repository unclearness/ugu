/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#pragma once

#include <mutex>
#include <thread>
#include <vector>

namespace ugu {

template <typename Index, class Func>
void parallel_for(Index st, Index ed, Func func, int num_theads = -1) {
  const unsigned int num_cpu =
      num_theads > 0 ? std::min(static_cast<unsigned int>(num_theads),
                                std::thread::hardware_concurrency())
                     : std::thread::hardware_concurrency();

  const Index jobs_per_thread =
      static_cast<Index>((ed - st + num_cpu - 1) / num_cpu);

  std::vector<std::thread> threads;

  for (Index i = st; i < ed; i += jobs_per_thread) {
    Index end_this_thread = std::min(i + jobs_per_thread, ed);
    threads.emplace_back([i, end_this_thread, &func]() {
      for (Index j = i; j < end_this_thread; j++) {
        func(j);
      }
    });
  }

  // Wait all threads...
  for (auto& t : threads) {
    t.join();
  }
}

}  // namespace ugu
