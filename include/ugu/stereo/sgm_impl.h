/*
 * Copyright (C) 2020, unclearness
 * All rights reserved.
 *
 *
 * Implementation of the following paper
 *
 * 
 * 
 */

#include "ugu/stereo/base.h"

namespace ugu {

template<typename T>
bool ComputeStereoSgmImpl(const Image<T>& left, const Image<T>& right,
                          Image1f* disparity, Image1f* cost, Image1f* depth,
                          const SgmParam& param) {
  return true;
}

}  // namespace ugu
