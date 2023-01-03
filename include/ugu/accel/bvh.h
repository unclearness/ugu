/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 *
 */

#pragma once

#include "ugu/accel/bvh_base.h"
#include "ugu/accel/bvh_naive.h"
#include "ugu/accel/bvh_nanort.h"

namespace ugu {

template <typename DataType, typename IndexType>
BvhPtr<DataType, IndexType> GetDefaultBvh() {
  BvhPtr<DataType, IndexType> bvh;
#ifdef UGU_USE_NANORT
  bvh = std::make_shared<BvhNanort<DataType, IndexType>>();
#else
  bvh = std::make_shared<BvhNaive<DataType, IndexType>>();
#endif
  return bvh;
}

}  // namespace ugu
