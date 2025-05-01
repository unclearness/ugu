#include "ugu/cuda/voxel.h"

#ifdef UGU_USE_CUDA
#include <cuda_runtime.h>

#include "./helper_cuda.h"
#include "./voxel.cuh"

#endif

namespace ugu {

#ifdef UGU_USE_CUDA

#else
VoxelGridCuda::VoxelGridCuda(int hash_table_size, int voxel_block_count,
                             float mu, float voxel_size) {}

VoxelGridCuda::~VoxelGridCuda() {}

void VoxelGridCuda::Init(int hash_table_size, int voxel_block_count, float mu,
                         float voxel_size) {}

#endif

}  // namespace ugu
