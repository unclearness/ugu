#include <cuda_runtime.h>

#include <iostream>
#include <cfloat>
#include <cstdint>

#include "./helper_cuda.h"
#include "./knn.cuh"

namespace {

// Shared kNN search body. `local_distances`/`local_indices` must have room
// for k entries; they hold the ascending sorted result when this returns.
__device__ inline void kNNSearchBody(
    const Eigen::Vector3f* d_points, const uint32_t* d_voxel_start_indices,
    const uint32_t* d_voxel_point_indices, uint32_t grid_size_x,
    uint32_t grid_size_y, uint32_t grid_size_z, float voxel_len,
    const Eigen::Vector3f& min_bound, const Eigen::Vector3f& query, uint32_t k,
    float* local_distances, uint32_t* local_indices) {
  uint32_t vx = static_cast<uint32_t>((query.x() - min_bound.x()) / voxel_len);
  uint32_t vy = static_cast<uint32_t>((query.y() - min_bound.y()) / voxel_len);
  uint32_t vz = static_cast<uint32_t>((query.z() - min_bound.z()) / voxel_len);

  vx = min(max(vx, 0), grid_size_x - 1);
  vy = min(max(vy, 0), grid_size_y - 1);
  vz = min(max(vz, 0), grid_size_z - 1);

  for (uint32_t i = 0; i < k; ++i) {
    local_distances[i] = FLT_MAX;
    local_indices[i] = UINT32_MAX;
  }

  // Determine the range of search. Count only the newly added shell
  // (Chebyshev distance == shell) per iteration instead of rescanning the
  // whole cube; the accumulated total matches the original full-cube count.
  const int max_range =
      static_cast<int>((grid_size_x + grid_size_y + grid_size_z) / 3 / 2);
  int range = 0;
  {
    int range_count = 0;
    uint32_t num_data_total = 0;
    for (int shell = 0; true; ++shell) {
      for (int dx = -shell; dx <= shell; ++dx) {
        for (int dy = -shell; dy <= shell; ++dy) {
          for (int dz = -shell; dz <= shell; ++dz) {
            if (max(abs(dx), max(abs(dy), abs(dz))) != shell) continue;

            int nx = vx + dx;
            int ny = vy + dy;
            int nz = vz + dz;

            if (nx < 0 || ny < 0 || nz < 0 || nx >= grid_size_x ||
                ny >= grid_size_y || nz >= grid_size_z)
              continue;

            uint32_t neighbor_voxel =
                nx + ny * grid_size_x + nz * grid_size_x * grid_size_y;
            num_data_total += d_voxel_start_indices[neighbor_voxel + 1] -
                              d_voxel_start_indices[neighbor_voxel];
          }
        }
      }
      if (shell == 0) continue;  // First check happens after shell 1
      range_count = shell - 1;
      if (range_count > max_range || k <= num_data_total) {
        range = range_count;
        break;
      }
    }
  }

  // TODO: This range is still not accurate. Accuracy depends on the postion of
  // the query point in the voxel.

  // Perform kNN search with the range
  int range_max = range + 1;
  int range_min = -range_max;
  // Current k-th best distance; skip the O(k) insertion scan for the vast
  // majority of candidates that cannot enter the list.
  float worst = FLT_MAX;
  for (int dx = range_min; dx <= range_max; ++dx) {
    for (int dy = range_min; dy <= range_max; ++dy) {
      for (int dz = range_min; dz <= range_max; ++dz) {
        int nx = vx + dx;
        int ny = vy + dy;
        int nz = vz + dz;

        if (nx < 0 || ny < 0 || nz < 0 || nx >= grid_size_x ||
            ny >= grid_size_y || nz >= grid_size_z)
          continue;

        uint32_t neighbor_voxel =
            nx + ny * grid_size_x + nz * grid_size_x * grid_size_y;
        uint32_t start = d_voxel_start_indices[neighbor_voxel];
        uint32_t end = d_voxel_start_indices[neighbor_voxel + 1];

        for (uint32_t p = start; p < end; ++p) {
          uint32_t point_idx = d_voxel_point_indices[p];
          if (point_idx == UINT32_MAX) {
            continue;
          }

          Eigen::Vector3f point = d_points[point_idx];
          float distance = (query - point).squaredNorm();
          if (distance >= worst) {
            continue;
          }

          // Update kNN (keep the list sorted ascending)
          for (uint32_t i = 0; i < k; ++i) {
            if (distance < local_distances[i]) {
              // Shift to the right
              for (uint32_t j = k - 1; j > i; --j) {
                local_distances[j] = local_distances[j - 1];
                local_indices[j] = local_indices[j - 1];
              }
              local_distances[i] = distance;
              local_indices[i] = point_idx;
              break;
            }
          }
          worst = local_distances[k - 1];
        }
      }
    }
  }
}

// This thread's slice of the output buffers is the working list. Never
// allocate device heap memory inside a kernel: it serializes on the heap
// allocator and easily exhausts the (default 8MB) heap.
// (A fixed-size local-memory variant was measured slower here due to
// register pressure/occupancy, so the output-slice approach is used for
// all k.)
__global__ void kNNKernel(const Eigen::Vector3f* d_points,
                          const uint32_t* d_voxel_start_indices,
                          const uint32_t* d_voxel_point_indices,
                          uint32_t grid_size_x, uint32_t grid_size_y,
                          uint32_t grid_size_z, float voxel_len,
                          Eigen::Vector3f min_bound,
                          const Eigen::Vector3f* d_queries,
                          uint32_t num_queries, uint32_t k,
                          uint32_t* d_knn_indices, float* d_knn_dists) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_queries) return;

  float* local_distances = d_knn_dists + static_cast<size_t>(idx) * k;
  uint32_t* local_indices = d_knn_indices + static_cast<size_t>(idx) * k;

  kNNSearchBody(d_points, d_voxel_start_indices, d_voxel_point_indices,
                grid_size_x, grid_size_y, grid_size_z, voxel_len, min_bound,
                d_queries[idx], k, local_distances, local_indices);
}

}  // namespace

namespace ugu {

void SearchKnnCuda(const float* d_data, const uint32_t* d_voxel_start_indices,
                   const uint32_t* d_voxel_point_indices, uint32_t grid_size_x,
                   uint32_t grid_size_y, uint32_t grid_size_z, float voxel_len,
                   Eigen::Vector3f min_bound, float* d_queries,
                   uint32_t num_queries, uint32_t k, uint32_t* d_knn_indices,
                   float* d_knn_dists) {
  int threads = 256;
  int blocks = (num_queries + threads - 1) / threads;

  const Eigen::Vector3f* points =
      reinterpret_cast<const Eigen::Vector3f*>(d_data);
  const Eigen::Vector3f* queries =
      reinterpret_cast<const Eigen::Vector3f*>(d_queries);

  kNNKernel<<<blocks, threads>>>(points, d_voxel_start_indices,
                                 d_voxel_point_indices, grid_size_x,
                                 grid_size_y, grid_size_z, voxel_len,
                                 min_bound, queries, num_queries, k,
                                 d_knn_indices, d_knn_dists);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
}

}  // namespace ugu
