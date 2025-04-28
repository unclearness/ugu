#pragma once

#include <memory>

#ifdef UGU_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace ugu {

#ifndef UGU_USE_CUDA
struct float3 {
  float x, y, z;
};
#endif

// Marching Cubes 用の頂点構造体
struct VertexHostDevice {
  float3 position;
  float3 normal;
};

// メッシュ構造体（ホスト側）
struct MeshHostDevice {
  VertexHostDevice* vertices{nullptr};
  int vertex_count{0};
  int* indices{nullptr};
  int index_count{0};
  int max_vertex_count{0};
  int max_index_count{0};
  MeshHostDevice();
  ~MeshHostDevice();
  void Reseave(int max_vertex_count, int max_index_count);
};

class VoxelGridCuda {
 public:
  VoxelGridCuda();
  VoxelGridCuda(int hash_table_size, int voxel_block_count, float mu,
                float voxel_size);
  ~VoxelGridCuda();
  void Init(int hash_table_size, int voxel_block_count, float mu,
            float voxel_size);
  void FusePointCloud(const float* d_points, const float* d_normals,
                      uint32_t num_points, bool sync = true);

  void FuseOrganizedPointCloudMulti(const float* d_points,
                                    const float* d_normals, int width,
                                    int height, int num_images,
                                    bool sync = true);
  void GenerateMesh(MeshHostDevice& mesh, bool connected = true); 

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ugu
