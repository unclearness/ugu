#pragma once

#include <memory>

#include "ugu/mesh.h"
#include "ugu/voxel/voxel.h"

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

class VoxelGridCudaHashing {
 public:
  VoxelGridCudaHashing();
  VoxelGridCudaHashing(int hash_table_size, int voxel_block_count, float mu,
                       float voxel_size);
  ~VoxelGridCudaHashing();
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

struct VoxelGridCudaNaiveFuseOption {
  // Followed OpenCV kinfu
  // https://github.com/opencv/opencv_contrib/blob/9d0a451bee4cdaf9d3f76912e5abac6000865f1a/modules/rgbd/src/kinfu.cpp#L66
  static inline float kDefaultResToTruncFactor = 7.f;

  float truncation_band {- 1.f};
  int sample_num{1};
  int nn_range{1};
  float weight{1.f};

  VoxelGridCudaNaiveFuseOption() = default;
  VoxelGridCudaNaiveFuseOption(float resolution){
    this->truncation_band = kDefaultResToTruncFactor * resolution;
  };
  ~VoxelGridCudaNaiveFuseOption() = default;

};

class VoxelGridCudaNaive {
 public:
  VoxelGridCudaNaive();
  ~VoxelGridCudaNaive();

  bool Init(const Eigen::Vector3f& bb_max, const Eigen::Vector3f& bb_min,
            float resolution);

  bool Init(const Eigen::Vector3f& bb_max, const Eigen::Vector3f& bb_min,
            const Eigen::Vector3f& resolution);

  void FusePointCloudMulti(const float* d_points, const float* d_normals,
                           int width, int height, int num_images,
                           const VoxelGridCudaNaiveFuseOption& option,
                           bool sync = true);

  void ExtractMesh(std::vector<Eigen::Vector3f>& vertices,
                   std::vector<Eigen::Vector3i>& faces);
  
  void ReadToCpu(ugu::VoxelGrid& grid_cpu) const;

  void Clear();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ugu
