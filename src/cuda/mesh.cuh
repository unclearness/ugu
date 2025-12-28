#pragma once

namespace ugu {

struct MeshDevice {
  float3* d_vertices{nullptr};
  int* d_faces{nullptr};  // length = 3*num_faces
  float3* d_face_normals{nullptr};
  int num_faces_{0};
  int num_vertices_{0};

  //MeshDevice() = default;
  //MeshDevice(int num_vertices, int num_faces)
  //    : num_faces_(num_faces), num_vertices_(num_vertices) {
  //  cudaMalloc(&d_vertices, sizeof(float3) * num_vertices_);
  //  cudaMalloc(&d_faces, sizeof(int) * 3 * num_faces_);
  //}
  //~MeshDevice(){
  //  cudaFree(d_vertices);
  //  cudaFree(d_faces);
  //  cudaFree(d_face_normals);
  //}
};

void RemoveSmallConnectedComponents(const MeshDevice& in, int K, int min_faces,
                           MeshDevice& out, cudaStream_t stream = 0);
}  // namespace ugu
