#pragma once

#include <thrust/device_vector.h>

namespace ugu {
struct MeshDevice {
  float3* d_vertices{nullptr};
  int* d_faces{nullptr};  // length = 3*num_faces
  float3* d_face_normals{nullptr};
  int num_faces_{0};
  int num_vertices_{0};

  // MeshDevice() = default;
  // MeshDevice(int num_vertices, int num_faces)
  //     : num_faces_(num_faces), num_vertices_(num_vertices) {
  //   cudaMalloc(&d_vertices, sizeof(float3) * num_vertices_);
  //   cudaMalloc(&d_faces, sizeof(int) * 3 * num_faces_);
  // }
  //~MeshDevice(){
  //   cudaFree(d_vertices);
  //   cudaFree(d_faces);
  //   cudaFree(d_face_normals);
  // }
};

class RemoveSmallConnectedComponentsBuf {
 public:
  thrust::device_vector<uint64_t> d_edge_key;
  thrust::device_vector<int> d_edge_face;
  thrust::device_vector<int> d_nbr;
  thrust::device_vector<int> d_label;
  thrust::device_vector<int> d_next;
  thrust::device_vector<int> d_face_id;
  thrust::device_vector<int> d_face_cc_size;
  thrust::device_vector<uint8_t> d_face_keep;
  thrust::device_vector<int> d_face_keep_i;
  thrust::device_vector<int> d_face_scan;

  thrust::device_vector<int> d_v_used;
  thrust::device_vector<int> d_v_scan;
  thrust::device_vector<int> d_v_new_id;

  // Scratch reused across calls (previously reallocated per call)
  thrust::device_vector<uint64_t> d_edge_key_tmp;
  thrust::device_vector<int> d_edge_face_tmp;
  thrust::device_vector<uint8_t> d_sort_tmp_storage;  // grow-only
  thrust::device_vector<int> d_head;
  thrust::device_vector<int> d_run_id;
  // num_runs <= num_faces, so these are sized at num_faces
  thrust::device_vector<int> d_run_start;
  thrust::device_vector<int> d_run_len;
  int* d_changed = nullptr;

  int* d_faces2 = nullptr;
  float3* d_face_normals2 = nullptr;
  float3* d_vertices2 = nullptr;

  int num_faces_ = 0;
  int num_vertices_ = 0;

  RemoveSmallConnectedComponentsBuf() {};
  ~RemoveSmallConnectedComponentsBuf() { Free(); }

  void EnsureCapacity(int num_faces, int num_vertices) {
    if (num_faces_ < num_faces || num_vertices_ < num_vertices) {
      // Reallocate memory
      Free();
      Malloc(num_faces, num_vertices);
    } else {
      // Resize only
      Resize(num_faces, num_vertices);
    }
    num_faces_ = num_faces;
    num_vertices_ = num_vertices;
  }

  void Malloc(int num_faces, int num_vertices) {
    Free();

    Resize(num_faces, num_vertices);
    cudaMalloc(&d_faces2, sizeof(int) * 3 * num_faces);
    cudaMalloc(&d_face_normals2, sizeof(float3) * num_faces);
    cudaMalloc(&d_vertices2, sizeof(float3) * num_vertices);
    cudaMalloc(&d_changed, sizeof(int));
  }

  void Resize(int num_faces, int num_vertices) {
    const int E = 3 * num_faces;

    d_edge_key.resize(E);
    d_edge_face.resize(E);
    d_nbr.resize(E);
    d_label.resize(num_faces);
    d_next.resize(num_faces);
    d_face_id.resize(num_faces);
    d_face_cc_size.resize(num_faces);
    d_face_keep.resize(num_faces);
    d_face_keep_i.resize(num_faces);
    d_face_scan.resize(num_faces);

    d_v_used.resize(num_vertices);
    d_v_scan.resize(num_vertices);
    d_v_new_id.resize(num_vertices);

    d_edge_key_tmp.resize(E);
    d_edge_face_tmp.resize(E);
    d_head.resize(num_faces);
    d_run_id.resize(num_faces);
    d_run_start.resize(num_faces);
    d_run_len.resize(num_faces);
  }

  void Free() {
    num_faces_ = 0;
    num_vertices_ = 0;

    d_edge_key.clear();
    d_edge_face.clear();
    d_nbr.clear();
    d_label.clear();
    d_next.clear();
    d_face_id.clear();
    d_face_cc_size.clear();
    d_face_keep.clear();
    d_face_keep_i.clear();
    d_face_scan.clear();
    d_v_used.clear();
    d_v_scan.clear();
    d_v_new_id.clear();
    d_edge_key_tmp.clear();
    d_edge_face_tmp.clear();
    d_sort_tmp_storage.clear();
    d_head.clear();
    d_run_id.clear();
    d_run_start.clear();
    d_run_len.clear();
    if (nullptr != d_changed) {
      cudaFree(d_changed);
      d_changed = nullptr;
    }
    if (nullptr != d_faces2) {
      cudaFree(d_faces2);
      d_faces2 = nullptr;
    }
    if (nullptr != d_face_normals2) {
      cudaFree(d_face_normals2);
      d_face_normals2 = nullptr;
    }
    if (nullptr != d_vertices2) {
      cudaFree(d_vertices2);
      d_vertices2 = nullptr;
    }
  }
};

void RemoveSmallConnectedComponents(const MeshDevice& in, int K, int min_faces,
                                    int early_exit_check_interval,
                                    MeshDevice& out,
                                    RemoveSmallConnectedComponentsBuf& buf,
                                    cudaStream_t stream = 0);

void BuildFaceAdjacencyNbr3(const int* d_faces, int num_faces,
                            // workspace (device)
                            uint64_t* d_edge_key, int* d_edge_face,
                            // output
                            int* d_nbr3 /* int[3*num_faces] */);

void BuildFaceAdjacencyNbr3WithLocalEdges(
    const int* d_faces, int num_faces,
    // workspace (device)
    uint64_t* d_edge_key, int* d_edge_face_local,
    // output
    int* d_nbr3 /* int[3*num_faces] */,
    int* d_nbr_local_edge3 /* int[3*num_faces] */);
}  // namespace ugu
