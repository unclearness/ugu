#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <cstdint>
#include <cub/cub.cuh>
#include <iostream>

#include "./mesh.cuh"
#include "ugu/timer.h"

namespace {

#define CUDA_CHECK(x)                                                 \
  do {                                                                \
    cudaError_t err = (x);                                            \
    if (err != cudaSuccess) {                                         \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << " @ " \
                << __FILE__ << ":" << __LINE__ << std::endl;          \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

static __host__ __device__ inline uint64_t pack_edge_u64(int a, int b) {
  uint32_t lo = (uint32_t)(a < b ? a : b);
  uint32_t hi = (uint32_t)(a < b ? b : a);
  return (uint64_t(hi) << 32) | uint64_t(lo);
}

__global__ void init_int_kernel(int* a, int n, int v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = v;
}
__global__ void init_u8_kernel(uint8_t* a, int n, uint8_t v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = v;
}

// ---------------------------
// (1) Build edge list (E=3F)
// ---------------------------
__global__ void build_edges_from_faces(const int* faces, int F,
                                       uint64_t* edge_key, int* edge_face) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= F) return;

  int i0 = faces[3 * f + 0];
  int i1 = faces[3 * f + 1];
  int i2 = faces[3 * f + 2];

  int base = 3 * f;
  edge_key[base + 0] = pack_edge_u64(i0, i1);
  edge_face[base + 0] = f;

  edge_key[base + 1] = pack_edge_u64(i1, i2);
  edge_face[base + 1] = f;

  edge_key[base + 2] = pack_edge_u64(i2, i0);
  edge_face[base + 2] = f;
}

// neighbors is int[3*F], -1 init.
// safely put a neighbor into one of 3 slots using atomicCAS
__device__ inline void try_set_neighbor3(int* nbr3, int v) {
  for (int k = 0; k < 3; ++k) {
    if (atomicCAS(&nbr3[k], -1, v) == -1) return;
  }
}

// Build adjacency from sorted edges:
// for each run(key) with len>=2, connect first two faces.
__global__ void build_adjacency_from_sorted_edges(const uint64_t* edge_key,
                                                  const int* edge_face, int E,
                                                  int* nbr, int F) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= E) return;

  bool is_head = (i == 0) || (edge_key[i] != edge_key[i - 1]);
  if (!is_head) return;

  uint64_t key = edge_key[i];
  int j = i + 1;
  while (j < E && edge_key[j] == key) ++j;

  if (j - i >= 2) {
    int f0 = edge_face[i];
    int f1 = edge_face[i + 1];
    if ((unsigned)f0 < (unsigned)F && (unsigned)f1 < (unsigned)F && f0 != f1) {
      try_set_neighbor3(&nbr[3 * f0], f1);
      try_set_neighbor3(&nbr[3 * f1], f0);
    }
  }
}

// ---------------------------
// (2) Label propagation (early-exit flag)
// ---------------------------
__global__ void label_propagation_one_iter(const int* nbr, int F,
                                           const int* label_in, int* label_out,
                                           int* d_changed) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= F) return;

  int old = label_in[f];
  int m = old;

  int n0 = nbr[3 * f + 0];
  int n1 = nbr[3 * f + 1];
  int n2 = nbr[3 * f + 2];

  if (n0 >= 0) m = min(m, label_in[n0]);
  if (n1 >= 0) m = min(m, label_in[n1]);
  if (n2 >= 0) m = min(m, label_in[n2]);

  // pointer jumping (OpenMP版の m = min(m, label[m]) と同等)
  m = min(m, label_in[m]);

  label_out[f] = m;
  if (m != old) atomicOr(d_changed, 1);
}

// ---------------------------
// (3) Compute cluster size per face from (label, faceId) sorted by label
//     Output: face_cc_size[faceId] = run_length(label)
// ---------------------------
__global__ void assign_runlen_to_faces(const int* sorted_label,
                                       const int* sorted_face, int F,
                                       int* face_cc_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= F) return;

  bool is_head = (i == 0) || (sorted_label[i] != sorted_label[i - 1]);
  if (!is_head) return;

  int lab = sorted_label[i];
  int j = i + 1;
  while (j < F && sorted_label[j] == lab) ++j;

  int runlen = j - i;
  for (int k = i; k < j; ++k) {
    int f = sorted_face[k];
    face_cc_size[f] = runlen;
  }
}

__global__ void make_run_head(const int* sorted_label, int F,
                              int* head)  // int[F], 0/1
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= F) return;
  int h = 0;
  if (i == 0)
    h = 1;
  else
    h = (sorted_label[i] != sorted_label[i - 1]) ? 1 : 0;
  head[i] = h;
}

// head[i]==1 の位置だけ run_start[run_id[i]] = i を書く
__global__ void write_run_start(const int* head, const int* run_id, int F,
                                int* run_start)  // int[num_runs]
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= F) return;
  if (head[i]) {
    int r = run_id[i];
    run_start[r] = i;
  }
}

__global__ void compute_run_len(const int* run_start, int num_runs, int F,
                                int* run_len) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= num_runs) return;
  int s = run_start[r];
  int e = (r + 1 < num_runs) ? run_start[r + 1] : F;
  run_len[r] = e - s;
}

__global__ void scatter_run_len_to_faces(const int* run_id, const int* run_len,
                                         const int* sorted_face, int F,
                                         int* face_cc_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= F) return;
  int r = run_id[i];
  int len = run_len[r];
  int f = sorted_face[i];
  face_cc_size[f] = len;
}

// ---------------------------
// (4) Face keep flag, face compaction, vertex mark, vertex compaction, remap
// ---------------------------
__global__ void make_face_keep(const int* face_cc_size, int F, int min_faces,
                               int* face_keep) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= F) return;
  face_keep[f] = (face_cc_size[f] >= min_faces) ? 1 : 0;
}

__global__ void compact_faces_kernel(
    const int* faces_in, int F, const uint8_t* face_keep,
    const int* face_scan_excl,  // exclusive scan of keep (int)
    int* faces_out) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= F) return;
  if (!face_keep[f]) return;

  int out_f = face_scan_excl[f];
  faces_out[3 * out_f + 0] = faces_in[3 * f + 0];
  faces_out[3 * out_f + 1] = faces_in[3 * f + 1];
  faces_out[3 * out_f + 2] = faces_in[3 * f + 2];
}

__global__ void compact_faces_and_normals_kernel(
    const int* faces_in, const float3* face_normals_in, int F,
    const int* face_keep,
    const int* face_scan_excl,  // exclusive scan of keep
    int* faces_out,
    float3* face_normals_out)  // nullable: if nullptr, normals are skipped
{
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= F) return;
  if (!face_keep[f]) return;

  int out_f = face_scan_excl[f];

  faces_out[3 * out_f + 0] = faces_in[3 * f + 0];
  faces_out[3 * out_f + 1] = faces_in[3 * f + 1];
  faces_out[3 * out_f + 2] = faces_in[3 * f + 2];

  if (face_normals_out && face_normals_in) {
    face_normals_out[out_f] = face_normals_in[f];
  }
}

__global__ void mark_used_vertices(const int* faces, int F2, uint8_t* v_used) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= F2) return;
  int i0 = faces[3 * f + 0];
  int i1 = faces[3 * f + 1];
  int i2 = faces[3 * f + 2];
  // atomicExchでOK（1を書くだけ）
  atomicExch((unsigned int*)&v_used[i0], 1u);
  atomicExch((unsigned int*)&v_used[i1], 1u);
  atomicExch((unsigned int*)&v_used[i2], 1u);
}

__global__ void mark_used_vertices_i32(const int* faces, int F, int V,
                                       int* v_used)  // int[V], 0/1
{
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= F) return;

  int i0 = faces[3 * f + 0];
  int i1 = faces[3 * f + 1];
  int i2 = faces[3 * f + 2];

  // 範囲外を潰す（デバッグにもなる）
  if ((unsigned)i0 < (unsigned)V) atomicExch(&v_used[i0], 1);
  if ((unsigned)i1 < (unsigned)V) atomicExch(&v_used[i1], 1);
  if ((unsigned)i2 < (unsigned)V) atomicExch(&v_used[i2], 1);
}

// v_used(uint8) -> v_scan_excl(int) を使って new_id を作る
__global__ void build_vertex_new_id(
    const int* v_used, const int* v_scan_excl, int V,
    int* v_new_id)  // -1 for unused, else new index
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= V) return;
  v_new_id[i] = v_used[i] ? v_scan_excl[i] : -1;
}

__global__ void gather_vertices(const float3* v_in, int V, const int* v_used,
                                const int* v_scan_excl, float3* v_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= V) return;
  if (!v_used[i]) return;
  int ni = v_scan_excl[i];
  v_out[ni] = v_in[i];
}

__global__ void remap_faces_vertices(int* faces, int F2, const int* v_new_id) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= F2) return;
  faces[3 * f + 0] = v_new_id[faces[3 * f + 0]];
  faces[3 * f + 1] = v_new_id[faces[3 * f + 1]];
  faces[3 * f + 2] = v_new_id[faces[3 * f + 2]];
}

static void sort_edges_by_key_cub(
    thrust::device_vector<uint64_t>& d_edge_key,
    thrust::device_vector<int>& d_edge_face,
    // 再利用用バッファ（呼び出し側で保持して毎回渡すと速い）
    thrust::device_vector<uint64_t>& d_edge_key_tmp,
    thrust::device_vector<int>& d_edge_face_tmp,
    thrust::device_vector<uint8_t>& d_temp_storage, cudaStream_t stream) {
  const int E = (int)d_edge_key.size();
  if ((int)d_edge_face.size() != E) std::exit(1);

  // tmp を確保（サイズが違うときだけリサイズ）
  if ((int)d_edge_key_tmp.size() != E) d_edge_key_tmp.resize(E);
  if ((int)d_edge_face_tmp.size() != E) d_edge_face_tmp.resize(E);

  uint64_t* keys_in = thrust::raw_pointer_cast(d_edge_key.data());
  uint64_t* keys_out = thrust::raw_pointer_cast(d_edge_key_tmp.data());
  int* vals_in = thrust::raw_pointer_cast(d_edge_face.data());
  int* vals_out = thrust::raw_pointer_cast(d_edge_face_tmp.data());

  void* temp_ptr = nullptr;
  size_t temp_bytes = 0;

  // 1) required temp size
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr, temp_bytes, keys_in, keys_out, vals_in, vals_out, E,
      /*begin_bit=*/0, /*end_bit=*/64, stream));

  // 2) allocate/reuse temp storage
  if (d_temp_storage.size() < temp_bytes) d_temp_storage.resize(temp_bytes);
  temp_ptr = thrust::raw_pointer_cast(d_temp_storage.data());

  // 3) sort
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      temp_ptr, temp_bytes, keys_in, keys_out, vals_in, vals_out, E,
      /*begin_bit=*/0, /*end_bit=*/64, stream));

  // 4) swap so that d_edge_key / d_edge_face become sorted
  d_edge_key.swap(d_edge_key_tmp);
  d_edge_face.swap(d_edge_face_tmp);
}

}  // namespace

namespace ugu {
void RemoveSmallConnectedComponents(const MeshDevice& in, int K, int min_faces,
                                    int early_exit_check_interval,
                                    MeshDevice& out,
                                    RemoveSmallConnectedComponentsBuf& buf,
                                    cudaStream_t stream) {
  const int F = in.num_faces_;
  const int V = in.num_vertices_;
  if (F == 0 || V == 0) {
    out = MeshDevice{};
    return;
  }

  buf.EnsureCapacity(F, V);
  CUDA_CHECK(cudaGetLastError());

  const int threads = 256;
  const int blocksF = (F + threads - 1) / threads;

  // ---- build neighbors ----
  const int E = 3 * F;
  thrust::device_vector<uint64_t>& d_edge_key = buf.d_edge_key;
  thrust::device_vector<int>& d_edge_face = buf.d_edge_face;

  build_edges_from_faces<<<blocksF, threads, 0, stream>>>(
      in.d_faces, F, thrust::raw_pointer_cast(d_edge_key.data()),
      thrust::raw_pointer_cast(d_edge_face.data()));
  CUDA_CHECK(cudaGetLastError());

#if 0
  thrust::sort_by_key(thrust::cuda::par.on(stream), d_edge_key.begin(),
                      d_edge_key.end(), d_edge_face.begin());
#else
  thrust::device_vector<uint64_t> d_edge_key_tmp;
  thrust::device_vector<int> d_edge_face_tmp;
  thrust::device_vector<uint8_t> d_sort_tmp_storage;

  sort_edges_by_key_cub(d_edge_key, d_edge_face, d_edge_key_tmp,
                        d_edge_face_tmp, d_sort_tmp_storage, stream);
#endif

  thrust::device_vector<int>& d_nbr = buf.d_nbr;
  init_int_kernel<<<(3 * F + threads - 1) / threads, threads, 0, stream>>>(
      thrust::raw_pointer_cast(d_nbr.data()), 3 * F, -1);
  CUDA_CHECK(cudaGetLastError());

  build_adjacency_from_sorted_edges<<<(E + threads - 1) / threads, threads, 0,
                                      stream>>>(
      thrust::raw_pointer_cast(d_edge_key.data()),
      thrust::raw_pointer_cast(d_edge_face.data()), E,
      thrust::raw_pointer_cast(d_nbr.data()), F);
  CUDA_CHECK(cudaGetLastError());

  // ---- label propagation with early exit ----
  thrust::device_vector<int>& d_label = buf.d_label;
  thrust::device_vector<int>& d_next = buf.d_next;
  thrust::sequence(thrust::cuda::par.on(stream), d_label.begin(),
                   d_label.end());

  int* d_changed = nullptr;
  CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));

  int end_iter = 0;
  for (int it = 0; it < K; ++it) {
    end_iter = it;
    CUDA_CHECK(cudaMemsetAsync(d_changed, 0, sizeof(int), stream));

    label_propagation_one_iter<<<blocksF, threads, 0, stream>>>(
        thrust::raw_pointer_cast(d_nbr.data()), F,
        thrust::raw_pointer_cast(d_label.data()),
        thrust::raw_pointer_cast(d_next.data()), d_changed);
    CUDA_CHECK(cudaGetLastError());

    d_label.swap(d_next);

    if (early_exit_check_interval > 0 && it > 0 &&
        it % early_exit_check_interval == 0) {
      int h_changed = 0;
      CUDA_CHECK(cudaMemcpyAsync(&h_changed, d_changed, sizeof(int),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));  // sync for early-exit
      if (!h_changed) {
        break;
      }
    }
  }
  CUDA_CHECK(cudaFree(d_changed));

  // ---- compute face_cc_size[f] from sorted (label, faceId) ----
  thrust::device_vector<int>& d_face_id = buf.d_face_id;
  thrust::sequence(thrust::cuda::par.on(stream), d_face_id.begin(),
                   d_face_id.end());

  // sort pairs by label
  thrust::sort_by_key(thrust::cuda::par.on(stream), d_label.begin(),
                      d_label.end(), d_face_id.begin());

  thrust::device_vector<int>& d_face_cc_size = buf.d_face_cc_size;
  init_int_kernel<<<blocksF, threads, 0, stream>>>(
      thrust::raw_pointer_cast(d_face_cc_size.data()), F, 0);
  CUDA_CHECK(cudaGetLastError());

#if 0
  assign_runlen_to_faces<<<blocksF, threads, 0, stream>>>(
      thrust::raw_pointer_cast(d_label.data()),
      thrust::raw_pointer_cast(d_face_id.data()), F,
      thrust::raw_pointer_cast(d_face_cc_size.data()));
    CUDA_CHECK(cudaGetLastError());
#else
  {
    thrust::device_vector<int> d_head(F);
    thrust::device_vector<int> d_run_id(F);

    int threads = 256;
    int blocksF = (F + threads - 1) / threads;

    make_run_head<<<blocksF, threads, 0, stream>>>(
        thrust::raw_pointer_cast(d_label.data()), F,
        thrust::raw_pointer_cast(d_head.data()));
    CUDA_CHECK(cudaGetLastError());

    // run_id = inclusive_scan(head) - 1
    thrust::inclusive_scan(thrust::cuda::par.on(stream), d_head.begin(),
                           d_head.end(), d_run_id.begin());

    thrust::transform(thrust::cuda::par.on(stream), d_run_id.begin(),
                      d_run_id.end(), d_run_id.begin(),
                      [] __host__ __device__(int x) { return x - 1; });

    // num_runs = run_id[last] + head[last]
    // * Synchronization occurs only when returning to the host here, so for
    // frame usage, the ideal design is "allocate up to F and hold num_runs on
    // the device".
    //  However, num_runs should be at most F, and buffers can be allocated up
    //  to F.
    int h_last_head = 0, h_last_run = 0;
    CUDA_CHECK(cudaMemcpyAsync(
        &h_last_head, thrust::raw_pointer_cast(d_head.data()) + (F - 1),
        sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        &h_last_run, thrust::raw_pointer_cast(d_run_id.data()) + (F - 1),
        sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int num_runs = h_last_run + h_last_head;

    // Use run_start/run_len at num_runs
    thrust::device_vector<int> d_run_start(num_runs);
    thrust::device_vector<int> d_run_len(num_runs);

    write_run_start<<<blocksF, threads, 0, stream>>>(
        thrust::raw_pointer_cast(d_head.data()),
        thrust::raw_pointer_cast(d_run_id.data()), F,
        thrust::raw_pointer_cast(d_run_start.data()));
    CUDA_CHECK(cudaGetLastError());

    compute_run_len<<<(num_runs + threads - 1) / threads, threads, 0, stream>>>(
        thrust::raw_pointer_cast(d_run_start.data()), num_runs, F,
        thrust::raw_pointer_cast(d_run_len.data()));
    CUDA_CHECK(cudaGetLastError());

    scatter_run_len_to_faces<<<blocksF, threads, 0, stream>>>(
        thrust::raw_pointer_cast(d_run_id.data()),
        thrust::raw_pointer_cast(d_run_len.data()),
        thrust::raw_pointer_cast(d_face_id.data()), F,
        thrust::raw_pointer_cast(d_face_cc_size.data()));
    CUDA_CHECK(cudaGetLastError());
  }

#endif

  //// ---- face_keep and face compaction ----
  thrust::device_vector<int>& d_face_keep_i = buf.d_face_keep_i;
  make_face_keep<<<blocksF, threads, 0, stream>>>(
      thrust::raw_pointer_cast(d_face_cc_size.data()), F, min_faces,
      thrust::raw_pointer_cast(d_face_keep_i.data()));
  CUDA_CHECK(cudaGetLastError());

  thrust::device_vector<int>& d_face_scan = buf.d_face_scan;
  thrust::exclusive_scan(thrust::cuda::par.on(stream), d_face_keep_i.begin(),
                         d_face_keep_i.end(), d_face_scan.begin(), 0);

  int kept_faces = 0;
  int last_keep = 0, last_scan = 0;
  {
    CUDA_CHECK(cudaMemcpyAsync(
        &last_keep, thrust::raw_pointer_cast(d_face_keep_i.data()) + (F - 1),
        sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        &last_scan, thrust::raw_pointer_cast(d_face_scan.data()) + (F - 1),
        sizeof(int), cudaMemcpyDeviceToHost, stream));
  }

  int* d_faces2 = buf.d_faces2;
  float3* d_face_normals2 = buf.d_face_normals2;

  compact_faces_and_normals_kernel<<<blocksF, threads, 0, stream>>>(
      in.d_faces, in.d_face_normals, F,
      thrust::raw_pointer_cast(d_face_keep_i.data()),
      thrust::raw_pointer_cast(d_face_scan.data()), d_faces2, d_face_normals2);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));
  kept_faces = last_scan + last_keep;

  const int blocksF2 = (kept_faces + threads - 1) / threads;

  thrust::device_vector<int>& d_v_used = buf.d_v_used;
  thrust::fill(thrust::cuda::par.on(stream), d_v_used.begin(), d_v_used.end(),
               0);

  mark_used_vertices_i32<<<blocksF2, threads, 0, stream>>>(
      d_faces2, kept_faces, V, thrust::raw_pointer_cast(d_v_used.data()));
  CUDA_CHECK(cudaGetLastError());

  thrust::device_vector<int>& d_v_scan = buf.d_v_scan;
  thrust::exclusive_scan(thrust::cuda::par.on(stream), d_v_used.begin(),
                         d_v_used.end(), d_v_scan.begin(), 0);
  CUDA_CHECK(cudaGetLastError());

  int kept_vertices = 0;
  int last_used = 0, last_vscan = 0;
  {
    CUDA_CHECK(cudaMemcpyAsync(
        &last_used, thrust::raw_pointer_cast(d_v_used.data()) + (V - 1),
        sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        &last_vscan, thrust::raw_pointer_cast(d_v_scan.data()) + (V - 1),
        sizeof(int), cudaMemcpyDeviceToHost, stream));
  }

  float3* d_vertices2 = buf.d_vertices2;

  gather_vertices<<<(V + threads - 1) / threads, threads, 0, stream>>>(
      in.d_vertices, V, thrust::raw_pointer_cast(d_v_used.data()),
      thrust::raw_pointer_cast(d_v_scan.data()), d_vertices2);
  CUDA_CHECK(cudaGetLastError());

  // build v_new_id and remap faces
  thrust::device_vector<int>& d_v_new_id = buf.d_v_new_id;
  build_vertex_new_id<<<(V + threads - 1) / threads, threads, 0, stream>>>(
      thrust::raw_pointer_cast(d_v_used.data()),
      thrust::raw_pointer_cast(d_v_scan.data()), V,
      thrust::raw_pointer_cast(d_v_new_id.data()));
  CUDA_CHECK(cudaGetLastError());

  remap_faces_vertices<<<blocksF2, threads, 0, stream>>>(
      d_faces2, kept_faces, thrust::raw_pointer_cast(d_v_new_id.data()));

  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));
  kept_vertices = last_vscan + last_used;

  // ---- output ----
  out.d_vertices = d_vertices2;
  out.d_faces = d_faces2;
  out.d_face_normals = d_face_normals2;
  out.num_vertices_ = kept_vertices;
  out.num_faces_ = kept_faces;
}

}  // namespace ugu
