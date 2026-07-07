#include "ugu/cuda/knn.h"

#ifdef UGU_USE_CUDA
#include <cuda_runtime.h>

#include "./helper_cuda.h"
#include "./knn.cuh"

#endif

namespace ugu {

#ifdef UGU_USE_CUDA

class KNnGridCuda::Impl {
 public:
  Impl();
  ~Impl();
  void SetData(const std::vector<Eigen::Vector3f>& data);
  void SetVoxelLen(float voxel_len) { m_voxel_len = voxel_len; }
  bool Build();
  std::vector<std::vector<KNnGridSearchResult>> SearchKnn(
      const std::vector<Eigen::Vector3f>& queries, uint32_t k) const;
  std::vector<KNnGridSearchResult> SearchKnn(const Eigen::Vector3f& query,
                                             uint32_t k) const;

 private:
  // Common
  uint32_t m_voxel_num;
  Eigen::Vector3f m_min_bound, m_max_bound;
  std::array<uint32_t, 3> m_grid_size;
  float m_voxel_len;

  // Host
  std::vector<Eigen::Vector3f> m_h_data;
  std::vector<uint32_t> m_h_voxel_start_indices;
  std::vector<uint32_t> m_h_voxel_point_indices;

  // Device
  float* m_d_data;
  uint32_t* m_d_voxel_start_indices;
  uint32_t* m_d_voxel_point_indices;
};

KNnGridCuda::Impl::Impl()
    : m_d_data(nullptr),
      m_d_voxel_start_indices(nullptr),
      m_d_voxel_point_indices(nullptr) {}

KNnGridCuda::Impl::~Impl() {
  if (m_d_data) {
    cudaFree(m_d_data);
  }
  if (m_d_voxel_start_indices) {
    cudaFree(m_d_voxel_start_indices);
  }
  if (m_d_voxel_point_indices) {
    cudaFree(m_d_voxel_point_indices);
  }
}

void KNnGridCuda::Impl::SetData(const std::vector<Eigen::Vector3f>& data) {
  m_h_data = data;
}

bool KNnGridCuda::Impl::Build() {
  if (m_h_data.empty()) {
    return false;
  }

  // Release buffers from a previous Build()
  if (m_d_data) {
    cudaFree(m_d_data);
    m_d_data = nullptr;
  }
  if (m_d_voxel_start_indices) {
    cudaFree(m_d_voxel_start_indices);
    m_d_voxel_start_indices = nullptr;
  }
  if (m_d_voxel_point_indices) {
    cudaFree(m_d_voxel_point_indices);
    m_d_voxel_point_indices = nullptr;
  }

  // Compute bounds
  m_min_bound = m_h_data[0];
  m_max_bound = m_h_data[0];
  for (const auto& p : m_h_data) {
    m_min_bound = m_min_bound.cwiseMin(p);
    m_max_bound = m_max_bound.cwiseMax(p);
  }

  // Compute grid size
  m_grid_size[0] = static_cast<uint32_t>(std::ceil(
                       (m_max_bound.x() - m_min_bound.x()) / m_voxel_len)) +
                   1;
  m_grid_size[1] = static_cast<uint32_t>(std::ceil(
                       (m_max_bound.y() - m_min_bound.y()) / m_voxel_len)) +
                   1;
  m_grid_size[2] = static_cast<uint32_t>(std::ceil(
                       (m_max_bound.z() - m_min_bound.z()) / m_voxel_len)) +
                   1;
  m_voxel_num = m_grid_size[0] * m_grid_size[1] * m_grid_size[2];

  // Counts the number of points in each voxel
  std::vector<uint32_t> voxel_counts(m_voxel_num, 0);

  // [point_index] -> voxel_index
  std::vector<uint32_t> point_voxel_indices(m_h_data.size(), 0);

  for (size_t i = 0; i < m_h_data.size(); ++i) {
    Eigen::Vector3f p = m_h_data[i];
    uint32_t vx =
        static_cast<uint32_t>((p.x() - m_min_bound.x()) / m_voxel_len);
    uint32_t vy =
        static_cast<uint32_t>((p.y() - m_min_bound.y()) / m_voxel_len);
    uint32_t vz =
        static_cast<uint32_t>((p.z() - m_min_bound.z()) / m_voxel_len);
    // Clamp to grid
    vx = std::min(std::max(vx, 0u), m_grid_size[0] - 1);
    vy = std::min(std::max(vy, 0u), m_grid_size[1] - 1);
    vz = std::min(std::max(vz, 0u), m_grid_size[2] - 1);
    uint32_t voxel_index =
        vx + vy * m_grid_size[0] + vz * m_grid_size[0] * m_grid_size[1];
    point_voxel_indices[i] = voxel_index;
    voxel_counts[voxel_index]++;
  }

  // Compute the start index of each voxel.
  // One extra sentinel entry holds the total count so that the kernel can
  // read offsets[voxel + 1] unconditionally.
  m_h_voxel_start_indices = std::vector<uint32_t>(m_voxel_num + 1, 0);

  uint32_t sum = 0;
  for (size_t i = 0; i < m_voxel_num; ++i) {
    m_h_voxel_start_indices[i] = sum;
    sum += voxel_counts[i];
  }
  m_h_voxel_start_indices[m_voxel_num] = sum;

  assert(sum == m_h_data.size());

  // Compute flatten voxel point indices
  m_h_voxel_point_indices = std::vector<uint32_t>(sum, -1);
  std::vector<uint32_t> voxel_current_indices = m_h_voxel_start_indices;

  for (size_t i = 0; i < m_h_data.size(); ++i) {
    uint32_t voxel_index = point_voxel_indices[i];
    uint32_t insert_pos = voxel_current_indices[voxel_index];
    voxel_current_indices[voxel_index]++;
    m_h_voxel_point_indices[insert_pos] = static_cast<uint32_t>(i);
  }

  // Send to device
  checkCudaErrors(
      cudaMalloc(&m_d_data, sizeof(Eigen::Vector3f) * m_h_data.size()));
  checkCudaErrors(cudaMemcpy(m_d_data, m_h_data.data(),
                             sizeof(Eigen::Vector3f) * m_h_data.size(),
                             cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&m_d_voxel_start_indices,
                             sizeof(uint32_t) * (m_voxel_num + 1)));
  checkCudaErrors(cudaMemcpy(m_d_voxel_start_indices,
                             m_h_voxel_start_indices.data(),
                             sizeof(uint32_t) * (m_voxel_num + 1),
                             cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&m_d_voxel_point_indices, sizeof(uint32_t) * sum));
  checkCudaErrors(cudaMemcpy(m_d_voxel_point_indices,
                             m_h_voxel_point_indices.data(),
                             sizeof(uint32_t) * sum, cudaMemcpyHostToDevice));
  return true;
}

std::vector<std::vector<KNnGridSearchResult>> KNnGridCuda::Impl::SearchKnn(
    const std::vector<Eigen::Vector3f>& queries, uint32_t k) const {
  std::vector<std::vector<KNnGridSearchResult>> results;
  if (!m_d_data || !m_d_voxel_point_indices || !m_d_voxel_start_indices) {
    return results;
  }

  // Clip k
  k = std::min(k, static_cast<uint32_t>(m_h_data.size()));

  uint32_t queries_num = static_cast<uint32_t>(queries.size());

  // Send queries to the device
  float* d_queries;
  checkCudaErrors(
      cudaMalloc(&d_queries, sizeof(Eigen::Vector3f) * queries_num));
  checkCudaErrors(cudaMemcpy(d_queries, queries.data(),
                             sizeof(Eigen::Vector3f) * queries_num,
                             cudaMemcpyHostToDevice));

  // Memory for device results
  uint32_t* d_knn_indices;
  checkCudaErrors(
      cudaMalloc(&d_knn_indices, sizeof(uint32_t) * queries_num * k));
  checkCudaErrors(
      cudaMemset(d_knn_indices, -1, sizeof(uint32_t) * queries_num * k));

  float* d_knn_dists;
  checkCudaErrors(cudaMalloc(&d_knn_dists, sizeof(float) * queries_num * k));
  checkCudaErrors(
      cudaMemset(d_knn_dists, -1, sizeof(float) * queries_num * k));

  // Execute kernel
  SearchKnnCuda(m_d_data, m_d_voxel_start_indices, m_d_voxel_point_indices,
                m_grid_size[0], m_grid_size[1], m_grid_size[2], m_voxel_len,
                m_min_bound, d_queries, queries_num, k, d_knn_indices,
                d_knn_dists);

  // Check errors
  checkCudaErrors(cudaGetLastError());

  // Copy to host
  std::vector<uint32_t> h_knn_indices(queries_num * k, -1);
  checkCudaErrors(cudaMemcpy(h_knn_indices.data(), d_knn_indices,
                             sizeof(uint32_t) * queries_num * k,
                             cudaMemcpyDeviceToHost));
  std::vector<float> h_knn_dists(queries_num * k, -1);
  checkCudaErrors(cudaMemcpy(h_knn_dists.data(), d_knn_dists,
                             sizeof(float) * queries_num * k,
                             cudaMemcpyDeviceToHost));

  results.resize(queries_num);
  for (int i = 0; i < queries_num; ++i) {
    results[i].resize(k);
    for (int j = 0; j < k; ++j) {
      results[i][j].index = h_knn_indices[i * k + j];
      results[i][j].dist = h_knn_dists[i * k + j];
    }
  }

  cudaFree(d_queries);
  cudaFree(d_knn_indices);
  cudaFree(d_knn_dists);

  return results;
}

std::vector<KNnGridSearchResult> KNnGridCuda::Impl::SearchKnn(
    const Eigen::Vector3f& query, uint32_t k) const {
  auto results = SearchKnn(std::vector<Eigen::Vector3f>{query}, k);
  return results[0];
}

KNnGridCuda::KNnGridCuda() : pimpl_(std::unique_ptr<Impl>(new Impl)) {}

KNnGridCuda::~KNnGridCuda() {}

void KNnGridCuda::SetData(const std::vector<Eigen::Vector3f>& data) {
  pimpl_->SetData(data);
}

void KNnGridCuda::SetVoxelLen(float voxel_len) {
  pimpl_->SetVoxelLen(voxel_len);
}

void KNnGridCuda::Build() { pimpl_->Build(); }

std::vector<std::vector<KNnGridSearchResult>> KNnGridCuda::SearchKnn(
    const std::vector<Eigen::Vector3f>& queries, uint32_t k) const {
  return pimpl_->SearchKnn(queries, k);
}

std::vector<KNnGridSearchResult> KNnGridCuda::SearchKnn(
    const Eigen::Vector3f& query, uint32_t k) const {
  return pimpl_->SearchKnn(query, k);
}

#else
class KNnGridCuda::Impl {
 public:
  Impl();
  ~Impl();
};

KNnGridCuda::Impl::Impl() {}
KNnGridCuda::Impl::~Impl() {}

KNnGridCuda::KNnGridCuda() : pimpl_(std::unique_ptr<Impl>(new Impl)) {}
KNnGridCuda::~KNnGridCuda() {}

void KNnGridCuda::SetData(const std::vector<Eigen::Vector3f>& data) {
  (void)data;
}

void KNnGridCuda::Build() {}

std::vector<std::vector<KNnGridSearchResult>> SearchKnn(
    const std::vector<Eigen::Vector3f>& queries, uint32_t k) {
  (void)queries;
  (void)k;
  return std::vector<std::vector<KNnGridSearchResult>>();
}

std::vector<KNnGridSearchResult> SearchKnn(const Eigen::Vector3f& query,
                                           uint32_t k) {
  (void)query;
  (void)k;
  return std::vector<KNnGridSearchResult>();
}

#endif

}  // namespace ugu
