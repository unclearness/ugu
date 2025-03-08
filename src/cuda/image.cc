#include "ugu/cuda/image.h"

#ifdef UGU_USE_CUDA
#include <cuda_runtime.h>

#include "./image.cuh"

#endif

namespace ugu {

#ifdef UGU_USE_CUDA
void BoxFilterCuda(Image3b& img, int kernel) {
  BoxFilterCuda3b(img.cols, img.rows, img.data, kernel);
}

void BilateralFilterCuda(const Image3b& src, Image3b& dst, int kernel) {}

bool ComputeNormalsCuda(const std::vector<Image1f>& depths,
                        const std::vector<PinholeCameraPtr>& cameras,
                        std::vector<Image3f>& normals, float max_connect_z_diff,
                        int step, bool gl_coord) {
  const size_t num_images = depths.size();
  const int width = depths[0].cols;
  const int height = depths[0].rows;

  std::vector<float> h_depths(num_images * width * height);

  // #pragma omp parallel for
  for (int i = 0; i < num_images; ++i) {
    std::memcpy(h_depths.data() + i * width * height, depths[i].data,
                sizeof(float) * width * height);
  }

  std::vector<float> h_fx(num_images);
  std::vector<float> h_fy(num_images);
  std::vector<float> h_cx(num_images);
  std::vector<float> h_cy(num_images);

  for (int i = 0; i < num_images; ++i) {
    h_fx[i] = cameras[i]->focal_length().x();
    h_fy[i] = cameras[i]->focal_length().y();
    h_cx[i] = cameras[i]->principal_point().x();
    h_cy[i] = cameras[i]->principal_point().y();
  }

  std::vector<float> h_normals(num_images * width * height * 3);

  ComputeNormalsCudaImpl(width, height, h_depths.data(), num_images,
                         h_fx.data(), h_fy.data(), h_cx.data(), h_cy.data(),
                         h_normals.data(), max_connect_z_diff, step, gl_coord);

  if (normals.size() != num_images) {
    normals.resize(num_images);
  }

  // #pragma omp parallel for
  for (int i = 0; i < num_images; ++i) {
    if (normals[i].cols != width || normals[i].rows != height) {
      normals[i] = ugu::Image1f::zeros(height, width);
    }
    std::memcpy(normals[i].data, h_normals.data() + i * width * height * 3,
                sizeof(float) * width * height * 3);
  }
  return true;
}

bool ComputeNormalsCuda(const int num_images, const int width, const int height,
                        std::vector<float>& h_depths,
                        std::vector<float>& h_normals,
                        const std::vector<float>& h_fx,
                        const std::vector<float>& h_fy,
                        const std::vector<float>& h_cx,
                        const std::vector<float>& h_cy,
                        float max_connect_z_diff, int step, bool gl_coord) {
  ComputeNormalsCudaImpl(width, height, h_depths.data(), num_images,
                         h_fx.data(), h_fy.data(), h_cx.data(), h_cy.data(),
                         h_normals.data(), max_connect_z_diff, step, gl_coord);

  return true;
}

#else
void BoxFilterCuda(Image3b& img, int kernel) {}

void BilateralFilterCuda(const Image3b& src, Image3b& dst, int kernel) {}

bool ComputeNormalsCuda(const std::vector<Image1f>& depths,
                        const std::vector<PinholeCameraPtr>& cameras,
                        std::vector<Image3f>& normals, float max_connect_z_diff,
                        int step, bool gl_coord) {
  return true;
}

bool ComputeNormalsCuda(const int num_images, const int width, const int height,
                        std::vector<float>& h_depths,
                        std::vector<float>& h_normals,
                        const std::vector<float>& h_fx,
                        const std::vector<float>& h_fy,
                        const std::vector<float>& h_cx,
                        const std::vector<float>& h_cy,
                        float max_connect_z_diff, int step, bool gl_coord) {
  return true;
}

NormalComputerCuda::NormalComputerCuda() {}
NormalComputerCuda(int width, int height, int num_images, const float* h_fx,
                   const float* h_fy, const float* h_cx, const float* h_cy,
                   float max_connect_z_diff, int step, bool gl_coord) {}
NormalComputerCuda::~NormalComputerCuda() {}

void NormalComputerCuda::ComputeNormals(float* h_depths, float* h_normals) {}

#endif

}  // namespace ugu
