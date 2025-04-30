#pragma once

#include "ugu/camera.h"
#include "ugu/image.h"

namespace ugu {

void BoxFilterCuda(Image3b& img, int kernel);

void BilateralFilterCuda(const Image3b& src, Image3b& dst, int kernel);

bool ComputeNormalsCuda(const std::vector<Image1f>& depths,
                        const std::vector<PinholeCameraPtr>& cameras,
                        std::vector<Image3f>& normals,
                        float max_connect_z_diff = 1e6f, int step = 1,
                        bool gl_coord = false);
bool ComputeNormalsCuda(
    const int num_images, const int width, const int height,
    std::vector<float>& h_depths, std::vector<float>& h_normals,
    const std::vector<float>& h_fx, const std::vector<float>& h_fy,
    const std::vector<float>& h_cx, const std::vector<float>& h_cy,
    float max_connect_z_diff = 1e6f, int step = 1, bool gl_coord = false);

class NormalComputerCuda {
 public:
  NormalComputerCuda();
  NormalComputerCuda(int width, int height, int num_images, const float* h_fx,
                     const float* h_fy, const float* h_cx, const float* h_cy,
                     float max_connect_z_diff = 1e6f, int step = 1,
                     bool gl_coord = false, const float* h_R = nullptr,
                     const float* h_t = nullptr);
  ~NormalComputerCuda();

  void Init(int width, int height, int num_images, const float* h_fx,
            const float* h_fy, const float* h_cx, const float* h_cy,
            float max_connect_z_diff = 1e6f, int step = 1,
            bool gl_coord = false, const float* h_R = nullptr,
            const float* h_t = nullptr);

  void ComputeNormals(const float* h_depths);

  void GetNormalsCpu(float* h_normals) const;
  void GetPointsCpu(float* h_points) const;

  const float* GetNormalsGpu() const;
  const float* GetPointsGpu() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ugu
