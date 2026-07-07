#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "./helper_cuda.h"
#include "./image.cuh"
#include "ugu/cuda/image.h"

namespace {

#define MAX_IMAGES 32

// ・ｽ關費ｿｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽﾌ茨ｿｽ
__constant__ int d_width;
__constant__ int d_height;
__constant__ int d_num_images;
__constant__ float d_max_connect_z_diff;
__constant__ int d_step;
__constant__ bool d_gl_coord;
__constant__ bool d_central_difference;

// ・ｽJ・ｽ・ｽ・ｽ・ｽ・ｽp・ｽ・ｽ・ｽ・ｽ・ｽ[・ｽ^・ｽi・ｽ鞫懶ｿｽ・ｽ・ｽﾆに異なるが・ｽ・ｽ・ｽ・ｽ・ｽﾍ擾ｿｽ・ｽﾈゑｿｽ・ｽﾆ会ｿｽ・ｽ・ｽj
__constant__ float d_fx[MAX_IMAGES];
__constant__ float d_fy[MAX_IMAGES];
__constant__ float d_cx[MAX_IMAGES];
__constant__ float d_cy[MAX_IMAGES];

// camera to world tranformation
__constant__ float d_R[MAX_IMAGES * 9];
__constant__ float d_t[MAX_IMAGES * 3];

#define BLOCK_W 16
#define BLOCK_H 16

#if 0
__global__ void BoxFilterNaive(const uint8_t* d_in, uint8_t* d_out, int width,
                               int height, int K) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  int R = (K - 1) / 2;
  int outIdxBase = 3 * (y * width + x);

  for (int c = 0; c < 3; c++) {
    float sumVal = 0.0f;
    int count = 0;

    for (int dy = -R; dy <= R; dy++) {
      for (int dx = -R; dx <= R; dx++) {
        int nx = x + dx;
        int ny = y + dy;
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
          int inIdx = 3 * (ny * width + nx) + c;
          sumVal += d_in[inIdx];
          count++;
        }
      }
    }
    d_out[outIdxBase + c] = (uint8_t)(sumVal / (float)count);
  }
}
#endif

#if 0
__global__ void BoxFilterShared(const uint8_t* d_in, uint8_t* d_out, int width,
                                int height, int K) {
  extern __shared__ float tile[];
  // tile uses 3 channels

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * blockDim.x + tx;
  int y = blockIdx.y * blockDim.y + ty;

  int R = (K - 1) / 2;
  int blockSize = blockDim.x * blockDim.y;

  // Write into shared memory if (x, y) is inside
  for (int c = 0; c < 3; c++) {
    int idxGlobal = 3 * (y * width + x) + c;
    int idxShared = c * blockSize + (ty * blockDim.x + tx);
    if (x < width && y < height) {
      tile[idxShared] = d_in[idxGlobal];
    } else {
      tile[idxShared] = 0.0f;  // 0 fill for outside
    }
  }

  __syncthreads();  // Wait shared memory writing

  if (x >= width || y >= height) {
    return;
  }

  // box filter ・ｽv・ｽZ
  float outVal[3] = {0.0f, 0.0f, 0.0f};
  int count = 0;
  // Loop window
  for (int dy = -R; dy <= R; dy++) {
    for (int dx = -R; dx <= R; dx++) {
      int xx = tx + dx;  // Index in shared memory
      int yy = ty + dy;
      // Refer to shared memory if inside
      if (xx >= 0 && xx < blockDim.x && yy >= 0 && yy < blockDim.y) {
        int idxSharedBase = (yy * blockDim.x + xx);
        for (int c = 0; c < 3; c++) {
          outVal[c] += tile[c * blockSize + idxSharedBase];
        }
        count++;
      } else {
        // Load from global memory if out of block boundary
        int gx = x + dx;
        int gy = y + dy;
        if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
          int idxGlobal = 3 * (gy * width + gx);
          for (int c = 0; c < 3; c++) {
            outVal[c] += d_in[idxGlobal + c];
          }
          count++;
        }
      }
    }
  }

  int outIdxBase = 3 * (y * width + x);
  for (int c = 0; c < 3; c++) {
    d_out[outIdxBase + c] = outVal[c] / (float)count;
  }
}
#endif

__global__ void BoxFilterRow(const uint8_t* d_in, uint8_t* d_out, int width,
                             int height, int K) {
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= width || y >= height) return;

  int R = (K - 1) / 2;
  int outIdxBase = 3 * (y * width + x);

  for (int c = 0; c < 3; c++) {
    float sumVal = 0.0f;
    int count = 0;
    for (int i = x - R; i <= x + R; i++) {
      if (i >= 0 && i < width) {
        int inIdx = 3 * (y * width + i) + c;
        sumVal += d_in[inIdx];
        count++;
      }
    }
    d_out[outIdxBase + c] = sumVal / (float)count;
  }
}

__global__ void BoxFilterCol(const uint8_t* d_in, uint8_t* d_out, int width,
                             int height, int K) {
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= width || y >= height) return;

  int R = (K - 1) / 2;
  int outIdxBase = 3 * (y * width + x);

  for (int c = 0; c < 3; c++) {
    float sumVal = 0.0f;
    int count = 0;
    for (int j = y - R; j <= y + R; j++) {
      if (j >= 0 && j < height) {
        int inIdx = 3 * (j * width + x) + c;
        sumVal += d_in[inIdx];
        count++;
      }
    }
    d_out[outIdxBase + c] = sumVal / (float)count;
  }
}

#if 0
__global__ void Transpose(const uint8_t* d_in, uint8_t* d_out, int width,
                          int height) {
  //__shared__ float tile[16][16 * 3];
  extern __shared__ float tile[];

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Write to shared memory
  if (x < width && y < height) {
    for (int c = 0; c < 3; c++) {
      tile[(threadIdx.y * blockDim.x + threadIdx.x) * 3 + c] =
          d_in[3 * (y * width + x) + c];
    }
  }

  __syncthreads();

  // Write to global memory from shared memory
  x = blockIdx.y * blockDim.x + threadIdx.x;  // transposed x
  y = blockIdx.x * blockDim.y + threadIdx.y;  // transposed y

  if (x < height && y < width) {
    for (int c = 0; c < 3; c++) {
      d_out[3 * (y * height + x) + c] =
          tile[(threadIdx.x * blockDim.y + threadIdx.y) * 3 + c];
    }
  }
}
#endif

#if 1
__global__ void ComputeNormalsTextureMultiCam(cudaTextureObject_t texDepth,
                                              float* normals) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int n = blockIdx.z * blockDim.z + threadIdx.z;  // ・ｽ鞫懶ｿｽC・ｽ・ｽ・ｽf・ｽb・ｽN・ｽX

  // ・ｽ鞫懶ｿｽ・ｽ・ｽ・ｽ・ｽﾆ具ｿｽ・ｽE・ｽ`・ｽF・ｽb・ｽN・ｽi・ｽ・ｽ・ｽE・ｽﾍ単・ｽ・ｽ・ｽﾉ擾ｿｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ[・ｽ・ｽ・ｽo・ｽﾍ）
  if (n >= d_num_images || x <= d_step - 1 || y <= d_step - 1 ||
      x >= d_width - d_step || y >= d_height - d_step) {
    return;
  }

  // ・ｽS・ｽﾌゑｿｽ1・ｽ・ｽ・ｽ・ｽ・ｽC・ｽ・ｽ・ｽf・ｽb・ｽN・ｽX・ｽv・ｽZ・ｽi・ｽ鞫・n ・ｽ・ｽ (y, x)・ｽj
  int idx = n * d_width * d_height + y * d_width + x;

  // ・ｽe・ｽ鞫懶ｿｽﾅ有・ｽﾌカ・ｽ・ｽ・ｽ・ｽ・ｽp・ｽ・ｽ・ｽ・ｽ・ｽ[・ｽ^・ｽ・ｽ・ｽ謫ｾ
  float fx_val = d_fx[n];
  float fy_val = d_fy[n];
  float cx_val = d_cx[n];
  float cy_val = d_cy[n];

  float inv_fx = 1.0f / fx_val;
  float inv_fy = 1.0f / fy_val;

  // Layered ・ｽe・ｽN・ｽX・ｽ`・ｽ・ｽ・ｽ・ｽ・ｽ迪ｻ・ｽﾝの会ｿｽf・ｽﾌ深・ｽx・ｽl・ｽ・ｽ・ｽ謫ｾ
  float d = tex2DLayered<float>(texDepth, x, y, n);
  if (d <= 0.0f) {
    normals[3 * idx + 0] = 0.0f;
    normals[3 * idx + 1] = 0.0f;
    normals[3 * idx + 2] = 0.0f;
    return;
  }

  // ・ｽﾗ接会ｿｽf・ｽi・ｽE・ｽ・ｽ・ｽ・ｽﾑ会ｿｽ・ｽj・ｽﾌ深・ｽx・ｽ・ｽ・ｽ謫ｾ
  float d_right = tex2DLayered<float>(texDepth, x + d_step, y, n);
  float d_bottom = tex2DLayered<float>(texDepth, x, y + d_step, n);
  if (d_right <= 0.0f || d_bottom <= 0.0f) {
    normals[3 * idx + 0] = 0.0f;
    normals[3 * idx + 1] = 0.0f;
    normals[3 * idx + 2] = 0.0f;
    return;
  }

  // ・ｽ・ｽf・ｽ・ｽ・ｽW (u, v)
  float u = (float)x;
  float v = (float)y;

  // ・ｽ・ｽ・ｽﾝの会ｿｽf・ｽ・ｽ3・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽW・ｽv・ｽZ
  float X = (u - cx_val) * d * inv_fx;
  float Y = (v - cy_val) * d * inv_fy;
  float Z = d;
  float dx_x;
  float dx_y;
  float dx_z;

  float dy_x;
  float dy_y;
  float dy_z;

  // ・ｽE・ｽﾗの会ｿｽf・ｽ・ｽ3・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽW・ｽv・ｽZ
  float Xr = ((u + d_step) - cx_val) * d_right * inv_fx;
  float Yr = (v - cy_val) * d_right * inv_fy;
  float Zr = d_right;

  // ・ｽ・ｽ・ｽﾗの会ｿｽf・ｽ・ｽ3・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽW・ｽv・ｽZ
  float Xb = (u - cx_val) * d_bottom * inv_fx;
  float Yb = ((v + d_step) - cy_val) * d_bottom * inv_fy;
  float Zb = d_bottom;

#if 1
  if (d_central_difference) {
    float d_left = tex2DLayered<float>(texDepth, x - d_step, y, n);
    float d_top = tex2DLayered<float>(texDepth, x, y - d_step, n);
    if (d_left <= 0.0f || d_top <= 0.0f) {
      normals[3 * idx + 0] = 0.0f;
      normals[3 * idx + 1] = 0.0f;
      normals[3 * idx + 2] = 0.0f;
      return;
    }

    // ・ｽE・ｽﾗの会ｿｽf・ｽ・ｽ3・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽW・ｽv・ｽZ
    float Xl = ((u - d_step) - cx_val) * d_left * inv_fx;
    float Yl = (v - cy_val) * d_left * inv_fy;
    float Zl = d_left;

    // ・ｽ・ｽ・ｽﾗの会ｿｽf・ｽ・ｽ3・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽW・ｽv・ｽZ
    float Xt = (u - cx_val) * d_top * inv_fx;
    float Yt = ((v - d_step) - cy_val) * d_top * inv_fy;
    float Zt = d_top;

    if (fabsf(Z - Zl) > d_max_connect_z_diff ||
        fabs(Z - Zt) > d_max_connect_z_diff) {
      normals[3 * idx + 0] = 0.0f;
      normals[3 * idx + 1] = 0.0f;
      normals[3 * idx + 2] = 0.0f;
      return;
    }

    dx_x = Xr - Xl;
    dx_y = Yr - Yl;
    dx_z = Zr - Zl;

    dy_x = Xb - Xt;
    dy_y = Yb - Yt;
    dy_z = Zb - Zt;

  } else {
    if (fabsf(Z - Zr) > d_max_connect_z_diff ||
        fabs(Z - Zb) > d_max_connect_z_diff) {
      normals[3 * idx + 0] = 0.0f;
      normals[3 * idx + 1] = 0.0f;
      normals[3 * idx + 2] = 0.0f;
      return;
    }

    dx_x = Xr - X;
    dx_y = Yr - Y;
    dx_z = Zr - Z;

    dy_x = Xb - X;
    dy_y = Yb - Y;
    dy_z = Zb - Z;
  }
#else
  if (fabsf(Z - Zr) > d_max_connect_z_diff ||
      fabs(Z - Zb) > d_max_connect_z_diff) {
    normals[3 * idx + 0] = 0.0f;
    normals[3 * idx + 1] = 0.0f;
    normals[3 * idx + 2] = 0.0f;
    return;
  }

  dx_x = Xr - X;
  dx_y = Yr - Y;
  dx_z = Zr - Z;

  dy_x = Xb - X;
  dy_y = Yb - Y;
  dy_z = Zb - Z;
#endif

  // ・ｽN・ｽ・ｽ・ｽX・ｽv・ｽ・ｽ・ｽ_・ｽN・ｽg・ｽﾅ法・ｽ・ｽ・ｽ・ｽ・ｽv・ｽZ
  float nx = dx_y * dy_z - dx_z * dy_y;
  float ny = dx_z * dy_x - dx_x * dy_z;
  float nz = dx_x * dy_y - dx_y * dy_x;

  // ・ｽ・ｽ・ｽK・ｽ・ｽ
  float norm = sqrtf(nx * nx + ny * ny + nz * nz);
  if (norm > 1e-6f) {
    nx /= norm;
    ny /= norm;
    nz /= norm;
  } else {
    nx = ny = nz = 0.0f;
  }

  if (d_gl_coord) {
    ny = -ny;
    nz = -nz;
  }

  normals[3 * idx + 0] = nx;
  normals[3 * idx + 1] = ny;
  normals[3 * idx + 2] = nz;
}
#endif

__global__ void ComputeNormalsTextureMultiCam_Shared(
    cudaTextureObject_t texDepth, float* normals, float* points) {
  // ・ｽe・ｽu・ｽ・ｽ・ｽb・ｽN・ｽ・ｽ 1 ・ｽ・ｽ・ｽﾌ画像・ｽ・ｽS・ｽ・ｽ
  int n = blockIdx.z;
  if (n >= d_num_images) return;

  // 2D ・ｽ^・ｽC・ｽ・ｽ・ｽ・ｽ・ｽﾌピ・ｽN・ｽZ・ｽ・ｽ
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * BLOCK_W + tx;
  int y = blockIdx.y * BLOCK_H + ty;
  if (x >= d_width || y >= d_height) return;

  // ・ｽ・ｽ・ｽL・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽF・ｽ^・ｽC・ｽ・ｽ・ｽ{・ｽ・ｽ・ｽE・ｽ・ｽ
  extern __shared__ float s_depth[];
  const int S_W = BLOCK_W + 2 * d_step;
  // const int S_H = BLOCK_H + 2 * d_step;

  // ・ｽ・ｽ・ｽL・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽﾌ搾ｿｽ・ｽW
  int sx = tx + d_step;
  int sy = ty + d_step;
  int sidx = sy * S_W + sx;

  // (1) ・ｽ・ｽ・ｽS・ｽ・ｽf・ｽﾌ深・ｽx・ｽ・ｽ・ｽ・ｽ・ｽ[・ｽh
  float d_center = tex2DLayered<float>(texDepth, x, y, n);
  s_depth[sidx] = d_center;

  // (2) ・ｽ・ｽ・ｽE・ｽs・ｽN・ｽZ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ[・ｽh
  //    ・ｽe・ｽX・ｽ・ｽ・ｽb・ｽh・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽﾌ趣ｿｽ・ｽ・ｽ step ・ｽ・ｽ・ｽﾌ具ｿｽ・ｽE・ｽ・ｽS・ｽ・ｽ・ｽ・ｽ・ｽﾜゑｿｽ
  if (tx < d_step) {
    // ・ｽ・ｽ・ｽ・ｽ・ｽE
    s_depth[sy * S_W + (sx - d_step)] =
        tex2DLayered<float>(texDepth, x - d_step, y, n);
  }
  if (tx >= BLOCK_W - d_step) {
    // ・ｽE・ｽ・ｽ・ｽE
    s_depth[sy * S_W + (sx + d_step)] =
        tex2DLayered<float>(texDepth, x + d_step, y, n);
  }
  if (ty < d_step) {
    // ・ｽ繼ｫ・ｽE
    s_depth[(sy - d_step) * S_W + sx] =
        tex2DLayered<float>(texDepth, x, y - d_step, n);
  }
  if (ty >= BLOCK_H - d_step) {
    // ・ｽ・ｽ・ｽ・ｽ・ｽE
    s_depth[(sy + d_step) * S_W + sx] =
        tex2DLayered<float>(texDepth, x, y + d_step, n);
  }

  // ・ｽp・ｽ・ｽ・ｽK・ｽv・ｽﾈら同・ｽl・ｽ・ｽ...
  __syncthreads();

  // ・ｽﾈ降・ｽ・ｽ shared ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽﾇみ出・ｽ・ｽ
  if (d_center <= 0.0f) {
    // ・ｽ・ｽ・ｽ・ｽ・ｽ[・ｽx
    int idx = n * d_width * d_height + y * d_width + x;
    normals[3 * idx + 0] = normals[3 * idx + 1] = normals[3 * idx + 2] = 0.0f;
    points[3 * idx + 0] = points[3 * idx + 1] = points[3 * idx + 2] = 0.0f;
    return;
  }

  // ・ｽ・ｽ・ｽL・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽE・ｽE・ｽ・ｽ・ｽ・ｽﾇみ出・ｽ・ｽ
  float d_r = s_depth[sidx + d_step];
  float d_b = s_depth[(sidx + S_W * d_step)];

  // 3D ・ｽ_・ｽﾌ計・ｽZ・ｽi・ｽ・ｽ・ｽS・ｽj
  float u = float(x), v = float(y);
  float fx_val = d_fx[n], fy_val = d_fy[n], cx_val = d_cx[n], cy_val = d_cy[n];
  float inv_fx = 1.0f / fx_val, inv_fy = 1.0f / fy_val;
  float X = (u - cx_val) * d_center * inv_fx;
  float Y = (v - cy_val) * d_center * inv_fy;
  float Z = d_center;
  int idx = n * d_width * d_height + y * d_width + x;

  if (d_gl_coord) {
    Y = -Y;
    Z = -Z;
  }

  // (3) ・ｽ關費ｿｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ Extrinsics ・ｽ・ｽﾇみ出・ｽ・ｽ
  const float* R = &d_R[n * 9];  // R[0]..R[8] ・ｽ・ｽ 3・ｽ~3 ・ｽs・ｽ・ｽ
  const float* t = &d_t[n * 3];  // t[0]..t[2] ・ｽ・ｽ・ｽ・ｽ・ｽi・ｽx・ｽN・ｽg・ｽ・ｽ

  // (4) ・ｽ・ｽ・ｽ[・ｽ・ｽ・ｽh・ｽ・ｽ・ｽW・ｽﾏ奇ｿｽ・ｽF・ｽ_ (X,Y,Z) ・ｽ・ｽ (Xw,Yw,Zw)
  float Xw = R[0] * X + R[1] * Y + R[2] * Z + t[0];
  float Yw = R[3] * X + R[4] * Y + R[5] * Z + t[1];
  float Zw = R[6] * X + R[7] * Y + R[8] * Z + t[2];

  // (6) ・ｽo・ｽﾍバ・ｽb・ｽt・ｽ@・ｽﾖ擾ｿｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ
  points[3 * idx + 0] = Xw;
  points[3 * idx + 1] = Yw;
  points[3 * idx + 2] = Zw;

  if (d_r <= 0.0f || d_b <= 0.0f) {
    normals[3 * idx + 0] = normals[3 * idx + 1] = normals[3 * idx + 2] = 0.0f;
    return;
  }

  // ・ｽﾗ接ピ・ｽN・ｽZ・ｽ・ｽ・ｽﾌ３・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽW
  float Xr = ((u + d_step) - cx_val) * d_r * inv_fx;
  float Yr = (v - cy_val) * d_r * inv_fy;
  float Zr = d_r;
  float Xb = (u - cx_val) * d_b * inv_fx;
  float Yb = ((v + d_step) - cy_val) * d_b * inv_fy;
  float Zb = d_b;

#if 0
  // ・ｽ@・ｽ・ｽ・ｽv・ｽZ・ｽi・ｽO・ｽi・ｽ・ｽ・ｽ・ｽ・ｽj
  float dx_x = Xr - X, dx_y = Yr - Y, dx_z = Zr - Z;
  float dy_x = Xb - X, dy_y = Yb - Y, dy_z = Zb - Z;
  //float nx = dx_y * dy_z - dx_z * dy_y;
  //float ny = dx_z * dy_x - dx_x * dy_z;
  //float nz = dx_x * dy_y - dx_y * dy_x;

#else
  float d_l = s_depth[sidx - d_step];
  float d_t = s_depth[(sidx - S_W * d_step)];

  if (d_l <= 0.0f || d_t <= 0.0f) {
    normals[3 * idx + 0] = normals[3 * idx + 1] = normals[3 * idx + 2] = 0.0f;
    return;
  }

  float Xl = ((u - d_step) - cx_val) * d_l * inv_fx;
  float Yl = (v - cy_val) * d_l * inv_fy;
  float Zl = d_l;
  float Xt = (u - cx_val) * d_t * inv_fx;
  float Yt = ((v - d_step) - cy_val) * d_t * inv_fy;
  float Zt = d_t;

  float dx_x = Xr - Xl, dx_y = Yr - Yl, dx_z = Zr - Zl;
  float dy_x = Xb - Xt, dy_y = Yb - Yt, dy_z = Zb - Zt;
#endif

  float nx = dx_z * dy_y - dx_y * dy_z;
  float ny = dx_x * dy_z - dx_z * dy_x;
  float nz = dx_y * dy_x - dx_x * dy_y;

  float norm = sqrtf(nx * nx + ny * ny + nz * nz);
  if (norm > 1e-6f) {
    nx /= norm;
    ny /= norm;
    nz /= norm;
  } else {
    nx = ny = nz = 0.0f;
  }

  if (d_gl_coord) {
    ny = -ny;
    nz = -nz;
  }

  // (5) ・ｽ@・ｽ・ｽ・ｽx・ｽN・ｽg・ｽ・ｽ・ｽﾍ包ｿｽ・ｽi・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽ・ｽﾈゑｿｽ・ｽﾌで会ｿｽ]・ｽﾌゑｿｽ
  float nxw = R[0] * nx + R[1] * ny + R[2] * nz;
  float nyw = R[3] * nx + R[4] * ny + R[5] * nz;
  float nzw = R[6] * nx + R[7] * ny + R[8] * nz;

  normals[3 * idx + 0] = nxw;
  normals[3 * idx + 1] = nyw;
  normals[3 * idx + 2] = nzw;
}

}  // namespace

namespace ugu {

namespace {

// Grow-only device workspace reused across calls so that per-call
// cudaMalloc/cudaFree does not dominate the filter cost.
// Buffers are intentionally not freed at process exit: the CUDA context may
// already be destroyed when static destructors run.
struct BoxFilterWorkspace {
  uint8_t* d_in = nullptr;
  uint8_t* d_temp = nullptr;
  uint8_t* d_out = nullptr;
  size_t capacity = 0;

  void Ensure(size_t size) {
    if (size <= capacity) {
      return;
    }
    if (d_in != nullptr) {
      checkCudaErrors(cudaFree(d_in));
      checkCudaErrors(cudaFree(d_temp));
      checkCudaErrors(cudaFree(d_out));
    }
    checkCudaErrors(cudaMalloc(&d_in, size));
    checkCudaErrors(cudaMalloc(&d_temp, size));
    checkCudaErrors(cudaMalloc(&d_out, size));
    capacity = size;
  }
};

}  // namespace

void BoxFilterCuda3b(int width, int height, void* data, int k) {
  // Not thread-safe, matching the rest of this API (default stream, shared
  // constant memory).
  static BoxFilterWorkspace ws;

  size_t totalSize = sizeof(uint8_t) * 3 * width * height;
  ws.Ensure(totalSize);

  checkCudaErrors(
      cudaMemcpy(ws.d_in, data, totalSize, cudaMemcpyHostToDevice));

  dim3 block(32, 32);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  // Separable two-pass filter. Both kernels run on the default stream, so
  // no synchronization is needed between them; the blocking D2H copy below
  // waits for completion.
  BoxFilterRow<<<grid, block>>>(ws.d_in, ws.d_temp, width, height, k);
  BoxFilterCol<<<grid, block>>>(ws.d_temp, ws.d_out, width, height, k);
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(
      cudaMemcpy(data, ws.d_out, totalSize, cudaMemcpyDeviceToHost));
}

namespace {

// Persistent depth texture + output buffer for the free-function normals
// path, so repeated calls do not pay cudaMalloc3DArray / texture-object
// creation / cudaMalloc / cudaFree every time. Reallocated only when the
// image dimensions change. Buffers are intentionally not freed at process
// exit (see BoxFilterWorkspace).
struct NormalsWorkspace {
  cudaArray* d_depthArray = nullptr;
  cudaTextureObject_t texDepth = 0;
  float* d_normals = nullptr;
  cudaExtent extent = {};
  int width = 0;
  int height = 0;
  int num_images = 0;

  void Ensure(int width_, int height_, int num_images_) {
    if (width == width_ && height == height_ && num_images == num_images_) {
      return;
    }
    if (texDepth != 0) {
      checkCudaErrors(cudaDestroyTextureObject(texDepth));
      texDepth = 0;
    }
    if (d_depthArray != nullptr) {
      checkCudaErrors(cudaFreeArray(d_depthArray));
      d_depthArray = nullptr;
    }
    if (d_normals != nullptr) {
      checkCudaErrors(cudaFree(d_normals));
      d_normals = nullptr;
    }

    width = width_;
    height = height_;
    num_images = num_images_;

    // Layered CUDA array for the depth image stack
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    extent = make_cudaExtent(width, height, num_images);
    checkCudaErrors(cudaMalloc3DArray(&d_depthArray, &channelDesc, extent,
                                      cudaArrayLayered));

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_depthArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    checkCudaErrors(
        cudaCreateTextureObject(&texDepth, &resDesc, &texDesc, NULL));

    size_t num_pixels = static_cast<size_t>(width) * height * num_images;
    checkCudaErrors(cudaMalloc(&d_normals, 3 * num_pixels * sizeof(float)));
  }
};

}  // namespace

void ComputeNormalsCudaImpl(int width, int height, float* h_depths,
                            int num_images, const float* h_fx,
                            const float* h_fy, const float* h_cx,
                            const float* h_cy, float* h_normals,
                            float max_connect_z_diff, int step, bool gl_coord) {
  // Not thread-safe, matching the rest of this API (default stream, shared
  // constant memory).
  static NormalsWorkspace ws;

  size_t num_pixels = static_cast<size_t>(width) * height * num_images;
  // The parameter constants are tiny; upload them every call so that
  // changing parameters between calls stays correct.
  cudaMemcpyToSymbol(d_height, &height, sizeof(int));
  cudaMemcpyToSymbol(d_width, &width, sizeof(int));
  cudaMemcpyToSymbol(d_num_images, &num_images, sizeof(int));
  cudaMemcpyToSymbol(d_max_connect_z_diff, &max_connect_z_diff, sizeof(float));
  cudaMemcpyToSymbol(d_step, &step, sizeof(int));
  cudaMemcpyToSymbol(d_gl_coord, &gl_coord, sizeof(bool));
  constexpr bool central_difference = true;
  cudaMemcpyToSymbol(d_central_difference, &central_difference, sizeof(bool));

  cudaMemcpyToSymbol(d_fx, h_fx, num_images * sizeof(float));
  cudaMemcpyToSymbol(d_fy, h_fy, num_images * sizeof(float));
  cudaMemcpyToSymbol(d_cx, h_cx, num_images * sizeof(float));
  cudaMemcpyToSymbol(d_cy, h_cy, num_images * sizeof(float));

  ws.Ensure(width, height, num_images);

  // Upload the depth stack into the persistent layered array
  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcPtr =
      make_cudaPitchedPtr(h_depths, width * sizeof(float), width, height);
  copyParams.dstArray = ws.d_depthArray;
  copyParams.extent = ws.extent;
  copyParams.kind = cudaMemcpyHostToDevice;
  checkCudaErrors(cudaMemcpy3D(&copyParams));

  // The kernel does not write pixels within `step` of the image border, so
  // clear the output first; otherwise stale/garbage values leak into the
  // result there.
  checkCudaErrors(
      cudaMemsetAsync(ws.d_normals, 0, 3 * num_pixels * sizeof(float)));

  dim3 block(16, 16, 1);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y,
            num_images);

  ComputeNormalsTextureMultiCam<<<grid, block>>>(ws.texDepth, ws.d_normals);
  checkCudaErrors(cudaGetLastError());

  // The blocking D2H copy below also serializes against the kernel
  checkCudaErrors(cudaMemcpy(h_normals, ws.d_normals,
                             3 * num_pixels * sizeof(float),
                             cudaMemcpyDeviceToHost));
}

class NormalComputerCuda::Impl {
 public:
  Impl() {}
  Impl(int width, int height, int num_images, const float* h_fx,
       const float* h_fy, const float* h_cx, const float* h_cy,
       float max_connect_z_diff, int step, bool gl_coord, const float* h_R,
       const float* h_t) {
    Init(width, height, num_images, h_fx, h_fy, h_cx, h_cy, max_connect_z_diff,
         step, gl_coord, h_R, h_t);
  }

  ~Impl() {
    if (texDepth != 0) {
      checkCudaErrors(cudaDestroyTextureObject(texDepth));
    }
    if (d_depthArray != nullptr) {
      checkCudaErrors(cudaFreeArray(d_depthArray));
    }
    if (d_normals != nullptr) {
      checkCudaErrors(cudaFree(d_normals));
    }
    if (d_points != nullptr) {
      checkCudaErrors(cudaFree(d_points));
    }
  }

  void Init(int width_, int height_, int num_images_, const float* h_fx,
            const float* h_fy, const float* h_cx, const float* h_cy,
            float max_connect_z_diff, int step_, bool gl_coord,
            const float* h_R, const float* h_t) {
    if (MAX_IMAGES < num_images_) {
      std::cerr << "Error: num_images (" << num_images_
                << ") exceeds MAX_IMAGES (" << MAX_IMAGES << ")" << std::endl;
      return;
    }

    // Release resources from a previous Init()
    if (texDepth != 0) {
      checkCudaErrors(cudaDestroyTextureObject(texDepth));
      texDepth = 0;
    }
    if (d_depthArray != nullptr) {
      checkCudaErrors(cudaFreeArray(d_depthArray));
      d_depthArray = nullptr;
    }
    if (d_normals != nullptr) {
      checkCudaErrors(cudaFree(d_normals));
      d_normals = nullptr;
    }
    if (d_points != nullptr) {
      checkCudaErrors(cudaFree(d_points));
      d_points = nullptr;
    }

    width = width_;
    height = height_;
    num_images = num_images_;
    step = step_;

    size_t num_pixels = width * height * num_images;
    cudaMemcpyToSymbol(d_height, &height, sizeof(int));
    cudaMemcpyToSymbol(d_width, &width, sizeof(int));
    cudaMemcpyToSymbol(d_num_images, &num_images, sizeof(int));
    cudaMemcpyToSymbol(d_max_connect_z_diff, &max_connect_z_diff,
                       sizeof(float));
    cudaMemcpyToSymbol(d_step, &step, sizeof(int));

    cudaMemcpyToSymbol(d_gl_coord, &gl_coord, sizeof(bool));
    constexpr bool central_difference = true;
    cudaMemcpyToSymbol(d_central_difference, &central_difference, sizeof(bool));

    cudaMemcpyToSymbol(d_fx, h_fx, num_images * sizeof(float));
    cudaMemcpyToSymbol(d_fy, h_fy, num_images * sizeof(float));
    cudaMemcpyToSymbol(d_cx, h_cx, num_images * sizeof(float));
    cudaMemcpyToSymbol(d_cy, h_cy, num_images * sizeof(float));

    if (h_R != nullptr) {
      cudaMemcpyToSymbol(d_R, h_R, num_images * 9 * sizeof(float));
    } else {
      std::vector<float> h_R_vec(num_images * 9, 0.0f);
      for (int i = 0; i < num_images; ++i) {
        h_R_vec[i * 9 + 0] = 1.0f;
        h_R_vec[i * 9 + 4] = 1.0f;
        h_R_vec[i * 9 + 8] = 1.0f;
      }
      // Identity matrix
      cudaMemcpyToSymbol(d_R, h_R_vec.data(), num_images * 9 * sizeof(float));
    }
    if (h_t != nullptr) {
      cudaMemcpyToSymbol(d_t, h_t, num_images * 3 * sizeof(float));
    } else {
      // Zero vector
      std::vector<float> h_t_vec(num_images * 3, 0.0f);
      cudaMemcpyToSymbol(d_t, h_t_vec.data(), num_images * 3 * sizeof(float));
    }

    // (4) ・ｽf・ｽo・ｽC・ｽX・ｽ・ｽ・ｽFLayered CUDA Array ・ｽﾌ確・ｽﾛ（・ｽ[・ｽx・ｽ鞫懶ｿｽp・ｽj
    channelDesc = cudaCreateChannelDesc<float>();
    extent = make_cudaExtent(width, height, num_images);
    checkCudaErrors(cudaMalloc3DArray(&d_depthArray, &channelDesc, extent,
                                      cudaArrayLayered));

    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_depthArray;

    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode =
        cudaFilterModePoint;  // ・ｽ・ｽﾔ不・ｽv・ｽﾌ場合・ｽﾍポ・ｽC・ｽ・ｽ・ｽg・ｽt・ｽB・ｽ・ｽ・ｽ^
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;  // ・ｽｳ規・ｽ・ｽ・ｽ・ｽ・ｽW・ｽﾅア・ｽN・ｽZ・ｽX

    // The depth array is persistent, so the texture object over it can be
    // created once here instead of per ComputeNormals call (the previous
    // per-call creation also leaked the old texture object every call).
    checkCudaErrors(
        cudaCreateTextureObject(&texDepth, &resDesc, &texDesc, NULL));

    checkCudaErrors(cudaMalloc(&d_normals, 3 * num_pixels * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_points, 3 * num_pixels * sizeof(float)));
  }

  void ComputeNormals(const float* h_depths) {
    // (5) cudaMemcpy3D ・ｽ・ｽp・ｽ・ｽ・ｽﾄホ・ｽX・ｽg・ｽﾌ深・ｽx・ｽ鞫懶ｿｽf・ｽ[・ｽ^・ｽ・ｽ CUDA Array ・ｽﾖ転・ｽ・ｽ
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(
        const_cast<float*>(h_depths), width * sizeof(float), width, height);
    copyParams.dstArray = d_depthArray;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));

    // The kernel does not write pixels within `step` of the image border,
    // so clear the outputs first to avoid stale values there.
    size_t num_pixels = static_cast<size_t>(width) * height * num_images;
    checkCudaErrors(
        cudaMemsetAsync(d_normals, 0, 3 * num_pixels * sizeof(float)));
    checkCudaErrors(
        cudaMemsetAsync(d_points, 0, 3 * num_pixels * sizeof(float)));

    dim3 block(16, 16, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y,
              num_images);
    unsigned long long shared_mem_size =
        3 * sizeof(float) * (BLOCK_W + 2 * step) * (BLOCK_H + 2 * step);
    ComputeNormalsTextureMultiCam_Shared<<<grid, block, shared_mem_size>>>(
        texDepth, d_normals, d_points);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
  }

  void GetNormalsCpu(float* h_normals) const {
    size_t num_pixels = width * height * num_images;
    checkCudaErrors(cudaMemcpy(h_normals, d_normals,
                               3 * num_pixels * sizeof(float),
                               cudaMemcpyDeviceToHost));
  }

  void GetPointsCpu(float* h_points) const {
    size_t num_pixels = width * height * num_images;
    checkCudaErrors(cudaMemcpy(h_points, d_points,
                               3 * num_pixels * sizeof(float),
                               cudaMemcpyDeviceToHost));
  }

  const float* GetNormalsGpu() const { return d_normals; }

  const float* GetPointsGpu() const { return d_points; }

 private:
  int width;
  int height;
  int num_images;
  int step;
  cudaChannelFormatDesc channelDesc;
  cudaExtent extent;
  cudaResourceDesc resDesc;
  cudaTextureDesc texDesc;
  cudaTextureObject_t texDepth = 0;
  cudaArray* d_depthArray = nullptr;
  float* d_normals = nullptr;
  float* d_points = nullptr;
};

NormalComputerCuda::NormalComputerCuda() { impl_ = std::make_unique<Impl>(); }

NormalComputerCuda::NormalComputerCuda(int width, int height, int num_images,
                                       const float* h_fx, const float* h_fy,
                                       const float* h_cx, const float* h_cy,
                                       float max_connect_z_diff, int step,
                                       bool gl_coord, const float* h_R,
                                       const float* h_t) {
  impl_ =
      std::make_unique<Impl>(width, height, num_images, h_fx, h_fy, h_cx, h_cy,
                             max_connect_z_diff, step, gl_coord, h_R, h_t);
}

NormalComputerCuda::~NormalComputerCuda() {}

void NormalComputerCuda::Init(int width, int height, int num_images,
                              const float* h_fx, const float* h_fy,
                              const float* h_cx, const float* h_cy,
                              float max_connect_z_diff, int step, bool gl_coord,
                              const float* h_R, const float* h_t) {
  impl_->Init(width, height, num_images, h_fx, h_fy, h_cx, h_cy,
              max_connect_z_diff, step, gl_coord, h_R, h_t);
}

const float* NormalComputerCuda::GetNormalsGpu() const {
  return impl_->GetNormalsGpu();
}

const float* NormalComputerCuda::GetPointsGpu() const {
  return impl_->GetPointsGpu();
}

void NormalComputerCuda::ComputeNormals(const float* h_depths) {
  impl_->ComputeNormals(h_depths);
}

void NormalComputerCuda::GetNormalsCpu(float* h_normals) const {
  impl_->GetNormalsCpu(h_normals);
}

void NormalComputerCuda::GetPointsCpu(float* h_points) const {
  impl_->GetPointsCpu(h_points);
}

}  // namespace ugu
