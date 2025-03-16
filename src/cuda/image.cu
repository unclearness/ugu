#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "./helper_cuda.h"
#include "./image.cuh"
#include "ugu/cuda/image.h"

namespace {

#define MAX_IMAGES 32

// 定数メモリ領域
__constant__ int d_width;
__constant__ int d_height;
__constant__ int d_num_images;
__constant__ float d_max_connect_z_diff;
__constant__ int d_step;
__constant__ bool d_gl_coord;
__constant__ bool d_central_difference;

// カメラパラメータ（画像ごとに異なるが枚数は少ないと仮定）
__constant__ float d_fx[MAX_IMAGES];
__constant__ float d_fy[MAX_IMAGES];
__constant__ float d_cx[MAX_IMAGES];
__constant__ float d_cy[MAX_IMAGES];

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

  // box filter 計算
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

__global__ void ComputeNormalsTextureMultiCam(cudaTextureObject_t texDepth,
                                              float* normals) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int n = blockIdx.z * blockDim.z + threadIdx.z;  // 画像インデックス

  // 画像枚数と境界チェック（境界は単純に処理せずゼロ出力）
  if (n >= d_num_images || x <= d_step - 1 || y <= d_step - 1 ||
      x >= d_width - d_step || y >= d_height - d_step) {
    return;
  }

  // 全体の1次元インデックス計算（画像 n の (y, x)）
  int idx = n * d_width * d_height + y * d_width + x;

  // 各画像固有のカメラパラメータを取得
  float fx_val = d_fx[n];
  float fy_val = d_fy[n];
  float cx_val = d_cx[n];
  float cy_val = d_cy[n];

  float inv_fx = 1.0f / fx_val;
  float inv_fy = 1.0f / fy_val;

  // Layered テクスチャから現在の画素の深度値を取得
  float d = tex2DLayered<float>(texDepth, x, y, n);
  if (d <= 0.0f) {
    normals[3 * idx + 0] = 0.0f;
    normals[3 * idx + 1] = 0.0f;
    normals[3 * idx + 2] = 0.0f;
    return;
  }

  // 隣接画素（右および下）の深度も取得
  float d_right = tex2DLayered<float>(texDepth, x + d_step, y, n);
  float d_bottom = tex2DLayered<float>(texDepth, x, y + d_step, n);
  if (d_right <= 0.0f || d_bottom <= 0.0f) {
    normals[3 * idx + 0] = 0.0f;
    normals[3 * idx + 1] = 0.0f;
    normals[3 * idx + 2] = 0.0f;
    return;
  }

  // 画素座標 (u, v)
  float u = (float)x;
  float v = (float)y;

  // 現在の画素の3次元座標計算
  float X = (u - cx_val) * d * inv_fx;
  float Y = (v - cy_val) * d * inv_fy;
  float Z = d;
  float dx_x;
  float dx_y;
  float dx_z;

  float dy_x;
  float dy_y;
  float dy_z;

  // 右隣の画素の3次元座標計算
  float Xr = ((u + d_step) - cx_val) * d_right * inv_fx;
  float Yr = (v - cy_val) * d_right * inv_fy;
  float Zr = d_right;

  // 下隣の画素の3次元座標計算
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

    // 右隣の画素の3次元座標計算
    float Xl = ((u - d_step) - cx_val) * d_left * inv_fx;
    float Yl = (v - cy_val) * d_left * inv_fy;
    float Zl = d_left;

    // 下隣の画素の3次元座標計算
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

  // クロスプロダクトで法線を計算
  float nx = dx_y * dy_z - dx_z * dy_y;
  float ny = dx_z * dy_x - dx_x * dy_z;
  float nz = dx_x * dy_y - dx_y * dy_x;

  // 正規化
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

}  // namespace

namespace ugu {

void BoxFilterCuda3b(int width, int height, void* data, int k) {
#if 1
  uint8_t *d_in, *d_out;
  size_t totalSize = sizeof(uint8_t) * 3 * width * height;
  checkCudaErrors(cudaMalloc(&d_in, totalSize));
  cudaMalloc(&d_out, totalSize);

  cudaMemcpy(d_in, data, totalSize, cudaMemcpyHostToDevice);

  int N = 1 << 20;
  int blocksize = 32;

  dim3 block(blocksize, blocksize);  // 32x32 = 1024 threads
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

#if 0
  BoxFilterNaive<<<grid, block>>>(d_in, d_out, width, height, k);
#endif

#if 0
  size_t sharedMemSize = block.x * block.y * 3 * sizeof(float);

  BoxFilterShared<<<grid, block, sharedMemSize>>>(d_in, d_out, width, height,
                                                  k);
#endif

#if 1
  uint8_t* d_temp;
  cudaMalloc(&d_temp, totalSize);

  BoxFilterRow<<<grid, block>>>(d_in, d_temp, width, height, k);
  cudaDeviceSynchronize();

  BoxFilterCol<<<grid, block>>>(d_temp, d_out, width, height, k);
#endif

  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(data, d_out, totalSize, cudaMemcpyDeviceToHost));

  cudaFree(d_in);
  cudaFree(d_out);
#if 1
  cudaFree(d_temp);
#endif
#else

  uint8_t *d_in, *d_temp, *d_transposed, *d_out_transposed, *d_final_out;
  size_t totalSize = sizeof(uint8_t) * 3 * width * height;

  cudaMalloc(&d_in, totalSize);
  cudaMalloc(&d_temp, totalSize);
  cudaMalloc(&d_transposed, totalSize);
  cudaMalloc(&d_out_transposed, totalSize);
  cudaMalloc(&d_final_out, totalSize);

  cudaMemcpy(d_in, data, totalSize, cudaMemcpyHostToDevice);

  dim3 block(32, 32);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  // 1st Pass: Row filtering
  BoxFilterRow<<<grid, block>>>(d_in, d_temp, width, height, k);
  cudaDeviceSynchronize();

  dim3 transposeBlock(16, 16);
  dim3 transposeGrid((width + transposeBlock.x - 1) / transposeBlock.x,
                     (height + transposeBlock.y - 1) / transposeBlock.y);

  size_t sharedMemSize = block.x * block.y * 3 * sizeof(float);

  // Transpose
  Transpose<<<transposeGrid, transposeBlock, sharedMemSize>>>(
      d_temp, d_transposed, width, height);
  cudaDeviceSynchronize();

  // 2nd Pass: Row filtering for transposed image
  dim3 grid2((height + block.x - 1) / block.x, (width + block.y - 1) / block.y);

  BoxFilterRow<<<grid2, block>>>(d_transposed, d_out_transposed, height, width,
                                 k);
  cudaDeviceSynchronize();

  // Transpose again（d_out_transposed -> d_final_out）
  dim3 transposeGridDim2((height + transposeBlock.y - 1) / transposeBlock.y,
                         (width + transposeBlock.x - 1) / transposeBlock.x);

  Transpose<<<transposeGridDim2, transposeBlock, sharedMemSize>>>(
      d_out_transposed, d_final_out, height, width);
  cudaDeviceSynchronize();

  cudaMemcpy(data, d_final_out, totalSize, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_temp);
  cudaFree(d_transposed);
  cudaFree(d_out_transposed);
  cudaFree(d_final_out);
#endif
}

void ComputeNormalsCudaImpl(int width, int height, float* h_depths,
                            int num_images, const float* h_fx,
                            const float* h_fy, const float* h_cx,
                            const float* h_cy, float* h_normals,
                            float max_connect_z_diff, int step, bool gl_coord) {
  size_t num_pixels = width * height * num_images;
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

  // (4) デバイス側：Layered CUDA Array の確保（深度画像用）
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaExtent extent = make_cudaExtent(width, height, num_images);
  cudaArray* d_depthArray = nullptr;
  checkCudaErrors(
      cudaMalloc3DArray(&d_depthArray, &channelDesc, extent, cudaArrayLayered));

  // (5) cudaMemcpy3D を用いてホストの深度画像データを CUDA Array へ転送
  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcPtr =
      make_cudaPitchedPtr(h_depths, width * sizeof(float), width, height);
  copyParams.dstArray = d_depthArray;
  copyParams.extent = extent;
  copyParams.kind = cudaMemcpyHostToDevice;
  checkCudaErrors(cudaMemcpy3D(&copyParams));

  // (6) テクスチャオブジェクトの設定（Layered 2D テクスチャ）
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = d_depthArray;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModePoint;  // 補間不要の場合はポイントフィルタ
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;  // 非正規化座標でアクセス

  cudaTextureObject_t texDepth = 0;
  checkCudaErrors(cudaCreateTextureObject(&texDepth, &resDesc, &texDesc, NULL));

  // (7) 出力法線用のデバイスメモリ確保（各画素3要素）
  float* d_normals;
  checkCudaErrors(cudaMalloc(&d_normals, 3 * num_pixels * sizeof(float)));

  // (8) カーネル呼び出し設定：ブロックは
  // (16,16,1)、グリッドは画像サイズと枚数に合わせる
  dim3 block(16, 16, 1);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y,
            num_images);

  // std::cout << block.x << " " << block.y << " " << block.z << std::endl;
  // std::cout << grid.x << " " << grid.y << " " << grid.z << std::endl;
  ComputeNormalsTextureMultiCam<<<grid, block>>>(texDepth, d_normals);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(h_normals, d_normals,
                             3 * num_pixels * sizeof(float),
                             cudaMemcpyDeviceToHost));

  // (11) 後始末：テクスチャオブジェクト、CUDA Array、各種メモリの解放
  checkCudaErrors(cudaDestroyTextureObject(texDepth));
  checkCudaErrors(cudaFreeArray(d_depthArray));
  checkCudaErrors(cudaFree(d_normals));
  // checkCudaErrors(cudaFree(d_fx));
  // checkCudaErrors(cudaFree(d_fy));
  // checkCudaErrors(cudaFree(d_cx));
  // checkCudaErrors(cudaFree(d_cy));
  //  free(h_depth);
  //  free(h_fx);
  //  free(h_fy);
  //  free(h_cx);
  //  free(h_cy);
  //  free(h_normals);
}

class NormalComputerCuda::Impl {
 public:
  Impl() {}
  Impl(int width, int height, int num_images, const float* h_fx,
       const float* h_fy, const float* h_cx, const float* h_cy,
       float max_connect_z_diff, int step, bool gl_coord) {
    this->width = width;
    this->height = height;
    this->num_images = num_images;

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

    // (4) デバイス側：Layered CUDA Array の確保（深度画像用）
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
        cudaFilterModePoint;  // 補間不要の場合はポイントフィルタ
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;  // 非正規化座標でアクセス

    checkCudaErrors(cudaMalloc(&d_normals, 3 * num_pixels * sizeof(float)));
  }
  ~Impl() {
    checkCudaErrors(cudaDestroyTextureObject(texDepth));
    checkCudaErrors(cudaFreeArray(d_depthArray));
    checkCudaErrors(cudaFree(d_normals));
  }

  void ComputeNormals(float* h_depths, float* h_normals) {
    // (5) cudaMemcpy3D を用いてホストの深度画像データを CUDA Array へ転送
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr =
        make_cudaPitchedPtr(h_depths, width * sizeof(float), width, height);
    copyParams.dstArray = d_depthArray;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));

    texDepth = 0;
    checkCudaErrors(
        cudaCreateTextureObject(&texDepth, &resDesc, &texDesc, NULL));

    dim3 block(16, 16, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y,
              num_images);
    ComputeNormalsTextureMultiCam<<<grid, block>>>(texDepth, d_normals);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    size_t num_pixels = width * height * num_images;

    checkCudaErrors(cudaMemcpy(h_normals, d_normals,
                               3 * num_pixels * sizeof(float),
                               cudaMemcpyDeviceToHost));
  }

 private:
  int width;
  int height;
  int num_images;
  cudaChannelFormatDesc channelDesc;
  cudaExtent extent;
  cudaResourceDesc resDesc;
  cudaTextureDesc texDesc;
  cudaTextureObject_t texDepth = 0;
  cudaArray* d_depthArray = nullptr;
  float* d_normals = nullptr;
};

NormalComputerCuda::NormalComputerCuda() { impl_ = std::make_unique<Impl>(); }

NormalComputerCuda::NormalComputerCuda(int width, int height, int num_images,
                                       const float* h_fx, const float* h_fy,
                                       const float* h_cx, const float* h_cy,
                                       float max_connect_z_diff, int step,
                                       bool gl_coord) {
  impl_ = std::make_unique<Impl>(width, height, num_images, h_fx, h_fy, h_cx,
                                 h_cy, max_connect_z_diff, step, gl_coord);
}
NormalComputerCuda::~NormalComputerCuda() {}

void NormalComputerCuda::ComputeNormals(float* h_depths, float* h_normals) {
  impl_->ComputeNormals(h_depths, h_normals);
}

}  // namespace ugu