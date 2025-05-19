#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "./helper_cuda.h"
#include "./voxel.cuh"
#include "ugu/cuda/voxel.h"
#include "ugu/util/image_util.h"

namespace {

#define MAX_IMAGES 32

// カメラパラメータ（画像ごとに異なるが枚数は少ないと仮定）
__constant__ float d_fx[MAX_IMAGES];
__constant__ float d_fy[MAX_IMAGES];
__constant__ float d_cx[MAX_IMAGES];
__constant__ float d_cy[MAX_IMAGES];

// camera to world tranformation
__constant__ float d_R[MAX_IMAGES * 9];
__constant__ float d_t[MAX_IMAGES * 3];

// 定数
#define BLOCK_SIZE 8  // 各VoxelBlockは BLOCK_SIZE^3 個のVoxelを持つ

// Voxel構造体：SDF、重み、法線を保持
struct Voxel {
  float sdf;
  float weight;
  float3 normal;
};

// VoxelBlock構造体
struct VoxelBlock {
  Voxel voxels[BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE];
};

// ハッシュテーブルのエントリ構造体
struct HashEntry {
  int3 pos;  // ボクセルブロック座標（ブロック単位）
  int ptr;   // d_voxelBlocks 配列内のインデックス。未割当は -1
};

//
// ---------------------
// Marching Cubes テーブル（標準テーブル：完全な内容は省略）
// ---------------------
// ※ 実装時は各テーブルの全要素を定義する必要があります。
//
__device__ __constant__ int d_edgeTable[256] = {
    0x0,   0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f,
    0xb06, 0xc0a, 0xd03, 0xe09, 0xf00, 0x190, 0x99,  0x393, 0x29a, 0x596, 0x49f,
    0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 0x230,
    0x339, 0x33,  0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936,
    0xe3a, 0xf33, 0xc39, 0xd30, 0x3a0, 0x2a9, 0x1a3, 0xaa,  0x7a6, 0x6af, 0x5a5,
    0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, 0x460, 0x569,
    0x663, 0x76a, 0x66,  0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a,
    0x963, 0xa69, 0xb60, 0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff,  0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0, 0x650, 0x759, 0x453,
    0x55a, 0x256, 0x35f, 0x55,  0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53,
    0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,  0xfcc,
    0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 0x8c0, 0x9c9, 0xac3, 0xbca,
    0xcc6, 0xdcf, 0xec5, 0xfcc, 0xcc,  0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9,
    0x7c0, 0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x55,
    0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650, 0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6,
    0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0xff,  0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f,
    0x66,  0x76a, 0x663, 0x569, 0x460, 0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af,
    0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa,  0x1a3, 0x2a9, 0x3a0, 0xd30,
    0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636,
    0x13a, 0x33,  0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895,
    0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99,  0x190, 0xf00, 0xe09,
    0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a,
    0x203, 0x109, 0x0};

__device__ __constant__ int d_triTable[256][16] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
    {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
    {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
    {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
    {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
    {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
    {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
    {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
    {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
    {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
    {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
    {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
    {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
    {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
    {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
    {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
    {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
    {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
    {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
    {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
    {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
    {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
    {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
    {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
    {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
    {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
    {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
    {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
    {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
    {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
    {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
    {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
    {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
    {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
    {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
    {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
    {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
    {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
    {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
    {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
    {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
    {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
    {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
    {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
    {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
    {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
    {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
    {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
    {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
    {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
    {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
    {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
    {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
    {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
    {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
    {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
    {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
    {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
    {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
    {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
    {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
    {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
    {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
    {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
    {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
    {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
    {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
    {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
    {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
    {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
    {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
    {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
    {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
    {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
    {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
    {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
    {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
    {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
    {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
    {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
    {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
    {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
    {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
    {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
    {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
    {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
    {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
    {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
    {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
    {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
    {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
    {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
    {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
    {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
    {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
    {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
    {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};

__device__ __constant__ float3 c_bb_min;
__device__ __constant__ float3 c_voxel_size;
__device__ __constant__ float3 c_inv_voxel_size;
__device__ __constant__ int3 c_voxel_num;
__device__ __constant__ float c_trunc;

//
// GPU内ユーティリティ関数
//
__device__ inline float dot(const float3 a, const float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ inline float3 operator+(const float3 a, const float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ inline float3 operator-(const float3 a, const float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ inline float3 operator*(const float3 a, const float b) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}
__device__ inline float3 operator/(const float3 a, const float b) {
  return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ inline float3 cross(const float3 a, const float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

__device__ inline float3 normalize(const float3 a) {
  float len = sqrtf(dot(a, a));
  return (len > 0.0f) ? a / len : make_float3(0.0f, 0.0f, 0.0f);
}

__device__ inline float3 voxel_idx2pos(int3 idx, float3 bb_min,
                                       float3 resolution) {
  return make_float3(bb_min.x + idx.x * resolution.x,
                     bb_min.y + idx.y * resolution.y,
                     bb_min.z + idx.z * resolution.z) +
         resolution * 0.5f;
}

__device__ inline int3 voxel_pos2voxel(const float3 pos, const float3 bb_min,
                                       const float3 resolution) {
  // グリッド原点からのオフセット
  float3 d = make_float3(pos.x - bb_min.x, pos.y - bb_min.y, pos.z - bb_min.z);
  // 各軸ごとにセル長で割り、floor して含まれるセルを得る
  int ix = floorf(d.x / resolution.x);
  int iy = floorf(d.y / resolution.y);
  int iz = floorf(d.z / resolution.z);
  return make_int3(ix, iy, iz);
}

//
// ハッシュテーブル初期化カーネル（各エントリの ptr を -1 に設定）
//
__global__ void initHashTable(HashEntry* hash_table, int hashTableSize) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < hashTableSize) {
    hash_table[idx].ptr = -1;
  }
}

__global__ void initVoxelBlocks(VoxelBlock* d_voxelBlocks,
                                int voxelBlockCount) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE; i++) {
    d_voxelBlocks[idx].voxels[i].sdf = 0.0f;
    d_voxelBlocks[idx].voxels[i].weight = 0.0f;
    d_voxelBlocks[idx].voxels[i].normal = make_float3(0.0f, 0.0f, 0.0f);
  }
}

struct VoxelCudaNaive {
  // float3 index{-1, -1, -1};  // voxel index
  // int id{-1};
  // float3 pos{0.0f, 0.0f, 0.0f};  // center of voxel
  // float3 col{0.0f, 0.0f, 0.0f};
  float sdf_sum{0.f};  // Signed Distance Function (SDF) value
  int update_num{0};
  VoxelCudaNaive(){};
  ~VoxelCudaNaive(){};
};

//
// 法線付き点群からVoxel Hash FusionによるSDF更新を行うカーネル
//
__global__ void fusePointCloudKernel(const float3* points,
                                     const float3* normals, int num_points,
                                     HashEntry* d_hashTable, int hashTableSize,
                                     VoxelBlock* d_voxel_blocks,
                                     int* d_globalVoxelBlockCounter, float mu,
                                     float voxel_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_points) return;

  // 入力点と法線を取得
  float3 pt = points[idx];
  float3 normal = normals[idx];

  // 点の位置からグローバルVoxel座標を計算
  int vx = floorf(pt.x / voxel_size);
  int vy = floorf(pt.y / voxel_size);
  int vz = floorf(pt.z / voxel_size);
  int3 voxel_coord = make_int3(vx, vy, vz);

  // ボクセルブロック座標およびブロック内ローカル座標を計算
  int bx = voxel_coord.x / BLOCK_SIZE;
  int by = voxel_coord.y / BLOCK_SIZE;
  int bz = voxel_coord.z / BLOCK_SIZE;
  int3 block_coord = make_int3(bx, by, bz);

  int lx = voxel_coord.x - bx * BLOCK_SIZE;
  int ly = voxel_coord.y - by * BLOCK_SIZE;
  int lz = voxel_coord.z - bz * BLOCK_SIZE;

  // ハッシュ関数（単純な組み合わせ）
  int h1 = (block_coord.x * 73856093) ^ (block_coord.y * 19349663) ^
           (block_coord.z * 83492791);
  h1 = h1 % hashTableSize;
  if (h1 < 0) h1 += hashTableSize;

  // 二重ハッシュ法による探索
  int h2 = 1 + (h1 % (hashTableSize - 1));
  int found = -1;
  for (int i = 0; i < hashTableSize; i++) {
    int h = (h1 + i * h2) % hashTableSize;
    if (d_hashTable[h].ptr != -1 && d_hashTable[h].pos.x == block_coord.x &&
        d_hashTable[h].pos.y == block_coord.y &&
        d_hashTable[h].pos.z == block_coord.z) {
      found = h;
      break;
    }
    if (d_hashTable[h].ptr == -1) {
      d_hashTable[h].pos = block_coord;
      int new_ptr = atomicAdd(d_globalVoxelBlockCounter, 1);
      d_hashTable[h].ptr = new_ptr;
      found = h;
      break;
    }
  }
  if (found == -1) return;

  int block_idx = d_hashTable[found].ptr;
  VoxelBlock* block = &d_voxel_blocks[block_idx];
  int voxel_index = lx + ly * BLOCK_SIZE + lz * BLOCK_SIZE * BLOCK_SIZE;

  // 対象Voxelの中心座標（ワールド空間）
  float3 voxel_center;
  voxel_center.x = ((bx * BLOCK_SIZE + lx) + 0.5f) * voxel_size;
  voxel_center.y = ((by * BLOCK_SIZE + ly) + 0.5f) * voxel_size;
  voxel_center.z = ((bz * BLOCK_SIZE + lz) + 0.5f) * voxel_size;

  // 点とVoxel中心との相対位置からSDFを計算（法線方向への射影距離）
  float3 diff = voxel_center - pt;
  float dist = dot(diff, normal);
  if (dist < -mu) return;
  float sdf = fminf(1.0f, dist / mu);

  // 更新：重み付き平均でSDFと法線を融合
  Voxel* voxel = &block->voxels[voxel_index];
  float new_weight = 1.0f;
  float total_weight = voxel->weight + new_weight;
  voxel->sdf = (voxel->sdf * voxel->weight + sdf * new_weight) / total_weight;
  float3 fused_normal =
      (voxel->normal * voxel->weight + normal * new_weight) / total_weight;
  voxel->normal = normalize(fused_normal);
  voxel->weight = total_weight;
}

__global__ void fuseOrganizedPointCloudMultiKernelHashing(
    const float3* d_points, const float3* d_normals, int width, int height,
    int num_images, HashEntry* d_hashTable, int hashTableSize,
    VoxelBlock* d_voxel_blocks, int* d_globalVoxelBlockCounter, float mu,
    float voxel_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int totalPixels = width * height * num_images;
  if (idx >= totalPixels) return;

  // 画像毎に連続して格納されているので、idx で直接アクセス
  float3 pt = d_points[idx];
  if (pt.x == 0.0f && pt.y == 0.0f && pt.z == 0.0f) {
    return;  // 無効な点はスキップ
  }

  float3 normal = d_normals[idx];

  // 更新するステップ数: トランケーション幅 [-mu, mu] を voxel_size ごとに走査
  int numSteps = (int)(2.0f * mu / voxel_size) +
                 1;  // 例: mu=0.1, voxel_size=0.005 → 約41ステップ

  for (int i = 0; i < numSteps; i++) {
    // t を -mu から +mu の間で走査し、法線方向の候補点を計算
    float t = -mu + i * voxel_size;
    float3 candidate = pt + normal * t;

    // candidate の位置からボクセル座標を計算
    int vx = floorf(candidate.x / voxel_size);
    int vy = floorf(candidate.y / voxel_size);
    int vz = floorf(candidate.z / voxel_size);
    int3 voxel_coord = make_int3(vx, vy, vz);

    // 対応するボクセルブロックの座標（各ブロックは BLOCK_SIZE 個のVoxelを持つ）
    int bx = voxel_coord.x / BLOCK_SIZE;
    int by = voxel_coord.y / BLOCK_SIZE;
    int bz = voxel_coord.z / BLOCK_SIZE;
    int3 block_coord = make_int3(bx, by, bz);

    // ブロック内でのローカル座標
    int lx = voxel_coord.x - bx * BLOCK_SIZE;
    int ly = voxel_coord.y - by * BLOCK_SIZE;
    int lz = voxel_coord.z - bz * BLOCK_SIZE;

    // ハッシュ関数によるブロック探索（ブロック単位で管理）
    int h1 = (block_coord.x * 73856093) ^ (block_coord.y * 19349663) ^
             (block_coord.z * 83492791);
    h1 = h1 % hashTableSize;
    if (h1 < 0) h1 += hashTableSize;
    int h2 = 1 + (h1 % (hashTableSize - 1));
    int found = -1;
    for (int j = 0; j < hashTableSize; j++) {
      int h = (h1 + j * h2) % hashTableSize;
      if (d_hashTable[h].ptr != -1 && d_hashTable[h].pos.x == block_coord.x &&
          d_hashTable[h].pos.y == block_coord.y &&
          d_hashTable[h].pos.z == block_coord.z) {
        found = h;
        break;
      }

      if (d_hashTable[h].ptr == -1) {
        d_hashTable[h].pos = block_coord;
        int new_ptr = atomicAdd(d_globalVoxelBlockCounter, 1);
        d_hashTable[h].ptr = new_ptr;
        found = h;
        break;
      }
    }
    if (found == -1)
      continue;  // ハッシュテーブルが満杯の場合、この候補はスキップ

    int block_idx = d_hashTable[found].ptr;
    VoxelBlock* block = &d_voxel_blocks[block_idx];
    int voxel_index = lx + ly * BLOCK_SIZE + lz * BLOCK_SIZE * BLOCK_SIZE;

    // 対象Voxelの中心座標（ワールド座標）を計算
    float3 voxel_center;
    voxel_center.x = ((bx * BLOCK_SIZE + lx) + 0.5f) * voxel_size;
    voxel_center.y = ((by * BLOCK_SIZE + ly) + 0.5f) * voxel_size;
    voxel_center.z = ((bz * BLOCK_SIZE + lz) + 0.5f) * voxel_size;

    // pt からの法線方向距離（candidate ではなく voxel_center を使って再計算）
    float3 diff = voxel_center - pt;
    float dist = dot(diff, normal);
    if (fabsf(dist) > mu) continue;  // トランケーション幅外は更新しない
    float sdf = fminf(1.0f, dist / mu);

    // 重み付き平均による SDF と法線の融合更新
    Voxel* voxel = &block->voxels[voxel_index];
    float new_weight = 1.0f;
    float total_weight = voxel->weight + new_weight;
    voxel->sdf = (voxel->sdf * voxel->weight + sdf * new_weight) / total_weight;
    float3 fused_normal =
        (voxel->normal * voxel->weight + normal * new_weight) / total_weight;
    voxel->normal = normalize(fused_normal);
    voxel->weight = total_weight;
  }
}

__global__ void FuseOrganizedPointCloudMultiKernelNaive(
    const float3* d_points, const float3* d_normals, int width, int height,
    int num_images, VoxelCudaNaive* d_voxel_, float3 voxel_size, float3 bb_max,
    float3 bb_min, int3 voxel_num, float truncation_band, float weight,
    int sample_num, int nn_range, float r, float height_half) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int totalPixels = width * height * num_images;
  if (idx >= totalPixels) {
    return;
  }

  float3 pt = d_points[idx];
  if (pt.x == 0.0f && pt.y == 0.0f && pt.z == 0.0f) {
    return;
  }

  float3 normal = d_normals[idx];

  if (0 < r && 0 < height_half) {
    // Update voxel with AABB surrounding cylinder
    float3 tangent = make_float3(0.0f, 0.0f, 0.0f);
    if (fabsf(normal.x) > fabsf(normal.y) &&
        fabsf(normal.x) > fabsf(normal.z)) {
      tangent = make_float3(0.0f, normal.z, -normal.y);
    } else if (fabsf(normal.y) > fabsf(normal.x) &&
               fabsf(normal.y) > fabsf(normal.z)) {
      tangent = make_float3(-normal.z, 0.0f, normal.x);
    } else {
      tangent = make_float3(-normal.y, normal.x, 0.0f);
    }

    tangent = normalize(tangent);
    float3 tangent_abs =
        make_float3(fabsf(tangent.x), fabsf(tangent.y), fabsf(tangent.z));

    float3 normal_abs =
        make_float3(fabsf(normal.x), fabsf(normal.y), fabsf(normal.z));

    float3 normal_offset = make_float3(height_half, height_half, height_half);

    float min_x = pt.x - normal_abs.x * normal_offset.x - r * tangent_abs.x;
    float max_x = pt.x + normal_abs.x * normal_offset.x + r * tangent_abs.x;
    float min_y = pt.y - normal_abs.y * normal_offset.y - r * tangent_abs.y;
    float max_y = pt.y + normal_abs.y * normal_offset.y + r * tangent_abs.y;
    float min_z = pt.z - normal_abs.z * normal_offset.z - r * tangent_abs.z;
    float max_z = pt.z + normal_abs.z * normal_offset.z + r * tangent_abs.z;

    float3 max_pos = make_float3(max_x, max_y, max_z);
    int3 max_idx = voxel_pos2voxel(max_pos, bb_min, voxel_size);
    int max_x_index = max_idx.x;
    int max_y_index = max_idx.y;
    int max_z_index = max_idx.z;

    float3 min_pos = make_float3(min_x, min_y, min_z);
    int3 min_idx = voxel_pos2voxel(min_pos, bb_min, voxel_size);
    int min_x_index = min_idx.x;
    int min_y_index = min_idx.y;
    int min_z_index = min_idx.z;

    if (min_x_index < 0) {
      min_x_index = 0;
    }
    if (max_x_index >= voxel_num.x) {
      max_x_index = voxel_num.x - 1;
    }
    if (min_y_index < 0) {
      min_y_index = 0;
    }
    if (max_y_index >= voxel_num.y) {
      max_y_index = voxel_num.y - 1;
    }
    if (min_z_index < 0) {
      min_z_index = 0;
    }
    if (max_z_index >= voxel_num.z) {
      max_z_index = voxel_num.z - 1;
    }

    for (int z_index = min_z_index; z_index <= max_z_index; z_index++) {
      for (int y_index = min_y_index; y_index <= max_y_index; y_index++) {
        for (int x_index = min_x_index; x_index <= max_x_index; x_index++) {
          float3 voxel_pos = voxel_idx2pos(make_int3(x_index, y_index, z_index),
                                           bb_min, voxel_size);

          // Distance from the voxel center to the point
          float3 diff = voxel_pos - pt;
          float d_dot_n = dot(diff, normal);
          float dist = d_dot_n;

          if (dist >= -truncation_band) {
            dist = fminf(1.0f, dist / truncation_band);
          } else {
            continue;
          }

          VoxelCudaNaive* voxel =
              &d_voxel_[x_index + y_index * voxel_num.x +
                        z_index * voxel_num.x * voxel_num.y];

          atomicAdd(&voxel->sdf_sum, dist * weight);
          atomicAdd(&voxel->update_num, 1);
        }
      }
    }
  }

  if (0 < nn_range) {
    // Update neighboring voxels
    int3 idx = voxel_pos2voxel(pt, bb_min, voxel_size);
    int x_index_ = idx.x;
    int y_index_ = idx.y;
    int z_index_ = idx.z;
    for (int z = -nn_range; z <= nn_range; z++) {
      int z_index = z_index_ + z;
      if (z_index < 0 || voxel_num.z - 1 < z_index) {
        continue;
      }
      for (int y = -nn_range; y <= nn_range; y++) {
        int y_index = y_index_ + y;
        if (y_index < 0 || voxel_num.y - 1 < y_index) {
          continue;
        }
        for (int x = -nn_range; x <= nn_range; x++) {
          int x_index = x_index_ + x;
          if (x_index < 0 || voxel_num.x - 1 < x_index) {
            continue;
          }

          float3 voxel_pos = voxel_idx2pos(make_int3(x_index, y_index, z_index),
                                           bb_min, voxel_size);

          // Distance from the voxel center to the point
          float3 diff = voxel_pos - pt;
          float d_dot_n = dot(diff, normal);
          float dist = d_dot_n;

          if (dist >= -truncation_band) {
            dist = fminf(1.0f, dist / truncation_band);
          } else {
            continue;
          }

          VoxelCudaNaive* voxel =
              &d_voxel_[x_index + y_index * voxel_num.x +
                        z_index * voxel_num.x * voxel_num.y];

          atomicAdd(&voxel->sdf_sum, dist * weight);
          atomicAdd(&voxel->update_num, 1);
        }
      }
    }
  }

  if (0 < sample_num) {
    // Update voxel along the normal direction
    for (int k = -sample_num; k < sample_num + 1; k++) {
      float3 offset;
      offset.x = k * voxel_size.x * normal.x;
      offset.y = k * voxel_size.y * normal.y;
      offset.z = k * voxel_size.z * normal.z;

      float3 ray_pos = pt + offset;
      if (ray_pos.x < bb_min.x || ray_pos.x > bb_max.x ||
          ray_pos.y < bb_min.y || ray_pos.y > bb_max.y ||
          ray_pos.z < bb_min.z || ray_pos.z > bb_max.z) {
        continue;
      }

      int3 idx = voxel_pos2voxel(ray_pos, bb_min, voxel_size);
      int x_index = idx.x;
      int y_index = idx.y;
      int z_index = idx.z;

      if (x_index < 0 || voxel_num.x - 1 < x_index || y_index < 0 ||
          voxel_num.y - 1 < y_index || z_index < 0 ||
          voxel_num.z - 1 < z_index) {
        continue;
      }

      float3 voxel_pos = voxel_idx2pos(make_int3(x_index, y_index, z_index),
                                       bb_min, voxel_size);

      float3 diff = voxel_pos - pt;
      float d_dot_n = dot(diff, normal);
      float dist = d_dot_n;
#
      if (dist >= -truncation_band) {
        dist = fminf(1.0f, dist / truncation_band);
      } else {
        continue;
      }

      VoxelCudaNaive* voxel = &d_voxel_[x_index + y_index * voxel_num.x +
                                        z_index * voxel_num.x * voxel_num.y];

      atomicAdd(&voxel->sdf_sum, dist * weight);
      atomicAdd(&voxel->update_num, 1);
    }
  }
}

// constexpr int NAIVE_KERNEL_HASH_SIZE = 2048;
// extern __shared__ int s_keys[];
// extern __shared__ float s_vals[];
// extern __shared__ int s_counts[];

__global__ void FuseOrganizedPointCloudMultiKernelNaiveOptimized(
    const float3* __restrict__ d_pts, const float3* __restrict__ d_nml,
    VoxelCudaNaive* d_voxels, int totalPixels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // int lane = threadIdx.x;
  //// 1) shared を初期化 (各スレッドで分担)
  // for (int i = lane; i < NAIVE_KERNEL_HASH_SIZE; i += blockDim.x) {
  //   s_keys[i] = -1;  // empty marker
  //   s_vals[i] = 0.0f;
  //   s_counts[i] = 0;
  // }
  //__syncthreads();

  if (idx >= totalPixels) return;

  float3 pt = d_pts[idx];
  if (pt.x == 0 && pt.y == 0 && pt.z == 0) return;

  // 1) インデックス計算（乗算＋キャスト）
  float3 diff = pt - c_bb_min;
  int xi = __float2int_rz(diff.x * c_inv_voxel_size.x);
  int yi = __float2int_rz(diff.y * c_inv_voxel_size.y);
  int zi = __float2int_rz(diff.z * c_inv_voxel_size.z);
  if (xi < 0 || xi >= c_voxel_num.x || yi < 0 || yi >= c_voxel_num.y ||
      zi < 0 || zi >= c_voxel_num.z) {
    return;
  }

  // 2) オフセットループはアンロール済み
  float3 normal = d_nml[idx];
  constexpr int nn_range = 1;  // nn_range=1 なら 3×3×3 ブロック
  constexpr int MAX_NEI =
      (2 * nn_range + 1) * (2 * nn_range + 1) * (2 * nn_range + 1);
  float local_contribs[MAX_NEI];  // nn_range=1 なら 3×3 ブロック
  int local_idxs[MAX_NEI];        // flatten したインデックス
  int cnt = 0;

#pragma unroll
  for (int dz = -nn_range; dz <= nn_range; ++dz) {
    for (int dy = -nn_range; dy <= nn_range; ++dy) {
      for (int dx = -nn_range; dx <= nn_range; ++dx) {
        int xj = xi + dx;
        int yj = yi + dy;
        int zj = zi + dz;
        // 範囲チェック
        if (xj < 0 || xj >= c_voxel_num.x || yj < 0 || yj >= c_voxel_num.y ||
            zj < 0 || zj >= c_voxel_num.z) {
          continue;
        }
        int vid = xj + yj * c_voxel_num.x + zj * c_voxel_num.x * c_voxel_num.y;
        float3 vpos = c_bb_min;
        vpos.x += xj * c_voxel_size.x;
        vpos.y += yj * c_voxel_size.y;
        vpos.z += zj * c_voxel_size.z;
        float3 dff = vpos - pt;
        // SDF の計算は sqrt を可能なら rsqrt で高速化
        float sign = dot(dff, normal) < 0 ? -1.f : 1.f;
        float dist = sqrtf(dot(dff, dff)) * sign;
        dist = (dist >= -c_trunc) ? fminf(1.f, dist / c_trunc) : 0.f;
        local_idxs[cnt] = vid;
        local_contribs[cnt] = dist;
        ++cnt;

        //// 3) shared-hash にインサート＋集約
        ////    simple linear‐probing
        // int slot = vid & (NAIVE_KERNEL_HASH_SIZE - 1);
        // while (true) {
        //   int old = atomicCAS(&s_keys[slot], -1, vid);
        //   if (old == -1 || old == vid) {
        //     // 同じ vid ならここで集約
        //     atomicAdd(&s_vals[slot], dist);
        //     atomicAdd(&s_counts[slot], 1);
        //     break;
        //   }
        //   slot = (slot + 1) & (NAIVE_KERNEL_HASH_SIZE - 1);
        // }
      }
    }
  }

  //// 3) スレッド内集約 → atomicAdd は cnt 回ではなく「有効な vid 数」回に
  for (int i = 0; i < cnt; ++i) {
    atomicAdd(&d_voxels[local_idxs[i]].sdf_sum, local_contribs[i]);
    atomicAdd(&d_voxels[local_idxs[i]].update_num, 1);
  }

  constexpr int sample_num = 2;
  constexpr int MAX_SAMPLE = sample_num * 2 + 1;
  float local_contribs_ray[MAX_SAMPLE];
  int local_idxs_ray[MAX_SAMPLE];  // flatten したインデックス
  int cnt_ray = 0;

#pragma unroll
  for (int k = -sample_num; k <= sample_num; k++) {
    float3 offset;
    offset.x = k * c_voxel_size.x * normal.x;
    offset.y = k * c_voxel_size.y * normal.y;
    offset.z = k * c_voxel_size.z * normal.z;

    float3 ray_pos = pt + offset;

    float3 ray_diff = ray_pos - c_bb_min;
    int x_index = floorf(ray_diff.x / c_voxel_size.x);
    int y_index = floorf(ray_diff.y / c_voxel_size.y);
    int z_index = floorf(ray_diff.z / c_voxel_size.z);

    if (x_index < 0 || c_voxel_num.x - 1 < x_index || y_index < 0 ||
        c_voxel_num.y - 1 < y_index || z_index < 0 ||
        c_voxel_num.z - 1 < z_index) {
      continue;
    }

    int vid = x_index + y_index * c_voxel_num.x +
              z_index * c_voxel_num.x * c_voxel_num.y;

    float3 vpos = c_bb_min;
    vpos.x += x_index * c_voxel_size.x;
    vpos.y += y_index * c_voxel_size.y;
    vpos.z += z_index * c_voxel_size.z;
    float3 dff = vpos - pt;
    // SDF の計算は sqrt を可能なら rsqrt で高速化
    float sign = dot(dff, normal) < 0 ? -1.f : 1.f;
    float dist = sqrtf(dot(dff, dff)) * sign;
    dist = (dist >= -c_trunc) ? fminf(1.f, dist / c_trunc) : 0.f;
    local_idxs_ray[cnt_ray] = vid;
    local_contribs_ray[cnt_ray] = dist;
    ++cnt_ray;

    // int slot = vid & (NAIVE_KERNEL_HASH_SIZE - 1);
    // while (true) {
    //   int old = atomicCAS(&s_keys[slot], -1, vid);
    //   if (old == -1 || old == vid) {
    //     // 同じ vid ならここで集約
    //     atomicAdd(&s_vals[slot], dist);
    //     atomicAdd(&s_counts[slot], 1);
    //     break;
    //   }
    //   slot = (slot + 1) & (NAIVE_KERNEL_HASH_SIZE - 1);
    // }
  }

  //__syncthreads();

  // if (lane == 0) {
  //   for (int i = 0; i < NAIVE_KERNEL_HASH_SIZE; ++i) {
  //     int vid = s_keys[i];
  //     if (vid >= 0) {
  //       float sum = s_vals[i];
  //       int cnt = s_counts[i];
  //       // ここでグローバルにまとめて加算
  //       atomicAdd(&d_voxels[vid].sdf_sum, sum);
  //       atomicAdd(&d_voxels[vid].update_num, cnt);
  //     }
  //   }
  // }

  // int uCnt = 0;
  //  // ユニーク化バッファ（最大近傍数に合わせたサイズ）
  // constexpr int MAX_BUF_NUM = MAX_NEI + MAX_SAMPLE;
  // int uidBuf[MAX_BUF_NUM];
  // float sumBuf[MAX_BUF_NUM];
  // int updBuf[MAX_BUF_NUM];

  // for (int i = 0; i < cnt; ++i) {
  //   int vid = local_idxs[i];
  //   float val = local_contribs[i];
  //   // 既に登録済みか線形検索
  //   int j = 0;
  //   for (; j < uCnt; ++j) {
  //     if (uidBuf[j] == vid) {
  //       sumBuf[j] += val;
  //       updBuf[j] += 1;  // update_num 用
  //       break;
  //     }
  //   }
  //   if (j == uCnt) {
  //     // 新規エントリ
  //     uidBuf[uCnt] = vid;
  //     sumBuf[uCnt] = val;
  //     updBuf[uCnt] = 1;
  //     ++uCnt;
  //   }
  // }

  // for (int i = 0; i < cnt_ray; ++i) {
  //   int vid = local_idxs_ray[i];
  //   float val = local_contribs_ray[i];
  //   // 既に登録済みか線形検索
  //   int j = 0;
  //   for (; j < uCnt; ++j) {
  //     if (uidBuf[j] == vid) {
  //       sumBuf[j] += val;
  //       updBuf[j] += 1;  // update_num 用
  //       break;
  //     }
  //   }
  //   if (j == uCnt) {
  //     // 新規エントリ
  //     uidBuf[uCnt] = vid;
  //     sumBuf[uCnt] = val;
  //     updBuf[uCnt] = 1;
  //     ++uCnt;
  //   }
  // }

  // // ここで uCnt はユニークなボクセル数
  // for (int j = 0; j < uCnt; ++j) {
  //   atomicAdd(&d_voxels[uidBuf[j]].sdf_sum, sumBuf[j]);
  //   atomicAdd(&d_voxels[uidBuf[j]].update_num, updBuf[j]);
  // }

  for (int i = 0; i < cnt_ray; ++i) {
    atomicAdd(&d_voxels[local_idxs_ray[i]].sdf_sum, local_contribs_ray[i]);
    atomicAdd(&d_voxels[local_idxs_ray[i]].update_num, 1);
  }
}

__global__ void FuseDepthMultiKernelNaive(const float* d_depth, int width,
                                          int height, int num_images,
                                          VoxelCudaNaive* d_voxel_,
                                          float3 voxel_size, float3 bb_max,
                                          float3 bb_min, int3 voxel_num,
                                          float truncation_band, float weight,
                                          int sample_num, int nn_range) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int n = blockIdx.z * blockDim.z + threadIdx.z;

  if (n >= num_images || x < 0 || y < 0 || x >= width || y >= height) {
    return;
  }

  int idx = n * width * height + y * width + x;
  int totalPixels = width * height * num_images;
  if (idx >= totalPixels) {
    return;
  }

  float d = d_depth[idx];

  if (d <= 0) {
    return;
  }

  float u = (float)x;
  float v = (float)y;

  float fx_val = d_fx[n];
  float fy_val = d_fy[n];
  float cx_val = d_cx[n];
  float cy_val = d_cy[n];

  float inv_fx = 1.0f / fx_val;
  float inv_fy = 1.0f / fy_val;

  float X = (u - cx_val) * d * inv_fx;
  float Y = (v - cy_val) * d * inv_fy;
  float Z = d;

  const float* R = &d_R[n * 9];
  const float* t = &d_t[n * 3];

  float R_inv[9];
  float t_inv[3];

  // Inverse rotation matrix
  R_inv[0] = R[0];
  R_inv[1] = R[3];
  R_inv[2] = R[6];
  R_inv[3] = R[1];
  R_inv[4] = R[4];
  R_inv[5] = R[7];
  R_inv[6] = R[2];
  R_inv[7] = R[5];
  R_inv[8] = R[8];
  // Inverse translation vector
  // t_inv[0] = -(R_inv[0] * t[0] + R_inv[1] * t[1] + R_inv[2] * t[2]);
  // t_inv[1] = -(R_inv[3] * t[0] + R_inv[4] * t[1] + R_inv[5] * t[2]);
  t_inv[2] = -(R_inv[6] * t[0] + R_inv[7] * t[1] + R_inv[8] * t[2]);

  // camera to world
  float3 pt;
  pt.x = R[0] * X + R[1] * Y + R[2] * Z + t[0];
  pt.y = R[3] * X + R[4] * Y + R[5] * Z + t[1];
  pt.z = R[6] * X + R[7] * Y + R[8] * Z + t[2];

  // Use reverse ray as pseudo normal
  float3 ray = normalize(make_float3(X, Y, Z));
  float3 pseudo_normal;
  pseudo_normal.x = R[0] * ray.x + R[1] * ray.y + R[2] * ray.z;
  pseudo_normal.y = R[3] * ray.x + R[4] * ray.y + R[5] * ray.z;
  pseudo_normal.z = R[6] * ray.x + R[7] * ray.y + R[8] * ray.z;
  pseudo_normal.x = -pseudo_normal.x;
  pseudo_normal.y = -pseudo_normal.y;
  pseudo_normal.z = -pseudo_normal.z;

  if (0 < nn_range) {
    float3 diff = pt - bb_min;
    int x_index_ = floorf(diff.x / voxel_size.x);
    int y_index_ = floorf(diff.y / voxel_size.y);
    int z_index_ = floorf(diff.z / voxel_size.z);
    for (int z = -nn_range; z <= nn_range; z++) {
      int z_index = z_index_ + z;
      if (z_index < 0 || voxel_num.z - 1 < z_index) {
        continue;
      }
      for (int y = -nn_range; y <= nn_range; y++) {
        int y_index = y_index_ + y;
        if (y_index < 0 || voxel_num.y - 1 < y_index) {
          continue;
        }
        for (int x = -nn_range; x <= nn_range; x++) {
          int x_index = x_index_ + x;
          if (x_index < 0 || voxel_num.x - 1 < x_index) {
            continue;
          }

          float3 voxel_pos;
          voxel_pos.x = bb_min.x + x_index * voxel_size.x;
          voxel_pos.y = bb_min.y + y_index * voxel_size.y;
          voxel_pos.z = bb_min.z + z_index * voxel_size.z;

          // Distance from the voxel center to the point
          float3 diff = voxel_pos - pt;
          float sign = 1.f;
          float voxel_z_cam = voxel_pos.x * R_inv[6] + voxel_pos.y * R_inv[7] +
                              voxel_pos.z * R_inv[8] + t_inv[2];
          if (Z < voxel_z_cam) {
            sign = -1.f;
          }
          // float d_dot_n = dot(diff, pseudo_normal);
          // if (d_dot_n < 0) {
          //   sign = -1.f;
          // }
          float dist = sqrtf(dot(diff, diff)) * sign;

          if (dist >= -truncation_band) {
            dist = fminf(1.0f, dist / truncation_band);
          } else {
            continue;
          }

          VoxelCudaNaive* voxel =
              &d_voxel_[x_index + y_index * voxel_num.x +
                        z_index * voxel_num.x * voxel_num.y];

          atomicAdd(&voxel->sdf_sum, dist * weight);
          atomicAdd(&voxel->update_num, 1);
        }
      }
    }
  }

  if (0 < sample_num) {
    for (int k = -sample_num; k < sample_num + 1; k++) {
      float3 offset;
      offset.x = k * voxel_size.x * pseudo_normal.x;
      offset.y = k * voxel_size.y * pseudo_normal.y;
      offset.z = k * voxel_size.z * pseudo_normal.z;

      float3 ray_pos = pt + offset;
      if (ray_pos.x < bb_min.x || ray_pos.x > bb_max.x ||
          ray_pos.y < bb_min.y || ray_pos.y > bb_max.y ||
          ray_pos.z < bb_min.z || ray_pos.z > bb_max.z) {
        continue;
      }

      float3 ray_diff = ray_pos - bb_min;
      int x_index = floorf(ray_diff.x / voxel_size.x);
      int y_index = floorf(ray_diff.y / voxel_size.y);
      int z_index = floorf(ray_diff.z / voxel_size.z);

      if (x_index < 0 || voxel_num.x - 1 < x_index || y_index < 0 ||
          voxel_num.y - 1 < y_index || z_index < 0 ||
          voxel_num.z - 1 < z_index) {
        continue;
      }

      float3 voxel_pos;
      voxel_pos.x = bb_min.x + x_index * voxel_size.x;
      voxel_pos.y = bb_min.y + y_index * voxel_size.y;
      voxel_pos.z = bb_min.z + z_index * voxel_size.z;

      // Distance from the voxel center to the point
      float3 diff = voxel_pos - pt;
      float sign = 1.f;
      // float d_dot_n = dot(diff, pseudo_normal);
      // if (d_dot_n < 0) {
      //   sign = -1.f;
      // }
      float voxel_z_cam = voxel_pos.x * R_inv[6] + voxel_pos.y * R_inv[7] +
                          voxel_pos.z * R_inv[8] + t_inv[2];
      if (Z < voxel_z_cam) {
        sign = -1.f;
      }
      float dist = sqrtf(dot(diff, diff)) * sign;

      if (dist >= -truncation_band) {
        dist = fminf(1.0f, dist / truncation_band);
      } else {
        continue;
      }

      VoxelCudaNaive* voxel = &d_voxel_[x_index + y_index * voxel_num.x +
                                        z_index * voxel_num.x * voxel_num.y];

      atomicAdd(&voxel->sdf_sum, dist * weight);
      atomicAdd(&voxel->update_num, 1);
    }
  }
}

__device__ float3 VertexInterp(float3 p1, float3 p2, float valp1, float valp2,
                               float iso_level = 0.f) {
  float diff = valp2 - valp1;
  float t = (fabsf(diff) < 1e-6f) ? 0.5f : (iso_level - valp1) / diff;
  return make_float3(p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y),
                     p1.z + t * (p2.z - p1.z));
}

//
// Marching Cubes によるメッシュ生成カーネル
// 各スレッドは有効なVoxelBlock内の1セル（(BLOCK_SIZE-1)^3個のセル）に対して処理を行う
// isoLevel は 0 を想定（SDF=0 の面）、d_vertices, d_indices
// は出力バッファ（d_indicesは connected==true の場合に書き出す）
// d_vertexCount はグローバル出力頂点数カウンター（各三角形につき3頂点を出力）
//
__global__ void marchingCubesKernel(VoxelBlock* d_voxelBlocks,
                                    int validBlockCount, float voxelSize,
                                    ugu::VertexHostDevice* d_vertices,
                                    int* d_indices, int* d_vertexCount,
                                    bool connected) {
  int cellsPerBlock = (BLOCK_SIZE - 1) * (BLOCK_SIZE - 1) * (BLOCK_SIZE - 1);
  int globalCellIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int totalCells = validBlockCount * cellsPerBlock;
  if (globalCellIdx >= totalCells) return;

  int cellIdxInBlock = globalCellIdx % cellsPerBlock;
  int blockIdxVoxel = globalCellIdx / cellsPerBlock;

  // セル内のローカル座標（各セルは8個の頂点を持つ）
  int cell_z = cellIdxInBlock / ((BLOCK_SIZE - 1) * (BLOCK_SIZE - 1));
  int rem = cellIdxInBlock % ((BLOCK_SIZE - 1) * (BLOCK_SIZE - 1));
  int cell_y = rem / (BLOCK_SIZE - 1);
  int cell_x = rem % (BLOCK_SIZE - 1);

  // 現在のVoxelBlockを取得
  VoxelBlock* curBlock = &d_voxelBlocks[blockIdxVoxel];

  // 各セルの8頂点のオフセット
  int3 offsets[8] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                     {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};

  float sdf[8];
  float3 pos[8];
  float3 norm[8];
  // 各頂点のSDF値、位置、法線を取得（セル内のVoxelはセルの左下奥のVoxelからのオフセット）
  for (int i = 0; i < 8; i++) {
    int vx = cell_x + offsets[i].x;
    int vy = cell_y + offsets[i].y;
    int vz = cell_z + offsets[i].z;
    int voxelIndex = vx + vy * BLOCK_SIZE + vz * BLOCK_SIZE * BLOCK_SIZE;
    Voxel v = curBlock->voxels[voxelIndex];
    sdf[i] = v.sdf;
    pos[i].x = (vx + 0.5f) * voxelSize;
    pos[i].y = (vy + 0.5f) * voxelSize;
    pos[i].z = (vz + 0.5f) * voxelSize;
    norm[i] = v.normal;
  }

  // Marching Cubes のキューブインデックスを計算（isoLevel=0）
  int cubeIndex = 0;
  if (sdf[0] < 0) cubeIndex |= 1;
  if (sdf[1] < 0) cubeIndex |= 2;
  if (sdf[2] < 0) cubeIndex |= 4;
  if (sdf[3] < 0) cubeIndex |= 8;
  if (sdf[4] < 0) cubeIndex |= 16;
  if (sdf[5] < 0) cubeIndex |= 32;
  if (sdf[6] < 0) cubeIndex |= 64;
  if (sdf[7] < 0) cubeIndex |= 128;

  // 交差がなければ処理しない
  if (d_edgeTable[cubeIndex] == 0) return;

  // 各エッジでの交差点と法線を計算するための補助ラムダ
  float3 edgeVertex[12];
  float3 edgeNormal[12];
  auto vertexInterp = [&](int edge, int a, int b) {
    float t = (0 - sdf[a]) / (sdf[b] - sdf[a]);
    edgeVertex[edge] = pos[a] + (pos[b] - pos[a]) * t;
    edgeNormal[edge] = normalize(norm[a] + (norm[b] - norm[a]) * t);
  };

  if (d_edgeTable[cubeIndex] & 1) vertexInterp(0, 0, 1);
  if (d_edgeTable[cubeIndex] & 2) vertexInterp(1, 1, 2);
  if (d_edgeTable[cubeIndex] & 4) vertexInterp(2, 2, 3);
  if (d_edgeTable[cubeIndex] & 8) vertexInterp(3, 3, 0);
  if (d_edgeTable[cubeIndex] & 16) vertexInterp(4, 4, 5);
  if (d_edgeTable[cubeIndex] & 32) vertexInterp(5, 5, 6);
  if (d_edgeTable[cubeIndex] & 64) vertexInterp(6, 6, 7);
  if (d_edgeTable[cubeIndex] & 128) vertexInterp(7, 7, 4);
  if (d_edgeTable[cubeIndex] & 256) vertexInterp(8, 0, 4);
  if (d_edgeTable[cubeIndex] & 512) vertexInterp(9, 1, 5);
  if (d_edgeTable[cubeIndex] & 1024) vertexInterp(10, 2, 6);
  if (d_edgeTable[cubeIndex] & 2048) vertexInterp(11, 3, 7);

  // triTable を参照して三角形を生成
  for (int i = 0; d_triTable[cubeIndex][i] != -1; i += 3) {
    float3 v0 = edgeVertex[d_triTable[cubeIndex][i]];
    float3 v1 = edgeVertex[d_triTable[cubeIndex][i + 1]];
    float3 v2 = edgeVertex[d_triTable[cubeIndex][i + 2]];
    float3 n0 = edgeNormal[d_triTable[cubeIndex][i]];
    float3 n1 = edgeNormal[d_triTable[cubeIndex][i + 1]];
    float3 n2 = edgeNormal[d_triTable[cubeIndex][i + 2]];

    int baseIndex = atomicAdd(d_vertexCount, 3);
    d_vertices[baseIndex + 0].position = v0;
    d_vertices[baseIndex + 0].normal = n0;
    d_vertices[baseIndex + 1].position = v1;
    d_vertices[baseIndex + 1].normal = n1;
    d_vertices[baseIndex + 2].position = v2;
    d_vertices[baseIndex + 2].normal = n2;

    if (connected) {
      d_indices[baseIndex + 0] = baseIndex + 0;
      d_indices[baseIndex + 1] = baseIndex + 1;
      d_indices[baseIndex + 2] = baseIndex + 2;
    }
  }
}

__global__ void InitVoxelsNaive(VoxelCudaNaive* voxels, size_t n,
                                float initSdf) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    voxels[idx].sdf_sum = initSdf;
    voxels[idx].update_num = 0;
  }
}

__global__ void MarchingCubesKernelNaive(
    const VoxelCudaNaive* __restrict__ voxels, float3 bb_min, float3 resolution,
    int3 voxel_num, float weight,
    float3* __restrict__ out_vertices,  // 三角形頂点バッファ
    int* __restrict__ out_counter       // 原子で増加させる頂点数
) {
  // 各スレッドは「セル」（voxel_num-1 の範囲）を担当
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;
  if (ix >= voxel_num.x - 1 || iy >= voxel_num.y - 1 || iz >= voxel_num.z - 1)
    return;

  // セル頂点の格子インデックス
  int3 base = make_int3(ix, iy, iz);

  // 1) 8 コーナーの SDF 値を読み込み
  float sdf[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    int3 offs = make_int3((k & 1) ? 1 : 0, (k & 2) ? 1 : 0, (k & 4) ? 1 : 0);
    int3 idx = make_int3(base.x + offs.x, base.y + offs.y, base.z + offs.z);
    int flat = idx.z * voxel_num.y * voxel_num.x + idx.y * voxel_num.x + idx.x;
    // Ignore if the voxel is not updated
    if (voxels[flat].update_num < 1) {
      return;
    }
    // Take average
    sdf[k] = voxels[flat].sdf_sum / (weight * voxels[flat].update_num);
  }

  const float iso_level = 0.0f;
  // 2) ケースインデックスを計算
  int cubeIndex = 0;
  if (sdf[0] < iso_level) cubeIndex |= 1;
  if (sdf[1] < iso_level) cubeIndex |= 2;
  if (sdf[2] < iso_level) cubeIndex |= 4;
  if (sdf[3] < iso_level) cubeIndex |= 8;
  if (sdf[4] < iso_level) cubeIndex |= 16;
  if (sdf[5] < iso_level) cubeIndex |= 32;
  if (sdf[6] < iso_level) cubeIndex |= 64;
  if (sdf[7] < iso_level) cubeIndex |= 128;

  // 立方体が完全に内部 or 外部なら何もしない
  int edges = d_edgeTable[cubeIndex];
  if (edges == 0) return;

  // 3) 8 頂点のワールド座標を計算
  float3 cornerPos[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    int3 offs = make_int3((k & 1) ? 1 : 0, (k & 2) ? 1 : 0, (k & 4) ? 1 : 0);
    float3 gridPos = make_float3(bb_min.x + (base.x + offs.x) * resolution.x,
                                 bb_min.y + (base.y + offs.y) * resolution.y,
                                 bb_min.z + (base.z + offs.z) * resolution.z);
    cornerPos[k] = gridPos;
  }

  // 4) エッジ上の交点を線形補間で求める
  float3 vertList[12];
  if (edges & 1)
    vertList[0] = VertexInterp(cornerPos[0], cornerPos[1], sdf[0], sdf[1]);
  if (edges & 2)
    vertList[1] = VertexInterp(cornerPos[1], cornerPos[2], sdf[1], sdf[2]);
  if (edges & 4)
    vertList[2] = VertexInterp(cornerPos[2], cornerPos[3], sdf[2], sdf[3]);
  if (edges & 8)
    vertList[3] = VertexInterp(cornerPos[3], cornerPos[0], sdf[3], sdf[0]);
  if (edges & 16)
    vertList[4] = VertexInterp(cornerPos[4], cornerPos[5], sdf[4], sdf[5]);
  if (edges & 32)
    vertList[5] = VertexInterp(cornerPos[5], cornerPos[6], sdf[5], sdf[6]);
  if (edges & 64)
    vertList[6] = VertexInterp(cornerPos[6], cornerPos[7], sdf[6], sdf[7]);
  if (edges & 128)
    vertList[7] = VertexInterp(cornerPos[7], cornerPos[4], sdf[7], sdf[4]);
  if (edges & 256)
    vertList[8] = VertexInterp(cornerPos[0], cornerPos[4], sdf[0], sdf[4]);
  if (edges & 512)
    vertList[9] = VertexInterp(cornerPos[1], cornerPos[5], sdf[1], sdf[5]);
  if (edges & 1024)
    vertList[10] = VertexInterp(cornerPos[2], cornerPos[6], sdf[2], sdf[6]);
  if (edges & 2048)
    vertList[11] = VertexInterp(cornerPos[3], cornerPos[7], sdf[3], sdf[7]);

  // 5) triTable を見て三角形を出力
  for (int t = 0; t < 16; t += 3) {
    int e0 = d_triTable[cubeIndex][t + 0];
    int e1 = d_triTable[cubeIndex][t + 1];
    int e2 = d_triTable[cubeIndex][t + 2];
    if (e0 < 0) break;  // テーブル終端

    // 出力バッファへ原子操作で書き込み
    int triIdx = atomicAdd(out_counter, 3);
    out_vertices[triIdx + 0] = vertList[e0];
    out_vertices[triIdx + 1] = vertList[e1];
    out_vertices[triIdx + 2] = vertList[e2];
  }
}

// edgeId: 0〜11 (Marching Cubes の仕様準拠)
__device__ int computeEdgeKey(int ix, int iy, int iz, int edgeId, int nx,
                              int ny, int nz) {
  int xCount = (nx - 1) * ny * nz;
  int yCount = nx * (ny - 1) * nz;
  // zCount = nx*ny*(nz-1)  // 使うのは後述のケース

  switch (edgeId) {
    // --- 底面 (z) の X, Y エッジ ---
    case 0:  // corner 0–1, X edge at (ix, iy, iz)
      return ix + iy * (nx - 1) + iz * (nx - 1) * ny;
    case 1:  // corner 1–2, Y edge at (ix+1, iy, iz)
      return xCount + (ix + 1) + iy * nx + iz * nx * (ny - 1);
    case 2:  // corner 2–3, X edge at (ix, iy+1, iz)
      return ix + (iy + 1) * (nx - 1) + iz * (nx - 1) * ny;
    case 3:  // corner 3–0, Y edge at (ix, iy, iz)
      return xCount + ix + iy * nx + iz * nx * (ny - 1);

    // --- 上面 (z+1) の X, Y エッジ ---
    case 4:  // corner 4–5, X edge at (ix, iy, iz+1)
      return ix + iy * (nx - 1) + (iz + 1) * (nx - 1) * ny;
    case 5:  // corner 5–6, Y edge at (ix+1, iy, iz+1)
      return xCount + (ix + 1) + iy * nx + (iz + 1) * nx * (ny - 1);
    case 6:  // corner 6–7, X edge at (ix, iy+1, iz+1)
      return ix + (iy + 1) * (nx - 1) + (iz + 1) * (nx - 1) * ny;
    case 7:  // corner 7–4, Y edge at (ix, iy, iz+1)
      return xCount + ix + iy * nx + (iz + 1) * nx * (ny - 1);

    // --- 垂直方向 (Z) のエッジ ---
    // Z-edge 数は nx*ny*(nz-1) ですが、Yオフセットの後ろに続くと考えます。
    case 8:  // corner 0–4, Z edge at (ix, iy, iz)
      return xCount + yCount + ix + iy * nx + iz * nx * ny;
    case 9:  // corner 1–5, Z edge at (ix+1, iy, iz)
      return xCount + yCount + (ix + 1) + iy * nx + iz * nx * ny;
    case 10:  // corner 2–6, Z edge at (ix+1, iy+1, iz)
      return xCount + yCount + (ix + 1) + (iy + 1) * nx + iz * nx * ny;
    case 11:  // corner 3–7, Z edge at (ix, iy+1, iz)
      return xCount + yCount + ix + (iy + 1) * nx + iz * nx * ny;
  }
  return -1;  // 不正な edgeId
}

// Compute cube index based on iso threshold
__device__ int calcCubeIndex(const VoxelCudaNaive* voxels, int ix, int iy,
                             int iz, int3 vn, float3 bb_min, float3 res,
                             float iso_level) {
  float sdf[8];
  int ids[8];
  // offsets for 8 corners
  const int offs[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                          {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
  for (int k = 0; k < 8; k++) {
    int x = ix + offs[k][0];
    int y = iy + offs[k][1];
    int z = iz + offs[k][2];
    int idx = z * vn.y * vn.x + y * vn.x + x;
    // skip if empty
    if (voxels[idx].update_num < 1) return -1;
    sdf[k] = voxels[idx].sdf_sum / float(voxels[idx].update_num);
  }
  int cubeIndex = 0;
  for (int k = 0; k < 8; k++) {
    if (sdf[k] < iso_level) cubeIndex |= (1 << k);
  }
  return cubeIndex;
}

__device__ const int cornerOffset[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0},
                                           {0, 1, 0}, {0, 0, 1}, {1, 0, 1},
                                           {1, 1, 1}, {0, 1, 1}};

// edgeId 0～11 に対して、エッジを構成する２つのコーナー番号
__device__ const int edgeCorners[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},  // 底面
    {4, 5}, {5, 6}, {6, 7}, {7, 4},  // 上面
    {0, 4}, {1, 5}, {2, 6}, {3, 7}   // 垂直エッジ
};

__global__ void BuildVerticesKernel(const VoxelCudaNaive* voxels, float3 bb_min,
                                    float3 resolution,
                                    int3 vn,  // voxel_num
                                    float iso_level, int* d_edgeVertexIds,
                                    int* d_vtxCounter, float3* d_vertices,
                                    float weight, int max_vertices_num) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;
  if (ix >= vn.x - 1 || iy >= vn.y - 1 || iz >= vn.z - 1) return;

  int cubeIndex =
      calcCubeIndex(voxels, ix, iy, iz, vn, bb_min, resolution, iso_level);
  if (cubeIndex < 0) return;
  int edges = d_edgeTable[cubeIndex];
  if (edges == 0) return;

  // ベースとなる flat インデックス
  int baseFlat = iz * vn.y * vn.x + iy * vn.x + ix;

  for (int e = 0; e < 12; e++) {
    if (!(edges & (1 << e))) continue;

    // 1) エッジキーを計算して頂点IDテーブルを予約
    int key = computeEdgeKey(ix, iy, iz, e, vn.x, vn.y, vn.z);
    int old = atomicCAS(&d_edgeVertexIds[key], -1, 0);
    if (old != -1) continue;  // 既に誰かが生成済み

    // 2) 新しい頂点ID を確保
    int vid = atomicAdd(d_vtxCounter, 1);

    if (max_vertices_num < vid) {
      return;
    }

    atomicExch(&d_edgeVertexIds[key], vid);

    // 3) エッジに対応する 2 つのコーナー番号
    int c0_id = edgeCorners[e][0];
    int c1_id = edgeCorners[e][1];

    // 4) それぞれのコーナーのグリッド座標 (gx,gy,gz) を計算
    int gx0 = ix + cornerOffset[c0_id][0];
    int gy0 = iy + cornerOffset[c0_id][1];
    int gz0 = iz + cornerOffset[c0_id][2];
    int gx1 = ix + cornerOffset[c1_id][0];
    int gy1 = iy + cornerOffset[c1_id][1];
    int gz1 = iz + cornerOffset[c1_id][2];

    // 5) flat index に戻す（必要なら使わずに直接 sdf 参照も可）
    int flat0 = gz0 * vn.y * vn.x + gy0 * vn.x + gx0;
    int flat1 = gz1 * vn.y * vn.x + gy1 * vn.x + gx1;

    // 6) 実ワールド座標を計算
    float3 p1 = voxel_idx2pos(make_int3(gx0, gy0, gz0), bb_min, resolution);
    float3 p2 = voxel_idx2pos(make_int3(gx1, gy1, gz1), bb_min, resolution);

    // 7) SDF 値を取得
    float v1 = voxels[flat0].sdf_sum / float(voxels[flat0].update_num * weight);
    float v2 = voxels[flat1].sdf_sum / float(voxels[flat1].update_num * weight);

    // 8) 線形補間で頂点位置を求めて書き込み
    d_vertices[vid] = VertexInterp(p1, p2, v1, v2, iso_level);
  }
}

// --- Kernel B: build faces ---
__global__ void BuildFacesKernel(const VoxelCudaNaive* voxels, float3 bb_min,
                                 float3 resolution, int3 vn, float iso_level,
                                 int* d_edgeVertexIds, int* d_idxCounter,
                                 int* d_faces, int max_faces) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;
  if (ix >= vn.x - 1 || iy >= vn.y - 1 || iz >= vn.z - 1) return;
  int cubeIndex =
      calcCubeIndex(voxels, ix, iy, iz, vn, bb_min, resolution, iso_level);
  if (cubeIndex < 0) return;
  int* tri = (int*)(&d_triTable[cubeIndex][0]);
  for (int i = 0; tri[i] != -1; i += 3) {
    int e0 = tri[i], e1 = tri[i + 1], e2 = tri[i + 2];
    int k0 = computeEdgeKey(ix, iy, iz, e0, vn.x, vn.y, vn.z);
    int k1 = computeEdgeKey(ix, iy, iz, e1, vn.x, vn.y, vn.z);
    int k2 = computeEdgeKey(ix, iy, iz, e2, vn.x, vn.y, vn.z);
    int v0 = d_edgeVertexIds[k0];
    int v1 = d_edgeVertexIds[k1];
    int v2 = d_edgeVertexIds[k2];
    int idx = atomicAdd(d_idxCounter, 3);
    if (max_faces < idx) {
      return;
    }
    d_faces[idx + 0] = v2;
    d_faces[idx + 1] = v1;
    d_faces[idx + 2] = v0;
  }
}
__global__ void BuildFacesKernelWithNormal(
    const VoxelCudaNaive* voxels, float3 bb_min, float3 resolution, int3 vn,
    float iso_level, int* d_edgeVertexIds, int* d_idxCounter, int* d_faces,
    float3* d_vertices, float3* d_face_normals, int max_faces) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;
  if (ix >= vn.x - 1 || iy >= vn.y - 1 || iz >= vn.z - 1) return;
  int cubeIndex =
      calcCubeIndex(voxels, ix, iy, iz, vn, bb_min, resolution, iso_level);
  if (cubeIndex < 0) return;
  int* tri = (int*)(&d_triTable[cubeIndex][0]);
  for (int i = 0; tri[i] != -1; i += 3) {
    int e0 = tri[i], e1 = tri[i + 1], e2 = tri[i + 2];
    int k0 = computeEdgeKey(ix, iy, iz, e0, vn.x, vn.y, vn.z);
    int k1 = computeEdgeKey(ix, iy, iz, e1, vn.x, vn.y, vn.z);
    int k2 = computeEdgeKey(ix, iy, iz, e2, vn.x, vn.y, vn.z);
    int v0 = d_edgeVertexIds[k0];
    int v1 = d_edgeVertexIds[k1];
    int v2 = d_edgeVertexIds[k2];
    int idx = atomicAdd(d_idxCounter, 3);
    if (max_faces < idx) {
      return;
    }
    d_faces[idx + 0] = v2;
    d_faces[idx + 1] = v1;
    d_faces[idx + 2] = v0;

    float3 face_normal =
        cross(d_vertices[v2] - d_vertices[v0], d_vertices[v1] - d_vertices[v0]);
    d_face_normals[idx / 3] = normalize(face_normal);
  }
}

__global__ void ComputeVertexNormalsKernel(
    const float3* __restrict__ face_normals,  // [numFaces]
    const int* __restrict__ faces,            // [numFaces*3]
    float3* vertex_normals,                   // [numVertices]
    int numFaces) {
  int fid = blockIdx.x * blockDim.x + threadIdx.x;
  if (fid >= numFaces) {
    return;
  }

  // 3 vertices of the face
  int3 v =
      make_int3(faces[fid * 3 + 0], faces[fid * 3 + 1], faces[fid * 3 + 2]);
  float3 fn = face_normals[fid];

  // Add to each vertex normal
  atomicAdd(&vertex_normals[v.x].x, fn.x);
  atomicAdd(&vertex_normals[v.x].y, fn.y);
  atomicAdd(&vertex_normals[v.x].z, fn.z);

  atomicAdd(&vertex_normals[v.y].x, fn.x);
  atomicAdd(&vertex_normals[v.y].y, fn.y);
  atomicAdd(&vertex_normals[v.y].z, fn.z);

  atomicAdd(&vertex_normals[v.z].x, fn.x);
  atomicAdd(&vertex_normals[v.z].y, fn.y);
  atomicAdd(&vertex_normals[v.z].z, fn.z);
}

__global__ void NormalizeVertexNormalsKernel(
    float3* vertex_normals,  // [numVertices]
    int numVertices) {
  int vid = blockIdx.x * blockDim.x + threadIdx.x;
  if (vid >= numVertices) {
    return;
  }

  float3 n = vertex_normals[vid];
  float len = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
  if (len > 1e-6f) {
    n.x /= len;
    n.y /= len;
    n.z /= len;
  }
  vertex_normals[vid] = n;
}

__global__ void SmoothFaceNormalsKernel(
    const int* __restrict__ faces,              // [numFaces*3]
    const float3* __restrict__ vertex_normals,  // [numVertices]
    float3* smooth_face_normals,                // [numFaces]  出力
    int numFaces) {
  int fid = blockIdx.x * blockDim.x + threadIdx.x;
  if (fid >= numFaces) {
    return;
  }

  int3 v =
      make_int3(faces[fid * 3 + 0], faces[fid * 3 + 1], faces[fid * 3 + 2]);

  // Average of 3 vertex normals
  float3 n0 = vertex_normals[v.x];
  float3 n1 = vertex_normals[v.y];
  float3 n2 = vertex_normals[v.z];
  float3 avg =
      make_float3(n0.x + n1.x + n2.x, n0.y + n1.y + n2.y, n0.z + n1.z + n2.z);

  // Normalize
  float len = sqrtf(avg.x * avg.x + avg.y * avg.y + avg.z * avg.z);
  if (len > 1e-6f) {
    avg.x /= len;
    avg.y /= len;
    avg.z /= len;
  }

  smooth_face_normals[fid] = avg;
}

}  // namespace

namespace ugu {

MeshHostDevice::MeshHostDevice()
    : vertices(nullptr), vertex_count(0), indices(nullptr), index_count(0) {}

MeshHostDevice::~MeshHostDevice() {
  if (vertices) {
    cudaFreeHost(vertices);
  }
  if (indices) {
    cudaFreeHost(indices);
  }
};

void MeshHostDevice::Reseave(int max_vertex_count_, int max_index_count_) {
  if (vertices) {
    cudaFreeHost(vertices);
    vertices = nullptr;
  }
  if (indices) {
    cudaFreeHost(indices);
    indices = nullptr;
  }
  max_vertex_count = max_vertex_count_;
  max_index_count = max_index_count_;
  cudaMallocHost(&vertices, sizeof(VertexHostDevice) * max_vertex_count);
  if (0 < max_index_count) {
    cudaMallocHost(&indices, sizeof(int) * max_index_count);
  }
}

class VoxelGridCudaHashing::Impl {
 public:
  Impl(){};

  // コンストラクタ：ハッシュテーブルサイズ、VoxelBlock配列の最大個数、トランケーション幅mu、1ボクセルの大きさを指定
  Impl(int hashTableSize, int voxelBlockCount, float mu, float voxelSize)
      : m_hashTableSize(hashTableSize),
        m_voxelBlockCount(voxelBlockCount),
        m_mu(mu),
        m_voxelSize(voxelSize),
        d_hashTable(nullptr),
        d_voxelBlocks(nullptr),
        d_globalVoxelBlockCounter(nullptr) {
    // GPU上に各データ構造を確保
    cudaMalloc(&d_hashTable, m_hashTableSize * sizeof(HashEntry));
    cudaMalloc(&d_voxelBlocks, m_voxelBlockCount * sizeof(VoxelBlock));
    cudaMalloc(&d_globalVoxelBlockCounter, sizeof(int));

    // ハッシュテーブル初期化（ptrを -1 に設定）
    {
      int threads = 256;
      int blocks = (m_hashTableSize + threads - 1) / threads;
      initHashTable<<<blocks, threads>>>(d_hashTable, m_hashTableSize);
    }
    // グローバルカウンター初期化
    cudaMemset(d_globalVoxelBlockCounter, 0, sizeof(int));

    // ※ 必要に応じてVoxelBlockの初期化カーネルを追加してください
    {
      int threads = 256;
      int blocks = (m_voxelBlockCount + threads - 1) / threads;
      initVoxelBlocks<<<blocks, threads>>>(d_voxelBlocks, m_voxelBlockCount);
    }

    cudaMalloc(&d_vertexCount, sizeof(int));
  }

  ~Impl() {
    if (d_hashTable) cudaFree(d_hashTable);
    if (d_voxelBlocks) cudaFree(d_voxelBlocks);
    if (d_globalVoxelBlockCounter) cudaFree(d_globalVoxelBlockCounter);

    if (d_vertices) cudaFree(d_vertices);
    if (d_indices) cudaFree(d_indices);
    cudaFree(d_vertexCount);
  }

  // void Init(int hash_table_size, int voxel_block_count, float mu,
  //   float voxel_size) {

  //}

  void FusePointCloud(const float3* d_points, const float3* d_normals,
                      int num_points, bool sync) {
    int threads = 256;
    int blocks = (num_points + threads - 1) / threads;
    fusePointCloudKernel<<<blocks, threads>>>(
        d_points, d_normals, num_points, d_hashTable, m_hashTableSize,
        d_voxelBlocks, d_globalVoxelBlockCounter, m_mu, m_voxelSize);
    if (sync) {
      cudaDeviceSynchronize();
    }
  }

  void FuseOrganizedPointCloudMulti(const float3* d_points,
                                    const float3* d_normals, int width,
                                    int height, int num_images, bool sync) {
    int totalPixels = width * height * num_images;
    int threads = 256;
    int blocks = (totalPixels + threads - 1) / threads;

    fuseOrganizedPointCloudMultiKernelHashing<<<blocks, threads>>>(
        d_points, d_normals, width, height, num_images, d_hashTable,
        m_hashTableSize, d_voxelBlocks, d_globalVoxelBlockCounter, m_mu,
        m_voxelSize);
    checkCudaErrors(cudaGetLastError());
    if (sync) {
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }

  // generateMesh(): Marching Cubesによりメッシュ生成を行う関数
  // connected が true
  // の場合、インデックスバッファを生成（各三角形は独立頂点ですが、インデックスで接続した状態とする）
  void GenerateMesh(MeshHostDevice& mesh, bool connected) {
    // デバイス上に割り当てられたVoxelBlock数（有効なブロック数）を取得
    int validBlockCount;
    cudaMemcpy(&validBlockCount, d_globalVoxelBlockCounter, sizeof(int),
               cudaMemcpyDeviceToHost);

    int cellsPerBlock = (BLOCK_SIZE - 1) * (BLOCK_SIZE - 1) * (BLOCK_SIZE - 1);
    int totalCells = validBlockCount * cellsPerBlock;
    // 各セルから最大5個の三角形が生成されると仮定（worst-case）
    int current_max_triangles = validBlockCount * cellsPerBlock * 5;
    int current_max_vertices = current_max_triangles * 3;

    {
      if (max_vertices < current_max_vertices) {
        if (d_vertices) {
          cudaFree(d_vertices);
          d_vertices = nullptr;
        }
        // if (connected) {
        if (d_indices) {
          cudaFree(d_indices);
          d_indices = nullptr;
        }
        //}
        max_vertices = current_max_vertices;

        cudaMalloc(&d_vertices, max_vertices * sizeof(VertexHostDevice));
        // if (connected) {
        cudaMalloc(&d_indices, max_vertices * sizeof(int));
        //}
      }
    }

    cudaMemset(d_vertexCount, 0, sizeof(int));

    int threads = 256;
    int blocks = (totalCells + threads - 1) / threads;
    marchingCubesKernel<<<blocks, threads>>>(d_voxelBlocks, validBlockCount,
                                             m_voxelSize, d_vertices, d_indices,
                                             d_vertexCount, connected);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // 出力頂点数を取得
    int h_vertexCount;
    cudaMemcpy(&h_vertexCount, d_vertexCount, sizeof(int),
               cudaMemcpyDeviceToHost);

    if (mesh.max_vertex_count < h_vertexCount ||
        (connected && mesh.max_index_count < h_vertexCount)) {
      mesh.Reseave(h_vertexCount, connected ? h_vertexCount : 0);
    }

    mesh.vertex_count = h_vertexCount;
    mesh.index_count = h_vertexCount;

    cudaMemcpy(mesh.vertices, d_vertices,
               h_vertexCount * sizeof(MeshHostDevice), cudaMemcpyDeviceToHost);
    if (connected) {
      cudaMemcpy(mesh.indices, d_indices, h_vertexCount * sizeof(int),
                 cudaMemcpyDeviceToHost);
    }
  }

  // ※
  // ここに、メッシュのリセットやホスト側への転送処理等、必要な関数を追加してください

 private:
  int m_hashTableSize;
  int m_voxelBlockCount;
  float m_mu;
  float m_voxelSize;

  HashEntry* d_hashTable{nullptr};
  VoxelBlock* d_voxelBlocks{nullptr};
  int* d_globalVoxelBlockCounter{nullptr};

  VertexHostDevice* d_vertices{nullptr};
  int* d_indices{nullptr};
  int* d_vertexCount{nullptr};
  // int max_triangles{0};
  int max_vertices{0};
};

VoxelGridCudaHashing::VoxelGridCudaHashing() {}

VoxelGridCudaHashing::VoxelGridCudaHashing(int hash_table_size,
                                           int voxel_block_count, float mu,
                                           float voxel_size) {
  Init(hash_table_size, voxel_block_count, mu, voxel_size);
}

VoxelGridCudaHashing::~VoxelGridCudaHashing() {}

void VoxelGridCudaHashing::Init(int hash_table_size, int voxel_block_count,
                                float mu, float voxel_size) {
  impl_ = std::make_unique<Impl>(hash_table_size, voxel_block_count, mu,
                                 voxel_size);
}

void VoxelGridCudaHashing::FusePointCloud(const float* d_points,
                                          const float* d_normals,
                                          uint32_t num_points, bool sync) {
  impl_->FusePointCloud(reinterpret_cast<const float3*>(d_points),
                        reinterpret_cast<const float3*>(d_normals), num_points,
                        sync);
}

void VoxelGridCudaHashing::FuseOrganizedPointCloudMulti(const float* d_points,
                                                        const float* d_normals,
                                                        int width, int height,
                                                        int num_images,
                                                        bool sync) {
  impl_->FuseOrganizedPointCloudMulti(
      reinterpret_cast<const float3*>(d_points),
      reinterpret_cast<const float3*>(d_normals), width, height, num_images,
      sync);
}

void VoxelGridCudaHashing::GenerateMesh(MeshHostDevice& mesh, bool connected) {
  impl_->GenerateMesh(mesh, connected);
}

class VoxelGridCudaNaive::Impl {
 public:
  Impl() {}
  ~Impl() { Free(); }
  bool Init(const Eigen::Vector3f& bb_max, const Eigen::Vector3f& bb_min,
            float resolution) {
    return Init(bb_max, bb_min, Eigen::Vector3f::Constant(resolution));
  }

  bool Init(const Eigen::Vector3f& bb_max, const Eigen::Vector3f& bb_min,
            const Eigen::Vector3f& resolution) {
    bb_max_.x = bb_max[0];
    bb_max_.y = bb_max[1];
    bb_max_.z = bb_max[2];

    bb_min_.x = bb_min[0];
    bb_min_.y = bb_min[1];
    bb_min_.z = bb_min[2];

    resolution_.x = resolution[0];
    resolution_.y = resolution[1];
    resolution_.z = resolution[2];

    inv_resolution_.x = 1.0f / resolution_.x;
    inv_resolution_.y = 1.0f / resolution_.y;
    inv_resolution_.z = 1.0f / resolution_.z;

    voxel_num_.x = static_cast<int>((bb_max_.x - bb_min_.x) / resolution_.x);
    voxel_num_.y = static_cast<int>((bb_max_.y - bb_min_.y) / resolution_.y);
    voxel_num_.z = static_cast<int>((bb_max_.z - bb_min_.z) / resolution_.z);

    Free();

    int total_voxel_num = voxel_num_.x * voxel_num_.y * voxel_num_.z;
    cudaMalloc(&d_voxels_, total_voxel_num * sizeof(VoxelCudaNaive));

    // Zero fill
    cudaMemset(d_voxels_, 0, sizeof(VoxelCudaNaive) * total_voxel_num);

    int totalCells =
        (voxel_num_.x - 1) * (voxel_num_.y - 1) * (voxel_num_.z - 1);
    int maxTris_ = totalCells * 5 * 3;  // Theoretical max;

    // Practical max
    max_tris_ = maxTris_ / max(max(voxel_num_.x, voxel_num_.y), voxel_num_.z);

    cudaMalloc(&d_vertices, sizeof(float3) * max_tris_);
    cudaMemset(d_vertices, 0, sizeof(float3) * max_tris_);

    cudaMalloc(&d_vertex_normals, sizeof(float3) * max_tris_);

    cudaMalloc(&d_vtxCounter, sizeof(int));
    cudaMemset(d_vtxCounter, 0, sizeof(int));

    // グリッドのサイズ
    int nx = voxel_num_.x;
    int ny = voxel_num_.y;
    int nz = voxel_num_.z;

    // 各方向のエッジ数
    int xCount = (nx - 1) * ny * nz;  // X 方向エッジ
    int yCount = nx * (ny - 1) * nz;  // Y 方向エッジ
    int zCount = nx * ny * (nz - 1);  // Z 方向エッジ

    // 全エッジ数
    numEdges = xCount + yCount + zCount;

    cudaMalloc(&d_edgeVertexIds, sizeof(int) * numEdges);
    cudaMemset(d_edgeVertexIds, -1, sizeof(int) * numEdges);

    // Face buffer and counter
    int maxF_ = totalCells * 15;

    // Practical max
    max_faces_ = maxF_ / max(max(voxel_num_.x, voxel_num_.y), voxel_num_.z);

    cudaMalloc(&d_faces, sizeof(int) * max_faces_);
    cudaMalloc(&d_idxCounter, sizeof(int));
    cudaMemset(d_idxCounter, 0, sizeof(int));

    cudaMalloc(&d_face_normals, sizeof(float3) * max_faces_ / 3);
    cudaMalloc(&d_face_smooth_normals, sizeof(float3) * max_faces_ / 3);

    cudaMallocHost(&h_vertices_pinned, sizeof(float3) * max_tris_);
    cudaMallocHost(&h_faces_pinned, sizeof(int) * max_faces_);
    cudaMallocHost(&h_face_normals_pinned, sizeof(float3) * max_faces_ / 3);

    // Constat
    cudaMemcpyToSymbol(c_bb_min, &bb_min_, sizeof(float3));
    cudaMemcpyToSymbol(c_voxel_size, &resolution_, sizeof(float3));
    cudaMemcpyToSymbol(c_inv_voxel_size, &inv_resolution_, sizeof(float3));
    cudaMemcpyToSymbol(c_voxel_num, &voxel_num_, sizeof(int3));
    cudaMemcpyToSymbol(c_trunc, &option_.truncation_band, sizeof(float));

    return true;
  }

  void FuseOrganizedPointCloudMulti(const float3* d_points,
                                    const float3* d_normals, int width,
                                    int height, int num_images,
                                    const VoxelGridCudaNaiveFuseOption& option,
                                    bool sync = true) {
    option_ = option;
    int totalPixels = width * height * num_images;
    int threads = 256;
    int blocks = (totalPixels + threads - 1) / threads;
#if 0
    if (fabsf(option.weight - 1.f) < 0.01f && option.sample_num == 2 &&
        option.nn_range == 1 && option.r < 0.f && option.height_half < 0.f) {
      // Need to send here
      cudaMemcpyToSymbol(c_trunc, &option_.truncation_band, sizeof(float));
      FuseOrganizedPointCloudMultiKernelNaiveOptimized<<<blocks, threads>>>(
          d_points, d_normals, d_voxels_, totalPixels);
    } else {
      FuseOrganizedPointCloudMultiKernelNaive<<<blocks, threads>>>(
          d_points, d_normals, width, height, num_images, d_voxels_,
          resolution_, bb_max_, bb_min_, voxel_num_, option.truncation_band,
          option.weight, option.sample_num, option.nn_range, option.r,
          option.height_half);
    }
#else
    FuseOrganizedPointCloudMultiKernelNaive<<<blocks, threads>>>(
        d_points, d_normals, width, height, num_images, d_voxels_, resolution_,
        bb_max_, bb_min_, voxel_num_, option.truncation_band, option.weight,
        option.sample_num, option.nn_range, option.r, option.height_half);
#endif
    checkCudaErrors(cudaGetLastError());
    if (sync) {
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }

  void FuseDepthMulti(const float* h_depth, int width, int height,
                      int num_images, const float* h_fx, const float* h_fy,
                      const float* h_cx, const float* h_cy, const float* h_R,
                      const float* h_t,
                      const VoxelGridCudaNaiveFuseOption& option,
                      bool sync = true) {
    if (MAX_IMAGES < num_images) {
      std::cerr << "Error: num_images (" << num_images
                << ") exceeds MAX_IMAGES (" << MAX_IMAGES << ")" << std::endl;
      return;
    }

    option_ = option;

    // Send camera parameters to GPU constant
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

    // TODO: Size/Num reset API
    if (d_depth == nullptr) {
      cudaMalloc(&d_depth, sizeof(float) * width * height * num_images);
    }
    cudaMemcpy(d_depth, h_depth, sizeof(float) * width * height * num_images,
               cudaMemcpyHostToDevice);

    dim3 block(16, 16, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y,
              num_images);
    FuseDepthMultiKernelNaive<<<grid, block>>>(
        d_depth, width, height, num_images, d_voxels_, resolution_, bb_max_,
        bb_min_, voxel_num_, option.truncation_band, option.weight,
        option.sample_num, option.nn_range);

    checkCudaErrors(cudaGetLastError());
    if (sync) {
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }

  void ExtractMesh(bool with_face_normals) {
    cudaMemset(d_vtxCounter, 0, sizeof(int));
    cudaMemset(d_edgeVertexIds, -1, sizeof(int) * numEdges);
    cudaMemset(d_idxCounter, 0, sizeof(int));

    dim3 block(8, 8, 8);
    dim3 grid((voxel_num_.x + block.x - 1) / block.x,
              (voxel_num_.y + block.y - 1) / block.y,
              (voxel_num_.z + block.z - 1) / block.z);

    // Launch kernels
    constexpr float iso_level = 0.f;
    constexpr int tri_ratio = 2;
    while (true) {
      BuildVerticesKernel<<<grid, block>>>(
          d_voxels_, bb_min_, resolution_, voxel_num_, iso_level,
          d_edgeVertexIds, d_vtxCounter, d_vertices, option_.weight, max_tris_);
      checkCudaErrors(cudaGetLastError());
      checkCudaErrors(cudaDeviceSynchronize());
      int h_vcount = 0;
      cudaMemcpy(&h_vcount, d_vtxCounter, sizeof(int), cudaMemcpyDeviceToHost);
      if (h_vcount < max_tris_) {
        num_vertices_ = h_vcount;
        break;
      }
      // If memory is not enough, reallocate
      cudaMemset(d_vtxCounter, 0, sizeof(int));
      cudaMemset(d_edgeVertexIds, -1, sizeof(int) * numEdges);
      EnsureTriangleVertexMemory(h_vcount * tri_ratio);
    }

    while (true) {
      if (with_face_normals) {
        BuildFacesKernelWithNormal<<<grid, block>>>(
            d_voxels_, bb_min_, resolution_, voxel_num_, iso_level,
            d_edgeVertexIds, d_idxCounter, d_faces, d_vertices, d_face_normals,
            max_faces_);
      } else {
        BuildFacesKernel<<<grid, block>>>(
            d_voxels_, bb_min_, resolution_, voxel_num_, iso_level,
            d_edgeVertexIds, d_idxCounter, d_faces, max_faces_);
      }
      checkCudaErrors(cudaGetLastError());
      checkCudaErrors(cudaDeviceSynchronize());

      int h_icount = 0;
      cudaMemcpy(&h_icount, d_idxCounter, sizeof(int), cudaMemcpyDeviceToHost);
      if (h_icount < max_faces_) {
        num_faces_ = h_icount / 3;
        break;
      }
      // if memory is not enough, reallocate
      cudaMemset(d_idxCounter, 0, sizeof(int));
      EnsureTriangleMemory(h_icount * tri_ratio);
    }
  }

  void ComputeVertexNormals() {
    // Zero clear for summation

    cudaMemset(d_vertex_normals, 0, sizeof(float3) * num_vertices_);

    const int THREADS = 256;

    int blocksF = (num_faces_ + THREADS - 1) / THREADS;
    int blocksV = (num_vertices_ + THREADS - 1) / THREADS;
    ComputeVertexNormalsKernel<<<blocksF, THREADS>>>(
        d_face_normals, d_faces, d_vertex_normals, num_faces_);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    NormalizeVertexNormalsKernel<<<blocksV, THREADS>>>(d_vertex_normals,
                                                       num_vertices_);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
  }

  void SmoothFaceNormalsWithVertexNormals() {
    const int THREADS = 256;
    int blocksF = (num_faces_ + THREADS - 1) / THREADS;

    SmoothFaceNormalsKernel<<<blocksF, THREADS>>>(
        d_faces, d_vertex_normals, d_face_smooth_normals, num_faces_);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
  }

  void GetVerticesCpu(std::vector<Eigen::Vector3f>& vertices) {
    // Copy data back
    cudaMemcpy(h_vertices_pinned, d_vertices, sizeof(float3) * num_vertices_,
               cudaMemcpyDeviceToHost);

    vertices.resize(num_vertices_);
    for (int i = 0; i < num_vertices_; ++i) {
      vertices[i] =
          Eigen::Vector3f(h_vertices_pinned[i].x, h_vertices_pinned[i].y,
                          h_vertices_pinned[i].z);
    }
  }

  void GetVertexNormalsCpu(std::vector<Eigen::Vector3f>& vertex_normals) {
    // Use the same host buffer, h_vertices_pinned, as vertices becasue byte
    // size is identical

    // Copy data back
    cudaMemcpy(h_vertices_pinned, d_vertex_normals,
               sizeof(float3) * num_vertices_, cudaMemcpyDeviceToHost);

    vertex_normals.resize(num_vertices_);
    for (int i = 0; i < num_vertices_; ++i) {
      vertex_normals[i] =
          Eigen::Vector3f(h_vertices_pinned[i].x, h_vertices_pinned[i].y,
                          h_vertices_pinned[i].z);
    }
  }

  void GetFacesCpu(std::vector<Eigen::Vector3i>& faces) {
    faces.resize(num_faces_);
    cudaMemcpy(h_faces_pinned, d_faces, sizeof(int) * num_faces_ * 3,
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < num_faces_; i++) {
      faces[i] =
          Eigen::Vector3i(h_faces_pinned[3 * i + 0], h_faces_pinned[3 * i + 1],
                          h_faces_pinned[3 * i + 2]);
    }
  }

  void GetFaceNormalsCpu(std::vector<Eigen::Vector3f>& face_normals,
                         bool smoothing = false) {
    face_normals.resize(num_faces_);
    float3* d_face_normals_source =
        smoothing ? d_face_smooth_normals : d_face_normals;
    cudaMemcpy(h_face_normals_pinned, d_face_normals_source,
               sizeof(float3) * num_faces_, cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_faces_; ++i) {
      face_normals[i] = Eigen::Vector3f(h_face_normals_pinned[i].x,
                                        h_face_normals_pinned[i].y,
                                        h_face_normals_pinned[i].z);
    }
  }

  void GetVoxelGridCpu(ugu::VoxelGrid& grid_cpu) const {
    // NOTE:
    // CPU version has resultion * 0.5 offset for both voxel definition and
    // marchingcubes. So the extracted mesh by CPU shows some shift from GPU
    // one.

    std::vector<VoxelCudaNaive> voxels_cpu(voxel_num_.x * voxel_num_.y *
                                           voxel_num_.z);
    cudaMemcpy(
        voxels_cpu.data(), d_voxels_,
        sizeof(VoxelCudaNaive) * voxel_num_.x * voxel_num_.y * voxel_num_.z,
        cudaMemcpyDeviceToHost);

    auto& voxels = grid_cpu.get_all();
#if 0
    for (int i = 0; i < voxel_num_.x * voxel_num_.y * voxel_num_.z; ++i) {
      voxels[i].update_num = voxels_cpu[i].update_num;

      if (voxels[i].update_num < 1) {
        voxels[i].sdf = ugu::InvalidSdf::kVal;
      } else {
        voxels[i].sdf =
            voxels_cpu[i].sdf_sum / (voxels_cpu[i].update_num * option_.weight);
      }
    }
#endif
    for (int z = 1; z < voxel_num_.z - 1; ++z) {
      for (int y = 1; y < voxel_num_.y - 1; ++y) {
        for (int x = 1; x < voxel_num_.x - 1; ++x) {
          int i = z * voxel_num_.y * voxel_num_.x + y * voxel_num_.x + x;
          int offset = 0;
          int j = (z + offset) * voxel_num_.y * voxel_num_.x +
                  (y + offset) * voxel_num_.x + (x + offset);
          voxels[j].update_num = voxels_cpu[i].update_num;

          // voxels[j].pos.x() += resolution_.x * 0.5f;
          // voxels[j].pos.y() += resolution_.y * 0.5f;
          // voxels[j].pos.z() += resolution_.z * 0.5f;

          if (voxels[j].update_num < 1) {
            voxels[j].sdf = ugu::InvalidSdf::kVal;
          } else {
            voxels[j].sdf = voxels_cpu[i].sdf_sum /
                            (voxels_cpu[i].update_num * option_.weight);
          }
        }
      }
    }
  }

  const float3* GetVerticesGpu() const { return d_vertices; }

  const float3* GetVertexNormalsGpu() const { return d_vertex_normals; }

  const int* GetFacesGpu() const { return d_faces; }

  const float3* GetFaceNormalsGpu() const { return d_face_normals; }

  const float3* GetSmoothFaceNormalsGpu() const {
    return d_face_smooth_normals;
  }

  const int* GetVerticesNumGpu() const { return d_vtxCounter; }

  const int* GetFacesNumGpu() const { return d_idxCounter; }

  void Clear() {
    int total_voxel_num = voxel_num_.x * voxel_num_.y * voxel_num_.z;
    cudaMemset(d_voxels_, 0, sizeof(VoxelCudaNaive) * total_voxel_num);
  }

 private:
  void Free() {
    if (d_voxels_ != nullptr) {
      cudaFree(d_voxels_);
      d_voxels_ = nullptr;
    }
    if (d_vertices != nullptr) {
      cudaFree(d_vertices);
      d_vertices = nullptr;
    }

    if (d_vertex_normals != nullptr) {
      cudaFree(d_vertex_normals);
      d_vertex_normals = nullptr;
    }

    if (d_vtxCounter != nullptr) {
      cudaFree(d_vtxCounter);
      d_vtxCounter = nullptr;
    }

    if (d_faces != nullptr) {
      cudaFree(d_faces);
      d_faces = nullptr;
    }

    if (d_idxCounter != nullptr) {
      cudaFree(d_idxCounter);
      d_idxCounter = nullptr;
    }

    if (d_face_normals != nullptr) {
      cudaFree(d_face_normals);
      d_face_normals = nullptr;
    }

    if (d_face_smooth_normals != nullptr) {
      cudaFree(d_face_smooth_normals);
      d_face_smooth_normals = nullptr;
    }

    if (d_depth != nullptr) {
      cudaFree(d_depth);
      d_depth = nullptr;
    }

    if (d_edgeVertexIds != nullptr) {
      cudaFree(d_edgeVertexIds);
      d_edgeVertexIds = nullptr;
    }

    if (h_vertices_pinned != nullptr) {
      cudaFreeHost(h_vertices_pinned);
      h_vertices_pinned = nullptr;
    }

    if (h_faces_pinned != nullptr) {
      cudaFreeHost(h_faces_pinned);
      h_faces_pinned = nullptr;
    }

    if (h_face_normals_pinned) {
      cudaFreeHost(h_face_normals_pinned);
      h_face_normals_pinned = nullptr;
    }

    max_faces_ = 0;
    max_tris_ = 0;
  }

  void EnsureTriangleVertexMemory(int tris_num) {
    if (tris_num <= max_tris_) {
      return;
    }

    max_tris_ = tris_num;

    cudaFree(d_vertices);
    cudaFree(d_vertex_normals);
    cudaFreeHost(h_vertices_pinned);

    cudaMalloc(&d_vertices, sizeof(float3) * max_tris_);
    cudaMalloc(&d_vertex_normals, sizeof(float3) * max_tris_);
    cudaMallocHost(&h_vertices_pinned, sizeof(float3) * max_tris_);
  }

  void EnsureTriangleMemory(int faces_num) {
    if (faces_num <= max_faces_) {
      return;
    }

    max_faces_ = faces_num;

    cudaFree(d_faces);
    cudaFree(d_face_normals);
    cudaFree(d_face_smooth_normals);

    cudaFreeHost(h_faces_pinned);
    cudaFreeHost(h_face_normals_pinned);

    cudaMalloc(&d_faces, sizeof(int) * max_faces_);
    cudaMalloc(&d_face_normals, sizeof(float3) * max_faces_ / 3);
    cudaMalloc(&d_face_smooth_normals, sizeof(float3) * max_faces_ / 3);

    cudaMallocHost(&h_faces_pinned, sizeof(int) * max_faces_);
    cudaMallocHost(&h_face_normals_pinned, sizeof(float3) * max_faces_ / 3);
  }

  VoxelCudaNaive* d_voxels_{nullptr};
  float3* d_vertices{nullptr};
  int* d_vtxCounter{nullptr};
  int* d_faces{nullptr};
  int* d_idxCounter{nullptr};
  int* d_edgeVertexIds{nullptr};
  float3* d_face_normals{nullptr};
  float3* d_vertex_normals{nullptr};
  float3* d_face_smooth_normals{nullptr};
  float* d_depth{nullptr};

  float3* h_vertices_pinned{nullptr};
  int* h_faces_pinned{nullptr};
  float3* h_face_normals_pinned{nullptr};

  int numEdges;
  float3 bb_max_;
  float3 bb_min_;
  float3 resolution_;
  float3 inv_resolution_;
  int3 voxel_num_{0, 0, 0};
  VoxelGridCudaNaiveFuseOption option_;
  int xy_slice_num_{0};

  int max_tris_{0};
  int max_faces_{0};

  int num_faces_{0};
  int num_vertices_{0};
};

VoxelGridCudaNaive::VoxelGridCudaNaive() { impl_ = std::make_unique<Impl>(); }

VoxelGridCudaNaive::~VoxelGridCudaNaive() {}

bool VoxelGridCudaNaive::Init(const Eigen::Vector3f& bb_max,
                              const Eigen::Vector3f& bb_min, float resolution) {
  return impl_->Init(bb_max, bb_min, resolution);
}

bool VoxelGridCudaNaive::Init(const Eigen::Vector3f& bb_max,
                              const Eigen::Vector3f& bb_min,
                              const Eigen::Vector3f& resolution) {
  return impl_->Init(bb_max, bb_min, resolution);
}

void VoxelGridCudaNaive::FusePointCloudMulti(
    const float* d_points, const float* d_normals, int width, int height,
    int num_images, const VoxelGridCudaNaiveFuseOption& option, bool sync) {
  impl_->FuseOrganizedPointCloudMulti(
      reinterpret_cast<const float3*>(d_points),
      reinterpret_cast<const float3*>(d_normals), width, height, num_images,
      option, sync);
}

void VoxelGridCudaNaive::FuseDepthMulti(
    const float* h_depth, int width, int height, int num_images,
    const float* h_fx, const float* h_fy, const float* h_cx, const float* h_cy,
    const float* h_R, const float* h_t,
    const VoxelGridCudaNaiveFuseOption& option, bool sync) {
  impl_->FuseDepthMulti(h_depth, width, height, num_images, h_fx, h_fy, h_cx,
                        h_cy, h_R, h_t, option, sync);
}

void VoxelGridCudaNaive::ExtractMesh(bool with_face_normals) {
  impl_->ExtractMesh(with_face_normals);
}

void VoxelGridCudaNaive::ComputeVertexNormals() {
  impl_->ComputeVertexNormals();
}

void VoxelGridCudaNaive::SmoothFaceNormalsWithVertexNormals() {
  impl_->SmoothFaceNormalsWithVertexNormals();
}

void VoxelGridCudaNaive::GetVerticesCpu(
    std::vector<Eigen::Vector3f>& vertices) {
  impl_->GetVerticesCpu(vertices);
}

void VoxelGridCudaNaive::GetVertexNormalsCpu(
    std::vector<Eigen::Vector3f>& vertex_normals) {
  impl_->GetVertexNormalsCpu(vertex_normals);
}

void VoxelGridCudaNaive::GetFacesCpu(std::vector<Eigen::Vector3i>& faces) {
  impl_->GetFacesCpu(faces);
}

void VoxelGridCudaNaive::GetFaceNormalsCpu(
    std::vector<Eigen::Vector3f>& face_normals, bool smoothing) {
  impl_->GetFaceNormalsCpu(face_normals, smoothing);
}

void VoxelGridCudaNaive::GetVoxelGridCpu(ugu::VoxelGrid& grid_cpu) const {
  impl_->GetVoxelGridCpu(grid_cpu);
}

const float3* VoxelGridCudaNaive::GetVerticesGpu() const {
  return impl_->GetVerticesGpu();
}

const int* VoxelGridCudaNaive::GetFacesGpu() const {
  return impl_->GetFacesGpu();
}

const float3* VoxelGridCudaNaive::GetFaceNormalsGpu() const {
  return impl_->GetFaceNormalsGpu();
}

const float3* VoxelGridCudaNaive::GetVertexNormalsGpu() const {
  return impl_->GetVertexNormalsGpu();
}

const float3* VoxelGridCudaNaive::GetSmoothFaceNormalsGpu() const {
  return impl_->GetSmoothFaceNormalsGpu();
}

const int* VoxelGridCudaNaive::GetVerticesNumGpu() const {
  return impl_->GetVerticesNumGpu();
}

const int* VoxelGridCudaNaive::GetFacesNumGpu() const {
  return impl_->GetFacesNumGpu();
}

void VoxelGridCudaNaive::Clear() { impl_->Clear(); }

}  // namespace ugu
