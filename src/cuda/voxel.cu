#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "./helper_cuda.h"
#include "./voxel.cuh"
#include "ugu/cuda/voxel.h"

namespace {
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
__device__ inline float3 normalize(const float3 a) {
  float len = sqrtf(dot(a, a));
  return (len > 0.0f) ? a / len : make_float3(0.0f, 0.0f, 0.0f);
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

__global__ void fuseOrganizedPointCloudMultiKernel(
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

//
// Marching Cubes によるメッシュ生成カーネル
// 各スレッドは有効なVoxelBlock内の1セル（(BLOCK_SIZE-1)^3個のセル）に対して処理を行う
// isoLevel は 0 を想定（SDF=0 の面）、d_vertices, d_indices
// は出力バッファ（d_indicesは connected==true の場合に書き出す） d_vertexCount
// はグローバル出力頂点数カウンター（各三角形につき3頂点を出力）
//
__global__ void marchingCubesKernel(VoxelBlock* d_voxelBlocks,
                                    int validBlockCount, float voxelSize,
                                    ugu::VertexCuda* d_vertices, int* d_indices,
                                    int* d_vertexCount, bool connected) {
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

}  // namespace

namespace ugu {

HostMeshCuda::HostMeshCuda()
    : vertices(nullptr), vertex_count(0), indices(nullptr), index_count(0) {}

HostMeshCuda::~HostMeshCuda() {
  if (vertices) {
    cudaFreeHost(vertices);
  }
  if (indices) {
    cudaFreeHost(indices);
  }
};

void HostMeshCuda::Reseave(int max_vertex_count_, int max_index_count_) {
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
  cudaMallocHost(&vertices, sizeof(VertexCuda) * max_vertex_count);
  if (0 < max_index_count) {
    cudaMallocHost(&indices, sizeof(int) * max_index_count);
  }
}

class VoxelGridCuda::Impl {
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

  // fusePointCloud(): 点群と法線情報からSDFフュージョンを実行（30FPSを想定）
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
    fuseOrganizedPointCloudMultiKernel<<<blocks, threads>>>(
        d_points, d_normals, width, height, num_images, d_hashTable,
        m_hashTableSize, d_voxelBlocks, d_globalVoxelBlockCounter, m_mu,
        m_voxelSize);
    if (sync) {
      cudaDeviceSynchronize();
    }
  }

  // generateMesh(): Marching Cubesによりメッシュ生成を行う関数
  // connected が true
  // の場合、インデックスバッファを生成（各三角形は独立頂点ですが、インデックスで接続した状態とする）
  void GenerateMesh(HostMeshCuda& mesh, bool connected) {
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

        cudaMalloc(&d_vertices, max_vertices * sizeof(VertexCuda));
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
    cudaDeviceSynchronize();

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

    cudaMemcpy(mesh.vertices, d_vertices, h_vertexCount * sizeof(HostMeshCuda),
               cudaMemcpyDeviceToHost);
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

  VertexCuda* d_vertices{nullptr};
  int* d_indices{nullptr};
  int* d_vertexCount{nullptr};
  // int max_triangles{0};
  int max_vertices{0};
};

VoxelGridCuda::VoxelGridCuda() {}

VoxelGridCuda::VoxelGridCuda(int hash_table_size, int voxel_block_count,
                             float mu, float voxel_size) {
  Init(hash_table_size, voxel_block_count, mu, voxel_size);
}

VoxelGridCuda::~VoxelGridCuda() {}

void VoxelGridCuda::Init(int hash_table_size, int voxel_block_count, float mu,
                         float voxel_size) {
  impl_ = std::make_unique<Impl>(hash_table_size, voxel_block_count, mu,
                                 voxel_size);
}

void VoxelGridCuda::FusePointCloud(const float* d_points,
                                   const float* d_normals, uint32_t num_points,
                                   bool sync) {
  impl_->FusePointCloud(reinterpret_cast<const float3*>(d_points),
                        reinterpret_cast<const float3*>(d_normals), num_points,
                        sync);
}

void VoxelGridCuda::FuseOrganizedPointCloudMulti(const float* d_points,
                                                 const float* d_normals,
                                                 int width, int height,
                                                 int num_images, bool sync) {
  impl_->FuseOrganizedPointCloudMulti(
      reinterpret_cast<const float3*>(d_points),
      reinterpret_cast<const float3*>(d_normals), width, height, num_images,
      sync);
}

}  // namespace ugu
