// This program computes matrix multiplication using shared memory tiling
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

// Matrix dimensions
constexpr int M = (1 << 9) + 7;
constexpr int N = (1 << 10) + 7;
constexpr int K = (1 << 11) + 7;

// Threads per CTA dimension
constexpr int THREADS = 32;

// Padded matrix dimensions
constexpr int M_padded = M + THREADS - M % THREADS;
constexpr int N_padded = N + THREADS - N % THREADS;
constexpr int K_padded = K + THREADS - K % THREADS;

// Size of shared memory per TB
constexpr int SHMEM_SIZE = THREADS * THREADS;

__global__ void matrixMul(const int *a, const int *b, int *c) {
  // Compute each thread's global row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Statically allocated shared memory
  __shared__ int s_a[SHMEM_SIZE];
  __shared__ int s_b[SHMEM_SIZE];

  // Accumulate in temporary variable
  int tmp = 0;

  // Sweep tile across matrix
  for (int i = 0; i < K_padded; i += blockDim.x) {
    // Load in elements for this tile
    s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * K + i + threadIdx.x];
    s_b[threadIdx.y * blockDim.x + threadIdx.x] =
        b[i * N + threadIdx.y * N + col];

    // Wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // Do matrix multiplication on the small matrix
    for (int j = 0; j < blockDim.x; j++) {
      tmp +=
          s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
    }

    // Wait for all threads to finish using current tiles before loading in new
    // ones
    __syncthreads();
  }

  // Write back results
  if (row < M && col < N) c[row * N + col] = tmp;
}

// Check result on the CPU
// MxN = MxK * KxN
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c) {
  // For every row...
  for (int row = 0; row < M_padded; row++) {
    if (row >= M) continue;
    // For every column...
    for (int col = 0; col < N_padded; col++) {
      if (col >= N) continue;
      // For every element in the row-column pair
      int tmp = 0;
      for (int i = 0; i < K_padded; i++) {
        // Accumulate the partial results
        tmp += a[row * K + i] * b[i * N + col];
      }

      // Check against the CPU result
      assert(tmp == c[row * N + col]);
    }
  }
}

int main() {
  // Size (in bytes) of matrix
  // MxN = MxK * KxN
  size_t bytes_a = M_padded * K_padded * sizeof(int);
  size_t bytes_b = K_padded * N_padded * sizeof(int);
  size_t bytes_c = M * N * sizeof(int);

  // Host vectors
  vector<int> h_a(M_padded * K_padded);
  vector<int> h_b(K_padded * N_padded);
  vector<int> h_c(M * N);

  // Initialize matrices
  // Padded matrix A
  for (int i = 0; i < M_padded; i++) {
    for (int j = 0; j < K_padded; j++) {
      if (i < M && j < K) h_a[i * K + j] = rand() % 100;
    }
  }
  // Padded matrix B
  for (int i = 0; i < K_padded; i++) {
    for (int j = 0; j < N_padded; j++) {
      if (i < K && j < N) h_b[i * N + j] = rand() % 100;
    }
  }

  // Allocate device memory
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes_a);
  cudaMalloc(&d_b, bytes_b);
  cudaMalloc(&d_c, bytes_c);

  // Copy data to the device
  cudaMemcpy(d_a, h_a.data(), bytes_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes_b, cudaMemcpyHostToDevice);

  // Blocks per grid dimension (assumes THREADS divides M and N evenly)
  int BLOCKS_X = N_padded / THREADS;
  int BLOCKS_Y = M_padded / THREADS;

  // Use dim3 structs for block  and grid dimensions
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS_X, BLOCKS_Y);

  // Launch kernel
  matrixMul<<<blocks, threads>>>(d_a, d_b, d_c);

  // Copy back to the host
  cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost);

  // Check result
  verify_result(h_a, h_b, h_c);

  cout << "COMPLETED SUCCESSFULLY\n";

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
