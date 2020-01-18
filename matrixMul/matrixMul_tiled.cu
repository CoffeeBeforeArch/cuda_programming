// This program is an optimized version of matrix multiplication
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

// Static shmem calculation for convenience (Int 32x32 matrix)
#define SHMEM_SIZE 32 * 32

__global__ void matrixMul(int *a, int *b, int *c, int N) {
  // Two statically-sized pieces of shared memory
  __shared__ int A[SHMEM_SIZE];
  __shared__ int B[SHMEM_SIZE];

  // Shorten these parameters for clean re-use
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Square tiles, so this could be blockDim.y as well
  int tile_size = blockDim.x;

  // Calculate global row and column positions for this thread
  int row = by * tile_size + ty;
  int col = bx * tile_size + tx;

  // Intermediate sum for element being written
  int tmp = 0;

  // Sweep tiles over entire matrix
  for (int i = 0; i < (N / tile_size); i++) {
    /*
            Every thread in a threadblock loads one element into shared memory
            The element location in shared memory corresponds to the thread's
            position in the threadblock (e.g. thread [0, 0] loads for
            A[0 * tile_size + 0], and B[0 * tile_size + 0].)

            Explanation of indexing parameters
            For A:
                            row*N: Indexes the global row for this thread
       (loop-invariant) i*tile_size: Indexes the new set of columns each
       iteration tx: Indexes the column within that set for B: i*tile_size*N:
       Indexes the next set of rows each iteration ty*N: Indexes the row within
       that set col: Indexes the global column (loop-invariant)
    */
    A[ty * tile_size + tx] = a[(row * N) + (i * tile_size + tx)];
    B[ty * tile_size + tx] = b[(i * tile_size * N + ty * N) + col];

    // Ensure all threads have loaded their data before proceeding
    __syncthreads();

    // Calculate all temp values for this tile
    for (int j = 0; j < tile_size; j++) {
      tmp += A[(ty * tile_size) + j] * B[(j * tile_size) + tx];
    }

    // Ensure some threads don't progress and stomp current shared memory values
    __syncthreads();
  }

  // Write back the result
  c[row * N + col] = tmp;
}

// Functional test for our program
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c, int N) {
  // For every row...
  for (int i = 0; i < N; i++) {
    // For every column...
    for (int j = 0; j < N; j++) {
      // For every element in the row/col pair...
      int tmp = 0;
      for (int k = 0; k < N; k++) {
        // Accumulate the partial results
        tmp += a[i * N + k] * b[k * N + j];
      }

      // Kill the program if something was wrong
      assert(tmp == c[i * N + j]);
    }
  }
}

int main() {
  // Problem size = 1024 x 1024 matrix
  constexpr int N = 1 << 10;
  size_t bytes = N * N * sizeof(int);

  // Host matrix pointers
  vector<int> h_a(N * N);
  vector<int> h_b(N * N);
  vector<int> h_c(N * N);

  // Initialize matrices
  generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
  generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

  // Allocate device memory
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // Copy matrices to the device
  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

  // Set the CTA and Grid dimensions
  int THREADS = 32;
  int BLOCKS = N / THREADS;

  // Use dim3 objects for 2-D grids and threadblocks
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  // Launch kernel
  matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N);

  // Copy result back from device
  cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  // Verify the result
  verify_result(h_a, h_b, h_c, N);

  cout << "COMPLETED SUCCESFULLY!\n";

  // Free device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
