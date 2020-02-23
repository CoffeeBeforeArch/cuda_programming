// This program performs sum reduction with an optimization
// removing warp divergence
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

using std::accumulate;
using std::cout;
using std::generate;
using std::vector;

#define SHMEM_SIZE 256

__global__ void sumReduction(int *v, int *v_r) {
  // Allocate shared memory
  __shared__ int partial_sum[SHMEM_SIZE];

  // Calculate thread ID
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Load elements into shared memory
  partial_sum[threadIdx.x] = v[tid];
  __syncthreads();

  // Increase the stride of the access until we exceed the CTA dimensions
  for (int s = 1; s < blockDim.x; s *= 2) {
    // Change the indexing to be sequential threads
    int index = 2 * s * threadIdx.x;

    // Each thread does work unless the index goes off the block
    if (index < blockDim.x) {
      partial_sum[index] += partial_sum[index + s];
    }
    __syncthreads();
  }

  // Let the thread 0 for this block write it's result to main memory
  // Result is inexed by this block
  if (threadIdx.x == 0) {
    v_r[blockIdx.x] = partial_sum[0];
  }
}

int main() {
  // Vector size
  int N = 1 << 16;
  size_t bytes = N * sizeof(int);

  // Host data
  vector<int> h_v(N);
  vector<int> h_v_r(N);

  // Initialize the input data
  generate(begin(h_v), end(h_v), []() { return rand() % 10; });

  // Allocate device memory
  int *d_v, *d_v_r;
  cudaMalloc(&d_v, bytes);
  cudaMalloc(&d_v_r, bytes);

  // Copy to device
  cudaMemcpy(d_v, h_v.data(), bytes, cudaMemcpyHostToDevice);

  // TB Size
  const int TB_SIZE = 256;

  // Grid Size (No padding)
  int GRID_SIZE = N / TB_SIZE;

  // Call kernels
  sumReduction<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r);

  sumReduction<<<1, TB_SIZE>>>(d_v_r, d_v_r);

  // Copy to host;
  cudaMemcpy(h_v_r.data(), d_v_r, bytes, cudaMemcpyDeviceToHost);

  // Print the result
  assert(h_v_r[0] == std::accumulate(begin(h_v), end(h_v), 0));

  cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}
