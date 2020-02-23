// This program computes the sum of two vectors of length N
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <vector>

using std::begin;
using std::copy;
using std::cout;
using std::end;
using std::generate;
using std::vector;

// CUDA kernel for vector addition
// __global__ means this is called from the CPU, and runs on the GPU
__global__ void vectorAdd(int* a, int* b, int* c, int N) {
  // Calculate global thread ID
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Boundary check
  if (tid < N) {
    // Each thread adds a single element
    c[tid] = a[tid] + b[tid];
  }
}

// Check vector add result
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c) {
  for (int i = 0; i < a.size(); i++) {
    assert(c[i] == a[i] + b[i]);
  }
}

int main() {
  // Array size of 2^16 (65536 elements)
  constexpr int N = 1 << 26;
  size_t bytes = sizeof(int) * N;

  // Vectors for holding the host-side (CPU-side) data
  vector<int> a(N);
  vector<int> b(N);
  vector<int> c(N);

  // Initialize random numbers in each array
  generate(begin(a), end(a), []() { return rand() % 100; });
  generate(begin(b), end(b), []() { return rand() % 100; });

  // Allocate memory on the device
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // Copy data from the host to the device (CPU -> GPU)
  cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

  // Threads per CTA (1024 threads per CTA)
  int NUM_THREADS = 1 << 10;

  // CTAs per Grid
  // We need to launch at LEAST as many threads as we have elements
  // This equation pads an extra CTA to the grid if N cannot evenly be divided
  // by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

  // Launch the kernel on the GPU
  // Kernel calls are asynchronous (the CPU program continues execution after
  // call, but no necessarily before the kernel finishes)
  vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);

  // Copy sum vector from device to host
  // cudaMemcpy is a synchronous operation, and waits for the prior kernel
  // launch to complete (both go to the default stream in this case).
  // Therefore, this cudaMemcpy acts as both a memcpy and synchronization
  // barrier.
  cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  // Check result for errors
  verify_result(a, b, c);

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}
