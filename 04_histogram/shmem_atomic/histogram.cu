// This program shows off a shared memory implementation of a histogram
// kernel in CUDA
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

using std::accumulate;
using std::cout;
using std::generate;
using std::ios;
using std::ofstream;
using std::vector;

// Number of bins for our plot
constexpr int BINS = 7;
constexpr int DIV = ((26 + BINS - 1) / BINS);

// GPU kernel for computing a histogram
// Takes:
//  a: Problem array in global memory
//  result: result array
//  N: Size of the array
__global__ void histogram(char *a, int *result, int N) {
  // Calculate global thread ID
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Allocate a local histogram for each TB
  __shared__ int s_result[BINS];

  // Initalize the shared memory to 0
  if (threadIdx.x < BINS) {
    s_result[threadIdx.x] = 0;
  }

  // Wait for shared memory writes to complete
  __syncthreads();

  // Calculate the bin positions locally
  int alpha_position;
  for (int i = tid; i < N; i += (gridDim.x * blockDim.x)) {
    // Calculate the position in the alphabet
    alpha_position = a[i] - 'a';
    atomicAdd(&s_result[(alpha_position / DIV)], 1);
  }

  // Wait for shared memory writes to complete
  __syncthreads();

  // Combine the partial results
  if (threadIdx.x < BINS) {
    atomicAdd(&result[threadIdx.x], s_result[threadIdx.x]);
  }
}

int main() {
  // Declare our problem size
  int N = 1 << 16;

  // Allocate memory on the host
  vector<char> h_input(N);
  vector<int> h_result(BINS);

  // Initialize the array
  srand(1);
  generate(begin(h_input), end(h_input), []() { return 'a' + (rand() % 26); });

  // Allocate memory on the device
  char *d_input;
  int *d_result;
  cudaMalloc(&d_input, N);
  cudaMalloc(&d_result, BINS * sizeof(int));

  // Copy the array to the device
  cudaMemcpy(d_input, h_input.data(), N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_result, h_result.data(), BINS * sizeof(int),
             cudaMemcpyHostToDevice);

  // Number of threads per threadblock
  int THREADS = 512;

  // Calculate the number of threadblocks
  int BLOCKS = N / THREADS;

  // Launch the kernel
  histogram<<<BLOCKS, THREADS>>>(d_input, d_result, N);

  // Copy the result back
  cudaMemcpy(h_result.data(), d_result, BINS * sizeof(int),
             cudaMemcpyDeviceToHost);

  // Functional test (the sum of all bins == N)
  assert(N == accumulate(begin(h_result), end(h_result), 0));

  // Dump the counts of the bins to a file
  ofstream output_file;
  output_file.open("histogram.dat", ios::out | ios::trunc);
  for (auto i : h_result) {
    output_file << i << " \n\n";
  }
  output_file.close();

  // Free memory
  cudaFree(d_input);
  cudaFree(d_result);

  return 0;
}
