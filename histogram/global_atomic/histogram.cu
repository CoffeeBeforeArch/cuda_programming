// This program shows off a global memory implementation of a histogram
// kernel in CUDA
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
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

  // Calculate the bin positions where threads are grouped together
  int alpha_position;
  for (int i = tid; i < N; i += (gridDim.x * blockDim.x)) {
    // Calculate the position in the alphabet
    alpha_position = a[i] - 'a';
    atomicAdd(&result[alpha_position / DIV], 1);
  }
}

int main() {
  // Declare our problem size
  int N = 1 << 24;

  // Allocate memory on the host
  vector<char> h_input(N);

  // Allocate space for the binned result
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

  // Functional test
  assert(N == accumulate(begin(h_result), end(h_result), 0));

  // Write the data out for gnuplot
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
