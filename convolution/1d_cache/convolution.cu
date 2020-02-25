// This program implements a 1D convolution using CUDA,
// and stores the mask in constant memory. It loads the
// primary array into shared memory, but not halo elements.
// By: Nick from CoffeeBeforeArch

#include <cassert>
#include <cstdlib>
#include <iostream>

// Length of our convolution mask
#define MASK_LENGTH 7

// Allocate space for the mask in constant memory
__constant__ int mask[MASK_LENGTH];

// 1-D convolution kernel
// All threads load 1 element into shared memory
// All threads compute 1 element in final array
//  Arguments:
//      array   = padded array
//      result  = result array
//      n       = number of elements in array
__global__ void convolution_1d(int *array, int *result, int n) {
  // Global thread ID calculation
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Store all elements needed to compute output in shared memory
  extern __shared__ int s_array[];

  // Load elements from the main array into shared memory
  // This is naturally offset by "r" due to padding
  s_array[threadIdx.x] = array[tid];

  __syncthreads();

  // Temp value for calculation
  int temp = 0;

  // Go over each element of the mask
  for (int j = 0; j < MASK_LENGTH; j++) {
    // Get the array value from the caches
    if ((threadIdx.x + j) >= blockDim.x) {
      temp += array[tid + j] * mask[j];
      // Get the value from shared memory
      // Only the last warp will be diverged (given mask size)
    } else {
      temp += s_array[threadIdx.x + j] * mask[j];
    }
  }

  // Write-back the results
  result[tid] = temp;
}

// Verify the result on the CPU
void verify_result(int *array, int *mask, int *result, int n) {
  int temp;
  for (int i = 0; i < n; i++) {
    temp = 0;
    for (int j = 0; j < MASK_LENGTH; j++) {
      temp += array[i + j] * mask[j];
    }
    assert(temp == result[i]);
  }
}

int main() {
  // Number of elements in result array
  int n = 1 << 20;

  // Size of the array in bytes
  int bytes_n = n * sizeof(int);

  // Size of the mask in bytes
  size_t bytes_m = MASK_LENGTH * sizeof(int);

  // Radius for padding the array
  int r = MASK_LENGTH / 2;
  int n_p = n + r * 2;

  // Size of the padded array in bytes
  size_t bytes_p = n_p * sizeof(int);

  // Allocate the array (include edge elements)...
  int *h_array = new int[n_p];

  // ... and initialize it
  for (int i = 0; i < n_p; i++) {
    if ((i < r) || (i >= (n + r))) {
      h_array[i] = 0;
    } else {
      h_array[i] = rand() % 100;
    }
  }

  // Allocate the mask and initialize it
  int *h_mask = new int[MASK_LENGTH];
  for (int i = 0; i < MASK_LENGTH; i++) {
    h_mask[i] = rand() % 10;
  }

  // Allocate space for the result
  int *h_result = new int[n];

  // Allocate space on the device
  int *d_array, *d_result;
  cudaMalloc(&d_array, bytes_p);
  cudaMalloc(&d_result, bytes_n);

  // Copy the data to the device
  cudaMemcpy(d_array, h_array, bytes_p, cudaMemcpyHostToDevice);

  // Copy the mask directly to the symbol
  // This would require 2 API calls with cudaMemcpy
  cudaMemcpyToSymbol(mask, h_mask, bytes_m);

  // Threads per TB
  int THREADS = 256;

  // Number of TBs
  int GRID = (n + THREADS - 1) / THREADS;

  // Amount of space per-block for shared memory
  // This is padded by the overhanging radius on either side
  size_t SHMEM = THREADS * sizeof(int);

  // Call the kernel
  convolution_1d<<<GRID, THREADS, SHMEM>>>(d_array, d_result, n);

  // Copy back the result
  cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);

  // Verify the result
  verify_result(h_array, h_mask, h_result, n);

  std::cout << "COMPLETED SUCCESSFULLY\n";

  // Free allocated memory on the device and host
  delete[] h_array;
  delete[] h_result;
  delete[] h_mask;
  cudaFree(d_array);
  cudaFree(d_result);

  return 0;
}
