// This program implements a 1D convolution using CUDA
// By: Nick from CoffeeBeforeArch

#include <assert.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

// 1-D convolution kernel
//  Arguments:
//      array   = padded array
//      mask    = convolution mask
//      result  = result array
//      n       = number of elements in array
//      m       = number of elements in the mask
__global__ void convolution_1d(int *array, int *mask, int *result, int n,
                               int m) {
  // Global thread ID calculation
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Calculate radius of the mask
  int r = m / 2;

  // Calculate the starting point for the element
  int start = tid - r;

  // Temp value for calculation
  int temp = 0;

  // Go over each element of the mask
  for (int j = 0; j < m; j++) {
    // Ignore elements that hang off (0s don't contribute)
    if (((start + j) >= 0) && (start + j < n)) {
      // accumulate partial results
      temp += array[start + j] * mask[j];
    }
  }

  // Write-back the results
  result[tid] = temp;
}

// Verify the result on the CPU
void verify_result(int *array, int *mask, int *result, int n, int m) {
  int radius = m / 2;
  int temp;
  int start;
  for (int i = 0; i < n; i++) {
    start = i - radius;
    temp = 0;
    for (int j = 0; j < m; j++) {
      if ((start + j >= 0) && (start + j < n)) {
        temp += array[start + j] * mask[j];
      }
    }
    assert(temp == result[i]);
  }
}

int main() {
  // Number of elements in result array
  int n = 1 << 20;

  // Size of the array in bytes
  int bytes_n = n * sizeof(int);

  // Number of elements in the convolution mask
  int m = 7;

  // Size of mask in bytes
  int bytes_m = m * sizeof(int);

  // Allocate the array (include edge elements)...
  int *h_array = new int[n];

  // ... and initialize it
  for (int i = 0; i < n; i++) {
    h_array[i] = rand() % 100;
  }

  // Allocate the mask and initialize it
  int *h_mask = new int[m];
  for (int i = 0; i < m; i++) {
    h_mask[i] = rand() % 10;
  }

  // Allocate space for the result
  int *h_result = new int[n];

  // Allocate space on the device
  int *d_array, *d_mask, *d_result;
  cudaMalloc(&d_array, bytes_n);
  cudaMalloc(&d_mask, bytes_m);
  cudaMalloc(&d_result, bytes_n);

  // Copy the data to the device
  cudaMemcpy(d_array, h_array, bytes_n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mask, h_mask, bytes_m, cudaMemcpyHostToDevice);

  // Threads per TB
  int THREADS = 256;

  // Number of TBs
  int GRID = (n + THREADS - 1) / THREADS;

  // Call the kernel
  convolution_1d<<<GRID, THREADS>>>(d_array, d_mask, d_result, n, m);

  // Copy back the result
  cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);

  // Verify the result
  verify_result(h_array, h_mask, h_result, n, m);

  cout << "COMPLETED SUCCESSFULLY\n";

  // Free allocated memory on the device and host
  delete[] h_array;
  delete[] h_result;
  delete[] h_mask;
  cudaFree(d_result);

  return 0;
}
