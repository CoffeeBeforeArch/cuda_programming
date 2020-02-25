// This program implements 2D convolution using Constant memory in CUDA
// By: Nick from CoffeeBeforeArch

#include <cassert>
#include <cstdlib>
#include <iostream>

// 7 x 7 convolutional mask
#define MASK_DIM 7

// Amount the the matrix will hang over the matrix
#define MASK_OFFSET (MASK_DIM / 2)

// Allocate mask in constant memory
__constant__ int mask[7 * 7];

// 2D Convolution Kernel
// Takes:
//  matrix: Input matrix
//  result: Convolution result
//  N:      Dimensions of the matrices
__global__ void convolution_2d(int *matrix, int *result, int N) {
  // Calculate the global thread positions
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Starting index for calculation
  int start_r = row - MASK_OFFSET;
  int start_c = col - MASK_OFFSET;

  // Temp value for accumulating the result
  int temp = 0;

  // Iterate over all the rows
  for (int i = 0; i < MASK_DIM; i++) {
    // Go over each column
    for (int j = 0; j < MASK_DIM; j++) {
      // Range check for rows
      if ((start_r + i) >= 0 && (start_r + i) < N) {
        // Range check for columns
        if ((start_c + j) >= 0 && (start_c + j) < N) {
          // Accumulate result
          temp += matrix[(start_r + i) * N + (start_c + j)] *
                  mask[i * MASK_DIM + j];
        }
      }
    }
  }

  // Write back the result
  result[row * N + col] = temp;
}

// Initializes an n x n matrix with random numbers
// Takes:
//  m : Pointer to the matrix
//  n : Dimension of the matrix (square)
void init_matrix(int *m, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      m[n * i + j] = rand() % 100;
    }
  }
}

// Verifies the 2D convolution result on the CPU
// Takes:
//  m:      Original matrix
//  mask:   Convolutional mask
//  result: Result from the GPU
//  N:      Dimensions of the matrix
void verify_result(int *m, int *mask, int *result, int N) {
  // Temp value for accumulating results
  int temp;

  // Intermediate value for more readable code
  int offset_r;
  int offset_c;

  // Go over each row
  for (int i = 0; i < N; i++) {
    // Go over each column
    for (int j = 0; j < N; j++) {
      // Reset the temp variable
      temp = 0;

      // Go over each mask row
      for (int k = 0; k < MASK_DIM; k++) {
        // Update offset value for row
        offset_r = i - MASK_OFFSET + k;

        // Go over each mask column
        for (int l = 0; l < MASK_DIM; l++) {
          // Update offset value for column
          offset_c = j - MASK_OFFSET + l;

          // Range checks if we are hanging off the matrix
          if (offset_r >= 0 && offset_r < N) {
            if (offset_c >= 0 && offset_c < N) {
              // Accumulate partial results
              temp += m[offset_r * N + offset_c] * mask[k * MASK_DIM + l];
            }
          }
        }
      }
      // Fail if the results don't match
      assert(result[i * N + j] == temp);
    }
  }
}

int main() {
  // Dimensions of the matrix (2 ^ 10 x 2 ^ 10)
  int N = 1 << 10;

  // Size of the matrix (in bytes)
  size_t bytes_n = N * N * sizeof(int);

  // Allocate the matrix and initialize it
  int *matrix = new int[N * N];
  int *result = new int[N * N];
  init_matrix(matrix, N);

  // Size of the mask in bytes
  size_t bytes_m = MASK_DIM * MASK_DIM * sizeof(int);

  // Allocate the mask and initialize it
  int *h_mask = new int[MASK_DIM * MASK_DIM];
  init_matrix(h_mask, MASK_DIM);

  // Allocate device memory
  int *d_matrix;
  int *d_result;
  cudaMalloc(&d_matrix, bytes_n);
  cudaMalloc(&d_result, bytes_n);

  // Copy data to the device
  cudaMemcpy(d_matrix, matrix, bytes_n, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(mask, h_mask, bytes_m);

  // Calculate grid dimensions
  int THREADS = 16;
  int BLOCKS = (N + THREADS - 1) / THREADS;

  // Dimension launch arguments
  dim3 block_dim(THREADS, THREADS);
  dim3 grid_dim(BLOCKS, BLOCKS);

  // Perform 2D Convolution
  convolution_2d<<<grid_dim, block_dim>>>(d_matrix, d_result, N);

  // Copy the result back to the CPU
  cudaMemcpy(result, d_result, bytes_n, cudaMemcpyDeviceToHost);

  // Functional test
  verify_result(matrix, h_mask, result, N);

  std::cout << "COMPLETED SUCCESSFULLY!";

  // Free the memory we allocated
  delete[] matrix;
  delete[] result;
  delete[] h_mask;

  cudaFree(d_matrix);
  cudaFree(d_result);

  return 0;
}

