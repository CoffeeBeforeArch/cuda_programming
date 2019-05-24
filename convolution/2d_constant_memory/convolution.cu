// This program implements 2D convolution using Constant memory in CUDA
// By: Nick from CoffeeBeforeArch

#include <iostream>
#include <assert.h>
#include <stdlib.h>

using namespace std;

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
__global__ void convolution_2d(int *matrix, int *result, int N){
    // Calculate the global thread positions
    int row = threadIdx.y * blockDim.y + gridDim.y;
    int col = threadIdx.x * blockDim.x + gridDim.x;

    // Starting index for calculation
    int start_r = row - MASK_OFFSET;
    int start_c = col - MASK_OFFSET;

    // Temp value for accumulating the result
    int temp = 0;

    // Iterate over all the rows
    for(int i = 0; i < MASK_DIM; i++){
        // Update the index
        start_r += i;

        // Range check
        if(start_r > 0 && start_r < N){
            // Go over each column
            for(int j = 0; j < MASK_DIM; j++){
                // Update the column index
                start_c += j;

                // Range check
                if(start_c > 0 && start_c < N){
                    temp += matrix[start_r * N + start_c] * mask[i * MASK_DIM + j];
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
void init_matrix(int *m, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            m[n * i + j] = rand() % 100;
        }
    }
}

int main(){
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

    convolution_2d<<<grid_dim, block_dim>>>(d_matrix, d_result, N);

    cudaMemcpy(result, d_result, bytes_m, cudaMemcpyDeviceToHost);

    return 0;
}
