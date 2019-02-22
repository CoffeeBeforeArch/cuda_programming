// This program computes the product of two matrices
// By: Nick from CoffeeBeforeArch

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

__global__ void matrixMul(int *a, int *b, int *c, int n) {
	// Compute each thread's row
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	// Compute each thread's column
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int temp_sum = 0;
	// Boundary protection
	if ((row < n) && (col < n)) {
		// Iterate over row, and down column
		for (int k = 0; k < n; k++) {
			// Accumulate result for a single element
			temp_sum += a[row * n + k] * b[k * n + col];
		}
		// Assign result
		c[row * n + col] = temp_sum;
	}

}

// Initialization function for matrices
void matrix_init(int *a, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			a[i * n + j] = rand() % 100;
		}
	}
}

// Check result
void check_answer(int *a, int *b, int *c, int n) {
	int *verify_c;
	verify_c = (int*)malloc(n * n * sizeof(int));
	int temp_sum;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			temp_sum = 0;
			for (int k = 0; k < n; k++) {
				temp_sum += a[i * n + k] * b[k * n + j];
			}
			verify_c[i * n + j] = temp_sum;
		}
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			assert(c[i * n + j] == verify_c[i * n + j]);
		}
	}
}

int main() {
	// Matrix size of 1024 x 1024;
	int n = 1 << 10;

	// Size (in bytes) of matrix
	size_t bytes = n * n * sizeof(int);

	// Host pointers
	int *h_a, *h_b, *h_c;

	// Allocate host memory
	h_a = (int*)malloc(bytes);
	h_b = (int*)malloc(bytes);
	h_c = (int*)malloc(bytes);

	// Device pointers
	int *d_a, *d_b, *d_c;

	// Allocated device memory
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	// Initialize matrices
	matrix_init(h_a, n);
	matrix_init(h_b, n);

	// Copy data to the device
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

	// Threads per block
	int BLOCK_SIZE = 16;

	// Blocks in each dimension
	int GRID_SIZE = (int)ceil(n / BLOCK_SIZE);

	// Use dim3 objects
	dim3 grid(GRID_SIZE, GRID_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	// Launch kernel
	matrixMul <<<grid, threads >>> (d_a, d_b, d_c, n);

	// Copy back to the host
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	// Check result
	check_answer(h_a, h_b, h_c, n);

    // Free memory on host
    free(h_a);
    free(h_b);
    free(h_c);

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

	printf("COMPLETED SUCCESSFULLY\n");

	return 0;
}
