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
			temp_sum += a[row * n + k] + b[k * n + col];
		}
	}
	// Assign result
	c[row * n + col] = temp_sum;
}

// Initialization function for matrices
void init_matrices(int *a, int *b, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			a[i * n + j] = rand() % 100;
			b[i * n + j] = rand() % 100;
		}
	}
}

// Check result
void verify_result() {

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
	init_matrices(h_a, h_b, n);

	// Copy data to the device
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

	// Threads per block
	int BLOCK_SIZE = 128;

	// Blocks in each dimension
	int GRID_SIZE = (int)ceil(n / 128);

	// Use dim3 objects
	dim3 grid(GRID_SIZE, GRID_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	// Launch kernel
	matrixMul<<<grid, threads>>>(d_a, d_b, d_c, n);

	// Copy back to the host
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	// Check result
	verify_result();

	return 0;
}