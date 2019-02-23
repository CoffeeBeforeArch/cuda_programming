// This program is an optimized version of Matrix Multiplication
// By: Nick from CoffeeBeforeArch

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

__global__ void matrixMulMisaligned(int *a, int *b, int *c, int n) {
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
			temp_sum += a[row * n + k] * b[col * n + k];
		}
		// Assign result
		c[row * n + col] = temp_sum;
	}
}

void transpose_matrix(int *a, int *a_t, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			a_t[i*n + j] = a[j*n + i];
		}
	}
}

void check_answer(int *a, int *b, int *c, int n) {
	int *verify_c;
	verify_c = (int*)malloc(n * n * sizeof(int));
	int temp_val;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			temp_val = 0;
			for (int k = 0; k < n; k++) {
				temp_val += a[i * n + k] * b[k * n + j];
			}
			verify_c[i * n + j] = temp_val;
		}
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			assert(c[i * n + j] == verify_c[i * n + j]);
		}
	}
}

void init_matrix(int *a, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			a[i * n + j] = rand() % 10;
		}
	}
}

int main() {
	// Problem size = 1024 x 1024 matrix
	int n = 1 << 10;

	// Matrix size (in bytes)
	size_t bytes = n * n * sizeof(int);

	// Host pointer to transposeed matrix
	int *h_b_t;

	// Host matrix pointers
	int *h_a, *h_b, *h_c;

	// Device pointer to transposeed matrix
	int *d_b_t;

	// Device matrix pointers
	int *d_a, *d_c;

	// Allocate host memory
	h_a = (int*)malloc(bytes);
	h_b = (int*)malloc(bytes);
	h_b_t = (int*)malloc(bytes);
	h_c = (int*)malloc(bytes);

	// Allocate device memory
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b_t, bytes);
	cudaMalloc(&d_c, bytes);


	// Initialize matrices
	init_matrix(h_a, n);
	init_matrix(h_b, n);

	// Transpose matrix a
	transpose_matrix(h_b, h_b_t, n);

	// Copy matrices to the device
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b_t, h_b_t, bytes, cudaMemcpyHostToDevice);

	// Threads per block (in both x and y dimensions)
	int BLOCK_SIZE = 16;

	// Blocks in each dimension
	int GRID_SIZE = (int)ceil(n / BLOCK_SIZE);

	// Use dim3 objects for 2-D grids and threadblocks
	dim3 grid(GRID_SIZE, GRID_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	// Launch kernel
	matrixMulMisaligned<<<grid, threads >>> (d_a, d_b_t, d_c, n);

	// Copy result back from device
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	// Verify the result
	check_answer(h_a, h_b, h_c, n);

	// Free host memory
	free(h_a);
	free(h_b);
	free(h_b_t);
	free(h_c);

	// Free device memory
	cudaFree(d_a);
	cudaFree(d_b_t);
	cudaFree(d_c);

	printf("COMPLETED SUCCESSFULLY\n");

	return 0;
}
