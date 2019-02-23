// This program is an optimized version of matrix multiplication
// By: Nick from CoffeeBeforeArch

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

// Static shmem calculation for convenience (Int 16x16 matrix)
#define SHMEM_SIZE 16 * 16 * 4

__global__ void tiledMatrixMul(int *a, int *b, int *c, int n, int tile_size) {
	// Two statically-sized pieces of shared memory
	__shared__ int A[SHMEM_SIZE];
	__shared__ int B[SHMEM_SIZE];

	// Shorten these parameters for clean re-use
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Calculate global row and column positions for this thread
	int row = by * tile_size + ty;
	int col = bx * tile_size + tx;

	// Intermediate sum for element being written
	int temp_val = 0;

	// Sweep tiles over entire matrix
	for (int i = 0; i < (n / tile_size); i++) {
		/*
			Every thread in a threadblock loads one element into shared memory
			The element location in shared memory corresponds to the thread's
			position in the threadblock (e.g. thread [0, 0] loads for 
			A[0 * tile_size + 0], and B[0 * tile_size + 0].)

			Explanation of indexing parameters
			For A:
				        row*n: Indexes the global row for this thread (loop-invariant)
				  i*tile_size: Indexes the new set of columns each iteration
				           tx: Indexes the column within that set
			for B:
				i*tile_size*n: Indexes the next set of rows each iteration
				         ty*n: Indexes the row within that set
						  col: Indexes the global column (loop-invariant)
		*/
		A[(ty * tile_size) + tx] = a[row * n + (i * tile_size + tx)];
		B[(ty * tile_size) + tx] = b[(i * tile_size * n + ty * n) + col];

		// Ensure all threads have loaded their data before proceeding
		__syncthreads();

		// Calculate all temp values for this tile
		for (int j = 0; j < tile_size; j++) {
			temp_val += A[(ty * tile_size) + j] * B[(j * tile_size) + tx];
		}

		// Ensure some threads don't progress and stomp current shared memory values
		__syncthreads();
	}
	c[(row * n) + col] = temp_val;
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

	// Host matrix pointers
	int *h_a, *h_b, *h_c;

	// Device matrix pointers
	int *d_a, *d_b, *d_c;

	// Allocate host memory
	h_a = (int*)malloc(bytes);
	h_b = (int*)malloc(bytes);
	h_c = (int*)malloc(bytes);

	// Allocate device memory
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	// Initialize matrices
	init_matrix(h_a, n);
	init_matrix(h_b, n);

	// Copy matrices to the device
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

	// Threads per block (in both x and y dimensions)
	int BLOCK_SIZE = 16;

	// Blocks in each dimension
	int GRID_SIZE = (int)ceil(n / BLOCK_SIZE);

	// Use dim3 objects for 2-D grids and threadblocks
	dim3 grid(GRID_SIZE, GRID_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	// Launch kernel
	tiledMatrixMul <<<grid, threads>>> (d_a, d_b, d_c, n, BLOCK_SIZE);

	// Copy result back from device
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);	

	// Verify the result
	check_answer(h_a, h_b, h_c, n);

	// Free host memory
	free(h_a);
	free(h_b);
	free(h_c);

	// Free device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	printf("COMPLETED SUCCESSFULLY\n");

	return 0;
}