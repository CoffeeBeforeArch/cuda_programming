// This program computes a sum reduction algortithm with warp divergence
// By: Nick from CoffeeBeforeArch

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#define SIZE 256

__global__ void sum_reduction(int *v, int *v_r) {
	// Allocate shared memory
	__shared__ int partial_sum[SIZE];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements into shared memory
	partial_sum[threadIdx.x] = v[tid];
	__syncthreads();

	// Iterate of log base 2 the block dimension
	for (int s = 1; s < blockDim.x; s *= 2) {
		// Reduce the threads performing work by half previous the previous
		// iteration each cycle
		if (threadIdx.x % (2 * s) == 0) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (blockIdx.x == 0) {
		v_r[blockIdx.x] = partial_sum[0];
	}
}

void initialize_vector(int *v, int n) {
	for (int i = 0; i < n; i++) {
		v[i] = 1;//rand() % 10;
	}
}

int main() {
	// Vector size
	int n = 1 << 7;
	size_t bytes = n * sizeof(int);

	// Original vector and result vector
	int *h_v, *h_v_r;
	int *d_v, *d_v_r;

	// Allocate memory
	h_v = (int*)malloc(bytes);
	h_v_r = (int*)malloc(bytes);
	cudaMalloc(&d_v, bytes);
	cudaMalloc(&d_v_r, bytes);
	
	// Initialize vector
	initialize_vector(h_v, n);

	// Copy to device
	cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);
	
	// TB Size
	int TB_SIZE = 128;

	// Grid Size
	int GRID_SIZE = (int)ceil(n / TB_SIZE);

	// Call kernel
	sum_reduction<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r);

	// Copy to host;
	cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

	// Print the result
	printf("%d \n", h_v_r[0]);

	assert(h_v_r[0] == 128);

	return 0;
}