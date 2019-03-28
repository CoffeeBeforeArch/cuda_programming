// This program performs sum reduction with an optimization
// removing warp bank conflicts
// By: Nick from CoffeeBeforeArch

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#define SIZE 256
#define SHMEM_SIZE 256 * 4

// For last iteration (saves useless work)
// Use volatile to prevent caching in registers (compiler optimization)
// No __syncthreads() necessary!
__device__ void warpReduce(volatile int* shmem_ptr, int t) {
	shmem_ptr[t] += shmem_ptr[t + 32];
	shmem_ptr[t] += shmem_ptr[t + 16];
	shmem_ptr[t] += shmem_ptr[t + 8];
	shmem_ptr[t] += shmem_ptr[t + 4];
	shmem_ptr[t] += shmem_ptr[t + 2];
	shmem_ptr[t] += shmem_ptr[t + 1];
}

__global__ void sum_reduction(int *v, int *v_r) {
	// Allocate shared memory
	__shared__ int partial_sum[SHMEM_SIZE];

	// Calculate thread ID
	int tid = threadIdx.x;

	// Load elements AND do first add of reduction
	// Vector now 2x as long as number of threads, so scale i
	int i = blockIdx.x * (blockDim.x * 2) + tid;

	// Store first partial result instead of just the elements
	partial_sum[tid] = v[i] + v[i + blockDim.x];
	__syncthreads();

	// Unroll the loop to remove un-ncessary control flow instructions
	if (tid < 128) {
		partial_sum[tid] += partial_sum[tid + 128];
		__syncthreads();
	}

	if (tid < 64) {
		partial_sum[tid] += partial_sum[tid + 64];
		__syncthreads();
	}

	if (threadIdx.x < 32) {
		warpReduce(partial_sum, tid);
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (tid == 0) {
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
	int n = 1 << 16;
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
	int TB_SIZE = SIZE;

	// Grid Size (cut in half)
	int GRID_SIZE = (int)ceil(n / TB_SIZE / 2);

	// Call kernel
	sum_reduction << <GRID_SIZE, TB_SIZE >> > (d_v, d_v_r);

	sum_reduction << <1, TB_SIZE >> > (d_v_r, d_v_r);

	// Copy to host;
	cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

	// Print the result
	//printf("Accumulated result is %d \n", h_v_r[0]);
	//scanf("Press enter to continue: ");
	assert(h_v_r[0] == 65536);

	printf("COMPLETED SUCCESSFULLY\n");

	return 0;
}