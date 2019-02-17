// This program computer the sum of two N-element vectors using unified memory
// By: Nick from CoffeeBeforeArch

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

__global__ void vectorAddUM(int *a, int *b, int *c, int n) {
	// Calculate global thread id
	int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

	// Boundary check
	if (tid < n) {
		c[tid] = a[tid] + b[tid];
	}
}

void init_vector(int *a, int *b, int n) {
	for (int i = 0; i < n; i++) {
		a[i] = rand() % 100;
		b[i] = rand() % 100;
	}
}

void check_answer(int *a, int *b, int *c, int n) {
	for (int i = 0; i < n; i++) {
		assert(c[i] == a[i] + b[i]);
	}
}

int main() {
	// Get the device ID for other CUDA calls
	int id = cudaGetDevice(&id);

	// Declare number of elements per-array
	int n = 1 << 16;

	// Size of each arrays in bytes
	size_t bytes = n * sizeof(int);
	
	// Declare unified memory pointers
	int *a, *b, *c;

	// Allocation memory for these pointers
	cudaMallocManaged(&a, bytes);
	cudaMallocManaged(&b, bytes);
	cudaMallocManaged(&c, bytes);

	// Initialize vectors
	init_vector(a, b, n);
	
	// Set threadblock size
	int BLOCK_SIZE = 256;

	// Set grid size
	int GRID_SIZE = (int)ceil(n / BLOCK_SIZE);

	// Call CUDA kernel
	// cudaMemPrefetchAsync(a, bytes, id);
	// cudaMemPrefetchAsync(b, bytes, id);
	vectorAddUM <<<GRID_SIZE, BLOCK_SIZE>>> (a, b, c, n);
	
	// Wait for all previous operations before using values
	cudaDeviceSynchronize();
	// cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

	// Check result
	check_answer(a, b, c, n);

	printf("COMPLETED SUCCESSFULLY\n");
	
	return 0;
}