// This program computes the sum of two vectors of length N
// By: Nick from CoffeeBeforeArch

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <iostream>

// CUDA kernel for vector addition
__global__ void vectorAdd(int* a, int* b, int* c, int n) {
	// Calculate global thread ID (tid)
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	// Vector boundary guard
	if (tid < n) {
		// Each thread adds a single element
		c[tid] = a[tid] + b[tid];
	}
}

// Initialize vector of size n to int between 0-99
void vector_init(int* a, int n) {
	for (int i = 0; i < n; i++) {
		a[i] = rand() % 100;
	}
}

// Check vector add result
void check_answer(int* a, int* b, int*c, int n) {
	for (int i = 0; i < n; i++) {
		assert(c[i] == a[i] + b[i]);
	}
}

int main() {
	// Vector size of 2^16 (65536 elements)
	int n = 1 << 16;
	// Host vector pointers
	int *h_a, *h_b, *h_c;
	// Device vector pointers
	int *d_a, *d_b, *d_c;
	// Allocation size for all vectors
	size_t bytes = sizeof(int) * n;

	// Allocate host memory
	h_a = (int*)malloc(bytes);
	h_b = (int*)malloc(bytes);
	h_c = (int*)malloc(bytes);

	// Allocate device memory
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	// Initialize vectors a and b with random values between 0 and 99
	vector_init(h_a, n);
	vector_init(h_b, n);

	// Copy data from
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

	// Threadblock size
	int NUM_THREADS = 256;
	
	// Grid size
	int NUM_BLOCKS = (n + NUM_THREADS - 1) / NUM_THREADS;

	// Launch kernel on default stream w/o shmem
	vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, n);

	// Copy sum vector from device to host
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	// Check result for errors
	check_answer(h_a, h_b, h_c, n);

    // Free memory on host
    free(h_a);
    free(h_b);
    free(h_c);

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

	printf("COMPLETED SUCCESFULLY\n");

	return 0;
}
