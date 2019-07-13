// This program performs sum reduction with an optimization
// removing warp bank conflicts
// By: Nick from CoffeeBeforeArch

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <iostream>

using namespace std;

#define SIZE 256
#define SHMEM_SIZE 256 * 4

__global__ void sum_reduction(int *v, int *v_r, clock_t *time) {
    // First thread gets the starting clock value
    if(threadIdx.x == 0){
        time[blockIdx.x] = clock();
    }

    // Allocate shared memory
	__shared__ int partial_sum[SHMEM_SIZE];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements into shared memory
	partial_sum[threadIdx.x] = v[tid];
	__syncthreads();

	// Start at 1/2 block stride and divide by two each iteration
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		// Each thread does work unless it is further than the stride
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = partial_sum[0];
	}
    
    // First thread gets the ending clock value
    if(threadIdx.x == 0){
        time[blockIdx.x + gridDim.x] = clock();
    }
}

void initialize_vector(int *v, int n) {
	for (int i = 0; i < n; i++) {
		v[i] = 1;
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

	// Grid Size (No padding)
	int GRID_SIZE = n / TB_SIZE;

    // Allocate space for the clock
    clock_t *time = new clock_t[GRID_SIZE * 2];
    clock_t *d_time;
    cudaMalloc(&d_time, sizeof(clock_t) * GRID_SIZE * 2);

	// Call kernel
	sum_reduction <<<GRID_SIZE, TB_SIZE >>> (d_v, d_v_r, d_time);

    // Just get the results for the first kernel 
    cudaMemcpy(time, d_time, sizeof(clock_t) * GRID_SIZE * 2, cudaMemcpyDeviceToHost);

	sum_reduction <<<1, TB_SIZE >>> (d_v_r, d_v_r, d_time);

	// Copy to host;
	cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

    // Calculate the number of Clocks/block
    cout << "Block,Clocks" << endl; 
    for(int i = 0; i < GRID_SIZE; i++){
        cout << i << "," << (time[i+GRID_SIZE] - time[i]) << endl;
    }

	return 0;
}
