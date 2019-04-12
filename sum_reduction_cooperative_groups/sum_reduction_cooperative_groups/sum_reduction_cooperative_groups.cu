// This program performs sum reduction with an optimization
// removing warp bank conflicts
// By: Nick from CoffeeBeforeArch

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <iostream>

using namespace cooperative_groups;

// Reduces a thread group to a single element
__device__ int reduce_sum(thread_group g, int *temp, int val){
	int lane = g.thread_rank();

	// Each thread adds its partial sum[i] to sum[lane+i]
	for (int i = g.size() / 2; i > 0; i /= 2){
		temp[lane] = val;
		// wait for all threads to store
		g.sync();
		if (lane < i) {
			val += temp[lane + i];
		}
		// wait for all threads to load
		g.sync();
	}
	// note: only thread 0 will return full sum
	return val; 
}

// Creates partials sums from the original array
__device__ int thread_sum(int *input, int n){
	int sum = 0;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = tid; i < n / 4; i += blockDim.x * gridDim.x){
		// Cast as int4 
		int4 in = ((int4*)input)[i];
		sum += in.x + in.y + in.z + in.w;
	}
	return sum;
}

__global__ void sum_reduction(int *sum, int *input, int n){
	// Create partial sums from the array
	int my_sum = thread_sum(input, n);

	// Dynamic shared memory allocation
	extern __shared__ int temp[];
	
	// Identifier for a TB
	auto g = this_thread_block();
	
	// Reudce each TB
	int block_sum = reduce_sum(g, temp, my_sum);

	// Collect the partial result from each TB
	if (g.thread_rank() == 0) {
		atomicAdd(sum, block_sum);
	}
}

void initialize_vector(int *v, int n) {
	for (int i = 0; i < n; i++) {
		v[i] = 1;//rand() % 10;
	}
}

int main() {
	// Vector size
	int n = 1 << 13;
	size_t bytes = n * sizeof(int);

	// Original vector and result vector
	int *sum;
	int *data;

	// Allocate using unified memory
	cudaMallocManaged(&sum, sizeof(int));
	cudaMallocManaged(&data, bytes);

	// Initialize vector
	initialize_vector(data, n);

	// TB Size
	int TB_SIZE = 256;

	// Grid Size (cut in half)
	int GRID_SIZE = (n + TB_SIZE - 1) / TB_SIZE;

	// Call kernel with dynamic shared memory (Could decrease this to fit larger data)
	sum_reduction <<<GRID_SIZE, TB_SIZE, n * sizeof(int)>>> (sum, data, n);

	// Synchronize the kernel
	cudaDeviceSynchronize();

	//printf("Accumulated result is %d \n", sum[0]);
	//scanf("Press enter to continue: ");
    assert(*sum == 8192);

	printf("COMPLETED SUCCESSFULLY\n");

	return 0;
}
