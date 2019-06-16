// This header file contains all the CUDA kernels an helper functions
// for the sum reduction performance test
// By: Nick from CoffeeBeforeArch

// For storing execution times
#include <vector>

// For rand()
#include <stdlib.h>

// To get linter to shut up
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

#define LOWER_BOUND 256
#define BLOCK_DIM 256
#define SHMEM_SIZE 256 * 4

// Array initalization function
// Takes:
//  a: Pointer to the array
//  N: Number of array elements
// Returns:
//  NA
void init_array(int *a, int N) {
    for (int i = 0; i < N; i++) {
        a[i] = rand() % 10;
    }
}

// Baseline sum reduction implementation
// Takes:
//  a:      Input array
//  result: Output array
// Returns:
//  NA
__global__ void sum_reduction_1(int *a, int *result) {
	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Allocate shared memory
	__shared__ int partial_sum[SHMEM_SIZE];

	// Load elements into shared memory
	partial_sum[threadIdx.x] = a[tid];
	__syncthreads();

	// Iterate log base two the block dimension times
	for (int s = 1; s < blockDim.x; s *= 2) {
		// Reduces active threads by half each iterations
        if (threadIdx.x % (2 * s) == 0) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	// Threads 0 writes back result to main memory
    if (threadIdx.x == 0) {
		result[blockIdx.x] = partial_sum[0];
	}
}

// Sum reduction implementation that uses sequential threads
// Takes:
//  a:      Input array
//  result: Output array
// Returns:
//  NA
__global__ void sum_reduction_2(int *a, int *result) {
	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Allocate shared memory
	__shared__ int partial_sum[SHMEM_SIZE];

	// Load elements into shared memory
	partial_sum[threadIdx.x] = a[tid];
	__syncthreads();

	// Iterate log base two the block dimension times
	for (int s = 1; s < blockDim.x; s *= 2) {
		// Change the indexing to be sequential threads
		int index = 2 * s * threadIdx.x;

		// Each thread does work unless the index goes off the block
		if (index < blockDim.x) {
			partial_sum[index] += partial_sum[index + s];
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x == 0) {
		result[blockIdx.x] = partial_sum[0];
	}
}

// Sum reduction implementation that reduces bank conflicts
// Takes:
//  a:      Input array
//  result: Output array
// Returns:
//  NA
__global__ void sum_reduction_3(int *a, int *result) {
	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Allocate shared memory
	__shared__ int partial_sum[SHMEM_SIZE];

	// Load elements into shared memory
	partial_sum[threadIdx.x] = a[tid];
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
		result[blockIdx.x] = partial_sum[0];
	}
}

// Sum reduction implementation that packs extra work per-thread
// Takes:
//  a:      Input array
//  result: Output array
// Returns:
//  NA
__global__ void sum_reduction_4(int *a, int *result) {
	// Allocate shared memory
	__shared__ int partial_sum[SHMEM_SIZE];

	// Load elements AND do first add of reduction
	// Vector now 2x as long as number of threads, so scale i
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// Store first partial result instead of just the elements
	partial_sum[threadIdx.x] = a[i] + a[i + blockDim.x];
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
		result[blockIdx.x] = partial_sum[0];
	}
}

// Unroll last loop iteration
__device__ void warpReduce(volatile int* shmem_ptr, int t) {
	shmem_ptr[t] += shmem_ptr[t + 32];
	shmem_ptr[t] += shmem_ptr[t + 16];
	shmem_ptr[t] += shmem_ptr[t + 8];
	shmem_ptr[t] += shmem_ptr[t + 4];
	shmem_ptr[t] += shmem_ptr[t + 2];
	shmem_ptr[t] += shmem_ptr[t + 1];
}

// Sum reduction implementations that unrolls the final loop iteration
// Takes:
//  a:      Input array
//  result: Output array
// Returns:
//  NA
__global__ void sum_reduction_5(int *a, int *result) {
	// Allocate shared memory
	__shared__ int partial_sum[SHMEM_SIZE];

	// Load elements AND do first add of reduction
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// Store first partial result instead of just the elements
	partial_sum[threadIdx.x] = a[i] + a[i + blockDim.x];
	__syncthreads();

	// Start at 1/2 block stride and divide by two each iteration
	// Stop early (call device function instead)
	for (int s = blockDim.x / 2; s > 32; s >>= 1) {
		// Each thread does work unless it is further than the stride
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	if (threadIdx.x < 32) {
		warpReduce(partial_sum, threadIdx.x);
	}

	if (threadIdx.x == 0) {
		result[blockIdx.x] = partial_sum[0];
	}
}

// Launches perf test for sum reduction kernel
// Takes:
//  N: Number of iterations
//  D: Upper bound of vector size
// Returns:
//  NA
vector<float> launch_perf_test(int D, int N){
    // Grid size will be set each loop iteration
    int GRID_DIM;

    // Host pointers (performance test, no need to copy back result)
    int *h_a, *h_result;

    // Device pointers
    int *d_a, *d_result;

    // Start and stop event times
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Variables to collect timing information
    float exec_time;
    float total_time;

    // Vector of average execution times
    vector<float> times;

    // Increase the size of matrix by 2x each iterate
    for(int i = LOWER_BOUND; i <= D; i *= 2){
        // Re-initialize total_time each iteration
        total_time = 0;

        // Allocate space for each matrix 
        h_a = new int[i];
        h_result = new int[i];
        cudaMalloc(&d_a, i * sizeof(int));
        cudaMalloc(&d_result, i * sizeof(int));

        // Initialize the input matrices
        init_array(h_a, i);
        cudaMemcpy(d_a, h_a, i * sizeof(int), cudaMemcpyHostToDevice);

        // Calculate grid dimension and create launch parameters
        GRID_DIM = i / BLOCK_DIM;

        // Average execution time for "N" kernel runs
        for(int j = 0; j < N; j++){
            // Profile the start and end time of each kernel launch
            cudaEventRecord(start);
            // Uncomment which implementation you would like to profile
            //sum_reduction_1<<<GRID_DIM, BLOCK_DIM>>>(d_a, d_result);
            //sum_reduction_2<<<GRID_DIM, BLOCK_DIM>>>(d_a, d_result);
            //sum_reduction_3<<<GRID_DIM, BLOCK_DIM>>>(d_a, d_result);
            //sum_reduction_4<<<GRID_DIM / 2, BLOCK_DIM>>>(d_a, d_result);
            sum_reduction_5<<<GRID_DIM / 2, BLOCK_DIM>>>(d_a, d_result);
            cudaEventRecord(stop);
        
            // Make sure the cuda kernel gets launched
            cudaMemcpy(h_result, d_result, i * sizeof(int), cudaMemcpyDeviceToHost);
            
            // Synchronize on the event that goes to default stream 0
            cudaEventSynchronize(stop);

            // Get the time between events
            cudaEventElapsedTime(&exec_time, start, stop);

            // Add the time to the total
            total_time += exec_time;
        }

        // Add the average time to the vector
        times.push_back(total_time / N);

        // Free memory each iteration
        delete [] h_a;
        delete [] h_result;
        cudaFree(d_a);
        cudaFree(d_result);
    }

    return times;
}

