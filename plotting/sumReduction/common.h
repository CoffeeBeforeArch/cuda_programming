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

#define SIZE 256
#define SHMEM_SIZE 256 * 4

// Array initalization function
// Takes:
//  a: Pointer to the array
//  N: Number of array elements
// Returns:
//  NA
void init_array(int *a, int N) {
    for (int i = 0; i < N; i++) {
        a[i] = 1;//rand() % 10;
    }
}

__global__ void sum_reduction(int *a, int *result) {
	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Allocate shared memory
	__shared__ int partial_sum[SHMEM_SIZE];

	// Load elements into shared memory
	partial_sum[threadIdx.x] = a[tid];
	__syncthreads();

	// Iterate of log base 2 the block dimension
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

// Launches perf test for sum reduction kernel
// Takes:
//  N: Number of iterations
//  D: Upper bound of vector size
// Returns:
//  NA
vector<float> launch_perf_test(int N, int D){
    // Grid size will be set each loop iteration
    int GRID_DIM;

    // Host pointers (performance test, no need to copy back result)
    int *h_a, *h_result;

    // Device pointers
    int *d_a, d_result;

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
    for(int i = LOWER_BOUND; i <= D; i += 128){
        // Re-initialize total_time each iteration
        total_time = 0;

        // Allocate space for each matrix 
        h_a = new int[i * i];
        h_result = new int[i];
        cudaMalloc(&d_a, i * sizeof(int));
        cudaMalloc(&d_result, i * sizeof(int));

        // Initialize the input matrices
        init_array(h_a, i);
        cudaMemcpy(d_a, h_a, i * sizeof(int), cudaMemcpyHostToDevice);

        // Calculate grid dimension and create launch parameters
        GRID_DIM = i / BLOCK_DIM;
        dim3 grid(GRID_DIM, GRID_DIM);
        dim3 block(BLOCK_DIM, BLOCK_DIM);

        // Average execution time for "N" kernel runs
        for(int j = 0; j < N; j++){
            // Profile the start and end time of each kernel launch
            cudaEventRecord(start);
            // Uncomment which implementation you would like to profile
            //naive_mmul<<<grid, block>>>(d_a, d_b, d_c, i);
            //aligned_mmul<<<grid, block>>>(d_a, d_b, d_c, i);
            tiled_mmul<<<grid, block>>>(d_a, d_b, d_c, i);
            cudaEventRecord(stop);
        
            // Make sure the cuda kernel gets launched
            cudaMemcpy(h_c, d_c, i * i * sizeof(int), cudaMemcpyDeviceToHost);
            
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
        delete [] h_b;
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    return times;

}
