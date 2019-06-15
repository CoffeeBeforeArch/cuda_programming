// This header file contains the utility functions and different CUDA
// kernels of matrix multiplication overviewed in CUDA Crash Course
// By: Nick from CoffeeBeforeArch

// For rand()
#include <stdlib.h>

// To store average execution times in
#include <vector>

// Headers to get Ale to shut up
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Lower bound of threads to launch
#define LOWER_BOUND 256

// Static shmem calculation for convenience (Int 16x16 matrix)
#define SHMEM_SIZE 256 * 4

using namespace std;

// Matrix initialization function
// Takes:
//  m:  Pointer to the matrix
//  N:  Dimension of the matrix (assumed to be square)
// Returns:
//  NA
void init_matrix(int *m, int N){
    for(int i = 0; i < N * N; i++){
        m[i] = rand() % 1000;
    }
}

// Naive matrix multiplication kernel
// Takes:
//  a:  Pointer to input matrix "a"
//  b:  Pointer to input matrix "b"
//  c:  Pointer to output matrix "c"
//  N:  Dimensions of the matrix (assumed to be square)
// Returns:
//  NA
__global__ void naive_mmul(int *a, int *b, int *c, int N){
    // Row and column of the element of "c" this thread computes
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    // Accumulate result in a temporary variable
    int temp = 0;

    // Loop over 1 row of "a", and column of "b"
    for(int i = 0; i < N; i++){
        // Accumulate the partial results in "temp"
        temp += a[row * N + i] * b[i * N + col];
    }

    // Store the result in the output matrix
    c[row * N + col] = temp;
}

// Read-aligned matrix multiplication kernel
// Takes:
//  a:  Pointer to input matrix "a"
//  b:  Pointer to input matrix "b"
//  c:  Pointer to output matrix "c"
//  N:  Dimensions of the matrix (assumed to be square)
// Returns:
//  NA
__global__ void aligned_mmul(int *a, int *b, int *c, int n) {
    // Row and column of the element of "c" this thread computes
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Accumulate result in a temporary variable
	int temp_sum = 0;
	
    // Iterate over row, and down column
	for (int i = 0; i < n; i++) {
	    // Accumulate result for a single element
		temp_sum += a[i * n + row] * b[i * n + col];
	}
    
    // Assign result
	c[row * n + col] = temp_sum;
}

// Cache tiled matrix multiplication kernel
// Takes:
//  a:  Pointer to input matrix "a"
//  b:  Pointer to input matrix "b"
//  c:  Pointer to output matrix "c"
//  N:  Dimensions of the matrix (assumed to be square)
// Returns:
//  NA
__global__ void tiled_mmul(int *a, int *b, int *c, int N) {
	// Two statically-sized pieces of shared memory
	__shared__ int A[SHMEM_SIZE];
	__shared__ int B[SHMEM_SIZE];

	// Shorten these parameters for clean re-use
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Calculate global row and column positions for this thread
	int row = by * blockDim.y + ty;
	int col = bx * blockDim.x + tx;

	// Intermediate sum for element being written
	int temp_val = 0;

	// Sweep tiles over entire matrix
	for (int i = 0; i < (N / blockDim.x); i++) {
		// Load sub-matrices into shared memory
        A[(ty * blockDim.x) + tx] = a[row * N + (i * blockDim.x + tx)];
		B[(ty * blockDim.x) + tx] = b[(i * blockDim.x * N + ty * N) + col];

		// Ensure all threads have loaded their data before proceeding
		__syncthreads();

		// Calculate all temp values for this tile
		for (int j = 0; j < blockDim.x; j++) {
			temp_val += A[(ty * blockDim.x) + j] * B[(j * blockDim.x) + tx];
		}

		// Ensure some threads don't progress and stomp current shared memory values
		__syncthreads();
	}
	
    // Write back to the result matrix
    c[row * N + col] = temp_val;
}


// Launches performance test for matrix multiplication kernel
// Takes:
//  D: Upper bound of square matrix dimension to test
//  N: Number if iterations to average over:
// Returns:
//  vector<float> of average execution times
vector<float> launch_mmul(int D, int N){
    // Set static number of threads per threadblock
    int BLOCK_DIM = 16;

    // Grid size will be set each loop iteration
    int GRID_DIM;

    // Host pointers (performance test, no need to copy back result)
    int *h_a, *h_b, *h_c;

    // Device pointers
    int *d_a, *d_b, *d_c;

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
        h_b = new int[i * i];
        h_c = new int[i * i];
        cudaMalloc(&d_a, i * i * sizeof(int));
        cudaMalloc(&d_b, i * i * sizeof(int));
        cudaMalloc(&d_c, i * i * sizeof(int));

        // Initialize the input matrices
        init_matrix(h_a, i);
        init_matrix(h_b, i);
        cudaMemcpy(d_a, h_a, i * i * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, i * i * sizeof(int), cudaMemcpyHostToDevice);

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

