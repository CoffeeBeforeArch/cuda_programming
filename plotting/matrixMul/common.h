// This header file contains the utility functions and different CUDA
// kernels of matrix multiplication overviewed in CUDA Crash Course
// By: Nick from CoffeeBeforeArch

// For rand()
#include <stdlib.h>

// To store average execution times in
#include <vector>

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

// Launches the naive matrix multiplication kernel
// Takes:
//  D: Upper bound of square matrix dimension to test
//  N: Number if iterations to average over:
// Returns:
//  vector<float> of average execution times
vector<float> launch_naive_mmul(int D, int N){
    // Set static number of threads per threadblock
    int BLOCK_DIM = 256;

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
    for(int i = BLOCK_DIM; i <= D; i *= 2){
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
            naive_mmul<<<grid, block>>>(d_a, d_b, d_c, i);
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
