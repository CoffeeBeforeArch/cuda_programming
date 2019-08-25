// This program computes matrix multiplication
// By: Nick from CoffeeBeforeArch

#include <cstdlib>
#include <cassert>

// Matrix Multiplication kernel
// Optimizations:
//  Accumulate partial results in a temporary variable
//  Ensure all threads in warps access consecutive memory
__global__ void matrixMul(int *a, int *b, int *c, int N){
    // Calculate the row and column for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if((row < N) && (col < N)){
        // Each thread computes one element
        int tmp = 0;
        for(int i = 0; i < N; i++){
            tmp += a[row * N + i] * b[i * N + col];
        }

        // Write back the tmp result
        c[row * N + col] = tmp;
    }
}

// Initialize a matrix with random numbers
void init_matrix(int *m, int N){
    for(int i = 0; i < N * N; i++){
        m[i] = rand() % 100;
    }
}

// Verify result (only needs to be run once to ensure functional
// correctness)
void verify_result(int *a, int *b, int *c, int N){
    int tmp = 0;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < N; k++){
                tmp += a[i * N + k] * b[k * N + j];
            }
            assert(c[i * N + j] == tmp);
        }
    }
}

int main(){
    // Problem size
    int N = 1 << 14;
    size_t bytes = N * N * sizeof(int);

    // Allocate host memory (make sure C is zeroed)
    int *h_a = (int*)malloc(bytes);
    int *h_b = (int*)malloc(bytes);
    int *h_c = (int*)malloc(bytes);
    
    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Initialized host-side matrices
    init_matrix(h_a, N * N);
    init_matrix(h_b, N * N);

    // Copy the matrices over
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, bytes, cudaMemcpyHostToDevice);

    // Set up the CTA and Grid Dimensions
    int threads = 32;
    int blocks = (N + threads -1) / threads;
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);

    // Call our kernel
    matrixMul<<<BLOCKS, THREADS>>>(d_a, d_b, d_c, N);
    
    // Copy data back to the host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Verify the result (comment out after confirmed)
    //verify_result(h_a, h_b, h_c, N);

    return 0;
}
