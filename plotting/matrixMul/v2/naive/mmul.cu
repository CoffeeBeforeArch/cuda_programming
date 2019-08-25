// This program computes matrix multiplication
// By: Nick from CoffeeBeforeArch

#include <cstdlib>

// Matrix Multiplication kernel
// Optimizations:
__global__ void matrixMul(float *a, float *b, float *c, int N){
    // Calculate the row and column for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if((row < N) && (col < N)){
        // Each thread computes one element
        for(int i = 0; i < N; i++){
            c[row * N + col] += a[row * N + i] * b[i * N + col];
        }
    }
}

// Initialize a matrix with random numbers
void init_matrix(float *m, int N){
    for(int i = 0; i < N * N; i++){
        m[i] = rand() % 100;
    }
}

int main(){
    // Problem size
    int N = 1 << 14;
    size_t bytes = N * N * sizeof(float);

    // Allocate host memory (make sure C is zeroed)
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)calloc(N * N, sizeof(float));
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
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
    int threads = 16;
    int blocks = (N + threads -1) / threads;
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);

    // Call our kernel
    matrixMul<<<BLOCKS, THREADS>>>(d_a, d_b, d_c, N);
    
    // Copy data back to the host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    return 0;
}
