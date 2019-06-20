// This program shows off the basics of handling non-perfect input
// sizes in CUDA
// By: Nick from CoffeeBeforeArch

#include <stdlib.h>

using namespace std;

// Naive vector addition kernel expecting perfect inputs
__global__ void vectorAdd(int *a, int *b, int *c, int N){
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid < N){
        // Perform vector addition
        c[tid] = a[tid] + b[tid];
    }
}

// Naive matrix multiplication kernel expecting perfect inputs
__global__ void matrixMul(int *a, int *b, int *c, int N){
    // Calculate row and column
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < N){
        // Temp variable for accumulating result
        int temp = 0;

        // Traverse row and column for this matrix
        for(int i = 0; i < N; i++){
            temp += a[row * N + i] * b[i * N + col];
        }

        // Write back the result
        c[row * N + col] = temp;
    
    }
}

// Matrix multiplication kernel with non-square matrix
__global__ void better_matrixMul(int *a, int *b, int *c, int M, int N, int K){
    // Calculate row and column
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < K){
        // Temp variable for accumulating result
        int temp = 0;

        for(int i = 0; i < N; i++){
            temp += a[row * N + i] * b[i * K + col];
        }

        c[row * K + col] = temp;
    }
}

int main(){
    // Problem size
    int v = 123456;
    int M = 86;
    int N = 530;
    int K = 74;

    // Host pointers and memory allocation
    int *h_v_a = new int[v];
    int *h_v_b = new int[v];
    int *h_v_c = new int[v];
    int *h_m_a = new int[M * N];
    int *h_m_b = new int[N * K];
    int *h_m_c = new int[M * K];

    // Device pointers and memory allocation
    int *d_v_a, *d_v_b, *d_v_c;
    int *d_m_a, *d_m_b, *d_m_c;
    cudaMalloc(&d_v_a, v * sizeof(int));
    cudaMalloc(&d_v_b, v * sizeof(int));
    cudaMalloc(&d_v_c, v * sizeof(int));
    cudaMalloc(&d_m_a, M * N * sizeof(int));
    cudaMalloc(&d_m_b, N * K * sizeof(int));
    cudaMalloc(&d_m_c, M * K * sizeof(int));

    // # of threads in each block dimension
    int THREADS_V = 256;
    int THREADS_M = 16;

    // Number of blocks for vector
    int BLOCKS_V = (v + THREADS_V - 1) / THREADS_V;

    // Number of blocks for matrix
    int BLOCKS_ROWS = (M + THREADS_M - 1) / THREADS_M;
    int BLOCKS_COLS = (K + THREADS_M - 1) / THREADS_M;
    
    // 2D block size for matrix
    dim3 threads_m(THREADS_M, THREADS_M);

    // 2D grid for matrix
    dim3 grid_m(BLOCKS_COLS, BLOCKS_ROWS);

    // Copy the data over
    cudaMemcpy(d_v_a, h_v_a, v * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_b, h_v_b, v * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m_a, h_m_a, M * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m_b, h_m_b, N * K * sizeof(int), cudaMemcpyHostToDevice);

    // Call the kernels and copy the data back
    vectorAdd<<<BLOCKS_V, THREADS_V>>>(d_v_a, d_v_b, d_v_c, v);
    cudaMemcpy(h_v_c, d_v_c, v * sizeof(int), cudaMemcpyDeviceToHost);
    better_matrixMul<<<grid_m, threads_m>>>(d_m_a, d_m_b, d_m_c, M, N, K);
    cudaMemcpy(h_m_c, d_m_c, M * K * sizeof(int), cudaMemcpyDeviceToHost);

    return 0;
}
