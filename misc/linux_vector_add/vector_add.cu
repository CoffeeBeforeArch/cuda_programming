// This program shows off a basic vector add program run in Linux
// By: Nick from CoffeeBeforeArch

#include <stdlib.h>
#include <assert.h>
#include <iostream>

__global__ void vectorAdd(int *A, int *B, int *C, int N){
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure this is a valid thread
    if(tid < N){
        C[tid] = A[tid] + B[tid]; 
    }
}

int main(){
    // Number of elements
    int n = 1 << 16;

    // Size of memory
    size_t bytes = n * sizeof(int);

    // Host pointers
    int *h_a, *h_b, *h_c;

    // Device pointers
    int *d_a, *d_b, *d_c;

    // Allocate host memory
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    // Initialize vectors
    for(int i = 0; i < n; i++){
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }

    // Allocate device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy memory to the device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Calculate block and grid sizes
    int block_size = 256;
    int grid_size = n / block_size;

    // Launch kernel
    vectorAdd<<<grid_size,block_size>>>(d_a, d_b, d_c, n);

    // Synchronize on copy back of data
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Check result
    for(int i = 0; i < n; i++){
        assert(h_c[i] == h_a[i] + h_b[i]);
    }

    std::cout << "COMPLETE SUCCESFULLY :^)" << std::endl;
}
