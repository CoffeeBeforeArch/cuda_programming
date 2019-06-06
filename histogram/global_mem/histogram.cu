// This program shows off a global memory implementation of a histogram
// kernel in CUDA
// By: Nick from CoffeeBeforeArch

#include <iostream>
#include <stdlib.h>

using namespace std;

// GPU kernel for computing a histogram
// Takes:
//  a: Problem array in global memory
//  result: result array
//  N: Size of the array
__global__ void histogram(int *a, int *result, int N){
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

}

// Initializes our input array
// Takes:
//  a: array of integers
//  N: Length of the array
void init_array(int *a, int N){
    for(int i = 0; i < N; i++){
            a[i] = rand() % 100;
    }
}

int main(){
    // Declare our problem size
    int N = 1 << 16;
    size_t bytes = N * sizeof(int);

    // Allocate memory on the host
    int *h_a = new int[N];
    int *h_result = new int[N]();

    // Initialize the array
    init_array(h_a, N);

    // Allocate memory on the device
    int *d_a;
    int *d_result;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_result, bytes);

    // Copy the array to the device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

    // Copy the result back
    cudaMemcpy(h_result, d_result, bytes, cudaMemcpyDeviceToHost);

    return 0;
}
