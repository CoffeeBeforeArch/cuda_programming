// This program shows off a global memory implementation of a histogram
// kernel in CUDA
// By: Nick from CoffeeBeforeArch

#include <iostream>
#include <stdlib.h>

// Number of bins for our plot
#define BINS 7
#define DIV ((26 + BINS - 1) / BINS)
using namespace std;

// GPU kernel for computing a histogram
// Takes:
//  a: Problem array in global memory
//  result: result array
//  N: Size of the array
__global__ void histogram(char *a, int *result, int N){
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate the bin positions where threads are grouped together
    for(int i = tid; i < N; i += (gridDim.x * blockDim.x)){
        atomicAdd(&result[(a[i] - 'a') / DIV], 1);
    } 
}

// Initializes our input array
// Takes:
//  a: array of integers
//  N: Length of the array
void init_array(char *a, int N){
    srand(1);
    for(int i = 0; i < N; i++){
            a[i] = 'a' +  rand() % 26;
    }
}

int main(){
    // Declare our problem size
    int N = 1 << 18;
    size_t bytes_n = N * sizeof(int);

    // Allocate memory on the host
    char *h_a = new char[N];

    // Allocate space for the binned result
    int *h_result = new int[BINS]();
    size_t bytes_r = BINS * sizeof(int);

    // Initialize the array
    init_array(h_a, N);
    
    // Allocate memory on the device
    char *d_a;
    int *d_result;
    cudaMalloc(&d_a, bytes_n);
    cudaMalloc(&d_result, bytes_r);

    // Copy the array to the device
    cudaMemcpy(d_a, h_a, bytes_n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, h_result, bytes_n, cudaMemcpyHostToDevice);

    // Number of threads per threadblock
    int THREADS = 512;

    // Calculate the number of threadblocks
    int BLOCKS = N / THREADS / 4;

    // Launch the kernel
    histogram<<<BLOCKS, THREADS>>>(d_a, d_result, N);

    // Copy the result back
    cudaMemcpy(h_result, d_result, bytes_r, cudaMemcpyDeviceToHost);

    for(int i = 0; i < BINS; i++){
        cout << h_result[i] << " ";
    }
    cout << endl;
    return 0;
}
