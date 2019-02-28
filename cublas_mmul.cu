// This program uses cuRAND and cuBLAS to perform matrix multiplication
// By: Nick from CoffeeBeforeArch

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>
#include <time.h>
#include <assert.h>

// Verify the result (Note, cuBLAS works in column-major format)
// Assumes contiguous memory is down columns, not across rows
void verify_result(float *a, float *b, float *c, int n){
    float temp = 0.0f;
    float epsilon = 0.00001f;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            temp = 0;
            for(int k = 0; k < n; k++){
                temp += a[k * n + i] * b[j * n + k];
            }
            assert(abs(c[j * n + i] - temp) < epsilon);
        }        
    }
}

int main(){
    // Problem size
    int n = 1 << 1024;
    size_t bytes = n * n * sizeof(float);

    // Declare pointers
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    
    // Allocate memory
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // RNG handle
    curandGenerator_t prng;

    // Create the Generator
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    // Initialize the matrices on the GPU
    curandGenerateUniform(prng, d_a, n * n);
    curandGenerateUniform(prng, d_b, n * n);

    // Scaling factors
    float alpha = 1;
    float beta = 0;

    // cuBLAS handle
    cublasHandle_t handle;
    
    // Create the handle
    cublasCreate(&handle);

    // Call SGEMM (alpha * a) * b + (beta * c)
    // Matrix definitions: op1(a) m*k, op1(b) k*n, c m*n
    // Function signature: handle, op1, op2, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_a, n, d_b, n, &beta, d_c, n);

    // Copy back the result, and original matrices
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Verify the result
    verify_result(h_a, h_b, h_c, n);

    return 0;
}
