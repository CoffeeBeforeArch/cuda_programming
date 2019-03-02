// This program calculates matrix multiplication (SGEMM) using cuBLAS
// By: Nick from CoffeeBeforeArch

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

void verify_solution(float *a, float *b, float *c, int n) {
	float temp;
	float epsilon = 0.001;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			temp = 0;
			for (int k = 0; k < n; k++) {
				temp += a[k * n + i] * b[j * n + k];
			}
			assert(fabs(c[j * n + i] - temp) < epsilon);
		}
	}
}

int main() {
	// Problem size
	int n = 1 << 10;
	size_t bytes = n * n * sizeof(float);

	// Declare pointers to matrices on device and host
	float *h_a, *h_b, *h_c;
	float *d_a, *d_b, *d_c;

	// Allocate memory
	h_a = (float*)malloc(bytes);
	h_b = (float*)malloc(bytes);
	h_c = (float*)malloc(bytes);
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	// Pseudo random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	// Set the seed
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

	// Fill the matrix with random numbers on the device
	curandGenerateUniform(prng, d_a, n*n);
	curandGenerateUniform(prng, d_b, n*n);

	// cuBLAS handle
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Scalaing factors
	float alpha = 1.0f;
	float beta = 0.0f;

	// Calculate: c = (alpha*a) * b + (beta*c)
	// (m X n) * (n X k) = (m X k)
	// Signature: handle, operation, operation, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_a, n, d_b, n, &beta, d_c, n);

	// Copy back the three matrices
	cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	// Verify solution
	verify_solution(h_a, h_b, h_c, n);

	printf("COMPLETED SUCCESSFULLY\n");

	return 0;
}