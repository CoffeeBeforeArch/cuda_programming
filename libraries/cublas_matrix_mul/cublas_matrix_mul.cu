// This program calculates matrix multiplication (SGEMM) using cuBLAS
// By: Nick from CoffeeBeforeArch

#include <cublas_v2.h>
#include <curand.h>
#include <cassert>
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

// Verify our result on the CPU
void verify_solution(float *a, float *b, float *c, int n) {
  float epsilon = 0.001;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      float temp = 0;
      for (int k = 0; k < n; k++) {
        temp += a[k * n + i] * b[j * n + k];
      }
      assert(fabs(c[j * n + i] - temp) < epsilon);
    }
  }
}

int main() {
  // Dimensions for our matrices
  // MxN * NxK = MxK
  int M = 1 << 8;
  int N = 1 << 9;
  int K = 1 << 10;

  // Pre-calculate the size (in bytes) of our matrices
  size_t bytes_a = M * N * sizeof(float);
  size_t bytes_b = N * K * sizeof(float);
  size_t bytes_c = M * K * sizeof(float);

  // Vectors for the host data
  std::vector<float> h_a(M * N);
  std::vector<float> h_b(N * K);
  std::vector<float> h_c(M * K);

  // Allocate device memory
  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // Pseudo random number generator
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

  // Set the seed
  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

  // Fill the matrix with random numbers on the device
  curandGenerateUniform(prng, d_a, M * N);
  curandGenerateUniform(prng, d_b, N * K);

  // cuBLAS handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Scalaing factors
  float alpha = 1.0f;
  float beta = 0.0f;

  // Calculate: c = (alpha*a) * b + (beta*c)
  // MxK = MxN * NxK
  // Signature: handle, operation, operation, m, n, k, alpha, A, lda, B, ldb,
  // beta, C, ldc
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_a, M, d_b, N,
              &beta, d_c, M);

  // Copy back the three matrices
  cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  // Verify solution
  verify_solution(h_a, h_b, h_c, n);
  std::cout << "COMPLETED SUCCESSFULLY\n";

  // Free our memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
