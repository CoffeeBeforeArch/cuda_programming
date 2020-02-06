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
// Indexing must account for the CUBLAS operating on column-major data
void verify_solution(float *a, float *b, float *c, int M, int N, int K) {
  // Tolerance for our result (floats are imperfect)
  float epsilon = 0.001f;

  // For every row...
  for (int row = 0; row < M; row++) {
    // For every column
    for (int col = 0; col < N; col++) {
      // For every element in the row-col pair...
      float temp = 0;
      for (int i = 0; i < K; i++) {
        temp += a[row + M * i] * b[col * K + i];
      }

      // Check to see if the difference falls within our tolerance
      assert(fabs(c[col * M + row] - temp) <= epsilon);
    }
  }
}

int main() {
  // Dimensions for our matrices
  // MxK * KxN = MxN
  const int M = 1 << 9;
  const int N = 1 << 8;
  const int K = 1 << 7;

  // Pre-calculate the size (in bytes) of our matrices
  const size_t bytes_a = M * K * sizeof(float);
  const size_t bytes_b = K * N * sizeof(float);
  const size_t bytes_c = M * N * sizeof(float);

  // Vectors for the host data
  std::vector<float> h_a(M * K);
  std::vector<float> h_b(K * N);
  std::vector<float> h_c(M * N);
  
  // Allocate device memory
  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes_a);
  cudaMalloc(&d_b, bytes_b);
  cudaMalloc(&d_c, bytes_c);

  // Pseudo random number generator
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

  // Set the seed
  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

  // Fill the matrix with random numbers on the device
  curandGenerateUniform(prng, d_a, M * K);
  curandGenerateUniform(prng, d_b, K * M);

  // cuBLAS handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Scalaing factors
  float alpha = 1.0f;
  float beta = 0.0f;

  // Calculate: c = (alpha*a) * b + (beta*c)
  // MxN = MxK * KxN
  // Signature: handle, operation, operation, M, N, K, alpha, A, lda, B, ldb,
  // beta, C, ldc
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_a, M, d_b, K,
              &beta, d_c, M);

  // Copy back the three matrices
  cudaMemcpy(h_a.data(), d_a, bytes_a, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_b.data(), d_b, bytes_b, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost);

  // Verify solution
  verify_solution(h_a.data(), h_b.data(), h_c.data(), M, N, K);
  std::cout << "COMPLETED SUCCESSFULLY\n";

  // Free our memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
