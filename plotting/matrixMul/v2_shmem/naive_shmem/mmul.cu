// This program is an optimized version of matrix multiplication
// By: Nick from CoffeeBeforeArch

#include <cstdlib>
#include <cassert>

// Static shmem calculation for convenience (Int 16x16 matrix)
#define SHMEM_SIZE 16 * 16

__global__ void matrixMul(int *a, int *b, int *c, int N) {
	// Two statically-sized pieces of shared memory
	__shared__ int A[SHMEM_SIZE];
	__shared__ int B[SHMEM_SIZE];

	// Shorten these parameters for clean re-use
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int dim = blockDim.x;

	// Calculate global row and column positions for this thread
	int row = blockIdx.y * dim + ty;
	int col = blockIdx.x * dim + tx;

	// Intermediate sum for element being written
	int temp_val = 0;

	// Sweep tiles over entire matrix
	for (int i = 0; i < (N / dim); i++) {
		A[(ty * dim) + tx] = a[row * N + (i * dim + tx)];
		B[(ty * dim) + tx] = b[(i * dim * N + ty * N) + col];

		// Ensure all threads have loaded their data before proceeding
		__syncthreads();

		// Calculate all temp values for this tile
		for (int j = 0; j < dim; j++) {
			temp_val += A[(ty * dim) + j] * B[(j * dim) + tx];
		}

		// Ensure some threads don't progress and stomp current shared memory values
		__syncthreads();
	}
    
    // Write back the result
	c[(row * N) + col] = temp_val;
}

void check_answer(int *a, int *b, int *c, int n) {
	int tmp;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			tmp = 0;
			for (int k = 0; k < n; k++) {
				 tmp += a[i * n + k] * b[k * n + j];
			}
            assert(tmp == c[i * n + j]);
		}
	}
}

void init_matrix(int *m, int N) {
	for (int i = 0; i < N * N; i++) {
	    m[i] = rand() % 100;
	}
}

int main() {
	// Set the problem size
    int N = 1 << 14;
	size_t bytes = N * N * sizeof(int);

	// Host matrix pointers
	int *h_a, *h_b, *h_c;

	// Device matrix pointers
	int *d_a, *d_b, *d_c;

	// Allocate host memory
	h_a = (int*)malloc(bytes);
	h_b = (int*)malloc(bytes);
	h_c = (int*)malloc(bytes);

	// Allocate device memory
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	// Initialize matrices
	init_matrix(h_a, N);
	init_matrix(h_b, N);

	// Copy matrices to the device
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

	// Threads per block (in both x and y dimensions)
	int BLOCK_SIZE = 16;

	// Blocks in each dimension
	int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	// Use dim3 objects for 2-D grids and threadblocks
	dim3 grid(GRID_SIZE, GRID_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	// Launch kernel
	matrixMul <<<grid, threads>>> (d_a, d_b, d_c, N);

	// Copy result back from device
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);	

	// Verify the result
	//check_answer(h_a, h_b, h_c, N);

	// Free host memory
	free(h_a);
	free(h_b);
	free(h_c);

	// Free device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}
