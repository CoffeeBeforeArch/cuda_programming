// This file includes a naive and shared memory kernels for
// matrix multiplication
// By: Nick from CoffeeBeforeArch

__global__ void matrixMulAligned(int *a, int *b, int *c, int n) {
	// Compute each thread's row
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	// Compute each thread's column
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int temp_sum = 0;
	// Boundary protection
	if ((row < n) && (col < n)) {
		// Iterate over row, and down column
		for (int k = 0; k < n; k++) {
			// Accumulate result for a single element
			temp_sum += a[k * n + row] * b[k * n + col];
		}
		// Assign result
		c[row * n + col] = temp_sum;
	}

}

// Static shmem calculation for convenience (Int 16x16 matrix)
#define SHMEM_SIZE 16 * 16 * 4

__global__ void tiledMatrixMul(int *a, int *b, int *c, int n, int tile_size) {
	// Two statically-sized pieces of shared memory
	__shared__ int A[SHMEM_SIZE];
	__shared__ int B[SHMEM_SIZE];

	// Shorten these parameters for clean re-use
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Calculate global row and column positions for this thread
	int row = by * tile_size + ty;
	int col = bx * tile_size + tx;

	// Intermediate sum for element being written
	int temp_val = 0;

	// Sweep tiles over entire matrix
	for (int i = 0; i < (n / tile_size); i++) {
		/*
			Every thread in a threadblock loads one element into shared memory
			The element location in shared memory corresponds to the thread's
			position in the threadblock (e.g. thread [0, 0] loads for
			A[0 * tile_size + 0], and B[0 * tile_size + 0].)

			Explanation of indexing parameters
			For A:
						row*n: Indexes the global row for this thread (loop-invariant)
				  i*tile_size: Indexes the new set of columns each iteration
						   tx: Indexes the column within that set
			for B:
				i*tile_size*n: Indexes the next set of rows each iteration
						 ty*n: Indexes the row within that set
						  col: Indexes the global column (loop-invariant)
		*/
		A[(ty * tile_size) + tx] = a[row * n + (i * tile_size + tx)];
		B[(ty * tile_size) + tx] = b[(i * tile_size * n + ty * n) + col];

		// Ensure all threads have loaded their data before proceeding
		__syncthreads();

		// Calculate all temp values for this tile
		for (int j = 0; j < tile_size; j++) {
			temp_val += A[(ty * tile_size) + j] * B[(j * tile_size) + tx];
		}

		// Ensure some threads don't progress and stomp current shared memory values
		__syncthreads();
	}
	c[(row * n) + col] = temp_val;
}