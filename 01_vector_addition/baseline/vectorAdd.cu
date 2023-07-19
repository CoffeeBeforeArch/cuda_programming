// This program computes the sum of two vectors of length N
// By: Nick from CoffeeBeforeArch
// revised by Topsy

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

// CUDA kernel for vector addition
// __global__ means this is called from the CPU, and runs on the GPU
/**
 * @brief                  Vector add
 *
 * @param d_a                array a, __restroct is used to indicate that a is not aliased with any other pointers in the code, no other pointer point the same memeory of a
 * @param d_b                array b
 * @param d_c                result array
 * @param N                the number of element
 * @return __global__
 */
__global__ void vectorAdd(const int *__restrict d_a, const int *__restrict d_b,
                          int *__restrict d_c, int N)
{
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Boundary check
    if (tid < N)
        d_c[tid] = d_a[tid] + d_b[tid];
}

// Check vector add result
void verify_result(std::vector<int> &a, std::vector<int> &b,
                   std::vector<int> &c)
{
    for (int i = 0; i < a.size(); i++)
    {
        assert(c[i] == a[i] + b[i]);
    }
}

int main()
{
    // Array size of 2^16 (65536 elements)
    constexpr int N = 1 << 16;
    constexpr size_t bytes = sizeof(int) * N;

    // Vectors for holding the host-side (CPU-side) data
    std::vector<int> a;
    a.reserve(N);
    std::vector<int> b;
    b.reserve(N);
    std::vector<int> c;
    c.reserve(N);

    // Initialize random numbers in each array
    for (int i = 0; i < N; i++)
    {
        a.push_back(rand() % 100);
        b.push_back(rand() % 100);
    }

    // Allocate memory on the device
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    // Threads per CTA (1024)
    // CTA stands for "Cooperative Thread Array."  -> block
    int NUM_THREADS = 1 << 10;

    // CTAs per Grid
    // We need to launch at LEAST as many threads as we have elements
    // This equation pads an extra CTA to the grid if N cannot evenly be divided
    // by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
    int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    // Launch the kernel on the GPU
    // Kernel calls are asynchronous (the CPU program continues execution after
    // call, but no necessarily before the kernel finishes)
    // specify the number of threads per CTA (block) and the number of CTAs per grid.
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);

    // Copy sum vector from device to host
    // cudaMemcpy is a synchronous operation, and waits for the prior kernel
    // launch to complete (both go to the default stream in this case).
    // Therefore, this cudaMemcpy acts as both a memcpy and synchronization
    // barrier.
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // Check result for errors
    verify_result(a, b, c);

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "COMPLETED SUCCESSFULLY\n";

    return 0;
}
