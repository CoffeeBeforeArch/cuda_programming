// Simple example of copying pointers
// By: Nick from CoffeeBeforeArch

#include <cstdio>
#include <vector>

__global__ void print_kernel(int **d_vec_of_ptrs, int num_ptrs,
                             int num_elements) {
  // Get the thread ID
  auto tid = threadIdx.x;

  // Print the thread ID and pointer address
  if (tid < num_ptrs)
    printf("threadIdx.x - %d, Ptr %p\n", tid, d_vec_of_ptrs[tid]);
}

int main() {
  // Number of ptrs
  int num_ptrs = 1 << 5;

  // Number of elements per pointer
  int num_elements = 1 << 10;

  // Create a host-side vector of ptrs
  std::vector<int *> h_vec_of_ptrs(num_ptrs);

  // Allocate the space behind each ptr
  for (auto &ptr : h_vec_of_ptrs) cudaMalloc(&ptr, num_elements * sizeof(int));

  // Print out the pointers on the host
  for (const auto ptr : h_vec_of_ptrs)
    printf("Device pointer from host - %p\n", ptr);

  // Create a device-side array of ptrs
  int **d_vec_of_ptrs;
  cudaMalloc(&d_vec_of_ptrs, num_ptrs * sizeof(int *));

  // Copy the the vec of device ptrs to the device
  cudaMemcpy(d_vec_of_ptrs, h_vec_of_ptrs.data(), num_ptrs * sizeof(int *),
             cudaMemcpyHostToDevice);

  // Call the kernel
  print_kernel<<<1, num_ptrs>>>(d_vec_of_ptrs, num_ptrs, num_elements);

  // Wait for the kernel to finish
  cudaDeviceSynchronize();

  return 0;
}
