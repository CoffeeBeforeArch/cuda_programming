// This program shows off a vfunc that will not work on the GPU
// By: Nick from CoffeeBeforeArch

#include <cstdio>
#include <cstring>

// Simple struct that contains a non-accessable pointer and two ints
struct VFuncStruct {
  // Some random data members
  int a;
  int b;
  
  // Must be marked __host__ __device__ to work on both CPU and GPU
  virtual __host__ __device__ void printValues() {
    printf("a = %d, b = %d\n", a, b);
  }
};

// Simple kernel that tries to call the vfunc from the copied object
__global__ void virtualFunctions(VFuncStruct *vf) {
  // Unsurprisingly, this fails.
  // The pointer stored in the struct is to CPU memory, and will not
  // resolve on the GPU.
  vf->printValues();
}

int main() {
  // Create a struct
  VFuncStruct vf_host{};
  vf_host.a = 5;
  vf_host.b = 10;
  
  // Reserve space for the struct on the GPU
  VFuncStruct *vf;
  cudaMalloc(&vf, sizeof(VFuncStruct));
  
  // Copy the struct to the GPU
  cudaMemcpy(vf, &vf_host, sizeof(VFuncStruct), cudaMemcpyHostToDevice);

  // Execute the kernel, and wait for execution to finish
  virtualFunctions<<<1, 1>>>(vf);
  cudaDeviceSynchronize();

  return 0;
}
