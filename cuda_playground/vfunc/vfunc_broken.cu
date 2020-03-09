// This program shows off a vfunc that will not work on the GPU
// By: Nick from CoffeeBeforeArch

#include <cstdio>
#include <cstring>

// Simple struct that should only contain a non-accessable vfunc-pointer
struct VFuncStruct {
  // Must be marked __host__ __device__ to work on both CPU and GPU
  virtual __host__ __device__ void getSize() {
    printf("Sizeof vfunc struct is %lu\n", sizeof(VFuncStruct));
  }
};

// Simple kernel that tries to call the vfunc from the copied object
__global__ void virtualFunctions(VFuncStruct *vf) {
  // Unsurprisingly, this fails.
  // The pointer stored in the struct is to CPU memory, and will not
  // resolve on the GPU.
  vf->getSize();
}

int main() {
  // Create a struct
  VFuncStruct vf_host{};
  
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
