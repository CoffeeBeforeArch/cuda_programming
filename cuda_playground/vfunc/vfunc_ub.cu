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

// Simple kernel that fixes the virtual function pointer for the copied
// struct
__global__ void virtualFunctions(VFuncStruct *vf) {
  // Create a new struct based on the CPU struct
  VFuncStruct vf_local = VFuncStruct(*vf);
  // Copy the bits (which includes the corrected pointer)
  std::memcpy(vf, &vf_local, sizeof(VFuncStruct));
  // Test the new object
  vf->getSize();
}

// Calls our virtual function from the struct fixed in the previous
// kernel
__global__ void callVFunc(VFuncStruct *vf) {
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

  // Execute the kernel, and wait for it to finish
  virtualFunctions<<<1, 1>>>(vf);
  cudaDeviceSynchronize();
  
  // Use the struct again to make sure we did things correctly
  callVFunc<<<1, 1>>>(vf);
  cudaDeviceSynchronize();
  
  return 0;
}
