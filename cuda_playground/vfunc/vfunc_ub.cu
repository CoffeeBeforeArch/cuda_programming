// This program shows off a vfunc that works without dynamic allocation
// but wades into the murky waters of UB
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

// Simple kernel that fixes the virtual function pointer for the copied
// struct
__global__ void virtualFunctions(VFuncStruct *vf) {
  // Create a new struct based on the CPU struct
  VFuncStruct vf_local = VFuncStruct(*vf);
  // Copy the bits (which includes the corrected pointer)
  std::memcpy(vf, &vf_local, sizeof(VFuncStruct));
  // Test the new object
  vf->printValues();
}

// Calls our virtual function from the struct fixed in the previous
// kernel
__global__ void callVFunc(VFuncStruct *vf) {
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

  // Execute the kernel, and wait for it to finish
  virtualFunctions<<<1, 1>>>(vf);
  cudaDeviceSynchronize();
  
  // Use the struct again to make sure we did things correctly
  callVFunc<<<1, 1>>>(vf);
  cudaDeviceSynchronize();
  
  return 0;
}
