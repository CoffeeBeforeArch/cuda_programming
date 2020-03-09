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

// Simple way of not losing the pointer to our device-side dynamic
// allocation
__managed__ VFuncStruct *vf;

// Simple kernel that creates the struct on the GPU
__global__ void virtualFunctions() {
  // Create a new struct
  VFuncStruct *v_test = new VFuncStruct();
  // Test the virtual function
  v_test->getSize();
  // Save the pointer
  vf = v_test;
}

// Calls our virtual function from the struct created in the previous
// kernel
__global__ void callVFunc() {
  vf->getSize();
}

int main() {
  // Execute the kernel, and wait for it to finish
  virtualFunctions<<<1, 1>>>();
  cudaDeviceSynchronize();
  
  // Use the struct again to make sure we did things correctly
  callVFunc<<<1, 1>>>();
  cudaDeviceSynchronize();
  
  return 0;
}
