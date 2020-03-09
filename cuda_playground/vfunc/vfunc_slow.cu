// This program shows off the defined way of using objects with virtual
// functions on the GPU using device-side dynamic allocation
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

// Simple way of not losing the pointer to our device-side dynamic
// allocation
__managed__ VFuncStruct *vf;

// Simple kernel that creates the struct on the GPU
__global__ void virtualFunctions() {
  // Create a new struct
  VFuncStruct *v_test = new VFuncStruct();
  v_test->a = 5;
  v_test->b = 10;
  // Test the virtual function
  v_test->printValues();
  // Save the pointer
  vf = v_test;
}

// Calls our virtual function from the struct created in the previous
// kernel
__global__ void callVFunc() {
  vf->printValues();
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
