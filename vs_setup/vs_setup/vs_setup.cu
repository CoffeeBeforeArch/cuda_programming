// This program is an example to show the V.S. setup for CUDA
// By: Nick from CoffeeBeforeArch

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void test() {
	// Empty Kernel
}

int main() {
	test <<<1, 1 >>>();
	return 0;
}