// This program shows how to get a number of device properties from API
// calls in CUDA
// By: Nick from CoffeeBeforeArch

#include <iostream>
#include <cuda_runtime.h>

using namespace std;

int main(){
    // We can get the number of devices in the system
    int device_count;
    cudaGetDeviceCount(&device_count);
    cout << "There are " << device_count << " GPU(s) in the system" << endl;

    for(int i = 0; i < device_count; i++){
        // We can se the device if we have multiple GPUs
        cudaSetDevice(i);

        // We can also get the properties from the GPU
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, i);
        cout << "Device " << i << " is a " << device_prop.name << endl;

        // We can also get information about the driver and runtime
        int driver;
        int runtime;
        cudaDriverGetVersion(&driver);
        cudaRuntimeGetVersion(&runtime);
        cout << "Driver: " << driver << " Runtime: " << runtime << endl;

        // We can compare this against the device capabilities
        cout << "CUDA capability: " << device_prop.major << "." <<
            device_prop.minor << endl;

        // We can also get the amount of global memory
        cout << "Global memory in GB: " <<
            device_prop.totalGlobalMem / (1 << 30) << endl;

        // The number of SMs
        cout << "Number of SMs: " << device_prop.multiProcessorCount <<
            endl;

        // The frequency
        cout << "Max clock rate: " << device_prop.clockRate * 1e-6 <<
            "GHz" << endl;

        // The L2 cache size
        cout << "The L2 cache size in MB: " <<
            device_prop.l2CacheSize / (1 << 20) << endl;

        // The shared memory per block
        cout << "Total shared memory per block in KB: " <<
            device_prop.sharedMemPerBlock / (1 << 10) << endl;

        // And much more!
    }

    return 0;
}
