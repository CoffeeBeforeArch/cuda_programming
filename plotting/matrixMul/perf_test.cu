// This program is used to get performance numbers for different
// implementations of matrix multiplication in CUDA
// By: Nick from CoffeeBeforeArch

#include <iostream>
#include <fstream>
#include "common.h"

int main(){
    // Number of iterations to run per-kernel (10 by default)
    int N = 10;

    // Upper bound of matrix size (2^16 by default)
    int D = 1 << 16;

    // Vector to get return average execution times
    vector<float> times;

    // Get execution time for naive implementation
    times = launch_naive_mmul(D, N);

    for(auto i : times){
        cout << i << endl;
    }

    return 0;
}
