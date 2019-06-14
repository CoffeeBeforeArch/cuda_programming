// This program is used to get performance numbers for different
// implementations of matrix multiplication in CUDA
// By: Nick from CoffeeBeforeArch

#include <iostream>
#include <fstream>
#include "common.h"

int main(){
    // Number of iterations to run per-kernel (10 by default)
    int N = 10;

    // Upper bound of matrix size (2^14 by default)
    int D = 1 << 12;

    // Vector to get return average execution times
    vector<float> times;

    // Get execution time for naive implementation
    times = launch_mmul(D, N);

    // Write out the times to a data file
    ofstream output_file;
    output_file.open("timing.dat", ios::out | ios::trunc);
    for(auto i : times){
        output_file << i << "\n"; 
    }

    return 0;
}
