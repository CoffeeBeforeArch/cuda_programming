// This program shows how to automatically execute on the GPU using
// OpenACC
// By: Nick from CoffeeBeforeArch

#include <stdlib.h>

// Simple function to init matrices with random numbers
void init_matrix(int *m, int N){
    for(int i = 0; i < N*N; i++){
        m[i] = rand() % 100;
    }
}

int main(){
    // Dimensions of matrices
    int N = 1 << 10;

    // Allocate space for matrices
    int *a = new int[N * N];
    int *b = new int[N * N];
    int *c = new int[N * N];

    // Init matrices
    init_matrix(a, N);
    init_matrix(b, N);

    // Perform matrix multiplication
    // "kernels" specifies this is offloaded to the accelerator
    // "copyin" specifies data to copy in a single direction to the device
    // "copy" specifies a two-way copy
    #pragma acc kernels copyin(a[0:N*N], b[0:N*N]) copy(c[0:N*N])
    {
        // All outer loops are independent
        #pragma acc loop independent
        for(int i = 0; i < N; i++){
            // All outer loops are independent
            #pragma acc loop independent
            for(int j = 0; j < N; j++){
                float sum = 0;
                #pragma acc loop independent reduction (+: sum)
                for(int k = 0; k < N; k++){
                    sum += a[i * N + k] * b[k * N + j];
                }
                c[i * N + j] = sum;
            }
        }
    }

    return 0;
}
