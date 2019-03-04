# GPGPU Programming with CUDA
This repository contains all code from the YouTube series "GPGPU Programming with CUDA" by CoffeeBeforeArch.

## Contact

Suggestions for specific content can be sent to: CoffeeBeforeArch@gmail.com

An up to date list on all series is available at: <a href="https://docs.google.com/spreadsheets/d/1cV4nuP-ZflfeGUn0Ay0w-ZKJrh9CqDEKvBJqCNMlTxI/edit?usp=sharing">Google Sheets</a>

## Environment 
Operating System: Windows 10

IDE: Visual Studio 2017

GPU: NVIDIA GTX 1050 Ti

CUDA version: 10.0

## Concepts covered in each video
| Video | Concepts | Files |
| ----- | -------- | ----- |
| <a href=https://youtu.be/2NgpYFdsduY>GPGPU Programming with CUDA: Vector Add</a> | GPU Threads, Memory Allocation, Memory Copy, GPU Kernels, Running Kernels | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/vector_add/vector_add/vector_add.cu >vector_add.cu</a> |
| <a href=https://youtu.be/84iwCupHW14>GPGPU Programming with CUDA: Vector Add with Unified Memory</a> | Unified Memory, Prefetching | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/vector_add_um/vector_add_um/vector_add_um.cu >vector_add_um.cu</a> |
| <a href=https://youtu.be/XEOc4HCf_pQ>GPGPU Programming with CUDA: Matrix Multiplication</a> | 2-D Threadblocks, Alligned Memory Accesses | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/matrix_mul/matrix_mul/matrix_mul.cu >matrix_mul.cu</a> |
| <a href=https://youtu.be/3xfyiWhtvZw>GPGPU Programming with CUDA: Tiled Matrix Multiplication</a> | Shared Memory, Cache Tiling, Performance Analysis, Optimization | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/tiled_matrix_mul/tiled_matrix_mul/tiled_matrix_mul.cu>tiled_matrix_mul.cu</a> |
| <a href=https://youtu.be/_qSP455IekE>CUDA Crash Course: Why Coalescing Matters</a> | Transposing Matrices, Coalescing Techniques | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/alignment_matrix_mul/alignment_matrix_mul/alignment_matrix_mul.cu>alignment_matrix_mul.cu</a>
| <a href=https://youtu.be/2_wZBq544gA>CUDA Crash Course: cuBLAS for Vector Add</a> | cuBLAS, SAXPY | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/simple_cublas/simple_cublas/simple_cublas.cu>simple_cublas.cu</a>
| <a href=https://youtu.be/MVutNZaNTkM>CUDA Crash Course: cuBLAS for Matrix Multiplication</a> | Column-Major Order, SGEMM, cuRAND | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/cublas_matrix_mul/cublas_matrix_mul/cublas_matrix_mul.cu>cublas_matrix_mul.cu</a>
| <a href=https://youtu.be/bpbit8SPMxU>CUDA Crash Course: Sum Reduction Part 1</a> | Sum Reduction, Warp Divergence | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/sum_reduction_diverged/sum_reduction_diverged/sum_reduction_diverged.cu>sum_reduction_diverged.cu</a>
