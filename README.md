# GPGPU Programming with CUDA
This repository contains all code from the YouTube series "GPGPU Programming with CUDA" by CoffeeBeforeArch.

## Contact

Suggestions for specific content can be sent to: CoffeeBeforeArch@gmail.com

An up to date list on all series is available at: <a href="https://docs.google.com/spreadsheets/d/1cV4nuP-ZflfeGUn0Ay0w-ZKJrh9CqDEKvBJqCNMlTxI/edit?usp=sharing">Google Sheets</a>

## Environment 
Operating System: Windows 10 & Ubuntu 18.04

IDE: Visual Studio 2017

Text Editor: VIM

GPU: NVIDIA GTX 1050 Ti

CUDA version: 10.0, 9.1

## Concepts covered in each video
# Environment Setup
| Video | Concepts | Files |
| ----- | -------- | ----- |
| <a href=https://youtu.be/3usDbpnn7E8>CUDA Crash Course: Visual Studio 2017 Environment Setup</a> | Setup, Linker, Visual Studio, Environmen, Build Paths | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/misc/vs_setup/vs_setup/vs_setup.cu>vs_setup.cu</a>
| <a href=https://youtu.be/AA7RIRxesD4>CUDA Crash Course: Programming in Linux</a> | NVCC, NVprof, Vector Addition | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/misc/linux_vector_add/vector_add.cu>vector_add.cu</a>

# Vector Addition
| Video | Concepts | Files |
| ----- | -------- | ----- |
| <a href=https://youtu.be/2NgpYFdsduY>GPGPU Programming with CUDA: Vector Add</a> | GPU Threads, Memory Allocation, Memory Copy, GPU Kernels, Running Kernels | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/vectorAdd/vector_add/vector_add/vector_add.cu >vector_add.cu</a> |
| <a href=https://youtu.be/84iwCupHW14>GPGPU Programming with CUDA: Vector Add with Unified Memory</a> | Unified Memory, Prefetching | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/vectorAdd/vector_add_um/vector_add_um/vector_add_um.cu >vector_add_um.cu</a> |

# Matrix Multiplication
| Video | Concepts | Files |
| ----- | -------- | ----- |
| <a href=https://youtu.be/XEOc4HCf_pQ>GPGPU Programming with CUDA: Matrix Multiplication</a> | 2-D Threadblocks, Alligned Memory Accesses | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/matrixMul/matrix_mul/matrix_mul/matrix_mul.cu >matrix_mul.cu</a> |
| <a href=https://youtu.be/3xfyiWhtvZw>GPGPU Programming with CUDA: Tiled Matrix Multiplication</a> | Shared Memory, Cache Tiling, Performance Analysis, Optimization | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/matrixMul/tiled_matrix_mul/tiled_matrix_mul/tiled_matrix_mul.cu>tiled_matrix_mul.cu</a> |
| <a href=https://youtu.be/_qSP455IekE>CUDA Crash Course: Why Coalescing Matters</a> | Transposing Matrices, Coalescing Techniques | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/matrixMul/alignment_matrix_mul/alignment_matrix_mul/alignment_matrix_mul.cu>alignment_matrix_mul.cu</a>

# CUDA Libraries
| Video | Concepts | Files |
| ----- | -------- | ----- |
| <a href=https://youtu.be/2_wZBq544gA>CUDA Crash Course: cuBLAS for Vector Add</a> | cuBLAS, SAXPY | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/libraries/simple_cublas/simple_cublas/simple_cublas.cu>simple_cublas.cu</a>
| <a href=https://youtu.be/MVutNZaNTkM>CUDA Crash Course: cuBLAS for Matrix Multiplication</a> | Column-Major Order, SGEMM, cuRAND | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/libraries/cublas_matrix_mul/cublas_matrix_mul/cublas_matrix_mul.cu>cublas_matrix_mul.cu</a>

# Sum Reduction
| Video | Concepts | Files |
| ----- | -------- | ----- |
| <a href=https://youtu.be/bpbit8SPMxU>CUDA Crash Course: Sum Reduction Part 1</a> | Sum Reduction, Warp Divergence | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/sumReduction/sum_reduction_diverged/sum_reduction_diverged/sum_reduction_diverged.cu>sum_reduction_diverged.cu</a>
| <a href=https://youtu.be/JmnPaOXxWLg>CUDA Crash Course: Sum Reduction Part 2</a> | Expensive Operations, Optimization, Warp Divergence | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/sumReduction/sum_reduction_bank_conflicts/sum_reduction_bank_conflicts/sum_reduction_bank_conflicts.cu>sum_reduction_bank_conflicts.cu</a>
| <a href=https://youtu.be/iHeze1VdxYA>CUDA Crash Course: Sum Reduction Part 3</a> | Optimization, Shared Memory Bank Conflicts | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/sumReduction/sum_reduction_no_conflicts/sum_reduction_no_conflicts/sum_reduction_no_conflicts.cu>sum_reduction_no_conflicts.cu</a>
| <a href=https://youtu.be/xXiA3dzl2UE>CUDA Crash Course: Sum Reduction Part 4</a> | Optimization, Idle Threads | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/sumReduction/sum_reduction_reduce_idle_threads/sum_reduction_reduce_idle_threads/sum_reduction_reduce_idle_threads.cu>sum_reduction_reduce_idle_threads.cu</a>
| <a href=https://youtu.be/Qpx227w6idA>CUDA Crash Course: Sum Reduction Part 5</a> | Optimization, Device Function, Loop Unrolling | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/sumReduction/sum_reduction_device_function/sum_reduction_device_function/sum_reduction_device_function.cu>sum_reduction_device_function.cu</a>
| <a href=https://youtu.be/dL6N0cdiMoU>CUDA Crash Course: Sum Reduction Part 6</a> | Cooperative Groups, Synchronization, Atomic Instructions | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/sumReduction/sum_reduction_cooperative_groups/sum_reduction_cooperative_groups/sum_reduction_cooperative_groups.cu>sum_reduction_cooperative_groups.cu</a> |

# Convolution
| Video | Concepts | Files |
| ----- | -------- | ----- |
|<a href=https://youtu.be/OlLquh9Lnbc>CUDA Crash Course: Naive 1-D Convolution</a> | 1-D Convolution | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/convolution/1d_naive/convolution.cu>convolution.cu</a> |
|<a href=https://youtu.be/n7vtr2hCzoc>CUDA Crash Course: 1-D Convolution with Constant Memory </a> | Constant Memory, Constant Cache | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/convolution/1d_constant_memory/convolution.cu>convolution.cu</a> |
|<a href=https://youtu.be/pBB8mZRM91A>CUDA Crash Course: Tiled 1-D Convolution </a> | Shared Memory, Tiling | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/convolution/1d_tiled/convolution.cu>convolution.cu</a> |
|<a href=https://youtu.be/miNGeE9ox20>CUDA Crash Course: 1-D Convolution Cache Simplification </a> | Shared Memory, Tiling, Programmability| <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/convolution/1d_cache/convolution.cu>convolution.cu</a>
|<a href=https://youtu.be/qxcfco89wvs>CUDA Crash Course: 2-D Convolution </a> | 2-D Convolution, Multi-Dimensional Thread Blocks | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/convolution/2d_constant_memory/convolution.cu>convolution.cu</a> |

# Histogram
| Video | Concepts | Files |
| ----- | -------- | ----- |
|<a href=https://youtu.be/v4DGiuU53h8>CUDA Crash Course: Optimizing Histogram Kernels </a> | Global Atomics, Shared Memory Atomics, Histograms, GNU Plot | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/histogram/global_mem/histogram.cu>histogram.cu</a><br><a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/histogram/shared_mem/histogram.cu>histogram.cu</a> |

# Misc. Topics
| Video | Concepts | Files |
| ----- | -------- | ----- |
| <a href=https://youtu.be/ZNaEyWYqiJ8>CUDA Crash Course: Video Corrections</a> | TB Calculations, Verification | <a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/vectorAdd/vector_add/vector_add/vector_add.cu >vector_add.cu</a><br><a href=https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/matrixMul/matrix_mul/matrix_mul/matrix_mul.cu >matrix_mul.cu</a> |
