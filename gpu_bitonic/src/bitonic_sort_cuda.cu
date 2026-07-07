/**
 * GPU Bitonic Sort Implementation
 *
 * Bitonic Sort is a parallel sorting algorithm with O(log² n) time complexity
 * on O(n) processors. It works by building larger and larger bitonic sequences
 * and then merging them.
 *
 * This implementation runs on NVIDIA GPUs using CUDA.
 * It supports both int and float data types.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

/* ================================================================
 * Utility: Check CUDA errors
 * ================================================================ */
#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while (0)

/* ================================================================
 * CPU Reference: Bitonic Sort (for verification)
 * ================================================================ */

// Compare and swap: if (ascending && a > b) or (!ascending && a < b), swap
static inline void compare_swap(float *a, float *b, int asc) {
    if ((*a > *b) == asc) {
        float t = *a;
        *a = *b;
        *b = t;
    }
}

/**
 * CPU bitonic sort for verification.
 * n must be a power of 2.
 */
void bitonic_sort_cpu(float *data, int n) {
    for (int k = 2; k <= n; k <<= 1) {          // stage: build larger bitonic sequences
        for (int j = k >> 1; j > 0; j >>= 1) {  // step: compare within distance j
            for (int i = 0; i < n; i++) {
                int ixj = i ^ j;
                if (ixj > i) {
                    int asc = ((i & k) == 0);   // 1 = ascending, 0 = descending
                    compare_swap(&data[i], &data[ixj], asc);
                }
            }
        }
    }
}

/* ================================================================
 * GPU Kernel: Bitonic Sort Step
 * ================================================================ */

/**
 * One step of the bitonic sort. Each thread handles one element.
 *
 * Parameters:
 *   data: array of n elements in global memory
 *   n:    number of elements (power of 2)
 *   j:    current step distance (power of 2)
 *   k:    current stage size (power of 2, > j)
 */
__global__ void bitonic_sort_step_global(float *data, int n, int j, int k) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;  // total threads (>= n)

    for (unsigned int i = tid; i < n; i += stride) {
        unsigned int ixj = i ^ j;
        if (ixj > i) {
            int asc = ((i & k) == 0);  // direction flag
            float a = data[i];
            float b = data[ixj];
            if ((a > b) == asc) {
                data[i] = b;
                data[ixj] = a;
            }
        }
    }
}

/* ================================================================
 * GPU Bitonic Sort (Shared Memory Version)
 * Optimized for small n that fits in shared memory.
 * For larger n, use the global memory version with multiple launches.
 * ================================================================ */

/**
 * Shared-memory bitonic sort for arrays up to 1024 elements per block.
 * Use one block to sort the entire array.
 */
__global__ void bitonic_sort_shared(float *data, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    sdata[tid] = (tid < n) ? data[tid] : 0.0f;
    __syncthreads();

    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                int asc = ((tid & k) == 0);
                float a = sdata[tid];
                float b = sdata[ixj];
                if ((a > b) == asc) {
                    sdata[tid] = b;
                    sdata[ixj] = a;
                }
            }
            __syncthreads();
        }
    }

    if (tid < n) {
        data[tid] = sdata[tid];
    }
}

/* ================================================================
 * Host: Orchestrate GPU Bitonic Sort (Global Memory)
 *
 * Launches the bitonic sort kernel repeatedly for each (stage, step).
 * ================================================================ */

void bitonic_sort_gpu(float *d_data, int n) {
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_sort_step_global<<<blocks, threads_per_block>>>(d_data, n, j, k);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

/* ================================================================
 * Utility Functions
 * ================================================================ */

/** Generate random float array (values in [0, 100)). */
void generate_random(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = (float)(rand() % 100000) / 1000.0f;
    }
}

/** Check if array is sorted ascending. */
int is_sorted(float *data, int n) {
    for (int i = 1; i < n; i++) {
        if (data[i - 1] > data[i]) {
            return 0;
        }
    }
    return 1;
}

/** Compare two arrays element-wise. */
int arrays_equal(float *a, float *b, int n) {
    for (int i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %d: GPU=%f, CPU=%f\n",
                    i, a[i], b[i]);
            return 0;
        }
    }
    return 1;
}

/* ================================================================
 * Timer (high-resolution)
 * ================================================================ */
typedef struct {
    cudaEvent_t start, stop;
} GpuTimer;

void timer_start(GpuTimer *t) {
    CUDA_CHECK(cudaEventCreate(&t->start));
    CUDA_CHECK(cudaEventCreate(&t->stop));
    CUDA_CHECK(cudaEventRecord(t->start, 0));
}

float timer_stop(GpuTimer *t) {
    float ms;
    CUDA_CHECK(cudaEventRecord(t->stop, 0));
    CUDA_CHECK(cudaEventSynchronize(t->stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t->start, t->stop));
    CUDA_CHECK(cudaEventDestroy(t->start));
    CUDA_CHECK(cudaEventDestroy(t->stop));
    return ms;
}

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char **argv) {
    // Configuration: test sizes (all powers of 2)
    int test_sizes[] = {256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    int num_runs = 5;  // repeat each test for stable timing

    // Read optional size override from command line
    if (argc > 1) {
        test_sizes[0] = atoi(argv[1]);
        num_tests = 1;
    }
    if (argc > 2) {
        num_runs = atoi(argv[2]);
    }

    // Check GPU
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    printf("CUDA Devices: %d\n", device_count);
    for (int d = 0; d < device_count; d++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, d));
        printf("  Device %d: %s\n", d, prop.name);
        printf("    Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("    Clock Rate: %.0f MHz\n", prop.clockRate / 1000.0);
        printf("    SMs: %d\n", prop.multiProcessorCount);
        printf("    Global Memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }
    printf("\n");

    printf("============================================================\n");
    printf("  GPU Bitonic Sort Performance Test\n");
    printf("============================================================\n");
    printf("%-12s %-14s %-14s %-10s %-10s\n",
           "Size", "GPU Time (ms)", "CPU Time (ms)", "Speedup", "Correct");
    printf("------------------------------------------------------------\n");

    srand(42);

    for (int t = 0; t < num_tests; t++) {
        int n = test_sizes[t];
        size_t bytes = n * sizeof(float);

        // Allocate host memory
        float *h_data = (float *)malloc(bytes);
        float *h_gpu_result = (float *)malloc(bytes);
        float *h_cpu_result = (float *)malloc(bytes);

        generate_random(h_data, n);

        // --- GPU Sort ---
        float *d_data;
        CUDA_CHECK(cudaMalloc(&d_data, bytes));

        // Warmup
        CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
        bitonic_sort_gpu(d_data, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed runs
        float gpu_total = 0.0f;
        for (int r = 0; r < num_runs; r++) {
            CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
            GpuTimer gt;
            timer_start(&gt);
            bitonic_sort_gpu(d_data, n);
            gpu_total += timer_stop(&gt);
        }
        float gpu_ms = gpu_total / num_runs;

        CUDA_CHECK(cudaMemcpy(h_gpu_result, d_data, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_data));

        // --- CPU Sort (Bitonic, for correctness check) ---
        memcpy(h_cpu_result, h_data, bytes);

        clock_t cpu_start = clock();
        bitonic_sort_cpu(h_cpu_result, n);
        clock_t cpu_end = clock();
        float cpu_ms = 1000.0f * (cpu_end - cpu_start) / CLOCKS_PER_SEC;

        // --- Verification ---
        int correct = is_sorted(h_gpu_result, n) && arrays_equal(h_gpu_result, h_cpu_result, n);
        float speedup = cpu_ms / gpu_ms;

        printf("%-12d %-14.3f %-14.3f %-10.2fx %s\n",
               n, gpu_ms, cpu_ms, speedup,
               correct ? "✓" : "✗ FAIL");

        free(h_data);
        free(h_gpu_result);
        free(h_cpu_result);
    }

    printf("------------------------------------------------------------\n");
    printf("All tests completed.\n");

    return 0;
}
