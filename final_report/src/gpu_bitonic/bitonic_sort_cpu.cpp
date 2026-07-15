/**
 * CPU Bitonic Sort Benchmark
 *
 * Compares bitonic sort CPU vs std::sort for different sizes.
 * Used to validate the theoretical analysis.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static inline void compare_swap(float *a, float *b, int asc) {
    if ((*a > *b) == asc) {
        float t = *a;
        *a = *b;
        *b = t;
    }
}

void bitonic_sort_cpu(float *data, int n) {
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = 0; i < n; i++) {
                int ixj = i ^ j;
                if (ixj > i) {
                    int asc = ((i & k) == 0);
                    compare_swap(&data[i], &data[ixj], asc);
                }
            }
        }
    }
}

void generate_random(float *data, int n) {
    for (int i = 0; i < n; i++)
        data[i] = (float)(rand() % 100000) / 1000.0f;
}

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    int test_sizes[] = {256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    int num_runs = 3;

    if (argc > 1) {
        test_sizes[0] = atoi(argv[1]);
        num_tests = 1;
    }
    if (argc > 2) num_runs = atoi(argv[2]);

#ifdef _OPENMP
    printf("OpenMP threads: %d\n", omp_get_max_threads());
#endif
    printf("%-12s %-16s %-16s %-10s\n",
           "Size", "Bitonic(ms)", "std::sort(ms)", "Ratio");
    printf("----------------------------------------------------------\n");

    srand(42);

    for (int t = 0; t < num_tests; t++) {
        int n = test_sizes[t];
        float *data = (float *)malloc(n * sizeof(float));
        float *copy = (float *)malloc(n * sizeof(float));
        generate_random(data, n);

        // Bitonic sort
        memcpy(copy, data, n * sizeof(float));
        double t0 = get_time_ms();
        for (int r = 0; r < num_runs; r++) {
            memcpy(copy, data, n * sizeof(float));
            bitonic_sort_cpu(copy, n);
        }
        double bitonic_ms = (get_time_ms() - t0) / num_runs;

        // std::sort
        std::vector<float> v(n);
        t0 = get_time_ms();
        for (int r = 0; r < num_runs; r++) {
            memcpy(v.data(), data, n * sizeof(float));
            std::sort(v.begin(), v.end());
        }
        double stdsort_ms = (get_time_ms() - t0) / num_runs;

        printf("%-12d %-16.3f %-16.3f %-10.2fx\n",
               n, bitonic_ms, stdsort_ms, bitonic_ms / stdsort_ms);

        free(data);
        free(copy);
    }

    return 0;
}
