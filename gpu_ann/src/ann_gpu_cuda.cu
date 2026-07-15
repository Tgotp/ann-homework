#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace {

constexpr int kTopK = 10;
constexpr int kBlockThreads = 256;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                      << ": " << cudaGetErrorString(err__) << std::endl;     \
            std::exit(1);                                                     \
        }                                                                    \
    } while (0)

struct Dataset {
    std::vector<float> base;
    std::vector<float> query;
    std::vector<int> gt;
    uint32_t base_n = 0;
    uint32_t query_n = 0;
    uint32_t dim = 0;
    uint32_t gt_dim = 0;
    bool has_gt = false;
    bool synthetic = false;
};

template <typename T>
bool load_bin_matrix(const std::string& path, std::vector<T>& data, uint32_t& n, uint32_t& d) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin.good()) {
        return false;
    }

    fin.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
    fin.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
    if (!fin.good() || n == 0 || d == 0) {
        return false;
    }

    data.resize(static_cast<size_t>(n) * d);
    fin.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
    return fin.good();
}

void generate_synthetic(Dataset& ds, uint32_t base_n, uint32_t query_n, uint32_t dim) {
    ds.base_n = base_n;
    ds.query_n = query_n;
    ds.dim = dim;
    ds.gt_dim = 0;
    ds.has_gt = false;
    ds.synthetic = true;
    ds.base.resize(static_cast<size_t>(base_n) * dim);
    ds.query.resize(static_cast<size_t>(query_n) * dim);

    std::mt19937 rng(20260715);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    auto fill_normalized = [&](std::vector<float>& data, uint32_t n) {
        for (uint32_t i = 0; i < n; ++i) {
            float norm = 0.0f;
            float* row = data.data() + static_cast<size_t>(i) * dim;
            for (uint32_t j = 0; j < dim; ++j) {
                row[j] = dist(rng);
                norm += row[j] * row[j];
            }
            norm = std::sqrt(std::max(norm, 1e-20f));
            for (uint32_t j = 0; j < dim; ++j) {
                row[j] /= norm;
            }
        }
    };

    fill_normalized(ds.base, base_n);
    fill_normalized(ds.query, query_n);
}

Dataset load_or_generate_dataset(
    const std::string& data_dir,
    uint32_t requested_queries,
    uint32_t requested_base) {
    Dataset ds;
    uint32_t base_dim = 0;
    uint32_t query_dim = 0;
    uint32_t gt_n = 0;

    std::string prefix = data_dir;
    if (!prefix.empty() && prefix.back() != '/') {
        prefix.push_back('/');
    }

    bool ok_base = load_bin_matrix<float>(prefix + "DEEP100K.base.100k.fbin", ds.base, ds.base_n, base_dim);
    bool ok_query = load_bin_matrix<float>(prefix + "DEEP100K.query.fbin", ds.query, ds.query_n, query_dim);
    if (ok_base && ok_query && base_dim == query_dim) {
        ds.dim = base_dim;
        if (requested_base > 0 && requested_base < ds.base_n) {
            ds.base_n = requested_base;
            ds.base.resize(static_cast<size_t>(ds.base_n) * ds.dim);
        }
        if (requested_queries > 0 && requested_queries < ds.query_n) {
            ds.query_n = requested_queries;
            ds.query.resize(static_cast<size_t>(ds.query_n) * ds.dim);
        }

        ds.has_gt = load_bin_matrix<int>(
            prefix + "DEEP100K.gt.query.100k.top100.bin",
            ds.gt,
            gt_n,
            ds.gt_dim);
        if (ds.has_gt && gt_n < ds.query_n) {
            ds.has_gt = false;
        }
        std::cerr << "Loaded DEEP100K from " << prefix << std::endl;
        return ds;
    }

    uint32_t base_n = requested_base > 0 ? requested_base : 100000;
    uint32_t query_n = requested_queries > 0 ? requested_queries : 2000;
    generate_synthetic(ds, base_n, query_n, 96);
    std::cerr << "DEEP100K not found under " << prefix
              << "; generated deterministic synthetic dataset." << std::endl;
    return ds;
}

__device__ __forceinline__ void insert_topk(float dis, int id, float* best_dis, int* best_id) {
    if (dis >= best_dis[kTopK - 1]) {
        return;
    }
    int pos = kTopK - 1;
    while (pos > 0 && dis < best_dis[pos - 1]) {
        best_dis[pos] = best_dis[pos - 1];
        best_id[pos] = best_id[pos - 1];
        --pos;
    }
    best_dis[pos] = dis;
    best_id[pos] = id;
}

__global__ void flat_ip_topk_kernel(
    const float* __restrict__ base,
    const float* __restrict__ query,
    int base_n,
    int query_n,
    int dim,
    float* partial_dis,
    int* partial_id) {
    int qid = blockIdx.x;
    if (qid >= query_n) {
        return;
    }

    int tid = threadIdx.x;
    float local_dis[kTopK];
    int local_id[kTopK];
#pragma unroll
    for (int i = 0; i < kTopK; ++i) {
        local_dis[i] = 1.0e30f;
        local_id[i] = -1;
    }

    const float* q = query + static_cast<size_t>(qid) * dim;
    for (int bid = tid; bid < base_n; bid += blockDim.x) {
        const float* b = base + static_cast<size_t>(bid) * dim;
        float ip = 0.0f;
        for (int d = 0; d < dim; ++d) {
            ip += b[d] * q[d];
        }
        float dis = 1.0f - ip;
        insert_topk(dis, bid, local_dis, local_id);
    }

    size_t offset = (static_cast<size_t>(qid) * blockDim.x + tid) * kTopK;
#pragma unroll
    for (int i = 0; i < kTopK; ++i) {
        partial_dis[offset + i] = local_dis[i];
        partial_id[offset + i] = local_id[i];
    }
}

using TopItem = std::pair<float, int>;

std::vector<int> merge_gpu_partials(
    const std::vector<float>& partial_dis,
    const std::vector<int>& partial_id,
    uint32_t query_n) {
    std::vector<int> result(static_cast<size_t>(query_n) * kTopK, -1);
    for (uint32_t q = 0; q < query_n; ++q) {
        std::priority_queue<TopItem> heap;
        for (int t = 0; t < kBlockThreads; ++t) {
            size_t offset = (static_cast<size_t>(q) * kBlockThreads + t) * kTopK;
            for (int j = 0; j < kTopK; ++j) {
                int id = partial_id[offset + j];
                if (id < 0) {
                    continue;
                }
                float dis = partial_dis[offset + j];
                if (heap.size() < kTopK) {
                    heap.push({dis, id});
                } else if (dis < heap.top().first) {
                    heap.push({dis, id});
                    heap.pop();
                }
            }
        }

        for (int j = kTopK - 1; j >= 0; --j) {
            result[static_cast<size_t>(q) * kTopK + j] = heap.top().second;
            heap.pop();
        }
    }
    return result;
}

std::vector<int> cpu_exact_topk_subset(
    const std::vector<float>& base,
    const std::vector<float>& query,
    uint32_t base_n,
    uint32_t query_n,
    uint32_t dim) {
    std::vector<int> result(static_cast<size_t>(query_n) * kTopK, -1);
    for (uint32_t q = 0; q < query_n; ++q) {
        std::priority_queue<TopItem> heap;
        const float* qv = query.data() + static_cast<size_t>(q) * dim;
        for (uint32_t i = 0; i < base_n; ++i) {
            const float* bv = base.data() + static_cast<size_t>(i) * dim;
            float ip = 0.0f;
            for (uint32_t d = 0; d < dim; ++d) {
                ip += bv[d] * qv[d];
            }
            float dis = 1.0f - ip;
            if (heap.size() < kTopK) {
                heap.push({dis, static_cast<int>(i)});
            } else if (dis < heap.top().first) {
                heap.push({dis, static_cast<int>(i)});
                heap.pop();
            }
        }
        for (int j = kTopK - 1; j >= 0; --j) {
            result[static_cast<size_t>(q) * kTopK + j] = heap.top().second;
            heap.pop();
        }
    }
    return result;
}

double recall_against_reference(
    const std::vector<int>& got,
    const std::vector<int>& ref,
    uint32_t query_n,
    uint32_t ref_stride) {
    double total = 0.0;
    for (uint32_t q = 0; q < query_n; ++q) {
        std::set<int> ref_set;
        for (int j = 0; j < kTopK; ++j) {
            ref_set.insert(ref[static_cast<size_t>(q) * ref_stride + j]);
        }
        int hit = 0;
        for (int j = 0; j < kTopK; ++j) {
            int id = got[static_cast<size_t>(q) * kTopK + j];
            if (ref_set.find(id) != ref_set.end()) {
                ++hit;
            }
        }
        total += static_cast<double>(hit) / kTopK;
    }
    return total / query_n;
}

}  // namespace

int main(int argc, char** argv) {
    std::string data_dir = argc > 1 ? argv[1] : "/anndata";
    uint32_t query_count = argc > 2 ? static_cast<uint32_t>(std::atoi(argv[2])) : 2000;
    uint32_t base_limit = argc > 3 ? static_cast<uint32_t>(std::atoi(argv[3])) : 100000;
    uint32_t check_queries = argc > 4 ? static_cast<uint32_t>(std::atoi(argv[4])) : 100;

    Dataset ds = load_or_generate_dataset(data_dir, query_count, base_limit);
    query_count = std::min(query_count, ds.query_n);
    check_queries = std::min(check_queries, query_count);

    std::cout << "dataset: " << (ds.synthetic ? "synthetic" : "DEEP100K") << "\n";
    std::cout << "base_n: " << ds.base_n << "\n";
    std::cout << "query_n: " << query_count << "\n";
    std::cout << "dim: " << ds.dim << "\n";

    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    CUDA_CHECK(cudaSetDevice(device));
    std::cout << "gpu: " << prop.name << "\n";
    std::cout << "sms: " << prop.multiProcessorCount << "\n";

    float* d_base = nullptr;
    float* d_query = nullptr;
    float* d_partial_dis = nullptr;
    int* d_partial_id = nullptr;

    size_t base_bytes = ds.base.size() * sizeof(float);
    size_t query_bytes = static_cast<size_t>(query_count) * ds.dim * sizeof(float);
    size_t partial_count = static_cast<size_t>(query_count) * kBlockThreads * kTopK;

    CUDA_CHECK(cudaMalloc(&d_base, base_bytes));
    CUDA_CHECK(cudaMalloc(&d_query, query_bytes));
    CUDA_CHECK(cudaMalloc(&d_partial_dis, partial_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial_id, partial_count * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_base, ds.base.data(), base_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_query, ds.query.data(), query_bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    flat_ip_topk_kernel<<<query_count, kBlockThreads>>>(
        d_base,
        d_query,
        static_cast<int>(ds.base_n),
        static_cast<int>(query_count),
        static_cast<int>(ds.dim),
        d_partial_dis,
        d_partial_id);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

    std::vector<float> partial_dis(partial_count);
    std::vector<int> partial_id(partial_count);
    auto copy_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(partial_dis.data(), d_partial_dis, partial_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(partial_id.data(), d_partial_id, partial_count * sizeof(int), cudaMemcpyDeviceToHost));
    auto gpu_result = merge_gpu_partials(partial_dis, partial_id, query_count);
    auto copy_end = std::chrono::high_resolution_clock::now();

    double host_merge_ms = std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
    double recall = 0.0;
    uint32_t recall_queries = 0;
    if (ds.has_gt && ds.gt_dim >= kTopK) {
        recall_queries = query_count;
        recall = recall_against_reference(gpu_result, ds.gt, recall_queries, ds.gt_dim);
    } else {
        recall_queries = check_queries;
        auto cpu_ref = cpu_exact_topk_subset(ds.base, ds.query, ds.base_n, recall_queries, ds.dim);
        recall = recall_against_reference(gpu_result, cpu_ref, recall_queries, kTopK);
    }

    double avg_kernel_us = static_cast<double>(kernel_ms) * 1000.0 / query_count;
    double avg_total_us = (static_cast<double>(kernel_ms) + host_merge_ms) * 1000.0 / query_count;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "recall@" << kTopK << ": " << recall << "\n";
    std::cout << "recall_queries: " << recall_queries << "\n";
    std::cout << "kernel_time_ms: " << kernel_ms << "\n";
    std::cout << "copy_merge_time_ms: " << host_merge_ms << "\n";
    std::cout << "avg_kernel_latency_us: " << avg_kernel_us << "\n";
    std::cout << "avg_total_latency_us: " << avg_total_us << "\n";

    CUDA_CHECK(cudaFree(d_base));
    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_partial_dis));
    CUDA_CHECK(cudaFree(d_partial_id));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}
