#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>
#include "hnswlib/hnswlib/hnswlib.h"

namespace {

constexpr int kTopK = 10;

template <typename T>
bool load_bin_matrix(const std::string& path, std::vector<T>& data, uint32_t& n, uint32_t& d) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin.good()) return false;
    fin.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
    fin.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
    if (!fin.good() || n == 0 || d == 0) return false;
    data.resize(static_cast<size_t>(n) * d);
    fin.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
    return fin.good();
}

float ip_distance(const float* a, const float* b, uint32_t dim) {
    float ip = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) ip += a[d] * b[d];
    return 1.0f - ip;
}

float l2_distance(const float* a, const float* b, uint32_t dim) {
    float sum = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) {
        float diff = a[d] - b[d];
        sum += diff * diff;
    }
    return sum;
}

using Item = std::pair<float, int>;

void push_topk(std::priority_queue<Item>& heap, float dis, int id, int k) {
    if (static_cast<int>(heap.size()) < k) {
        heap.push({dis, id});
    } else if (dis < heap.top().first) {
        heap.push({dis, id});
        heap.pop();
    }
}

std::vector<int> exact_flat_batch(
    const std::vector<float>& base,
    const std::vector<float>& query,
    uint32_t base_n,
    uint32_t query_n,
    uint32_t dim) {
    std::vector<int> result(static_cast<size_t>(query_n) * kTopK, -1);
#pragma omp parallel for schedule(dynamic)
    for (int qi = 0; qi < static_cast<int>(query_n); ++qi) {
        std::priority_queue<Item> heap;
        const float* q = query.data() + static_cast<size_t>(qi) * dim;
        for (uint32_t bi = 0; bi < base_n; ++bi) {
            float dis = ip_distance(base.data() + static_cast<size_t>(bi) * dim, q, dim);
            push_topk(heap, dis, static_cast<int>(bi), kTopK);
        }
        for (int j = kTopK - 1; j >= 0; --j) {
            result[static_cast<size_t>(qi) * kTopK + j] = heap.top().second;
            heap.pop();
        }
    }
    return result;
}

double recall_at_10(const std::vector<int>& got, const std::vector<int>& gt, uint32_t query_n) {
    double total = 0.0;
    for (uint32_t qi = 0; qi < query_n; ++qi) {
        std::set<int> gtset;
        for (int j = 0; j < kTopK; ++j) gtset.insert(gt[static_cast<size_t>(qi) * kTopK + j]);
        int hit = 0;
        for (int j = 0; j < kTopK; ++j) {
            if (gtset.find(got[static_cast<size_t>(qi) * kTopK + j]) != gtset.end()) ++hit;
        }
        total += static_cast<double>(hit) / kTopK;
    }
    return total / query_n;
}

struct IVFIndex {
    uint32_t nlist = 0;
    uint32_t dim = 0;
    std::vector<float> centroids;
    std::vector<std::vector<int>> lists;
};

IVFIndex build_ivf(const std::vector<float>& base, uint32_t base_n, uint32_t dim, uint32_t nlist, uint32_t train_n, int iters) {
    IVFIndex index;
    index.nlist = nlist;
    index.dim = dim;
    index.centroids.assign(static_cast<size_t>(nlist) * dim, 0.0f);
    index.lists.resize(nlist);
    train_n = std::min(train_n, base_n);

    for (uint32_t c = 0; c < nlist; ++c) {
        uint32_t src = static_cast<uint64_t>(c) * train_n / nlist;
        std::copy(base.data() + static_cast<size_t>(src) * dim,
                  base.data() + static_cast<size_t>(src + 1) * dim,
                  index.centroids.data() + static_cast<size_t>(c) * dim);
    }

    std::vector<int> label(train_n, 0);
    for (int iter = 0; iter < iters; ++iter) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(train_n); ++i) {
            const float* v = base.data() + static_cast<size_t>(i) * dim;
            float best = std::numeric_limits<float>::max();
            int best_id = 0;
            for (uint32_t c = 0; c < nlist; ++c) {
                float dis = l2_distance(v, index.centroids.data() + static_cast<size_t>(c) * dim, dim);
                if (dis < best) {
                    best = dis;
                    best_id = static_cast<int>(c);
                }
            }
            label[i] = best_id;
        }

        std::vector<float> sums(static_cast<size_t>(nlist) * dim, 0.0f);
        std::vector<int> counts(nlist, 0);
        for (uint32_t i = 0; i < train_n; ++i) {
            int c = label[i];
            ++counts[c];
            const float* v = base.data() + static_cast<size_t>(i) * dim;
            float* sum = sums.data() + static_cast<size_t>(c) * dim;
            for (uint32_t d = 0; d < dim; ++d) sum[d] += v[d];
        }
        for (uint32_t c = 0; c < nlist; ++c) {
            if (counts[c] == 0) continue;
            float inv = 1.0f / counts[c];
            float* dst = index.centroids.data() + static_cast<size_t>(c) * dim;
            float* sum = sums.data() + static_cast<size_t>(c) * dim;
            for (uint32_t d = 0; d < dim; ++d) dst[d] = sum[d] * inv;
        }
    }

    std::vector<std::vector<int>> local_lists(nlist);
    for (uint32_t i = 0; i < base_n; ++i) {
        const float* v = base.data() + static_cast<size_t>(i) * dim;
        float best = std::numeric_limits<float>::max();
        uint32_t best_id = 0;
        for (uint32_t c = 0; c < nlist; ++c) {
            float dis = l2_distance(v, index.centroids.data() + static_cast<size_t>(c) * dim, dim);
            if (dis < best) {
                best = dis;
                best_id = c;
            }
        }
        local_lists[best_id].push_back(static_cast<int>(i));
    }
    index.lists.swap(local_lists);
    return index;
}

std::vector<int> search_ivf_batch(
    const IVFIndex& index,
    const std::vector<float>& base,
    const std::vector<float>& query,
    uint32_t query_n,
    uint32_t base_n,
    uint32_t dim,
    uint32_t nprobe) {
    (void)base_n;
    std::vector<int> result(static_cast<size_t>(query_n) * kTopK, -1);
#pragma omp parallel for schedule(dynamic)
    for (int qi = 0; qi < static_cast<int>(query_n); ++qi) {
        const float* q = query.data() + static_cast<size_t>(qi) * dim;
        std::priority_queue<Item> probe_heap;
        for (uint32_t c = 0; c < index.nlist; ++c) {
            float dis = l2_distance(q, index.centroids.data() + static_cast<size_t>(c) * dim, dim);
            push_topk(probe_heap, dis, static_cast<int>(c), nprobe);
        }
        std::vector<int> probes;
        while (!probe_heap.empty()) {
            probes.push_back(probe_heap.top().second);
            probe_heap.pop();
        }

        std::priority_queue<Item> heap;
        for (int c : probes) {
            for (int id : index.lists[c]) {
                float dis = ip_distance(base.data() + static_cast<size_t>(id) * dim, q, dim);
                push_topk(heap, dis, id, kTopK);
            }
        }
        int out = kTopK - 1;
        while (!heap.empty() && out >= 0) {
            result[static_cast<size_t>(qi) * kTopK + out] = heap.top().second;
            heap.pop();
            --out;
        }
    }
    return result;
}

std::vector<int> search_hnsw_batch(
    hnswlib::HierarchicalNSW<float>& index,
    const std::vector<float>& query,
    uint32_t query_n,
    uint32_t dim,
    int ef_search) {
    index.setEf(ef_search);
    std::vector<int> result(static_cast<size_t>(query_n) * kTopK, -1);
    for (uint32_t qi = 0; qi < query_n; ++qi) {
        auto res = index.searchKnn(query.data() + static_cast<size_t>(qi) * dim, kTopK);
        for (int j = kTopK - 1; j >= 0 && !res.empty(); --j) {
            result[static_cast<size_t>(qi) * kTopK + j] = static_cast<int>(res.top().second);
            res.pop();
        }
    }
    return result;
}

template <typename Fn>
std::pair<std::vector<int>, double> timed_search(Fn&& fn) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> result = fn();
    auto stop = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(stop - start).count();
    return {std::move(result), ms};
}

}  // namespace

int main(int argc, char** argv) {
    std::string data_dir = argc > 1 ? argv[1] : "/anndata";
    uint32_t query_n = argc > 2 ? static_cast<uint32_t>(std::atoi(argv[2])) : 200;
    uint32_t base_limit = argc > 3 ? static_cast<uint32_t>(std::atoi(argv[3])) : 100000;
    if (!data_dir.empty() && data_dir.back() != '/') data_dir.push_back('/');

    std::vector<float> base, query;
    uint32_t base_n = 0, base_dim = 0, q_n = 0, q_dim = 0;
    if (!load_bin_matrix<float>(data_dir + "DEEP100K.base.100k.fbin", base, base_n, base_dim) ||
        !load_bin_matrix<float>(data_dir + "DEEP100K.query.fbin", query, q_n, q_dim) ||
        base_dim != q_dim) {
        std::cerr << "failed to load DEEP100K from " << data_dir << std::endl;
        return 1;
    }
    uint32_t dim = base_dim;
    base_n = std::min(base_n, base_limit);
    query_n = std::min(query_n, q_n);
    base.resize(static_cast<size_t>(base_n) * dim);
    query.resize(static_cast<size_t>(query_n) * dim);
    omp_set_num_threads(std::min(16, omp_get_max_threads()));

    std::cout << "name\trecall\tlatency_us\tbuild_ms\textra\n";

    auto exact_pair = timed_search([&] {
        return exact_flat_batch(base, query, base_n, query_n, dim);
    });
    const auto& gt = exact_pair.first;
    std::cout << "flat_exact_cpu\t1.000000\t"
              << (exact_pair.second * 1000.0 / query_n)
              << "\t0\tthreads=" << omp_get_max_threads() << "\n";

    auto build_ivf_start = std::chrono::high_resolution_clock::now();
    IVFIndex ivf = build_ivf(base, base_n, dim, 128, std::min<uint32_t>(20000, base_n), 5);
    auto build_ivf_stop = std::chrono::high_resolution_clock::now();
    double ivf_build_ms = std::chrono::duration<double, std::milli>(build_ivf_stop - build_ivf_start).count();

    for (uint32_t nprobe : {4u, 8u, 16u, 32u}) {
        auto pair = timed_search([&] {
            return search_ivf_batch(ivf, base, query, query_n, base_n, dim, nprobe);
        });
        std::cout << "ivf_flat_nprobe" << nprobe << "\t"
                  << std::fixed << std::setprecision(6) << recall_at_10(pair.first, gt, query_n)
                  << "\t" << std::setprecision(3) << (pair.second * 1000.0 / query_n)
                  << "\t" << ivf_build_ms
                  << "\tnlist=128\n";
    }

    hnswlib::InnerProductSpace ipspace(dim);
    auto hnsw_build_start = std::chrono::high_resolution_clock::now();
    hnswlib::HierarchicalNSW<float> hnsw(&ipspace, base_n, 16, 100);
    for (uint32_t i = 0; i < base_n; ++i) {
        hnsw.addPoint(base.data() + static_cast<size_t>(i) * dim, i);
    }
    auto hnsw_build_stop = std::chrono::high_resolution_clock::now();
    double hnsw_build_ms = std::chrono::duration<double, std::milli>(hnsw_build_stop - hnsw_build_start).count();
    for (int ef : {32, 64, 128}) {
        auto pair = timed_search([&] {
            return search_hnsw_batch(hnsw, query, query_n, dim, ef);
        });
        std::cout << "hnsw_ef" << ef << "\t"
                  << std::fixed << std::setprecision(6) << recall_at_10(pair.first, gt, query_n)
                  << "\t" << std::setprecision(3) << (pair.second * 1000.0 / query_n)
                  << "\t" << hnsw_build_ms
                  << "\tM=16 efC=100\n";
    }
    return 0;
}
