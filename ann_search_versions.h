#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <limits>
#include <omp.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define ANN_USE_NEON 1
#elif defined(__AVX__) || defined(__SSE__)
#include <immintrin.h>
#define ANN_USE_X86_SIMD 1
#endif

using AnnQueue = std::priority_queue<std::pair<float, uint32_t> >;

struct AnnSearchConfig
{
    int mode = 0;
    int threads = 1;
    int chunk = 256;
    int top_p = 2000;
    int pq_m = 8;
    int pq_ks = 256;
    int pq_iters = 4;
    int pq_train = 10000;
    std::string schedule = "static";
};

static AnnSearchConfig g_ann_config;

static inline void push_topk(AnnQueue& q, float dis, uint32_t id, size_t k)
{
    if (q.size() < k) {
        q.push({dis, id});
    } else if (dis < q.top().first) {
        q.push({dis, id});
        q.pop();
    }
}

static inline float inner_product_scalar(const float* a, const float* b, size_t dim)
{
    float sum = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        sum += a[d] * b[d];
    }
    return sum;
}

static inline float inner_product_simd(const float* a, const float* b, size_t dim)
{
#if defined(ANN_USE_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t d = 0;
    for (; d + 4 <= dim; d += 4) {
        float32x4_t va = vld1q_f32(a + d);
        float32x4_t vb = vld1q_f32(b + d);
#if defined(__aarch64__)
        acc = vfmaq_f32(acc, va, vb);
#else
        acc = vaddq_f32(acc, vmulq_f32(va, vb));
#endif
    }

#if defined(__aarch64__)
    float sum = vaddvq_f32(acc);
#else
    float32x2_t low = vget_low_f32(acc);
    float32x2_t high = vget_high_f32(acc);
    float32x2_t pair_sum = vadd_f32(low, high);
    pair_sum = vpadd_f32(pair_sum, pair_sum);
    float sum = vget_lane_f32(pair_sum, 0);
#endif

#elif defined(__AVX__)
    __m256 acc = _mm256_setzero_ps();
    size_t d = 0;
    for (; d + 8 <= dim; d += 8) {
        __m256 va = _mm256_loadu_ps(a + d);
        __m256 vb = _mm256_loadu_ps(b + d);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
#elif defined(__SSE__)
    __m128 acc = _mm_setzero_ps();
    size_t d = 0;
    for (; d + 4 <= dim; d += 4) {
        __m128 va = _mm_loadu_ps(a + d);
        __m128 vb = _mm_loadu_ps(b + d);
        acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
    }
    alignas(16) float tmp[4];
    _mm_store_ps(tmp, acc);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
#else
    return inner_product_scalar(a, b, dim);
#endif

    for (; d < dim; ++d) {
        sum += a[d] * b[d];
    }
    return sum;
}

struct SQIndex
{
    size_t base_number = 0;
    size_t vecdim = 0;
    std::vector<float> minv;
    std::vector<float> scale;
    std::vector<uint8_t> codes;
};

static SQIndex g_sq_index;

struct PQIndex
{
    size_t base_number = 0;
    size_t vecdim = 0;
    int m = 0;
    int ks = 0;
    int subdim = 0;
    std::vector<float> codebook;
    std::vector<uint8_t> codes;
};

static PQIndex g_pq_index;

static inline int effective_top_p(size_t base_number, size_t k)
{
    int top_p = g_ann_config.top_p;
    if (top_p < static_cast<int>(k)) {
        top_p = static_cast<int>(k);
    }
    if (top_p > static_cast<int>(base_number)) {
        top_p = static_cast<int>(base_number);
    }
    return top_p;
}

static inline void quantize_vector_u8(
    const float* src,
    uint8_t* dst,
    const std::vector<float>& minv,
    const std::vector<float>& scale,
    size_t vecdim)
{
    for (size_t d = 0; d < vecdim; ++d) {
        float q = (src[d] - minv[d]) * scale[d];
        int qi = static_cast<int>(q + 0.5f);
        if (qi < 0) {
            qi = 0;
        } else if (qi > 255) {
            qi = 255;
        }
        dst[d] = static_cast<uint8_t>(qi);
    }
}

static inline void build_sq_index(float* base, size_t base_number, size_t vecdim)
{
    if (g_sq_index.base_number == base_number && g_sq_index.vecdim == vecdim) {
        return;
    }

    g_sq_index.base_number = base_number;
    g_sq_index.vecdim = vecdim;
    g_sq_index.minv.assign(vecdim, std::numeric_limits<float>::max());
    std::vector<float> maxv(vecdim, -std::numeric_limits<float>::max());

    for (size_t i = 0; i < base_number; ++i) {
        const float* v = base + i * vecdim;
        for (size_t d = 0; d < vecdim; ++d) {
            g_sq_index.minv[d] = std::min(g_sq_index.minv[d], v[d]);
            maxv[d] = std::max(maxv[d], v[d]);
        }
    }

    g_sq_index.scale.resize(vecdim);
    for (size_t d = 0; d < vecdim; ++d) {
        float range = maxv[d] - g_sq_index.minv[d];
        g_sq_index.scale[d] = range > 1e-12f ? 255.0f / range : 1.0f;
    }

    g_sq_index.codes.resize(base_number * vecdim);
    for (size_t i = 0; i < base_number; ++i) {
        quantize_vector_u8(
            base + i * vecdim,
            g_sq_index.codes.data() + i * vecdim,
            g_sq_index.minv,
            g_sq_index.scale,
            vecdim);
    }
}

static inline uint32_t dot_u8_scalar(const uint8_t* a, const uint8_t* b, size_t dim)
{
    uint32_t sum = 0;
    for (size_t d = 0; d < dim; ++d) {
        sum += static_cast<uint32_t>(a[d]) * static_cast<uint32_t>(b[d]);
    }
    return sum;
}

static inline uint32_t dot_u8_simd(const uint8_t* a, const uint8_t* b, size_t dim)
{
#if defined(ANN_USE_NEON)
    uint32x4_t acc = vdupq_n_u32(0);
    size_t d = 0;
    for (; d + 16 <= dim; d += 16) {
        uint8x16_t va = vld1q_u8(a + d);
        uint8x16_t vb = vld1q_u8(b + d);

        uint16x8_t prod0 = vmull_u8(vget_low_u8(va), vget_low_u8(vb));
        uint16x8_t prod1 = vmull_u8(vget_high_u8(va), vget_high_u8(vb));

        acc = vaddq_u32(acc, vpaddlq_u16(prod0));
        acc = vaddq_u32(acc, vpaddlq_u16(prod1));
    }

#if defined(__aarch64__)
    uint32_t sum = vaddvq_u32(acc);
#else
    uint64x2_t pair64 = vpaddlq_u32(acc);
    uint64_t tmp[2];
    vst1q_u64(tmp, pair64);
    uint32_t sum = static_cast<uint32_t>(tmp[0] + tmp[1]);
#endif
    for (; d < dim; ++d) {
        sum += static_cast<uint32_t>(a[d]) * static_cast<uint32_t>(b[d]);
    }
    return sum;
#else
    return dot_u8_scalar(a, b, dim);
#endif
}

static inline AnnQueue rerank_float_candidates(
    float* base,
    float* query,
    size_t vecdim,
    size_t k,
    AnnQueue candidates)
{
    AnnQueue result;
    while (!candidates.empty()) {
        uint32_t id = candidates.top().second;
        candidates.pop();
        float dis = 1.0f - inner_product_simd(base + 1ll * id * vecdim, query, vecdim);
        push_topk(result, dis, id, k);
    }
    return result;
}

static inline AnnQueue flat_search_sq_simd(float* base, float* query, size_t base_number, size_t vecdim, size_t k)
{
    build_sq_index(base, base_number, vecdim);

    std::vector<uint8_t> qcode(vecdim);
    quantize_vector_u8(query, qcode.data(), g_sq_index.minv, g_sq_index.scale, vecdim);

    AnnQueue candidates;
    int top_p = effective_top_p(base_number, k);
    for (size_t i = 0; i < base_number; ++i) {
        uint32_t ip = dot_u8_simd(g_sq_index.codes.data() + i * vecdim, qcode.data(), vecdim);
        float approx_dis = -static_cast<float>(ip);
        push_topk(candidates, approx_dis, static_cast<uint32_t>(i), top_p);
    }

    return rerank_float_candidates(base, query, vecdim, k, candidates);
}

static inline float l2_sqr_scalar(const float* a, const float* b, size_t dim)
{
    float sum = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        float diff = a[d] - b[d];
        sum += diff * diff;
    }
    return sum;
}

static inline void build_pq_index(float* base, size_t base_number, size_t vecdim)
{
    int m = g_ann_config.pq_m;
    int ks = g_ann_config.pq_ks;
    if (m <= 0 || ks <= 0 || vecdim % static_cast<size_t>(m) != 0 || ks > 256) {
        m = 8;
        ks = 256;
    }
    int subdim = static_cast<int>(vecdim / m);

    if (g_pq_index.base_number == base_number && g_pq_index.vecdim == vecdim &&
        g_pq_index.m == m && g_pq_index.ks == ks) {
        return;
    }

    g_pq_index.base_number = base_number;
    g_pq_index.vecdim = vecdim;
    g_pq_index.m = m;
    g_pq_index.ks = ks;
    g_pq_index.subdim = subdim;
    g_pq_index.codebook.assign(static_cast<size_t>(m) * ks * subdim, 0.0f);
    g_pq_index.codes.assign(base_number * m, 0);
    size_t train_size = static_cast<size_t>(std::max(ks, g_ann_config.pq_train));
    train_size = std::min(train_size, base_number);
    int iters = std::max(1, g_ann_config.pq_iters);

    for (int part = 0; part < m; ++part) {
        std::vector<int> labels(train_size, 0);
        std::vector<float> sums(static_cast<size_t>(ks) * subdim, 0.0f);
        std::vector<int> counts(ks, 0);

        for (int c = 0; c < ks; ++c) {
            size_t train_pos = (static_cast<size_t>(c) * train_size) / ks;
            size_t src_id = (train_pos * base_number) / train_size;
            const float* src = base + src_id * vecdim + part * subdim;
            float* dst = g_pq_index.codebook.data() + (static_cast<size_t>(part) * ks + c) * subdim;
            std::copy(src, src + subdim, dst);
        }

        for (int iter = 0; iter < iters; ++iter) {
            std::fill(sums.begin(), sums.end(), 0.0f);
            std::fill(counts.begin(), counts.end(), 0);

            for (size_t t = 0; t < train_size; ++t) {
                size_t src_id = (t * base_number) / train_size;
                const float* sub = base + src_id * vecdim + part * subdim;
                float best = std::numeric_limits<float>::max();
                int best_id = 0;

                for (int c = 0; c < ks; ++c) {
                    const float* centroid = g_pq_index.codebook.data() + (static_cast<size_t>(part) * ks + c) * subdim;
                    float dis = l2_sqr_scalar(sub, centroid, subdim);
                    if (dis < best) {
                        best = dis;
                        best_id = c;
                    }
                }

                labels[t] = best_id;
                ++counts[best_id];
                float* sum = sums.data() + static_cast<size_t>(best_id) * subdim;
                for (int d = 0; d < subdim; ++d) {
                    sum[d] += sub[d];
                }
            }

            for (int c = 0; c < ks; ++c) {
                if (counts[c] == 0) {
                    continue;
                }
                float inv_count = 1.0f / counts[c];
                float* centroid = g_pq_index.codebook.data() + (static_cast<size_t>(part) * ks + c) * subdim;
                const float* sum = sums.data() + static_cast<size_t>(c) * subdim;
                for (int d = 0; d < subdim; ++d) {
                    centroid[d] = sum[d] * inv_count;
                }
            }
        }
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(base_number); ++i) {
        for (int part = 0; part < m; ++part) {
            const float* sub = base + 1ll * i * vecdim + part * subdim;
            float best = std::numeric_limits<float>::max();
            int best_id = 0;
            for (int c = 0; c < ks; ++c) {
                const float* centroid = g_pq_index.codebook.data() + (static_cast<size_t>(part) * ks + c) * subdim;
                float dis = l2_sqr_scalar(sub, centroid, subdim);
                if (dis < best) {
                    best = dis;
                    best_id = c;
                }
            }
            g_pq_index.codes[static_cast<size_t>(i) * m + part] = static_cast<uint8_t>(best_id);
        }
    }
}

static inline AnnQueue flat_search_pq_simd(float* base, float* query, size_t base_number, size_t vecdim, size_t k)
{
    build_pq_index(base, base_number, vecdim);

    int m = g_pq_index.m;
    int ks = g_pq_index.ks;
    int subdim = g_pq_index.subdim;
    std::vector<float> lut(static_cast<size_t>(m) * ks);

    for (int part = 0; part < m; ++part) {
        const float* qsub = query + part * subdim;
        for (int c = 0; c < ks; ++c) {
            const float* centroid = g_pq_index.codebook.data() + (static_cast<size_t>(part) * ks + c) * subdim;
            lut[static_cast<size_t>(part) * ks + c] = l2_sqr_scalar(qsub, centroid, subdim);
        }
    }

    AnnQueue candidates;
    int top_p = effective_top_p(base_number, k);
    for (size_t i = 0; i < base_number; ++i) {
        float approx_dis = 0.0f;
        const uint8_t* code = g_pq_index.codes.data() + i * m;
        for (int part = 0; part < m; ++part) {
            approx_dis += lut[static_cast<size_t>(part) * ks + code[part]];
        }
        push_topk(candidates, approx_dis, static_cast<uint32_t>(i), top_p);
    }

    return rerank_float_candidates(base, query, vecdim, k, candidates);
}

static inline AnnQueue flat_search_pq_openmp_simd(float* base, float* query, size_t base_number, size_t vecdim, size_t k)
{
    build_pq_index(base, base_number, vecdim);

    int m = g_pq_index.m;
    int ks = g_pq_index.ks;
    int subdim = g_pq_index.subdim;
    std::vector<float> lut(static_cast<size_t>(m) * ks);

    for (int part = 0; part < m; ++part) {
        const float* qsub = query + part * subdim;
        for (int c = 0; c < ks; ++c) {
            const float* centroid = g_pq_index.codebook.data() + (static_cast<size_t>(part) * ks + c) * subdim;
            lut[static_cast<size_t>(part) * ks + c] = l2_sqr_scalar(qsub, centroid, subdim);
        }
    }

    int max_threads = omp_get_max_threads();
    int top_p = effective_top_p(base_number, k);
    std::vector<AnnQueue> local(max_threads);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
#pragma omp for schedule(runtime)
        for (int i = 0; i < static_cast<int>(base_number); ++i) {
            float approx_dis = 0.0f;
            const uint8_t* code = g_pq_index.codes.data() + static_cast<size_t>(i) * m;
            for (int part = 0; part < m; ++part) {
                approx_dis += lut[static_cast<size_t>(part) * ks + code[part]];
            }
            push_topk(local[tid], approx_dis, static_cast<uint32_t>(i), top_p);
        }
    }

    AnnQueue candidates;
    for (int t = 0; t < max_threads; ++t) {
        while (!local[t].empty()) {
            auto item = local[t].top();
            local[t].pop();
            push_topk(candidates, item.first, item.second, top_p);
        }
    }

    return rerank_float_candidates(base, query, vecdim, k, candidates);
}

static inline AnnQueue flat_search_simd(float* base, float* query, size_t base_number, size_t vecdim, size_t k)
{
    AnnQueue q;
    for (size_t i = 0; i < base_number; ++i) {
        float dis = 1.0f - inner_product_simd(base + i * vecdim, query, vecdim);
        push_topk(q, dis, static_cast<uint32_t>(i), k);
    }
    return q;
}

static inline AnnQueue flat_search_openmp(float* base, float* query, size_t base_number, size_t vecdim, size_t k)
{
    int max_threads = omp_get_max_threads();
    std::vector<AnnQueue> local(max_threads);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
#pragma omp for schedule(runtime)
        for (int i = 0; i < static_cast<int>(base_number); ++i) {
            float dis = 1.0f - inner_product_scalar(base + 1ll * i * vecdim, query, vecdim);
            push_topk(local[tid], dis, static_cast<uint32_t>(i), k);
        }
    }

    AnnQueue merged;
    for (int t = 0; t < max_threads; ++t) {
        while (!local[t].empty()) {
            auto item = local[t].top();
            local[t].pop();
            push_topk(merged, item.first, item.second, k);
        }
    }
    return merged;
}

static inline AnnQueue flat_search_openmp_simd(float* base, float* query, size_t base_number, size_t vecdim, size_t k)
{
    int max_threads = omp_get_max_threads();
    std::vector<AnnQueue> local(max_threads);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
#pragma omp for schedule(runtime)
        for (int i = 0; i < static_cast<int>(base_number); ++i) {
            float dis = 1.0f - inner_product_simd(base + 1ll * i * vecdim, query, vecdim);
            push_topk(local[tid], dis, static_cast<uint32_t>(i), k);
        }
    }

    AnnQueue merged;
    for (int t = 0; t < max_threads; ++t) {
        while (!local[t].empty()) {
            auto item = local[t].top();
            local[t].pop();
            push_topk(merged, item.first, item.second, k);
        }
    }
    return merged;
}

static inline int read_int_env(const char* name, int default_value)
{
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return default_value;
    }
    return std::atoi(value);
}

static inline void load_ann_config_file(AnnSearchConfig& config)
{
    std::ifstream fin("files/ann_config.txt");
    if (!fin.good()) {
        return;
    }

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        for (char& c : line) {
            if (c == '=') {
                c = ' ';
            }
        }

        std::istringstream iss(line);
        std::string key;
        std::string value;
        if (!(iss >> key >> value)) {
            continue;
        }

        if (key == "mode") {
            config.mode = std::atoi(value.c_str());
        } else if (key == "threads") {
            config.threads = std::atoi(value.c_str());
        } else if (key == "chunk") {
            config.chunk = std::atoi(value.c_str());
        } else if (key == "top_p") {
            config.top_p = std::atoi(value.c_str());
        } else if (key == "pq_m") {
            config.pq_m = std::atoi(value.c_str());
        } else if (key == "pq_ks") {
            config.pq_ks = std::atoi(value.c_str());
        } else if (key == "pq_iters") {
            config.pq_iters = std::atoi(value.c_str());
        } else if (key == "pq_train") {
            config.pq_train = std::atoi(value.c_str());
        } else if (key == "schedule") {
            config.schedule = value;
        }
    }
}

static inline void configure_ann_search_from_env()
{
    load_ann_config_file(g_ann_config);

    const char* mode_env = std::getenv("ANN_MODE");
    const char* threads_env = std::getenv("ANN_THREADS");
    const char* chunk_env = std::getenv("ANN_CHUNK");
    const char* top_p_env = std::getenv("ANN_TOP_P");
    const char* pq_m_env = std::getenv("ANN_PQ_M");
    const char* pq_ks_env = std::getenv("ANN_PQ_KS");
    const char* pq_iters_env = std::getenv("ANN_PQ_ITERS");
    const char* pq_train_env = std::getenv("ANN_PQ_TRAIN");
    const char* schedule_env = std::getenv("ANN_SCHEDULE");

    if (mode_env && *mode_env) {
        g_ann_config.mode = std::atoi(mode_env);
    }
    if (threads_env && *threads_env) {
        g_ann_config.threads = std::atoi(threads_env);
    }
    if (chunk_env && *chunk_env) {
        g_ann_config.chunk = std::atoi(chunk_env);
    }
    if (top_p_env && *top_p_env) {
        g_ann_config.top_p = std::atoi(top_p_env);
    }
    if (pq_m_env && *pq_m_env) {
        g_ann_config.pq_m = std::atoi(pq_m_env);
    }
    if (pq_ks_env && *pq_ks_env) {
        g_ann_config.pq_ks = std::atoi(pq_ks_env);
    }
    if (pq_iters_env && *pq_iters_env) {
        g_ann_config.pq_iters = std::atoi(pq_iters_env);
    }
    if (pq_train_env && *pq_train_env) {
        g_ann_config.pq_train = std::atoi(pq_train_env);
    }
    if (schedule_env && *schedule_env) {
        g_ann_config.schedule = schedule_env;
    }

    if (g_ann_config.threads > 0) {
        omp_set_num_threads(g_ann_config.threads);
    }

    if (g_ann_config.schedule == "dynamic") {
        omp_set_schedule(omp_sched_dynamic, g_ann_config.chunk);
    } else if (g_ann_config.schedule == "guided") {
        omp_set_schedule(omp_sched_guided, g_ann_config.chunk);
    } else {
        omp_set_schedule(omp_sched_static, g_ann_config.chunk);
    }
}

static inline AnnQueue ann_search_dispatch(float* base, float* query, size_t base_number, size_t vecdim, size_t k)
{
    switch (g_ann_config.mode) {
        case 1:
            return flat_search_simd(base, query, base_number, vecdim, k);
        case 2:
            return flat_search_openmp(base, query, base_number, vecdim, k);
        case 3:
            return flat_search_openmp_simd(base, query, base_number, vecdim, k);
        case 4:
            return flat_search_sq_simd(base, query, base_number, vecdim, k);
        case 5:
            return flat_search_pq_simd(base, query, base_number, vecdim, k);
        case 6:
            return flat_search_pq_openmp_simd(base, query, base_number, vecdim, k);
        default:
            return flat_search(base, query, base_number, vecdim, k);
    }
}

static inline void ann_prepare_index(float* base, size_t base_number, size_t vecdim)
{
    if (g_ann_config.mode == 4) {
        build_sq_index(base, base_number, vecdim);
    } else if (g_ann_config.mode == 5 || g_ann_config.mode == 6) {
        build_pq_index(base, base_number, vecdim);
    }
}

static inline void print_ann_search_config()
{
    std::cerr << "ANN_MODE: " << g_ann_config.mode
              << "  ANN_THREADS: " << omp_get_max_threads()
              << "  ANN_SCHEDULE: " << g_ann_config.schedule
              << "  ANN_CHUNK: " << g_ann_config.chunk
              << "  ANN_TOP_P: " << g_ann_config.top_p
              << "  ANN_PQ_M: " << g_ann_config.pq_m
              << "  ANN_PQ_KS: " << g_ann_config.pq_ks
              << "  ANN_PQ_ITERS: " << g_ann_config.pq_iters
              << "  ANN_PQ_TRAIN: " << g_ann_config.pq_train
              << "\n";
}
