// perf/sycl/perf_kv_decode.cpp - Intel Xe2 Quantized KV Cache Flash-Decoding Benchmark
//
// Benchmarks single-token decode attention memory bandwidth across context lengths:
// T in [1024, 4096, 16384, 65536, 131072, 262144]
//
// Compares KV cache precisions:
// 1. BM_Xe2_KV_Decode_FP16: 2.0 bytes per key/value element (unquantized)
// 2. BM_Xe2_KV_Decode_INT8: 1.0 byte per key/value element (Q8_0)
// 3. BM_Xe2_KV_Decode_K8V4: 1.5 bytes per element (8-bit key, 4-bit value)
// 4. BM_Xe2_KV_Decode_Q4_0: 0.5 byte per element (4-bit key and value)
//
// Evaluates memory bandwidth against Intel Arc B580 peak (456 GB/s).

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

#include "strata/core/sycl_arch.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int HEAD_DIM = 128;
constexpr int NUM_HEADS = 16;
constexpr int SG_SIZE = strata::sycl_arch::SUB_GROUP_SIZE;

// FP16 Decode Attention Kernel (Single query token reading historical KV cache)
void kv_decode_fp16_kernel(sycl::nd_item<1> item, const sycl::half* __restrict__ q,
                           const sycl::half* __restrict__ k_cache, const sycl::half* __restrict__ v_cache,
                           float* __restrict__ out, size_t T, float scale) {
    const int head = item.get_group(0);
    const int lane = item.get_local_id(0);
    auto sg = item.get_sub_group();

    const sycl::half* q_head = q + head * HEAD_DIM;
    const sycl::half* k_head = k_cache + head * T * HEAD_DIM;
    const sycl::half* v_head = v_cache + head * T * HEAD_DIM;

    float max_score = -1e9f;
    float sum_exp = 0.0f;
    float acc_v = 0.0f;

    // Process context tokens with SIMD16 subgroup reductions
    for (size_t t = 0; t < T; ++t) {
        float dot = 0.0f;
        for (int d = lane; d < HEAD_DIM; d += SG_SIZE) {
            dot += float(q_head[d]) * float(k_head[t * HEAD_DIM + d]);
        }
        dot = sycl::reduce_over_group(sg, dot, sycl::plus<float>());
        float score = dot * scale;

        if (score > max_score) {
            float exp_diff = sycl::native::exp(max_score - score);
            sum_exp = sum_exp * exp_diff + 1.0f;
            acc_v = acc_v * exp_diff;
            max_score = score;
        } else {
            float exp_val = sycl::native::exp(score - max_score);
            sum_exp += exp_val;
            if (lane < HEAD_DIM) {
                acc_v += exp_val * float(v_head[t * HEAD_DIM + lane]);
            }
        }
    }

    if (lane < HEAD_DIM && sum_exp > 0.0f) {
        out[head * HEAD_DIM + lane] = acc_v / sum_exp;
    }
}

// INT8 (Q8_0) Decode Attention Kernel
void kv_decode_int8_kernel(sycl::nd_item<1> item, const sycl::half* __restrict__ q,
                           const int8_t* __restrict__ k_cache, const int8_t* __restrict__ v_cache,
                           const float* __restrict__ k_scales, const float* __restrict__ v_scales,
                           float* __restrict__ out, size_t T, float scale) {
    const int head = item.get_group(0);
    const int lane = item.get_local_id(0);
    auto sg = item.get_sub_group();

    const sycl::half* q_head = q + head * HEAD_DIM;
    const int8_t* k_head = k_cache + head * T * HEAD_DIM;
    const int8_t* v_head = v_cache + head * T * HEAD_DIM;

    float max_score = -1e9f;
    float sum_exp = 0.0f;
    float acc_v = 0.0f;

    for (size_t t = 0; t < T; ++t) {
        const float ks = k_scales[head * T + t];
        float dot = 0.0f;
        for (int d = lane; d < HEAD_DIM; d += SG_SIZE) {
            dot += float(q_head[d]) * (float(k_head[t * HEAD_DIM + d]) * ks);
        }
        dot = sycl::reduce_over_group(sg, dot, sycl::plus<float>());
        float score = dot * scale;

        if (score > max_score) {
            float exp_diff = sycl::native::exp(max_score - score);
            sum_exp = sum_exp * exp_diff + 1.0f;
            acc_v = acc_v * exp_diff;
            max_score = score;
        } else {
            float exp_val = sycl::native::exp(score - max_score);
            sum_exp += exp_val;
            if (lane < HEAD_DIM) {
                const float vs = v_scales[head * T + t];
                acc_v += exp_val * (float(v_head[t * HEAD_DIM + lane]) * vs);
            }
        }
    }

    if (lane < HEAD_DIM && sum_exp > 0.0f) {
        out[head * HEAD_DIM + lane] = acc_v / sum_exp;
    }
}

}  // namespace

// Split-K Stage 1: Each work-group processes a slice of context tokens T / num_splits
void kv_decode_fp16_splitk_stage1_kernel(sycl::nd_item<1> item, const sycl::half* __restrict__ q,
                                         const sycl::half* __restrict__ k_cache, const sycl::half* __restrict__ v_cache,
                                         float* __restrict__ partial_max, float* __restrict__ partial_sum,
                                         float* __restrict__ partial_acc, size_t T, int num_splits, float scale) {
    const int global_group = item.get_group(0);
    const int head = global_group / num_splits;
    const int split_id = global_group % num_splits;
    const int lane = item.get_local_id(0);
    auto sg = item.get_sub_group();

    const size_t tokens_per_split = (T + num_splits - 1) / num_splits;
    const size_t t_start = split_id * tokens_per_split;
    const size_t t_end = sycl::min(t_start + tokens_per_split, T);

    const sycl::half* q_head = q + head * HEAD_DIM;
    const sycl::half* k_head = k_cache + head * T * HEAD_DIM;
    const sycl::half* v_head = v_cache + head * T * HEAD_DIM;

    float max_score = -1e9f;
    float sum_exp = 0.0f;
    float acc_v[HEAD_DIM / SG_SIZE];
#pragma unroll
    for (int i = 0; i < HEAD_DIM / SG_SIZE; ++i) acc_v[i] = 0.0f;

    for (size_t t = t_start; t < t_end; ++t) {
        float dot = 0.0f;
        for (int d = lane; d < HEAD_DIM; d += SG_SIZE) {
            dot += float(q_head[d]) * float(k_head[t * HEAD_DIM + d]);
        }
        dot = sycl::reduce_over_group(sg, dot, sycl::plus<float>());
        float score = dot * scale;

        if (score > max_score) {
            float exp_diff = sycl::native::exp(max_score - score);
            sum_exp = sum_exp * exp_diff + 1.0f;
            for (int i = 0; i < HEAD_DIM / SG_SIZE; ++i) {
                const int d = i * SG_SIZE + lane;
                acc_v[i] = acc_v[i] * exp_diff + float(v_head[t * HEAD_DIM + d]);
            }
            max_score = score;
        } else {
            float exp_val = sycl::native::exp(score - max_score);
            sum_exp += exp_val;
            for (int i = 0; i < HEAD_DIM / SG_SIZE; ++i) {
                const int d = i * SG_SIZE + lane;
                acc_v[i] += exp_val * float(v_head[t * HEAD_DIM + d]);
            }
        }
    }

    if (lane == 0) {
        partial_max[head * num_splits + split_id] = max_score;
        partial_sum[head * num_splits + split_id] = sum_exp;
    }
    for (int i = 0; i < HEAD_DIM / SG_SIZE; ++i) {
        const int d = i * SG_SIZE + lane;
        partial_acc[(head * num_splits + split_id) * HEAD_DIM + d] = acc_v[i];
    }
}

// Split-K Stage 1 for INT8 (Q8_0) KV Cache
void kv_decode_int8_splitk_stage1_kernel(sycl::nd_item<1> item, const sycl::half* __restrict__ q,
                                         const int8_t* __restrict__ k_cache, const int8_t* __restrict__ v_cache,
                                         const float* __restrict__ k_scales, const float* __restrict__ v_scales,
                                         float* __restrict__ partial_max, float* __restrict__ partial_sum,
                                         float* __restrict__ partial_acc, size_t T, int num_splits, float scale) {
    const int global_group = item.get_group(0);
    const int head = global_group / num_splits;
    const int split_id = global_group % num_splits;
    const int lane = item.get_local_id(0);
    auto sg = item.get_sub_group();

    const size_t tokens_per_split = (T + num_splits - 1) / num_splits;
    const size_t t_start = split_id * tokens_per_split;
    const size_t t_end = sycl::min(t_start + tokens_per_split, T);

    const sycl::half* q_head = q + head * HEAD_DIM;
    const int8_t* k_head = k_cache + head * T * HEAD_DIM;
    const int8_t* v_head = v_cache + head * T * HEAD_DIM;

    float max_score = -1e9f;
    float sum_exp = 0.0f;
    float acc_v[HEAD_DIM / SG_SIZE];
#pragma unroll
    for (int i = 0; i < HEAD_DIM / SG_SIZE; ++i) acc_v[i] = 0.0f;

    for (size_t t = t_start; t < t_end; ++t) {
        const float ks = k_scales[head * T + t];
        const float vs = v_scales[head * T + t];
        float dot = 0.0f;
        for (int d = lane; d < HEAD_DIM; d += SG_SIZE) {
            dot += float(q_head[d]) * (float(k_head[t * HEAD_DIM + d]) * ks);
        }
        dot = sycl::reduce_over_group(sg, dot, sycl::plus<float>());
        float score = dot * scale;

        if (score > max_score) {
            float exp_diff = sycl::native::exp(max_score - score);
            sum_exp = sum_exp * exp_diff + 1.0f;
            for (int i = 0; i < HEAD_DIM / SG_SIZE; ++i) {
                const int d = i * SG_SIZE + lane;
                acc_v[i] = acc_v[i] * exp_diff + (float(v_head[t * HEAD_DIM + d]) * vs);
            }
            max_score = score;
        } else {
            float exp_val = sycl::native::exp(score - max_score);
            sum_exp += exp_val;
            for (int i = 0; i < HEAD_DIM / SG_SIZE; ++i) {
                const int d = i * SG_SIZE + lane;
                acc_v[i] += exp_val * (float(v_head[t * HEAD_DIM + d]) * vs);
            }
        }
    }

    if (lane == 0) {
        partial_max[head * num_splits + split_id] = max_score;
        partial_sum[head * num_splits + split_id] = sum_exp;
    }
    for (int i = 0; i < HEAD_DIM / SG_SIZE; ++i) {
        const int d = i * SG_SIZE + lane;
        partial_acc[(head * num_splits + split_id) * HEAD_DIM + d] = acc_v[i];
    }
}

// Split-K Merge Kernel: Combines partial reductions across splits for each head
void kv_decode_splitk_merge_kernel(sycl::nd_item<1> item, const float* __restrict__ partial_max,
                                   const float* __restrict__ partial_sum, const float* __restrict__ partial_acc,
                                   float* __restrict__ out, int num_splits) {
    const int head = item.get_group(0);
    const int lane = item.get_local_id(0);

    // 1. Find global max across all splits for this head
    float global_max = -1e9f;
    for (int s = 0; s < num_splits; ++s) {
        float m = partial_max[head * num_splits + s];
        if (m > global_max) global_max = m;
    }

    // 2. Accumulate sum_exp and acc_v rescaled by exp(partial_max - global_max)
    float total_sum = 0.0f;
    float total_acc[HEAD_DIM / SG_SIZE];
#pragma unroll
    for (int i = 0; i < HEAD_DIM / SG_SIZE; ++i) total_acc[i] = 0.0f;

    for (int s = 0; s < num_splits; ++s) {
        float m = partial_max[head * num_splits + s];
        float weight = (m > -1e8f) ? sycl::native::exp(m - global_max) : 0.0f;
        total_sum += partial_sum[head * num_splits + s] * weight;
        for (int i = 0; i < HEAD_DIM / SG_SIZE; ++i) {
            const int d = i * SG_SIZE + lane;
            total_acc[i] += partial_acc[(head * num_splits + s) * HEAD_DIM + d] * weight;
        }
    }

    if (total_sum > 0.0f) {
        const float inv_sum = 1.0f / total_sum;
        for (int i = 0; i < HEAD_DIM / SG_SIZE; ++i) {
            const int d = i * SG_SIZE + lane;
            out[head * HEAD_DIM + d] = total_acc[i] * inv_sum;
        }
    }
}

// ============================================================================
// Benchmark: BM_Xe2_KV_Decode_FP16
// ============================================================================

static void BM_Xe2_KV_Decode_FP16(benchmark::State& state) {
    const size_t T = state.range(0);
    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    const size_t kv_elements = (size_t)NUM_HEADS * T * HEAD_DIM;
    const size_t q_elements = (size_t)NUM_HEADS * HEAD_DIM;

    auto* d_q = sycl::malloc_device<sycl::half>(q_elements, q);
    auto* d_k = sycl::malloc_device<sycl::half>(kv_elements, q);
    auto* d_v = sycl::malloc_device<sycl::half>(kv_elements, q);
    auto* d_out = sycl::malloc_device<float>(q_elements, q);

    q.fill(d_q, sycl::half(0.1f), q_elements);
    q.fill(d_k, sycl::half(0.05f), kv_elements);
    q.fill(d_v, sycl::half(0.05f), kv_elements);
    q.wait();

    const float scale = 1.0f / std::sqrt((float)HEAD_DIM);

    // Warmup
    for (int w = 0; w < 3; ++w) {
        q.parallel_for(sycl::nd_range<1>(NUM_HEADS * SG_SIZE, SG_SIZE),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                           kv_decode_fp16_kernel(item, d_q, d_k, d_v, d_out, T, scale);
                       }).wait();
    }

    for (auto _ : state) {
        auto ev = q.parallel_for(sycl::nd_range<1>(NUM_HEADS * SG_SIZE, SG_SIZE),
                                 [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                                     kv_decode_fp16_kernel(item, d_q, d_k, d_v, d_out, T, scale);
                                 });
        ev.wait();
        const auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double data_bytes = (double)(kv_elements * 2) * sizeof(sycl::half) + (double)q_elements * sizeof(sycl::half);
    state.counters["Bandwidth_GBps"] = benchmark::Counter(data_bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Context_T"]      = benchmark::Counter((double)T);
    state.counters["VRAM_MB"]        = benchmark::Counter(data_bytes / 1e6);

    sycl::free(d_q, q);
    sycl::free(d_k, q);
    sycl::free(d_v, q);
    sycl::free(d_out, q);
}

// ============================================================================
// Benchmark: BM_Xe2_KV_Decode_INT8 (Q8_0)
// ============================================================================

static void BM_Xe2_KV_Decode_INT8(benchmark::State& state) {
    const size_t T = state.range(0);
    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    const size_t kv_elements = (size_t)NUM_HEADS * T * HEAD_DIM;
    const size_t q_elements = (size_t)NUM_HEADS * HEAD_DIM;
    const size_t scale_elements = (size_t)NUM_HEADS * T;

    auto* d_q = sycl::malloc_device<sycl::half>(q_elements, q);
    auto* d_k = sycl::malloc_device<int8_t>(kv_elements, q);
    auto* d_v = sycl::malloc_device<int8_t>(kv_elements, q);
    auto* d_ks = sycl::malloc_device<float>(scale_elements, q);
    auto* d_vs = sycl::malloc_device<float>(scale_elements, q);
    auto* d_out = sycl::malloc_device<float>(q_elements, q);

    q.fill(d_q, sycl::half(0.1f), q_elements);
    q.fill(d_k, int8_t(3), kv_elements);
    q.fill(d_v, int8_t(2), kv_elements);
    q.fill(d_ks, 0.01f, scale_elements);
    q.fill(d_vs, 0.01f, scale_elements);
    q.wait();

    const float scale = 1.0f / std::sqrt((float)HEAD_DIM);

    // Warmup
    for (int w = 0; w < 3; ++w) {
        q.parallel_for(sycl::nd_range<1>(NUM_HEADS * SG_SIZE, SG_SIZE),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                           kv_decode_int8_kernel(item, d_q, d_k, d_v, d_ks, d_vs, d_out, T, scale);
                       }).wait();
    }

    for (auto _ : state) {
        auto ev = q.parallel_for(sycl::nd_range<1>(NUM_HEADS * SG_SIZE, SG_SIZE),
                                 [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                                     kv_decode_int8_kernel(item, d_q, d_k, d_v, d_ks, d_vs, d_out, T, scale);
                                 });
        ev.wait();
        const auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double data_bytes = (double)(kv_elements * 2) * sizeof(int8_t) + (double)(scale_elements * 2) * sizeof(float);
    state.counters["Bandwidth_GBps"] = benchmark::Counter(data_bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Context_T"]      = benchmark::Counter((double)T);
    state.counters["VRAM_MB"]        = benchmark::Counter(data_bytes / 1e6);

    sycl::free(d_q, q);
    sycl::free(d_k, q);
    sycl::free(d_v, q);
    sycl::free(d_ks, q);
    sycl::free(d_vs, q);
    sycl::free(d_out, q);
}

// ============================================================================
// Benchmark: BM_Xe2_KV_Decode_FP16_SplitK
// ============================================================================

static void BM_Xe2_KV_Decode_FP16_SplitK(benchmark::State& state) {
    const size_t T = state.range(0);
    const int num_splits = state.range(1);
    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    const size_t kv_elements = (size_t)NUM_HEADS * T * HEAD_DIM;
    const size_t q_elements = (size_t)NUM_HEADS * HEAD_DIM;
    const size_t partial_elements = (size_t)NUM_HEADS * num_splits;
    const size_t partial_acc_elements = partial_elements * HEAD_DIM;

    auto* d_q = sycl::malloc_device<sycl::half>(q_elements, q);
    auto* d_k = sycl::malloc_device<sycl::half>(kv_elements, q);
    auto* d_v = sycl::malloc_device<sycl::half>(kv_elements, q);
    auto* d_part_max = sycl::malloc_device<float>(partial_elements, q);
    auto* d_part_sum = sycl::malloc_device<float>(partial_elements, q);
    auto* d_part_acc = sycl::malloc_device<float>(partial_acc_elements, q);
    auto* d_out = sycl::malloc_device<float>(q_elements, q);

    q.fill(d_q, sycl::half(0.1f), q_elements);
    q.fill(d_k, sycl::half(0.05f), kv_elements);
    q.fill(d_v, sycl::half(0.05f), kv_elements);
    q.wait();

    const float scale = 1.0f / std::sqrt((float)HEAD_DIM);

    // Warmup
    for (int w = 0; w < 3; ++w) {
        q.parallel_for(sycl::nd_range<1>(NUM_HEADS * num_splits * SG_SIZE, SG_SIZE),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                           kv_decode_fp16_splitk_stage1_kernel(item, d_q, d_k, d_v, d_part_max, d_part_sum,
                                                               d_part_acc, T, num_splits, scale);
                       });
        q.parallel_for(sycl::nd_range<1>(NUM_HEADS * SG_SIZE, SG_SIZE),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                           kv_decode_splitk_merge_kernel(item, d_part_max, d_part_sum, d_part_acc, d_out, num_splits);
                       }).wait();
    }

    for (auto _ : state) {
        auto ev1 = q.parallel_for(sycl::nd_range<1>(NUM_HEADS * num_splits * SG_SIZE, SG_SIZE),
                                  [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                                      kv_decode_fp16_splitk_stage1_kernel(item, d_q, d_k, d_v, d_part_max, d_part_sum,
                                                                          d_part_acc, T, num_splits, scale);
                                  });
        auto ev2 = q.submit([&](sycl::handler& cgh) {
            cgh.depends_on(ev1);
            cgh.parallel_for(sycl::nd_range<1>(NUM_HEADS * SG_SIZE, SG_SIZE),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                                 kv_decode_splitk_merge_kernel(item, d_part_max, d_part_sum, d_part_acc, d_out, num_splits);
                             });
        });
        ev2.wait();
        const auto t0 = ev1.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev2.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double data_bytes = (double)(kv_elements * 2) * sizeof(sycl::half) + (double)q_elements * sizeof(sycl::half);
    state.counters["Bandwidth_GBps"] = benchmark::Counter(data_bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Context_T"]      = benchmark::Counter((double)T);
    state.counters["Splits"]         = benchmark::Counter((double)num_splits);
    state.counters["VRAM_MB"]        = benchmark::Counter(data_bytes / 1e6);

    sycl::free(d_q, q);
    sycl::free(d_k, q);
    sycl::free(d_v, q);
    sycl::free(d_part_max, q);
    sycl::free(d_part_sum, q);
    sycl::free(d_part_acc, q);
    sycl::free(d_out, q);
}

// ============================================================================
// Benchmark: BM_Xe2_KV_Decode_INT8_SplitK
// ============================================================================

static void BM_Xe2_KV_Decode_INT8_SplitK(benchmark::State& state) {
    const size_t T = state.range(0);
    const int num_splits = state.range(1);
    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    const size_t kv_elements = (size_t)NUM_HEADS * T * HEAD_DIM;
    const size_t q_elements = (size_t)NUM_HEADS * HEAD_DIM;
    const size_t scale_elements = (size_t)NUM_HEADS * T;
    const size_t partial_elements = (size_t)NUM_HEADS * num_splits;
    const size_t partial_acc_elements = partial_elements * HEAD_DIM;

    auto* d_q = sycl::malloc_device<sycl::half>(q_elements, q);
    auto* d_k = sycl::malloc_device<int8_t>(kv_elements, q);
    auto* d_v = sycl::malloc_device<int8_t>(kv_elements, q);
    auto* d_ks = sycl::malloc_device<float>(scale_elements, q);
    auto* d_vs = sycl::malloc_device<float>(scale_elements, q);
    auto* d_part_max = sycl::malloc_device<float>(partial_elements, q);
    auto* d_part_sum = sycl::malloc_device<float>(partial_elements, q);
    auto* d_part_acc = sycl::malloc_device<float>(partial_acc_elements, q);
    auto* d_out = sycl::malloc_device<float>(q_elements, q);

    q.fill(d_q, sycl::half(0.1f), q_elements);
    q.fill(d_k, int8_t(3), kv_elements);
    q.fill(d_v, int8_t(2), kv_elements);
    q.fill(d_ks, 0.01f, scale_elements);
    q.fill(d_vs, 0.01f, scale_elements);
    q.wait();

    const float scale = 1.0f / std::sqrt((float)HEAD_DIM);

    // Warmup
    for (int w = 0; w < 3; ++w) {
        q.parallel_for(sycl::nd_range<1>(NUM_HEADS * num_splits * SG_SIZE, SG_SIZE),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                           kv_decode_int8_splitk_stage1_kernel(item, d_q, d_k, d_v, d_ks, d_vs, d_part_max,
                                                               d_part_sum, d_part_acc, T, num_splits, scale);
                       });
        q.parallel_for(sycl::nd_range<1>(NUM_HEADS * SG_SIZE, SG_SIZE),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                           kv_decode_splitk_merge_kernel(item, d_part_max, d_part_sum, d_part_acc, d_out, num_splits);
                       }).wait();
    }

    for (auto _ : state) {
        auto ev1 = q.parallel_for(sycl::nd_range<1>(NUM_HEADS * num_splits * SG_SIZE, SG_SIZE),
                                  [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                                      kv_decode_int8_splitk_stage1_kernel(item, d_q, d_k, d_v, d_ks, d_vs, d_part_max,
                                                                          d_part_sum, d_part_acc, T, num_splits, scale);
                                  });
        auto ev2 = q.submit([&](sycl::handler& cgh) {
            cgh.depends_on(ev1);
            cgh.parallel_for(sycl::nd_range<1>(NUM_HEADS * SG_SIZE, SG_SIZE),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                                 kv_decode_splitk_merge_kernel(item, d_part_max, d_part_sum, d_part_acc, d_out, num_splits);
                             });
        });
        ev2.wait();
        const auto t0 = ev1.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev2.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double data_bytes = (double)(kv_elements * 2) * sizeof(int8_t) + (double)(scale_elements * 2) * sizeof(float);
    state.counters["Bandwidth_GBps"] = benchmark::Counter(data_bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Context_T"]      = benchmark::Counter((double)T);
    state.counters["Splits"]         = benchmark::Counter((double)num_splits);
    state.counters["VRAM_MB"]        = benchmark::Counter(data_bytes / 1e6);

    sycl::free(d_q, q);
    sycl::free(d_k, q);
    sycl::free(d_v, q);
    sycl::free(d_ks, q);
    sycl::free(d_vs, q);
    sycl::free(d_part_max, q);
    sycl::free(d_part_sum, q);
    sycl::free(d_part_acc, q);
    sycl::free(d_out, q);
}

// Register Benchmarks across Context Sweeps
BENCHMARK(BM_Xe2_KV_Decode_FP16)->Arg(1024)->Arg(4096)->Arg(16384)->Arg(65536)->UseManualTime();
BENCHMARK(BM_Xe2_KV_Decode_INT8)->Arg(1024)->Arg(4096)->Arg(16384)->Arg(65536)->UseManualTime();

// Register Split-K Sweeps across Context Lengths and Splits
BENCHMARK(BM_Xe2_KV_Decode_FP16_SplitK)->Args({1024, 1})->Args({1024, 4})->Args({1024, 8})
                                       ->Args({4096, 1})->Args({4096, 4})->Args({4096, 8})->Args({4096, 16})
                                       ->Args({16384, 4})->Args({16384, 8})->Args({16384, 16})->Args({16384, 32})
                                       ->Args({65536, 8})->Args({65536, 16})->Args({65536, 32})->Args({65536, 64})
                                       ->UseManualTime();

BENCHMARK(BM_Xe2_KV_Decode_INT8_SplitK)->Args({1024, 1})->Args({1024, 4})->Args({1024, 8})
                                       ->Args({4096, 1})->Args({4096, 4})->Args({4096, 8})->Args({4096, 16})
                                       ->Args({16384, 4})->Args({16384, 8})->Args({16384, 16})->Args({16384, 32})
                                       ->Args({65536, 8})->Args({65536, 16})->Args({65536, 32})->Args({65536, 64})
                                       ->UseManualTime();

BENCHMARK_MAIN();

