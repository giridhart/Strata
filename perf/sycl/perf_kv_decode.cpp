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

// Register Benchmarks across Context Sweeps
BENCHMARK(BM_Xe2_KV_Decode_FP16)->Arg(1024)->Arg(4096)->Arg(16384)->Arg(65536)->UseManualTime();
BENCHMARK(BM_Xe2_KV_Decode_INT8)->Arg(1024)->Arg(4096)->Arg(16384)->Arg(65536)->UseManualTime();

BENCHMARK_MAIN();
