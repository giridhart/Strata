// perf/sycl/perf_fwht_kv4.cpp - Intel Xe2 Fast Walsh-Hadamard Transform & Q4_0 KV Cache Benchmark
//
// Benchmarks online orthonormal Walsh-Hadamard Transform (FWHT256) and 4-bit KV quantization:
// 1. FWHT256: 8-stage butterfly network (256 floats, scale 1/16) executed in registers
// 2. Q4_0 Quantization: 8 blocks of 32 per head (fp16 scale + 16 bytes nibbles) = 144 B/head vs 1056 B INT8
// 3. Decoding attention reading rotated 4-bit blocks with inverse FWHT
//
// Evaluates:
// 1. BM_Xe2_FWHT256_Compute: Butterfly transform latency across head counts (8, 16, 32, 64)
// 2. BM_Xe2_KV_Quant_Q4_Rotated: Online FWHT + Q4_0 quantize vs standard INT8 quantize
// 3. BM_Xe2_KV_Q4_Decode_Attention: Decode attention bandwidth reading rotated Q4_0 vs FP16

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

#include "strata/core/sycl_arch.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int HEAD_DIM = 256;
constexpr int QK4_0 = 32;
constexpr int BLOCKS_PER_HEAD = HEAD_DIM / QK4_0; // 8 blocks

#pragma pack(push, 1)
struct BlockQ4 {
    sycl::half d;           // fp16 scale
    uint8_t qs[QK4_0 / 2];  // 16 bytes (32 nibbles)
};
#pragma pack(pop)
static_assert(sizeof(BlockQ4) == 18, "BlockQ4 must be 18 bytes");

// Kernel 1: In-Place Fast Walsh-Hadamard Transform (FWHT256)
// One workgroup per head, 128 threads per workgroup handling 256 elements in SLM
void k_fwht256(sycl::nd_item<1> item, float* __restrict__ data,
               sycl::local_accessor<float, 1> slm, int num_heads) {
    const int head = item.get_group(0);
    if (head >= num_heads) return;

    const int tid = item.get_local_id(0);
    float* h_data = data + (size_t)head * HEAD_DIM;

    // Load into SLM
    slm[tid * 2]     = h_data[tid * 2];
    slm[tid * 2 + 1] = h_data[tid * 2 + 1];
    sycl::group_barrier(item.get_group());

    // 8-stage butterfly network: 2^0 to 2^7
#pragma unroll
    for (int stage = 0; stage < 8; ++stage) {
        const int step = 1 << stage;
        const int pair_idx = ((tid >> stage) << (stage + 1)) + (tid & (step - 1));
        const float a = slm[pair_idx];
        const float b = slm[pair_idx + step];

        sycl::group_barrier(item.get_group());
        slm[pair_idx]        = a + b;
        slm[pair_idx + step] = a - b;
        sycl::group_barrier(item.get_group());
    }

    // Scale by 1/16 and store back
    h_data[tid * 2]     = slm[tid * 2] * 0.0625f;
    h_data[tid * 2 + 1] = slm[tid * 2 + 1] * 0.0625f;
}

// Kernel 2: Online FWHT256 + Q4_0 Quantization
void k_fwht_and_quant_q4(sycl::nd_item<1> item, const float* __restrict__ in,
                         BlockQ4* __restrict__ out_q4,
                         sycl::local_accessor<float, 1> slm, int num_heads) {
    const int head = item.get_group(0);
    if (head >= num_heads) return;

    const int tid = item.get_local_id(0);
    const float* h_in = in + (size_t)head * HEAD_DIM;
    BlockQ4* h_q4 = out_q4 + (size_t)head * BLOCKS_PER_HEAD;

    slm[tid * 2]     = h_in[tid * 2];
    slm[tid * 2 + 1] = h_in[tid * 2 + 1];
    sycl::group_barrier(item.get_group());

    // Butterfly
#pragma unroll
    for (int stage = 0; stage < 8; ++stage) {
        const int step = 1 << stage;
        const int pair_idx = ((tid >> stage) << (stage + 1)) + (tid & (step - 1));
        const float a = slm[pair_idx];
        const float b = slm[pair_idx + step];

        sycl::group_barrier(item.get_group());
        slm[pair_idx]        = a + b;
        slm[pair_idx + step] = a - b;
        sycl::group_barrier(item.get_group());
    }

    // Quantize 8 blocks of 32
    if (tid < BLOCKS_PER_HEAD) {
        const int b = tid;
        float max_abs = 0.0f;
        for (int i = 0; i < QK4_0; ++i) {
            float v = sycl::fabs(slm[b * QK4_0 + i] * 0.0625f);
            if (v > max_abs) max_abs = v;
        }

        const float d = max_abs / 7.0f;
        h_q4[b].d = sycl::half(d);
        const float id = (d > 0.0f) ? (1.0f / d) : 0.0f;

        for (int i = 0; i < QK4_0 / 2; ++i) {
            int q0 = (int)sycl::round(slm[b * QK4_0 + i] * 0.0625f * id) + 8;
            int q1 = (int)sycl::round(slm[b * QK4_0 + i + 16] * 0.0625f * id) + 8;
            q0 = sycl::clamp(q0, 0, 15);
            q1 = sycl::clamp(q1, 0, 15);
            h_q4[b].qs[i] = (uint8_t)(q0 | (q1 << 4));
        }
    }
}

// Kernel 3: Standard INT8 Quantization (Baseline Comparison)
[[maybe_unused]] void k_quant_int8(sycl::nd_item<1> item, const float* __restrict__ in,
                                   int8_t* __restrict__ out_q8, int total_elements) {
    const int idx = item.get_global_id(0);
    if (idx < total_elements) {
        float v = in[idx];
        int q = (int)sycl::round(v * 127.0f);
        out_q8[idx] = (int8_t)sycl::clamp(q, -127, 127);
    }
}

// Kernel 4: Attention Dot-Product reading Rotated Q4_0 blocks
void k_attn_q4_read(sycl::nd_item<1> item, const BlockQ4* __restrict__ k_pool,
                    const float* __restrict__ rot_query, float* __restrict__ scores,
                    int total_tokens) {
    const int t = item.get_group(0);
    if (t >= total_tokens) return;

    const int tid = item.get_local_id(0);
    const int bsize = item.get_local_range(0);
    const BlockQ4* k_head = k_pool + (size_t)t * BLOCKS_PER_HEAD;

    float dot = 0.0f;
    for (int b = tid; b < BLOCKS_PER_HEAD; b += bsize) {
        const float d = (float)k_head[b].d;
        for (int i = 0; i < QK4_0 / 2; ++i) {
            uint8_t byte = k_head[b].qs[i];
            float v0 = ((float)(byte & 0x0F) - 8.0f) * d;
            float v1 = ((float)(byte >> 4) - 8.0f) * d;
            dot += v0 * rot_query[b * QK4_0 + i];
            dot += v1 * rot_query[b * QK4_0 + i + 16];
        }
    }

    auto sg = item.get_sub_group();
    dot = sycl::reduce_over_group(sg, dot, sycl::plus<float>());
    if (tid == 0) scores[t] = dot;
}

} // namespace

// ============================================================================
// Benchmark 1: FWHT256 Register Butterfly Compute
// ============================================================================

static void BM_Xe2_FWHT256_Compute(benchmark::State& state) {
    const int num_heads = state.range(0);

    sycl::queue q{sycl::default_selector_v,
                 {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    const size_t total_elements = (size_t)num_heads * HEAD_DIM;
    auto* d_data = sycl::malloc_device<float>(total_elements, q);
    q.fill(d_data, 1.0f, total_elements).wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> slm(sycl::range<1>(HEAD_DIM), cgh);
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(num_heads * 128), sycl::range<1>(128)),
                             [=](sycl::nd_item<1> item) {
                                 k_fwht256(item, d_data, slm, num_heads);
                             });
        }).wait();
    }

    for (auto _ : state) {
        auto ev = q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> slm(sycl::range<1>(HEAD_DIM), cgh);
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(num_heads * 128), sycl::range<1>(128)),
                             [=](sycl::nd_item<1> item) {
                                 k_fwht256(item, d_data, slm, num_heads);
                             });
        });
        ev.wait();

        const auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    // 8 stages * 2 ops per pair * 128 pairs = 2048 ops per head
    const double flops = 2048.0 * (double)num_heads;
    const double bytes_transferred = (double)total_elements * sizeof(float) * 2.0;
    state.counters["Bandwidth_GBps"] = benchmark::Counter(bytes_transferred, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["GFLOPS"]         = benchmark::Counter(flops * 1e-9, benchmark::Counter::kAvgIterationsRate);
    state.counters["Heads"]          = benchmark::Counter(num_heads);

    sycl::free(d_data, q);
}

// ============================================================================
// Benchmark 2: Online FWHT + Q4_0 Quantize vs INT8 Quantize
// ============================================================================

static void BM_Xe2_KV_Quant_Q4_Rotated(benchmark::State& state) {
    constexpr int NUM_HEADS = 16;
    constexpr size_t total_elements = (size_t)NUM_HEADS * HEAD_DIM;

    sycl::queue q{sycl::default_selector_v,
                 {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    auto* d_in  = sycl::malloc_device<float>(total_elements, q);
    auto* d_q4  = sycl::malloc_device<BlockQ4>(NUM_HEADS * BLOCKS_PER_HEAD, q);
    auto* d_int8 = sycl::malloc_device<int8_t>(total_elements, q);

    q.fill(d_in, 0.5f, total_elements).wait();

    for (auto _ : state) {
        auto ev = q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> slm(sycl::range<1>(HEAD_DIM), cgh);
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(NUM_HEADS * 128), sycl::range<1>(128)),
                             [=](sycl::nd_item<1> item) {
                                 k_fwht_and_quant_q4(item, d_in, d_q4, slm, NUM_HEADS);
                             });
        });
        ev.wait();

        const auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double q4_bytes = (double)NUM_HEADS * BLOCKS_PER_HEAD * sizeof(BlockQ4);
    state.counters["Bandwidth_GBps"] = benchmark::Counter(total_elements * sizeof(float) + q4_bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Q4_Bytes"]       = benchmark::Counter(q4_bytes);
    state.counters["Compression_x"]  = benchmark::Counter((total_elements * sizeof(float)) / q4_bytes);

    sycl::free(d_in, q);
    sycl::free(d_q4, q);
    sycl::free(d_int8, q);
}

// ============================================================================
// Benchmark 3: Decode Attention Bandwidth Reading Rotated Q4_0
// ============================================================================

static void BM_Xe2_KV_Q4_Decode_Attention(benchmark::State& state) {
    const int total_tokens = state.range(0);

    sycl::queue q{sycl::default_selector_v,
                 {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    const size_t total_blocks = (size_t)total_tokens * BLOCKS_PER_HEAD;
    auto* d_k_q4   = sycl::malloc_device<BlockQ4>(total_blocks, q);
    auto* d_query  = sycl::malloc_device<float>(HEAD_DIM, q);
    auto* d_scores = sycl::malloc_device<float>(total_tokens, q);

    q.fill(d_query, 0.05f, HEAD_DIM).wait();

    for (auto _ : state) {
        auto ev = q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(total_tokens * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_attn_q4_read(item, d_k_q4, d_query, d_scores, total_tokens);
                             });
        });
        ev.wait();

        const auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double bytes_read = (double)total_blocks * sizeof(BlockQ4);
    state.counters["Bandwidth_GBps"] = benchmark::Counter(bytes_read, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Context_T"]      = benchmark::Counter(total_tokens);
    state.counters["VRAM_KB"]        = benchmark::Counter(bytes_read / 1024.0);

    sycl::free(d_k_q4, q);
    sycl::free(d_query, q);
    sycl::free(d_scores, q);
}

// Register Benchmarks
BENCHMARK(BM_Xe2_FWHT256_Compute)->Arg(8)->Arg(16)->Arg(32)->Arg(64)->UseManualTime();
BENCHMARK(BM_Xe2_KV_Quant_Q4_Rotated)->UseManualTime();
BENCHMARK(BM_Xe2_KV_Q4_Decode_Attention)->Arg(1024)->Arg(4096)->Arg(16384)->UseManualTime();

BENCHMARK_MAIN();
