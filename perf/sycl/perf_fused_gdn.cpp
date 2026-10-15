// perf/sycl/perf_fused_gdn.cpp - Intel Xe2 Fused GDN Recurrence Benchmark
//
// Benchmarks Gated Delta Network (GDN) linear attention recurrence:
// S_t = g S_{t-1} + k (beta (v - g S_{t-1}^T k))^T
// o   = (S_t^T q) / sqrt(128)
// y   = rmsnorm(o) * gamma * sigmoid(z)
//
// Compares:
// 1. BM_Xe2_GDN_Unfused: Separate global memory state roundtrips (conv/norm -> alpha/beta -> step)
// 2. BM_Xe2_GDN_Fused_SLM: Fused step + norm keeping intermediate state in SLM
// 3. BM_Xe2_GDN_SubgroupWidth: SIMD16 vs SIMD32 subgroup reduction and shuffles
//
// Parameters:
// - Heads: h_v = 16, h_k = 16, S = 128 (Qwen3.8-Flash-Next GDN layers)
// - State size: 16 * 128 * 128 * 4 bytes = 1,048,576 bytes (1 MB per layer)

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/prefill/kernels.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int S = 128;
constexpr int DEFAULT_HV = 16;
constexpr int DEFAULT_HK = 16;

// ============================================================================
// GDN Computational Kernels
// ============================================================================

/// 1. Unfused GDN Step (Global memory state update per thread)
template <int SUB_SIZE = 16>
void gdn_unfused_step_kernel(sycl::nd_item<2> item, float* __restrict__ state, const float* __restrict__ q,
                             const float* __restrict__ k, const float* __restrict__ v, const float* __restrict__ gate,
                             const float* __restrict__ beta, float* __restrict__ output, int h_k, int h_v,
                             float scale) {
    constexpr int R_COUNT = S / SUB_SIZE;
    const int head = item.get_group(0);
    const int lane = item.get_local_id(1);
    const int col = item.get_group(1) * 4 + item.get_local_id(0);
    const int q_head = head % h_k;

    float s_shard[R_COUNT], k_reg[R_COUNT], q_reg[R_COUNT];
#pragma unroll
    for (int r = 0; r < R_COUNT; ++r) {
        const int i = r * SUB_SIZE + lane;
        s_shard[r] = state[(size_t(i) * h_v + head) * S + col];
        k_reg[r] = k[q_head * S + i];
        q_reg[r] = q[q_head * S + i];
    }

    const float g_val = sycl::native::exp(gate[head]);
    float kv_shard = 0.0f;
#pragma unroll
    for (int r = 0; r < R_COUNT; ++r) {
        kv_shard += s_shard[r] * k_reg[r];
    }

    auto sg = item.get_sub_group();
    const float kv_col = sycl::reduce_over_group(sg, kv_shard, sycl::plus<float>());
    const float delta_col = (v[head * S + col] - g_val * kv_col) * beta[head];

    float attn_partial = 0.0f;
#pragma unroll
    for (int r = 0; r < R_COUNT; ++r) {
        s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
        attn_partial += s_shard[r] * q_reg[r];
    }

    const float attn_col = sycl::reduce_over_group(sg, attn_partial, sycl::plus<float>());
    if (lane == 0) output[head * S + col] = attn_col * scale;

#pragma unroll
    for (int r = 0; r < R_COUNT; ++r) {
        const int i = r * SUB_SIZE + lane;
        state[(size_t(i) * h_v + head) * S + col] = s_shard[r];
    }
}

/// 2. Fused GDN Step + Norm (State resident in registers, reductions in SLM)
void gdn_fused_step_norm_kernel(float* __restrict__ state, const float* __restrict__ q, const float* __restrict__ k,
                                const float* __restrict__ v, const float* __restrict__ gate,
                                const float* __restrict__ beta, const float* __restrict__ z,
                                const float* __restrict__ gamma, float eps, float* __restrict__ y, int h_k, int h_v,
                                sycl::nd_item<2> item, float* sk, float* sq,
                                sycl::local_accessor<float, 2> red, float* wsum) {
    constexpr int RG = 4;
    constexpr int RPG = S / RG;  // 32 rows per thread
    const int head = item.get_group(0);
    const int rg = item.get_local_id(0);   // 0..3
    const int col = item.get_local_id(1);  // 0..127
    const int tid = rg * S + col;
    const int qh = head % h_k;

    if (tid < S) {
        sk[tid] = k[qh * S + tid];
        sq[tid] = q[qh * S + tid];
    }

    float s[RPG];
    float* base = state + ((size_t)(rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t)h_v * S;

#pragma unroll
    for (int r = 0; r < RPG; ++r) {
        s[r] = base[r * row_stride];
    }

    item.barrier(sycl::access::fence_space::local_space);
    const float g = sycl::native::exp(gate[head]);
    float kv = 0.0f;
#pragma unroll
    for (int r = 0; r < RPG; ++r) {
        kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
    }
    red[rg][col] = kv;
    item.barrier(sycl::access::fence_space::local_space);

    const float kv_col = red[0][col] + red[1][col] + red[2][col] + red[3][col];
    const float delta = (v[head * S + col] - g * kv_col) * beta[head];

    float o = 0.0f;
#pragma unroll
    for (int r = 0; r < RPG; ++r) {
        s[r] = sycl::fma(g, s[r], sk[rg * RPG + r] * delta);
        o = sycl::fma(s[r], sq[rg * RPG + r], o);
        base[r * row_stride] = s[r];
    }
    item.barrier(sycl::access::fence_space::local_space);
    red[rg][col] = o;
    item.barrier(sycl::access::fence_space::local_space);

    float oc = 0.0f, sq_part = 0.0f;
    if (rg == 0) {
        oc = (red[0][col] + red[1][col] + red[2][col] + red[3][col]) * sycl::rsqrt((float)S);
        sq_part = oc * oc;
    }

    auto sg = item.get_sub_group();
    sq_part = sycl::reduce_over_group(sg, sq_part, sycl::plus<float>());
    if ((tid & 15) == 0) wsum[tid >> 4] = sq_part;
    item.barrier(sycl::access::fence_space::local_space);

    if (rg == 0) {
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < (512 / 16); ++i) ss += wsum[i];
        const float scale = sycl::rsqrt(ss / (float)S + eps);
        const float zz = z[head * S + col];
        y[head * S + col] = oc * scale * gamma[col] * (1.0f / (1.0f + sycl::native::exp(-zz)));
    }
}

}  // namespace

// ============================================================================
// Benchmark: BM_Xe2_GDN_Unfused (Standard Multi-Kernel Global Roundtrip)
// ============================================================================

static void BM_Xe2_GDN_Unfused(benchmark::State& state) {
    const int h_v = state.range(0);
    const int h_k = state.range(1);

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    const size_t state_elements = (size_t)S * S * h_v;
    const size_t qk_elements = (size_t)S * h_k;
    const size_t out_elements = (size_t)S * h_v;

    auto* d_state  = sycl::malloc_device<float>(state_elements, q);
    auto* d_q      = sycl::malloc_device<float>(qk_elements, q);
    auto* d_k      = sycl::malloc_device<float>(qk_elements, q);
    auto* d_v      = sycl::malloc_device<float>(out_elements, q);
    auto* d_gate   = sycl::malloc_device<float>(h_v, q);
    auto* d_beta   = sycl::malloc_device<float>(h_v, q);
    auto* d_output = sycl::malloc_device<float>(out_elements, q);

    q.fill(d_state, 0.01f, state_elements);
    q.fill(d_q, 0.05f, qk_elements);
    q.fill(d_k, 0.05f, qk_elements);
    q.fill(d_v, 0.1f, out_elements);
    q.fill(d_gate, -0.5f, h_v);
    q.fill(d_beta, 0.8f, h_v);
    q.wait();

    const float scale = 1.0f / std::sqrt((float)S);

    // Warmup
    for (int w = 0; w < 5; ++w) {
        q.parallel_for(sycl::nd_range<2>(sycl::range<2>(h_v * 4, (S / 4) * 16), sycl::range<2>(4, 16)),
                       [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                           gdn_unfused_step_kernel<16>(item, d_state, d_q, d_k, d_v, d_gate, d_beta, d_output,
                                                       h_k, h_v, scale);
                       }).wait();
    }

    for (auto _ : state) {
        auto ev = q.parallel_for(sycl::nd_range<2>(sycl::range<2>(h_v * 4, (S / 4) * 16), sycl::range<2>(4, 16)),
                                 [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                                     gdn_unfused_step_kernel<16>(item, d_state, d_q, d_k, d_v, d_gate, d_beta, d_output,
                                                                 h_k, h_v, scale);
                                 });
        ev.wait();
        const auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double bytes_transferred = (double)(state_elements * 2 + qk_elements * 2 + out_elements * 2) * sizeof(float);
    state.counters["Bandwidth_GBps"] = benchmark::Counter(bytes_transferred, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Heads_HV"]       = benchmark::Counter(h_v);
    state.counters["Heads_HK"]       = benchmark::Counter(h_k);
    state.counters["State_MB"]       = benchmark::Counter(state_elements * sizeof(float) / 1e6);

    sycl::free(d_state, q);
    sycl::free(d_q, q);
    sycl::free(d_k, q);
    sycl::free(d_v, q);
    sycl::free(d_gate, q);
    sycl::free(d_beta, q);
    sycl::free(d_output, q);
}

// ============================================================================
// Benchmark: BM_Xe2_GDN_Fused_SLM (Fused Step + Norm with Local Memory)
// ============================================================================

static void BM_Xe2_GDN_Fused_SLM(benchmark::State& state) {
    const int h_v = state.range(0);
    const int h_k = state.range(1);

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    const size_t state_elements = (size_t)S * S * h_v;
    const size_t qk_elements = (size_t)S * h_k;
    const size_t out_elements = (size_t)S * h_v;

    auto* d_state = sycl::malloc_device<float>(state_elements, q);
    auto* d_q     = sycl::malloc_device<float>(qk_elements, q);
    auto* d_k     = sycl::malloc_device<float>(qk_elements, q);
    auto* d_v     = sycl::malloc_device<float>(out_elements, q);
    auto* d_gate  = sycl::malloc_device<float>(h_v, q);
    auto* d_beta  = sycl::malloc_device<float>(h_v, q);
    auto* d_z     = sycl::malloc_device<float>(out_elements, q);
    auto* d_gamma = sycl::malloc_device<float>(S, q);
    auto* d_y     = sycl::malloc_device<float>(out_elements, q);

    q.fill(d_state, 0.01f, state_elements);
    q.fill(d_q, 0.05f, qk_elements);
    q.fill(d_k, 0.05f, qk_elements);
    q.fill(d_v, 0.1f, out_elements);
    q.fill(d_gate, -0.5f, h_v);
    q.fill(d_beta, 0.8f, h_v);
    q.fill(d_z, 0.2f, out_elements);
    q.fill(d_gamma, 1.0f, S);
    q.wait();

    // Warmup
    for (int w = 0; w < 5; ++w) {
        q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> sk(sycl::range<1>(S), cgh);
            sycl::local_accessor<float, 1> sq(sycl::range<1>(S), cgh);
            sycl::local_accessor<float, 2> red(sycl::range<2>(4, S), cgh);
            sycl::local_accessor<float, 1> wsum(sycl::range<1>(32), cgh);

            cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(h_v, 512), sycl::range<2>(4, 128)),
                             [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                                 gdn_fused_step_norm_kernel(
                                     d_state, d_q, d_k, d_v, d_gate, d_beta, d_z, d_gamma, 1e-5f, d_y, h_k, h_v, item,
                                     sk.get_multi_ptr<sycl::access::decorated::no>().get(),
                                     sq.get_multi_ptr<sycl::access::decorated::no>().get(), red,
                                     wsum.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        }).wait();
    }

    for (auto _ : state) {
        auto ev = q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> sk(sycl::range<1>(S), cgh);
            sycl::local_accessor<float, 1> sq(sycl::range<1>(S), cgh);
            sycl::local_accessor<float, 2> red(sycl::range<2>(4, S), cgh);
            sycl::local_accessor<float, 1> wsum(sycl::range<1>(32), cgh);

            cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(h_v, 512), sycl::range<2>(4, 128)),
                             [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                                 gdn_fused_step_norm_kernel(
                                     d_state, d_q, d_k, d_v, d_gate, d_beta, d_z, d_gamma, 1e-5f, d_y, h_k, h_v, item,
                                     sk.get_multi_ptr<sycl::access::decorated::no>().get(),
                                     sq.get_multi_ptr<sycl::access::decorated::no>().get(), red,
                                     wsum.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });
        ev.wait();
        const auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double bytes_transferred = (double)(state_elements * 2 + qk_elements * 2 + out_elements * 3) * sizeof(float);
    state.counters["Bandwidth_GBps"] = benchmark::Counter(bytes_transferred, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Heads_HV"]       = benchmark::Counter(h_v);
    state.counters["Heads_HK"]       = benchmark::Counter(h_k);
    state.counters["State_MB"]       = benchmark::Counter(state_elements * sizeof(float) / 1e6);

    sycl::free(d_state, q);
    sycl::free(d_q, q);
    sycl::free(d_k, q);
    sycl::free(d_v, q);
    sycl::free(d_gate, q);
    sycl::free(d_beta, q);
    sycl::free(d_z, q);
    sycl::free(d_gamma, q);
    sycl::free(d_y, q);
}

// ============================================================================
// Benchmark: BM_Xe2_GDN_Fused_SLM_Padded (Bank Conflict Mitigation)
// ============================================================================

template <int PAD>
static void BM_Xe2_GDN_Fused_SLM_Padded(benchmark::State& state) {
    const int h_v = state.range(0);
    const int h_k = state.range(1);

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    const size_t state_elements = (size_t)S * S * h_v;
    const size_t qk_elements = (size_t)S * h_k;
    const size_t out_elements = (size_t)S * h_v;

    auto* d_state = sycl::malloc_device<float>(state_elements, q);
    auto* d_q     = sycl::malloc_device<float>(qk_elements, q);
    auto* d_k     = sycl::malloc_device<float>(qk_elements, q);
    auto* d_v     = sycl::malloc_device<float>(out_elements, q);
    auto* d_gate  = sycl::malloc_device<float>(h_v, q);
    auto* d_beta  = sycl::malloc_device<float>(h_v, q);
    auto* d_z     = sycl::malloc_device<float>(out_elements, q);
    auto* d_gamma = sycl::malloc_device<float>(S, q);
    auto* d_y     = sycl::malloc_device<float>(out_elements, q);

    q.fill(d_state, 0.01f, state_elements);
    q.fill(d_q, 0.05f, qk_elements);
    q.fill(d_k, 0.05f, qk_elements);
    q.fill(d_v, 0.1f, out_elements);
    q.fill(d_gate, -0.5f, h_v);
    q.fill(d_beta, 0.8f, h_v);
    q.fill(d_z, 0.2f, out_elements);
    q.fill(d_gamma, 1.0f, S);
    q.wait();

    // Warmup
    for (int w = 0; w < 5; ++w) {
        q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> sk(sycl::range<1>(S), cgh);
            sycl::local_accessor<float, 1> sq(sycl::range<1>(S), cgh);
            sycl::local_accessor<float, 2> red(sycl::range<2>(4, S + PAD), cgh);
            sycl::local_accessor<float, 1> wsum(sycl::range<1>(32), cgh);

            cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(h_v, 512), sycl::range<2>(4, 128)),
                             [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                                 gdn_fused_step_norm_kernel(
                                     d_state, d_q, d_k, d_v, d_gate, d_beta, d_z, d_gamma, 1e-5f, d_y, h_k, h_v, item,
                                     sk.get_multi_ptr<sycl::access::decorated::no>().get(),
                                     sq.get_multi_ptr<sycl::access::decorated::no>().get(), red,
                                     wsum.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        }).wait();
    }

    for (auto _ : state) {
        auto ev = q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> sk(sycl::range<1>(S), cgh);
            sycl::local_accessor<float, 1> sq(sycl::range<1>(S), cgh);
            sycl::local_accessor<float, 2> red(sycl::range<2>(4, S + PAD), cgh);
            sycl::local_accessor<float, 1> wsum(sycl::range<1>(32), cgh);

            cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(h_v, 512), sycl::range<2>(4, 128)),
                             [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                                 gdn_fused_step_norm_kernel(
                                     d_state, d_q, d_k, d_v, d_gate, d_beta, d_z, d_gamma, 1e-5f, d_y, h_k, h_v, item,
                                     sk.get_multi_ptr<sycl::access::decorated::no>().get(),
                                     sq.get_multi_ptr<sycl::access::decorated::no>().get(), red,
                                     wsum.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });
        ev.wait();
        const auto t0 = ev.template get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.template get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double bytes_transferred = (double)(state_elements * 2 + qk_elements * 2 + out_elements * 3) * sizeof(float);
    state.counters["Bandwidth_GBps"] = benchmark::Counter(bytes_transferred, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Heads_HV"]       = benchmark::Counter(h_v);
    state.counters["Heads_HK"]       = benchmark::Counter(h_k);
    state.counters["Pad_Elements"]   = benchmark::Counter(PAD);
    state.counters["State_MB"]       = benchmark::Counter(state_elements * sizeof(float) / 1e6);

    sycl::free(d_state, q);
    sycl::free(d_q, q);
    sycl::free(d_k, q);
    sycl::free(d_v, q);
    sycl::free(d_gate, q);
    sycl::free(d_beta, q);
    sycl::free(d_z, q);
    sycl::free(d_gamma, q);
    sycl::free(d_y, q);
}

static void BM_Xe2_GDN_Fused_Production_StepNormSiLU(benchmark::State& state) {
    const int h_v = state.range(0);
    const int h_k = state.range(1);

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    const size_t state_elements = (size_t)S * S * h_v;
    const size_t qk_elements = (size_t)S * h_k;
    const size_t out_elements = (size_t)S * h_v;

    auto* d_state  = sycl::malloc_device<float>(state_elements, q);
    auto* d_q      = sycl::malloc_device<float>(qk_elements, q);
    auto* d_k      = sycl::malloc_device<float>(qk_elements, q);
    auto* d_v      = sycl::malloc_device<float>(out_elements, q);
    auto* d_gate   = sycl::malloc_device<float>(h_v, q);
    auto* d_beta   = sycl::malloc_device<float>(h_v, q);
    auto* d_z      = sycl::malloc_device<float>(out_elements, q);
    auto* d_gamma  = sycl::malloc_device<float>(S, q);
    auto* d_y      = sycl::malloc_device<float>(out_elements, q);

    q.fill(d_state, 0.01f, state_elements);
    q.fill(d_q, 0.05f, qk_elements);
    q.fill(d_k, 0.05f, qk_elements);
    q.fill(d_v, 0.1f, out_elements);
    q.fill(d_gate, -0.5f, h_v);
    q.fill(d_beta, 0.8f, h_v);
    q.fill(d_z, 0.2f, out_elements);
    q.fill(d_gamma, 1.0f, S);
    q.wait();

    // Warmup
    for (int w = 0; w < 5; ++w) {
        strata::kernels::fused_gdn_step_norm_silu(d_state, d_q, d_k, d_v, d_gate, d_beta, d_z, d_gamma, 1e-5f, d_y, h_k, h_v, &q);
    }
    q.wait();

    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        strata::kernels::fused_gdn_step_norm_silu(d_state, d_q, d_k, d_v, d_gate, d_beta, d_z, d_gamma, 1e-5f, d_y, h_k, h_v, &q);
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    const double bytes_transferred = (double)(state_elements * 2 + qk_elements * 2 + out_elements * 3) * sizeof(float);
    state.counters["Bandwidth_GBps"] = benchmark::Counter(bytes_transferred, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Heads_HV"]       = benchmark::Counter(h_v);
    state.counters["Heads_HK"]       = benchmark::Counter(h_k);
    state.counters["State_MB"]       = benchmark::Counter(state_elements * sizeof(float) / 1e6);

    sycl::free(d_state, q);
    sycl::free(d_q, q);
    sycl::free(d_k, q);
    sycl::free(d_v, q);
    sycl::free(d_gate, q);
    sycl::free(d_beta, q);
    sycl::free(d_z, q);
    sycl::free(d_gamma, q);
    sycl::free(d_y, q);
}

/// Prefill GDN Recurrence Benchmark: compares STRATA_GDN_XE (variant 6, 16-lane subgroup shuffle)
/// against baseline column-split recurrence (variant 0) across prompt sequence lengths T.
static void BM_Xe2_GDN_Prefill_Prompt_Recurrence_XE(benchmark::State& state) {
    const int64_t T = state.range(0);
    const int variant = state.range(1); // 0 = baseline column-split, 6 = STRATA_GDN_XE subgroup-shuffle

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    constexpr int HV_P = 48;
    constexpr int C_P = 10240;
    constexpr int S_P = 128;

    const size_t state_elements = (size_t)S_P * HV_P * S_P;
    const size_t h_elements = (size_t)T * C_P;
    const size_t gb_elements = (size_t)T * HV_P;
    const size_t z_elements = (size_t)T * HV_P * S_P;
    const size_t y_elements = (size_t)T * HV_P * S_P;

    auto* d_state = sycl::malloc_device<float>(state_elements, q);
    auto* d_h     = sycl::malloc_device<float>(h_elements, q);
    auto* d_gate  = sycl::malloc_device<float>(gb_elements, q);
    auto* d_beta  = sycl::malloc_device<float>(gb_elements, q);
    auto* d_z     = sycl::malloc_device<float>(z_elements, q);
    auto* d_gamma = sycl::malloc_device<float>(S_P, q);
    auto* d_y     = sycl::malloc_device<float>(y_elements, q);
    auto* d_y16   = sycl::malloc_device<uint16_t>(y_elements, q);

    if (!d_state || !d_h || !d_gate || !d_beta || !d_z || !d_gamma || !d_y || !d_y16) {
        state.SkipWithError("USM allocation failure in BM_Xe2_GDN_Prefill_Prompt_Recurrence_XE");
        if (d_state) sycl::free(d_state, q);
        if (d_h)     sycl::free(d_h, q);
        if (d_gate)  sycl::free(d_gate, q);
        if (d_beta)  sycl::free(d_beta, q);
        if (d_z)     sycl::free(d_z, q);
        if (d_gamma) sycl::free(d_gamma, q);
        if (d_y)     sycl::free(d_y, q);
        if (d_y16)   sycl::free(d_y16, q);
        return;
    }

    q.fill(d_state, 0.01f, state_elements);
    q.fill(d_h, 0.05f, h_elements);
    q.fill(d_gate, -0.5f, gb_elements);
    q.fill(d_beta, 0.8f, gb_elements);
    q.fill(d_z, 0.2f, z_elements);
    q.fill(d_gamma, 1.0f, S_P);
    q.wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        strata::prefill::gdn_recurrence_variant(variant, d_state, d_h, d_gate, d_beta, d_z, d_gamma, 1e-6f, d_y, d_y16, T, &q);
    }
    q.wait();

    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        strata::prefill::gdn_recurrence_variant(variant, d_state, d_h, d_gate, d_beta, d_z, d_gamma, 1e-6f, d_y, d_y16, T, &q);
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    const double bytes_transferred = (double)(h_elements + gb_elements * 2 + z_elements + y_elements) * sizeof(float)
                                   + (double)y_elements * sizeof(uint16_t)
                                   + (double)state_elements * sizeof(float) * 2;
    state.counters["tokens_per_sec"] = benchmark::Counter(
        (double)T, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Bandwidth_GBps"] = benchmark::Counter(
        bytes_transferred, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Variant"] = benchmark::Counter(variant);
    state.counters["SeqLen_T"] = benchmark::Counter(T);

    sycl::free(d_state, q);
    sycl::free(d_h, q);
    sycl::free(d_gate, q);
    sycl::free(d_beta, q);
    sycl::free(d_z, q);
    sycl::free(d_gamma, q);
    sycl::free(d_y, q);
    sycl::free(d_y16, q);
}

// Register Benchmarks
BENCHMARK(BM_Xe2_GDN_Unfused)->Args({DEFAULT_HV, DEFAULT_HK})->Args({DEFAULT_HV, 4})->UseManualTime();
BENCHMARK(BM_Xe2_GDN_Fused_SLM)->Args({DEFAULT_HV, DEFAULT_HK})->Args({DEFAULT_HV, 4})->UseManualTime();
BENCHMARK(BM_Xe2_GDN_Fused_Production_StepNormSiLU)->Args({DEFAULT_HV, DEFAULT_HK})->Args({DEFAULT_HV, 4})->UseManualTime();

// Register Padded SLM sweeps (4, 8, 16 floats bank offset) across head configurations
BENCHMARK(BM_Xe2_GDN_Fused_SLM_Padded<4>)->Args({DEFAULT_HV, DEFAULT_HK})->Args({DEFAULT_HV, 4})->UseManualTime();
BENCHMARK(BM_Xe2_GDN_Fused_SLM_Padded<8>)->Args({DEFAULT_HV, DEFAULT_HK})->Args({DEFAULT_HV, 4})->UseManualTime();
BENCHMARK(BM_Xe2_GDN_Fused_SLM_Padded<16>)->Args({DEFAULT_HV, DEFAULT_HK})->Args({DEFAULT_HV, 4})->UseManualTime();

// Register Prefill GDN Recurrence (Baseline Variant 0 vs STRATA_GDN_XE Variant 6)
BENCHMARK(BM_Xe2_GDN_Prefill_Prompt_Recurrence_XE)->UseManualTime()->Args({64, 0})->Args({64, 6});
BENCHMARK(BM_Xe2_GDN_Prefill_Prompt_Recurrence_XE)->UseManualTime()->Args({128, 0})->Args({128, 6});
BENCHMARK(BM_Xe2_GDN_Prefill_Prompt_Recurrence_XE)->UseManualTime()->Args({256, 0})->Args({256, 6});
BENCHMARK(BM_Xe2_GDN_Prefill_Prompt_Recurrence_XE)->UseManualTime()->Args({512, 0})->Args({512, 6});
BENCHMARK(BM_Xe2_GDN_Prefill_Prompt_Recurrence_XE)->UseManualTime()->Args({1024, 0})->Args({1024, 6});

BENCHMARK_MAIN();

