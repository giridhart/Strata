// perf/sycl/perf_moe_accumulate.cpp - Intel Xe2 MoE Expert Weighted Reduction Benchmark
//
// Benchmarks MoE weighted accumulation across 10 routed experts (D = 2560 embedding dimensions):
// output[b, col] = sum_{e=0}^{9} parts[b, e, col] * weights[b, e]
//
// Compares:
// 1. BM_Xe2_MoE_Atomic: Global FP32 atomic additions (atomic_ref in device memory)
// 2. BM_Xe2_MoE_DirectCombine: Column-strided reduction (native_moe combine kernel)
// 3. BM_Xe2_MoE_SubgroupReduce: SIMD16 hierarchical subgroup cooperative reduction
//
// Evaluated across batch sizes B = 1, 4, 8, 16, 32.

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr int N_EXPERTS = 10;
constexpr int N_EMBD = 2560;

// ============================================================================
// MoE Reduction Kernels
// ============================================================================

/// 1. Atomic Add Reduction
/// Each work-item processes one expert for one column and atomically accumulates to output.
void moe_atomic_kernel(sycl::nd_item<1> item, const float* __restrict__ parts, const float* __restrict__ weights,
                       float* __restrict__ output, int B, int n_embd, int k) {
    const int idx = item.get_global_id(0);
    const int total_items = B * k * n_embd;
    if (idx >= total_items) return;

    const int b = idx / (k * n_embd);
    const int rem = idx % (k * n_embd);
    const int expert = rem / n_embd;
    const int col = rem % n_embd;

    const float val = parts[idx] * weights[b * k + expert];

    auto out_ref = sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(output[b * n_embd + col]);
    out_ref.fetch_add(val);
}

/// 2. Direct Column-Strided Reduction (from native_moe.cpp)
/// One thread per output column, iterating across all K experts sequentially.
void moe_combine_kernel(sycl::nd_item<2> item, const float* __restrict__ parts, const float* __restrict__ weights,
                        float* __restrict__ output, int n_embd, int k) {
    const int64_t tk = item.get_group(0);  // Token / Batch index
    parts += tk * k * n_embd;
    weights += tk * k;
    output += tk * n_embd;

    const int64_t col = int64_t(item.get_group(1)) * item.get_local_range(1) + item.get_local_id(1);
    if (col >= n_embd) return;

    float sum = parts[col] * weights[0];
#pragma unroll
    for (int expert = 1; expert < k; ++expert) {
        sum += parts[int64_t(expert) * n_embd + col] * weights[expert];
    }
    output[col] = sum;
}

/// 3. SIMD16 Subgroup Cooperative Reduction
/// Subgroups stride across columns with unrolled vectorized memory operations.
void moe_subgroup_kernel(sycl::nd_item<1> item, const float* __restrict__ parts, const float* __restrict__ weights,
                         float* __restrict__ output, int B, int n_embd, int k) {
    const int total_cols = B * n_embd;
    const int col_idx = item.get_global_id(0);
    if (col_idx >= total_cols) return;

    const int b = col_idx / n_embd;
    const int col = col_idx % n_embd;

    const float* p_tok = parts + static_cast<size_t>(b) * k * n_embd + col;
    const float* w_tok = weights + b * k;

    float sum = 0.0f;
#pragma unroll 10
    for (int e = 0; e < k; ++e) {
        sum += p_tok[static_cast<size_t>(e) * n_embd] * w_tok[e];
    }
    output[col_idx] = sum;
}

}  // namespace

// ============================================================================
// Benchmarks
// ============================================================================

static void BM_Xe2_MoE_Atomic(benchmark::State& state) {
    const int B = static_cast<int>(state.range(0));
    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    const size_t parts_size = static_cast<size_t>(B) * N_EXPERTS * N_EMBD;
    const size_t weights_size = static_cast<size_t>(B) * N_EXPERTS;
    const size_t out_size = static_cast<size_t>(B) * N_EMBD;

    float* d_parts = sycl::malloc_device<float>(parts_size, q);
    float* d_weights = sycl::malloc_device<float>(weights_size, q);
    float* d_out = sycl::malloc_device<float>(out_size, q);

    if (!d_parts || !d_weights || !d_out) {
        state.SkipWithError("Failed to allocate USM memory for MoE benchmark");
        if (d_parts) sycl::free(d_parts, q);
        if (d_weights) sycl::free(d_weights, q);
        if (d_out) sycl::free(d_out, q);
        return;
    }

    q.fill(d_parts, 0.1f, parts_size);
    q.fill(d_weights, 0.1f, weights_size);
    q.fill(d_out, 0.0f, out_size);
    q.wait_and_throw();

    constexpr int WG = 256;
    const size_t total_items = parts_size;
    const size_t n_wgs = (total_items + WG - 1) / WG;

    auto launch = [&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(n_wgs * WG, WG),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                             moe_atomic_kernel(item, d_parts, d_weights, d_out, B, N_EMBD, N_EXPERTS);
                         });
    };

    // Warmup
    for (int i = 0; i < 5; ++i) {
        q.fill(d_out, 0.0f, out_size);
        q.submit(launch).wait();
    }

    for (auto _ : state) {
        q.fill(d_out, 0.0f, out_size);
        auto ev = q.submit(launch);
        ev.wait();
        auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double bytes = static_cast<double>(parts_size + weights_size + out_size) * sizeof(float);
    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["TFLOPS"] =
        benchmark::Counter(2.0 * static_cast<double>(parts_size), benchmark::Counter::kIsIterationInvariantRate,
                           benchmark::Counter::kIs1000);
    state.counters["tokens_per_sec"] = benchmark::Counter(
        static_cast<double>(B), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_parts, q);
    sycl::free(d_weights, q);
    sycl::free(d_out, q);
}

static void BM_Xe2_MoE_DirectCombine(benchmark::State& state) {
    const int B = static_cast<int>(state.range(0));
    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    const size_t parts_size = static_cast<size_t>(B) * N_EXPERTS * N_EMBD;
    const size_t weights_size = static_cast<size_t>(B) * N_EXPERTS;
    const size_t out_size = static_cast<size_t>(B) * N_EMBD;

    float* d_parts = sycl::malloc_device<float>(parts_size, q);
    float* d_weights = sycl::malloc_device<float>(weights_size, q);
    float* d_out = sycl::malloc_device<float>(out_size, q);

    if (!d_parts || !d_weights || !d_out) {
        state.SkipWithError("Failed to allocate USM memory for MoE benchmark");
        if (d_parts) sycl::free(d_parts, q);
        if (d_weights) sycl::free(d_weights, q);
        if (d_out) sycl::free(d_out, q);
        return;
    }

    q.fill(d_parts, 0.1f, parts_size);
    q.fill(d_weights, 0.1f, weights_size);
    q.fill(d_out, 0.0f, out_size);
    q.wait_and_throw();

    constexpr int WG = 256;
    const size_t n_blocks = (N_EMBD + WG - 1) / WG;

    auto launch = [&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(B, n_blocks * WG), sycl::range<2>(1, WG)),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                             moe_combine_kernel(item, d_parts, d_weights, d_out, N_EMBD, N_EXPERTS);
                         });
    };

    // Warmup
    for (int i = 0; i < 5; ++i) {
        q.submit(launch).wait();
    }

    for (auto _ : state) {
        auto ev = q.submit(launch);
        ev.wait();
        auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double bytes = static_cast<double>(parts_size + weights_size + out_size) * sizeof(float);
    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["TFLOPS"] =
        benchmark::Counter(2.0 * static_cast<double>(parts_size), benchmark::Counter::kIsIterationInvariantRate,
                           benchmark::Counter::kIs1000);
    state.counters["tokens_per_sec"] = benchmark::Counter(
        static_cast<double>(B), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_parts, q);
    sycl::free(d_weights, q);
    sycl::free(d_out, q);
}

static void BM_Xe2_MoE_SubgroupReduce(benchmark::State& state) {
    const int B = static_cast<int>(state.range(0));
    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    const size_t parts_size = static_cast<size_t>(B) * N_EXPERTS * N_EMBD;
    const size_t weights_size = static_cast<size_t>(B) * N_EXPERTS;
    const size_t out_size = static_cast<size_t>(B) * N_EMBD;

    float* d_parts = sycl::malloc_device<float>(parts_size, q);
    float* d_weights = sycl::malloc_device<float>(weights_size, q);
    float* d_out = sycl::malloc_device<float>(out_size, q);

    if (!d_parts || !d_weights || !d_out) {
        state.SkipWithError("Failed to allocate USM memory for MoE benchmark");
        if (d_parts) sycl::free(d_parts, q);
        if (d_weights) sycl::free(d_weights, q);
        if (d_out) sycl::free(d_out, q);
        return;
    }

    q.fill(d_parts, 0.1f, parts_size);
    q.fill(d_weights, 0.1f, weights_size);
    q.fill(d_out, 0.0f, out_size);
    q.wait_and_throw();

    constexpr int WG = 256;
    const size_t total_cols = out_size;
    const size_t n_wgs = (total_cols + WG - 1) / WG;

    auto launch = [&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(n_wgs * WG, WG),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                             moe_subgroup_kernel(item, d_parts, d_weights, d_out, B, N_EMBD, N_EXPERTS);
                         });
    };

    // Warmup
    for (int i = 0; i < 5; ++i) {
        q.submit(launch).wait();
    }

    for (auto _ : state) {
        auto ev = q.submit(launch);
        ev.wait();
        auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double bytes = static_cast<double>(parts_size + weights_size + out_size) * sizeof(float);
    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["TFLOPS"] =
        benchmark::Counter(2.0 * static_cast<double>(parts_size), benchmark::Counter::kIsIterationInvariantRate,
                           benchmark::Counter::kIs1000);
    state.counters["tokens_per_sec"] = benchmark::Counter(
        static_cast<double>(B), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_parts, q);
    sycl::free(d_weights, q);
    sycl::free(d_out, q);
}

// ============================================================================
// Grouped MoE Expert GEMV Projections (16-Lane Subgroup vs Legacy 32-Lane)
// ============================================================================

template <int GU_TYPE = 2, int D_TYPE = 2, bool SUB16 = true>
static void BM_Xe2_MoE_GroupedExpert(benchmark::State& state) {
    const int NT = static_cast<int>(state.range(0));
    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    const int gu_type = GU_TYPE, d_type = D_TYPE;
    strata::kernels::NativeExpertLayout L = strata::kernels::native_expert_layout(gu_type, d_type, N_EMBD, 640);
    const size_t scratch_bytes = strata::kernels::native_expert_scratch_bytes(NT, 640);

    uint8_t* d_blob = sycl::malloc_device<uint8_t>(L.bytes, q);
    uint8_t* d_scratch = sycl::malloc_device<uint8_t>(scratch_bytes, q);
    float* d_out = sycl::malloc_device<float>(static_cast<size_t>(NT) * N_EMBD, q);
    const size_t q8_1_bytes = static_cast<size_t>(NT) * (N_EMBD / 32) * 36; // sizeof(block_q8_1) == 36
    void* d_xq = sycl::malloc_device<uint8_t>(q8_1_bytes, q);

    unsigned long long* d_grp_ptr = sycl::malloc_device<unsigned long long>(1, q);
    int32_t* d_grp_start = sycl::malloc_device<int32_t>(2, q);
    int32_t* d_ngroups = sycl::malloc_device<int32_t>(1, q);
    int32_t* d_ent_dst = sycl::malloc_device<int32_t>(NT, q);
    int32_t* d_ent_tok = sycl::malloc_device<int32_t>(NT, q);

    if (!d_blob || !d_scratch || !d_out || !d_xq || !d_grp_ptr || !d_grp_start || !d_ngroups || !d_ent_dst || !d_ent_tok) {
        state.SkipWithError("Failed to allocate USM memory for MoE expert benchmark");
        if (d_blob) sycl::free(d_blob, q);
        if (d_scratch) sycl::free(d_scratch, q);
        if (d_out) sycl::free(d_out, q);
        if (d_xq) sycl::free(d_xq, q);
        if (d_grp_ptr) sycl::free(d_grp_ptr, q);
        if (d_grp_start) sycl::free(d_grp_start, q);
        if (d_ngroups) sycl::free(d_ngroups, q);
        if (d_ent_dst) sycl::free(d_ent_dst, q);
        if (d_ent_tok) sycl::free(d_ent_tok, q);
        return;
    }

    q.memset(d_blob, 0x11, L.bytes);
    q.memset(d_scratch, 0, scratch_bytes);
    q.fill(d_out, 0.0f, static_cast<size_t>(NT) * N_EMBD);
    q.memset(d_xq, 0x22, q8_1_bytes);

    unsigned long long h_ptr = reinterpret_cast<unsigned long long>(d_blob);
    int32_t h_start[2] = {0, NT};
    int32_t h_ngroups = 1;
    std::vector<int32_t> h_dst(NT), h_tok(NT);
    for (int i = 0; i < NT; ++i) {
        h_dst[i] = i;
        h_tok[i] = i;
    }

    q.copy(&h_ptr, d_grp_ptr, 1);
    q.copy(h_start, d_grp_start, 2);
    q.copy(&h_ngroups, d_ngroups, 1);
    q.copy(h_dst.data(), d_ent_dst, NT);
    q.copy(h_tok.data(), d_ent_tok, NT);
    q.wait_and_throw();

    const char* prev_env = std::getenv("STRATA_NO_SUB16_GU");
    std::string prev_val = prev_env ? prev_env : "";
    if constexpr (SUB16) {
        setenv("STRATA_NO_SUB16_GU", "0", 1);
    } else {
        setenv("STRATA_NO_SUB16_GU", "1", 1);
    }

    // Warmup
    for (int i = 0; i < 5; ++i) {
        strata::kernels::native_expert_grouped(L, d_grp_ptr, d_grp_start, d_ngroups, d_ent_dst, d_ent_tok,
                                              1, NT, d_xq, d_scratch, d_out, &q);
        q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();
        strata::kernels::native_expert_grouped(L, d_grp_ptr, d_grp_start, d_ngroups, d_ent_dst, d_ent_tok,
                                              1, NT, d_xq, d_scratch, d_out, &q);
        q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    if (!prev_val.empty()) {
        setenv("STRATA_NO_SUB16_GU", prev_val.c_str(), 1);
    } else {
        unsetenv("STRATA_NO_SUB16_GU");
    }

    const double flops = static_cast<double>(NT) * (2.0 * 2560.0 * 640.0 * 2.0 + 2.0 * 640.0 * 2560.0);
    const double bytes = static_cast<double>(L.bytes) + scratch_bytes + static_cast<double>(NT) * N_EMBD * sizeof(float);
    state.counters["TFLOPS"] = benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Bandwidth_GBps"] = benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_blob, q);
    sycl::free(d_scratch, q);
    sycl::free(d_out, q);
    sycl::free(d_xq, q);
    sycl::free(d_grp_ptr, q);
    sycl::free(d_grp_start, q);
    sycl::free(d_ngroups, q);
    sycl::free(d_ent_dst, q);
    sycl::free(d_ent_tok, q);
}

// ============================================================================
// Dual-Column Gate + Up GEMV with In-Register SwiGLU
// ============================================================================

constexpr int INTER_DIM = 6144;

void moe_gemv_single_kernel(sycl::nd_item<1> item, const float* __restrict__ x, const float* __restrict__ w,
                            float* __restrict__ out, int B, int K, int N) {
    const int global_col = item.get_group(0);
    const int b = global_col / N;
    const int c = global_col % N;
    const int lane = item.get_local_id(0);
    auto sg = item.get_sub_group();

    if (b >= B) return;

    const float* x_tok = x + b * K;
    float dot = 0.0f;
    for (int k = lane; k < K; k += 16) {
        dot += x_tok[k] * w[k * N + c];
    }
    dot = sycl::reduce_over_group(sg, dot, sycl::plus<float>());
    if (lane == 0) out[b * N + c] = dot;
}

void moe_swiglu_elementwise_kernel(sycl::nd_item<1> item, const float* __restrict__ g, const float* __restrict__ u,
                                   float* __restrict__ y, int total_elements) {
    const int idx = item.get_global_id(0);
    if (idx >= total_elements) return;
    const float g_val = g[idx];
    const float u_val = u[idx];
    const float silu_g = g_val / (1.0f + sycl::native::exp(-g_val));
    y[idx] = silu_g * u_val;
}

void moe_dual_column_swiglu_kernel(sycl::nd_item<1> item, const float* __restrict__ x,
                                   const float* __restrict__ wg, const float* __restrict__ wu,
                                   float* __restrict__ y, int B, int K, int N) {
    const int global_col = item.get_group(0);
    const int b = global_col / N;
    const int c = global_col % N;
    const int lane = item.get_local_id(0);
    auto sg = item.get_sub_group();

    if (b >= B) return;

    const float* x_tok = x + b * K;
    float dot_g = 0.0f;
    float dot_u = 0.0f;
    for (int k = lane; k < K; k += 16) {
        const float x_val = x_tok[k];
        dot_g += x_val * wg[k * N + c];
        dot_u += x_val * wu[k * N + c];
    }
    dot_g = sycl::reduce_over_group(sg, dot_g, sycl::plus<float>());
    dot_u = sycl::reduce_over_group(sg, dot_u, sycl::plus<float>());

    if (lane == 0) {
        const float silu_g = dot_g / (1.0f + sycl::native::exp(-dot_g));
        y[b * N + c] = silu_g * dot_u;
    }
}

static void BM_Xe2_MoE_Separate_Gate_Up_SwiGLU(benchmark::State& state) {
    const int B = state.range(0);
    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    const size_t x_elements = (size_t)B * N_EMBD;
    const size_t w_elements = (size_t)N_EMBD * INTER_DIM;
    const size_t act_elements = (size_t)B * INTER_DIM;

    auto* d_x  = sycl::malloc_device<float>(x_elements, q);
    auto* d_wg = sycl::malloc_device<float>(w_elements, q);
    auto* d_wu = sycl::malloc_device<float>(w_elements, q);
    auto* d_g  = sycl::malloc_device<float>(act_elements, q);
    auto* d_u  = sycl::malloc_device<float>(act_elements, q);
    auto* d_y  = sycl::malloc_device<float>(act_elements, q);

    q.fill(d_x, 0.05f, x_elements);
    q.fill(d_wg, 0.01f, w_elements);
    q.fill(d_wu, 0.01f, w_elements);
    q.wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        q.parallel_for(sycl::nd_range<1>(B * INTER_DIM * 16, 16),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                           moe_gemv_single_kernel(item, d_x, d_wg, d_g, B, N_EMBD, INTER_DIM);
                       });
        q.parallel_for(sycl::nd_range<1>(B * INTER_DIM * 16, 16),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                           moe_gemv_single_kernel(item, d_x, d_wu, d_u, B, N_EMBD, INTER_DIM);
                       });
        q.parallel_for(sycl::nd_range<1>(((B * INTER_DIM + 127) / 128) * 128, 128),
                       [=](sycl::nd_item<1> item) {
                           moe_swiglu_elementwise_kernel(item, d_g, d_u, d_y, B * INTER_DIM);
                       }).wait();
    }

    for (auto _ : state) {
        auto ev1 = q.parallel_for(sycl::nd_range<1>(B * INTER_DIM * 16, 16),
                                  [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                      moe_gemv_single_kernel(item, d_x, d_wg, d_g, B, N_EMBD, INTER_DIM);
                                  });
        auto ev2 = q.parallel_for(sycl::nd_range<1>(B * INTER_DIM * 16, 16),
                                  [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                      moe_gemv_single_kernel(item, d_x, d_wu, d_u, B, N_EMBD, INTER_DIM);
                                  });
        auto ev3 = q.submit([&](sycl::handler& cgh) {
            cgh.depends_on({ev1, ev2});
            cgh.parallel_for(sycl::nd_range<1>(((B * INTER_DIM + 127) / 128) * 128, 128),
                             [=](sycl::nd_item<1> item) {
                                 moe_swiglu_elementwise_kernel(item, d_g, d_u, d_y, B * INTER_DIM);
                             });
        });
        ev3.wait();
        const auto t0 = ev1.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev3.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double flops = 4.0 * (double)B * N_EMBD * INTER_DIM + 3.0 * (double)B * INTER_DIM;
    const double bytes = (w_elements * 2 + x_elements + act_elements * 5) * sizeof(float);
    state.counters["GFLOPS"] = benchmark::Counter(flops * 1e-9, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Bandwidth_GBps"] = benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Batch_B"] = benchmark::Counter((double)B);

    sycl::free(d_x, q);
    sycl::free(d_wg, q);
    sycl::free(d_wu, q);
    sycl::free(d_g, q);
    sycl::free(d_u, q);
    sycl::free(d_y, q);
}

static void BM_Xe2_MoE_DualColumn_SwiGLU(benchmark::State& state) {
    const int B = state.range(0);
    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    const size_t x_elements = (size_t)B * N_EMBD;
    const size_t w_elements = (size_t)N_EMBD * INTER_DIM;
    const size_t act_elements = (size_t)B * INTER_DIM;

    auto* d_x  = sycl::malloc_device<float>(x_elements, q);
    auto* d_wg = sycl::malloc_device<float>(w_elements, q);
    auto* d_wu = sycl::malloc_device<float>(w_elements, q);
    auto* d_y  = sycl::malloc_device<float>(act_elements, q);

    q.fill(d_x, 0.05f, x_elements);
    q.fill(d_wg, 0.01f, w_elements);
    q.fill(d_wu, 0.01f, w_elements);
    q.wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        q.parallel_for(sycl::nd_range<1>(B * INTER_DIM * 16, 16),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                           moe_dual_column_swiglu_kernel(item, d_x, d_wg, d_wu, d_y, B, N_EMBD, INTER_DIM);
                       }).wait();
    }

    for (auto _ : state) {
        auto ev = q.parallel_for(sycl::nd_range<1>(B * INTER_DIM * 16, 16),
                                 [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                     moe_dual_column_swiglu_kernel(item, d_x, d_wg, d_wu, d_y, B, N_EMBD, INTER_DIM);
                                 });
        ev.wait();
        const auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double flops = 4.0 * (double)B * N_EMBD * INTER_DIM + 3.0 * (double)B * INTER_DIM;
    const double bytes = (w_elements * 2 + x_elements + act_elements) * sizeof(float);
    state.counters["GFLOPS"] = benchmark::Counter(flops * 1e-9, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Bandwidth_GBps"] = benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Batch_B"] = benchmark::Counter((double)B);

    sycl::free(d_x, q);
    sycl::free(d_wg, q);
    sycl::free(d_wu, q);
    sycl::free(d_y, q);
}

// ============================================================================
// Registration: MoE Expert Reduction Across Batch Sizes
// ============================================================================

BENCHMARK(BM_Xe2_MoE_Atomic)->UseManualTime()->Args({1});
BENCHMARK(BM_Xe2_MoE_Atomic)->UseManualTime()->Args({4});
BENCHMARK(BM_Xe2_MoE_Atomic)->UseManualTime()->Args({8});
BENCHMARK(BM_Xe2_MoE_Atomic)->UseManualTime()->Args({16});
BENCHMARK(BM_Xe2_MoE_Atomic)->UseManualTime()->Args({32});

BENCHMARK(BM_Xe2_MoE_DirectCombine)->UseManualTime()->Args({1});
BENCHMARK(BM_Xe2_MoE_DirectCombine)->UseManualTime()->Args({4});
BENCHMARK(BM_Xe2_MoE_DirectCombine)->UseManualTime()->Args({8});
BENCHMARK(BM_Xe2_MoE_DirectCombine)->UseManualTime()->Args({16});
BENCHMARK(BM_Xe2_MoE_DirectCombine)->UseManualTime()->Args({32});

BENCHMARK(BM_Xe2_MoE_SubgroupReduce)->UseManualTime()->Args({1});
BENCHMARK(BM_Xe2_MoE_SubgroupReduce)->UseManualTime()->Args({4});
BENCHMARK(BM_Xe2_MoE_SubgroupReduce)->UseManualTime()->Args({8});
BENCHMARK(BM_Xe2_MoE_SubgroupReduce)->UseManualTime()->Args({16});
BENCHMARK(BM_Xe2_MoE_SubgroupReduce)->UseManualTime()->Args({32});

BENCHMARK(BM_Xe2_MoE_GroupedExpert<2, 2, true>)->Name("BM_Xe2_MoE_GroupedExpert_Q4_0_Sub16")->UseManualTime()->Args({1})->Args({4})->Args({8});
BENCHMARK(BM_Xe2_MoE_GroupedExpert<2, 2, false>)->Name("BM_Xe2_MoE_GroupedExpert_Q4_0_Legacy32")->UseManualTime()->Args({1})->Args({4})->Args({8});
BENCHMARK(BM_Xe2_MoE_GroupedExpert<40, 40, true>)->Name("BM_Xe2_MoE_GroupedExpert_NVFP4")->UseManualTime()->Args({1})->Args({4})->Args({8});
BENCHMARK(BM_Xe2_MoE_GroupedExpert<17, 20, true>)->Name("BM_Xe2_MoE_GroupedExpert_IQ2_XS_IQ4NL")->UseManualTime()->Args({1})->Args({4})->Args({8});

BENCHMARK(BM_Xe2_MoE_Separate_Gate_Up_SwiGLU)->UseManualTime()->Args({1})->Args({4})->Args({8})->Args({16})->Args({32});
BENCHMARK(BM_Xe2_MoE_DualColumn_SwiGLU)->UseManualTime()->Args({1})->Args({4})->Args({8})->Args({16})->Args({32});


// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char** argv) {
    try {
        sycl::queue q{sycl::default_selector_v};
        auto dev = q.get_device();
        std::printf("=================================================================\n");
        std::printf(" Intel Xe2 Performance Tuning Suite - MoE Accumulate Benchmark\n");
        std::printf(" Device:       %s\n", dev.get_info<sycl::info::device::name>().c_str());
        std::printf(" Driver:       %s\n", dev.get_info<sycl::info::device::driver_version>().c_str());
        std::printf(" Subgroup:     %d (Native Xe2 SIMD16)\n", strata::sycl_arch::SUB_GROUP_SIZE);
        std::printf("=================================================================\n\n");
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "Warning: Unable to probe default SYCL device: %s\n", e.what());
    }

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
