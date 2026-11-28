// perf/sycl/perf_residual_pipeline.cpp - Intel Xe2 Residual Stream Inter-Kernel Handover & Fusion Benchmark
//
// Benchmarks the high-traffic residual spine across attention and FFN halves:
// R is (hc, n_embd) floats = 4 * 2560 * 4 bytes = 40.96 KB
//
// Every layer executes:
//   gr_read(attn) -> mixer (GEMV) -> gr_write(attn) -> gr_read(ffn) -> MoE -> gr_write(ffn)
//
// Compares:
// 1. BM_Xe2_Residual_Unfused: Separate global memory roundtrips for R between halves
// 2. BM_Xe2_Residual_Fused: Folding write(attn) directly into read(ffn) in registers/SLM
// 3. BM_Xe2_Residual_Chained_Layers: Simulating 1, 2, 4, 8 back-to-back layers to test L2/L3 cache residency

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>

#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/fused_gr.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int HC = 4;
constexpr int N_EMBD = 2560;
constexpr int SUB_SIZE = 16;

// ============================================================================
// Computational Kernels for Residual Stream Pipeline
// ============================================================================

/// 1. Standalone gr_read / RMSNorm Kernel: Computes RMSNorm on R and combines streams into mixed
void gr_norm_kernel(sycl::nd_item<1> item, const float* __restrict__ R, const float* __restrict__ w_norm,
                    float* __restrict__ mixed, float* __restrict__ xn, float eps,
                    float* __restrict__ local_sums) {
    const int c = item.get_group(0); // Stream index (0..HC-1)
    if (c >= HC) return;

    const int tid = item.get_local_id(0);
    const int bsize = item.get_local_range(0);
    const float* Rc = R + (size_t)c * N_EMBD;
    float* xnc = xn + (size_t)c * N_EMBD;

    float ss = 0.0f;
    for (int d = tid; d < N_EMBD; d += bsize) {
        const float v = Rc[d];
        ss += v * v;
    }

    auto sg = item.get_sub_group();
    ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());

    const int warp = tid / SUB_SIZE;
    const int lane = tid % SUB_SIZE;
    if (lane == 0) {
        local_sums[warp] = ss;
    }
    sycl::group_barrier(item.get_group());

    if (warp == 0) {
        const int num_warps = bsize / SUB_SIZE;
        float block_ss = (lane < num_warps) ? local_sums[lane] : 0.0f;
        block_ss = sycl::reduce_over_group(sg, block_ss, sycl::plus<float>());
        if (lane == 0) {
            local_sums[0] = block_ss;
        }
    }
    sycl::group_barrier(item.get_group());

    const float ms = local_sums[0] / (float)N_EMBD;
    const float rs = sycl::rsqrt(ms + eps);

    for (int d = tid; d < N_EMBD; d += bsize) {
        const float val = Rc[d] * rs * w_norm[(size_t)c * N_EMBD + d];
        xnc[d] = val;
        // Stream combine into mixed
        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(mixed[d]);
        ref.fetch_add(val * (1.0f / (float)HC));
    }
}

/// 2. Synthetic GEMV Projection (simulating mixer projection)
void residual_gemv_kernel(sycl::nd_item<1> item, const float* __restrict__ in,
                          const sycl::ext::oneapi::bfloat16* __restrict__ weight,
                          float* __restrict__ out) {
    const int row = item.get_group(0);
    if (row >= N_EMBD) return;

    const int tid = item.get_local_id(0);
    const int bsize = item.get_local_range(0);
    const auto* w_row = weight + (size_t)row * N_EMBD;

    float acc = 0.0f;
    for (int col = tid; col < N_EMBD; col += bsize) {
        acc += in[col] * (float)w_row[col];
    }

    auto sg = item.get_sub_group();
    acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());

    // Direct reduction over subgroup
    if (tid == 0) {
        out[row] = acc;
    }
}

/// 2b. Synthetic GEMV Projection with L3 Streaming Load Hints (Bypassing L3 cache for large weights)
void residual_gemv_streaming_kernel(sycl::nd_item<1> item, const float* __restrict__ in,
                                    const sycl::ext::oneapi::bfloat16* __restrict__ weight,
                                    float* __restrict__ out) {
    const int row = item.get_group(0);
    if (row >= N_EMBD) return;

    const int tid = item.get_local_id(0);
    const int bsize = item.get_local_range(0);
    const auto* w_row = weight + (size_t)row * N_EMBD;

    auto stream_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::read_hint<
            sycl::ext::intel::experimental::cache_control<
                sycl::ext::intel::experimental::cache_mode::streaming,
                sycl::ext::oneapi::experimental::cache_level::L3>>};
    sycl::ext::oneapi::experimental::annotated_ptr<
        const sycl::ext::oneapi::bfloat16, decltype(stream_props)> w_stream(w_row, stream_props);

    float acc = 0.0f;
    for (int col = tid; col < N_EMBD; col += bsize) {
        acc += in[col] * float(sycl::ext::oneapi::bfloat16(w_stream[col]));
    }

    auto sg = item.get_sub_group();
    acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());

    if (tid == 0) {
        out[row] = acc;
    }
}

/// 3. Standalone gr_write Kernel: updates R with output vector across streams
void gr_write_kernel(sycl::nd_item<1> item, float* __restrict__ R, const float* __restrict__ out,
                     float alpha) {
    const int idx = item.get_global_id(0);
    const int total = HC * N_EMBD;
    if (idx < total) {
        const int d = idx % N_EMBD;
        R[idx] += alpha * out[d];
    }
}

/// 4. Fused gr_write + gr_read Kernel:
/// Folds the previous half's residual write into the next half's read directly in registers/SLM,
/// updating R and immediately computing the RMSNorm sum of squares without an extra memory roundtrip.
void fused_gr_write_read_kernel(sycl::nd_item<1> item, float* __restrict__ R,
                                const float* __restrict__ prev_out, const float* __restrict__ w_norm,
                                float* __restrict__ mixed, float* __restrict__ xn,
                                float alpha, float eps, float* __restrict__ local_sums) {
    const int c = item.get_group(0);
    if (c >= HC) return;

    const int tid = item.get_local_id(0);
    const int bsize = item.get_local_range(0);
    float* Rc = R + (size_t)c * N_EMBD;
    float* xnc = xn + (size_t)c * N_EMBD;

    float ss = 0.0f;
    for (int d = tid; d < N_EMBD; d += bsize) {
        // Read, apply write in-register, store updated R
        float v = Rc[d] + alpha * prev_out[d];
        Rc[d] = v;
        ss += v * v;
    }

    auto sg = item.get_sub_group();
    ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());

    const int warp = tid / SUB_SIZE;
    const int lane = tid % SUB_SIZE;
    if (lane == 0) {
        local_sums[warp] = ss;
    }
    sycl::group_barrier(item.get_group());

    if (warp == 0) {
        const int num_warps = bsize / SUB_SIZE;
        float block_ss = (lane < num_warps) ? local_sums[lane] : 0.0f;
        block_ss = sycl::reduce_over_group(sg, block_ss, sycl::plus<float>());
        if (lane == 0) {
            local_sums[0] = block_ss;
        }
    }
    sycl::group_barrier(item.get_group());

    const float ms = local_sums[0] / (float)N_EMBD;
    const float rs = sycl::rsqrt(ms + eps);

    for (int d = tid; d < N_EMBD; d += bsize) {
        const float val = Rc[d] * rs * w_norm[(size_t)c * N_EMBD + d];
        xnc[d] = val;
        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(mixed[d]);
        ref.fetch_add(val * (1.0f / (float)HC));
    }
}

} // namespace

// ============================================================================
// Benchmark 1: Unfused Residual Pipeline (Separate Kernels)
// ============================================================================

static void BM_Xe2_Residual_Unfused(benchmark::State& state) {
    sycl::queue q{sycl::default_selector_v,
                 {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    const size_t r_elements = (size_t)HC * N_EMBD;
    const size_t w_elements = (size_t)N_EMBD * N_EMBD;

    auto* d_R       = sycl::malloc_device<float>(r_elements, q);
    auto* d_wnorm   = sycl::malloc_device<float>(r_elements, q);
    auto* d_mixed   = sycl::malloc_device<float>(N_EMBD, q);
    auto* d_xn      = sycl::malloc_device<float>(r_elements, q);
    auto* d_out     = sycl::malloc_device<float>(N_EMBD, q);
    auto* d_weight  = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(w_elements, q);

    q.fill(d_R, 1.0f, r_elements);
    q.fill(d_wnorm, 1.0f, r_elements);
    q.fill(d_mixed, 0.0f, N_EMBD);
    q.fill(d_out, 0.5f, N_EMBD);
    q.fill(d_weight, sycl::ext::oneapi::bfloat16(0.01f), w_elements);
    q.wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> lsums(sycl::range<1>(16), cgh);
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(HC * 256), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 gr_norm_kernel(item, d_R, d_wnorm, d_mixed, d_xn, 1e-5f,
                                                lsums.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 residual_gemv_kernel(item, d_mixed, d_weight, d_out);
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(r_elements), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) {
                                 gr_write_kernel(item, d_R, d_out, 0.25f);
                             });
        });
        q.wait();
    }

    for (auto _ : state) {
        q.fill(d_mixed, 0.0f, N_EMBD);

        // Half 1: gr_read -> GEMV -> gr_write
        auto ev1 = q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> lsums(sycl::range<1>(16), cgh);
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(HC * 256), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 gr_norm_kernel(item, d_R, d_wnorm, d_mixed, d_xn, 1e-5f,
                                                lsums.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 residual_gemv_kernel(item, d_mixed, d_weight, d_out);
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(r_elements), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) {
                                 gr_write_kernel(item, d_R, d_out, 0.25f);
                             });
        });

        // Half 2: gr_read -> GEMV -> gr_write
        q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> lsums(sycl::range<1>(16), cgh);
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(HC * 256), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 gr_norm_kernel(item, d_R, d_wnorm, d_mixed, d_xn, 1e-5f,
                                                lsums.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 residual_gemv_kernel(item, d_mixed, d_weight, d_out);
                             });
        });
        auto ev_last = q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(r_elements), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) {
                                 gr_write_kernel(item, d_R, d_out, 0.25f);
                             });
        });
        ev_last.wait();

        const auto t0 = ev1.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev_last.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    // Bytes transferred across 2 halves:
    // Read R (2x), Write R (2x), Weights (2x), Activations
    const double bytes_transferred = (double)(r_elements * 4 * sizeof(float) + w_elements * 2 * sizeof(uint16_t));
    state.counters["Bandwidth_GBps"]         = benchmark::Counter(bytes_transferred, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Roundtrips_Saved"]       = benchmark::Counter(0);
    state.counters["L3_Working_Set_KB"]      = benchmark::Counter((r_elements * sizeof(float) + N_EMBD * 4 * sizeof(float)) / 1024.0);

    sycl::free(d_R, q);
    sycl::free(d_wnorm, q);
    sycl::free(d_mixed, q);
    sycl::free(d_xn, q);
    sycl::free(d_out, q);
    sycl::free(d_weight, q);
}

// ============================================================================
// Benchmark 2: Fused Residual Pipeline (Folding Write into Read)
// ============================================================================

static void BM_Xe2_Residual_Fused(benchmark::State& state) {
    sycl::queue q{sycl::default_selector_v,
                 {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    const size_t r_elements = (size_t)HC * N_EMBD;
    const size_t w_elements = (size_t)N_EMBD * N_EMBD;

    auto* d_R       = sycl::malloc_device<float>(r_elements, q);
    auto* d_wnorm   = sycl::malloc_device<float>(r_elements, q);
    auto* d_mixed   = sycl::malloc_device<float>(N_EMBD, q);
    auto* d_xn      = sycl::malloc_device<float>(r_elements, q);
    auto* d_out     = sycl::malloc_device<float>(N_EMBD, q);
    auto* d_weight  = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(w_elements, q);

    q.fill(d_R, 1.0f, r_elements);
    q.fill(d_wnorm, 1.0f, r_elements);
    q.fill(d_mixed, 0.0f, N_EMBD);
    q.fill(d_out, 0.5f, N_EMBD);
    q.fill(d_weight, sycl::ext::oneapi::bfloat16(0.01f), w_elements);
    q.wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> lsums(sycl::range<1>(16), cgh);
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(HC * 256), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 gr_norm_kernel(item, d_R, d_wnorm, d_mixed, d_xn, 1e-5f,
                                                lsums.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 residual_gemv_kernel(item, d_mixed, d_weight, d_out);
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> lsums(sycl::range<1>(16), cgh);
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(HC * 256), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 fused_gr_write_read_kernel(item, d_R, d_out, d_wnorm, d_mixed, d_xn, 0.25f, 1e-5f,
                                                            lsums.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });
        q.wait();
    }

    for (auto _ : state) {
        q.fill(d_mixed, 0.0f, N_EMBD);

        // Half 1: gr_read -> GEMV
        auto ev1 = q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> lsums(sycl::range<1>(16), cgh);
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(HC * 256), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 gr_norm_kernel(item, d_R, d_wnorm, d_mixed, d_xn, 1e-5f,
                                                lsums.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 residual_gemv_kernel(item, d_mixed, d_weight, d_out);
                             });
        });

        // Fused transition: write(attn) folded into read(ffn)
        q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> lsums(sycl::range<1>(16), cgh);
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(HC * 256), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 fused_gr_write_read_kernel(item, d_R, d_out, d_wnorm, d_mixed, d_xn, 0.25f, 1e-5f,
                                                            lsums.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });

        // Half 2 finish: GEMV -> gr_write
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 residual_gemv_kernel(item, d_mixed, d_weight, d_out);
                             });
        });
        auto ev_last = q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(r_elements), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) {
                                 gr_write_kernel(item, d_R, d_out, 0.25f);
                             });
        });
        ev_last.wait();

        const auto t0 = ev1.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev_last.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    // Saved 1 roundtrip of R (write + read eliminated)
    const double bytes_transferred = (double)(r_elements * 3 * sizeof(float) + w_elements * 2 * sizeof(uint16_t));
    state.counters["Bandwidth_GBps"]         = benchmark::Counter(bytes_transferred, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Roundtrips_Saved"]       = benchmark::Counter(1);
    state.counters["L3_Working_Set_KB"]      = benchmark::Counter((r_elements * sizeof(float) + N_EMBD * 4 * sizeof(float)) / 1024.0);

    sycl::free(d_R, q);
    sycl::free(d_wnorm, q);
    sycl::free(d_mixed, q);
    sycl::free(d_xn, q);
    sycl::free(d_out, q);
    sycl::free(d_weight, q);
}

// ============================================================================
// Benchmark 3: Multi-Layer Chaining (Testing Cache Warmth across Layers)
// ============================================================================

static void BM_Xe2_Residual_Layer_Chaining(benchmark::State& state) {
    const int num_layers = state.range(0);

    sycl::queue q{sycl::default_selector_v,
                 {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    const size_t r_elements = (size_t)HC * N_EMBD;
    const size_t w_elements = (size_t)N_EMBD * N_EMBD;

    auto* d_R       = sycl::malloc_device<float>(r_elements, q);
    auto* d_wnorm   = sycl::malloc_device<float>(r_elements, q);
    auto* d_mixed   = sycl::malloc_device<float>(N_EMBD, q);
    auto* d_xn      = sycl::malloc_device<float>(r_elements, q);
    auto* d_out     = sycl::malloc_device<float>(N_EMBD, q);
    auto* d_weight  = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(w_elements, q);

    q.fill(d_R, 1.0f, r_elements);
    q.fill(d_wnorm, 1.0f, r_elements);
    q.fill(d_mixed, 0.0f, N_EMBD);
    q.fill(d_out, 0.5f, N_EMBD);
    q.fill(d_weight, sycl::ext::oneapi::bfloat16(0.01f), w_elements);
    q.wait();

    for (auto _ : state) {
        sycl::event first_ev, last_ev;

        for (int l = 0; l < num_layers; ++l) {
            auto ev = q.submit([&](sycl::handler& cgh) {
                sycl::local_accessor<float, 1> lsums(sycl::range<1>(16), cgh);
                cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(HC * 256), sycl::range<1>(256)),
                                 [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                     gr_norm_kernel(item, d_R, d_wnorm, d_mixed, d_xn, 1e-5f,
                                                    lsums.get_multi_ptr<sycl::access::decorated::no>().get());
                                 });
            });
            if (l == 0) first_ev = ev;

            q.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD * 16), sycl::range<1>(16)),
                                 [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                     residual_gemv_kernel(item, d_mixed, d_weight, d_out);
                                 });
            });

            last_ev = q.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(r_elements), sycl::range<1>(256)),
                                 [=](sycl::nd_item<1> item) {
                                     gr_write_kernel(item, d_R, d_out, 0.1f);
                                 });
            });
        }
        last_ev.wait();

        const auto t0 = first_ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = last_ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double bytes_transferred = (double)num_layers * (r_elements * 2 * sizeof(float) + w_elements * sizeof(uint16_t));
    state.counters["Bandwidth_GBps"] = benchmark::Counter(bytes_transferred, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Chained_Layers"] = benchmark::Counter(num_layers);

    sycl::free(d_R, q);
    sycl::free(d_wnorm, q);
    sycl::free(d_mixed, q);
    sycl::free(d_xn, q);
    sycl::free(d_out, q);
    sycl::free(d_weight, q);
}

static void BM_Xe2_Residual_Streaming_Layer_Chaining(benchmark::State& state) {
    const int num_layers = state.range(0);

    sycl::queue q{sycl::default_selector_v,
                 {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    const size_t r_elements = (size_t)HC * N_EMBD;
    const size_t w_elements = (size_t)N_EMBD * N_EMBD;

    auto* d_R       = sycl::malloc_device<float>(r_elements, q);
    auto* d_wnorm   = sycl::malloc_device<float>(r_elements, q);
    auto* d_mixed   = sycl::malloc_device<float>(N_EMBD, q);
    auto* d_xn      = sycl::malloc_device<float>(r_elements, q);
    auto* d_out     = sycl::malloc_device<float>(N_EMBD, q);
    auto* d_weight  = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(w_elements, q);

    q.fill(d_R, 1.0f, r_elements);
    q.fill(d_wnorm, 1.0f, r_elements);
    q.fill(d_mixed, 0.0f, N_EMBD);
    q.fill(d_out, 0.5f, N_EMBD);
    q.fill(d_weight, sycl::ext::oneapi::bfloat16(0.01f), w_elements);
    q.wait();

    for (auto _ : state) {
        sycl::event first_ev, last_ev;

        for (int l = 0; l < num_layers; ++l) {
            auto ev = q.submit([&](sycl::handler& cgh) {
                sycl::local_accessor<float, 1> lsums(sycl::range<1>(16), cgh);
                cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(HC * 256), sycl::range<1>(256)),
                                 [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                     gr_norm_kernel(item, d_R, d_wnorm, d_mixed, d_xn, 1e-5f,
                                                    lsums.get_multi_ptr<sycl::access::decorated::no>().get());
                                 });
            });
            if (l == 0) first_ev = ev;

            q.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD * 16), sycl::range<1>(16)),
                                 [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                     residual_gemv_streaming_kernel(item, d_mixed, d_weight, d_out);
                                 });
            });

            last_ev = q.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(r_elements), sycl::range<1>(256)),
                                 [=](sycl::nd_item<1> item) {
                                     gr_write_kernel(item, d_R, d_out, 0.1f);
                                 });
            });
        }
        last_ev.wait();

        const auto t0 = first_ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = last_ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double bytes_transferred = (double)num_layers * (r_elements * 2 * sizeof(float) + w_elements * sizeof(uint16_t));
    state.counters["Bandwidth_GBps"] = benchmark::Counter(bytes_transferred, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Chained_Layers"] = benchmark::Counter(num_layers);
    state.counters["Streaming_Hint"] = benchmark::Counter(1);

    sycl::free(d_R, q);
    sycl::free(d_wnorm, q);
    sycl::free(d_mixed, q);
    sycl::free(d_xn, q);
    sycl::free(d_out, q);
    sycl::free(d_weight, q);
}

// ============================================================================
// Benchmark 4: Multi-Token Hyper-Connection GR Read (Sequential vs Batched)
// ============================================================================

static void BM_Xe2_GR_Read_Sequential(benchmark::State& state) {
    const int B = state.range(0);
    const int64_t hc_dim = (int64_t)HC * N_EMBD;
    const int64_t hc_lr = 320;
    const float eps = 1e-6f;
    const strata::kernels::GrShapes sh{N_EMBD, HC, hc_lr};

    sycl::queue q{sycl::default_selector_v,
                 {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    strata::kernels::gr_set_native_mmvf(true);

    const size_t r_total = (size_t)B * hc_dim;
    const size_t down_total = (size_t)hc_lr * hc_dim;
    const size_t up_total = (size_t)hc_dim * hc_lr;
    const size_t inj_total = (size_t)HC * hc_dim;

    auto* d_R       = sycl::malloc_device<float>(r_total, q);
    auto* d_wnorm   = sycl::malloc_device<float>(hc_dim, q);
    auto* d_wdown   = sycl::malloc_device<uint16_t>(down_total, q);
    auto* d_wup     = sycl::malloc_device<uint16_t>(up_total, q);
    auto* d_winj    = sycl::malloc_device<uint16_t>(inj_total, q);
    auto* d_mixed   = sycl::malloc_device<float>((size_t)B * N_EMBD, q);
    auto* d_inject  = sycl::malloc_device<float>((size_t)B * HC, q);

    void* ws_raw = sycl::malloc_device<uint8_t>(strata::kernels::gr_workspace_bytes(sh) + 64, q);
    strata::kernels::GrWorkspace ws;
    strata::kernels::gr_workspace_init(sh, ws_raw, ws);

    q.fill(d_R, 1.0f, r_total);
    q.fill(d_wnorm, 1.0f, hc_dim);
    q.fill(d_wdown, (uint16_t)0x3c00, down_total);
    q.fill(d_wup, (uint16_t)0x3c00, up_total);
    q.fill(d_winj, (uint16_t)0x3c00, inj_total);
    q.fill(d_mixed, 0.0f, (size_t)B * N_EMBD);
    q.fill(d_inject, 0.0f, (size_t)B * HC).wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        for (int t = 0; t < B; ++t) {
            strata::kernels::gr_read(d_R + (size_t)t * hc_dim, d_wnorm, d_wdown, d_wup, d_winj,
                                     eps, sh, ws, d_mixed + (size_t)t * N_EMBD,
                                     d_inject + (size_t)t * HC, &q);
        }
        q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < B; ++t) {
            strata::kernels::gr_read(d_R + (size_t)t * hc_dim, d_wnorm, d_wdown, d_wup, d_winj,
                                     eps, sh, ws, d_mixed + (size_t)t * N_EMBD,
                                     d_inject + (size_t)t * HC, &q);
        }
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    state.counters["Batch_B"] = benchmark::Counter((double)B);
    state.counters["tokens_per_sec"] = benchmark::Counter((double)B, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    const double bytes_transferred = (double)B * (hc_dim * sizeof(float) + N_EMBD * sizeof(float) + HC * sizeof(float))
                                   + (hc_dim * sizeof(float) + down_total * sizeof(uint16_t) + up_total * sizeof(uint16_t) + inj_total * sizeof(uint16_t));
    state.counters["Bandwidth_GBps"] = benchmark::Counter(bytes_transferred, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_R, q);
    sycl::free(d_wnorm, q);
    sycl::free(d_wdown, q);
    sycl::free(d_wup, q);
    sycl::free(d_winj, q);
    sycl::free(d_mixed, q);
    sycl::free(d_inject, q);
    sycl::free(ws_raw, q);
}

static void BM_Xe2_GR_Read_Batched(benchmark::State& state) {
    const int B = state.range(0);
    const int64_t hc_dim = (int64_t)HC * N_EMBD;
    const int64_t hc_lr = 320;
    const float eps = 1e-6f;
    const strata::kernels::GrShapes sh{N_EMBD, HC, hc_lr};

    sycl::queue q{sycl::default_selector_v,
                 {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    strata::kernels::gr_set_native_mmvf(true);

    const size_t r_total = (size_t)B * hc_dim;
    const size_t down_total = (size_t)hc_lr * hc_dim;
    const size_t up_total = (size_t)hc_dim * hc_lr;
    const size_t inj_total = (size_t)HC * hc_dim;

    auto* d_R       = sycl::malloc_device<float>(r_total, q);
    auto* d_wnorm   = sycl::malloc_device<float>(hc_dim, q);
    auto* d_wdown   = sycl::malloc_device<uint16_t>(down_total, q);
    auto* d_wup     = sycl::malloc_device<uint16_t>(up_total, q);
    auto* d_winj    = sycl::malloc_device<uint16_t>(inj_total, q);
    auto* d_mixed   = sycl::malloc_device<float>((size_t)B * N_EMBD, q);
    auto* d_inject  = sycl::malloc_device<float>((size_t)B * HC, q);
    auto* d_xn      = sycl::malloc_device<float>(r_total, q);
    auto* d_lo      = sycl::malloc_device<float>((size_t)B * hc_lr, q);
    auto* d_gated   = sycl::malloc_device<float>(r_total, q);

    void* ws_raw = sycl::malloc_device<uint8_t>(strata::kernels::gr_workspace_bytes(sh) + 64, q);
    strata::kernels::GrWorkspace ws;
    strata::kernels::gr_workspace_init(sh, ws_raw, ws);

    q.fill(d_R, 1.0f, r_total);
    q.fill(d_wnorm, 1.0f, hc_dim);
    q.fill(d_wdown, (uint16_t)0x3c00, down_total);
    q.fill(d_wup, (uint16_t)0x3c00, up_total);
    q.fill(d_winj, (uint16_t)0x3c00, inj_total);
    q.fill(d_mixed, 0.0f, (size_t)B * N_EMBD);
    q.fill(d_inject, 0.0f, (size_t)B * HC);
    q.fill(d_xn, 0.0f, r_total);
    q.fill(d_lo, 0.0f, (size_t)B * hc_lr);
    q.fill(d_gated, 0.0f, r_total).wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        strata::kernels::gr_read_multi(d_R, d_wnorm, d_wdown, d_wup, d_winj, eps, sh, ws,
                                       d_xn, d_lo, d_gated, d_mixed, d_inject, B, &q);
        q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        strata::kernels::gr_read_multi(d_R, d_wnorm, d_wdown, d_wup, d_winj, eps, sh, ws,
                                       d_xn, d_lo, d_gated, d_mixed, d_inject, B, &q);
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    state.counters["Batch_B"] = benchmark::Counter((double)B);
    state.counters["tokens_per_sec"] = benchmark::Counter((double)B, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    const double bytes_transferred = (double)B * (hc_dim * sizeof(float) + N_EMBD * sizeof(float) + HC * sizeof(float))
                                   + (hc_dim * sizeof(float) + down_total * sizeof(uint16_t) + up_total * sizeof(uint16_t) + inj_total * sizeof(uint16_t));
    state.counters["Bandwidth_GBps"] = benchmark::Counter(bytes_transferred, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_R, q);
    sycl::free(d_wnorm, q);
    sycl::free(d_wdown, q);
    sycl::free(d_wup, q);
    sycl::free(d_winj, q);
    sycl::free(d_mixed, q);
    sycl::free(d_inject, q);
    sycl::free(d_xn, q);
    sycl::free(d_lo, q);
    sycl::free(d_gated, q);
    sycl::free(ws_raw, q);
}

static void BM_Xe2_Fused_GR_Read_Batched(benchmark::State& state) {
    const int B = state.range(0);
    const int64_t hc_dim = (int64_t)HC * N_EMBD;
    const int64_t hc_lr = 320;
    const float eps = 1e-6f;

    sycl::queue q{sycl::default_selector_v,
                 {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    const size_t r_total = (size_t)B * hc_dim;
    const size_t down_total = (size_t)hc_lr * hc_dim;
    const size_t up_total = (size_t)hc_dim * hc_lr;
    const size_t inj_total = (size_t)HC * hc_dim;

    auto* d_R       = sycl::malloc_device<float>(r_total, q);
    auto* d_wnorm   = sycl::malloc_device<float>(hc_dim, q);
    auto* d_wdown   = sycl::malloc_device<uint16_t>(down_total, q);
    auto* d_wup     = sycl::malloc_device<uint16_t>(up_total, q);
    auto* d_winj    = sycl::malloc_device<uint16_t>(inj_total, q);
    auto* d_mixed   = sycl::malloc_device<float>((size_t)B * N_EMBD, q);
    auto* d_inject  = sycl::malloc_device<float>((size_t)B * HC, q);
    auto* d_lo      = sycl::malloc_device<float>((size_t)B * hc_lr, q);
    auto* d_rs      = sycl::malloc_device<float>((size_t)B * HC, q);
    auto* d_xn      = sycl::malloc_device<float>((size_t)B * hc_dim, q);

    q.fill(d_R, 1.0f, r_total);
    q.fill(d_wnorm, 1.0f, hc_dim);
    q.fill(d_wdown, (uint16_t)0x3c00, down_total);
    q.fill(d_wup, (uint16_t)0x3c00, up_total);
    q.fill(d_winj, (uint16_t)0x3c00, inj_total);
    q.fill(d_mixed, 0.0f, (size_t)B * N_EMBD);
    q.fill(d_inject, 0.0f, (size_t)B * HC);
    q.fill(d_lo, 0.0f, (size_t)B * hc_lr);
    q.fill(d_rs, 0.0f, (size_t)B * HC);
    q.fill(d_xn, 0.0f, (size_t)B * hc_dim).wait();

    strata::kernels::FusedGrArgs args[strata::kernels::kFusedGrMaxT];
    for (int t = 0; t < B; ++t) {
        args[t].R = d_R + (size_t)t * hc_dim;
        args[t].R_out = d_R + (size_t)t * hc_dim;
        args[t].apply = false;
        args[t].bo_prev = nullptr;
        args[t].inj_prev = nullptr;
        args[t].w_norm = d_wnorm;
        args[t].w_down = d_wdown;
        args[t].w_up = d_wup;
        args[t].w_inject = d_winj;
        args[t].eps = eps;
        args[t].lo = d_lo + (size_t)t * hc_lr;
        args[t].rs = d_rs + (size_t)t * HC;
        args[t].inject_out = d_inject + (size_t)t * HC;
        args[t].mixed = d_mixed + (size_t)t * N_EMBD;
    }

    // Warmup
    for (int w = 0; w < 3; ++w) {
        strata::kernels::fused_gr_read_multi(args, B, d_xn, &q);
        q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        strata::kernels::fused_gr_read_multi(args, B, d_xn, &q);
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    state.counters["Batch_B"] = benchmark::Counter((double)B);
    state.counters["tokens_per_sec"] = benchmark::Counter((double)B, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    const double bytes_transferred = (double)B * (hc_dim * sizeof(float) + N_EMBD * sizeof(float) + HC * sizeof(float))
                                   + (hc_dim * sizeof(float) + down_total * sizeof(uint16_t) + up_total * sizeof(uint16_t) + inj_total * sizeof(uint16_t));
    state.counters["Bandwidth_GBps"] = benchmark::Counter(bytes_transferred, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_R, q);
    sycl::free(d_wnorm, q);
    sycl::free(d_wdown, q);
    sycl::free(d_wup, q);
    sycl::free(d_winj, q);
    sycl::free(d_mixed, q);
    sycl::free(d_inject, q);
    sycl::free(d_lo, q);
    sycl::free(d_rs, q);
    sycl::free(d_xn, q);
}

// ============================================================================
// Benchmark 5: Multi-Token Hyper-Connection GR Write (Sequential vs Batched)
// ============================================================================

static void BM_Xe2_GR_Write_Sequential(benchmark::State& state) {
    const int B = state.range(0);
    const int64_t hc_dim = (int64_t)HC * N_EMBD;
    const int64_t hc_lr = 320;
    const strata::kernels::GrShapes sh{N_EMBD, HC, hc_lr};

    sycl::queue q{sycl::default_selector_v,
                 {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    strata::kernels::gr_set_native_mmvf(true);

    const size_t r_total = (size_t)B * hc_dim;
    const size_t blk_total = (size_t)B * N_EMBD;
    const size_t inj_total = (size_t)B * HC;

    auto* d_R         = sycl::malloc_device<float>(r_total, q);
    auto* d_block_out = sycl::malloc_device<float>(blk_total, q);
    auto* d_inject    = sycl::malloc_device<float>(inj_total, q);
    auto* d_R_out     = sycl::malloc_device<float>(r_total, q);

    q.fill(d_R, 1.0f, r_total);
    q.fill(d_block_out, 0.5f, blk_total);
    q.fill(d_inject, 0.1f, inj_total);
    q.fill(d_R_out, 0.0f, r_total).wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        for (int t = 0; t < B; ++t) {
            strata::kernels::gr_write(d_R + (size_t)t * hc_dim, d_block_out + (size_t)t * N_EMBD,
                                      d_inject + (size_t)t * HC, sh, d_R_out + (size_t)t * hc_dim, &q);
        }
        q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < B; ++t) {
            strata::kernels::gr_write(d_R + (size_t)t * hc_dim, d_block_out + (size_t)t * N_EMBD,
                                      d_inject + (size_t)t * HC, sh, d_R_out + (size_t)t * hc_dim, &q);
        }
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    state.counters["Batch_B"] = benchmark::Counter((double)B);
    state.counters["tokens_per_sec"] = benchmark::Counter((double)B, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    const double bytes_transferred = (double)B * (2 * hc_dim * sizeof(float) + N_EMBD * sizeof(float) + HC * sizeof(float));
    state.counters["Bandwidth_GBps"] = benchmark::Counter(bytes_transferred, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_R, q);
    sycl::free(d_block_out, q);
    sycl::free(d_inject, q);
    sycl::free(d_R_out, q);
}

static void BM_Xe2_GR_Write_Batched(benchmark::State& state) {
    const int B = state.range(0);
    const int64_t hc_dim = (int64_t)HC * N_EMBD;
    const int64_t hc_lr = 320;
    const strata::kernels::GrShapes sh{N_EMBD, HC, hc_lr};

    sycl::queue q{sycl::default_selector_v,
                 {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    strata::kernels::gr_set_native_mmvf(true);

    const size_t r_total = (size_t)B * hc_dim;
    const size_t blk_total = (size_t)B * N_EMBD;
    const size_t inj_total = (size_t)B * HC;

    auto* d_R         = sycl::malloc_device<float>(r_total, q);
    auto* d_block_out = sycl::malloc_device<float>(blk_total, q);
    auto* d_inject    = sycl::malloc_device<float>(inj_total, q);
    auto* d_R_out     = sycl::malloc_device<float>(r_total, q);

    q.fill(d_R, 1.0f, r_total);
    q.fill(d_block_out, 0.5f, blk_total);
    q.fill(d_inject, 0.1f, inj_total);
    q.fill(d_R_out, 0.0f, r_total).wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        strata::kernels::gr_write_multi(d_R, d_block_out, d_inject, sh, d_R_out, B, &q);
        q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        strata::kernels::gr_write_multi(d_R, d_block_out, d_inject, sh, d_R_out, B, &q);
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    state.counters["Batch_B"] = benchmark::Counter((double)B);
    state.counters["tokens_per_sec"] = benchmark::Counter((double)B, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    const double bytes_transferred = (double)B * (2 * hc_dim * sizeof(float) + N_EMBD * sizeof(float) + HC * sizeof(float));
    state.counters["Bandwidth_GBps"] = benchmark::Counter(bytes_transferred, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_R, q);
    sycl::free(d_block_out, q);
    sycl::free(d_inject, q);
    sycl::free(d_R_out, q);
}

// Register Benchmarks
BENCHMARK(BM_Xe2_Residual_Unfused)->UseManualTime();
BENCHMARK(BM_Xe2_Residual_Fused)->UseManualTime();
BENCHMARK(BM_Xe2_Residual_Layer_Chaining)->RangeMultiplier(2)->Range(1, 8)->UseManualTime();
BENCHMARK(BM_Xe2_Residual_Streaming_Layer_Chaining)->RangeMultiplier(2)->Range(1, 8)->UseManualTime();
BENCHMARK(BM_Xe2_GR_Read_Sequential)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->UseManualTime();
BENCHMARK(BM_Xe2_GR_Read_Batched)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->UseManualTime();
BENCHMARK(BM_Xe2_Fused_GR_Read_Batched)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->UseManualTime();
BENCHMARK(BM_Xe2_GR_Write_Sequential)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->UseManualTime();
BENCHMARK(BM_Xe2_GR_Write_Batched)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->UseManualTime();

BENCHMARK_MAIN();
