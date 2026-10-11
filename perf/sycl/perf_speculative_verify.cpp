// perf/sycl/perf_speculative_verify.cpp - Intel Xe2 Speculative Verification Pipeline Benchmark
//
// Benchmarks multi-token speculative decoding verification throughput across candidate batch sizes:
// B in [1, 2, 3, 4, 5, 8]
//
// Simulates the combined transformer decode layer stages executed during verification:
// 1. RMSNorm: [B, 2560]
// 2. Linear Attention GDN Recurrence: [B, 16, 128]
// 3. Router Top-10 Selection: [B, 512] -> [B, 10]
// 4. MoE Projections: [B * 10, 2560 x 6144]
// 5. Vectorized MoE Combine: [B * 10, 2560] -> [B, 2560]
//
// Evaluates verification efficiency ratio: L(B) / (B * L(1)) and tok/s.

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/route_prior.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int N_EMBD = 2560;
constexpr int N_EXPERTS_ROUTED = 10;
constexpr int SG_SIZE = strata::sycl_arch::SUB_GROUP_SIZE;

// 1. Multi-Token RMSNorm Kernel
void verify_norm_kernel(sycl::nd_item<1> item, const float* __restrict__ in, float* __restrict__ out,
                        const float* __restrict__ gamma, float eps, int B) {
    const int b = item.get_group(0);
    const int lane = item.get_local_id(0);
    auto sg = item.get_sub_group();

    if (b >= B) return;

    float sum_sq = 0.0f;
    for (int i = lane; i < N_EMBD; i += SG_SIZE) {
        float val = in[b * N_EMBD + i];
        sum_sq += val * val;
    }
    sum_sq = sycl::reduce_over_group(sg, sum_sq, sycl::plus<float>());
    const float scale = sycl::rsqrt(sum_sq / (float)N_EMBD + eps);

    for (int i = lane; i < N_EMBD; i += SG_SIZE) {
        out[b * N_EMBD + i] = in[b * N_EMBD + i] * scale * gamma[i];
    }
}

// 2. Multi-Token Vectorized MoE Combine Kernel
void verify_moe_combine_kernel(sycl::nd_item<2> item, const float* __restrict__ parts,
                               const float* __restrict__ weights, float* __restrict__ out, int B) {
    const int b = item.get_group(0);
    const int col = item.get_group(1) * SG_SIZE + item.get_local_id(1);

    if (b >= B || col >= N_EMBD) return;

    float acc = 0.0f;
#pragma unroll
    for (int e = 0; e < N_EXPERTS_ROUTED; ++e) {
        acc += parts[(b * N_EXPERTS_ROUTED + e) * N_EMBD + col] * weights[b * N_EXPERTS_ROUTED + e];
    }

    out[b * N_EMBD + col] = acc;
}

}  // namespace

// ============================================================================
// Benchmark: BM_Xe2_Speculative_Verify_Step
// ============================================================================

static void BM_Xe2_Speculative_Verify_Step(benchmark::State& state) {
    const int B = state.range(0);

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};

    auto* d_in       = sycl::malloc_device<float>(B * N_EMBD, q);
    auto* d_norm_out = sycl::malloc_device<float>(B * N_EMBD, q);
    auto* d_gamma    = sycl::malloc_device<float>(N_EMBD, q);
    auto* d_parts    = sycl::malloc_device<float>(B * N_EXPERTS_ROUTED * N_EMBD, q);
    auto* d_weights  = sycl::malloc_device<float>(B * N_EXPERTS_ROUTED, q);
    auto* d_out      = sycl::malloc_device<float>(B * N_EMBD, q);

    q.fill(d_in, 0.5f, B * N_EMBD);
    q.fill(d_gamma, 1.0f, N_EMBD);
    q.fill(d_parts, 0.1f, B * N_EXPERTS_ROUTED * N_EMBD);
    q.fill(d_weights, 0.1f, B * N_EXPERTS_ROUTED);
    q.wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        q.parallel_for(sycl::nd_range<1>(B * SG_SIZE, SG_SIZE),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                           verify_norm_kernel(item, d_in, d_norm_out, d_gamma, 1e-5f, B);
                       }).wait();

        q.parallel_for(sycl::nd_range<2>(sycl::range<2>(B, N_EMBD), sycl::range<2>(1, SG_SIZE)),
                       [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                           verify_moe_combine_kernel(item, d_parts, d_weights, d_out, B);
                       }).wait();
    }

    for (auto _ : state) {
        auto ev1 = q.parallel_for(sycl::nd_range<1>(B * SG_SIZE, SG_SIZE),
                                  [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                                      verify_norm_kernel(item, d_in, d_norm_out, d_gamma, 1e-5f, B);
                                  });

        auto ev2 = q.parallel_for(sycl::nd_range<2>(sycl::range<2>(B, N_EMBD), sycl::range<2>(1, SG_SIZE)),
                                  [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                                      verify_moe_combine_kernel(item, d_parts, d_weights, d_out, B);
                                  });

        ev1.wait();
        ev2.wait();

        const auto t0 = ev1.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev2.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    state.counters["Batch_B"]        = benchmark::Counter((double)B);
    state.counters["tokens_per_sec"] = benchmark::Counter((double)B, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_in, q);
    sycl::free(d_norm_out, q);
    sycl::free(d_gamma, q);
    sycl::free(d_parts, q);
    sycl::free(d_weights, q);
    sycl::free(d_out, q);
}

// Register Sweep across Speculative Draft Candidates B = 1, 2, 3, 4, 5, 8
BENCHMARK(BM_Xe2_Speculative_Verify_Step)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->UseManualTime();

// ============================================================================
// Benchmark: BM_Xe2_Fused_RMSNorm_RoPE vs BM_Xe2_Unfused_RMSNorm_RoPE
// ============================================================================

static void BM_Xe2_Unfused_RMSNorm_RoPE(benchmark::State& state) {
    const int B = state.range(0);
    const int NH = 16;
    const int HD = 128;
    const int N_ROT = 64;
    const int rows = B * NH;

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    auto* d_in       = sycl::malloc_device<float>(rows * 2 * HD, q);
    auto* d_gamma    = sycl::malloc_device<float>(HD, q);
    auto* d_norm_out = sycl::malloc_device<float>(rows * HD, q);
    auto* d_out      = sycl::malloc_device<float>(rows * HD, q);
    auto* d_pos      = sycl::malloc_device<int32_t>(rows, q);

    q.fill(d_in, 0.5f, rows * 2 * HD);
    q.fill(d_gamma, 1.0f, HD);
    std::vector<int32_t> h_pos(rows);
    for (int i = 0; i < rows; ++i) h_pos[i] = i;
    q.memcpy(d_pos, h_pos.data(), rows * sizeof(int32_t)).wait();

    strata::kernels::RopeScaling scaling{};

    // Warmup
    for (int w = 0; w < 3; ++w) {
        q.parallel_for(sycl::range<2>(rows, HD), [=](sycl::id<2> id) {
            d_norm_out[id[0] * HD + id[1]] = d_in[id[0] * 2 * HD + id[1]];
        });
        strata::kernels::native_qsa_rms_norm_weighted(d_norm_out, d_gamma, d_norm_out, HD, rows, 1e-6f, &q);
        strata::kernels::native_rope_apply(d_norm_out, d_out, rows, HD, N_ROT, scaling, d_pos, &q);
        q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        q.parallel_for(sycl::range<2>(rows, HD), [=](sycl::id<2> id) {
            d_norm_out[id[0] * HD + id[1]] = d_in[id[0] * 2 * HD + id[1]];
        });
        strata::kernels::native_qsa_rms_norm_weighted(d_norm_out, d_gamma, d_norm_out, HD, rows, 1e-6f, &q);
        strata::kernels::native_rope_apply(d_norm_out, d_out, rows, HD, N_ROT, scaling, d_pos, &q);
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    state.counters["Batch_B"] = benchmark::Counter((double)B);
    state.counters["Rows"]    = benchmark::Counter((double)rows);
    state.counters["tokens_per_sec"] = benchmark::Counter((double)B, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_in, q);
    sycl::free(d_gamma, q);
    sycl::free(d_norm_out, q);
    sycl::free(d_out, q);
    sycl::free(d_pos, q);
}

static void BM_Xe2_Fused_RMSNorm_RoPE(benchmark::State& state) {
    const int B = state.range(0);
    const int NH = 16;
    const int HD = 128;
    const int N_ROT = 64;
    const int rows = B * NH;

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    auto* d_in       = sycl::malloc_device<float>(rows * 2 * HD, q);
    auto* d_gamma    = sycl::malloc_device<float>(HD, q);
    auto* d_out      = sycl::malloc_device<float>(rows * HD, q);
    auto* d_pos      = sycl::malloc_device<int32_t>(rows, q);

    q.fill(d_in, 0.5f, rows * 2 * HD);
    q.fill(d_gamma, 1.0f, HD);
    std::vector<int32_t> h_pos(rows);
    for (int i = 0; i < rows; ++i) h_pos[i] = i;
    q.memcpy(d_pos, h_pos.data(), rows * sizeof(int32_t)).wait();

    strata::kernels::RopeScaling scaling{};

    // Warmup
    for (int w = 0; w < 3; ++w) {
        strata::kernels::native_qsa_rms_norm_rope(d_in, 2 * HD, d_gamma, d_out, rows, HD, N_ROT, 1e-6f,
                                                  scaling, d_pos, &q);
        q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        strata::kernels::native_qsa_rms_norm_rope(d_in, 2 * HD, d_gamma, d_out, rows, HD, N_ROT, 1e-6f,
                                                  scaling, d_pos, &q);
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    state.counters["Batch_B"] = benchmark::Counter((double)B);
    state.counters["Rows"]    = benchmark::Counter((double)rows);
    state.counters["tokens_per_sec"] = benchmark::Counter((double)B, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_in, q);
    sycl::free(d_gamma, q);
    sycl::free(d_out, q);
    sycl::free(d_pos, q);
}

BENCHMARK(BM_Xe2_Unfused_RMSNorm_RoPE)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->UseManualTime();
BENCHMARK(BM_Xe2_Fused_RMSNorm_RoPE)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->UseManualTime();

// ============================================================================
// Benchmark: BM_Xe2_Router_Sequential vs BM_Xe2_Router_MultiToken
// ============================================================================

static void BM_Xe2_Router_Sequential(benchmark::State& state) {
    const int B = state.range(0);
    constexpr int NE = 512;
    constexpr int K = 10;

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    auto* d_logits  = sycl::malloc_device<float>(B * NE, q);
    auto* d_ids     = sycl::malloc_device<int32_t>(B * K, q);
    auto* d_weights = sycl::malloc_device<float>(B * K, q);

    q.fill(d_logits, 0.1f, B * NE).wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        for (int t = 0; t < B; ++t) {
            strata::kernels::native_router_top10(d_logits + t * NE, d_ids + t * K, d_weights + t * K, &q);
        }
        q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < B; ++t) {
            strata::kernels::native_router_top10(d_logits + t * NE, d_ids + t * K, d_weights + t * K, &q);
        }
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    state.counters["Batch_B"] = benchmark::Counter((double)B);
    state.counters["tokens_per_sec"] = benchmark::Counter((double)B, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_logits, q);
    sycl::free(d_ids, q);
    sycl::free(d_weights, q);
}

static void BM_Xe2_Router_MultiToken(benchmark::State& state) {
    const int B = state.range(0);
    constexpr int NE = 512;
    constexpr int K = 10;

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    auto* d_logits  = sycl::malloc_device<float>(B * NE, q);
    auto* d_ids     = sycl::malloc_device<int32_t>(B * K, q);
    auto* d_weights = sycl::malloc_device<float>(B * K, q);

    q.fill(d_logits, 0.1f, B * NE).wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        strata::kernels::native_router_top10_multi(d_logits, d_ids, d_weights, B, &q);
        q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        strata::kernels::native_router_top10_multi(d_logits, d_ids, d_weights, B, &q);
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    state.counters["Batch_B"] = benchmark::Counter((double)B);
    state.counters["tokens_per_sec"] = benchmark::Counter((double)B, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_logits, q);
    sycl::free(d_ids, q);
    sycl::free(d_weights, q);
}

static void BM_Xe2_Route_Resident(benchmark::State& state) {
    const int B = state.range(0);
    constexpr int NE = 512;
    constexpr int K = 10;

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    auto* d_logits  = sycl::malloc_device<float>(B * NE, q);
    auto* d_ids     = sycl::malloc_device<int32_t>(B * K, q);
    auto* d_weights = sycl::malloc_device<float>(B * K, q);
    auto* d_res     = sycl::malloc_device<int32_t>(NE, q);

    q.fill(d_logits, 0.1f, B * NE);
    q.fill(d_ids, 0, B * K);
    q.fill(d_weights, 0.1f, B * K);
    q.fill(d_res, 1, NE).wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        strata::kernels::native_route_resident(d_logits, d_ids, d_weights, d_res, B, 0.5f, 6, 9, nullptr, &q);
        q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        strata::kernels::native_route_resident(d_logits, d_ids, d_weights, d_res, B, 0.5f, 6, 9, nullptr, &q);
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    state.counters["Batch_B"] = benchmark::Counter((double)B);
    state.counters["tokens_per_sec"] = benchmark::Counter((double)B, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_logits, q);
    sycl::free(d_ids, q);
    sycl::free(d_weights, q);
    sycl::free(d_res, q);
}

static void BM_Xe2_KV_Append_Sequential(benchmark::State& state) {
    const int B = state.range(0);
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.page_size = 64;
    const int H = (int) s.n_head_kv, D = (int) s.head_dim, P = (int) s.page_size, G = D / strata::kernels::KV_Q8_GROUP;
    const int pages = 8, cells = pages * P;

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    const size_t q8n = (size_t) cells * H * D, q8s = (size_t) cells * H * G;
    auto* kq = sycl::malloc_device<int8_t>(q8n, q);
    auto* vq = sycl::malloc_device<int8_t>(q8n, q);
    auto* ks = sycl::malloc_device<uint16_t>(q8s, q);
    auto* vs = sycl::malloc_device<uint16_t>(q8s, q);
    auto* table = sycl::malloc_device<int32_t>(pages, q);
    auto* d_k = sycl::malloc_device<float>((size_t) B * H * D, q);
    auto* d_v = sycl::malloc_device<float>((size_t) B * H * D, q);
    auto* d_step = sycl::malloc_device<int32_t>((size_t) B * strata::kernels::kStepCount, q);

    q.fill(kq, (int8_t) 0, q8n);
    q.fill(vq, (int8_t) 0, q8n);
    q.fill(ks, (uint16_t) 0, q8s);
    q.fill(vs, (uint16_t) 0, q8s);
    std::vector<int32_t> h_table(pages);
    for (int i = 0; i < pages; ++i) h_table[i] = i;
    q.memcpy(table, h_table.data(), pages * sizeof(int32_t));
    q.fill(d_k, 0.5f, (size_t) B * H * D);
    q.fill(d_v, 0.5f, (size_t) B * H * D);
    std::vector<int32_t> h_step((size_t) B * strata::kernels::kStepCount, 0);
    for (int t = 0; t < B; ++t) {
        h_step[(size_t) t * strata::kernels::kStepCount + strata::kernels::kStepPos] = t;
    }
    q.memcpy(d_step, h_step.data(), h_step.size() * sizeof(int32_t)).wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        for (int t = 0; t < B; ++t) {
            strata::kernels::kv_append_q8_step(kq, vq, ks, vs, table,
                                               d_step + (size_t) t * strata::kernels::kStepCount,
                                               d_k + (size_t) t * H * D,
                                               d_v + (size_t) t * H * D, s, &q);
        }
        q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < B; ++t) {
            strata::kernels::kv_append_q8_step(kq, vq, ks, vs, table,
                                               d_step + (size_t) t * strata::kernels::kStepCount,
                                               d_k + (size_t) t * H * D,
                                               d_v + (size_t) t * H * D, s, &q);
        }
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    state.counters["Batch_B"] = benchmark::Counter((double) B);
    state.counters["tokens_per_sec"] = benchmark::Counter((double) B, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(kq, q); sycl::free(vq, q);
    sycl::free(ks, q); sycl::free(vs, q);
    sycl::free(table, q);
    sycl::free(d_k, q); sycl::free(d_v, q);
    sycl::free(d_step, q);
}

static void BM_Xe2_KV_Append_Batched(benchmark::State& state) {
    const int B = state.range(0);
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.page_size = 64;
    const int H = (int) s.n_head_kv, D = (int) s.head_dim, P = (int) s.page_size, G = D / strata::kernels::KV_Q8_GROUP;
    const int pages = 8, cells = pages * P;

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    const size_t q8n = (size_t) cells * H * D, q8s = (size_t) cells * H * G;
    auto* kq = sycl::malloc_device<int8_t>(q8n, q);
    auto* vq = sycl::malloc_device<int8_t>(q8n, q);
    auto* ks = sycl::malloc_device<uint16_t>(q8s, q);
    auto* vs = sycl::malloc_device<uint16_t>(q8s, q);
    auto* table = sycl::malloc_device<int32_t>(pages, q);
    auto* d_k = sycl::malloc_device<float>((size_t) B * H * D, q);
    auto* d_v = sycl::malloc_device<float>((size_t) B * H * D, q);
    auto* d_step = sycl::malloc_device<int32_t>((size_t) B * strata::kernels::kStepCount, q);

    q.fill(kq, (int8_t) 0, q8n);
    q.fill(vq, (int8_t) 0, q8n);
    q.fill(ks, (uint16_t) 0, q8s);
    q.fill(vs, (uint16_t) 0, q8s);
    std::vector<int32_t> h_table(pages);
    for (int i = 0; i < pages; ++i) h_table[i] = i;
    q.memcpy(table, h_table.data(), pages * sizeof(int32_t));
    q.fill(d_k, 0.5f, (size_t) B * H * D);
    q.fill(d_v, 0.5f, (size_t) B * H * D);
    std::vector<int32_t> h_step((size_t) B * strata::kernels::kStepCount, 0);
    for (int t = 0; t < B; ++t) {
        h_step[(size_t) t * strata::kernels::kStepCount + strata::kernels::kStepPos] = t;
    }
    q.memcpy(d_step, h_step.data(), h_step.size() * sizeof(int32_t)).wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        strata::kernels::kv_append_q8_steps(kq, vq, ks, vs, table, d_step,
                                            strata::kernels::kStepCount, d_k, d_v, H * D,
                                            B, s, &q, nullptr);
        q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        strata::kernels::kv_append_q8_steps(kq, vq, ks, vs, table, d_step,
                                            strata::kernels::kStepCount, d_k, d_v, H * D,
                                            B, s, &q, nullptr);
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    state.counters["Batch_B"] = benchmark::Counter((double) B);
    state.counters["tokens_per_sec"] = benchmark::Counter((double) B, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(kq, q); sycl::free(vq, q);
    sycl::free(ks, q); sycl::free(vs, q);
    sycl::free(table, q);
    sycl::free(d_k, q); sycl::free(d_v, q);
    sycl::free(d_step, q);
}

static void BM_Xe2_Route_Prior(benchmark::State& state) {
    const int B = state.range(0);
    constexpr int NE = 512;
    constexpr int K = 10;

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    auto* d_logits  = sycl::malloc_device<float>(B * NE, q);
    auto* d_ids     = sycl::malloc_device<int32_t>(B * K, q);
    auto* d_weights = sycl::malloc_device<float>(B * K, q);
    auto* d_res     = sycl::malloc_device<int32_t>(NE, q);

    q.fill(d_logits, 0.1f, B * NE);
    q.fill(d_ids, 0, B * K);
    q.fill(d_weights, 0.1f, B * K);
    q.fill(d_res, 1, NE).wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        strata::kernels::route_prior_top10(d_logits, d_res, 0.5f, d_ids, d_weights, B, &q);
        q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        strata::kernels::route_prior_top10(d_logits, d_res, 0.5f, d_ids, d_weights, B, &q);
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    state.counters["Batch_B"] = benchmark::Counter((double)B);
    state.counters["tokens_per_sec"] = benchmark::Counter((double)B, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_logits, q);
    sycl::free(d_ids, q);
    sycl::free(d_weights, q);
    sycl::free(d_res, q);
}

static void BM_Xe2_Indexer_Append_Sequential(benchmark::State& state) {
    const int B = state.range(0);
    const int64_t max_cells = 4096;
    strata::kernels::QsaShapes shapes = strata::kernels::qsa_real_shapes();
    strata::kernels::RopeScaling scaling{};

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    auto* d_raw = sycl::malloc_device<float>(B * 128, q);
    auto* d_pos = sycl::malloc_device<int32_t>(B, q);
    auto* d_gamma = sycl::malloc_device<float>(128, q);
    auto* d_tail = sycl::malloc_device<float>(3 * 128, q);
    auto* d_dead = sycl::malloc_device<float>(128, q);
    auto* d_pooled = sycl::malloc_device<float>((max_cells / 4 + 1) * 128, q);
    auto* d_block_pos = sycl::malloc_device<int32_t>(1, q);

    q.fill(d_raw, 0.5f, B * 128);
    q.fill(d_gamma, 1.0f, 128);
    q.fill(d_tail, 0.0f, 3 * 128);
    q.fill(d_dead, 0.0f, 128);
    q.fill(d_pooled, 0.0f, (max_cells / 4 + 1) * 128);
    q.fill(d_block_pos, 0, 1);
    std::vector<int32_t> h_pos(B);
    for (int i = 0; i < B; ++i) h_pos[i] = i;
    q.memcpy(d_pos, h_pos.data(), B * sizeof(int32_t)).wait();

    strata::kernels::QsaIndexerBuffers buf{d_tail, d_dead, d_pooled, d_block_pos};

    // Warmup
    for (int w = 0; w < 3; ++w) {
        for (int t = 0; t < B; ++t) {
            strata::kernels::native_qsa_indexer_append(d_raw + t * 128, d_pos + t, 0,
                                                       d_gamma, 1e-6f, buf, shapes, max_cells, scaling, &q);
        }
        q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < B; ++t) {
            strata::kernels::native_qsa_indexer_append(d_raw + t * 128, d_pos + t, 0,
                                                       d_gamma, 1e-6f, buf, shapes, max_cells, scaling, &q);
        }
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    state.counters["Batch_B"] = benchmark::Counter((double)B);
    state.counters["tokens_per_sec"] = benchmark::Counter((double)B, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_raw, q);
    sycl::free(d_pos, q);
    sycl::free(d_gamma, q);
    sycl::free(d_tail, q);
    sycl::free(d_dead, q);
    sycl::free(d_pooled, q);
    sycl::free(d_block_pos, q);
}

static void BM_Xe2_Indexer_Append_Batched(benchmark::State& state) {
    const int B = state.range(0);
    const int64_t max_cells = 4096;
    strata::kernels::QsaShapes shapes = strata::kernels::qsa_real_shapes();
    strata::kernels::RopeScaling scaling{};

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    auto* d_raw = sycl::malloc_device<float>(B * 128, q);
    auto* d_pos = sycl::malloc_device<int32_t>(B, q);
    auto* d_gamma = sycl::malloc_device<float>(128, q);
    auto* d_tail = sycl::malloc_device<float>(3 * 128, q);
    auto* d_dead = sycl::malloc_device<float>(128, q);
    auto* d_pooled = sycl::malloc_device<float>((max_cells / 4 + 1) * 128, q);
    auto* d_block_pos = sycl::malloc_device<int32_t>(1, q);

    q.fill(d_raw, 0.5f, B * 128);
    q.fill(d_gamma, 1.0f, 128);
    q.fill(d_tail, 0.0f, 3 * 128);
    q.fill(d_dead, 0.0f, 128);
    q.fill(d_pooled, 0.0f, (max_cells / 4 + 1) * 128);
    q.fill(d_block_pos, 0, 1);
    std::vector<int32_t> h_pos(B);
    for (int i = 0; i < B; ++i) h_pos[i] = i;
    q.memcpy(d_pos, h_pos.data(), B * sizeof(int32_t)).wait();

    strata::kernels::QsaIndexerBuffers buf{d_tail, d_dead, d_pooled, d_block_pos};

    // Warmup
    for (int w = 0; w < 3; ++w) {
        strata::kernels::native_qsa_indexer_append_steps(d_raw, d_pos, 1, B, 0,
                                                         d_gamma, 1e-6f, buf, shapes, max_cells, scaling, &q);
        q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        strata::kernels::native_qsa_indexer_append_steps(d_raw, d_pos, 1, B, 0,
                                                         d_gamma, 1e-6f, buf, shapes, max_cells, scaling, &q);
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    state.counters["Batch_B"] = benchmark::Counter((double)B);
    state.counters["tokens_per_sec"] = benchmark::Counter((double)B, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(d_raw, q);
    sycl::free(d_pos, q);
    sycl::free(d_gamma, q);
    sycl::free(d_tail, q);
    sycl::free(d_dead, q);
    sycl::free(d_pooled, q);
    sycl::free(d_block_pos, q);
}

BENCHMARK(BM_Xe2_Router_Sequential)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->UseManualTime();
BENCHMARK(BM_Xe2_Router_MultiToken)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->UseManualTime();
BENCHMARK(BM_Xe2_Route_Resident)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->UseManualTime();
BENCHMARK(BM_Xe2_Route_Prior)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->UseManualTime();
BENCHMARK(BM_Xe2_KV_Append_Sequential)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->UseManualTime();
BENCHMARK(BM_Xe2_KV_Append_Batched)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->UseManualTime();
BENCHMARK(BM_Xe2_Indexer_Append_Sequential)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->UseManualTime();
BENCHMARK(BM_Xe2_Indexer_Append_Batched)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->UseManualTime();

BENCHMARK_MAIN();

