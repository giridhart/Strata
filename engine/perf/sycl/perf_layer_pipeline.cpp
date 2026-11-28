// perf/sycl/perf_layer_pipeline.cpp - Intel Xe2 Full Composite Layer Pipeline Benchmark
//
// Simulates an entire 9-kernel transformer/MoE layer end-to-end:
// 1. gr_norm(attn): RMSNorm on R (40.96 KB) -> mixed (10.24 KB)
// 2. qkv_gemv: 2560 -> 10240 projection
// 3. gdn_recurrent: Fused linear attention recurrence in 64 KB SLM
// 4. ssm_out_gemv: 2560 -> 2560 projection
// 5. gr_write(attn): residual update on R
// 6. ffn_norm: RMSNorm on R -> mixed_ffn
// 7. moe_router: 2560 -> 64 router GEMV + top-10 selection
// 8. moe_shared_gemv: 2560 -> 6144 -> 2560 shared expert
// 9. moe_combine: 10-expert weighted reduction + residual update
//
// Compares:
// 1. BM_Xe2_Composite_Layer_Pipelined: 9 kernels queued back-to-back in an in-order queue
// 2. BM_Xe2_Composite_Layer_Graph: 9 kernels captured into strata::core::CapturedGraph

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

#include "strata/core/graph.hpp"
#include "strata/core/sycl_arch.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int HC = 4;
constexpr int N_EMBD = 2560;
constexpr int QKV_DIM = 10240;
constexpr int ROUTER_EXPERTS = 64;
constexpr int K_EXP = 10;
constexpr int S_DIM = 16;
constexpr int SUB_SIZE = 16;

// Kernel 1: gr_norm
void k_gr_norm(sycl::nd_item<1> item, const float* __restrict__ R, const float* __restrict__ w_norm,
               float* __restrict__ mixed, float eps, float* __restrict__ local_sums) {
    const int c = item.get_group(0);
    if (c >= HC) return;
    const int tid = item.get_local_id(0);
    const int bsize = item.get_local_range(0);
    const float* Rc = R + (size_t)c * N_EMBD;

    float ss = 0.0f;
    for (int d = tid; d < N_EMBD; d += bsize) {
        float v = Rc[d];
        ss += v * v;
    }
    auto sg = item.get_sub_group();
    ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());

    const int warp = tid / SUB_SIZE;
    const int lane = tid % SUB_SIZE;
    if (lane == 0) local_sums[warp] = ss;
    sycl::group_barrier(item.get_group());

    if (warp == 0) {
        float bss = (lane < bsize / SUB_SIZE) ? local_sums[lane] : 0.0f;
        bss = sycl::reduce_over_group(sg, bss, sycl::plus<float>());
        if (lane == 0) local_sums[0] = bss;
    }
    sycl::group_barrier(item.get_group());

    const float ms = local_sums[0] / (float)N_EMBD;
    const float rs = sycl::rsqrt(ms + eps);

    for (int d = tid; d < N_EMBD; d += bsize) {
        float val = Rc[d] * rs * w_norm[(size_t)c * N_EMBD + d];
        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(mixed[d]);
        ref.fetch_add(val * (1.0f / (float)HC));
    }
}

// Kernel 2: qkv_gemv
void k_qkv_gemv(sycl::nd_item<1> item, const float* __restrict__ in,
                const sycl::ext::oneapi::bfloat16* __restrict__ weight,
                float* __restrict__ out) {
    const int row = item.get_group(0);
    if (row >= QKV_DIM) return;
    const int tid = item.get_local_id(0);
    const int bsize = item.get_local_range(0);
    const auto* w_row = weight + (size_t)row * N_EMBD;

    float acc = 0.0f;
    for (int col = tid; col < N_EMBD; col += bsize) {
        acc += in[col] * (float)w_row[col];
    }
    auto sg = item.get_sub_group();
    acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (tid == 0) out[row] = acc;
}

// Kernel 3: gdn_recurrent step
void k_gdn_step(sycl::nd_item<1> item, float* __restrict__ state, const float* __restrict__ qkv,
                float* __restrict__ out) {
    const int head = item.get_group(0);
    const int tid = item.get_local_id(0);
    if (head >= 16) return;

    // Local SLM state update
    float* h_state = state + (size_t)head * S_DIM * S_DIM;
    if (tid < S_DIM) {
        float q_val = qkv[head * S_DIM + tid];
        float acc = 0.0f;
        for (int j = 0; j < S_DIM; ++j) {
            h_state[tid * S_DIM + j] = h_state[tid * S_DIM + j] * 0.95f + 0.05f;
            acc += h_state[tid * S_DIM + j] * q_val;
        }
        out[head * S_DIM + tid] = acc * 0.25f;
    }
}

// Kernel 4: ssm_out_gemv
void k_ssm_out_gemv(sycl::nd_item<1> item, const float* __restrict__ in,
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
    if (tid == 0) out[row] = acc;
}

// Kernel 5: gr_write
void k_gr_write(sycl::nd_item<1> item, float* __restrict__ R, const float* __restrict__ out, float alpha) {
    const int idx = item.get_global_id(0);
    if (idx < HC * N_EMBD) {
        R[idx] += alpha * out[idx % N_EMBD];
    }
}

// Kernel 6: moe_router (2560 -> 64)
void k_moe_router(sycl::nd_item<1> item, const float* __restrict__ in,
                  const sycl::ext::oneapi::bfloat16* __restrict__ w_router,
                  float* __restrict__ logits, int32_t* __restrict__ topk_ids,
                  float* __restrict__ topk_weights) {
    const int exp = item.get_group(0);
    if (exp >= ROUTER_EXPERTS) return;
    const int tid = item.get_local_id(0);
    const int bsize = item.get_local_range(0);
    const auto* w_row = w_router + (size_t)exp * N_EMBD;

    float acc = 0.0f;
    for (int col = tid; col < N_EMBD; col += bsize) {
        acc += in[col] * (float)w_row[col];
    }
    auto sg = item.get_sub_group();
    acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (tid == 0) {
        logits[exp] = acc;
        if (exp < K_EXP) {
            topk_ids[exp] = exp;
            topk_weights[exp] = 0.1f;
        }
    }
}

// Kernel 7: moe_shared_gemv
void k_moe_shared_gemv(sycl::nd_item<1> item, const float* __restrict__ in,
                       const sycl::ext::oneapi::bfloat16* __restrict__ w_gate,
                       float* __restrict__ out) {
    const int row = item.get_group(0);
    if (row >= N_EMBD) return;
    const int tid = item.get_local_id(0);
    const int bsize = item.get_local_range(0);
    const auto* w_row = w_gate + (size_t)row * N_EMBD;

    float acc = 0.0f;
    for (int col = tid; col < N_EMBD; col += bsize) {
        acc += in[col] * (float)w_row[col];
    }
    auto sg = item.get_sub_group();
    acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (tid == 0) out[row] = acc;
}

// Kernel 8: moe_combine
void k_moe_combine(sycl::nd_item<1> item, const float* __restrict__ shared,
                   const float* __restrict__ parts, const float* __restrict__ weights,
                   float* __restrict__ R) {
    const int d = item.get_global_id(0);
    if (d >= N_EMBD) return;

    float sum = shared[d];
    for (int k = 0; k < K_EXP; ++k) {
        sum += parts[k * N_EMBD + d] * weights[k];
    }
    for (int c = 0; c < HC; ++c) {
        R[c * N_EMBD + d] += 0.25f * sum;
    }
}

} // namespace

// ============================================================================
// Benchmark: BM_Xe2_Composite_Layer_Pipelined
// ============================================================================

static void BM_Xe2_Composite_Layer_Pipelined(benchmark::State& state) {
    sycl::queue q{sycl::default_selector_v,
                 {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    const size_t r_bytes = (size_t)HC * N_EMBD * sizeof(float);
    const size_t qkv_w_bytes = (size_t)QKV_DIM * N_EMBD * sizeof(uint16_t);
    const size_t ssm_w_bytes = (size_t)N_EMBD * N_EMBD * sizeof(uint16_t);
    const size_t router_w_bytes = (size_t)ROUTER_EXPERTS * N_EMBD * sizeof(uint16_t);
    const size_t shared_w_bytes = (size_t)N_EMBD * N_EMBD * sizeof(uint16_t);

    auto* d_R       = sycl::malloc_device<float>(HC * N_EMBD, q);
    auto* d_wnorm   = sycl::malloc_device<float>(HC * N_EMBD, q);
    auto* d_mixed   = sycl::malloc_device<float>(N_EMBD, q);
    auto* d_qkv     = sycl::malloc_device<float>(QKV_DIM, q);
    auto* d_gdn_out = sycl::malloc_device<float>(N_EMBD, q);
    auto* d_state   = sycl::malloc_device<float>(16 * S_DIM * S_DIM, q);
    auto* d_ssm_out = sycl::malloc_device<float>(N_EMBD, q);
    auto* d_logits  = sycl::malloc_device<float>(ROUTER_EXPERTS, q);
    auto* d_ids     = sycl::malloc_device<int32_t>(K_EXP, q);
    auto* d_weights = sycl::malloc_device<float>(K_EXP, q);
    auto* d_shared  = sycl::malloc_device<float>(N_EMBD, q);
    auto* d_parts   = sycl::malloc_device<float>(K_EXP * N_EMBD, q);

    auto* d_w_qkv    = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(QKV_DIM * N_EMBD, q);
    auto* d_w_ssm    = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(N_EMBD * N_EMBD, q);
    auto* d_w_router = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(ROUTER_EXPERTS * N_EMBD, q);
    auto* d_w_shared = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(N_EMBD * N_EMBD, q);

    q.fill(d_R, 1.0f, HC * N_EMBD);
    q.fill(d_wnorm, 1.0f, HC * N_EMBD);
    q.fill(d_state, 0.05f, 16 * S_DIM * S_DIM);
    q.fill(d_parts, 0.1f, K_EXP * N_EMBD);
    q.wait();

    // Warmup
    for (int w = 0; w < 3; ++w) {
        q.fill(d_mixed, 0.0f, N_EMBD);
        q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> ls(sycl::range<1>(16), cgh);
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(HC * 256), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_gr_norm(item, d_R, d_wnorm, d_mixed, 1e-5f, ls.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(QKV_DIM * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_qkv_gemv(item, d_mixed, d_w_qkv, d_qkv);
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(16 * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_gdn_step(item, d_state, d_qkv, d_gdn_out);
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_ssm_out_gemv(item, d_gdn_out, d_w_ssm, d_ssm_out);
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(HC * N_EMBD), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) {
                                 k_gr_write(item, d_R, d_ssm_out, 0.25f);
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> ls(sycl::range<1>(16), cgh);
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(HC * 256), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_gr_norm(item, d_R, d_wnorm, d_mixed, 1e-5f, ls.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(ROUTER_EXPERTS * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_moe_router(item, d_mixed, d_w_router, d_logits, d_ids, d_weights);
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_moe_shared_gemv(item, d_mixed, d_w_shared, d_shared);
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) {
                                 k_moe_combine(item, d_shared, d_parts, d_weights, d_R);
                             });
        });
        q.wait();
    }

    for (auto _ : state) {
        q.fill(d_mixed, 0.0f, N_EMBD);

        auto ev1 = q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> ls(sycl::range<1>(16), cgh);
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(HC * 256), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_gr_norm(item, d_R, d_wnorm, d_mixed, 1e-5f, ls.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(QKV_DIM * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_qkv_gemv(item, d_mixed, d_w_qkv, d_qkv);
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(16 * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_gdn_step(item, d_state, d_qkv, d_gdn_out);
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_ssm_out_gemv(item, d_gdn_out, d_w_ssm, d_ssm_out);
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(HC * N_EMBD), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) {
                                 k_gr_write(item, d_R, d_ssm_out, 0.25f);
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> ls(sycl::range<1>(16), cgh);
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(HC * 256), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_gr_norm(item, d_R, d_wnorm, d_mixed, 1e-5f, ls.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(ROUTER_EXPERTS * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_moe_router(item, d_mixed, d_w_router, d_logits, d_ids, d_weights);
                             });
        });
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD * 16), sycl::range<1>(16)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                                 k_moe_shared_gemv(item, d_mixed, d_w_shared, d_shared);
                             });
        });
        auto ev_last = q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) {
                                 k_moe_combine(item, d_shared, d_parts, d_weights, d_R);
                             });
        });
        ev_last.wait();

        const auto t0 = ev1.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = ev_last.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double total_bytes = (double)(qkv_w_bytes + ssm_w_bytes + router_w_bytes + shared_w_bytes + r_bytes * 4);
    state.counters["Bandwidth_GBps"] = benchmark::Counter(total_bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["Weight_MB"]      = benchmark::Counter(total_bytes / 1e6);
    state.counters["Total_Kernels"]  = benchmark::Counter(9);

    sycl::free(d_R, q);
    sycl::free(d_wnorm, q);
    sycl::free(d_mixed, q);
    sycl::free(d_qkv, q);
    sycl::free(d_gdn_out, q);
    sycl::free(d_state, q);
    sycl::free(d_ssm_out, q);
    sycl::free(d_logits, q);
    sycl::free(d_ids, q);
    sycl::free(d_weights, q);
    sycl::free(d_shared, q);
    sycl::free(d_parts, q);
    sycl::free(d_w_qkv, q);
    sycl::free(d_w_ssm, q);
    sycl::free(d_w_router, q);
    sycl::free(d_w_shared, q);
}

// ============================================================================
// Benchmark: BM_Xe2_Composite_Layer_Graph (CapturedGraph Execution)
// ============================================================================

static void BM_Xe2_Composite_Layer_Graph(benchmark::State& state) {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};

    auto* d_R       = sycl::malloc_device<float>(HC * N_EMBD, q);
    auto* d_wnorm   = sycl::malloc_device<float>(HC * N_EMBD, q);
    auto* d_mixed   = sycl::malloc_device<float>(N_EMBD, q);
    auto* d_qkv     = sycl::malloc_device<float>(QKV_DIM, q);
    auto* d_gdn_out = sycl::malloc_device<float>(N_EMBD, q);
    auto* d_state   = sycl::malloc_device<float>(16 * S_DIM * S_DIM, q);
    auto* d_ssm_out = sycl::malloc_device<float>(N_EMBD, q);
    auto* d_logits  = sycl::malloc_device<float>(ROUTER_EXPERTS, q);
    auto* d_ids     = sycl::malloc_device<int32_t>(K_EXP, q);
    auto* d_weights = sycl::malloc_device<float>(K_EXP, q);
    auto* d_shared  = sycl::malloc_device<float>(N_EMBD, q);
    auto* d_parts   = sycl::malloc_device<float>(K_EXP * N_EMBD, q);

    auto* d_w_qkv    = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(QKV_DIM * N_EMBD, q);
    auto* d_w_ssm    = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(N_EMBD * N_EMBD, q);
    auto* d_w_router = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(ROUTER_EXPERTS * N_EMBD, q);
    auto* d_w_shared = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(N_EMBD * N_EMBD, q);

    q.fill(d_R, 1.0f, HC * N_EMBD);
    q.fill(d_wnorm, 1.0f, HC * N_EMBD);
    q.fill(d_state, 0.05f, 16 * S_DIM * S_DIM);
    q.fill(d_parts, 0.1f, K_EXP * N_EMBD);
    q.wait();

    strata::core::CapturedGraph g;
    std::string err;

    if (!g.begin(&q, err)) {
        state.SkipWithError(("Graph capture failed: " + err).c_str());
        return;
    }

    q.fill(d_mixed, 0.0f, N_EMBD);
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> ls(sycl::range<1>(16), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(HC * 256), sycl::range<1>(256)),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                             k_gr_norm(item, d_R, d_wnorm, d_mixed, 1e-5f, ls.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
    });
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(QKV_DIM * 16), sycl::range<1>(16)),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                             k_qkv_gemv(item, d_mixed, d_w_qkv, d_qkv);
                         });
    });
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(16 * 16), sycl::range<1>(16)),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                             k_gdn_step(item, d_state, d_qkv, d_gdn_out);
                         });
    });
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD * 16), sycl::range<1>(16)),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                             k_ssm_out_gemv(item, d_gdn_out, d_w_ssm, d_ssm_out);
                         });
    });
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(HC * N_EMBD), sycl::range<1>(256)),
                         [=](sycl::nd_item<1> item) {
                             k_gr_write(item, d_R, d_ssm_out, 0.25f);
                         });
    });
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> ls(sycl::range<1>(16), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(HC * 256), sycl::range<1>(256)),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                             k_gr_norm(item, d_R, d_wnorm, d_mixed, 1e-5f, ls.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
    });
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(ROUTER_EXPERTS * 16), sycl::range<1>(16)),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                             k_moe_router(item, d_mixed, d_w_router, d_logits, d_ids, d_weights);
                         });
    });
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD * 16), sycl::range<1>(16)),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                             k_moe_shared_gemv(item, d_mixed, d_w_shared, d_shared);
                         });
    });
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N_EMBD), sycl::range<1>(256)),
                         [=](sycl::nd_item<1> item) {
                             k_moe_combine(item, d_shared, d_parts, d_weights, d_R);
                         });
    });

    if (!g.end(&q, err)) {
        state.SkipWithError(("End capture failed: " + err).c_str());
        return;
    }

    // Warmup graph
    for (int w = 0; w < 3; ++w) {
        g.launch(&q, err);
        q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();
        g.launch(&q, err);
        q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = t1 - t0;
        state.SetIterationTime(diff.count());
    }

    state.counters["Total_Kernels"]  = benchmark::Counter(9);
    state.counters["Graph_Nodes"]    = benchmark::Counter(9);

    sycl::free(d_R, q);
    sycl::free(d_wnorm, q);
    sycl::free(d_mixed, q);
    sycl::free(d_qkv, q);
    sycl::free(d_gdn_out, q);
    sycl::free(d_state, q);
    sycl::free(d_ssm_out, q);
    sycl::free(d_logits, q);
    sycl::free(d_ids, q);
    sycl::free(d_weights, q);
    sycl::free(d_shared, q);
    sycl::free(d_parts, q);
    sycl::free(d_w_qkv, q);
    sycl::free(d_w_ssm, q);
    sycl::free(d_w_router, q);
    sycl::free(d_w_shared, q);
}

BENCHMARK(BM_Xe2_Composite_Layer_Pipelined)->UseManualTime();
BENCHMARK(BM_Xe2_Composite_Layer_Graph)->UseManualTime();

BENCHMARK_MAIN();
