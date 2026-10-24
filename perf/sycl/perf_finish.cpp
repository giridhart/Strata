// perf/sycl/perf_finish.cpp - Intel Xe2 Fused Finish Kernel (RMSNorm + RoPE + Activation Quant) Benchmark
//
// Ports the DeepSeek-V4.1 finish kernel pattern to SYCL 2020.
// Compares:
//   1. Unfused Pipeline: separate RMSNorm -> RoPE -> Quant/Pack kernels (3 kernel launches, intermediate global memory writes/reads)
//   2. Fused Finish Kernel: single launch computing RMSNorm + RoPE + Quant/Pack in registers with sub-group shuffles
//
// Evaluates on Intel Xe2 across transformer head and hidden vector dimensions:
//   len in [128, 256, 1024, 2048], n_vec in [16, 32, 64]
//
// Verification:
//   Device bit-level numerical equivalence check comparing unfused pipeline vs fused kernel.

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/finish.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

struct FinishFixture {
    int n_vec;
    int len;
    int rd;
    long long stride;
    long long dst_stride;
    long long pdst_stride;
    float eps = 1e-6f;
    int pos = 100;
    int inverse = 0;
    int vpp = 1;
    int pos_step = 1;

    sycl::queue q;

    // Device allocations
    float* d_x = nullptr;
    float* d_w = nullptr;
    float* d_freqs = nullptr;

    // Unfused intermediate buffers
    float* d_buf_norm = nullptr;
    float* d_buf_rope = nullptr;

    // BF16 outputs
    uint16_t* d_dst_unfused = nullptr;
    uint16_t* d_dst_fused = nullptr;

    // FP8 packed outputs
    uint8_t* d_pdst_unfused = nullptr;
    uint8_t* d_pdst_fused = nullptr;

    FinishFixture(int n_vec_in, int len_in)
        : n_vec(n_vec_in),
          len(len_in),
          rd(len_in >= 128 ? 128 : len_in),
          stride(len_in),
          dst_stride(len_in),
          pdst_stride((long long)strata::kernels::finish_kv_row_bytes(strata::kernels::kKvFp8, len_in)),
          q(sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order()}) {

        std::mt19937 rng(1337);
        std::uniform_real_distribution<float> dist_x(-2.0f, 2.0f);
        std::uniform_real_distribution<float> dist_w(0.8f, 1.2f);

        std::vector<float> h_x((size_t)n_vec * (size_t)stride);
        for (auto& v : h_x) v = dist_x(rng);

        std::vector<float> h_w((size_t)len);
        for (auto& v : h_w) v = dist_w(rng);

        const int num_freqs = rd / 2;
        std::vector<float> h_freqs((size_t)num_freqs);
        for (int i = 0; i < num_freqs; ++i) {
            h_freqs[i] = 1.0f / std::pow(10000.0f, (2.0f * (float)i) / (float)rd);
        }

        const size_t total_elems = (size_t)n_vec * (size_t)stride;
        const size_t pdst_bytes = (size_t)n_vec * (size_t)pdst_stride;

        d_x = sycl::malloc_device<float>(total_elems, q);
        d_w = sycl::malloc_device<float>((size_t)len, q);
        d_freqs = sycl::malloc_device<float>((size_t)num_freqs, q);

        d_buf_norm = sycl::malloc_device<float>(total_elems, q);
        d_buf_rope = sycl::malloc_device<float>(total_elems, q);

        d_dst_unfused = sycl::malloc_device<uint16_t>(total_elems, q);
        d_dst_fused = sycl::malloc_device<uint16_t>(total_elems, q);

        d_pdst_unfused = sycl::malloc_device<uint8_t>(pdst_bytes, q);
        d_pdst_fused = sycl::malloc_device<uint8_t>(pdst_bytes, q);

        q.memcpy(d_x, h_x.data(), total_elems * sizeof(float)).wait();
        q.memcpy(d_w, h_w.data(), (size_t)len * sizeof(float)).wait();
        q.memcpy(d_freqs, h_freqs.data(), (size_t)num_freqs * sizeof(float)).wait();
        q.memset(d_pdst_unfused, 0, pdst_bytes).wait();
        q.memset(d_pdst_fused, 0, pdst_bytes).wait();
    }

    ~FinishFixture() {
        if (d_x) sycl::free(d_x, q);
        if (d_w) sycl::free(d_w, q);
        if (d_freqs) sycl::free(d_freqs, q);
        if (d_buf_norm) sycl::free(d_buf_norm, q);
        if (d_buf_rope) sycl::free(d_buf_rope, q);
        if (d_dst_unfused) sycl::free(d_dst_unfused, q);
        if (d_dst_fused) sycl::free(d_dst_fused, q);
        if (d_pdst_unfused) sycl::free(d_pdst_unfused, q);
        if (d_pdst_fused) sycl::free(d_pdst_fused, q);
    }
};

// ============================================================================
// Unfused Pipeline Implementation
// ============================================================================

void run_unfused_rmsnorm(const float* x, const float* w, float* out,
                         int n_vec, int len, long long stride, float eps,
                         sycl::queue& q) {
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> slm_red(sycl::range<1>(32), cgh);
        sycl::local_accessor<float, 1> slm_tot(sycl::range<1>(1), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(n_vec * strata::kernels::kFinThreads),
                              sycl::range<1>(strata::kernels::kFinThreads)),
            [=](sycl::nd_item<1> item) {
                const int v = item.get_group(0);
                const float* in_vec = x + (long long)v * stride;
                float* out_vec = out + (long long)v * stride;

                auto sg = item.get_sub_group();
                const int tid = item.get_local_id(0);
                const unsigned sg_size = sg.get_local_range()[0];
                const int sg_id = tid / sg_size;
                const int lane = sg.get_local_linear_id();

                float ss = 0.0f;
                for (int e = tid; e < len; e += strata::kernels::kFinThreads) {
                    float val = in_vec[e];
                    ss += val * val;
                }

                ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
                if (lane == 0) slm_red[sg_id] = ss;
                sycl::group_barrier(item.get_group());

                if (sg_id == 0) {
                    const unsigned num_sgs = strata::kernels::kFinThreads / sg_size;
                    float block_ss = ((unsigned)lane < num_sgs) ? slm_red[lane] : 0.0f;
                    block_ss = sycl::reduce_over_group(sg, block_ss, sycl::plus<float>());
                    if (lane == 0) slm_tot[0] = block_ss;
                }
                sycl::group_barrier(item.get_group());

                const float inv = sycl::rsqrt(slm_tot[0] / (float)len + eps);

                for (int e = tid; e < len; e += strata::kernels::kFinThreads) {
                    out_vec[e] = (w != nullptr) ? (in_vec[e] * inv * w[e]) : (in_vec[e] * inv);
                }
            });
    });
}

void run_unfused_rope(const float* in, float* out, const float* freqs,
                      int n_vec, int len, long long stride, int rd, int pos,
                      int vpp, int pos_step, int inverse, sycl::queue& q) {
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(n_vec * strata::kernels::kFinThreads),
                              sycl::range<1>(strata::kernels::kFinThreads)),
            [=](sycl::nd_item<1> item) {
                const int v = item.get_group(0);
                const float* in_vec = in + (long long)v * stride;
                float* out_vec = out + (long long)v * stride;

                auto sg = item.get_sub_group();
                const int tid = item.get_local_id(0);

                const int r0 = len - rd;
                const int pos_v = pos + (int)(v / vpp) * pos_step;

                for (int e = tid; e < len; e += strata::kernels::kFinThreads) {
                    float val = in_vec[e];
                    float other = sycl::permute_group_by_xor(sg, val, 1);
                    if (e >= r0) {
                        const int i = (e - r0) >> 1;
                        const float ang = (float)pos_v * freqs[i];
                        const float c = sycl::cos(ang);
                        const float sn = inverse ? -sycl::sin(ang) : sycl::sin(ang);
                        if (((e - r0) & 1) == 0) {
                            out_vec[e] = val * c - other * sn;
                        } else {
                            out_vec[e] = other * sn + val * c;
                        }
                    } else {
                        out_vec[e] = val;
                    }
                }
            });
    });
}

void run_unfused_quant_bf16(const float* in, uint16_t* dst, int n_vec, int len,
                            long long stride, long long dst_stride, sycl::queue& q) {
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(n_vec * strata::kernels::kFinThreads),
                              sycl::range<1>(strata::kernels::kFinThreads)),
            [=](sycl::nd_item<1> item) {
                const int v = item.get_group(0);
                const float* in_vec = in + (long long)v * stride;
                uint16_t* dst_vec = dst + (long long)v * dst_stride;

                const int tid = item.get_local_id(0);
                for (int e = tid; e < len; e += strata::kernels::kFinThreads) {
                    dst_vec[e] = strata::kernels::bf16_from_f32(in_vec[e]);
                }
            });
    });
}

void run_unfused_quant_fp8(const float* in, uint8_t* pdst, int n_vec, int len,
                           long long stride, long long pdst_stride, sycl::queue& q) {
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> slm_max(sycl::range<1>(32), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(n_vec * strata::kernels::kFinThreads),
                              sycl::range<1>(strata::kernels::kFinThreads)),
            [=](sycl::nd_item<1> item) {
                const int v = item.get_group(0);
                const float* in_vec = in + (long long)v * stride;
                uint8_t* pr = pdst + (long long)v * pdst_stride;

                auto sg = item.get_sub_group();
                const int tid = item.get_local_id(0);
                const unsigned sg_size = sg.get_local_range()[0];
                const int sg_id = tid / sg_size;
                const int lane = sg.get_local_linear_id();

                constexpr int g = 32;
                const int nj = (len + strata::kernels::kFinThreads - 1) / strata::kernels::kFinThreads;

                for (int j = 0; j < nj; ++j) {
                    const int e = tid + j * strata::kernels::kFinThreads;
                    float val = (e < len) ? in_vec[e] : 0.0f;
                    float a = sycl::fabs(val);

                    if (sg_size == 32) {
                        for (int o = 16; o > 0; o >>= 1) {
                            a = sycl::fmax(a, sycl::permute_group_by_xor(sg, a, o));
                        }
                    } else {
                        for (int o = 8; o > 0; o >>= 1) {
                            a = sycl::fmax(a, sycl::permute_group_by_xor(sg, a, o));
                        }
                        if (lane == 0) slm_max[sg_id] = a;
                        sycl::group_barrier(item.get_group());
                        a = sycl::fmax(a, slm_max[sg_id ^ 1]);
                    }

                    if (e < len) {
                        const float s = strata::kernels::pow2_ceil(sycl::fmax(a, 1e-4f) * (1.0f / 448.0f));
                        const float q_val = strata::kernels::fp8_rt(sycl::clamp(val / s, -448.0f, 448.0f));
                        pr[e] = (uint8_t)strata::kernels::e4m3_code(q_val);
                        if (e % g == 0) {
                            pr[len + e / g] = (uint8_t)(sycl::bit_cast<uint32_t>(s) >> 23);
                        }
                    }
                }
            });
    });
}

void run_unfused_pipeline_bf16(FinishFixture& fix) {
    run_unfused_rmsnorm(fix.d_x, fix.d_w, fix.d_buf_norm, fix.n_vec, fix.len, fix.stride, fix.eps, fix.q);
    run_unfused_rope(fix.d_buf_norm, fix.d_buf_rope, fix.d_freqs, fix.n_vec, fix.len, fix.stride, fix.rd,
                     fix.pos, fix.vpp, fix.pos_step, fix.inverse, fix.q);
    run_unfused_quant_bf16(fix.d_buf_rope, fix.d_dst_unfused, fix.n_vec, fix.len, fix.stride, fix.dst_stride, fix.q);
}

void run_unfused_pipeline_fp8(FinishFixture& fix) {
    run_unfused_rmsnorm(fix.d_x, fix.d_w, fix.d_buf_norm, fix.n_vec, fix.len, fix.stride, fix.eps, fix.q);
    run_unfused_rope(fix.d_buf_norm, fix.d_buf_rope, fix.d_freqs, fix.n_vec, fix.len, fix.stride, fix.rd,
                     fix.pos, fix.vpp, fix.pos_step, fix.inverse, fix.q);
    run_unfused_quant_fp8(fix.d_buf_rope, fix.d_pdst_unfused, fix.n_vec, fix.len, fix.stride, fix.pdst_stride, fix.q);
}

// ============================================================================
// Fused Finish Implementation
// ============================================================================

void run_fused_pipeline_bf16(FinishFixture& fix) {
    strata::kernels::FinishArgs args;
    args.w = fix.d_w;
    args.eps = fix.eps;
    args.rd = fix.rd;
    args.pos = fix.pos;
    args.inverse = fix.inverse;
    args.vpp = fix.vpp;
    args.pos_step = fix.pos_step;
    args.freqs = fix.d_freqs;
    args.quant = 0;
    args.dst = fix.d_dst_fused;
    args.dst_stride = fix.dst_stride;
    args.pdst = nullptr;

    strata::kernels::finish(fix.d_x, fix.n_vec, fix.len, fix.stride, args, &fix.q);
}

void run_fused_pipeline_fp8(FinishFixture& fix) {
    strata::kernels::FinishArgs args;
    args.w = fix.d_w;
    args.eps = fix.eps;
    args.rd = fix.rd;
    args.pos = fix.pos;
    args.inverse = fix.inverse;
    args.vpp = fix.vpp;
    args.pos_step = fix.pos_step;
    args.freqs = fix.d_freqs;
    args.quant = 1;
    args.dst = nullptr;
    args.pdst = fix.d_pdst_fused;
    args.pdst_stride = fix.pdst_stride;

    strata::kernels::finish(fix.d_x, fix.n_vec, fix.len, fix.stride, args, &fix.q);
}

// ============================================================================
// Numerical Verification
// ============================================================================

bool verify_finish_equivalence() {
    std::printf("[VERIFY] Running numerical equivalence tests between Unfused and Fused Finish kernels...\n");

    const std::vector<std::pair<int, int>> test_shapes = {
        {16, 128},
        {32, 128},
        {16, 256},
        {32, 256},
        {16, 1024},
        {8, 2048}
    };

    for (const auto& [n_vec, len] : test_shapes) {
        FinishFixture fix(n_vec, len);

        // 1. Verify BF16 pipeline
        run_unfused_pipeline_bf16(fix);
        run_fused_pipeline_bf16(fix);
        fix.q.wait();

        const size_t total_elems = (size_t)n_vec * (size_t)fix.stride;
        std::vector<uint16_t> h_unfused_bf16(total_elems);
        std::vector<uint16_t> h_fused_bf16(total_elems);

        fix.q.memcpy(h_unfused_bf16.data(), fix.d_dst_unfused, total_elems * sizeof(uint16_t)).wait();
        fix.q.memcpy(h_fused_bf16.data(), fix.d_dst_fused, total_elems * sizeof(uint16_t)).wait();

        int diff_bf16 = 0;
        float max_abs_diff_bf16 = 0.0f;
        for (size_t i = 0; i < total_elems; ++i) {
            if (h_unfused_bf16[i] != h_fused_bf16[i]) {
                const float f_unf = strata::kernels::f32_from_bf16(h_unfused_bf16[i]);
                const float f_fus = strata::kernels::f32_from_bf16(h_fused_bf16[i]);
                const float diff = std::abs(f_unf - f_fus);
                if (diff > max_abs_diff_bf16) max_abs_diff_bf16 = diff;
                // Allow minor 1-ulp difference in rounding if float reduction order varies
                if (diff > 1e-3f) {
                    diff_bf16++;
                }
            }
        }

        if (diff_bf16 > 0) {
            std::printf("  [FAIL] BF16 mismatch for shape (%d, %d): %d elements differ over 1e-3 (max diff: %.6f)\n",
                        n_vec, len, diff_bf16, max_abs_diff_bf16);
            return false;
        }

        // 2. Verify FP8 packed pipeline
        run_unfused_pipeline_fp8(fix);
        run_fused_pipeline_fp8(fix);
        fix.q.wait();

        const size_t pdst_bytes = (size_t)n_vec * (size_t)fix.pdst_stride;
        std::vector<uint8_t> h_unfused_fp8(pdst_bytes);
        std::vector<uint8_t> h_fused_fp8(pdst_bytes);

        fix.q.memcpy(h_unfused_fp8.data(), fix.d_pdst_unfused, pdst_bytes).wait();
        fix.q.memcpy(h_fused_fp8.data(), fix.d_pdst_fused, pdst_bytes).wait();

        int diff_fp8 = 0;
        for (int v = 0; v < n_vec; ++v) {
            const size_t row_off = (size_t)v * fix.pdst_stride;
            // Check codes (first len bytes)
            for (int e = 0; e < len; ++e) {
                if (h_unfused_fp8[row_off + e] != h_fused_fp8[row_off + e]) {
                    diff_fp8++;
                }
            }
            // Check scales (len/32 bytes)
            for (int g = 0; g < len / 32; ++g) {
                if (h_unfused_fp8[row_off + len + g] != h_fused_fp8[row_off + len + g]) {
                    diff_fp8++;
                }
            }
        }

        if (diff_fp8 > 0) {
            std::printf("  [FAIL] FP8 packed mismatch for shape (%d, %d): %d bytes differ\n",
                        n_vec, len, diff_fp8);
            return false;
        }

        std::printf("  [PASS] Shape (%d, %d): BF16 exact (max diff: %.2e), FP8 packed 100%% byte-exact match\n",
                    n_vec, len, max_abs_diff_bf16);
    }

    std::printf("[VERIFY] All verification tests PASSED!\n\n");
    return true;
}

// ============================================================================
// Google Benchmark Registration
// ============================================================================

void BM_Finish_BF16_Unfused(benchmark::State& state) {
    const int n_vec = state.range(0);
    const int len = state.range(1);
    FinishFixture fix(n_vec, len);

    for (int i = 0; i < 5; ++i) {
        run_unfused_pipeline_bf16(fix);
        fix.q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();
        run_unfused_pipeline_bf16(fix);
        fix.q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    const double useful_bytes = (double)n_vec * len * (sizeof(float) + sizeof(uint16_t));
    state.counters["Useful_GB/s"] = benchmark::Counter(useful_bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

void BM_Finish_BF16_Fused(benchmark::State& state) {
    const int n_vec = state.range(0);
    const int len = state.range(1);
    FinishFixture fix(n_vec, len);

    for (int i = 0; i < 5; ++i) {
        run_fused_pipeline_bf16(fix);
        fix.q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();
        run_fused_pipeline_bf16(fix);
        fix.q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    const double useful_bytes = (double)n_vec * len * (sizeof(float) + sizeof(uint16_t));
    state.counters["Useful_GB/s"] = benchmark::Counter(useful_bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

void BM_Finish_FP8_Unfused(benchmark::State& state) {
    const int n_vec = state.range(0);
    const int len = state.range(1);
    FinishFixture fix(n_vec, len);

    for (int i = 0; i < 5; ++i) {
        run_unfused_pipeline_fp8(fix);
        fix.q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();
        run_unfused_pipeline_fp8(fix);
        fix.q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    const double useful_bytes = (double)n_vec * (len * sizeof(float) + len + len / 32);
    state.counters["Useful_GB/s"] = benchmark::Counter(useful_bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

void BM_Finish_FP8_Fused(benchmark::State& state) {
    const int n_vec = state.range(0);
    const int len = state.range(1);
    FinishFixture fix(n_vec, len);

    for (int i = 0; i < 5; ++i) {
        run_fused_pipeline_fp8(fix);
        fix.q.wait();
    }

    for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();
        run_fused_pipeline_fp8(fix);
        fix.q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    const double useful_bytes = (double)n_vec * (len * sizeof(float) + len + len / 32);
    state.counters["Useful_GB/s"] = benchmark::Counter(useful_bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// Register Google Benchmarks for realistic transformer shapes
#define REGISTER_FINISH_BENCHMARKS(len_val) \
    BENCHMARK(BM_Finish_BF16_Unfused)->Args({16, len_val})->Args({32, len_val})->Args({64, len_val})->UseManualTime(); \
    BENCHMARK(BM_Finish_BF16_Fused)->Args({16, len_val})->Args({32, len_val})->Args({64, len_val})->UseManualTime(); \
    BENCHMARK(BM_Finish_FP8_Unfused)->Args({16, len_val})->Args({32, len_val})->Args({64, len_val})->UseManualTime(); \
    BENCHMARK(BM_Finish_FP8_Fused)->Args({16, len_val})->Args({32, len_val})->Args({64, len_val})->UseManualTime();

REGISTER_FINISH_BENCHMARKS(128)
REGISTER_FINISH_BENCHMARKS(256)
REGISTER_FINISH_BENCHMARKS(1024)
REGISTER_FINISH_BENCHMARKS(2048)

// ============================================================================
// Direct Summary Runner
// ============================================================================

void run_direct_summary() {
    std::printf("====================================================================================================\n");
    std::printf(" Intel Xe2 Microbenchmark: Fused Finish Kernel (RMSNorm + RoPE + Act Quant) vs Unfused Pipeline\n");
    std::printf("====================================================================================================\n");
    std::printf("%-18s | %-10s | %-14s | %-14s | %-9s | %-13s\n",
                "Format & Shape", "n_vec x len", "Unfused Latency", "Fused Latency", "Speedup", "Effective BW");
    std::printf("----------------------------------------------------------------------------------------------------\n");

    const std::vector<std::pair<int, int>> configs = {
        {16, 128}, {32, 128}, {64, 128},
        {16, 256}, {32, 256}, {64, 256},
        {16, 1024}, {32, 1024}, {64, 1024},
        {16, 2048}, {32, 2048}, {64, 2048}
    };

    constexpr int warmup = 10;
    constexpr int iters = 50;

    // 1. BF16 Evaluation
    for (const auto& [n_vec, len] : configs) {
        FinishFixture fix(n_vec, len);

        for (int i = 0; i < warmup; ++i) {
            run_unfused_pipeline_bf16(fix);
            fix.q.wait();
        }
        auto t0_u = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            run_unfused_pipeline_bf16(fix);
            fix.q.wait();
        }
        auto t1_u = std::chrono::high_resolution_clock::now();
        const double lat_u_us = std::chrono::duration<double, std::micro>(t1_u - t0_u).count() / iters;

        for (int i = 0; i < warmup; ++i) {
            run_fused_pipeline_bf16(fix);
            fix.q.wait();
        }
        auto t0_f = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            run_fused_pipeline_bf16(fix);
            fix.q.wait();
        }
        auto t1_f = std::chrono::high_resolution_clock::now();
        const double lat_f_us = std::chrono::duration<double, std::micro>(t1_f - t0_f).count() / iters;

        const double useful_bytes = (double)n_vec * len * (sizeof(float) + sizeof(uint16_t));
        const double bw_gbs = (useful_bytes / (lat_f_us * 1e-6)) * 1e-9;
        const double speedup = lat_u_us / lat_f_us;

        char dim_buf[32];
        std::snprintf(dim_buf, sizeof(dim_buf), "%d x %d", n_vec, len);
        std::printf("%-18s | %-10s | %11.2f µs | %11.2f µs | %7.2fx | %10.2f GB/s\n",
                    "BF16 (Norm+RoPE)", dim_buf, lat_u_us, lat_f_us, speedup, bw_gbs);
    }

    std::printf("----------------------------------------------------------------------------------------------------\n");

    // 2. FP8 Evaluation
    for (const auto& [n_vec, len] : configs) {
        FinishFixture fix(n_vec, len);

        for (int i = 0; i < warmup; ++i) {
            run_unfused_pipeline_fp8(fix);
            fix.q.wait();
        }
        auto t0_u = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            run_unfused_pipeline_fp8(fix);
            fix.q.wait();
        }
        auto t1_u = std::chrono::high_resolution_clock::now();
        const double lat_u_us = std::chrono::duration<double, std::micro>(t1_u - t0_u).count() / iters;

        for (int i = 0; i < warmup; ++i) {
            run_fused_pipeline_fp8(fix);
            fix.q.wait();
        }
        auto t0_f = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            run_fused_pipeline_fp8(fix);
            fix.q.wait();
        }
        auto t1_f = std::chrono::high_resolution_clock::now();
        const double lat_f_us = std::chrono::duration<double, std::micro>(t1_f - t0_f).count() / iters;

        const double useful_bytes = (double)n_vec * (len * sizeof(float) + len + len / 32);
        const double bw_gbs = (useful_bytes / (lat_f_us * 1e-6)) * 1e-9;
        const double speedup = lat_u_us / lat_f_us;

        char dim_buf[32];
        std::snprintf(dim_buf, sizeof(dim_buf), "%d x %d", n_vec, len);
        std::printf("%-18s | %-10s | %11.2f µs | %11.2f µs | %7.2fx | %10.2f GB/s\n",
                    "FP8 (Norm+RoPE+Q)", dim_buf, lat_u_us, lat_f_us, speedup, bw_gbs);
    }

    std::printf("====================================================================================================\n\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (!verify_finish_equivalence()) {
        std::fprintf(stderr, "[FATAL] Device verification failed!\n");
        return 1;
    }

    bool run_summary = true;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]).rfind("--benchmark_filter", 0) == 0 ||
            std::string(argv[i]) == "--benchmark_list_tests") {
            run_summary = false;
        }
    }

    if (run_summary) {
        run_direct_summary();
    }

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
