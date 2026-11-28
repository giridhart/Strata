// engine/perf/sycl/perf_inkling2_kernels.cpp
// Google Benchmark suite for Inkling-2-Mini SYCL kernels on Intel Xe2 (Arc B580 / Lunar Lake).
// Reports effective GB/s and us/call for:
//   - MoE w13+SITU and w2 (FMT_SF4, FMT_MX8, FMT_S4)
//   - Dense decode GEMV (Q8_ROW dp4a, MX8, S4_K)
//   - Prefill GEMM (k_gemm_q8row_sg16)
// MIT License - Copyright (c) 2026 Strata contributors

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ink_sy_codecs.h"
#include "ink_sy_intelarc_xmx.h"

using namespace ink_sycl;

namespace {

constexpr int kD = 2304;
constexpr int kInter = 1152;
constexpr int kExperts = 128;
constexpr int kTopK = 8;

struct MoeFixture {
    int fmt = FMT_SF4;
    uint32_t dn_off = 0;
    size_t body_bytes = 0;
    size_t w13_bytes_per_expert = 0;
    size_t w2_bytes_per_expert = 0;
    std::vector<uint8_t*> d_pool;
    InkSyExpertDev* d_ex = nullptr;

    explicit MoeFixture(int f) : fmt(f) {
        if (fmt == FMT_SF4) {
            w13_bytes_per_expert = 16u + 2654208u + 331776u;
            w2_bytes_per_expert = 1327104u + 165888u;
            dn_off = (uint32_t)w13_bytes_per_expert;
            body_bytes = w13_bytes_per_expert + w2_bytes_per_expert;
        } else if (fmt == FMT_MX8) {
            w13_bytes_per_expert = 5308416u + 165888u;
            w2_bytes_per_expert = 2654208u + 82944u;
            dn_off = (uint32_t)w13_bytes_per_expert;
            body_bytes = w13_bytes_per_expert + w2_bytes_per_expert;
        } else {
            const int SB = sk_sb_bytes(fmt);
            w13_bytes_per_expert = 2304u + 2304ul * 9ul * (size_t)SB;
            w2_bytes_per_expert = 1152ul * 9ul * (size_t)SB;
            dn_off = (uint32_t)w13_bytes_per_expert;
            body_bytes = w13_bytes_per_expert + w2_bytes_per_expert;
        }
        // Use 16 distinct expert bodies so caches are realistically stressed without OOM
        constexpr int kPool = 16;
        std::vector<uint8_t> h_buf(body_bytes, 0x3Cu);
        if (fmt == FMT_SF4) {
            float* hdr = reinterpret_cast<float*>(h_buf.data());
            hdr[0] = 0.25f; hdr[1] = 1.0f; hdr[2] = 0.25f; hdr[3] = 1.0f;
        } else if (fmt != FMT_MX8) {
            sycl::half one(1.0f);
            uint16_t bits = 0;
            std::memcpy(&bits, &one, sizeof(bits));
            for (int r = 0; r < kInter; ++r) {
                h_buf[2 * r] = (uint8_t)(bits & 0xFF);
                h_buf[2 * r + 1] = (uint8_t)(bits >> 8);
            }
        }
        d_pool.resize(kPool);
        for (int i = 0; i < kPool; ++i) {
            d_pool[i] = sycl::malloc_device<uint8_t>(body_bytes, Q());
            Q().memcpy(d_pool[i], h_buf.data(), body_bytes);
        }
        std::vector<InkSyExpertDev> h_ex(kExperts);
        for (int e = 0; e < kExperts; ++e) {
            h_ex[e].body_ptr = d_pool[e % kPool];
            h_ex[e].dn_off = dn_off;
            h_ex[e].gu_code = (uint8_t)fmt;
            h_ex[e].dn_code = (uint8_t)fmt;
            h_ex[e].in_vram = 1;
            h_ex[e].pad = 0;
        }
        d_ex = sycl::malloc_device<InkSyExpertDev>(kExperts, Q());
        Q().memcpy(d_ex, h_ex.data(), kExperts * sizeof(InkSyExpertDev)).wait();
    }

    ~MoeFixture() {
        if (d_ex) sycl::free(d_ex, Q());
        for (auto* p : d_pool) {
            if (p) sycl::free(p, Q());
        }
    }
};

void bench_moe_kernel(benchmark::State& state, int fmt, bool is_w2) {
    const int M = (int)state.range(0);
    const int P = M * kTopK;
    const int max_uniq = std::min(P, kExperts);

    MoeFixture fix(fmt);

    std::vector<int> h_topk_idx(P);
    std::vector<float> h_topk_w(P, 0.125f);
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < kTopK; ++k) {
            h_topk_idx[m * kTopK + k] = (m * 8 + k) % kExperts;
        }
    }
    int n_uniq_actual = 0;
    {
        std::vector<int> seen(kExperts, 0);
        for (int idx : h_topk_idx) seen[idx] = 1;
        for (int s : seen) n_uniq_actual += s;
    }

    float* d_X = sycl::malloc_device<float>((size_t)M * kD, Q());
    int* d_topk_idx = sycl::malloc_device<int>(P, Q());
    float* d_topk_w = sycl::malloc_device<float>(P, Q());
    int* d_uexp = sycl::malloc_device<int>(kExperts, Q());
    int* d_uoff = sycl::malloc_device<int>(kExperts, Q());
    int* d_ucnt = sycl::malloc_device<int>(kExperts, Q());
    int* d_plist = sycl::malloc_device<int>(P, Q());
    int* d_nuniq = sycl::malloc_device<int>(1, Q());
    float* d_hact = sycl::malloc_device<float>((size_t)P * kInter, Q());
    float* d_ypairs = sycl::malloc_device<float>((size_t)P * kD, Q());

    Q().fill(d_X, 0.25f, (size_t)M * kD);
    Q().fill(d_hact, 0.25f, (size_t)P * kInter);
    Q().memcpy(d_topk_idx, h_topk_idx.data(), P * sizeof(int));
    Q().memcpy(d_topk_w, h_topk_w.data(), P * sizeof(float));
    Q().wait();

    k_moe_plan_sg16(d_topk_idx, P, d_uexp, d_uoff, d_ucnt, d_plist, d_nuniq);
    Q().wait_and_throw();

    auto run_once = [&]() {
        if (!is_w2) {
            k_moe_gu_situ_sg16(fix.d_ex, d_X, d_uexp, d_uoff, d_ucnt, d_plist, d_nuniq, max_uniq, d_hact);
        } else {
            k_moe_dn_sg16(fix.d_ex, d_topk_w, d_uexp, d_uoff, d_ucnt, d_plist, d_nuniq, max_uniq, d_hact, d_ypairs);
        }
    };

    // Warmup
    for (int w = 0; w < 3; ++w) run_once();
    Q().wait_and_throw();

    const size_t bytes_per_call = (size_t)n_uniq_actual * (is_w2 ? fix.w2_bytes_per_expert : fix.w13_bytes_per_expert);
    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        run_once();
        Q().wait();
        auto t1 = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }

    state.SetBytesProcessed((int64_t)state.iterations() * (int64_t)bytes_per_call);
    state.counters["GB/s"] = benchmark::Counter((double)state.iterations() * (double)bytes_per_call / 1e9,
                                                benchmark::Counter::kIsRate);

    sycl::free(d_X, Q());
    sycl::free(d_topk_idx, Q());
    sycl::free(d_topk_w, Q());
    sycl::free(d_uexp, Q());
    sycl::free(d_uoff, Q());
    sycl::free(d_ucnt, Q());
    sycl::free(d_plist, Q());
    sycl::free(d_nuniq, Q());
    sycl::free(d_hact, Q());
    sycl::free(d_ypairs, Q());
}

void BM_MoE_W13_SF4(benchmark::State& st) { bench_moe_kernel(st, FMT_SF4, false); }
void BM_MoE_W2_SF4(benchmark::State& st)  { bench_moe_kernel(st, FMT_SF4, true); }
void BM_MoE_W13_MX8(benchmark::State& st) { bench_moe_kernel(st, FMT_MX8, false); }
void BM_MoE_W2_MX8(benchmark::State& st)  { bench_moe_kernel(st, FMT_MX8, true); }
void BM_MoE_W13_S4(benchmark::State& st)  { bench_moe_kernel(st, FMT_S4, false); }
void BM_MoE_W2_S4(benchmark::State& st)   { bench_moe_kernel(st, FMT_S4, true); }

void BM_Dense_Q8Row_DP4A(benchmark::State& state) {
    const int N = (int)state.range(0);
    const int K = (int)state.range(1);
    const int M = (int)state.range(2);
    const size_t w_bytes = (size_t)N * 4 + (size_t)N * K;

    uint8_t* d_W = sycl::malloc_device<uint8_t>(w_bytes, Q());
    int8_t* d_xq = sycl::malloc_device<int8_t>((size_t)M * K, Q());
    float* d_xsc = sycl::malloc_device<float>(M, Q());
    float* d_Y = sycl::malloc_device<float>((size_t)M * N, Q());
    Q().memset(d_W, 0x11, w_bytes);
    Q().fill(d_xq, (int8_t)7, (size_t)M * K);
    Q().fill(d_xsc, 0.01f, M);
    Q().wait();

    for (int w = 0; w < 3; ++w) k_linear_q8row_dp4a_sg16(d_W, N, K, d_xq, d_xsc, d_Y, M);
    Q().wait_and_throw();

    const size_t bytes = w_bytes + (size_t)M * K + (size_t)M * N * sizeof(float);
    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        k_linear_q8row_dp4a_sg16(d_W, N, K, d_xq, d_xsc, d_Y, M);
        Q().wait();
        auto t1 = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }
    state.SetBytesProcessed((int64_t)state.iterations() * (int64_t)bytes);
    state.counters["GB/s"] = benchmark::Counter((double)state.iterations() * (double)bytes / 1e9,
                                                benchmark::Counter::kIsRate);

    sycl::free(d_W, Q());
    sycl::free(d_xq, Q());
    sycl::free(d_xsc, Q());
    sycl::free(d_Y, Q());
}

void BM_Dense_MX8(benchmark::State& state) {
    const int N = (int)state.range(0);
    const int K = (int)state.range(1);
    const int M = (int)state.range(2);
    const size_t w_bytes = (size_t)N * K + (size_t)N * (K >> 5);

    uint8_t* d_W = sycl::malloc_device<uint8_t>(w_bytes, Q());
    float* d_X = sycl::malloc_device<float>((size_t)M * K, Q());
    float* d_Y = sycl::malloc_device<float>((size_t)M * N, Q());
    Q().memset(d_W, 0x38, w_bytes);
    Q().fill(d_X, 0.25f, (size_t)M * K);
    Q().wait();

    for (int w = 0; w < 3; ++w) k_linear_mx8_sg16(d_W, N, K, d_X, d_Y, M);
    Q().wait_and_throw();

    const size_t bytes = w_bytes + (size_t)M * K * sizeof(float) + (size_t)M * N * sizeof(float);
    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        k_linear_mx8_sg16(d_W, N, K, d_X, d_Y, M);
        Q().wait();
        auto t1 = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }
    state.SetBytesProcessed((int64_t)state.iterations() * (int64_t)bytes);
    state.counters["GB/s"] = benchmark::Counter((double)state.iterations() * (double)bytes / 1e9,
                                                benchmark::Counter::kIsRate);

    sycl::free(d_W, Q());
    sycl::free(d_X, Q());
    sycl::free(d_Y, Q());
}

void BM_Dense_S4K(benchmark::State& state) {
    const int N = (int)state.range(0);
    const int K = (int)state.range(1);
    const int M = (int)state.range(2);
    constexpr int SB = sk_sb_bytes(FMT_S4);
    const size_t w_bytes = (size_t)N * (K >> 8) * SB;

    uint8_t* d_W = sycl::malloc_device<uint8_t>(w_bytes, Q());
    float* d_X = sycl::malloc_device<float>((size_t)M * K, Q());
    float* d_Y = sycl::malloc_device<float>((size_t)M * N, Q());
    Q().memset(d_W, 0x38, w_bytes);
    Q().fill(d_X, 0.25f, (size_t)M * K);
    Q().wait();

    for (int w = 0; w < 3; ++w) k_linear_sk_sg16<FMT_S4>(d_W, N, K, d_X, d_Y, M);
    Q().wait_and_throw();

    const size_t bytes = w_bytes + (size_t)M * K * sizeof(float) + (size_t)M * N * sizeof(float);
    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        k_linear_sk_sg16<FMT_S4>(d_W, N, K, d_X, d_Y, M);
        Q().wait();
        auto t1 = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }
    state.SetBytesProcessed((int64_t)state.iterations() * (int64_t)bytes);
    state.counters["GB/s"] = benchmark::Counter((double)state.iterations() * (double)bytes / 1e9,
                                                benchmark::Counter::kIsRate);

    sycl::free(d_W, Q());
    sycl::free(d_X, Q());
    sycl::free(d_Y, Q());
}

void BM_Prefill_Q8Row_GEMM(benchmark::State& state) {
    const int N = (int)state.range(0);
    const int K = (int)state.range(1);
    const int M = (int)state.range(2);
    const size_t w_bytes = (size_t)N * 4 + (size_t)N * K;

    uint8_t* d_W = sycl::malloc_device<uint8_t>(w_bytes, Q());
    float* d_X = sycl::malloc_device<float>((size_t)M * K, Q());
    float* d_Y = sycl::malloc_device<float>((size_t)M * N, Q());
    Q().memset(d_W, 0x11, w_bytes);
    Q().fill(d_X, 0.25f, (size_t)M * K);
    Q().wait();

    for (int w = 0; w < 3; ++w) k_gemm_q8row_sg16(d_W, N, K, d_X, d_Y, M);
    Q().wait_and_throw();

    const double flops = 2.0 * (double)M * (double)N * (double)K;
    const size_t bytes = w_bytes + (size_t)M * K * sizeof(float) + (size_t)M * N * sizeof(float);
    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        k_gemm_q8row_sg16(d_W, N, K, d_X, d_Y, M);
        Q().wait();
        auto t1 = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }
    state.SetBytesProcessed((int64_t)state.iterations() * (int64_t)bytes);
    state.counters["GB/s"] = benchmark::Counter((double)state.iterations() * (double)bytes / 1e9,
                                                benchmark::Counter::kIsRate);
    state.counters["TFLOPS"] = benchmark::Counter((double)state.iterations() * flops / 1e12,
                                                  benchmark::Counter::kIsRate);

    sycl::free(d_W, Q());
    sycl::free(d_X, Q());
    sycl::free(d_Y, Q());
}

void BM_Prefill_Q8Row_XMX_JM(benchmark::State& state) {
#if INK_SY_HAS_JOINT_MATRIX
    if (!Q().get_device().has(sycl::aspect::ext_intel_matrix)) {
        state.SkipWithMessage("Device lacks ext_intel_matrix");
        return;
    }
    const int N = (int)state.range(0);
    const int K = (int)state.range(1);
    const int M = (int)state.range(2);
    const size_t w_bytes = (size_t)N * 4 + (size_t)N * K;

    uint8_t* d_W = sycl::malloc_device<uint8_t>(w_bytes, Q());
    float* d_X = sycl::malloc_device<float>((size_t)M * K, Q());
    int8_t* d_xq = sycl::malloc_device<int8_t>((size_t)M * K, Q());
    float* d_xsc = sycl::malloc_device<float>(M, Q());
    float* d_Y = sycl::malloc_device<float>((size_t)M * N, Q());
    Q().memset(d_W, 0x11, w_bytes);
    Q().fill(d_X, 0.25f, (size_t)M * K);
    Q().wait();

    for (int w = 0; w < 3; ++w) {
        k_quantize_x_q8_sg16(d_X, M, K, d_xq, d_xsc);
        k_jm_gemm_q8row_s8_sg16(Q(), d_W, N, K, N, d_xq, d_xsc, d_Y, M);
    }
    Q().wait_and_throw();

    const double flops = 2.0 * (double)M * (double)N * (double)K;
    const size_t bytes = w_bytes + (size_t)M * K * sizeof(float) + (size_t)M * N * sizeof(float);
    for (auto _ : state) {
        auto t0 = std::chrono::steady_clock::now();
        k_quantize_x_q8_sg16(d_X, M, K, d_xq, d_xsc);
        k_jm_gemm_q8row_s8_sg16(Q(), d_W, N, K, N, d_xq, d_xsc, d_Y, M);
        Q().wait();
        auto t1 = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
    }
    state.SetBytesProcessed((int64_t)state.iterations() * (int64_t)bytes);
    state.counters["GB/s"] = benchmark::Counter((double)state.iterations() * (double)bytes / 1e9,
                                                benchmark::Counter::kIsRate);
    state.counters["TFLOPS"] = benchmark::Counter((double)state.iterations() * flops / 1e12,
                                                  benchmark::Counter::kIsRate);

    sycl::free(d_W, Q());
    sycl::free(d_X, Q());
    sycl::free(d_xq, Q());
    sycl::free(d_xsc, Q());
    sycl::free(d_Y, Q());
#else
    state.SkipWithMessage("INK_SY_HAS_JOINT_MATRIX=0");
#endif
}

BENCHMARK(BM_MoE_W13_SF4)->Arg(1)->Arg(4)->Arg(16)->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MoE_W2_SF4)->Arg(1)->Arg(4)->Arg(16)->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MoE_W13_MX8)->Arg(1)->Arg(4)->Arg(16)->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MoE_W2_MX8)->Arg(1)->Arg(4)->Arg(16)->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MoE_W13_S4)->Arg(1)->Arg(4)->Arg(16)->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MoE_W2_S4)->Arg(1)->Arg(4)->Arg(16)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Dense_Q8Row_DP4A)->Args({2304, 2304, 1})->Args({2304, 2304, 4})->Args({2304, 2304, 8})->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Dense_MX8)->Args({4096, 2304, 1})->Args({4096, 2304, 4})->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Dense_S4K)->Args({2304, 2304, 1})->Args({2304, 2304, 4})->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Prefill_Q8Row_GEMM)->Args({2304, 2304, 16})->Args({2304, 2304, 64})->Args({2304, 2304, 128})->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Prefill_Q8Row_XMX_JM)->Args({2304, 2304, 16})->Args({2304, 2304, 64})->Args({2304, 2304, 128})->UseManualTime()->Unit(benchmark::kMicrosecond);

}  // namespace

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    const auto dev = Q().get_device();
    benchmark::AddCustomContext("sycl_device", dev.get_info<sycl::info::device::name>());
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
