// tests/sycl/prefill_actions_perf.cpp - Performance benchmarking and empirical comparison
// for Action 2 (Adaptive MoE Tile Sizing Micro BM=32 vs Small BM=64) and
// Action 3 (Vectorized 128-bit moe_combine4 vs 32-bit scalar moe_combine).
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix-unified.hpp>
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/prefill/kernels.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static uint16_t f2h(float f) {
    return sycl::bit_cast<uint16_t>(sycl::half(f));
}

// ----------------------------------------------------------------
// Action 2 Benchmark: Adaptive Tile Sizing (BM=32 vs BM=64)
// ----------------------------------------------------------------
static void benchmark_action2(sycl::queue& q) {
    std::printf("\n========================================================================================\n");
    std::printf(" ACTION 2 BENCHMARK: Adaptive MoE Tile Sizing (Micro BM=32 vs Small BM=64)\n");
    std::printf(" Target Hardware: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("========================================================================================\n");
    std::printf(" %-10s | %-15s | %-10s | %-12s | %-12s | %-10s\n",
                "Mean Rows", "Format", "Pass", "Small (BM=64)", "Micro (BM=32)", "Speedup");
    std::printf("----------------------------------------------------------------------------------------\n");

    const int mean_row_tests[] = {8, 12, 16, 20, 24, 32, 43};
    const int n = 8;
    std::mt19937 rng(42);

    struct Case {
        int ty;
        bool gu;
        const char* name;
    };
    const Case cases[] = {
        {21, true,  "IQ3_S"},
        {20, false, "IQ4_NL"},
        {42, false, "Q2_0"}
    };

    for (int mean_rows : mean_row_tests) {
        std::vector<int32_t> cnt(n);
        std::gamma_distribution<double> gd(1.5, mean_rows / 1.5);
        int64_t tot = 0;
        for (auto& c : cnt) {
            c = std::max(1, (int)gd(rng));
            tot += c;
        }

        for (const auto& cs : cases) {
            const int64_t K = cs.gu ? 2560 : 640;
            const int64_t N = cs.gu ? 1280 : 2560;
            const int64_t n_ff = 640;
            const int blk_elems = (cs.ty == 20) ? 32 : (cs.ty == 42 ? 64 : 256);
            const size_t blk_bytes = strata::kernels::iq_row_bytes(cs.ty, blk_elems);
            const size_t mat_bytes = strata::kernels::iq_row_bytes(cs.ty, K) * (cs.gu ? n_ff : N);

            if (blk_bytes == 0 || mat_bytes == 0) continue;

            auto make_blob = [&]() {
                std::vector<uint8_t> b(mat_bytes);
                for (auto& x : b) x = (uint8_t)rng();
                if (cs.ty != 29) {
                    for (size_t o = 0; o + 1 < b.size(); o += blk_bytes) {
                        b[o] = 0x00;
                        b[o + 1] = 0x2C;
                    }
                }
                return b;
            };

            std::vector<uint8_t*> dg(n, nullptr), du(n, nullptr);
            std::vector<const uint8_t*> pg(n, nullptr), pu(n, nullptr);
            for (int i = 0; i < n; ++i) {
                auto g = make_blob();
                dg[i] = sycl::malloc_device<uint8_t>(mat_bytes, q);
                q.memcpy(dg[i], g.data(), mat_bytes);
                pg[i] = dg[i];
                if (cs.gu) {
                    auto u = make_blob();
                    du[i] = sycl::malloc_device<uint8_t>(mat_bytes, q);
                    q.memcpy(du[i], u.data(), mat_bytes);
                    pu[i] = du[i];
                }
            }

            std::vector<uint16_t> hx((size_t)tot * K);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            for (auto& x : hx) x = f2h(nd(rng) * 0.25f);

            uint16_t* dx = sycl::malloc_device<uint16_t>(hx.size(), q);
            q.memcpy(dx, hx.data(), hx.size() * sizeof(uint16_t));
            float* yx = sycl::malloc_device<float>((size_t)tot * N, q);
            q.wait();

            // Time with Small (BM=64)
            setenv("STRATA_XMXG_TILE", "small", 1);
            for (int w = 0; w < 5; ++w) {
                strata::kernels::iq_xmx_grouped(cs.ty, pg.data(), cs.gu ? pu.data() : nullptr,
                                                cnt.data(), n, dx, yx, (int)N, (int)K, &q);
            }
            q.wait();
            const auto t0_small = std::chrono::steady_clock::now();
            constexpr int iters = 25;
            for (int it = 0; it < iters; ++it) {
                strata::kernels::iq_xmx_grouped(cs.ty, pg.data(), cs.gu ? pu.data() : nullptr,
                                                cnt.data(), n, dx, yx, (int)N, (int)K, &q);
            }
            q.wait();
            const auto t1_small = std::chrono::steady_clock::now();
            const double ms_small = std::chrono::duration<double, std::milli>(t1_small - t0_small).count() / iters;

            // Time with Micro (BM=32)
            setenv("STRATA_XMXG_TILE", "micro", 1);
            for (int w = 0; w < 5; ++w) {
                strata::kernels::iq_xmx_grouped(cs.ty, pg.data(), cs.gu ? pu.data() : nullptr,
                                                cnt.data(), n, dx, yx, (int)N, (int)K, &q);
            }
            q.wait();
            const auto t0_micro = std::chrono::steady_clock::now();
            for (int it = 0; it < iters; ++it) {
                strata::kernels::iq_xmx_grouped(cs.ty, pg.data(), cs.gu ? pu.data() : nullptr,
                                                cnt.data(), n, dx, yx, (int)N, (int)K, &q);
            }
            q.wait();
            const auto t1_micro = std::chrono::steady_clock::now();
            const double ms_micro = std::chrono::duration<double, std::milli>(t1_micro - t0_micro).count() / iters;
            unsetenv("STRATA_XMXG_TILE");

            const double speedup = ms_small / ms_micro;
            char mean_str[32];
            std::snprintf(mean_str, sizeof(mean_str), "%d (tot %lld)", mean_rows, (long long)tot);
            std::printf(" %-10s | %-15s | %-10s | %9.3f ms | %9.3f ms | %6.2fx (%+.1f%%)\n",
                        mean_str, cs.name, cs.gu ? "gate/up" : "down",
                        ms_small, ms_micro, speedup, (speedup - 1.0) * 100.0);

            for (int i = 0; i < n; ++i) {
                sycl::free(dg[i], q);
                if (cs.gu) sycl::free(du[i], q);
            }
            sycl::free(dx, q);
            sycl::free(yx, q);
        }
    }
}

// ----------------------------------------------------------------
// Action 3 Benchmark: moe_combine4 (Vectorized 128-bit vs Scalar)
// ----------------------------------------------------------------
static void benchmark_action3(sycl::queue& q) {
    constexpr int N = 2560;
    std::printf("\n========================================================================================\n");
    std::printf(" ACTION 3 BENCHMARK: 128-bit Vectorized moe_combine4 vs 32-bit Scalar moe_combine\n");
    std::printf(" Target Hardware: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("========================================================================================\n");
    std::printf(" %-8s | %-10s | %-15s | %-15s | %-10s | %-14s\n",
                "Tokens", "Scale", "Scalar Combine", "Vector Combine4", "Speedup", "Bandwidth");
    std::printf("----------------------------------------------------------------------------------------\n");

    const int token_tests[] = {512, 2201, 4096};
    const int rows = 4000;
    std::mt19937 g(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int32_t> pick_row(0, rows - 1);

    std::vector<float> Dm((size_t)rows * N);
    for (auto& x : Dm) x = nd(g);
    std::vector<float> row_sd(rows);
    for (auto& x : row_sd) x = std::fabs(nd(g)) + 0.1f;

    float* d_Dm = sycl::malloc_device<float>(Dm.size(), q);
    float* d_row_sd = sycl::malloc_device<float>(row_sd.size(), q);
    q.memcpy(d_Dm, Dm.data(), Dm.size() * sizeof(float));
    q.memcpy(d_row_sd, row_sd.data(), row_sd.size() * sizeof(float));
    q.wait();

    for (int tokens : token_tests) {
        std::vector<int32_t> slot((size_t)tokens * 10);
        for (auto& s : slot) s = pick_row(g);
        std::vector<float> w((size_t)tokens * 10);
        for (auto& x : w) x = std::fabs(nd(g)) * 0.1f;
        std::vector<float> shared((size_t)tokens * N), sg(tokens);
        for (auto& x : shared) x = nd(g);
        for (auto& x : sg) x = nd(g) * 2.0f;

        int32_t* d_slot = sycl::malloc_device<int32_t>(slot.size(), q);
        float* d_w = sycl::malloc_device<float>(w.size(), q);
        float* d_shared = sycl::malloc_device<float>(shared.size(), q);
        float* d_sg = sycl::malloc_device<float>(sg.size(), q);
        float* d_bo = sycl::malloc_device<float>((size_t)tokens * N, q);

        q.memcpy(d_slot, slot.data(), slot.size() * sizeof(int32_t));
        q.memcpy(d_w, w.data(), w.size() * sizeof(float));
        q.memcpy(d_shared, shared.data(), shared.size() * sizeof(float));
        q.memcpy(d_sg, sg.data(), sg.size() * sizeof(float));
        q.wait();

        for (bool with_sd : {false, true}) {
            // Measure scalar combine
            setenv("STRATA_NO_COMB4", "1", 1);
            for (int w = 0; w < 5; ++w) {
                strata::prefill::moe_combine(d_Dm, d_slot, d_w, d_shared, d_sg, d_bo, tokens, &q, with_sd ? d_row_sd : nullptr);
            }
            q.wait();
            const auto t0_scalar = std::chrono::steady_clock::now();
            constexpr int iters = 50;
            for (int it = 0; it < iters; ++it) {
                strata::prefill::moe_combine(d_Dm, d_slot, d_w, d_shared, d_sg, d_bo, tokens, &q, with_sd ? d_row_sd : nullptr);
            }
            q.wait();
            const auto t1_scalar = std::chrono::steady_clock::now();
            const double ms_scalar = std::chrono::duration<double, std::milli>(t1_scalar - t0_scalar).count() / iters;

            // Measure vectorized combine4
            unsetenv("STRATA_NO_COMB4");
            for (int w = 0; w < 5; ++w) {
                strata::prefill::moe_combine(d_Dm, d_slot, d_w, d_shared, d_sg, d_bo, tokens, &q, with_sd ? d_row_sd : nullptr);
            }
            q.wait();
            const auto t0_vec = std::chrono::steady_clock::now();
            for (int it = 0; it < iters; ++it) {
                strata::prefill::moe_combine(d_Dm, d_slot, d_w, d_shared, d_sg, d_bo, tokens, &q, with_sd ? d_row_sd : nullptr);
            }
            q.wait();
            const auto t1_vec = std::chrono::steady_clock::now();
            const double ms_vec = std::chrono::duration<double, std::milli>(t1_vec - t0_vec).count() / iters;

            const double speedup = ms_scalar / ms_vec;
            // Read 10 rows of Dm (10 * N * 4 bytes) + 1 shared row (N * 4 bytes) + write 1 bo row (N * 4 bytes) per token
            const double bytes_transferred = (double)tokens * (12.0 * N * sizeof(float));
            const double bw_gbs = (bytes_transferred / (ms_vec * 1e-3)) / 1e9;

            std::printf(" %-8d | %-10s | %10.3f ms   | %10.3f ms   | %6.2fx (%+.1f%%) | %7.1f GB/s\n",
                        tokens, with_sd ? "NVFP4 (sd)" : "None",
                        ms_scalar, ms_vec, speedup, (speedup - 1.0) * 100.0, bw_gbs);
        }

        sycl::free(d_slot, q);
        sycl::free(d_w, q);
        sycl::free(d_shared, q);
        sycl::free(d_sg, q);
        sycl::free(d_bo, q);
    }

    sycl::free(d_Dm, q);
    sycl::free(d_row_sd, q);
}

int main() {
    sycl::device dev;
    try {
        dev = sycl::device(sycl::default_selector_v);
    } catch (...) {
        std::printf("SKIP: no SYCL device\n");
        return 77;
    }
    if (!dev.is_gpu()) {
        std::printf("SKIP: default SYCL device is not GPU\n");
        return 77;
    }
    sycl::queue q{dev, sycl::property::queue::in_order()};

    benchmark_action2(q);
    benchmark_action3(q);
    std::printf("\n[SUCCESS] Performance benchmarks completed successfully.\n");
    return 0;
}
