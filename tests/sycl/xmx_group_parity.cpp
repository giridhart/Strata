// tests/sycl/xmx_group_parity.cpp - Numerical parity and benchmark verification
// for iq_xmx_grouped (fused weight decode + grouped XMX joint_matrix DPAS).
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix-unified.hpp>
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static uint16_t f2h(float f) {
    return sycl::bit_cast<uint16_t>(sycl::half(f));
}

// Reference GPU GEMM: Y[M x N] = X[M x K] * W[N x K]^T
static void ref_gemm(sycl::queue& q, const uint16_t* X, const uint16_t* W, float* Y,
                     int64_t M, int64_t N, int64_t K) {
    const size_t local_m = 16, local_n = 16;
    const size_t global_m = ((size_t)M + local_m - 1) / local_m * local_m;
    const size_t global_n = ((size_t)N + local_n - 1) / local_n * local_n;

    q.parallel_for(sycl::nd_range<2>({global_m, global_n}, {local_m, local_n}),
                   [=](sycl::nd_item<2> it) {
        const int64_t m = (int64_t)it.get_global_id(0);
        const int64_t n = (int64_t)it.get_global_id(1);
        if (m >= M || n >= N) return;
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            float fx = static_cast<float>(sycl::bit_cast<sycl::half>(X[(size_t)m * K + k]));
            float fw = static_cast<float>(sycl::bit_cast<sycl::half>(W[(size_t)n * K + k]));
            sum += fx * fw;
        }
        Y[(size_t)m * N + n] = sum;
    });
}

int main(int argc, char** argv) {
    const int mean_rows = argc > 1 ? std::atoi(argv[1]) : 128;
    const int n = argc > 2 ? std::atoi(argv[2]) : 8;

    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    sycl::device dev = q.get_device();
    std::printf("Running xmx_group_parity on: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    bool has_xmx = false;
    try {
        auto combos = dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
        if (!combos.empty()) has_xmx = true;
    } catch (...) {}

    if (!has_xmx) {
        std::printf("  Device does not report hardware matrix support; skipping.\n");
        return 0;
    }

    std::mt19937 rng(42);
    std::vector<int32_t> cnt(n);
    std::gamma_distribution<double> gd(1.5, mean_rows / 1.5);
    int64_t tot = 0;
    for (auto& c : cnt) {
        c = std::max(1, (int)gd(rng));
        tot += c;
    }

    int fails = 0;
    struct Case {
        int ty;
        bool gu;
        const char* name;
    };

    // Test the essential formats used by DeepSeek and Qwen native packs
    const Case cases[] = {
        {21, true,  "IQ3_S  gate/up"},
        {20, false, "IQ4_NL down"},
        {42, false, "Q2_0   down"},
        {18, true,  "IQ3_XXS gate/up"},
        {22, true,  "IQ2_S  gate/up"}
    };

    for (const auto& cs : cases) {
        const int64_t K = cs.gu ? 2560 : 640;
        const int64_t N = cs.gu ? 1280 : 2560;
        const int64_t n_ff = 640;
        const int blk_elems = (cs.ty == 20) ? 32 : (cs.ty == 42 ? 64 : 256);
        const size_t blk_bytes = strata::kernels::iq_row_bytes(cs.ty, blk_elems);
        const size_t mat_bytes = strata::kernels::iq_row_bytes(cs.ty, K) * (cs.gu ? n_ff : N);

        if (blk_bytes == 0 || mat_bytes == 0) {
            std::printf("  Type %d (%s): not supported by iq_row_bytes\n", cs.ty, cs.name);
            continue;
        }

        auto make_blob = [&]() {
            std::vector<uint8_t> b(mat_bytes);
            for (auto& x : b) x = (uint8_t)rng();
            // Set valid scale header for blocks
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
        uint16_t* dw = sycl::malloc_device<uint16_t>((size_t)N * K, q);
        float* yr = sycl::malloc_device<float>((size_t)tot * N, q);
        float* yx = sycl::malloc_device<float>((size_t)tot * N, q);
        q.wait();

        auto run_ref = [&]() {
            int64_t o = 0;
            for (int i = 0; i < n; ++i) {
                if (cs.gu) {
                    strata::kernels::iq_dequant_gu_f16(cs.ty, dg[i], du[i], n_ff, K, dw, &q);
                } else {
                    strata::kernels::iq_dequant_f16(cs.ty, dg[i], N * K, dw, &q);
                }
                ref_gemm(q, dx + o * K, dw, yr + o * N, cnt[i], N, K);
                o += cnt[i];
            }
        };

        auto run_xmx = [&]() {
            strata::kernels::iq_xmx_grouped(cs.ty, pg.data(), cs.gu ? pu.data() : nullptr,
                                            cnt.data(), n, dx, yx, (int)N, (int)K, &q);
        };

        // Warm up and verify correctness
        run_ref();
        run_xmx();
        q.wait();

        std::vector<float> hr((size_t)tot * N), hx_out((size_t)tot * N);
        q.memcpy(hr.data(), yr, hr.size() * sizeof(float));
        q.memcpy(hx_out.data(), yx, hx_out.size() * sizeof(float)).wait();

        double max_err = 0.0, max_val = 0.0;
        size_t bad_ref = 0, bad_xmx = 0;
        for (size_t i = 0; i < hr.size(); ++i) {
            if (!std::isfinite(hr[i])) {
                ++bad_ref;
                continue;
            }
            if (!std::isfinite(hx_out[i])) {
                ++bad_xmx;
                continue;
            }
            const double diff = std::fabs((double)hr[i] - (double)hx_out[i]);
            max_err = std::max(max_err, diff);
            max_val = std::max(max_val, (double)std::fabs(hr[i]));
        }

        const double rel_err = max_err / (max_val > 0.0 ? max_val : 1.0);
        const bool ok = (bad_xmx == 0) && (max_val > 0.0) && (rel_err <= 3e-3);
        if (!ok) {
            std::printf("      DEBUG: bad_ref=%zu, bad_xmx=%zu, max_val=%e, max_err=%e, rel_err=%e\n",
                        bad_ref, bad_xmx, max_val, max_err, rel_err);
            ++fails;
        }

        // Benchmark timings
        auto time_fn = [&](auto fn) {
            fn();
            q.wait();
            const auto t0 = std::chrono::steady_clock::now();
            constexpr int iters = 10;
            for (int it = 0; it < iters; ++it) fn();
            q.wait();
            const auto t1 = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        };

        const double t_ref = time_fn(run_ref);
        const double t_xmx = time_fn(run_xmx);

        std::printf("  Type %2d (%-15s) rows=%5lld: dequant+GEMM %7.2f ms | fused XMX %6.2f ms (%.2fx speedup)  relerr=%.2e  [%s]\n",
                    cs.ty, cs.name, (long long)tot, t_ref, t_xmx, t_ref / t_xmx, rel_err, ok ? "OK" : "FAIL");

        for (int i = 0; i < n; ++i) {
            sycl::free(dg[i], q);
            if (cs.gu) sycl::free(du[i], q);
        }
        sycl::free(dx, q);
        sycl::free(dw, q);
        sycl::free(yr, q);
        sycl::free(yx, q);
    }

    std::printf("\nxmx_group_parity summary: %d failures\n", fails);
    return fails == 0 ? 0 : 1;
}
