// src/kernels/sycl/fused_gr_bench.cpp - pure SYCL 2020 benchmark of fused_gr_read_multi across T = 1..6.
#include "strata/kernels/fused_gr.hpp"

#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

static uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

template <typename T>
static T* dalloc(size_t n, sycl::queue& q) {
    T* p = sycl::malloc_device<T>(n + 64, q);
    if (!p) {
        std::fprintf(stderr, "malloc_device failed\n");
        std::exit(2);
    }
    return p;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    auto dev = q.get_device();
    std::printf("fused_gr_bench: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
    const int t_lo = argc > 2 ? std::atoi(argv[2]) : 1, t_hi = argc > 3 ? std::atoi(argv[3]) : 6;
    const int N = 2560, HC = 4, D = N * HC, LR = 320, TM = K::kFusedGrMaxT;

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto fill = [&](std::vector<float>& v, float sc) {
        for (auto& x : v) x = sc * nd(rng);
    };

    std::vector<float> R((size_t)TM * D), bo((size_t)TM * N), inj((size_t)TM * HC), wn(D);
    fill(R, 1.0f);
    fill(bo, 0.5f);
    fill(inj, 1.0f);
    fill(wn, 0.3f);
    for (auto& x : wn) x += 1.0f;

    std::vector<uint16_t> wd((size_t)LR * D), wu((size_t)D * LR), wi((size_t)HC * D);
    for (auto& x : wd) x = bf16(0.02f * nd(rng));
    for (auto& x : wu) x = bf16(0.05f * nd(rng));
    for (auto& x : wi) x = bf16(0.02f * nd(rng));

    float* dR = dalloc<float>(R.size(), q);
    float* dbo = dalloc<float>(bo.size(), q);
    float* dinj = dalloc<float>(inj.size(), q);
    float* dwn = dalloc<float>(wn.size(), q);
    float* dxn = dalloc<float>((size_t)TM * D, q);
    float* dlo = dalloc<float>((size_t)TM * LR, q);
    float* drs = dalloc<float>((size_t)TM * HC, q);
    float* dio = dalloc<float>((size_t)TM * HC, q);
    float* dmix = dalloc<float>((size_t)TM * N, q);
    uint16_t* dwd = dalloc<uint16_t>(wd.size(), q);
    uint16_t* dwu = dalloc<uint16_t>(wu.size(), q);
    uint16_t* dwi = dalloc<uint16_t>(wi.size(), q);

    q.memcpy(dbo, bo.data(), bo.size() * sizeof(float)).wait();
    q.memcpy(dinj, inj.data(), inj.size() * sizeof(float)).wait();
    q.memcpy(dwn, wn.data(), wn.size() * sizeof(float)).wait();
    q.memcpy(dwd, wd.data(), wd.size() * sizeof(uint16_t)).wait();
    q.memcpy(dwu, wu.data(), wu.size() * sizeof(uint16_t)).wait();
    q.memcpy(dwi, wi.data(), wi.size() * sizeof(uint16_t)).wait();

    int total_runs = 0;
    for (int T = t_lo; T <= t_hi; ++T) {
        for (int apply = 0; apply < 2; ++apply) {
            for (int inject = 0; inject < 2; ++inject) {
                K::FusedGrArgs a[TM];
                for (int t = 0; t < T; ++t) {
                    a[t].R = dR + (size_t)t * D;
                    a[t].R_out = dR + (size_t)t * D;
                    a[t].apply = apply;
                    a[t].bo_prev = dbo + (size_t)t * N;
                    a[t].inj_prev = dinj + (size_t)t * HC;
                    a[t].w_norm = dwn;
                    a[t].w_down = dwd;
                    a[t].w_up = dwu;
                    a[t].w_inject = inject ? dwi : nullptr;
                    a[t].lo = dlo + (size_t)t * LR;
                    a[t].rs = drs + (size_t)t * HC;
                    a[t].inject_out = dio + (size_t)t * HC;
                    a[t].mixed = dmix + (size_t)t * N;
                }

                q.memcpy(dR, R.data(), R.size() * sizeof(float)).wait();
                q.memset(dlo, 0xff, (size_t)TM * LR * sizeof(float)).wait();
                q.memset(drs, 0xff, (size_t)TM * HC * sizeof(float)).wait();
                q.memset(dio, 0xff, (size_t)TM * HC * sizeof(float)).wait();
                q.memset(dmix, 0xff, (size_t)TM * N * sizeof(float)).wait();

                // Warmup
                for (int i = 0; i < 10; ++i) {
                    K::fused_gr_read_multi(a, T, dxn, &q);
                }
                q.wait();

                // Benchmark
                double min_us = 1e30;
                for (int round = 0; round < 3; ++round) {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    for (int i = 0; i < iters; ++i) {
                        K::fused_gr_read_multi(a, T, dxn, &q);
                    }
                    q.wait();
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
                    min_us = std::min(min_us, elapsed_us);
                }

                std::vector<float> out((size_t)TM * (D + LR + HC + HC + N));
                float* p = out.data();
                q.memcpy(p, dR, (size_t)TM * D * sizeof(float)).wait();
                p += (size_t)TM * D;
                q.memcpy(p, dlo, (size_t)TM * LR * sizeof(float)).wait();
                p += (size_t)TM * LR;
                q.memcpy(p, drs, (size_t)TM * HC * sizeof(float)).wait();
                p += (size_t)TM * HC;
                q.memcpy(p, dio, (size_t)TM * HC * sizeof(float)).wait();
                p += (size_t)TM * HC;
                q.memcpy(p, dmix, (size_t)TM * N * sizeof(float)).wait();

                unsigned long long h = 1469598103934665603ull;
                for (size_t b = 0; b < out.size() * sizeof(float); ++b) {
                    h = (h ^ ((const unsigned char*)out.data())[b]) * 1099511628211ull;
                }

                std::printf("T %d apply %d inject %d | time %6.1f us | hash %016llx\n", T, apply, inject, min_us, h);
                ++total_runs;
            }
        }
    }

    sycl::free(dR, q);
    sycl::free(dbo, q);
    sycl::free(dinj, q);
    sycl::free(dwn, q);
    sycl::free(dxn, q);
    sycl::free(dlo, q);
    sycl::free(drs, q);
    sycl::free(dio, q);
    sycl::free(dmix, q);
    sycl::free(dwd, q);
    sycl::free(dwu, q);
    sycl::free(dwi, q);

    std::printf("fused_gr_bench: %d configurations evaluated OK\n", total_runs);
    return 0;
}
