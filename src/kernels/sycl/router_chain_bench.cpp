// src/kernels/sycl/router_chain_bench.cpp - pure SYCL 2020 benchmark of the router chain (MMVF + router top-10 + doorbell).
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

template <typename T>
static T* dalloc(size_t n, sycl::queue& q, int fill = 0) {
    T* p = sycl::malloc_device<T>(n + 64, q);
    if (!p) {
        std::fprintf(stderr, "malloc_device failed\n");
        std::exit(2);
    }
    q.memset(p, fill, (n + 64) * sizeof(T)).wait();
    return p;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    auto dev = q.get_device();
    std::printf("router_chain_bench: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    const int T = argc > 1 ? std::atoi(argv[1]) : 2;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
    const int L = 48, N = 2560, NE = 512, KK = 10;

    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> w((size_t)NE * N);
    for (auto& x : w) {
        const float f = 0.05f * nd(rng);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        x = (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
    }
    std::vector<float> xh((size_t)T * N);
    for (auto& x : xh) x = nd(rng);

    std::vector<uint16_t*> dw(L);
    for (int l = 0; l < L; ++l) {
        dw[l] = dalloc<uint16_t>(w.size(), q);
        q.memcpy(dw[l], w.data(), w.size() * sizeof(uint16_t)).wait();
    }

    float* dx = dalloc<float>(xh.size(), q);
    q.memcpy(dx, xh.data(), xh.size() * sizeof(float)).wait();
    float* dlog = dalloc<float>((size_t)T * NE, q);
    float* dwt = dalloc<float>((size_t)T * KK, q);
    int32_t* dids = dalloc<int32_t>((size_t)T * KK, q);
    int32_t* dids_o = dalloc<int32_t>((size_t)T * KK, q);
    float* dxo = dalloc<float>(xh.size(), q);
    int32_t* dres = dalloc<int32_t>(NE, q);
    std::vector<int32_t> res(NE);
    for (int i = 0; i < NE; ++i) res[i] = i;
    q.memcpy(dres, res.data(), NE * sizeof(int32_t)).wait();

    uint32_t* dseq = sycl::malloc_shared<uint32_t>(1, q);
    *dseq = 0;
    int* dz = dalloc<int>(4, q);

    {
        // 1. Parity across shapes and token counts: check bitwise FNV hash
        const int shapes[][2] = {{2560, 512}, {2560, 256}, {2560, 64}, {1280, 513}, {640, 100},
                                 {4096, 50},  {2048, 7},   {256, 33},   {8192, 90}};
        for (const auto& sh : shapes) {
            const int ni = sh[0], no = sh[1];
            std::vector<uint16_t> wh((size_t)no * ni);
            for (auto& v : wh) {
                const float f = 0.05f * nd(rng);
                uint32_t u;
                std::memcpy(&u, &f, 4);
                v = (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
            }
            std::vector<float> xs((size_t)8 * ni);
            for (auto& v : xs) v = nd(rng);

            uint16_t* dwp = dalloc<uint16_t>(wh.size(), q);
            q.memcpy(dwp, wh.data(), wh.size() * sizeof(uint16_t)).wait();
            float* dxp = dalloc<float>(xs.size(), q);
            q.memcpy(dxp, xs.data(), xs.size() * sizeof(float)).wait();
            float* dyp = dalloc<float>((size_t)8 * no, q);

            for (int tk = 1; tk <= 8; ++tk) {
                q.memset(dyp, 0xff, (size_t)8 * no * sizeof(float)).wait();
                K::bf16_gemv_fp32_mmvf_multi(dxp, ni, dwp, dyp, no, ni, no, tk, &q);
                q.wait();

                std::vector<float> yh((size_t)8 * no);
                q.memcpy(yh.data(), dyp, yh.size() * sizeof(float)).wait();
                unsigned long long h = 1469598103934665603ull;
                for (size_t b = 0; b < yh.size() * sizeof(float); ++b) {
                    h = (h ^ ((const unsigned char*)yh.data())[b]) * 1099511628211ull;
                }
                std::printf("PARITY n_in %d n_out %d T %d hash %016llx\n", ni, no, tk, h);
            }
            sycl::free(dwp, q);
            sycl::free(dxp, q);
            sycl::free(dyp, q);
        }
    }

    {
        // 2. Shape microbenchmarks across token counts
        const int outs[] = {512, 256, 128, 64, 48, 8, 1};
        const int ins[] = {2560, 2560, 2560, 2560, 2560, 2560, 2560};
        for (int si = 0; si < 7; ++si) {
            const int no = outs[si], ni = ins[si];
            std::vector<uint16_t*> dws(L);
            for (int l = 0; l < L; ++l) {
                dws[l] = dalloc<uint16_t>((size_t)no * ni, q);
                q.memcpy(dws[l], w.data(),
                         ((size_t)no * ni < w.size() ? (size_t)no * ni : w.size()) * sizeof(uint16_t)).wait();
            }
            for (int tk : {1, 2, 3, 4, 6}) {
                // Warmup
                for (int l = 0; l < L; ++l) {
                    K::bf16_gemv_fp32_mmvf_multi(dx, N, dws[l], dlog, NE, ni, no, tk, &q);
                }
                q.wait();

                std::vector<double> us_times;
                for (int r = 0; r < reps; ++r) {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    for (int l = 0; l < L; ++l) {
                        K::bf16_gemv_fp32_mmvf_multi(dx, N, dws[l], dlog, NE, ni, no, tk, &q);
                    }
                    q.wait();
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double elapsed = std::chrono::duration<double, std::micro>(t1 - t0).count();
                    us_times.push_back(elapsed / L);
                }
                std::sort(us_times.begin(), us_times.end());
                std::printf("SHAPE n_in %d n_out %3d T %d : %6.2f us per call\n", ni, no, tk,
                            us_times[us_times.size() / 2]);
            }
            for (int l = 0; l < L; ++l) sycl::free(dws[l], q);
        }
    }

    // 3. Chain pipeline stages
    const char* names[] = {"empty node", "mmvf", "route", "mmvf+route", "doorbell", "mmvf+doorbell",
                           "route+doorbell", "mmvf+route+doorbell"};
    for (int mode = 0; mode < 8; ++mode) {
        // Warmup
        for (int l = 0; l < L; ++l) {
            if (mode == 0) K::gpu_stamp((unsigned long long*)dz, 0, &q);
            if (mode & 1) K::bf16_gemv_fp32_mmvf_multi(dx, N, dw[l], dlog, NE, N, NE, T, &q);
            if (mode & 2) K::native_router_top10_multi(dlog, dids, dwt, T, &q);
            if (mode & 4) K::doorbell_publish_res(dx, dids, dres, NE, (int64_t)T * N, (int64_t)T * KK, dxo, dids_o, dseq, &q);
        }
        q.wait();

        std::vector<double> us_times;
        for (int r = 0; r < reps; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int l = 0; l < L; ++l) {
                if (mode == 0) K::gpu_stamp((unsigned long long*)dz, 0, &q);
                if (mode & 1) K::bf16_gemv_fp32_mmvf_multi(dx, N, dw[l], dlog, NE, N, NE, T, &q);
                if (mode & 2) K::native_router_top10_multi(dlog, dids, dwt, T, &q);
                if (mode & 4) K::doorbell_publish_res(dx, dids, dres, NE, (int64_t)T * N, (int64_t)T * KK, dxo, dids_o, dseq, &q);
            }
            q.wait();
            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double, std::micro>(t1 - t0).count();
            us_times.push_back(elapsed / L);
        }
        std::sort(us_times.begin(), us_times.end());
        std::printf("T %d %-20s : %6.2f us per layer (median of %d reps of %d layers)\n", T, names[mode],
                    us_times[us_times.size() / 2], reps, L);
    }

    for (int l = 0; l < L; ++l) sycl::free(dw[l], q);
    sycl::free(dx, q);
    sycl::free(dlog, q);
    sycl::free(dwt, q);
    sycl::free(dids, q);
    sycl::free(dids_o, q);
    sycl::free(dxo, q);
    sycl::free(dres, q);
    sycl::free(dseq, q);
    sycl::free(dz, q);

    std::printf("router_chain_bench OK\n");
    return 0;
}
