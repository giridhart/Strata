// src/kernels/sycl/mmvq_sg_bench.cpp - SYCL MMVQ sub-group and GRF benchmarking on model shapes.
// Tests all dense quantization formats with configurable STRATA_MMVQ_SG, STRATA_MMVQ_LARGE_GRF,
// and side-by-side 128-GRF vs 256-GRF single-token decode comparisons.
// Usage: mmvq_sg_bench [reps=200]
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    const int it_user = argc > 1 ? std::atoi(argv[1]) : 200;
    sycl::queue queue{sycl::default_selector_v, sycl::property::queue::in_order()};
    sycl::queue* s = &queue;

    const char* sg_env = std::getenv("STRATA_MMVQ_SG");
    const char* grf_env = std::getenv("STRATA_MMVQ_LARGE_GRF");
    const char* dec_grf_env = std::getenv("STRATA_MMVQ_DECODE_256GRF");
    std::printf("Device: %s\n", s->get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("Configuration: STRATA_MMVQ_SG=%s, STRATA_MMVQ_LARGE_GRF=%s, STRATA_MMVQ_DECODE_256GRF=%s\n",
                sg_env ? sg_env : "default(32)", grf_env ? grf_env : "default(auto)",
                dec_grf_env ? dec_grf_env : "default(0/128-GRF)");

    // Enable wide kernels for Q6_K and IQ3_S
    strata::kernels::native_mmvq_set_q6k_wide(true);
    setenv("STRATA_IQ3S_WIDE", "1", 0);

    struct T {
        const char* name;
        int type, bytes, elems;
    };  // ggml type, bytes per block of `elems` weights
    const T types[] = {
        {"iq3_s", 21, 110, 256},
        {"q6_k", 14, 210, 256},
        {"q5_k", 13, 176, 256},
        {"q4_k", 12, 144, 256},
        {"iq4_xs", 23, 136, 256},
        {"q8_0", 8, 34, 32},
        {"iq4_nl", 20, 18, 32}
    };
    const int shapes[][2] = {
        {2560, 2560},   // Square hidden projection
        {2560, 10240},  // Wide MLP expansion
        {6144, 2560},   // Attention projection
        {2560, 640}     // Short projection
    };
    const int cols[] = {1, 2, 4};

    for (const T& t : types) {
        for (auto& sh : shapes) {
            const int n_in = sh[0], n_out = sh[1];
            if (n_in % t.elems != 0) continue;
            const size_t wbytes = (size_t)n_out * (n_in / t.elems) * t.bytes;
            std::vector<uint8_t> hw(wbytes);
            for (size_t i = 0; i < wbytes; ++i)
                hw[i] = (uint8_t)(i * 2654435761u >> 13) & 0x3b;  // keeps fp16 fields finite
            void* w = sycl::malloc_device(wbytes, *s);
            s->memcpy(w, hw.data(), wbytes).wait();

            for (int nc : cols) {
                std::vector<float> hx((size_t)nc * n_in);
                for (size_t i = 0; i < hx.size(); ++i)
                    hx[i] = (float)((i * 7919) % 1000) / 500.f - 1.f;
                float* x = sycl::malloc_device<float>(hx.size(), *s);
                void* xq = sycl::malloc_device((size_t)nc * (n_in / 32) * 36, *s);
                float* y = sycl::malloc_device<float>((size_t)nc * n_out, *s);
                s->memcpy(x, hx.data(), hx.size() * 4).wait();
                strata::kernels::native_quantize_q8_1(x, xq, n_in, nc, s);

                auto run = [&] { strata::kernels::native_mmvq(t.type, w, xq, y, n_in, n_out, nc, s); };

                auto bench = [&](bool large_grf) -> double {
                    strata::kernels::native_mmvq_set_decode_256grf(large_grf);
                    strata::kernels::iq_set_decode_256grf(large_grf);
                    // Warm up
                    const auto w0 = std::chrono::steady_clock::now();
                    while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() < 80) {
                        for (int i = 0; i < 10; ++i) run();
                        s->wait();
                    }
                    const int it = it_user;
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int i = 0; i < it; ++i) run();
                    s->wait();
                    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
                };

                if (nc == 1) {
                    const double us_128 = bench(false);
                    const double us_256 = bench(true);
                    const double delta = ((us_128 - us_256) / us_128) * 100.0;
                    std::printf("%-7s %5d x %5d  cols 1 [128-GRF]  %7.1f us  %6.1f GB/s\n",
                                t.name, n_in, n_out, us_128, wbytes / us_128 / 1e3);
                    std::printf("%-7s %5d x %5d  cols 1 [256-GRF]  %7.1f us  %6.1f GB/s  (delta: %+.1f%% %s)\n",
                                t.name, n_in, n_out, us_256, wbytes / us_256 / 1e3,
                                delta, delta > 0 ? "faster" : "slower");
                } else {
                    const double us = bench(false);
                    std::printf("%-7s %5d x %5d  cols %d            %7.1f us  %6.1f GB/s\n",
                                t.name, n_in, n_out, nc, us, wbytes / us / 1e3);
                }

                sycl::free(x, *s);
                sycl::free(xq, *s);
                sycl::free(y, *s);
            }
            sycl::free(w, *s);
        }
    }
    return 0;
}
