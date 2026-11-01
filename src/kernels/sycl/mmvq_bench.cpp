// src/kernels/sycl/mmvq_bench.cpp - Standalone dense native_mmvq benchmark.
// Tests Q6_K wide and planar kernels, including A/B comparison of 128-GRF vs 256-GRF for single-token decode.
// Usage: mmvq_bench [n_in] [n_out] [ncols]
#include "strata/kernels/native_mmvq.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int n_in = argc > 1 ? std::atoi(argv[1]) : 2560;
    const int n_out = argc > 2 ? std::atoi(argv[2]) : 2560;
    const int ncols = argc > 3 ? std::atoi(argv[3]) : 1;
    sycl::queue queue{sycl::default_selector_v, sycl::property::queue::in_order()};
    sycl::queue* s = &queue;

    std::printf("Device: %s\n", s->get_device().get_info<sycl::info::device::name>().c_str());

    // Q6_K: 210 bytes per 256 weights
    const size_t wbytes = (size_t)n_out * (n_in / 256) * 210;
    std::vector<uint8_t> hw(wbytes);
    for (size_t i = 0; i < wbytes; ++i)
        hw[i] = (uint8_t)(i * 2654435761u >> 13);
    for (size_t b = 0; b < wbytes / 210; ++b) {
        hw[b * 210 + 208] = 0x00;
        hw[b * 210 + 209] = 0x3c;
    }  // d = 1.0 (finite)
    std::vector<float> hx((size_t)ncols * n_in);
    for (size_t i = 0; i < hx.size(); ++i)
        hx[i] = (float)((i * 7919) % 1000) / 500.f - 1.f;
    void* w = sycl::malloc_device(wbytes, *s);
    float* x = sycl::malloc_device<float>(hx.size(), *s);
    void* xq = sycl::malloc_device((size_t)ncols * (n_in / 32) * 36, *s);
    float* y = sycl::malloc_device<float>((size_t)ncols * n_out, *s);
    s->memcpy(w, hw.data(), wbytes).wait();
    s->memcpy(x, hx.data(), hx.size() * 4).wait();
    strata::kernels::native_quantize_q8_1(x, xq, n_in, ncols, s);
    strata::kernels::native_mmvq_set_q6k_wide(true);

    const int it = 400;

    auto time_run = [&](void* mat, float* out_buf) -> double {
        // warm up (~150 ms)
        const auto w0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() < 150) {
            for (int i = 0; i < 20; ++i)
                strata::kernels::native_q6_k_mmvq(mat, xq, out_buf, n_in, n_out, ncols, s);
            s->wait();
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < it; ++i)
            strata::kernels::native_q6_k_mmvq(mat, xq, out_buf, n_in, n_out, ncols, s);
        s->wait();
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
    };

    auto calc_cs = [&](float* dev_ptr) -> unsigned long long {
        std::vector<float> hy((size_t)ncols * n_out);
        s->memcpy(hy.data(), dev_ptr, hy.size() * 4).wait();
        unsigned long long cs = 1469598103934665603ull;
        for (float v : hy) {
            unsigned u;
            std::memcpy(&u, &v, 4);
            cs = (cs ^ u) * 1099511628211ull;
        }
        return cs;
    };

    if (ncols == 1) {
        std::printf("\n=== Single-token Decode (ncols=1): 128-GRF vs 256-GRF A/B Comparison ===\n");
        // 1. Q6_K Wide 128-GRF
        strata::kernels::native_mmvq_set_decode_256grf(false);
        const double us_128 = time_run(w, y);
        const unsigned long long cs_128 = calc_cs(y);
        std::vector<float> y_128((size_t)n_out);
        s->memcpy(y_128.data(), y, (size_t)n_out * 4).wait();

        // 2. Q6_K Wide 256-GRF
        float* y_256_dev = sycl::malloc_device<float>((size_t)n_out, *s);
        strata::kernels::native_mmvq_set_decode_256grf(true);
        const double us_256 = time_run(w, y_256_dev);
        const unsigned long long cs_256 = calc_cs(y_256_dev);
        std::vector<float> y_256((size_t)n_out);
        s->memcpy(y_256.data(), y_256_dev, (size_t)n_out * 4).wait();
        const bool wide_equal = (std::memcmp(y_128.data(), y_256.data(), (size_t)n_out * 4) == 0);

        const double wide_delta = ((us_128 - us_256) / us_128) * 100.0;
        std::printf("q6_k wide  128-GRF: %6.1f us (%6.1f GB/s)  checksum %016llx\n",
                    us_128, wbytes / us_128 / 1e3, cs_128);
        std::printf("q6_k wide  256-GRF: %6.1f us (%6.1f GB/s)  checksum %016llx  [%s, delta: %+.1f%% %s]\n",
                    us_256, wbytes / us_256 / 1e3, cs_256,
                    wide_equal ? "bitwise EQUAL" : "DIFF", wide_delta,
                    wide_delta > 0 ? "faster" : "slower");

        // 3. Planar Layout A/B
        void* w2 = sycl::malloc_device(wbytes, *s);
        s->memcpy(w2, hw.data(), wbytes).wait();
        void* wp = strata::kernels::native_q6_k_planar_adopt(w2, n_in, n_out, "bench matrix");
        if (strata::kernels::native_q6_k_is_planar(wp)) {
            strata::kernels::native_mmvq_set_decode_256grf(false);
            const double pus_128 = time_run(wp, y);
            s->memcpy(y_128.data(), y, (size_t)n_out * 4).wait();

            strata::kernels::native_mmvq_set_decode_256grf(true);
            const double pus_256 = time_run(wp, y_256_dev);
            s->memcpy(y_256.data(), y_256_dev, (size_t)n_out * 4).wait();
            const bool planar_equal = (std::memcmp(y_128.data(), y_256.data(), (size_t)n_out * 4) == 0);
            const double planar_delta = ((pus_128 - pus_256) / pus_128) * 100.0;

            std::printf("q6_k planar 128-GRF: %6.1f us (%6.1f GB/s)\n",
                        pus_128, wbytes / pus_128 / 1e3);
            std::printf("q6_k planar 256-GRF: %6.1f us (%6.1f GB/s)  [%s, delta: %+.1f%% %s]\n",
                        pus_256, wbytes / pus_256 / 1e3,
                        planar_equal ? "bitwise EQUAL" : "DIFF", planar_delta,
                        planar_delta > 0 ? "faster" : "slower");
        }
        sycl::free(y_256_dev, *s);
    } else {
        // Multi-column run
        const double us = time_run(w, y);
        const unsigned long long cs = calc_cs(y);
        std::printf("q6_k mmvq %d x %d, %d cols: %.1f us  (%.1f GB/s of weights)  checksum %016llx\n",
                    n_in, n_out, ncols, us, wbytes / us / 1e3, cs);

        // Test planar layout
        void* w2 = sycl::malloc_device(wbytes, *s);
        s->memcpy(w2, hw.data(), wbytes).wait();
        void* wp = strata::kernels::native_q6_k_planar_adopt(w2, n_in, n_out, "bench matrix");
        if (strata::kernels::native_q6_k_is_planar(wp)) {
            float* y2 = sycl::malloc_device<float>((size_t)ncols * n_out, *s);
            const double pus = time_run(wp, y2);
            std::vector<float> hp((size_t)ncols * n_out), hy((size_t)ncols * n_out);
            s->memcpy(hp.data(), y2, hp.size() * 4).wait();
            s->memcpy(hy.data(), y, hy.size() * 4).wait();
            const bool same = std::memcmp(hp.data(), hy.data(), hp.size() * 4) == 0;
            std::printf("q6_k planar %d x %d, %d cols: %.1f us  (%.1f GB/s of weights)  bitwise %s\n",
                        n_in, n_out, ncols, pus, wbytes / pus / 1e3, same ? "EQUAL" : "DIFFERENT");
            sycl::free(y2, *s);
        }
    }

    sycl::free(w, *s);
    sycl::free(x, *s);
    sycl::free(xq, *s);
    sycl::free(y, *s);
    return 0;
}
