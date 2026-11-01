// src/kernels/sycl/mmvq_bench.cpp - Standalone dense native_mmvq benchmark.
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
    strata::kernels::native_q6_k_mmvq(w, xq, y, n_in, n_out, ncols, s);
    s->wait();
    {  // warm up (~300 ms of work)
        const auto w0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() < 300) {
            for (int i = 0; i < 20; ++i)
                strata::kernels::native_q6_k_mmvq(w, xq, y, n_in, n_out, ncols, s);
            s->wait();
        }
    }
    const int it = 400;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < it; ++i)
        strata::kernels::native_q6_k_mmvq(w, xq, y, n_in, n_out, ncols, s);
    s->wait();
    const double us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
    std::vector<float> hy((size_t)ncols * n_out);
    s->memcpy(hy.data(), y, hy.size() * 4).wait();
    unsigned long long cs = 1469598103934665603ull;
    for (float v : hy) {
        unsigned u;
        std::memcpy(&u, &v, 4);
        cs = (cs ^ u) * 1099511628211ull;
    }
    std::printf("q6_k mmvq %d x %d, %d cols: %.1f us  (%.1f GB/s of weights)  checksum %016llx\n", n_in, n_out, ncols,
                us, wbytes / us / 1e3, cs);

    // Test planar layout
    void* w2 = sycl::malloc_device(wbytes, *s);
    s->memcpy(w2, hw.data(), wbytes).wait();
    void* wp = strata::kernels::native_q6_k_planar_adopt(w2, n_in, n_out, "bench matrix");
    if (strata::kernels::native_q6_k_is_planar(wp)) {
        float* y2 = sycl::malloc_device<float>((size_t)ncols * n_out, *s);
        strata::kernels::native_q6_k_mmvq(wp, xq, y2, n_in, n_out, ncols, s);
        s->wait();
        std::vector<float> hp((size_t)ncols * n_out);
        s->memcpy(hp.data(), y2, hp.size() * 4).wait();
        const bool same = std::memcmp(hp.data(), hy.data(), hp.size() * 4) == 0;
        for (int i = 0; i < 20; ++i)
            strata::kernels::native_q6_k_mmvq(wp, xq, y2, n_in, n_out, ncols, s);
        s->wait();
        const auto p0 = std::chrono::steady_clock::now();
        for (int i = 0; i < it; ++i)
            strata::kernels::native_q6_k_mmvq(wp, xq, y2, n_in, n_out, ncols, s);
        s->wait();
        const double pus =
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - p0).count() / it;
        std::printf("q6_k planar %d x %d, %d cols: %.1f us  (%.1f GB/s of weights)  bitwise %s\n", n_in, n_out,
                    ncols, pus, wbytes / pus / 1e3, same ? "EQUAL" : "DIFFERENT");
        sycl::free(y2, *s);
    }

    sycl::free(w, *s);
    sycl::free(x, *s);
    sycl::free(xq, *s);
    sycl::free(y, *s);
    return 0;
}
