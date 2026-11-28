// src/kernels/sycl/mmvf_rows_parity.cpp - Exact multi-row BF16/F32 GEMV checks against independent single-row calls.
// Run with STRATA_MMVF_ROWS=1; synthetic data, no model or private prompts.
#include "strata/kernels/bf16_gemv.hpp"
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

template<class T>
static T* alloc(size_t n, sycl::queue& q) {
    T* p = sycl::malloc_device<T>(n + 16, q);
    if (!p) {
        std::fprintf(stderr, "malloc_device failed\n");
        std::exit(2);
    }
    return p;
}

int main() try {
    const char* flag = std::getenv("STRATA_MMVF_ROWS");
    if (!flag || flag[0] != '1') { std::fprintf(stderr, "set STRATA_MMVF_ROWS=1\n"); return 2; }
    auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.empty()) {
        std::puts("no GPU device: skipped");
        return 77;
    }
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    const int shapes[][2] = {{64,64},{192,67},{512,65},{640,128},{2560,48},
                             {2560,128},{2560,512},{2560,2560},{10240,320}};
    size_t cases = 0, values = 0, mismatches = 0;
    for (const auto& shape : shapes) for (int nt = 1; nt <= 8; ++nt)
    for (int pad : {0,2}) for (int range : {0,1,2}) {
        const int ni = shape[0], no = shape[1], ldx = ni + pad, ldy = no + pad + 1;
        std::mt19937 rng(1234 + nt * 17 + pad * 31 + range * 53 + ni + no);
        std::uniform_real_distribution<float> d(-1.f, 1.f);
        const float xs = range == 0 ? 1.f : range == 1 ? 1.e-12f : 1.e12f;
        const float ws = range == 0 ? 1.f : range == 1 ? 1.e-8f : 1.e8f;
        std::vector<float> x((size_t) nt * ldx);
        std::vector<uint16_t> w((size_t) (no + 1) * ni);
        for (auto& v : x) v = d(rng) * xs;
        for (auto& v : w) { const float f = d(rng) * ws; uint32_t b; std::memcpy(&b,&f,4); v = uint16_t(b >> 16); }
        float* dx = alloc<float>(x.size(), q);
        uint16_t* dw = alloc<uint16_t>(w.size(), q);
        const size_t count = (size_t) nt * ldy;
        float* ref = alloc<float>(count, q);
        float* got = alloc<float>(count, q);
        float* aux = alloc<float>((size_t)nt * 3, q);
        float* aref = alloc<float>((size_t)nt * 3, q);
        q.memcpy(dx, x.data(), x.size() * sizeof(float)).wait();
        q.memcpy(dw, w.data(), w.size() * sizeof(uint16_t)).wait();
        q.memset(ref, 0xff, count * sizeof(float)).wait();
        q.memset(got, 0xff, count * sizeof(float)).wait();
        q.memset(aux, 0xff, (size_t)nt * 3 * sizeof(float)).wait();
        q.memset(aref, 0xff, (size_t)nt * 3 * sizeof(float)).wait();
        for (int t = 0; t < nt; ++t) {
            k::bf16_gemv_fp32_mmvf(dx + (size_t)t * ldx, dw, ref + (size_t)t * ldy, ni, no, &q);
            k::bf16_gemv_fp32_mmvf(dx + (size_t)t * ldx, dw + (size_t)no * ni, aref + t * 3, ni, 1, &q);
        }
        q.wait();
        k::bf16_gemv_fp32_mmvf_multi(dx, ldx, dw, got, ldy, ni, no, nt, &q);
        q.wait();
        auto compare = [&](float* a, float* b, size_t n) {
            std::vector<uint32_t> aa(n), bb(n);
            q.memcpy(aa.data(), a, n * sizeof(uint32_t)).wait();
            q.memcpy(bb.data(), b, n * sizeof(uint32_t)).wait();
            size_t bad = 0; for (size_t i = 0; i < n; ++i) bad += (aa[i] != bb[i]);
            values += n; mismatches += bad;
            if (bad) std::printf("FAIL ni=%d no=%d nt=%d pad=%d range=%d different=%zu\n", ni, no, nt, pad, range, bad);
        };
        compare(ref, got, count);
        q.memset(got, 0xff, count * sizeof(float)).wait();
        const bool fused = k::bf16_gemv_fp32_mmvf_multi_aux(dx, ldx, dw, got, ldy, ni, no, nt,
                                                            dw + (size_t)no * ni, aux, 3, &q);
        if (fused) {
            q.wait();
            compare(ref, got, count);
            compare(aref, aux, (size_t)nt * 3);
        }
        if (fused != (nt >= 2 && no >= 64)) { std::printf("FAIL eligibility\n"); ++mismatches; }
        sycl::free(dx, q); sycl::free(dw, q); sycl::free(ref, q);
        sycl::free(got, q); sycl::free(aux, q); sycl::free(aref, q);
        ++cases;
    }
    std::printf("mmvf_rows_parity: %zu cases, %zu values, %zu mismatches\n", cases, values, mismatches);
    return mismatches == 0 ? 0 : 1;
} catch (const sycl::exception& e) {
    std::fprintf(stderr, "SYCL exception: %s\n", e.what());
    return 1;
}
