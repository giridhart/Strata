// src/kernels/sycl/b6_mmvq_bench.cpp - B6: what a verify row costs in the dense projections, per kernel, for SYCL.
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

namespace {

struct Case {
    const char* name;
    int type;          // GGML type id; 30 = BF16 (bf16_gemv_fp32_mmvf_multi)
    int n_in, n_out;
    int block_elems, block_bytes;
    int scale_at[2];   // byte offsets of the fp16 scales in a block; -1 = none
};

const Case CASES[] = {
    {"gdn qkv Q6_K", 14, 2560, 10240, 256, 210, {208, -1}},
    {"gdn qkv Q4_K", 12, 2560, 10240, 256, 144, {0, 2}},
    {"gdn qkv Q5_K", 13, 2560, 10240, 256, 176, {0, 2}},
    {"gdn qkv IQ4_XS", 23, 2560, 10240, 256, 136, {0, -1}},
    {"gdn gate Q6_K", 14, 2560, 6144, 256, 210, {208, -1}},
    {"gdn out Q6_K", 14, 6144, 2560, 256, 210, {208, -1}},
    {"gdn out IQ4_XS", 23, 6144, 2560, 256, 136, {0, -1}},
    {"gdn out Q4_K", 12, 6144, 2560, 256, 144, {0, 2}},
    {"qsa q IQ4_XS", 23, 2560, 12288, 256, 136, {0, -1}},
    {"qsa q Q6_K", 14, 2560, 12288, 256, 210, {208, -1}},
    {"shexp up IQ3_S", 21, 2560, 640, 256, 110, {0, -1}},
    {"shexp gate Q4_K", 12, 2560, 640, 256, 144, {0, 2}},
    {"shexp down IQ4_NL", 20, 640, 2560, 32, 18, {0, -1}},
    {"router BF16", 30, 2560, 512, 1, 2, {-1, -1}},
    {"head Q5_K", 13, 2560, 248320, 256, 176, {0, 2}},
};
const int WIDTHS[] = {1, 2, 3, 4, 5, 8};

uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}
uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

struct Buf {
    std::vector<void*> w;   // weight copies
    std::size_t wbytes = 0;
    void* xq = nullptr;     // Q8_1 of 8 columns
    float* xf = nullptr;    // the float activations (BF16 path)
    float* y = nullptr;     // 8 columns of output
};

void call(const Case& c, const Buf& b, int k, int ncols, float* y, sycl::queue& s) {
    if (c.type == 30)
        K::bf16_gemv_fp32_mmvf_multi(b.xf, c.n_in, (const uint16_t*) b.w[(size_t) k], y, c.n_out, c.n_in, c.n_out, ncols, &s);
    else
        K::native_mmvq(c.type, b.w[(size_t) k], b.xq, y, c.n_in, c.n_out, ncols, &s);
}

void call_col(const Case& c, const Buf& b, int j, float* y, sycl::queue& s) {
    if (c.type == 30)
        K::bf16_gemv_fp32_mmvf_multi(b.xf + (size_t) j * c.n_in, c.n_in, (const uint16_t*) b.w[0],
                                     y + (size_t) j * c.n_out, c.n_out, c.n_in, c.n_out, 1, &s);
    else
        K::native_mmvq(c.type, b.w[0], (const uint8_t*) b.xq + K::native_q8_1_bytes(c.n_in, 1) * (size_t) j,
                       y + (size_t) j * c.n_out, c.n_in, c.n_out, 1, &s);
}

double median_ms(const Case& c, const Buf& b, int ncols, int reps, sycl::queue& s) {
    const int nk = (int) b.w.size();
    for (int r = 0; r < 20; ++r) call(c, b, r % nk, ncols, b.y, s);   // warm-up
    s.wait();
    std::vector<float> t((size_t) reps);
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        call(c, b, r % nk, ncols, b.y, s);
        s.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        t[(size_t) r] = std::chrono::duration<float, std::milli>(t1 - t0).count();
    }
    std::nth_element(t.begin(), t.begin() + reps / 2, t.end());
    return t[(size_t) reps / 2];
}

}  // namespace

int main(int argc, char** argv) try {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int reps = argc > 1 ? std::max(20, std::atoi(argv[1])) : 100;
    auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.empty()) {
        std::puts("SKIP: no GPU");
        return 77;
    }
    sycl::queue s{sycl::default_selector_v, sycl::property::queue::in_order()};
    auto dev = s.get_device();
    const int bmode = argc > 3 ? std::atoi(argv[3]) : 1;
    long long bad = 0;

    std::printf("\n=== device: %s, %d reps, median ms per call, B = rows mode %d ===\n",
                dev.get_info<sycl::info::device::name>().c_str(), reps, bmode);
    std::printf("%-18s %4s %9s %9s %8s %8s  %s\n", "case", "cols", "A ms", "B ms", "A +col", "B +col",
                "bitwise B=A, A=1col, B=1col");

    for (const Case& c : CASES) {
        if (c.type != 30 && !K::native_mmvq_supported(c.type)) {
            std::printf("%-18s unsupported type %d\n", c.name, c.type);
            continue;
        }
        std::mt19937 rng(77u + (unsigned) c.type * 131u + (unsigned) c.n_out);
        Buf b;
        b.wbytes = c.type == 30 ? (size_t) c.n_in * c.n_out * 2 : K::native_mmvq_weight_bytes(c.type, c.n_in, c.n_out);
        std::vector<uint8_t> w(b.wbytes);
        if (c.type == 30) {
            std::normal_distribution<float> nd(0.f, 0.05f);
            auto* p = (uint16_t*) w.data();
            for (size_t i = 0; i < b.wbytes / 2; ++i) p[i] = bf16(nd(rng));
        } else {
            for (auto& x : w) x = (uint8_t) (rng() & 0xff);
            const size_t nb = (size_t) c.n_out * (size_t) (c.n_in / c.block_elems);
            for (size_t k = 0; k < nb; ++k)
                for (int at : c.scale_at)
                    if (at >= 0) {
                        const uint16_t h = sane_half(rng);
                        std::memcpy(&w[k * (size_t) c.block_bytes + (size_t) at], &h, 2);
                    }
        }
        const size_t copies = std::max<size_t>(1, std::min<size_t>(8, (96u << 20) / b.wbytes + 1));
        for (size_t k = 0; k < copies; ++k) {
            void* d = sycl::malloc_device<uint8_t>(b.wbytes + 64, s);
            s.memcpy(d, w.data(), b.wbytes).wait();
            b.w.push_back(d);
        }
        std::vector<float> x((size_t) 8 * c.n_in);
        std::normal_distribution<float> nx(0.f, 1.f);
        for (auto& v : x) v = nx(rng);
        b.xf = sycl::malloc_device<float>(x.size() + 16, s);
        s.memcpy(b.xf, x.data(), x.size() * sizeof(float)).wait();
        b.y = sycl::malloc_device<float>((size_t) 8 * c.n_out + 16, s);
        if (c.type != 30) {
            b.xq = sycl::malloc_device<uint8_t>(K::native_q8_1_bytes(c.n_in, 8) + 64, s);
            K::quantize_q8_1_rows(b.xf, 8, c.n_in, b.xq, &s);
        }
        float *ya = sycl::malloc_device<float>((size_t) 8 * c.n_out + 16, s);
        float *yb = sycl::malloc_device<float>((size_t) 8 * c.n_out + 16, s);
        float *y1 = sycl::malloc_device<float>((size_t) 8 * c.n_out + 16, s);

        double a1 = 0, b1 = 0;
        for (int T : WIDTHS) {
            std::vector<uint32_t> ha((size_t) T * c.n_out), hb(ha.size()), h1(ha.size());
            K::native_mmvq_set_b6_rows(0);
            const double ta = median_ms(c, b, T, reps, s);
            call(c, b, 0, T, ya, s);
            for (int j = 0; j < T; ++j) call_col(c, b, j, y1, s);
            K::native_mmvq_set_b6_rows(bmode);
            const double tb = median_ms(c, b, T, reps, s);
            call(c, b, 0, T, yb, s);
            K::native_mmvq_set_b6_rows(0);
            s.wait();
            s.memcpy(ha.data(), ya, ha.size() * 4).wait();
            s.memcpy(hb.data(), yb, hb.size() * 4).wait();
            s.memcpy(h1.data(), y1, h1.size() * 4).wait();
            const bool ba = std::memcmp(ha.data(), hb.data(), ha.size() * 4) == 0;
            const bool a_1 = std::memcmp(ha.data(), h1.data(), ha.size() * 4) == 0;
            const bool b_1 = std::memcmp(hb.data(), h1.data(), hb.size() * 4) == 0;
            long long nonfinite = 0;
            for (uint32_t u : ha) { float f; std::memcpy(&f, &u, 4); if (!std::isfinite(f)) ++nonfinite; }
            if (!ba || !b_1 || nonfinite) ++bad;
            if (T == 1) { a1 = ta; b1 = tb; }
            std::printf("%-18s %4d %9.4f %9.4f %8.4f %8.4f  %s %s %s%s\n", c.name, T, ta, tb,
                        T > 1 ? (ta - a1) / (T - 1) : 0.0, T > 1 ? (tb - b1) / (T - 1) : 0.0, ba ? "ok" : "DIFF",
                        a_1 ? "ok" : "diff", b_1 ? "ok" : "DIFF", nonfinite ? " NONFINITE" : "");
        }
        for (void* d : b.w) sycl::free(d, s);
        sycl::free(b.xq, s); sycl::free(b.xf, s); sycl::free(b.y, s);
        sycl::free(ya, s); sycl::free(yb, s); sycl::free(y1, s);
    }
    std::printf("\nb6_mmvq_bench: %s (%lld case/width comparisons failed)\n", bad ? "FAILED" : "ok", bad);
    return bad ? 1 : 0;
} catch (const sycl::exception& e) {
    std::fprintf(stderr, "SYCL exception: %s\n", e.what());
    return 1;
}
