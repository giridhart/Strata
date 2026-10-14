// tests/sycl/dequant_wide_test.cpp - Wide8 vectorized dequantization bitwise parity test
#include <sycl/sycl.hpp>
#include "strata/kernels/dequant_bf16.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace K = strata::kernels;

static uint16_t f2h(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    int e = (int)((u >> 23) & 0xff) - 127 + 15;
    uint32_t m = u & 0x7fffffu;
    if (e <= 0) return (uint16_t)sign;
    if (e >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t h = sign | ((uint32_t)e << 10) | (m >> 13);
    if ((m & 0x1fffu) > 0x1000u || ((m & 0x1fffu) == 0x1000u && (h & 1u))) ++h;
    return (uint16_t)h;
}

struct Type {
    int id;
    const char* name;
    int block_bytes;
};

static const Type kTypes[] = {
    {12, "Q4_K", 144},
    {13, "Q5_K", 176},
    {14, "Q6_K", 210},
    {23, "IQ4_XS", 136}
};

static std::vector<uint8_t> make_blocks(const Type& t, int64_t rows, int64_t cols, bool sane, uint32_t seed) {
    std::mt19937 rng(seed);
    const int64_t nb = rows * (cols / 256);
    std::vector<uint8_t> b((size_t)nb * t.block_bytes);
    for (size_t i = 0; i + 4 <= b.size(); i += 4) {
        const uint32_t r = rng();
        std::memcpy(&b[i], &r, 4);
    }
    for (size_t i = b.size() & ~(size_t)3; i < b.size(); ++i) b[i] = (uint8_t)rng();
    if (sane) {
        std::uniform_real_distribution<float> ud(1e-4f, 3e-2f);
        for (int64_t k = 0; k < nb; ++k) {
            uint8_t* p = b.data() + (size_t)k * t.block_bytes;
            const uint16_t d = f2h((rng() & 1 ? 1.0f : -1.0f) * ud(rng)), dm = f2h(-ud(rng));
            if (t.id == 14) std::memcpy(p + 208, &d, 2);
            else if (t.id == 23) std::memcpy(p, &d, 2);
            else { std::memcpy(p, &d, 2); std::memcpy(p + 2, &dm, 2); }
        }
    }
    return b;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";

    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    std::printf("Running dequant_wide_test on: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    struct Shape { int64_t rows, cols; };
    const Shape shapes[] = {
        {1024, 2560},
        {512, 2560},
        {256, 4096},
        {64, 768},
        {7, 256}
    };

    int cases = 0, failures = 0;
    for (const Type& t : kTypes) {
        for (const Shape& sh : shapes) {
            for (int sane = 1; sane >= 0; --sane) {
                const int64_t rows = sh.rows, cols = sh.cols, n = rows * cols;
                const std::vector<uint8_t> hb = make_blocks(t, rows, cols, sane != 0,
                                                            (uint32_t)(t.id * 1000003 + rows * 31 + cols + sane));
                uint8_t* db = sycl::malloc_device<uint8_t>(hb.size(), q);
                q.memcpy(db, hb.data(), hb.size()).wait();

                const int64_t ldp = cols + 64; // padded stride
                const size_t cap = (size_t)rows * (size_t)ldp * 4;
                uint8_t* o0 = sycl::malloc_device<uint8_t>(cap, q);
                uint8_t* o1 = sycl::malloc_device<uint8_t>(cap, q);
                std::vector<uint8_t> h0(cap), h1(cap);

                // what: 0 FP16, 1 BF16, 2 FP32, 3 FP16 with padded stride, 4 FP16 row slice
                for (int what = 0; what < 5; ++what) {
                    const int64_t r0 = what == 4 ? rows / 3 : 0;
                    const int64_t nr = what == 4 ? rows - rows / 3 - rows / 5 : rows;
                    if (nr <= 0) continue;

                    for (int wide = 0; wide < 2; ++wide) {
                        uint8_t* o = wide ? o1 : o0;
                        q.memset(o, 0xA5, cap).wait();
                        K::dequant_set_wide(wide);
                        if (what == 0 || what == 4) K::dequant_f16(t.id, db, r0, nr, cols, (uint16_t*)o, &q);
                        else if (what == 1) K::dequant_bf16(t.id, db, r0, nr, cols, (uint16_t*)o, &q);
                        else if (what == 2) K::dequant_f32(t.id, db, r0, nr, cols, (float*)o, &q);
                        else if (!K::dequant_f16_ld(t.id, db, r0, nr, cols, ldp, (uint16_t*)o, &q)) {
                            std::printf("FAIL %s: dequant_f16_ld refused\n", t.name);
                            ++failures;
                        }
                        q.memcpy(wide ? h1.data() : h0.data(), o, cap).wait();
                    }
                    ++cases;
                    if (std::memcmp(h0.data(), h1.data(), cap) != 0) {
                        size_t nd = 0, first = 0;
                        for (size_t i = 0; i < cap; ++i) {
                            if (h0[i] != h1[i]) {
                                if (!nd) first = i;
                                ++nd;
                            }
                        }
                        std::printf("FAIL %s %lldx%lld %s what %d: %zu bytes differ, first at %zu\n",
                                    t.name, (long long)rows, (long long)cols, sane ? "sane" : "raw", what, nd, first);
                        ++failures;
                    }

                    if (what == 3) {
                        size_t bad = 0;
                        for (int64_t r = 0; r < nr; ++r) {
                            for (int64_t c = cols; c < ldp; ++c) {
                                if (h1[(size_t)(r * ldp + c) * 2] != 0xA5 ||
                                    h1[(size_t)(r * ldp + c) * 2 + 1] != 0xA5)
                                    ++bad;
                            }
                        }
                        if (bad) {
                            std::printf("FAIL %s %lldx%lld: wide kernel wrote %zu values into row padding\n",
                                        t.name, (long long)rows, (long long)cols, bad);
                            ++failures;
                        }
                    }
                }

                if (bench && sane && rows >= 512) {
                    double ms[2] = {0, 0};
                    constexpr int reps = 20;
                    for (int wide = 0; wide < 2; ++wide) {
                        K::dequant_set_wide(wide);
                        for (int i = 0; i < 3; ++i) K::dequant_f16(t.id, db, 0, rows, cols, (uint16_t*)o0, &q);
                        q.wait();
                        auto t0 = std::chrono::steady_clock::now();
                        for (int i = 0; i < reps; ++i) K::dequant_f16(t.id, db, 0, rows, cols, (uint16_t*)o0, &q);
                        q.wait();
                        auto t1 = std::chrono::steady_clock::now();
                        ms[wide] = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
                    }
                    const double bytes = (double)hb.size() + (double)n * 2;
                    std::printf("BENCH %-6s %6lldx%-6lld per-group %8.3f ms (%5.1f GB/s) | wide %8.3f ms (%5.1f GB/s) | %.2fx\n",
                                t.name, (long long)rows, (long long)cols, ms[0], bytes / ms[0] / 1e6, ms[1],
                                bytes / ms[1] / 1e6, ms[0] / ms[1]);
                }

                sycl::free(db, q);
                sycl::free(o0, q);
                sycl::free(o1, q);
            }
        }
    }
    K::dequant_set_wide(-1);
    std::printf("\ndequant_wide_test: %d cases, %s (failures: %d)\n", cases,
                failures ? "FAILED" : "PASSED (100% bitwise parity)", failures);
    return failures ? 1 : 0;
}
