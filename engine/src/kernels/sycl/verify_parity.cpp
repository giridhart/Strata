// src/kernels/sycl/verify_parity.cpp - pure SYCL 2020 batched / split verify-window kernels against the kernels they replace, BITWISE.
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

template <typename T>
T* dev(size_t n, sycl::queue& q) {
    T* p = sycl::malloc_device<T>(n + 16, q);
    if (!p) {
        std::fprintf(stderr, "malloc_device failed\n");
        std::exit(1);
    }
    return p;
}

template <typename T>
void up(T* d, const std::vector<T>& h, sycl::queue& q) {
    q.memcpy(d, h.data(), h.size() * sizeof(T)).wait();
}

template <typename T>
std::vector<T> down(const T* d, size_t n, sycl::queue& q) {
    std::vector<T> h(n);
    q.memcpy(h.data(), d, n * sizeof(T)).wait();
    return h;
}

int test_argmax(std::mt19937& rng, sycl::queue& q) {
    using namespace strata::kernels;
    int bad = 0;
    std::normal_distribution<float> nd(0.0f, 4.0f);
    uint8_t* sa = dev<uint8_t>(argmax_rows_scratch_bytes(8), q);
    uint8_t* st = dev<uint8_t>(row_top_prob_scratch_bytes(8), q);
    q.memset(sa, 0, argmax_rows_scratch_bytes(8)).wait();
    q.memset(st, 0, row_top_prob_scratch_bytes(8)).wait();

    for (const int n : {248320, 40525, 4097, 33}) {
        for (const int rows : {4, 8, 1, 3, 8, 2}) {
            std::vector<float> h((size_t)rows * n);
            for (float& v : h) v = nd(rng);
            for (int r = 0; r < rows; ++r) {
                float* row = h.data() + (size_t)r * n;
                const int a = (int)(rng() % n), b = (int)(rng() % n);
                row[a] = row[b] = 40.0f;                       // a tie at the maximum
                row[(int)(rng() % n)] = std::nanf("");        // never picked
                row[(int)(rng() % n)] = -INFINITY;
                if (r == 2)                                    // no value above -inf: 0
                    for (int i = 0; i < n; ++i) row[i] = i % 7 ? -INFINITY : std::nanf("");
            }
            float* d_l = dev<float>(h.size(), q);
            up(d_l, h, q);
            int* d_ref = dev<int>(rows, q);
            int32_t* d_out = dev<int32_t>(rows, q);
            float* d_p1 = dev<float>(rows, q);
            float* d_p2 = dev<float>(rows, q);
            SamplerParams sp;
            sp.greedy = true;
            sp.temperature = 0.0f;
            sample_tokens(d_l, rows, n, nullptr, 0, sp, d_ref, &q);
            row_top_prob(d_l, rows, n, d_ref, d_p1, &q);
            for (int rep = 0; rep < 3; ++rep) {
                argmax_rows(d_l, rows, n, sa, d_out, &q);
                row_top_prob_split(d_l, rows, n, d_out, d_p2, st, &q);
                q.wait();
                const std::vector<int> ref = down(d_ref, rows, q);
                const std::vector<int32_t> out = down(d_out, rows, q);
                const std::vector<float> p1 = down(d_p1, rows, q), p2 = down(d_p2, rows, q);
                for (int r = 0; r < rows; ++r) {
                    if (ref[r] != out[r] || std::memcmp(&p1[r], &p2[r], 4) != 0) {
                        if (bad < 5)
                            std::printf("  argmax n %d rows %d row %d: %d / %d, p %.9g / %.9g\n", n, rows, r, ref[r],
                                        out[r], p1[r], p2[r]);
                        ++bad;
                    }
                }
            }
            sycl::free(d_l, q);
            sycl::free(d_ref, q);
            sycl::free(d_out, q);
            sycl::free(d_p1, q);
            sycl::free(d_p2, q);
        }
    }
    sycl::free(sa, q);
    sycl::free(st, q);
    std::printf("argmax_rows, row_top_prob_split: %s\n",
                bad ? "MISMATCH" : "bitwise equal to the one-block kernels (ties, NaN, -inf, 1-8 rows on one scratch)");
    return bad;
}

int test_logits_nonfinite(std::mt19937& rng, sycl::queue& q) {
    (void)rng;
    using namespace strata::kernels;
    int bad = 0;
    const int n = 4096, rows = 4;
    std::vector<float> h((size_t)rows * n, 1.0f);
    // Put NaN in row 0, inf in row 2
    h[100] = std::nanf("");
    h[(size_t)2 * n + 500] = INFINITY;
    float* d_l = dev<float>(h.size(), q);
    up(d_l, h, q);
    uint32_t* d_flags = dev<uint32_t>((size_t)rows * kNonfiniteBlocks, q);
    logits_nonfinite_rows(d_l, rows, n, d_flags, &q);
    q.wait();
    std::vector<uint32_t> flags = down(d_flags, (size_t)rows * kNonfiniteBlocks, q);
    bool r0_has_nan = false, r1_has_nan = false, r2_has_inf = false, r3_has_nan = false;
    for (int b = 0; b < kNonfiniteBlocks; ++b) {
        if (flags[0 * kNonfiniteBlocks + b]) r0_has_nan = true;
        if (flags[1 * kNonfiniteBlocks + b]) r1_has_nan = true;
        if (flags[2 * kNonfiniteBlocks + b]) r2_has_inf = true;
        if (flags[3 * kNonfiniteBlocks + b]) r3_has_nan = true;
    }
    if (!r0_has_nan || r1_has_nan || !r2_has_inf || r3_has_nan) {
        std::printf("  logits_nonfinite mismatch: r0=%d r1=%d r2=%d r3=%d\n",
                    r0_has_nan, r1_has_nan, r2_has_inf, r3_has_nan);
        ++bad;
    }
    sycl::free(d_l, q);
    sycl::free(d_flags, q);
    std::printf("logits_nonfinite_rows: %s\n", bad ? "MISMATCH" : "PASS");
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) != "--selftest") {
            std::fprintf(stderr, "usage: verify_parity [--selftest]\n");
            return 2;
        }
    }
    std::mt19937 rng(20260925);
    sycl::queue q{sycl::property::queue::in_order()};
    int bad = 0;
    bad += test_argmax(rng, q);
    bad += test_logits_nonfinite(rng, q);
    std::printf("verify_parity: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
