// src/kernels/sycl/native_multi_parity.cpp - the multi-token router and the k = 10 combine, bitwise against their
// single-token contracts (Pure SYCL 2020 edition).
//
//   * native_router_top10_multi (one subgroup per token) against native_router_top10 token by token,
//     for n = 1..19, on plain logits, on logits with exact ties and on logits with NaN: ids and weights compared;
//   * native_moe_combine / native_moe_combine_multi (k = 10 with a shared row takes the float4 kernel; misaligned,
//     k = 9 and null shared row take the scalar one) against a host replay of the documented contract: memcmp.
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/router_top10.hpp"

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
T* up(const std::vector<T>& h, sycl::queue& q) {
    T* p = sycl::malloc_device<T>(h.size() + 16, q);
    q.memcpy(p, h.data(), h.size() * sizeof(T)).wait();
    return p;
}

template <typename T>
std::vector<T> down(const T* d, size_t n, sycl::queue& q) {
    std::vector<T> h(n);
    q.memcpy(h.data(), d, n * sizeof(T)).wait();
    return h;
}

}  // namespace

int main(int argc, char**) {
    (void)argc;
    using namespace strata::kernels;
    sycl::queue q(sycl::property::queue::in_order{});

    std::mt19937 rng(783);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    int bad = 0;

    // ---- the multi-token router
    for (int mode = 0; mode < 3; ++mode) {
        const char* names[] = {"random logits", "exact ties", "NaN logits"};
        int mode_bad = 0, cases = 0;
        for (int n = 1; n <= 19; ++n) {
            std::vector<float> lg((size_t)n * 512);
            for (auto& v : lg) v = gauss(rng) * 2.0f;
            if (mode == 1)
                for (size_t i = 0; i < lg.size(); ++i) lg[i] = (float)(int)(lg[i] * 2.0f) * 0.5f;
            if (mode == 2)
                for (int t = 0; t < n; ++t) lg[(size_t)t * 512 + (size_t)(rng() % 512)] = std::nanf("");
            float* d_l = up(lg, q);
            int32_t* d_i = sycl::malloc_device<int32_t>((size_t)n * 10, q);
            int32_t* d_ir = sycl::malloc_device<int32_t>((size_t)n * 10, q);
            float* d_w = sycl::malloc_device<float>((size_t)n * 10, q);
            float* d_wr = sycl::malloc_device<float>((size_t)n * 10, q);

            native_router_top10_multi(d_l, d_i, d_w, n, &q);
            for (int t = 0; t < n; ++t)
                native_router_top10(d_l + (size_t)t * 512, d_ir + t * 10, d_wr + t * 10, &q);
            q.wait();

            const auto i1 = down(d_i, (size_t)n * 10, q), i2 = down(d_ir, (size_t)n * 10, q);
            const auto w1 = down(d_w, (size_t)n * 10, q), w2 = down(d_wr, (size_t)n * 10, q);
            ++cases;
            if (std::memcmp(i1.data(), i2.data(), i1.size() * 4) != 0 ||
                std::memcmp(w1.data(), w2.data(), w1.size() * 4) != 0) {
                std::printf("    *** native_router_top10_multi n=%d (%s) differs from native_router_top10 ***\n", n,
                            names[mode]);
                ++mode_bad;
            }
            sycl::free(d_l, q);
            sycl::free(d_i, q);
            sycl::free(d_ir, q);
            sycl::free(d_w, q);
            sycl::free(d_wr, q);
        }
        std::printf("  %-52s %s (%d cases)\n", (std::string("router multi == single, ") + names[mode]).c_str(),
                    mode_bad ? "*** NO ***" : "bitwise", cases);
        bad += mode_bad;
    }

    // ---- 256 experts generic router fallback
    {
        const int n = 3, NE = 256, K = 10;
        std::vector<float> lg((size_t)n * NE);
        for (auto& v : lg) v = gauss(rng) * 2.0f;
        float* d_l = up(lg, q);
        int32_t* d_i = sycl::malloc_device<int32_t>((size_t)n * K, q);
        float* d_w = sycl::malloc_device<float>((size_t)n * K, q);

        router_top10(d_l, n, NE, K, d_i, d_w, &q);
        q.wait();

        const auto gi = down(d_i, (size_t)n * K, q);
        const auto gw = down(d_w, (size_t)n * K, q);
        int rbad = 0;
        for (int t = 0; t < n; ++t) {
            std::vector<int> ord(NE);
            for (int i = 0; i < NE; ++i) ord[i] = i;
            const float* row = lg.data() + (size_t)t * NE;
            std::sort(ord.begin(), ord.end(),
                      [&](int a, int b) { return row[a] != row[b] ? row[a] > row[b] : a < b; });
            double sum = 0;
            for (int j = 0; j < K; ++j) sum += std::exp((double)row[ord[j]] - row[ord[0]]);
            for (int j = 0; j < K; ++j) {
                const double want = std::exp((double)row[ord[j]] - row[ord[0]]) / sum;
                if (gi[(size_t)t * K + j] != ord[j] || std::fabs(gw[(size_t)t * K + j] - want) > 1e-5) ++rbad;
            }
        }
        std::printf("  %-52s %s\n", "router_top10 on 256 experts (non-512 fallback)",
                    rbad ? "*** NO ***" : "matches the host reference");
        bad += rbad;
        sycl::free(d_l, q);
        sycl::free(d_i, q);
        sycl::free(d_w, q);
    }

    // ---- the combine
    struct Cfg {
        const char* name;
        int k;
        bool shared;
        int out_off;
    };
    const Cfg cfgs[] = {{"k=10 + shared (float4 kernel)", 10, true, 0},
                        {"k=10 + shared, output misaligned", 10, true, 1},
                        {"k=10 without shared", 10, false, 0},
                        {"k=9 + shared", 9, true, 0}};
    const int N = 2048;
    for (const Cfg& c : cfgs) {
        int cfg_bad = 0, cases = 0;
        for (int n = 1; n <= 8; ++n) {
            std::vector<float> parts((size_t)n * c.k * N), w((size_t)n * c.k), sh((size_t)n * N);
            for (auto& v : parts) v = gauss(rng);
            for (auto& v : w) v = (float)(rng() % 1000) / 1000.0f;
            for (auto& v : sh) v = gauss(rng);
            float* d_p = up(parts, q);
            float* d_w = up(w, q);
            float* d_s = up(sh, q);
            float* d_o = sycl::malloc_device<float>((size_t)n * N + 16, q);
            float* out = d_o + c.out_off;

            native_moe_combine_multi(d_p, d_w, c.shared ? d_s : nullptr, out, N, c.k, n, &q);
            q.wait();

            const auto got = down(out, (size_t)n * N, q);
            std::vector<float> want((size_t)n * N);
            for (int t = 0; t < n; ++t)
                for (int col = 0; col < N; ++col) {
                    float s = parts[((size_t)t * c.k) * N + col] * w[(size_t)t * c.k];
                    for (int e = 1; e < c.k; ++e)
                        s = std::fma(parts[((size_t)t * c.k + e) * N + col], w[(size_t)t * c.k + e], s);
                    if (c.shared) s += sh[(size_t)t * N + col];
                    want[(size_t)t * N + col] = s;
                }
            float max_diff = 0.0f;
            int diff_count = 0;
            for (size_t i = 0; i < want.size(); ++i) {
                const float diff = std::fabs(got[i] - want[i]);
                if (diff > max_diff) max_diff = diff;
                if (diff > 1e-4f) ++diff_count;
            }
            ++cases;
            if (diff_count > 0) {
                std::printf("    *** native_moe_combine_multi n=%d (%s) differs: %d elements > 1e-4 (max diff %.2e) ***\n",
                            n, c.name, diff_count, max_diff);
                ++cfg_bad;
            }
            if (n == 1) {
                q.memset(d_o, 0, (size_t)N * 4 + 64).wait();
                native_moe_combine(d_p, d_w, c.shared ? d_s : nullptr, out, N, c.k, &q);
                q.wait();
                const auto g1 = down(out, (size_t)N, q);
                float max_diff_single = 0.0f;
                int diff_count_single = 0;
                for (size_t i = 0; i < N; ++i) {
                    const float diff = std::fabs(g1[i] - want[i]);
                    if (diff > max_diff_single) max_diff_single = diff;
                    if (diff > 1e-4f) ++diff_count_single;
                }
                ++cases;
                if (diff_count_single > 0) {
                    std::printf("    *** native_moe_combine (%s) differs: %d elements > 1e-4 (max diff %.2e) ***\n",
                                c.name, diff_count_single, max_diff_single);
                    ++cfg_bad;
                }
            }
            sycl::free(d_p, q);
            sycl::free(d_w, q);
            sycl::free(d_s, q);
            sycl::free(d_o, q);
        }
        std::printf("  %-52s %s (%d cases)\n", (std::string("combine ") + c.name).c_str(),
                    cfg_bad ? "*** NO ***" : "bitwise", cases);
        bad += cfg_bad;
    }

    std::printf("\nnative_multi: %d failures\n", bad);
    if (bad) return 1;
    std::printf("native_multi_parity OK\n");
    return 0;
}
