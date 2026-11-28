// src/kernels/sycl/route_window_parity.cpp - pure SYCL 2020 verify window's batched routing against the per-token calls, bit for bit.
//
// moe_route_window replaces n per-token (projection + top-k) pairs by one multi-column BF16 projection and one
// n-token top-k. Speculative decoding is exact only if a token's route does not depend on how many tokens share
// its window, so both batched kernels must equal their one-token forms exactly - this checks that on random data
// shaped like the router (2560 -> 256, top-10) and like the shared-expert gate (2560 -> 1).
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/router_top10.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
uint16_t bf16(float f) {
    uint32_t i;
    std::memcpy(&i, &f, 4);
    return (uint16_t)((i + ((i >> 16) & 1u) + 0x7FFFu) >> 16);
}
}  // namespace

int main() {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    const int N = 2560, E = 256, K = 10;
    int bad = 0;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int n_out : {E, 1}) {
        for (int T : {2, 3, 4, 6}) {
            std::vector<uint16_t> w((size_t)n_out * N);
            std::vector<float> x((size_t)T * N);
            for (auto& v : w) v = bf16(nd(rng) * 0.05f);
            for (auto& v : x) v = nd(rng);
            uint16_t* dw = sycl::malloc_device<uint16_t>(w.size(), q);
            float* dx = sycl::malloc_device<float>(x.size(), q);
            float* dy1 = sycl::malloc_device<float>((size_t)T * n_out, q);
            float* dyn = sycl::malloc_device<float>((size_t)T * n_out, q);
            q.memcpy(dw, w.data(), w.size() * sizeof(uint16_t));
            q.memcpy(dx, x.data(), x.size() * sizeof(float)).wait();

            for (int t = 0; t < T; ++t)
                strata::kernels::bf16_gemv_fp32_mmvf(dx + (size_t)t * N, dw, dy1 + (size_t)t * n_out, N, n_out, &q);
            strata::kernels::bf16_gemv_fp32_mmvf_cols(dx, dw, dyn, N, n_out, T, &q);
            q.wait();

            std::vector<float> y1((size_t)T * n_out), yn((size_t)T * n_out);
            q.memcpy(y1.data(), dy1, y1.size() * sizeof(float));
            q.memcpy(yn.data(), dyn, yn.size() * sizeof(float)).wait();
            const bool same = std::memcmp(y1.data(), yn.data(), y1.size() * sizeof(float)) == 0;
            std::printf("  mmvf cols  n_out %3d  T %d  %s\n", n_out, T, same ? "bit-identical" : "*** DIFFERS ***");
            if (!same) ++bad;

            if (n_out == E) {
                int* i1 = sycl::malloc_device<int>((size_t)T * K, q);
                int* in_ = sycl::malloc_device<int>((size_t)T * K, q);
                float* w1 = sycl::malloc_device<float>((size_t)T * K, q);
                float* wn = sycl::malloc_device<float>((size_t)T * K, q);
                for (int t = 0; t < T; ++t)
                    strata::kernels::router_top10(dy1 + (size_t)t * E, 1, E, K, i1 + t * K, w1 + t * K, &q);
                strata::kernels::router_top10(dy1, T, E, K, in_, wn, &q);
                q.wait();

                std::vector<int> a((size_t)T * K), b((size_t)T * K);
                std::vector<float> fa((size_t)T * K), fb((size_t)T * K);
                q.memcpy(a.data(), i1, a.size() * sizeof(int));
                q.memcpy(b.data(), in_, b.size() * sizeof(int));
                q.memcpy(fa.data(), w1, fa.size() * sizeof(float));
                q.memcpy(fb.data(), wn, fb.size() * sizeof(float)).wait();

                const bool s2 = a == b && std::memcmp(fa.data(), fb.data(), fa.size() * sizeof(float)) == 0;
                std::printf("  top10 n    T %d            %s\n", T, s2 ? "bit-identical" : "*** DIFFERS ***");
                if (!s2) ++bad;
                sycl::free(i1, q); sycl::free(in_, q); sycl::free(w1, q); sycl::free(wn, q);
            }
            sycl::free(dw, q); sycl::free(dx, q); sycl::free(dy1, q); sycl::free(dyn, q);
        }
    }
    std::printf(bad ? "route_window_parity: %d FAILURES\n" : "route_window_parity OK\n", bad);
    return bad ? 1 : 0;
}
