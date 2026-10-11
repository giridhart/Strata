// src/kernels/sycl/gr_multi_parity.cpp - pure SYCL 2020 multi-token hyper-connection read and write parity
// (gr_read_multi, gr_write_multi and the exact-T BF16 projections under them) against the single-token calls, memcmp.
#include "strata/kernels/gr.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

template <typename T>
T* up(const std::vector<T>& h, sycl::queue& q) {
    T* p = sycl::malloc_device<T>(h.size() + 16, q);
    if (!p) {
        std::fprintf(stderr, "malloc_device failed\n");
        std::exit(2);
    }
    q.memcpy(p, h.data(), h.size() * sizeof(T)).wait();
    return p;
}

template <typename T>
T* dev(size_t n, sycl::queue& q) {
    T* p = sycl::malloc_device<T>(n + 16, q);
    if (!p) {
        std::fprintf(stderr, "malloc_device failed\n");
        std::exit(2);
    }
    q.memset(p, 0, (n + 16) * sizeof(T)).wait();
    return p;
}

template <typename T>
std::vector<T> down(const T* d, size_t n, sycl::queue& q) {
    std::vector<T> h(n);
    q.memcpy(h.data(), d, n * sizeof(T)).wait();
    return h;
}

uint16_t bf16_of(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}

}  // namespace

int main() {
    sycl::queue q{sycl::property::queue::in_order()};

    const long long n_embd = 2560, hc = 4, hc_lr = 320, hc_dim = hc * n_embd;
    const float eps = 1e-6f;
    std::mt19937 rng(783);
    std::normal_distribution<float> gauss(0.f, 1.f);
    k::gr_set_native_mmvf(true);

    std::vector<float> w_norm((size_t)hc_dim);
    for (auto& x : w_norm) x = 1.0f + 0.1f * gauss(rng);
    std::vector<uint16_t> w_down((size_t)(hc_lr * hc_dim)), w_up((size_t)(hc_dim * hc_lr)), w_inj((size_t)(hc * hc_dim));
    for (auto& x : w_down) x = bf16_of(gauss(rng) * 0.05f);
    for (auto& x : w_up) x = bf16_of(gauss(rng) * 0.05f);
    for (auto& x : w_inj) x = bf16_of(gauss(rng) * 0.02f);
    float* d_norm = up(w_norm, q);
    uint16_t* d_down = up(w_down, q);
    uint16_t* d_up = up(w_up, q);
    uint16_t* d_inj = up(w_inj, q);

    const k::GrShapes sh{n_embd, hc, hc_lr};
    void* ws_raw = sycl::malloc_device<uint8_t>(k::gr_workspace_bytes(sh) + 64, q);
    k::GrWorkspace ws;
    k::gr_workspace_init(sh, ws_raw, ws);

    int cases = 0, bad = 0;
    for (int T = 1; T <= 8; ++T) {
        std::vector<float> R((size_t)T * hc_dim);
        for (int t = 0; t < T; ++t)
            for (long long c = 0; c < hc; ++c)
                for (long long d = 0; d < n_embd; ++d) R[(size_t)t * hc_dim + c * n_embd + d] = gauss(rng) * (float)(1 << (2 * c));
        float* d_R = up(R, q);
        float* d_xn = dev<float>((size_t)T * hc_dim, q);
        float* d_lo = dev<float>((size_t)T * hc_lr, q);
        float* d_g = dev<float>((size_t)T * hc_dim, q);
        for (int with_inj = 0; with_inj < 2; ++with_inj) {
            float* d_mixed_m = dev<float>((size_t)T * n_embd, q);
            float* d_mixed_s = dev<float>((size_t)T * n_embd, q);
            float* d_inject_m = dev<float>((size_t)T * hc, q);
            float* d_inject_s = dev<float>((size_t)T * hc, q);
            const uint16_t* wi = with_inj ? d_inj : nullptr;
            k::gr_read_multi(d_R, d_norm, d_down, d_up, wi, eps, sh, ws, d_xn, d_lo, d_g, d_mixed_m,
                             with_inj ? d_inject_m : nullptr, T, &q);
            for (int t = 0; t < T; ++t)
                k::gr_read(d_R + (size_t)t * hc_dim, d_norm, d_down, d_up, wi, eps, sh, ws, d_mixed_s + (size_t)t * n_embd,
                           with_inj ? d_inject_s + (size_t)t * hc : nullptr, &q);
            q.wait();
            bool ok = std::memcmp(down(d_mixed_m, (size_t)T * n_embd, q).data(), down(d_mixed_s, (size_t)T * n_embd, q).data(),
                                  (size_t)T * n_embd * 4) == 0;
            if (with_inj)
                ok = ok && std::memcmp(down(d_inject_m, (size_t)T * hc, q).data(), down(d_inject_s, (size_t)T * hc, q).data(),
                                       (size_t)T * hc * 4) == 0;
            ++cases;
            if (!ok) {
                std::printf("    *** gr_read_multi T=%d %s differs from gr_read ***\n", T, with_inj ? "with injection" : "final mixer");
                ++bad;
            }
            sycl::free(d_mixed_m, q);
            sycl::free(d_mixed_s, q);
            sycl::free(d_inject_m, q);
            sycl::free(d_inject_s, q);
        }

        // the write: R_out[t] = R[t] + block_out[t] * 2 sigmoid(inject[t] / hc)
        std::vector<float> bo((size_t)T * n_embd), inj((size_t)T * hc);
        for (auto& x : bo) x = gauss(rng);
        for (auto& x : inj) x = gauss(rng) * 3.0f;
        float* d_bo = up(bo, q);
        float* d_in = up(inj, q);
        for (int inplace = 0; inplace < 2; ++inplace) {
            float* d_Rm = up(R, q);
            float* d_Rs = up(R, q);
            float* d_om = inplace ? d_Rm : dev<float>((size_t)T * hc_dim, q);
            float* d_os = inplace ? d_Rs : dev<float>((size_t)T * hc_dim, q);
            k::gr_write_multi(d_Rm, d_bo, d_in, sh, d_om, T, &q);
            for (int t = 0; t < T; ++t)
                k::gr_write(d_Rs + (size_t)t * hc_dim, d_bo + (size_t)t * n_embd, d_in + (size_t)t * hc, sh,
                            d_os + (size_t)t * hc_dim, &q);
            q.wait();
            const bool ok = std::memcmp(down(d_om, (size_t)T * hc_dim, q).data(), down(d_os, (size_t)T * hc_dim, q).data(),
                                        (size_t)T * hc_dim * 4) == 0;
            ++cases;
            if (!ok) {
                std::printf("    *** gr_write_multi T=%d %s differs from gr_write ***\n", T, inplace ? "in place" : "out of place");
                ++bad;
            }
            sycl::free(d_Rm, q);
            sycl::free(d_Rs, q);
            if (!inplace) {
                sycl::free(d_om, q);
                sycl::free(d_os, q);
            }
        }
        sycl::free(d_R, q);
        sycl::free(d_xn, q);
        sycl::free(d_lo, q);
        sycl::free(d_g, q);
        sycl::free(d_bo, q);
        sycl::free(d_in, q);
    }
    sycl::free(ws_raw, q);
    sycl::free(d_norm, q);
    sycl::free(d_down, q);
    sycl::free(d_up, q);
    sycl::free(d_inj, q);

    std::printf("gr_multi: %d cases, %d failures\n", cases, bad);
    if (bad) return 1;
    std::printf("gr_multi_parity OK\n");
    return 0;
}
