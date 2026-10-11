// src/kernels/sycl/verify_batch_parity.cpp - pure SYCL 2020 verify window's one-launch forms against per-token launches, bitwise.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_gr_postops.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_fail = 0;

template <typename T>
T* dalloc(size_t n, sycl::queue& q, int fill = 0) {
    T* p = sycl::malloc_device<T>(n + 64, q);
    if (!p) {
        std::fprintf(stderr, "malloc_device failed\n");
        std::exit(2);
    }
    q.memset(p, fill, (n + 64) * sizeof(T)).wait();
    return p;
}

template <typename T>
std::vector<T> read(const T* d, size_t n, sycl::queue& q) {
    std::vector<T> h(n);
    q.memcpy(h.data(), d, n * sizeof(T)).wait();
    return h;
}

template <typename T>
T* upload(const std::vector<T>& h, sycl::queue& q) {
    T* d = sycl::malloc_device<T>(h.size() + 64, q);
    if (!d) {
        std::fprintf(stderr, "malloc_device failed\n");
        std::exit(2);
    }
    q.memcpy(d, h.data(), h.size() * sizeof(T)).wait();
    return d;
}

void report(const char* what, bool ok) {
    std::printf("%-64s %s\n", what, ok ? "bitwise equal" : "DIFFERENT");
    if (!ok) ++g_fail;
}

// 1. the hyper-connection write
void gr_post(sycl::queue& q) {
    const int N = 2560, HC = 4;
    std::mt19937 rng(12);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int n : {2, 4, 8}) {
        std::vector<float> r((size_t)n * HC * N), bo((size_t)n * N), inj((size_t)n * HC);
        for (auto& x : r) x = nd(rng);
        for (auto& x : bo) x = nd(rng);
        for (auto& x : inj) x = 3.f * nd(rng);
        float* r0 = upload(r, q);
        float* r1 = upload(r, q);
        float* d_bo = upload(bo, q);
        float* d_inj = upload(inj, q);
        for (int t = 0; t < n; ++t)
            k::native_gr_post(r0 + (size_t)t * HC * N, d_bo + (size_t)t * N, d_inj + (size_t)t * HC,
                              r0 + (size_t)t * HC * N, N, HC, &q);
        k::native_gr_post_multi(r1, d_bo, d_inj, r1, N, HC, n, (long long)HC * N, N, HC, &q);
        q.wait();
        char what[128];
        std::snprintf(what, sizeof what, "native_gr_post_multi, %d tokens (in place)", n);
        report(what, read(r0, r.size(), q) == read(r1, r.size(), q));
        sycl::free(r0, q); sycl::free(r1, q); sycl::free(d_bo, q); sycl::free(d_inj, q);
    }
}

// 2. the PLE post-ops of a window, with the commit's history snapshots
void ple(sycl::queue& q) {
    constexpr int N = k::NG_N_EMBD, D = k::NG_HC_DIM, HS = k::NG_HIST * k::NG_HC_DIM;
    std::mt19937 rng(13);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> nk(D), nq(D), nc(D);
    for (auto* v : {&nk, &nq, &nc})
        for (auto& x : *v) x = 1.f + 0.1f * nd(rng);
    std::vector<uint16_t> conv((size_t)4 * D);
    for (auto& x : conv) x = k::f16_from_f32(0.3f * nd(rng));
    k::PleWeights w;
    w.norm_key = upload(nk, q);
    w.norm_query = upload(nq, q);
    w.norm_conv = upload(nc, q);
    w.conv1d_f16 = upload(conv, q);
    std::vector<float> hist0((size_t)HS);
    for (auto& x : hist0) x = nd(rng);
    for (int n : {2, 3, 4, 8}) {
        std::vector<float> key((size_t)n * D), hid((size_t)n * D), val((size_t)n * N);
        for (auto& x : key) x = 4.f * nd(rng);
        for (auto& x : hid) x = nd(rng);
        for (auto& x : val) x = nd(rng);
        float* key0 = upload(key, q);
        float* hid0 = upload(hid, q);
        float* val0 = upload(val, q);
        float* hist_a = upload(hist0, q);
        float* snap_a = dalloc<float>((size_t)n * HS, q);
        float *q_buf = dalloc<float>(D, q), *nrm = dalloc<float>(D, q), *g = dalloc<float>(4, q), *gd = dalloc<float>(D, q);
        float* cv = dalloc<float>(D, q);
        for (int t = 0; t < n; ++t) {
            k::NativePlePostopsBuffers b{q_buf, nrm, g, gd, nrm, cv, hid0 + (size_t)t * D};
            k::native_ple_postops(key0 + (size_t)t * D, hid0 + (size_t)t * D, val0 + (size_t)t * N, hist_a, w, b, &q);
            k::ple_history_advance(hist_a, nrm, &q);
            q.memcpy(snap_a + (size_t)t * HS, hist_a, (size_t)HS * sizeof(float));
        }
        float* key1 = upload(key, q);
        float* hid1 = upload(hid, q);
        float* val1 = upload(val, q);
        float* hist_b = upload(hist0, q);
        float* snap_b = dalloc<float>((size_t)n * HS, q);
        float *qn = dalloc<float>((size_t)n * D, q), *gated = dalloc<float>((size_t)n * D, q), *gate = dalloc<float>((size_t)n * 4, q);
        k::native_ple_postops_batch_snap(key1, hid1, val1, hist_b, w, qn, gated, gate, n, snap_b, &q);
        q.wait();
        char what[128];
        std::snprintf(what, sizeof what, "native_ple_postops_batch_snap, %d tokens: results", n);
        report(what, read(hid0, hid.size(), q) == read(hid1, hid.size(), q));
        std::snprintf(what, sizeof what, "native_ple_postops_batch_snap, %d tokens: history", n);
        report(what, read(hist_a, (size_t)HS, q) == read(hist_b, (size_t)HS, q));
        std::snprintf(what, sizeof what, "native_ple_postops_batch_snap, %d tokens: snapshots", n);
        report(what, read(snap_a, (size_t)n * HS, q) == read(snap_b, (size_t)n * HS, q));

        for (float* p : {key0, hid0, val0, hist_a, snap_a, q_buf, nrm, g, gd, cv, key1, hid1, val1, hist_b, snap_b, qn, gated, gate})
            sycl::free(p, q);
    }
    sycl::free((void*)w.norm_key, q);
    sycl::free((void*)w.norm_query, q);
    sycl::free((void*)w.norm_conv, q);
    sycl::free((void*)w.conv1d_f16, q);
}

}  // namespace

int main() {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    auto dev = q.get_device();
    std::printf("verify_batch_parity: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    gr_post(q);
    ple(q);
    std::printf("verify_batch_parity: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
