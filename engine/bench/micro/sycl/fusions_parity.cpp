// bench/micro/sycl/fusions_parity.cpp - the fusion work against its own unfused pipeline, BITWISE, plus
// the before/after kernel timings (std::chrono).
//
//     build_sycl/fusions_parity            every bitwise check below; exit 1 on any difference
//     build_sycl/fusions_parity --bench    fused vs unfused kernel times at decode shapes (M=1 and M=8)
//
// WHAT IS CHECKED (synthetic data, GPU, no model):
//   1. `swilu_quantize_q8_0` / `_scaled` against `swiglu` + `quantize_q8_0` / `_scaled` - the S2 expert
//      path's intermediate.  Products, q8 bytes and the fp32 scales all bitwise.
//   2. `swilu_quantize_q8_0` / `_q8_K` with the LEGACY silu against `shared_expert`'s swiglu kernel +
//      `quantize_q8_0` / `quantize_q8_K`.
//   3. `quantize_act_images` against the three standalone quantizers (q8_0 bytes, q8_K bytes, bf16 bits).
//   4. `bf16_gemv_pair` against two `bf16_gemv_split(32)` calls (the subgroup accumulation it replicates).
#include <sycl/sycl.hpp>
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s_gemv.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_fail = 0;

void report(const char* name, bool ok) {
    std::printf("  %-52s %s\n", name, ok ? "bitwise identical" : "DIFFERS");
    if (!ok) ++g_fail;
}

template <typename T>
T* dalloc(size_t n, sycl::queue& q) {
    T* p = sycl::malloc_device<T>(n, q);
    if (!p) {
        std::fprintf(stderr, "dalloc: malloc_device failed\n");
        std::exit(2);
    }
    q.memset(p, 0, n * sizeof(T)).wait();
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

// the `gemv` dispatch `shared_expert`'s lambda uses (defined later; declared for `run_bench`)
void gemv_dispatch(const k::SForm& f, const uint8_t* codes, const float* scales, const float* off,
                   const uint8_t* x80, const uint8_t* x8k, float* y, long long n_in, long long n_out,
                   void* stream);

// ---- the unfused SwiGLU references, one thread per element (the shapes the standalone kernels had) ----
void swilu_legacy_kernel(sycl::queue& q, const float* g, const float* u, float* out, int n) {
    const int block_size = 128;
    const int blocks = (n + block_size - 1) / block_size;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * block_size), sycl::range<1>(block_size)),
                   [=](sycl::nd_item<1> item) {
                       const int i = item.get_global_id(0);
                       if (i >= n) return;
                       const double x = (double)g[i];
                       out[i] = (float)(x / (1.0 + sycl::exp(-x))) * u[i];
                   });
}

void swilu_fast_kernel(sycl::queue& q, const float* g, const float* u, float* out, int n) {
    const int block_size = 128;
    const int blocks = (n + block_size - 1) / block_size;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * block_size), sycl::range<1>(block_size)),
                   [=](sycl::nd_item<1> item) {
                       const int i = item.get_global_id(0);
                       if (i >= n) return;
                       out[i] = (g[i] / (1.0f + sycl::native::exp(-g[i]))) * u[i];
                   });
}

void swilu_native_kernel(sycl::queue& q, const float* g, const float* u, float* out, int n) {
    const int block_size = 128;
    const int blocks = (n + block_size - 1) / block_size;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * block_size), sycl::range<1>(block_size)),
                   [=](sycl::nd_item<1> item) {
                       const int i = item.get_global_id(0);
                       if (i >= n) return;
                       out[i] = (g[i] / (1.0f + sycl::native::exp(-g[i]))) * u[i];
                   });
}

void fill_pairs(std::vector<float>& g, std::vector<float>& u, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> gv(-6.0f, 6.0f), uv(-3.0f, 3.0f);
    for (size_t i = 0; i < g.size(); ++i) { g[i] = gv(rng); u[i] = uv(rng); }
}

void check_swilu_quantize(sycl::queue& q, int kind, bool scaled, const char* name) {
    const long long n = 640 * 4;                       // FF x a few hits, multiple of 256
    std::vector<float> hg(n), hu(n);
    fill_pairs(hg, hu, 7u + (unsigned) kind * 13 + (unsigned) scaled);
    float *dg = dalloc<float>(n, q), *du = dalloc<float>(n, q), *dfused = dalloc<float>(n, q), *dref = dalloc<float>(n, q);
    up(dg, hg, q); up(du, hu, q);
    const size_t nb80 = (size_t) (n / 32);
    uint8_t* db80 = dalloc<uint8_t>(nb80 * 34 + 256, q);
    uint8_t* db80r = dalloc<uint8_t>(nb80 * 34 + 256, q);
    float* ds = dalloc<float>(nb80 + 16, q);
    float* dsr = dalloc<float>(nb80 + 16, q);

    // fused
    if (scaled) k::swilu_quantize_q8_0_scaled(dg, du, dfused, n, kind, db80, ds, &q);
    else k::swilu_quantize_q8_0(dg, du, dfused, n, kind, db80, &q);
    // unfused: standalone swilu then quantize
    if (kind == 0) swilu_legacy_kernel(q, dg, du, dref, (int) n);
    else swilu_fast_kernel(q, dg, du, dref, (int) n);
    if (scaled) k::quantize_q8_0_scaled(dref, db80r, dsr, n, &q);
    else k::quantize_q8_0(dref, db80r, n, &q);
    q.wait();

    const std::vector<float> p1 = down(dfused, n, q), p2 = down(dref, n, q);
    const std::vector<uint8_t> q1 = down(db80, nb80 * 34, q), q2 = down(db80r, nb80 * 34, q);
    bool ok = std::memcmp(p1.data(), p2.data(), n * sizeof(float)) == 0 &&
              std::memcmp(q1.data(), q2.data(), nb80 * 34) == 0;
    if (scaled) {
        const std::vector<float> s1 = down(ds, nb80, q), s2 = down(dsr, nb80, q);
        ok = ok && std::memcmp(s1.data(), s2.data(), nb80 * sizeof(float)) == 0;
    }
    report(name, ok);
    sycl::free(dg, q); sycl::free(du, q); sycl::free(dfused, q); sycl::free(dref, q);
    sycl::free(db80, q); sycl::free(db80r, q);
    sycl::free(ds, q); sycl::free(dsr, q);
}

void check_swilu_quantize_q8_K(sycl::queue& q) {
    const long long n = 256 * 3;
    std::vector<float> hg(n), hu(n);
    fill_pairs(hg, hu, 99u);
    float *dg = dalloc<float>(n, q), *du = dalloc<float>(n, q), *df = dalloc<float>(n, q), *dr = dalloc<float>(n, q);
    up(dg, hg, q); up(du, hu, q);
    const size_t nbK = (size_t) (n / 256);
    uint8_t* dK = dalloc<uint8_t>(nbK * 292 + 256, q);
    uint8_t* dKr = dalloc<uint8_t>(nbK * 292 + 256, q);
    k::swilu_quantize_q8_K(dg, du, df, n, 0, dK, &q);
    swilu_legacy_kernel(q, dg, du, dr, (int) n);
    k::quantize_q8_K(dr, dKr, n, &q);
    q.wait();
    bool ok = down(df, n, q) == down(dr, n, q) && down(dK, nbK * 292, q) == down(dKr, nbK * 292, q);
    report("swilu_quantize_q8_K (legacy silu)", ok);
    sycl::free(dg, q); sycl::free(du, q); sycl::free(df, q); sycl::free(dr, q);
    sycl::free(dK, q); sycl::free(dKr, q);
}

void check_images(sycl::queue& q) {
    const long long n = 2560;
    std::vector<float> hx(n);
    std::mt19937 rng(4u);
    std::uniform_real_distribution<float> d(-4.0f, 4.0f);
    for (auto& v : hx) v = d(rng);
    float* dx = dalloc<float>(n, q);
    up(dx, hx, q);
    const size_t nb80 = (size_t) (n / 32), nbK = (size_t) (n / 256);
    uint8_t* i80 = dalloc<uint8_t>(nb80 * 34 + 256, q);
    uint8_t* iK = dalloc<uint8_t>(nbK * 292 + 256, q);
    uint16_t* i16 = dalloc<uint16_t>(n + 8, q);
    uint8_t* r80 = dalloc<uint8_t>(nb80 * 34 + 256, q);
    uint8_t* rK = dalloc<uint8_t>(nbK * 292 + 256, q);
    k::quantize_act_images(dx, n, i80, iK, i16, &q);
    k::quantize_q8_0(dx, r80, n, &q);
    k::quantize_q8_K(dx, rK, n, &q);
    q.wait();
    // the standalone bf16 conversion is `bf16_from_f32` per element (`f32_to_bf16_bulk`)
    std::vector<uint16_t> hb(n);
    for (long long i = 0; i < n; ++i) hb[(size_t) i] = k::bf16_from_f32(hx[(size_t) i]);
    const std::vector<uint16_t> got16 = down(i16, n, q);
    report("quantize_act_images (q8_0 + q8_K + bf16)",
           down(i80, nb80 * 34, q) == down(r80, nb80 * 34, q) &&
           down(iK, nbK * 292, q) == down(rK, nbK * 292, q) &&
           std::memcmp(got16.data(), hb.data(), n * sizeof(uint16_t)) == 0);
    sycl::free(dx, q); sycl::free(i80, q); sycl::free(iK, q); sycl::free(i16, q);
    sycl::free(r80, q); sycl::free(rK, q);
}

void check_pair(sycl::queue& q) {
    const long long n_in = 2560, n1 = 48, n2 = 48;    // ssm_alpha / ssm_beta shapes
    std::vector<uint16_t> hx(n_in), hw1(n_in * n1), hw2(n_in * n2);
    std::mt19937 rng(11u);
    std::uniform_real_distribution<float> d(-2.0f, 2.0f);
    auto put = [&](std::vector<uint16_t>& v) {
        for (auto& t : v) t = k::bf16_from_f32(d(rng));
    };
    put(hx); put(hw1); put(hw2);
    uint16_t *dx = dalloc<uint16_t>(n_in, q), *dw1 = dalloc<uint16_t>(n_in * n1, q), *dw2 = dalloc<uint16_t>(n_in * n2, q);
    up(dx, hx, q); up(dw1, hw1, q); up(dw2, hw2, q);
    float *dy1 = dalloc<float>(n1 + 8, q), *dy2 = dalloc<float>(n2 + 8, q), *dr1 = dalloc<float>(n1 + 8, q),
          *dr2 = dalloc<float>(n2 + 8, q);
    k::bf16_gemv_pair(dx, dw1, dy1, n1, dw2, dy2, n2, n_in, &q);
    k::bf16_gemv_split(dx, dw1, dr1, n_in, n1, 32, &q);
    k::bf16_gemv_split(dx, dw2, dr2, n_in, n2, 32, &q);
    q.wait();
    report("bf16_gemv_pair vs 2x bf16_gemv_split(32)",
           down(dy1, n1, q) == down(dr1, n1, q) && down(dy2, n2, q) == down(dr2, n2, q));
    sycl::free(dx, q); sycl::free(dw1, q); sycl::free(dw2, q); sycl::free(dy1, q);
    sycl::free(dy2, q); sycl::free(dr1, q); sycl::free(dr2, q);
}

// ---- micro bench: fused vs unfused kernel time at decode shapes, steady_clock ------------------------
//
// The decode geometry is the model's: n_embd 2560, FF 640 for the experts (H/FF from `cpu/expert.hpp`).
// M = 1 is the single-token decode; M = 8 runs the M=8 pair shapes (8 columns of the same weights).

float time_us(sycl::queue& q, const std::function<void()>& f, int reps) {
    f();
    q.wait();  // warmup
    const auto a = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) f();
    q.wait();
    const auto b = std::chrono::steady_clock::now();
    const double elapsed_us = std::chrono::duration<double, std::micro>(b - a).count();
    return (float)(elapsed_us / reps);
}

void bench_pair(sycl::queue& q, const char* what, const std::function<void()>& before,
                const std::function<void()>& after, int reps) {
    const float t0 = time_us(q, before, reps), t1 = time_us(q, after, reps);
    std::printf("  %-44s unfused %8.1f us   fused %8.1f us   %+.1f%%\n", what, t0, t1,
                (t1 / t0 - 1.0f) * 100.0f);
}

// ---- (a) the gate/up dual GEMV + SwiGLU kernel against its unfused pipeline, bitwise ----------------
// The unfused pipeline is the one `shared_expert` ran: a standalone GEMV per projection (dispatching the
// same way its `gemv` lambda does) then the standalone SwiGLU kernel.  Both silu kinds are checked.

void gemv_dispatch(const k::SForm& f, const uint8_t* codes, const float* scales, const float* off,
                   const uint8_t* x80, const uint8_t* x8k, float* y, long long n_in, long long n_out,
                   void* stream) {
    if (f.code_bits == 2) {
        std::fprintf(stderr, "gemv_dispatch: code_bits 2 goes through s2_gemv_q8 - not exercised here\n");
        std::exit(2);
    }
    if (f.act_kind == 1) k::s_gemv_q8k_split(x8k, codes, scales, off, y, n_in, n_out, f, stream);
    else k::s_gemv_q8_0_split(x80, codes, scales, off, y, n_in, n_out, f, stream);
}

void check_dual_gemv(sycl::queue& q, int swilu_kind, const char* name) {
    const long long n_in = 2560, n_out = 128;
    const long long n_groups = n_in / 64;
    std::vector<uint8_t> hcg((size_t) n_out * n_in / 2), hcu((size_t) n_out * n_in / 2);  // S4 codes
    std::vector<float> hsg((size_t) n_out * n_groups), hsu((size_t) n_out * n_groups);
    std::vector<float> hog((size_t) n_out * n_groups), hou((size_t) n_out * n_groups);
    std::vector<float> hx(n_in);
    std::mt19937 rng(77u + (unsigned) swilu_kind);
    for (auto& v : hcg) v = (uint8_t) rng();
    for (auto& v : hcu) v = (uint8_t) rng();
    std::uniform_real_distribution<float> sc(1e-3f, 3e-2f), xv(-4.0f, 4.0f), ov(-1e-2f, 1e-2f);
    for (auto& v : hsg) v = sc(rng);
    for (auto& v : hsu) v = sc(rng);
    for (auto& v : hog) v = ov(rng);
    for (auto& v : hou) v = ov(rng);
    for (auto& v : hx) v = xv(rng);

    // Q8_0 activation image of x (both sides use it)
    const size_t nb80 = (size_t) (n_in / 32);
    std::vector<uint8_t> hx80(nb80 * 34 + 256);
    uint8_t* dx80 = dalloc<uint8_t>(hx80.size(), q);
    float* dx = dalloc<float>(n_in, q);
    up(dx, hx, q);
    k::quantize_q8_0(dx, dx80, n_in, &q);
    q.wait();

    uint8_t *dcg = dalloc<uint8_t>(hcg.size(), q), *dcu = dalloc<uint8_t>(hcu.size(), q);
    float *dsg = dalloc<float>(hsg.size(), q), *dsu = dalloc<float>(hsu.size(), q);
    float *dog = dalloc<float>(hog.size(), q), *dou = dalloc<float>(hou.size(), q);
    up(dcg, hcg, q); up(dcu, hcu, q); up(dsg, hsg, q); up(dsu, hsu, q); up(dog, hog, q); up(dou, hou, q);
    float* d_pair = dalloc<float>(n_out + 8, q);
    float* d_gate = dalloc<float>(n_out + 8, q), *d_up = dalloc<float>(n_out + 8, q), *d_ref = dalloc<float>(n_out + 8, q);

    const k::SForm fg{4, -8, 64, k::Codebook::Affine, true, 0};
    const k::SForm fu{4, -8, 64, k::Codebook::Affine, true, 0};
    const bool ok_launch = k::s_gemv_pair_silu(dx80, nullptr, fg, dcg, dsg, dog, fu, dcu, dsu, dou,
                                               d_pair, n_in, n_out, 32, swilu_kind, &q);
    // unfused: the same two GEMVs the `gemv` lambda would run, then the standalone swilu
    gemv_dispatch(fg, dcg, dsg, dog, dx80, nullptr, d_gate, n_in, n_out, &q);
    gemv_dispatch(fu, dcu, dsu, dou, dx80, nullptr, d_up, n_in, n_out, &q);
    if (swilu_kind == 0) swilu_legacy_kernel(q, d_gate, d_up, d_ref, (int) n_out);
    else if (swilu_kind == 1) swilu_fast_kernel(q, d_gate, d_up, d_ref, (int) n_out);
    else swilu_native_kernel(q, d_gate, d_up, d_ref, (int) n_out);
    q.wait();
    report(name, ok_launch && down(d_pair, n_out, q) == down(d_ref, n_out, q));

    sycl::free(dx80, q); sycl::free(dx, q); sycl::free(dcg, q); sycl::free(dcu, q);
    sycl::free(dsg, q); sycl::free(dsu, q);
    sycl::free(dog, q); sycl::free(dou, q); sycl::free(d_pair, q); sycl::free(d_gate, q);
    sycl::free(d_up, q); sycl::free(d_ref, q);
}

void run_bench(sycl::queue& q) {
    std::printf("\nfusions bench: decode shapes (n_embd 2560, FF 640), SYCL in-order queue, %s\n",
                "200 reps");
    const int reps = 200;

    for (int m : {1, 8}) {
        const long long pairs = (long long) 640 * m;             // one expert hit's rows x tokens
        std::vector<float> hg(pairs), hu(pairs);
        fill_pairs(hg, hu, 5u + (unsigned) m);
        float *dg = dalloc<float>(pairs, q), *du = dalloc<float>(pairs, q), *dout = dalloc<float>(pairs, q);
        up(dg, hg, q); up(du, hu, q);
        const size_t nb = (size_t) (pairs / 32);
        uint8_t* d80 = dalloc<uint8_t>(nb * 34 + 256, q);
        float* ds = dalloc<float>(nb + 16, q);
        char label[96];
        std::snprintf(label, sizeof label, "expert intermediate  M=%d  (swilu + q8_0)", m);
        bench_pair(q, label,
                   [&] {
                       swilu_fast_kernel(q, dg, du, dout, (int) pairs);
                       k::quantize_q8_0(dout, d80, pairs, &q);
                   },
                   [&] { k::swilu_quantize_q8_0(dg, du, dout, pairs, 1, d80, &q); }, reps);
        sycl::free(dg, q); sycl::free(du, q); sycl::free(dout, q); sycl::free(d80, q); sycl::free(ds, q);
    }

    {
        const long long n = 2560;
        std::vector<float> hx(n);
        std::mt19937 rng(21u);
        std::uniform_real_distribution<float> d(-4.0f, 4.0f);
        for (auto& v : hx) v = d(rng);
        float* dx = dalloc<float>(n, q);
        up(dx, hx, q);
        const size_t nb80 = (size_t) (n / 32), nbK = (size_t) (n / 256);
        uint8_t* o80 = dalloc<uint8_t>(nb80 * 34 + 256, q);
        uint8_t* oK = dalloc<uint8_t>(nbK * 292 + 256, q);
        uint16_t* o16 = dalloc<uint16_t>(n + 8, q);
        bench_pair(q, "layer activation images  (q8_0 + q8_K + bf16)",
                   [&] {
                       k::quantize_q8_0(dx, o80, n, &q);
                       k::quantize_q8_K(dx, oK, n, &q);
                       k::f32_to_bf16_bulk(dx, o16, n, &q);
                   },
                   [&] { k::quantize_act_images(dx, n, o80, oK, o16, &q); }, reps);
        sycl::free(dx, q); sycl::free(o80, q); sycl::free(oK, q); sycl::free(o16, q);
    }

    {
        // (a) gate/up dual GEMV + SwiGLU: one kernel vs gemv(gate) + gemv(up) + swilu, shared-expert shapes
        const long long n_in = 2560, n_out = 640;
        const long long n_groups = n_in / 64;
        std::vector<uint8_t> hcg((size_t) n_out * n_in / 2), hcu((size_t) n_out * n_in / 2);
        std::vector<float> hsg((size_t) n_out * n_groups), hsu((size_t) n_out * n_groups);
        std::vector<float> hog((size_t) n_out * n_groups), hou((size_t) n_out * n_groups);
        std::vector<float> hx(n_in);
        std::mt19937 rng(55u);
        for (auto& v : hcg) v = (uint8_t) rng();
        for (auto& v : hcu) v = (uint8_t) rng();
        std::uniform_real_distribution<float> sc(1e-3f, 3e-2f), xv(-4.0f, 4.0f), ov(-1e-2f, 1e-2f);
        for (auto& v : hsg) v = sc(rng);
        for (auto& v : hsu) v = sc(rng);
        for (auto& v : hog) v = ov(rng);
        for (auto& v : hou) v = ov(rng);
        for (auto& v : hx) v = xv(rng);
        float* dx2 = dalloc<float>(n_in, q);
        up(dx2, hx, q);
        uint8_t* dx80 = dalloc<uint8_t>((size_t) (n_in / 32) * 34 + 256, q);
        k::quantize_q8_0(dx2, dx80, n_in, &q);
        uint8_t *dcg = dalloc<uint8_t>(hcg.size(), q), *dcu = dalloc<uint8_t>(hcu.size(), q);
        float *dsg = dalloc<float>(hsg.size(), q), *dsu = dalloc<float>(hsu.size(), q);
        float *dog = dalloc<float>(hog.size(), q), *dou = dalloc<float>(hou.size(), q);
        up(dcg, hcg, q); up(dcu, hcu, q); up(dsg, hsg, q); up(dsu, hsu, q); up(dog, hog, q); up(dou, hou, q);
        float *d_gate = dalloc<float>(n_out + 8, q), *d_up = dalloc<float>(n_out + 8, q), *d_pair = dalloc<float>(n_out + 8, q);
        const k::SForm fg{4, -8, 64, k::Codebook::Affine, true, 0};
        const k::SForm fu{4, -8, 64, k::Codebook::Affine, true, 0};
        bench_pair(q, "dual GEMV + swilu (gate,up)  FF=640",
                   [&] {
                       gemv_dispatch(fg, dcg, dsg, dog, dx80, nullptr, d_gate, n_in, n_out, &q);
                       gemv_dispatch(fu, dcu, dsu, dou, dx80, nullptr, d_up, n_in, n_out, &q);
                       swilu_legacy_kernel(q, d_gate, d_up, d_pair, (int) n_out);
                   },
                   [&] {
                       k::s_gemv_pair_silu(dx80, nullptr, fg, dcg, dsg, dog, fu, dcu, dsu, dou, d_pair,
                                           n_in, n_out, 32, 0, &q);
                   }, reps);
        sycl::free(dx2, q); sycl::free(dx80, q); sycl::free(dcg, q); sycl::free(dcu, q);
        sycl::free(dsg, q); sycl::free(dsu, q);
        sycl::free(dog, q); sycl::free(dou, q); sycl::free(d_gate, q); sycl::free(d_up, q);
        sycl::free(d_pair, q);
    }

    for (int m : {1, 8}) {
        const long long n_in = 2560, n1 = 48, n2 = 48;
        std::vector<uint16_t> hx(n_in), hw1(n_in * n1), hw2(n_in * n2);
        std::mt19937 rng(31u + (unsigned) m);
        std::uniform_real_distribution<float> d(-2.0f, 2.0f);
        auto put = [&](std::vector<uint16_t>& v) { for (auto& t : v) t = k::bf16_from_f32(d(rng)); };
        put(hx); put(hw1); put(hw2);
        uint16_t *dx = dalloc<uint16_t>(n_in, q), *dw1 = dalloc<uint16_t>(n_in * n1, q), *dw2 = dalloc<uint16_t>(n_in * n2, q);
        up(dx, hx, q); up(dw1, hw1, q); up(dw2, hw2, q);
        float *dy1 = dalloc<float>(n1 + 8, q), *dy2 = dalloc<float>(n2 + 8, q);
        char label[96];
        std::snprintf(label, sizeof label, "bf16 pair (alpha,beta)   M=%d", m);
        bench_pair(q, label,
                   [&] {
                       k::bf16_gemv_split(dx, dw1, dy1, n_in, n1, 32, &q);
                       k::bf16_gemv_split(dx, dw2, dy2, n_in, n2, 32, &q);
                   },
                   [&] { k::bf16_gemv_pair(dx, dw1, dy1, n1, dw2, dy2, n2, n_in, &q); }, reps);
        sycl::free(dx, q); sycl::free(dw1, q); sycl::free(dw2, q); sycl::free(dy1, q);
        sycl::free(dy2, q);
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool bench = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--bench") bench = true;
        else { std::fprintf(stderr, "usage: fusions_parity [--bench]\n"); return 2; }

    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};

    std::printf("fusions_parity: V100 fusions vs their unfused pipelines, bitwise\n");
    check_swilu_quantize(q, 1, false, "swilu_quantize_q8_0 (fast silu, expert path)");
    check_swilu_quantize(q, 1, true, "swilu_quantize_q8_0_scaled (fast silu, expert path)");
    check_swilu_quantize(q, 0, false, "swilu_quantize_q8_0 (legacy silu, shared expert)");
    check_swilu_quantize_q8_K(q);
    check_images(q);
    check_pair(q);
    check_dual_gemv(q, 0, "s_gemv_pair_silu (legacy silu) vs gemv+gemv+swilu");
    check_dual_gemv(q, 1, "s_gemv_pair_silu (fast silu) vs gemv+gemv+swilu");
    check_dual_gemv(q, 2, "s_gemv_pair_silu (native silu) vs gemv+gemv+swilu");
    if (bench) run_bench(q);
    std::printf(g_fail ? "fusions_parity: %d failures\n" : "fusions_parity: 0 failures\n", g_fail);
    return g_fail ? 1 : 0;
}
