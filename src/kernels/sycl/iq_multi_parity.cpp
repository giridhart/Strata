// src/kernels/iq_multi_parity.cpp - the IQ kernels that decode each weight part once for every
// column / entry, against the older kernels that decode it again per column.  Synthetic blocks, no model.
//
//     build/iq_multi_parity            the checks (a ctest; needs the GPU)
//     build/iq_multi_parity --bench    the checks, then old/new timings of both paths
//
// (1) iq_mmvq, every format it takes, ncols 1..8 and 11 (the > 8 split): the new output must be BITWISE equal to
//     the old one (iq_set_old_kernels), and within float rounding of a double reference built from the GPU's own
//     dequantizer (validated against gguf-py by iq_parity) and the dequantized q8_1 activations - a gross-error
//     guard, not the contract.
// (2) native_expert_grouped, every gate/up format with the down formats: groups of 0..11 entries (more than a pass
//     of GRP_NC), unused grid rows (cap_groups > groups), scattered destinations; bitwise, new vs old, the rows it
//     must not write included.
//
// Random bytes are valid codes for every format here (all grid indices are in range); only the fp16 block scales
// are set, small enough that the grouped path's SwiGLU output keeps a finite fp16 q8_1 scale.
#include <sycl/sycl.hpp>
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_fail = 0;

inline sycl::queue& get_queue() {
    static sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    return q;
}

#define ck(expr, label)                                          \
    do {                                                         \
        try {                                                    \
            (expr);                                              \
        } catch (const std::exception& e) {                      \
            std::fprintf(stderr, "%s: %s\n", (label), e.what()); \
        }                                                        \
    } while (0)

template <typename T>
T* dalloc(size_t n) {
    sycl::queue& q_ct1 = get_queue();
    T* p = (T*)sycl::malloc_device(n * sizeof(T) + 256, q_ct1);
    q_ct1.memset(p, 0, n * sizeof(T) + 256).wait();
    return p;
}

const char* name_of(int t) {
    switch (t) {
        case 16:
            return "IQ2_XXS";
        case 17:
            return "IQ2_XS";
        case 18:
            return "IQ3_XXS";
        case 20:
            return "IQ4_NL";
        case 21:
            return "IQ3_S";
        case 22:
            return "IQ2_S";
        case 23:
            return "IQ4_XS";
        case 29:
            return "IQ1_M";
        case 40:
            return "NVFP4";
        case 42:
            return "Q2_0";
        default:
            return "?";
    }
}
int block_values(int t) { return t == 20 ? 32 : (t == 42 || t == 40) ? 64 : 256; }

// `rows` rows of `n` values of format t: random bytes, then a finite fp16 scale in every block
std::vector<uint8_t> random_rows(int t, int64_t rows, int64_t n, std::mt19937& rng) {
    const size_t rb = k::iq_row_bytes(t, n), bs = k::iq_row_bytes(t, block_values(t));
    std::vector<uint8_t> w((size_t)rows * rb);
    std::uniform_int_distribution<int> byte(0, 255), ex(2, 8), man(0, 1023), sgn(0, 3);
    for (auto& b : w)
        b = (uint8_t)byte(rng);
    for (size_t o = 0; o < w.size(); o += bs) {
        if (t == 29) {
            // IQ1_M: the fp16 scale is the top nibbles of the four scale words (bytes 48..55); nibble 3 carries the
            // sign and the exponent's top bits: 0x1 / 0x2 (or 0x9 / 0xA) keeps it in 2^-11 .. 2^-3
            const uint8_t nib = (uint8_t)((sgn(rng) == 0 ? 0x8 : 0x0) | (1 + (byte(rng) & 1)));
            w[o + 55] = (uint8_t)((w[o + 55] & 0x0F) | (nib << 4));
        } else if (t == 40) {
            // NVFP4: 4 bytes of UE4M3 scales in bytes 0..3 of each 36-byte block
            std::uniform_int_distribution<int> uexp(4, 10), uman(0, 7);
            for (int s = 0; s < 4; ++s) {
                w[o + s] = (uint8_t)((uexp(rng) << 3) | uman(rng));
            }
        } else {
            const uint16_t h = (uint16_t)((sgn(rng) == 0 ? 0x8000 : 0) | (ex(rng) << 10) | man(rng));  // 2^-13 .. 2^-6
            std::memcpy(&w[o], &h, 2);
        }
    }
    return w;
}

std::vector<float> random_x(size_t n, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(n);
    for (auto& v : x)
        v = nd(rng);
    return x;
}

// q8_1 blocks (fp16 scale, fp16 sum, 32 int8) back to floats
std::vector<double> dequant_q8_1(const std::vector<uint8_t>& q, size_t n) {
    std::vector<double> v(n);
    for (size_t b = 0; b < n / 32; ++b) {
        uint16_t dh;
        std::memcpy(&dh, &q[b * 36], 2);
        const double d = k::f32_from_f16(dh);
        for (int i = 0; i < 32; ++i)
            v[b * 32 + i] = d * (int8_t)q[b * 36 + 4 + i];
    }
    return v;
}

// ------------------------------------------------------------------------------------------------ (1) iq_mmvq
void check_mmvq(int t, int n_in, int n_out, sycl::queue* s, std::mt19937& rng) {
    sycl::queue& q_ct1 = *s;
    const auto w = random_rows(t, n_out, n_in, rng);
    uint8_t* dw = dalloc<uint8_t>(w.size());
    ck((q_ct1.memcpy(dw, w.data(), w.size()).wait()), "w");
    // the double reference's weights: the GPU dequantizer (iq_parity checks it against gguf-py)
    float* dwf = dalloc<float>((size_t)n_out * n_in);
    k::iq_dequant_f32(t, dw, (int64_t)n_out * n_in, dwf, s);
    std::vector<float> wf((size_t)n_out * n_in);
    ck((q_ct1.memcpy(wf.data(), dwf, wf.size() * 4).wait()), "wf");
    const int max_cols = 11;
    const auto x = random_x((size_t)max_cols * n_in, rng);
    float* dx = dalloc<float>(x.size());
    ck((q_ct1.memcpy(dx, x.data(), x.size() * 4).wait()), "x");
    const size_t xq_bytes = (size_t)max_cols * (n_in / 32) * 36;
    uint8_t* dxq = dalloc<uint8_t>(xq_bytes);
    k::quantize_q8_1_rows(dx, max_cols, n_in, dxq, s);
    std::vector<uint8_t> xq(xq_bytes);
    ck((q_ct1.memcpy(xq.data(), dxq, xq_bytes).wait()), "xq");
    const auto xd = dequant_q8_1(xq, (size_t)max_cols * n_in);
    float* dy_old = dalloc<float>((size_t)max_cols * n_out);
    float* dy_new = dalloc<float>((size_t)max_cols * n_out);
    std::vector<float> y_old((size_t)max_cols * n_out), y_new(y_old.size());
    const int cols[] = {1, 2, 3, 4, 5, 6, 7, 8, 11};
    int bad = 0;
    double worst = 0.0;
    for (int nc : cols) {
        ck((q_ct1.memset(dy_old, 0xFF, y_old.size() * 4).wait()), "memset");
        ck((q_ct1.memset(dy_new, 0xFF, y_new.size() * 4).wait()), "memset");
        k::iq_set_old_kernels(true);
        k::iq_mmvq(t, dw, dxq, dy_old, n_in, n_out, nc, s);
        k::iq_set_old_kernels(false);
        k::iq_mmvq(t, dw, dxq, dy_new, n_in, n_out, nc, s);
        ck((s->wait()), "sync");
        ck((q_ct1.memcpy(y_old.data(), dy_old, y_old.size() * 4).wait()), "y_old");
        ck((q_ct1.memcpy(y_new.data(), dy_new, y_new.size() * 4).wait()), "y_new");
        size_t diff = 0;
        for (size_t i = 0; i < y_old.size(); ++i)
            diff += std::memcmp(&y_old[i], &y_new[i], 4) != 0;
        double num = 0, den = 0;
        bool finite = true;
        for (int c = 0; c < nc; ++c)
            for (int r = 0; r < n_out; ++r) {
                double acc = 0;
                for (int i = 0; i < n_in; ++i)
                    acc += (double)wf[(size_t)r * n_in + i] * xd[(size_t)c * n_in + i];
                const float g = y_new[(size_t)c * n_out + r];
                finite = finite && std::isfinite(g);
                num += std::fabs(g - acc);
                den += std::fabs(acc);
            }
        const double rel = num / (den + 1e-30);
        worst = std::max(worst, rel);
        if (diff != 0 || !finite || !(rel < 1e-2)) {
            std::printf("  %-8s %5d x %5d  ncols %2d: %zu outputs differ from the old kernel, ref rel %.2e%s  FAIL\n",
                        name_of(t), n_out, n_in, nc, diff, rel, finite ? "" : ", non-finite");
            ++bad;
        }
    }
    std::printf("%-8s type %2d  %5d x %5d  iq_mmvq ncols 1..8, 11: %s (ref rel <= %.2e)\n", name_of(t), t, n_out, n_in,
                bad ? "FAIL" : "bitwise equal to the old kernel", worst);
    g_fail += bad;
    sycl::free(dw, q_ct1);
    sycl::free(dwf, q_ct1);
    sycl::free(dx, q_ct1);
    sycl::free(dxq, q_ct1);
    sycl::free(dy_old, q_ct1);
    sycl::free(dy_new, q_ct1);
}

// ---------------------------------------------------------------------------------- (2) native_expert_grouped
struct Grouped {
    k::NativeExpertLayout L;
    int n_groups = 0, n_ent = 0, cap_groups = 0, cap_ent = 0, n_tok = 0;
    std::vector<uint8_t*> blobs;
    unsigned long long* dptr = nullptr;
    int32_t *dstart = nullptr, *dn = nullptr, *ddst = nullptr, *dtok = nullptr;
    float* dx = nullptr;
    uint8_t *dxq = nullptr, *dscr = nullptr;
    float* dout = nullptr;
    size_t out_floats = 0;

    Grouped(int gu, int dt, int64_t H, int64_t FF, const std::vector<int>& counts, int tokens, std::mt19937& rng) {
        sycl::queue& q_ct1 = get_queue();
        L = k::native_expert_layout(gu, dt, H, FF);
        n_groups = (int)counts.size();
        cap_groups = n_groups + 2;  // the extra grid rows must exit
        n_tok = tokens;
        std::vector<int32_t> start(1, 0), tok, dst;
        for (int c : counts)
            start.push_back(start.back() + c);
        n_ent = start.back();
        cap_ent = n_ent + 3;
        std::uniform_int_distribution<int> tk(0, tokens - 1);
        for (int e = 0; e < n_ent; ++e)
            tok.push_back(tk(rng));
        dst.resize(n_ent);
        std::iota(dst.begin(), dst.end(), 0);
        std::shuffle(dst.begin(), dst.end(), rng);
        std::vector<unsigned long long> ptr;
        for (int g = 0; g < n_groups; ++g) {
            // one blob per group: gate rows | up rows (format gu, H values) | down rows (format dt, FF values)
            auto b = random_rows(gu, 2 * FF, H, rng);
            const auto d = random_rows(dt, H, FF, rng);
            b.insert(b.end(), d.begin(), d.end());
            if (L.tail_off) {
                const float tail[4] = {0.85f, 1.15f, 0.95f, 0.0f};
                const auto* tp = reinterpret_cast<const uint8_t*>(tail);
                b.insert(b.end(), tp, tp + 16);
            }
            if (b.size() != L.bytes) {
                std::fprintf(stderr, "blob size %zu != %zu\n", b.size(), L.bytes);
                std::exit(2);
            }
            uint8_t* db = dalloc<uint8_t>(b.size());
            ck((q_ct1.memcpy(db, b.data(), b.size()).wait()), "blob");
            blobs.push_back(db);
            ptr.push_back((unsigned long long)db);
        }
        dptr = dalloc<unsigned long long>(cap_groups);
        dstart = dalloc<int32_t>(cap_groups + 1);
        dn = dalloc<int32_t>(1);
        ddst = dalloc<int32_t>(cap_ent);
        dtok = dalloc<int32_t>(cap_ent);
        ck((q_ct1.memcpy(dptr, ptr.data(), ptr.size() * 8).wait()), "ptr");
        ck((q_ct1.memcpy(dstart, start.data(), start.size() * 4).wait()), "start");
        ck((q_ct1.memcpy(dn, &n_groups, 4).wait()), "n");
        if (n_ent) {
            ck((q_ct1.memcpy(ddst, dst.data(), dst.size() * 4).wait()), "dst");
            ck((q_ct1.memcpy(dtok, tok.data(), tok.size() * 4).wait()), "tok");
        }
        const auto x = random_x((size_t)tokens * H, rng);
        dx = dalloc<float>(x.size());
        ck((q_ct1.memcpy(dx, x.data(), x.size() * 4).wait()), "x");
        dxq = dalloc<uint8_t>((size_t)tokens * (H / 32) * 36);
        dscr = dalloc<uint8_t>(k::native_expert_scratch_bytes(cap_ent, FF));
        out_floats = (size_t)cap_ent * H;
        dout = dalloc<float>(out_floats);
    }
    ~Grouped() {
        sycl::queue& q_ct1 = get_queue();
        for (auto* b : blobs)
            sycl::free(b, q_ct1);
        sycl::free(dptr, q_ct1);
        sycl::free(dstart, q_ct1);
        sycl::free(dn, q_ct1);
        sycl::free(ddst, q_ct1);
        sycl::free(dtok, q_ct1);
        sycl::free(dx, q_ct1);
        sycl::free(dxq, q_ct1);
        sycl::free(dscr, q_ct1);
        sycl::free(dout, q_ct1);
    }
    void run(bool old, sycl::queue* s) {
        k::iq_set_old_kernels(old);
        k::quantize_q8_1_rows(dx, n_tok, L.n_embd, dxq, s);
        k::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, cap_groups, cap_ent, dxq, dscr, dout, s);
        k::iq_set_old_kernels(false);
    }
    std::vector<float> result(bool old, sycl::queue* s) {
        sycl::queue& q_ct1 = *s;
        ck((q_ct1.memset(dout, 0xFF, out_floats * 4).wait()),
           "memset");  // NaN rows: must stay NaN where nothing writes
        ck((q_ct1.memset(dscr, 0, k::native_expert_scratch_bytes(cap_ent, L.n_ff)).wait()), "memset");
        run(old, s);
        ck((s->wait()), "sync");
        std::vector<float> o(out_floats);
        ck((q_ct1.memcpy(o.data(), dout, o.size() * 4).wait()), "out");
        return o;
    }
};

void check_grouped(int gu, int dt, int64_t H, int64_t FF, sycl::queue* s, std::mt19937& rng) {
    // 0..11 entries per group: one pass, a partial pass, two passes and three passes of GRP_NC = 4
    Grouped G(gu, dt, H, FF, {1, 3, 4, 5, 0, 8, 2, 11, 1}, 8, rng);
    const auto a = G.result(true, s), b = G.result(false, s);
    size_t diff = 0;
    [[maybe_unused]] size_t written = 0;
    bool finite = true;
    for (size_t i = 0; i < a.size(); ++i) {
        diff += std::memcmp(&a[i], &b[i], 4) != 0;
        if (i < (size_t)G.n_ent * H) {
            ++written;
            finite = finite && std::isfinite(b[i]);
        }
    }
    const bool ok = diff == 0 && finite;
    std::printf("%-8s/%-7s %5lld x %4lld  native_expert_grouped, %d groups, %d entries: %s\n", name_of(gu), name_of(dt),
                (long long)H, (long long)FF, G.n_groups, G.n_ent, ok ? "bitwise equal to the old kernels" : "FAIL");
    if (!ok) {
        std::printf("  %zu of %zu floats differ%s\n", diff, a.size(), finite ? "" : ", non-finite outputs");
        ++g_fail;
    }
}

// ------------------------------------------------------------------------------------------------ --bench
template <typename F>
float time_ms(sycl::queue* s, int it, const F& fn) {
    fn();
    s->wait();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < it; ++i)
        fn();
    s->wait();
    auto end = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float, std::milli>(end - start).count();
    return ms / (float)it;
}

void bench(sycl::queue* s, std::mt19937& rng) {
    sycl::queue& q_ct1 = *s;
    std::printf("\n--bench: microseconds per call, old (per-column decode) / new (decode once), idle GPU assumed\n");
    const int n_in = 2560, n_out = 8192, it = 200;
    for (int t : {16, 17, 18, 21, 22, 29}) {
        const auto w = random_rows(t, n_out, n_in, rng);
        uint8_t* dw = dalloc<uint8_t>(w.size());
        ck((q_ct1.memcpy(dw, w.data(), w.size()).wait()), "w");
        const auto x = random_x((size_t)8 * n_in, rng);
        float* dx = dalloc<float>(x.size());
        ck((q_ct1.memcpy(dx, x.data(), x.size() * 4).wait()), "x");
        uint8_t* dxq = dalloc<uint8_t>((size_t)8 * (n_in / 32) * 36);
        k::quantize_q8_1_rows(dx, 8, n_in, dxq, s);
        float* dy = dalloc<float>((size_t)8 * n_out);
        std::printf("%-8s iq_mmvq %d x %d:", name_of(t), n_out, n_in);
        for (int nc = 1; nc <= 8; ++nc) {
            float us[2];
            for (int old = 1; old >= 0; --old) {
                k::iq_set_old_kernels(old != 0);
                us[old] = 1e3f * time_ms(s, it, [&] { k::iq_mmvq(t, dw, dxq, dy, n_in, n_out, nc, s); });
            }
            std::printf("  %d: %.1f/%.1f", nc, us[1], us[0]);
        }
        std::printf("\n");
        k::iq_set_old_kernels(false);
        sycl::free(dw, q_ct1);
        sycl::free(dx, q_ct1);
        sycl::free(dxq, q_ct1);
        sycl::free(dy, q_ct1);
    }
    // the verify window's shape: 2560 x 640 experts, 16 groups of m entries each
    for (int gu : {16, 17, 18, 21, 22, 29}) {
        std::printf("%-8s/Q2_0 grouped, 16 groups of m entries:", name_of(gu));
        for (int m : {1, 2, 3, 4, 8}) {
            Grouped G(gu, 42, 2560, 640, std::vector<int>(16, m), 8, rng);
            float us[2];
            for (int old = 1; old >= 0; --old)
                us[old] = 1e3f * time_ms(s, 50, [&] { G.run(old != 0, s); });
            std::printf("  m=%d: %.1f/%.1f", m, us[1], us[0]);
        }
        std::printf("\n");
    }
}

}  // namespace

int main(int argc, char** argv) {
    sycl::queue q_ct1{sycl::default_selector_v, sycl::property::queue::in_order()};
    const bool do_bench = argc > 1 && std::string(argv[1]) == "--bench";
    sycl::queue* s = &q_ct1;
    std::mt19937 rng(18);
    for (int t : {16, 17, 18, 20, 21, 22, 23, 29, 40, 42}) {
        check_mmvq(t, 2560, 67, s, rng);  // the model's n_embd; 67 rows: a partial block of 4 rows
        check_mmvq(t, 1024, 5, s, rng);
    }
    for (int gu : {16, 17, 18, 21, 22, 23, 29, 40, 42}) {
        for (int dt : {20, 40, 42})
            check_grouped(gu, dt, 2560, 640, s, rng);  // the model's shape
        check_grouped(gu, 23, 1024, 512, s, rng);      // IQ4_XS down needs n_ff % 256 == 0
    }
    if (do_bench) bench(s, rng);
    std::printf("iq_multi_parity: %d failures\n", g_fail);

    return g_fail ? 1 : 0;
}
