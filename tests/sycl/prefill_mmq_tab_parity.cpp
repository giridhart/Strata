// tests/sycl/prefill_mmq_tab_parity.cpp - STRATA_MMQ_DIRECT for SYCL 2020:
// Direct pointer-table MMQ (Product::tab) bit for bit exact against gathered MMQ
// on Intel Arc Xe2 (Battlemage) GPUs.
//
// Part 1: Per product type (IQ3_S gate/up [1280, 2560], IQ4_NL down [2560, 640],
//         Q8_0 down [2560, 640], Q2_0, IQ4_XS, etc.):
//         Random valid matrices at scattered aligned addresses in a device arena,
//         each followed by a guard filled with fp16 NaN (0xFF 0x7C).
//         prep_group zeroes over-read bytes, then table launch runs.
//         Assert 100% bitwise exact equality (memcmp == 0) against gathered launch.
//         Sweep: 1..16 experts with varying row counts {0, 1, 7, 8, 31, 64, 100, 512, 1000}.
// Part 2: Negative control: same table launch with guards NOT zeroed must yield
//         non-finite values (NaN / Inf) for formats that over-read (IQ4_NL: 72 B, Q8_0: 136 B).
// Part 3: Mixed direct + gathered groups as prefill.cpp builds them:
//         Native expert blobs for IQ3_S + IQ4_NL and IQ3_S + Q8_0, some direct,
//         some gathered at compact indices c != q, including all-direct, all-gathered,
//         and interleaved groups. Both products bitwise exact (memcmp == 0) to all-gathered.

#include <sycl/sycl.hpp>
#include "strata/prefill/moe_mmq.hpp"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace mmq = strata::prefill::mmq;

namespace {

constexpr int G = mmq::kGatherGroupMax;
constexpr size_t TAIL = 4096;
constexpr int64_t kIdsPad = 2048 + 128;

const char* type_name(int t) {
    switch (t) {
        case GGML_TYPE_Q8_0: return "Q8_0";
        case GGML_TYPE_IQ4_NL: return "IQ4_NL";
        case GGML_TYPE_IQ3_S: return "IQ3_S";
        case GGML_TYPE_Q2_0: return "Q2_0";
        case GGML_TYPE_IQ2_XXS: return "IQ2_XXS";
        case GGML_TYPE_IQ2_XS: return "IQ2_XS";
        case GGML_TYPE_IQ2_S: return "IQ2_S";
        case GGML_TYPE_IQ3_XXS: return "IQ3_XXS";
        case GGML_TYPE_IQ4_XS: return "IQ4_XS";
        default: return "UNKNOWN";
    }
}

inline int64_t blck_size(int t) {
    switch (t) {
        case GGML_TYPE_Q8_0: return 32;
        case GGML_TYPE_IQ4_NL: return 32;
        case GGML_TYPE_IQ3_S: return 256;
        case GGML_TYPE_Q2_0: return 64;
        case GGML_TYPE_IQ2_XXS: return 256;
        case GGML_TYPE_IQ2_XS: return 256;
        case GGML_TYPE_IQ2_S: return 256;
        case GGML_TYPE_IQ3_XXS: return 256;
        case GGML_TYPE_IQ4_XS: return 256;
        default: return 32;
    }
}

inline uint16_t fp32_to_fp16(float f) {
    sycl::half h = sycl::half(f);
    return sycl::bit_cast<uint16_t>(h);
}

struct Dev {
    sycl::queue* q = nullptr;
    void* p = nullptr;
    size_t n = 0;

    Dev(size_t bytes, sycl::queue& queue) : q(&queue), n(bytes) {
        p = sycl::malloc_device(bytes > 0 ? bytes : 1, *q);
        if (!p) throw std::runtime_error("malloc_device failed");
    }
    ~Dev() {
        if (p && q) sycl::free(p, *q);
    }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    uint8_t* b() const { return (uint8_t*)p; }
};

void upload(void* dst, const void* src, size_t n, sycl::queue& q) {
    q.memcpy(dst, src, n).wait();
}

void fill_nan(uint8_t* dev, size_t n, sycl::queue& q) {
    std::vector<uint8_t> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = (i & 1) ? 0x7C : 0xFF;
    upload(dev, h.data(), n, q);
}

void fill_matrix(uint8_t* m, int t, int64_t rows, int64_t cols, std::mt19937& rng) {
    const size_t total_bytes = mmq::matrix_bytes(t, rows, cols);
    const int64_t qk = blck_size(t);
    const size_t nb = (size_t)rows * (size_t)(cols / qk);
    const size_t bs = total_bytes / nb;

    std::uniform_int_distribution<int> byte_dist(0, 255);
    const float lo = (t == GGML_TYPE_Q8_0) ? 0.0001f :
                     (t == GGML_TYPE_IQ3_S) ? 0.0004f :
                     (t == GGML_TYPE_IQ4_XS) ? 0.00005f :
                     (t == GGML_TYPE_IQ4_NL) ? 0.0002f :
                     (t == GGML_TYPE_Q2_0) ? 0.004f : 0.0008f;
    std::uniform_real_distribution<float> sc(lo, 6.0f * lo);

    for (size_t i = 0; i < total_bytes; ++i) m[i] = (uint8_t)byte_dist(rng);
    for (size_t b = 0; b < nb; ++b) {
        const uint16_t h = fp32_to_fp16(sc(rng));
        std::memcpy(m + b * bs, &h, 2);
    }
}

struct Shape {
    int t;
    int64_t rows, cols;
    const char* role;
};

struct Acts {
    Dev x, xq, ids, bounds;
    int64_t total;

    Acts(int t, int64_t cols, const std::vector<int>& cnt, std::mt19937& rng, sycl::queue& q)
        : x((size_t)std::max<int64_t>(1, sum(cnt)) * cols * 4, q),
          xq(mmq::q8_bytes(std::max<int64_t>(1, sum(cnt)), cols), q),
          ids((size_t)(sum(cnt) + kIdsPad) * 4, q),
          bounds((cnt.size() + 1) * 4, q),
          total(sum(cnt)) {
        std::vector<float> hx((size_t)std::max<int64_t>(1, total) * cols);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& v : hx) v = nd(rng);
        upload(x.p, hx.data(), hx.size() * 4, q);

        std::vector<int32_t> hb(cnt.size() + 1, 0);
        for (size_t i = 0; i < cnt.size(); ++i) hb[i + 1] = hb[i] + cnt[i];
        upload(bounds.p, hb.data(), hb.size() * 4, q);

        mmq::iota((int32_t*)ids.p, total + kIdsPad, &q);
        if (total > 0) mmq::quantize((const float*)x.p, nullptr, xq.p, t, cols, cols, total, &q);
        q.wait();
    }

    static int64_t sum(const std::vector<int>& c) {
        int64_t n = 0;
        for (int v : c) n += v;
        return n;
    }
};

mmq::Product make_product(const Shape& sh, const Acts& a, int n, int64_t max_rows, float* dst) {
    mmq::Product p;
    p.type = sh.t;
    p.w_rows = sh.rows;
    p.w_cols = sh.cols;
    p.n = n;
    p.xq = a.xq.p;
    p.bounds = (const int32_t*)a.bounds.p;
    p.ids = (const int32_t*)a.ids.p;
    p.total_rows = a.total;
    p.max_rows = max_rows;
    p.dst = dst;
    p.ld_dst = sh.rows;
    return p;
}

int g_cases = 0, g_fail = 0;

bool same_bits(const Dev& a, const Dev& b, size_t n, sycl::queue& q, const char* what) {
    std::vector<uint8_t> ha(n), hb(n);
    q.memcpy(ha.data(), a.p, n).wait();
    q.memcpy(hb.data(), b.p, n).wait();
    ++g_cases;
    if (std::memcmp(ha.data(), hb.data(), n) == 0) return true;

    size_t i = 0;
    while (i < n && ha[i] == hb[i]) ++i;
    std::printf("  FAIL %s: first differing byte %zu of %zu\n", what, i, n);
    ++g_fail;
    return false;
}

bool any_nonfinite(const Dev& d, size_t floats, sycl::queue& q) {
    std::vector<float> h(floats);
    q.memcpy(h.data(), d.p, floats * 4).wait();
    for (float v : h) {
        if (!std::isfinite(v)) return true;
    }
    return false;
}

void product_case(mmq::Context& ctx, const Shape& sh, const std::vector<int>& cnt, int64_t max_rows,
                  std::mt19937& rng, sycl::queue& q, bool negative, const char* label) {
    const int n = (int)cnt.size();
    const size_t eb = mmq::matrix_bytes(sh.t, sh.rows, sh.cols);
    const size_t over = mmq::overread_bytes(sh.t, sh.cols);
    Acts a(sh.t, sh.cols, cnt, rng, q);

    std::vector<std::vector<uint8_t>> w((size_t)n, std::vector<uint8_t>(eb));
    for (auto& m : w) fill_matrix(m.data(), sh.t, sh.rows, sh.cols, rng);

    // Gathered buffer: back-to-back, zeroed tail, space after it NaN-patterned
    Dev grp((size_t)n * eb + TAIL + 4096, q);
    fill_nan(grp.b(), grp.n, q);
    for (int z = 0; z < n; ++z) upload(grp.b() + (size_t)z * eb, w[(size_t)z].data(), eb, q);
    q.memset(grp.b() + (size_t)n * eb, 0, TAIL).wait();

    // Table buffer: scattered 256-aligned slots in one arena, each with NaN guard
    const size_t stride = (eb + mmq::kSlotGuard + 255) / 256 * 256;
    std::vector<size_t> pos((size_t)n);
    for (int z = 0; z < n; ++z) pos[(size_t)z] = (size_t)z;
    std::shuffle(pos.begin(), pos.end(), rng);
    std::uniform_int_distribution<int> gap(0, 8);
    Dev arena((size_t)n * (stride + 8 * 256) + 4096, q);
    fill_nan(arena.b(), arena.n, q);

    mmq::WeightTab tab;
    mmq::GuardTab guards;
    size_t at = 0;
    std::vector<size_t> where((size_t)n);
    for (int k = 0; k < n; ++k) {
        at += (size_t)gap(rng) * 256;
        where[pos[(size_t)k]] = at;
        at += stride;
    }
    for (int z = 0; z < n; ++z) {
        uint8_t* p = arena.b() + where[(size_t)z];
        upload(p, w[(size_t)z].data(), eb, q);
        tab.p[z] = p;
        guards.p[guards.n++] = p + eb;
    }

    const size_t out = (size_t)std::max<int64_t>(1, a.total) * (size_t)sh.rows * 4;
    Dev ref(out, q), got(out, q);
    fill_nan(ref.b(), out, q);
    fill_nan(got.b(), out, q);

    // 1. Gathered run
    mmq::Product pr = make_product(sh, a, n, max_rows, (float*)ref.p);
    pr.w = grp.p;
    pr.expert_bytes = eb;
    ctx.run(pr, &q);

    // 2. Direct table run
    if (!negative) mmq::prep_group(guards, over, nullptr, nullptr, 0, &q);
    mmq::Product pt = make_product(sh, a, n, max_rows, (float*)got.p);
    pt.w = nullptr;
    pt.expert_bytes = 0;
    pt.tab = &tab;
    ctx.run(pt, &q);
    q.wait();

    char what[256];
    std::snprintf(what, sizeof what, "%s %s n=%d rows=%lld max_rows=%lld (%s)", type_name(sh.t), sh.role, n,
                  (long long)a.total, (long long)max_rows, label);

    if (!negative) {
        same_bits(ref, got, out, q, what);
        return;
    }
    ++g_cases;
    if (!any_nonfinite(got, out / 4, q)) {
        std::printf("  FAIL negative control %s: no non-finite value with guards left as NaN\n", what);
        ++g_fail;
    } else if (any_nonfinite(ref, (size_t)a.total * (size_t)sh.rows, q)) {
        std::printf("  FAIL negative control %s: gathered reference is not finite\n", what);
        ++g_fail;
    }
}

std::vector<int> random_counts(int n, std::mt19937& rng) {
    static const int kRows[] = {0, 1, 7, 8, 9, 31, 64, 100, 512, 1000};
    std::uniform_int_distribution<int> pick(0, 9);
    std::vector<int> c((size_t)n);
    for (auto& v : c) v = kRows[pick(rng)];
    return c;
}

int64_t max_of(const std::vector<int>& c) {
    int64_t m = 0;
    for (int v : c) m = std::max<int64_t>(m, v);
    return m;
}

void sweep(mmq::Context& ctx, const Shape& sh, bool full, std::mt19937& rng, sycl::queue& q) {
    const int f0 = g_fail, c0 = g_cases;
    // 1..16 experts, mixed counts
    for (int n = 1; n <= G; ++n) {
        if (!full && n != 1 && n != 3 && n != 8 && n != G) continue;
        for (int rep = 0; rep < (full ? 2 : 1); ++rep) {
            std::vector<int> c = random_counts(n, rng);
            if (max_of(c) == 0) c[0] = 1;
            product_case(ctx, sh, c, max_of(c), rng, q, false, "mixed");
        }
    }
    // Specific J values
    static const int kMax[] = {1, 8, 16, 24, 31, 32, 64, 100, 128, 512, 1000};
    for (int m : kMax) {
        for (int n : {1, 4, 16}) {
            if (!full && n == 4) continue;
            std::uniform_int_distribution<int> fewer(0, m);
            std::vector<int> c((size_t)n);
            for (auto& v : c) v = fewer(rng);
            c[(size_t)(rng() % (unsigned)n)] = m;
            product_case(ctx, sh, c, m, rng, q, false, "J sweep");
        }
    }
    // Empty experts
    product_case(ctx, sh, {0, 0, 5, 0, 100, 0, 0}, 100, rng, q, false, "empty experts");
    product_case(ctx, sh, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}, 1, rng, q, false, "only last expert");

    std::printf("%-8s %-7s %4d cases, %s\n", type_name(sh.t), sh.role, g_cases - c0,
                g_fail == f0 ? "all bitwise identical (memcmp == 0)" : "FAILURES");

    // Part 2: Negative control for types with over-read
    if (mmq::overread_bytes(sh.t, sh.cols) > 0) {
        const int f1 = g_fail;
        product_case(ctx, sh, {3, 64, 0, 17}, 64, rng, q, true, "negative control");
        product_case(ctx, sh, {200}, 200, rng, q, true, "negative control");
        std::printf("%-8s %-7s negative control (guards unzeroed): %s\n", type_name(sh.t), sh.role,
                    g_fail == f1 ? "non-finite output as required" : "FAILED");
    }
}

// Part 3: Mixed groups
struct Pair {
    int gu, d;
};

void group_case(mmq::Context& ctx, const Pair& pr, const std::vector<int>& cnt, const std::vector<char>& direct,
                std::mt19937& rng, sycl::queue& q, const char* label) {
    const int n = (int)cnt.size();
    const Shape gs{pr.gu, 1280, 2560, "gate/up"}, ds{pr.d, 2560, 640, "down"};
    const size_t gub = mmq::matrix_bytes(pr.gu, 1280, 2560), db = mmq::matrix_bytes(pr.d, 2560, 640);
    const size_t up_off = gub / 2, down_off = gub, bytes = gub + db;
    const size_t over = mmq::overread_bytes(pr.d, 640);

    std::vector<std::vector<uint8_t>> blob((size_t)n, std::vector<uint8_t>(bytes));
    for (auto& b : blob) {
        fill_matrix(b.data(), pr.gu, 1280, 2560, rng);
        fill_matrix(b.data() + down_off, pr.d, 2560, 640, rng);
    }
    const size_t stride = (bytes + mmq::kSlotGuard + 255) / 256 * 256;
    Dev arena((size_t)n * (stride + 4 * 256) + 4096, q);
    fill_nan(arena.b(), arena.n, q);
    std::vector<uint8_t*> at((size_t)n);
    size_t o = 0;
    for (int z = 0; z < n; ++z) {
        o += (size_t)(rng() % 4) * 256;
        at[(size_t)z] = arena.b() + o;
        o += stride;
        upload(at[(size_t)z], blob[(size_t)z].data(), bytes, q);
    }

    Acts ag(pr.gu, 2560, cnt, rng, q), ad(pr.d, 640, cnt, rng, q);
    const int64_t mr = std::max<int64_t>(1, max_of(cnt));
    const size_t out_g = (size_t)std::max<int64_t>(1, ag.total) * 1280 * 4;
    const size_t out_d = (size_t)std::max<int64_t>(1, ad.total) * 2560 * 4;

    // Reference: all-gathered
    Dev rgu((size_t)G * gub + TAIL, q), rd((size_t)G * db + TAIL, q);
    Dev ref_g(out_g, q), ref_d(out_d, q);
    fill_nan(rgu.b(), rgu.n, q);
    fill_nan(rd.b(), rd.n, q);
    fill_nan(ref_g.b(), out_g, q);
    fill_nan(ref_d.b(), out_d, q);

    for (int qidx = 0; qidx < n; ++qidx) {
        const uint8_t* b = at[(size_t)qidx];
        mmq::gather_native(b, b + up_off, gub / 2, b + down_off, db,
                           rgu.b() + (size_t)qidx * gub, rd.b() + (size_t)qidx * db, &q);
    }
    q.memset(rgu.b() + (size_t)n * gub, 0, TAIL);
    q.memset(rd.b() + (size_t)n * db, 0, TAIL).wait();

    mmq::Product p = make_product(gs, ag, n, mr, (float*)ref_g.p);
    p.w = rgu.p;
    p.expert_bytes = gub;
    ctx.run(p, &q);

    p = make_product(ds, ad, n, mr, (float*)ref_d.p);
    p.w = rd.p;
    p.expert_bytes = db;
    ctx.run(p, &q);

    // Got: direct mode with mixed resident slots & gathered fallback
    Dev tgu_buf((size_t)G * gub + TAIL, q), td_buf((size_t)G * db + TAIL, q);
    Dev got_g(out_g, q), got_d(out_d, q);
    fill_nan(tgu_buf.b(), tgu_buf.n, q);
    fill_nan(td_buf.b(), td_buf.n, q);
    fill_nan(got_g.b(), out_g, q);
    fill_nan(got_d.b(), out_d, q);

    mmq::WeightTab tgu, td;
    mmq::GuardTab gd;
    int ns = 0;

    for (int qidx = 0; qidx < n; ++qidx) {
        uint8_t* b = at[(size_t)qidx];
        if (direct[(size_t)qidx]) {
            tgu.p[qidx] = b;
            td.p[qidx] = b + down_off;
            gd.p[gd.n++] = b + bytes;
        } else {
            const int c = ns++;
            tgu.p[qidx] = tgu_buf.b() + (size_t)c * gub;
            td.p[qidx] = td_buf.b() + (size_t)c * db;
            mmq::gather_native(b, b + up_off, gub / 2, b + down_off, db,
                               tgu_buf.b() + (size_t)c * gub, td_buf.b() + (size_t)c * db, &q);
        }
    }

    mmq::prep_group(gd, over, ns ? tgu_buf.b() + (size_t)ns * gub : nullptr,
                    ns ? td_buf.b() + (size_t)ns * db : nullptr, TAIL, &q);

    p = make_product(gs, ag, n, mr, (float*)got_g.p);
    p.tab = &tgu;
    ctx.run(p, &q);

    p = make_product(ds, ad, n, mr, (float*)got_d.p);
    p.tab = &td;
    ctx.run(p, &q);
    q.wait();

    char what[256];
    std::snprintf(what, sizeof what, "%s+%s group n=%d direct=%d (%s)", type_name(pr.gu), type_name(pr.d), n,
                  gd.n, label);
    std::string w1 = std::string(what) + " gate/up", w2 = std::string(what) + " down";
    same_bits(ref_g, got_g, out_g, q, w1.c_str());
    same_bits(ref_d, got_d, out_d, q, w2.c_str());
}

}  // namespace

int main() {
    try {
        sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
        auto dev = q.get_device();
        std::printf("Running on device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

        if (!mmq::tab_built()) {
            std::printf("SKIP: pointer-table MMQ not built\n");
            return 77;
        }

        std::mt19937 rng(42);
        mmq::Context ctx;

        // Verify over-read calculations
        const Shape main_shapes[] = {
            {GGML_TYPE_IQ3_S, 1280, 2560, "gate/up"},
            {GGML_TYPE_IQ4_NL, 2560, 640, "down"},
            {GGML_TYPE_Q8_0, 2560, 640, "down"}
        };
        const size_t want_over[] = {0, 72, 136};

        for (size_t i = 0; i < 3; ++i) {
            const Shape& sh = main_shapes[i];
            const size_t ov = mmq::overread_bytes(sh.t, sh.cols);
            if (ov != want_over[i]) {
                std::printf("FAIL: %s over-read %zu B (expected %zu B)\n", type_name(sh.t), ov, want_over[i]);
                return 1;
            }
        }

        // Part 1 & Part 2: Main shapes
        for (const Shape& sh : main_shapes) {
            if (!mmq::tab_fits(sh.t, sh.rows)) {
                std::printf("FAIL: tab_fits false for %s %s\n", type_name(sh.t), sh.role);
                return 1;
            }
            std::printf("%-8s %-7s over-read: %zu B, starting sweep...\n", type_name(sh.t), sh.role,
                        mmq::overread_bytes(sh.t, sh.cols));
            sweep(ctx, sh, true, rng, q);
        }

        // Secondary shapes
        const Shape other_shapes[] = {
            {GGML_TYPE_IQ4_NL, 1280, 2560, "gate/up"},
            {GGML_TYPE_Q8_0, 1280, 2560, "gate/up"}
        };
        for (const Shape& sh : other_shapes) {
            sweep(ctx, sh, false, rng, q);
        }

        // Part 3: Mixed groups
        const int f0 = g_fail, c0 = g_cases;
        for (const Pair pr : {Pair{GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL}, Pair{GGML_TYPE_IQ3_S, GGML_TYPE_Q8_0}}) {
            // alternating direct / gathered, last entry direct
            group_case(ctx, pr, {5, 0, 31, 64, 1, 100, 8, 9}, {0, 1, 0, 1, 1, 0, 0, 1}, rng, q, "interleaved, last direct");
            // full group, last gathered
            group_case(ctx, pr, {7, 8, 9, 31, 64, 100, 1, 0, 512, 3, 17, 33, 2, 64, 90, 128},
                       {0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0}, rng, q, "full group, last gathered");
            // all direct & all gathered
            group_case(ctx, pr, {12, 40, 0, 7}, {1, 1, 1, 1}, rng, q, "all direct");
            group_case(ctx, pr, {12, 40, 0, 7}, {0, 0, 0, 0}, rng, q, "all gathered");
            group_case(ctx, pr, {1000}, {1}, rng, q, "one direct entry");

            // random mixes
            for (int r = 0; r < 4; ++r) {
                const int n = 1 + (int)(rng() % G);
                std::vector<int> c = random_counts(n, rng);
                if (max_of(c) == 0) c[0] = 1;
                std::vector<char> d((size_t)n);
                for (auto& v : d) v = (char)(rng() & 1);
                group_case(ctx, pr, c, d, rng, q, "random mix");
            }
        }
        std::printf("Part 3 mixed groups: %d comparisons, %s\n", g_cases - c0,
                    g_fail == f0 ? "all bitwise exact identical" : "FAILURES");

        std::printf("\n========================================\n");
        std::printf("Total tests: %d comparisons, %d failed\n", g_cases, g_fail);
        std::printf("========================================\n");

        return g_fail == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("Exception: %s\n", e.what());
        return 1;
    }
}
