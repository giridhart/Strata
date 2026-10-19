// src/kernels/sycl/qsa_merge_daily_parity.cpp - the Daily's decode attention (STRATA_ATTN_DAILY=1)
// against upstream's, bitwise (GPU, synthetic, no model).
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

template <typename T>
T* dalloc(size_t n, sycl::queue& q) {
    T* p = sycl::malloc_device<T>(n + 256, q);
    q.memset(p, 0, (n + 256) * sizeof(T)).wait();
    return p;
}

template <typename T>
T* upload(const std::vector<T>& h, sycl::queue& q) {
    T* d = dalloc<T>(h.size(), q);
    q.memcpy(d, h.data(), h.size() * sizeof(T)).wait();
    return d;
}

uint16_t f16_value(uint32_t r) { return (uint16_t) ((r & 0x8000u) | ((11u + (r >> 16) % 5u) << 10) | (r & 0x3FFu)); }
uint16_t f16_scale_q8(uint32_t r) { return (uint16_t) ((9u << 10) | (r & 0x3FFu)); }
uint16_t f16_scale_q4(uint32_t r) { return (uint16_t) ((13u << 10) | (r & 0x3FFu)); }

[[maybe_unused]] constexpr int kFmtF16 = 0, kFmtInt8 = 1, kFmtQ4 = 2, kFmtK8V4 = 3;
const char* fmt_name(int f) { return f == kFmtF16 ? "fp16" : f == kFmtInt8 ? "int8" : f == kFmtQ4 ? "q4_0" : "k8v4"; }

struct DevicePools {
    uint16_t *k16 = nullptr, *v16 = nullptr, *k8s = nullptr, *v8s = nullptr;
    int8_t *k8 = nullptr, *v8 = nullptr;
    uint8_t *k4 = nullptr, *v4 = nullptr;
    k::QsaAttnPools view(int fmt, const int32_t* table) const {
        k::QsaAttnPools p;
        p.page_table = table;
        if (fmt == kFmtF16) { p.k_pool = k16; p.v_pool = v16; }
        else if (fmt == kFmtInt8) { p.k_q = k8; p.v_q = v8; p.k_scale = k8s; p.v_scale = v8s; }
        else if (fmt == kFmtQ4) { p.k_q4 = k4; p.v_q4 = v4; }
        else { p.k_q = k8; p.k_scale = k8s; p.v_q4 = v4; }
        return p;
    }
};

DevicePools make_pools(int64_t rows, int64_t D, std::mt19937& rng, sycl::queue& q) {
    DevicePools p;
    {
        std::vector<uint16_t> h((size_t) (rows * D));
        for (auto& x : h) x = f16_value(rng());
        p.k16 = upload(h, q);
        for (auto& x : h) x = f16_value(rng());
        p.v16 = upload(h, q);
    }
    {
        std::vector<int8_t> c((size_t) (rows * D));
        for (auto& x : c) x = (int8_t) ((int) (rng() % 255u) - 127);
        p.k8 = upload(c, q);
        for (auto& x : c) x = (int8_t) ((int) (rng() % 255u) - 127);
        p.v8 = upload(c, q);
        std::vector<uint16_t> sc((size_t) (rows * (D / k::KV_Q8_GROUP)));
        for (auto& x : sc) x = f16_scale_q8(rng());
        p.k8s = upload(sc, q);
        for (auto& x : sc) x = f16_scale_q8(rng());
        p.v8s = upload(sc, q);
    }
    {
        const size_t blocks = (size_t) (rows * (D / k::QK4_0));
        std::vector<uint8_t> b(blocks * sizeof(k::block_q4_0));
        for (int side = 0; side < 2; ++side) {
            for (size_t i = 0; i < blocks; ++i) {
                k::block_q4_0 blk;
                blk.d = f16_scale_q4(rng());
                for (auto& qb : blk.qs) qb = (uint8_t) (rng() & 0xFFu);
                std::memcpy(b.data() + i * sizeof(k::block_q4_0), &blk, sizeof(blk));
            }
            (side == 0 ? p.k4 : p.v4) = upload(b, q);
        }
    }
    return p;
}

constexpr int64_t kCells = 32768, kHole = 12800;

struct Case {
    int fmt = 0;
    int64_t cap = 0;
    int n_q = 0;
    std::vector<int32_t> widths;
};

void selection(int32_t* ids, int64_t cap, int32_t width, int shape, std::mt19937& rng) {
    const int64_t w = std::max<int64_t>(0, std::min<int64_t>(cap, width < 0 ? cap : width));
    if (shape == 0) {
        const int64_t start = kHole - w / 2;
        for (int64_t j = 0; j < w; ++j) ids[j] = (int32_t) (start + j);
        for (int64_t j = w; j < cap; ++j) ids[j] = 0;
    } else if (shape == 1) {
        std::vector<int32_t> v((size_t) cap);
        for (int64_t j = 0; j < cap; ++j) v[(size_t) j] = (int32_t) (rng() % (uint32_t) kCells);
        std::sort(v.begin(), v.begin() + w);
        std::copy(v.begin(), v.end(), ids);
    } else {
        for (int64_t j = 0; j < cap; ++j) ids[j] = (int32_t) (rng() % (uint32_t) kCells);
    }
}

struct Harness {
    k::QsaShapes s = k::qsa_real_shapes();
    DevicePools pools;
    int32_t* table = nullptr;
    float *q = nullptr, *scratch = nullptr, *attn_old = nullptr, *attn_new = nullptr;
    int32_t *ids = nullptr, *steps = nullptr;
    int64_t max_q = 0, max_cap = 0;
    sycl::queue* sq = nullptr;

    void init(std::mt19937& rng, int64_t max_q_, int64_t max_cap_, sycl::queue& queue_ref) {
        sq = &queue_ref;
        max_q = max_q_;
        max_cap = max_cap_;
        const int64_t PS = s.page_size, pages = kCells / PS, H = s.n_head_kv, D = s.head_dim;
        std::vector<int32_t> tab((size_t) pages);
        for (int64_t i = 0; i < pages; ++i) tab[(size_t) i] = (int32_t) i;
        std::shuffle(tab.begin(), tab.end(), rng);
        for (int64_t i = 0; i < pages; ++i)
            if (rng() % 32u == 0) tab[(size_t) i] = -1;
        for (int64_t i = kHole / PS; i < (kHole + 256) / PS; ++i) tab[(size_t) i] = -1;
        table = upload(tab, *sq);
        pools = make_pools(pages * H * PS, D, rng, *sq);
        q = dalloc<float>((size_t) (max_q * s.n_head * D), *sq);
        ids = dalloc<int32_t>((size_t) (max_q * max_cap), *sq);
        steps = dalloc<int32_t>((size_t) (max_q * k::kStepCount), *sq);
        scratch = dalloc<float>((size_t) (k::qsa_decode_attn_scratch_floats(max_cap, s) * (uint64_t) max_q), *sq);
        attn_old = dalloc<float>((size_t) (max_q * s.n_head * D), *sq);
        attn_new = dalloc<float>((size_t) (max_q * s.n_head * D), *sq);
    }

    ~Harness() {
        if (!sq) return;
        for (void* p : {static_cast<void*>(table), static_cast<void*>(q), static_cast<void*>(ids),
                       static_cast<void*>(steps), static_cast<void*>(scratch), static_cast<void*>(attn_old),
                       static_cast<void*>(attn_new), static_cast<void*>(pools.k16), static_cast<void*>(pools.v16),
                       static_cast<void*>(pools.k8), static_cast<void*>(pools.v8), static_cast<void*>(pools.k8s),
                       static_cast<void*>(pools.v8s), static_cast<void*>(pools.k4), static_cast<void*>(pools.v4)})
            if (p) sycl::free(p, *sq);
    }

    void prepare(const Case& c, std::mt19937& rng) {
        const int64_t D = s.head_dim, NH = s.n_head;
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> hq((size_t) (c.n_q * NH * D));
        for (auto& x : hq) x = nd(rng);
        sq->memcpy(q, hq.data(), hq.size() * sizeof(float)).wait();
        std::vector<int32_t> hid((size_t) (c.n_q * c.cap));
        std::vector<int32_t> hst((size_t) (c.n_q * k::kStepCount), 0);
        for (int i = 0; i < c.n_q; ++i) {
            selection(hid.data() + (size_t) i * c.cap, c.cap, c.widths[(size_t) i], (int) (rng() % 3u), rng);
            hst[(size_t) i * k::kStepCount + k::kStepWidth] = c.widths[(size_t) i];
        }
        sq->memcpy(ids, hid.data(), hid.size() * sizeof(int32_t)).wait();
        sq->memcpy(steps, hst.data(), hst.size() * sizeof(int32_t)).wait();
        const size_t scratch_bytes = (size_t) k::qsa_decode_attn_scratch_floats(c.cap, s) * (size_t) c.n_q * 4;
        sq->memset(scratch, 0xFF, scratch_bytes).wait();
        sq->memset(attn_old, 0xFF, (size_t) (c.n_q * NH * D) * 4).wait();
        sq->memset(attn_new, 0x7F, (size_t) (c.n_q * NH * D) * 4).wait();
        k::qsa_decode_attn_chunks_only(q, pools.view(c.fmt, table), ids, steps, c.cap, s, scratch, c.n_q, 0, sq);
        sq->wait();
    }

    bool check(const Case& c, long long& nonzero) {
        const int64_t D = s.head_dim, NH = s.n_head;
        k::qsa_decode_attn_merge_only(steps, c.cap, s, scratch, attn_old, c.n_q, 0, sq);
        k::qsa_decode_attn_merge_only(steps, c.cap, s, scratch, attn_new, c.n_q, 1, sq);
        sq->wait();
        const size_t n = (size_t) (c.n_q * NH * D);
        std::vector<float> a(n), b(n);
        sq->memcpy(a.data(), attn_old, n * sizeof(float)).wait();
        sq->memcpy(b.data(), attn_new, n * sizeof(float)).wait();
        nonzero = 0;
        long long nan = 0;
        for (size_t i = 0; i < n; ++i) {
            if (a[i] != 0.0f) ++nonzero;
            if (!std::isfinite(a[i])) ++nan;
        }
        const bool same = std::memcmp(a.data(), b.data(), n * sizeof(float)) == 0;
        if (!same || nan) {
            size_t first = 0;
            while (first < n && std::memcmp(&a[first], &b[first], sizeof(float)) == 0) ++first;
            std::printf("FAIL %s cap %lld n_q %d widths", fmt_name(c.fmt), (long long) c.cap, c.n_q);
            for (int32_t w : c.widths) std::printf(" %d", w);
            if (first < n) {
                uint32_t ua, ub;
                std::memcpy(&ua, &a[first], 4);
                std::memcpy(&ub, &b[first], 4);
                std::printf(": first difference at query %lld head %lld dim %lld: %.9g (0x%08x) vs %.9g (0x%08x)",
                            (long long) (first / (NH * D)), (long long) (first / D % NH), (long long) (first % D),
                            a[first], ua, b[first], ub);
            }
            if (nan) std::printf(" (%lld nonfinite values in the original's output)", nan);
            std::printf("\n");
        }
        return same && !nan;
    }
};

}  // namespace

int main(int argc, char** argv) try {
    (void)argc;
    (void)argv;
    auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.empty()) {
        std::puts("SKIP: no GPU");
        return 77;
    }
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    auto prop = q.get_device();
    std::printf("qsa_merge_daily_parity on device (%s)\n", prop.get_info<sycl::info::device::name>().c_str());

    std::mt19937 rng(20261004u);
    Harness hx;
    hx.init(rng, 7, 32768, q);

    const int32_t widths[] = {-1, 0, 1, 63, 64, 65, 1023, 1024, 1025, 2051, 4096, 16383, 16384, 16385, 32768, std::numeric_limits<int32_t>::max()};
    const int n_widths = (int) (sizeof(widths) / sizeof(widths[0]));
    long long cases = 0, failed = 0, nonzero_total = 0;
    for (int fmt = 0; fmt < 4; ++fmt) {
        long long fmt_cases = 0, fmt_failed = 0;
        for (int64_t cap : {(int64_t) 2051, (int64_t) 2112, (int64_t) 32768}) {
            for (int n_q = 1; n_q <= 6; ++n_q) {
                for (int wc = 0; wc < n_widths + 3; ++wc) {
                    Case c;
                    c.fmt = fmt;
                    c.cap = cap;
                    c.n_q = n_q;
                    for (int i = 0; i < n_q; ++i)
                        c.widths.push_back(wc < n_widths ? widths[wc] : widths[rng() % (uint32_t) n_widths]);
                    if (cap % 64 != 0 && std::any_of(c.widths.begin(), c.widths.end(),
                            [cap](int32_t w) { return (int64_t) w > cap; })) continue;
                    hx.prepare(c, rng);
                    long long nz = 0;
                    const bool ok = hx.check(c, nz);
                    nonzero_total += nz;
                    ++cases;
                    ++fmt_cases;
                    if (!ok) { ++failed; ++fmt_failed; }
                }
            }
        }
        std::printf("[%s] %lld cases, %s (%lld differ)\n", fmt_name(fmt), fmt_cases, fmt_failed ? "FAIL" : "bit-identical",
                    fmt_failed);
    }
    std::printf("qsa_merge_daily_parity: %lld cases, %lld differ, %lld nonzero output values in all -> %s\n", cases, failed,
                nonzero_total, failed == 0 && nonzero_total > 0 ? "PASS" : "FAIL");
    return failed == 0 && nonzero_total > 0 ? 0 : 1;
} catch (const sycl::exception& e) {
    std::fprintf(stderr, "SYCL exception: %s\n", e.what());
    return 1;
}
