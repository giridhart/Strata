// tests/sycl/sycl_iq_fused_bench.cpp
// Correctness and benchmark suite for:
//   - Mode 8 fused Gate/Up + SwiGLU + q8_1 quantization (DecG<TY> + nib_neg4 + dynamic LNT=1/2/4)
//     across IQ2_XXS(16), IQ2_XS(17), IQ3_XXS(18), IQ3_S(21), IQ2_S(22), IQ1_M(29)
//   - Q2_0(42) down-projection with 11-dword bank-conflict-free SLM staging (kDtW=11)

#include "strata/kernels/iq_kernels.hpp"
#include "strata/core/sycl_arch.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {

struct Q81Host {
    sycl::half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q81Host) == 36);

struct FmtInfo {
    int type;
    const char* name;
    int block_bytes;
    int qk;
};

constexpr FmtInfo kIqFmts[] = {
    {16, "IQ2_XXS", 66, 256},
    {17, "IQ2_XS",  74, 256},
    {18, "IQ3_XXS", 98, 256},
    {21, "IQ3_S",  110, 256},
    {22, "IQ2_S",   82, 256},
    {29, "IQ1_M",   56, 256},
};

int block_bytes_of(int ty) {
    for (const auto& f : kIqFmts) {
        if (f.type == ty) return f.block_bytes;
    }
    if (ty == 42) return 16;  // block_q2_0 (32 weights -> wait: 2 + 8 = 10? no, check iq_row_bytes)
    return 0;
}

const char* name_of(int ty) {
    for (const auto& f : kIqFmts) {
        if (f.type == ty) return f.name;
    }
    if (ty == 42) return "Q2_0";
    if (ty == 23) return "IQ4_XS";
    return "?";
}

// Fill random quantized expert weights with valid finite fp16 block scales.
std::vector<uint8_t> make_random_weights(int ty, int64_t rows, int64_t cols, std::mt19937& rng) {
    const size_t rb = k::iq_row_bytes(ty, cols);
    std::vector<uint8_t> buf((size_t)rows * rb);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_int_distribution<int> scale_dist(0x2800, 0x3400);  // finite positive fp16 in [0.03125, 0.25]
    for (size_t i = 0; i < buf.size(); ++i) buf[i] = (uint8_t)byte_dist(rng);

    if (ty == 29) {
        // IQ1_M: 56 bytes per 256-weight block; last 8 bytes are 4 uint16_t whose top 4 bits form fp16 scale.
        const size_t bs = 56;
        const size_t nb = rb / bs;
        for (int64_t r = 0; r < rows; ++r) {
            uint8_t* row = buf.data() + (size_t)r * rb;
            for (size_t b = 0; b < nb; ++b) {
                uint8_t* blk = row + b * bs;
                uint16_t sc[4];
                std::memcpy(sc, blk + 48, 8);
                const uint16_t d = (uint16_t)scale_dist(rng);
                sc[0] = (sc[0] & 0x0FFFu) | (uint16_t)((d & 0x000Fu) << 12);
                sc[1] = (sc[1] & 0x0FFFu) | (uint16_t)((d & 0x00F0u) << 8);
                sc[2] = (sc[2] & 0x0FFFu) | (uint16_t)((d & 0x0F00u) << 4);
                sc[3] = (sc[3] & 0x0FFFu) | (uint16_t)(d & 0xF000u);
                std::memcpy(blk + 48, sc, 8);
            }
        }
    } else {
        const int bs = (ty == 42) ? (int)(k::iq_row_bytes(42, 256) / 8) : block_bytes_of(ty);
        const size_t nb = rb / (size_t)bs;
        for (int64_t r = 0; r < rows; ++r) {
            uint8_t* row = buf.data() + (size_t)r * rb;
            for (size_t b = 0; b < nb; ++b) {
                const uint16_t d = (uint16_t)scale_dist(rng);
                std::memcpy(row + b * (size_t)bs, &d, 2);
            }
        }
    }
    return buf;
}

std::vector<float> make_random_x(int64_t n_tok, int64_t n_embd, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
    std::vector<float> x((size_t)n_tok * (size_t)n_embd);
    for (float& v : x) v = dist(rng);
    return x;
}

template <typename T>
T* dalloc(size_t n, sycl::queue& q) {
    T* p = sycl::malloc_device<T>(n, q);
    if (!p) {
        std::fprintf(stderr, "sycl::malloc_device failed (%zu elements)\n", n);
        std::abort();
    }
    return p;
}

struct GroupedCase {
    k::NativeExpertLayout L;
    int cap_groups = 0;
    int cap_entries = 0;
    int n_tok = 0;
    std::vector<uint8_t*> blobs;
    unsigned long long* dptr = nullptr;
    int32_t* dstart = nullptr;
    int32_t* dn = nullptr;
    int32_t* ddst = nullptr;
    int32_t* dtok = nullptr;
    float* dx = nullptr;
    uint8_t* dxq = nullptr;
    uint8_t* dscr = nullptr;
    float* dout = nullptr;
    sycl::queue& q;

    GroupedCase(int gu_type, int d_type, int64_t n_embd, int64_t n_ff,
                const std::vector<int>& counts, int tokens, sycl::queue& queue, std::mt19937& rng)
        : L(k::native_expert_layout(gu_type, d_type, n_embd, n_ff)),
          cap_groups((int)counts.size()),
          n_tok(tokens),
          q(queue) {
        std::vector<int32_t> start(counts.size() + 1, 0);
        for (size_t i = 0; i < counts.size(); ++i) start[i + 1] = start[i] + counts[i];
        cap_entries = start.back();
        if (cap_entries <= 0) cap_entries = 1;

        std::vector<int32_t> dst(cap_entries), tok(cap_entries);
        for (int i = 0; i < cap_entries; ++i) {
            dst[i] = i;
            tok[i] = i % tokens;
        }
        std::vector<unsigned long long> ptrs(cap_groups);
        for (int g = 0; g < cap_groups; ++g) {
            auto g_w = make_random_weights(gu_type, n_ff, n_embd, rng);
            auto u_w = make_random_weights(gu_type, n_ff, n_embd, rng);
            auto d_w = make_random_weights(d_type, n_embd, n_ff, rng);
            std::vector<uint8_t> blob(L.bytes, 0);
            std::memcpy(blob.data(), g_w.data(), g_w.size());
            std::memcpy(blob.data() + L.up_off, u_w.data(), u_w.size());
            std::memcpy(blob.data() + L.down_off, d_w.data(), d_w.size());
            uint8_t* db = dalloc<uint8_t>(L.bytes, q);
            q.memcpy(db, blob.data(), L.bytes).wait();
            blobs.push_back(db);
            ptrs[g] = (unsigned long long)db;
        }
        dptr = dalloc<unsigned long long>(cap_groups, q);
        dstart = dalloc<int32_t>(cap_groups + 1, q);
        dn = dalloc<int32_t>(1, q);
        ddst = dalloc<int32_t>(cap_entries, q);
        dtok = dalloc<int32_t>(cap_entries, q);
        q.memcpy(dptr, ptrs.data(), ptrs.size() * 8);
        q.memcpy(dstart, start.data(), start.size() * 4);
        q.memcpy(dn, &cap_groups, 4);
        q.memcpy(ddst, dst.data(), dst.size() * 4);
        q.memcpy(dtok, tok.data(), tok.size() * 4);

        auto hx = make_random_x(tokens, n_embd, rng);
        dx = dalloc<float>(hx.size(), q);
        q.memcpy(dx, hx.data(), hx.size() * 4);
        dxq = dalloc<uint8_t>((size_t)tokens * (size_t)(n_embd / 32) * sizeof(Q81Host), q);
        dscr = dalloc<uint8_t>(k::native_expert_scratch_bytes(cap_entries, n_ff), q);
        dout = dalloc<float>((size_t)cap_entries * (size_t)n_embd, q);
        q.wait();
        k::quantize_q8_1_rows(dx, tokens, n_embd, dxq, &q);
        q.wait();
    }

    ~GroupedCase() {
        for (uint8_t* b : blobs) sycl::free(b, q);
        sycl::free(dptr, q);
        sycl::free(dstart, q);
        sycl::free(dn, q);
        sycl::free(ddst, q);
        sycl::free(dtok, q);
        sycl::free(dx, q);
        sycl::free(dxq, q);
        sycl::free(dscr, q);
        sycl::free(dout, q);
    }

    size_t hq_offset() const {
        const size_t f = (size_t)cap_entries * (size_t)L.n_ff * sizeof(float);
        return 3 * ((f + 255) & ~(size_t)255);
    }

    std::vector<Q81Host> run_hq(bool old_kernels, int mode) {
        const size_t nblk = (size_t)cap_entries * (size_t)(L.n_ff / 32);
        q.memset(dscr, 0, k::native_expert_scratch_bytes(cap_entries, L.n_ff)).wait();
        k::iq_set_old_kernels(old_kernels);
        k::native_expert_set_mode(mode, 1);
        k::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, cap_groups, cap_entries, dxq, dscr, dout, &q);
        q.wait();
        k::native_expert_set_mode(-1, 0);
        k::iq_set_old_kernels(false);
        std::vector<Q81Host> hq(nblk);
        q.memcpy(hq.data(), dscr + hq_offset(), nblk * sizeof(Q81Host)).wait();
        return hq;
    }

    std::vector<float> run_e2e(bool old_kernels, int mode) {
        const size_t nout = (size_t)cap_entries * (size_t)L.n_embd;
        q.memset(dout, 0xFF, nout * sizeof(float)).wait();
        q.memset(dscr, 0, k::native_expert_scratch_bytes(cap_entries, L.n_ff)).wait();
        k::iq_set_old_kernels(old_kernels);
        k::native_expert_set_mode(mode, 0);
        k::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, cap_groups, cap_entries, dxq, dscr, dout, &q);
        q.wait();
        k::native_expert_set_mode(-1, 0);
        k::iq_set_old_kernels(false);
        std::vector<float> out(nout);
        q.memcpy(out.data(), dout, nout * sizeof(float)).wait();
        return out;
    }
};

std::vector<int> build_counts(int n_exp, int T_per_exp) {
    if (n_exp == 1) return {T_per_exp};
    std::vector<int> c(n_exp, T_per_exp);
    // Include uneven and zero-entry expert when batching multi-expert with T>=3
    if (n_exp >= 4 && T_per_exp >= 3) {
        c[0] = 0;
        c[1] = T_per_exp + 1;
        c[2] = std::max(1, T_per_exp - 1);
    }
    return c;
}

int test_mode8_correctness(sycl::queue& q) {
    int failures = 0;
    std::mt19937 rng(20261113);

    const int ts[] = {1, 2, 3, 4, 8};
    struct Shape { int64_t k, rows; int n_exp; };
    const Shape shapes[] = {
        {1024, 256, 1},
        {1024, 640, 8},
        {2560, 256, 8},
        {2560, 640, 16},
    };

    for (const auto& fmt : kIqFmts) {
        int fmt_fails = 0;
        for (const auto& sh : shapes) {
            for (int t : ts) {
                // Keep total work bounded on iGPU while testing every T, shape, and expert count.
                if (sh.n_exp == 16 && (t == 3 || sh.k == 1024)) continue;
                const auto counts = build_counts(sh.n_exp, t);
                int sum_ent = 0;
                for (int c : counts) sum_ent += c;
                GroupedCase gc(fmt.type, 42, sh.k, sh.rows, counts, std::max(1, sum_ent), q, rng);

                const auto ref_hq = gc.run_hq(true, 0);
                const auto m8_hq = gc.run_hq(false, 8);
                int qs_diff = 0, ds_diff = 0;
                float max_deq_diff = 0.0f, max_deq_rel = 0.0f;
                bool finite_hq = true;
                for (size_t b = 0; b < ref_hq.size(); ++b) {
                    const float rd = (float)ref_hq[b].ds[0];
                    const float md = (float)m8_hq[b].ds[0];
                    const float rsum = (float)ref_hq[b].ds[1];
                    const float msum = (float)m8_hq[b].ds[1];
                    if (std::memcmp(&ref_hq[b].ds, &m8_hq[b].ds, sizeof(sycl::half2)) != 0) ++ds_diff;
                    if (!std::isfinite(md) || !std::isfinite(msum)) finite_hq = false;
                    for (int j = 0; j < 32; ++j) {
                        if (ref_hq[b].qs[j] != m8_hq[b].qs[j]) ++qs_diff;
                        const float rv = (float)ref_hq[b].qs[j] * rd;
                        const float mv = (float)m8_hq[b].qs[j] * md;
                        const float ad = std::fabs(rv - mv);
                        max_deq_diff = std::max(max_deq_diff, ad);
                        max_deq_rel = std::max(max_deq_rel, ad / (std::fabs(rv) + 1e-6f));
                    }
                    (void)rsum;
                }
                const bool hq_ok = finite_hq && (qs_diff == 0 || max_deq_rel < 2e-2f);

                const auto ref_out = gc.run_e2e(true, 0);
                const auto m8_out = gc.run_e2e(false, 8);
                float max_out_diff = 0.0f, max_out_rel = 0.0f;
                bool finite_out = true;
                for (size_t i = 0; i < ref_out.size(); ++i) {
                    if (!std::isfinite(m8_out[i])) finite_out = false;
                    const float ad = std::fabs(ref_out[i] - m8_out[i]);
                    max_out_diff = std::max(max_out_diff, ad);
                    max_out_rel = std::max(max_out_rel, ad / (std::fabs(ref_out[i]) + 1e-6f));
                }
                const bool out_ok = finite_out && max_out_diff < 5.0f && max_out_rel < 0.15f;
                if (!hq_ok || !out_ok) {
                    ++failures;
                    ++fmt_fails;
                    std::printf("FAIL Mode8 %-7s k=%4lld rows=%3lld exp=%2d T=%d: "
                                "qs_diff=%d ds_diff=%d max_deq_diff=%.4g max_out_diff=%.4g max_out_rel=%.4g\n",
                                fmt.name, (long long)sh.k, (long long)sh.rows, sh.n_exp, t,
                                qs_diff, ds_diff, max_deq_diff, max_out_diff, max_out_rel);
                }
            }
        }
        if (fmt_fails == 0) {
            std::printf("[PASS] Mode 8 Gate/Up+SwiGLU+q8_1 %-7s (T=1,2,3,4,8; k=1024,2560; rows=256,640; n_exp=1,8,16)\n",
                        fmt.name);
        }
    }

    // Dedicated Q2_0 down-projection parity with 11-dword SLM layout across T=1,2,4,8
    for (int t : {1, 2, 4, 8}) {
        const auto counts = build_counts(8, t);
        int sum_ent = 0;
        for (int c : counts) sum_ent += c;
        GroupedCase gc(22, 42, 2560, 640, counts, sum_ent, q, rng);
        const auto ref_out = gc.run_e2e(true, 0);
        const auto dt_out = gc.run_e2e(false, 0);
        float max_diff = 0.0f, max_rel = 0.0f;
        bool finite = true;
        for (size_t i = 0; i < ref_out.size(); ++i) {
            if (!std::isfinite(dt_out[i])) finite = false;
            const float ad = std::fabs(ref_out[i] - dt_out[i]);
            max_diff = std::max(max_diff, ad);
            max_rel = std::max(max_rel, ad / (std::fabs(ref_out[i]) + 1e-6f));
        }
        if (!finite || max_rel >= 0.05f) {
            ++failures;
            std::printf("FAIL Q2_0 down kDtW=11 T=%d: max_diff=%g max_rel=%g\n", t, max_diff, max_rel);
        } else {
            std::printf("[PASS] Q2_0 down (kDtW=11) T=%d max_diff=%.4g max_rel=%.4g\n", t, max_diff, max_rel);
        }
    }
    return failures;
}

void run_bench(sycl::queue& q) {
    std::mt19937 rng(777);
    const int64_t k = 2560, rows = 640, n_exp = 16;
    const int ts[] = {1, 2, 4, 8};

    // Use a ring of weight slots so repeated runs stream cold weights from VRAM instead of L2.
    const int kSlots = 8;
    const int warmup = 4, reps = 24;

    std::printf("\n=== Mode 8 Fused Gate/Up+SwiGLU+q8_1 vs Baseline (k=%lld, rows=%lld, n_exp=%lld) ===\n",
                (long long)k, (long long)rows, (long long)n_exp);
    std::printf("%-8s %3s %10s %10s %10s %10s %8s\n",
                "Fmt", "T", "Base(us)", "Mode8(us)", "Base(GB/s)", "M8(GB/s)", "Speedup");

    for (const auto& fmt : kIqFmts) {
        const auto L = k::native_expert_layout(fmt.type, 42, k, rows);
        const size_t gu_bytes_per_exp = L.down_off;  // gate + up per expert
        // Allocate kSlots * n_exp expert blobs once per format.
        std::vector<uint8_t*> d_blobs(kSlots * n_exp);
        std::vector<unsigned long long> h_ptrs(kSlots * n_exp);
        auto g_w = make_random_weights(fmt.type, rows, k, rng);
        auto u_w = make_random_weights(fmt.type, rows, k, rng);
        std::vector<uint8_t> h_blob(L.bytes, 0);
        std::memcpy(h_blob.data(), g_w.data(), g_w.size());
        std::memcpy(h_blob.data() + L.up_off, u_w.data(), u_w.size());
        for (int s = 0; s < kSlots * n_exp; ++s) {
            d_blobs[s] = dalloc<uint8_t>(L.bytes, q);
            q.memcpy(d_blobs[s], h_blob.data(), L.bytes);
            h_ptrs[s] = (unsigned long long)d_blobs[s];
        }
        unsigned long long* d_ptr = dalloc<unsigned long long>(kSlots * n_exp, q);
        q.memcpy(d_ptr, h_ptrs.data(), h_ptrs.size() * 8).wait();

        for (int t : ts) {
            const int cap_ent = (int)n_exp * t;
            std::vector<int32_t> h_start(n_exp + 1, 0), h_dst(cap_ent), h_tok(cap_ent);
            for (int e = 0; e < n_exp; ++e) h_start[e + 1] = h_start[e] + t;
            for (int i = 0; i < cap_ent; ++i) { h_dst[i] = i; h_tok[i] = i; }

            int32_t* d_start = dalloc<int32_t>(n_exp + 1, q);
            int32_t* d_n = dalloc<int32_t>(1, q);
            int32_t* d_dst = dalloc<int32_t>(cap_ent, q);
            int32_t* d_tok = dalloc<int32_t>(cap_ent, q);
            const int32_t ng = (int32_t)n_exp;
            q.memcpy(d_start, h_start.data(), h_start.size() * 4);
            q.memcpy(d_n, &ng, 4);
            q.memcpy(d_dst, h_dst.data(), h_dst.size() * 4);
            q.memcpy(d_tok, h_tok.data(), h_tok.size() * 4);

            auto hx = make_random_x(cap_ent, k, rng);
            float* dx = dalloc<float>(hx.size(), q);
            uint8_t* dxq = dalloc<uint8_t>((size_t)cap_ent * (size_t)(k / 32) * sizeof(Q81Host), q);
            uint8_t* dscr = dalloc<uint8_t>(k::native_expert_scratch_bytes(cap_ent, rows), q);
            float* dout = dalloc<float>((size_t)cap_ent * (size_t)k, q);
            q.memcpy(dx, hx.data(), hx.size() * 4).wait();
            k::quantize_q8_1_rows(dx, cap_ent, k, dxq, &q);
            q.wait();

            auto time_mode = [&](int mode) {
                k::native_expert_set_mode(mode, 1);
                for (int i = 0; i < warmup; ++i) {
                    const unsigned long long* p = d_ptr + (size_t)(i % kSlots) * n_exp;
                    k::native_expert_grouped(L, p, d_start, d_n, d_dst, d_tok, n_exp, cap_ent, dxq, dscr, dout, &q);
                }
                q.wait();
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < reps; ++i) {
                    const unsigned long long* p = d_ptr + (size_t)(i % kSlots) * n_exp;
                    k::native_expert_grouped(L, p, d_start, d_n, d_dst, d_tok, n_exp, cap_ent, dxq, dscr, dout, &q);
                }
                q.wait();
                const auto t1 = std::chrono::steady_clock::now();
                k::native_expert_set_mode(-1, 0);
                return std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
            };

            const double us_base = time_mode(0);
            const double us_m8 = time_mode(8);
            const double total_w_gb = (double)(n_exp * gu_bytes_per_exp) / 1e9;
            std::printf("%-8s %3d %10.1f %10.1f %10.2f %10.2f %7.2fx\n",
                        fmt.name, t, us_base, us_m8,
                        total_w_gb / (us_base * 1e-6),
                        total_w_gb / (us_m8 * 1e-6),
                        us_base / us_m8);

            sycl::free(d_start, q);
            sycl::free(d_n, q);
            sycl::free(d_dst, q);
            sycl::free(d_tok, q);
            sycl::free(dx, q);
            sycl::free(dxq, q);
            sycl::free(dscr, q);
            sycl::free(dout, q);
        }
        for (uint8_t* p : d_blobs) sycl::free(p, q);
        sycl::free(d_ptr, q);
    }

    // Q2_0 down-projection benchmark (phase=2)
    std::printf("\n=== Q2_0 Down-Projection (kDtW=%s vs old, k=%lld, rows=%lld, n_exp=%lld) ===\n",
                std::getenv("STRATA_IQ_DT_W") ? std::getenv("STRATA_IQ_DT_W") : "11",
                (long long)k, (long long)rows, (long long)n_exp);
    std::printf("%-8s %3s %10s %10s %10s %10s %8s\n",
                "Fmt", "T", "Old(us)", "Fast(us)", "Old(GB/s)", "Fast(GB/s)", "Speedup");
    {
        const auto L = k::native_expert_layout(22, 42, k, rows);
        const size_t d_bytes_per_exp = L.bytes - L.down_off;
        std::vector<uint8_t*> d_blobs(kSlots * n_exp);
        std::vector<unsigned long long> h_ptrs(kSlots * n_exp);
        auto d_w = make_random_weights(42, k, rows, rng);
        std::vector<uint8_t> h_blob(L.bytes, 0);
        std::memcpy(h_blob.data() + L.down_off, d_w.data(), d_w.size());
        for (int s = 0; s < kSlots * n_exp; ++s) {
            d_blobs[s] = dalloc<uint8_t>(L.bytes, q);
            q.memcpy(d_blobs[s], h_blob.data(), L.bytes);
            h_ptrs[s] = (unsigned long long)d_blobs[s];
        }
        unsigned long long* d_ptr = dalloc<unsigned long long>(kSlots * n_exp, q);
        q.memcpy(d_ptr, h_ptrs.data(), h_ptrs.size() * 8).wait();

        for (int t : ts) {
            const int cap_ent = (int)n_exp * t;
            std::vector<int32_t> h_start(n_exp + 1, 0), h_dst(cap_ent), h_tok(cap_ent);
            for (int e = 0; e < n_exp; ++e) h_start[e + 1] = h_start[e] + t;
            for (int i = 0; i < cap_ent; ++i) { h_dst[i] = i; h_tok[i] = i; }

            int32_t* d_start = dalloc<int32_t>(n_exp + 1, q);
            int32_t* d_n = dalloc<int32_t>(1, q);
            int32_t* d_dst = dalloc<int32_t>(cap_ent, q);
            int32_t* d_tok = dalloc<int32_t>(cap_ent, q);
            const int32_t ng = (int32_t)n_exp;
            q.memcpy(d_start, h_start.data(), h_start.size() * 4);
            q.memcpy(d_n, &ng, 4);
            q.memcpy(d_dst, h_dst.data(), h_dst.size() * 4);
            q.memcpy(d_tok, h_tok.data(), h_tok.size() * 4);

            uint8_t* dxq = dalloc<uint8_t>((size_t)cap_ent * (size_t)(k / 32) * sizeof(Q81Host), q);
            uint8_t* dscr = dalloc<uint8_t>(k::native_expert_scratch_bytes(cap_ent, rows), q);
            float* dout = dalloc<float>((size_t)cap_ent * (size_t)k, q);
            q.memset(dscr, 0, k::native_expert_scratch_bytes(cap_ent, rows)).wait();

            auto time_down = [&](bool old_k) {
                k::iq_set_old_kernels(old_k);
                k::native_expert_set_mode(0, 2);
                for (int i = 0; i < warmup; ++i) {
                    const unsigned long long* p = d_ptr + (size_t)(i % kSlots) * n_exp;
                    k::native_expert_grouped(L, p, d_start, d_n, d_dst, d_tok, n_exp, cap_ent, dxq, dscr, dout, &q);
                }
                q.wait();
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < reps; ++i) {
                    const unsigned long long* p = d_ptr + (size_t)(i % kSlots) * n_exp;
                    k::native_expert_grouped(L, p, d_start, d_n, d_dst, d_tok, n_exp, cap_ent, dxq, dscr, dout, &q);
                }
                q.wait();
                const auto t1 = std::chrono::steady_clock::now();
                k::native_expert_set_mode(-1, 0);
                k::iq_set_old_kernels(false);
                return std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
            };

            const double us_old = time_down(true);
            const double us_fast = time_down(false);
            const double total_w_gb = (double)(n_exp * d_bytes_per_exp) / 1e9;
            std::printf("%-8s %3d %10.1f %10.1f %10.2f %10.2f %7.2fx\n",
                        "Q2_0", t, us_old, us_fast,
                        total_w_gb / (us_old * 1e-6),
                        total_w_gb / (us_fast * 1e-6),
                        us_old / us_fast);

            sycl::free(d_start, q);
            sycl::free(d_n, q);
            sycl::free(d_dst, q);
            sycl::free(d_tok, q);
            sycl::free(dxq, q);
            sycl::free(dscr, q);
            sycl::free(dout, q);
        }
        for (uint8_t* p : d_blobs) sycl::free(p, q);
        sycl::free(d_ptr, q);
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool do_bench = false;
    bool bench_only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) do_bench = true;
        if (std::strcmp(argv[i], "--bench-only") == 0) { do_bench = true; bench_only = true; }
    }

    sycl::queue q;
    try {
        q = sycl::queue(sycl::gpu_selector_v, sycl::property::queue::in_order());
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "sycl_iq_fused_bench: no SYCL GPU device (%s)\n", e.what());
        return 0;
    }
    const auto dev = q.get_device();
    std::printf("Device: %s (SLM %zu B, CUs %u)\n",
                dev.get_info<sycl::info::device::name>().c_str(),
                (size_t)dev.get_info<sycl::info::device::local_mem_size>(),
                dev.get_info<sycl::info::device::max_compute_units>());

    int failures = 0;
    if (!bench_only) {
        failures = test_mode8_correctness(q);
        std::printf("sycl_iq_fused_bench: %d failures\n", failures);
    }
    if (do_bench && failures == 0) {
        run_bench(q);
    }
    return failures ? 1 : 0;
}
