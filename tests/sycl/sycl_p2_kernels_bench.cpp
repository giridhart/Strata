// tests/sycl/sycl_p2_kernels_bench.cpp
// Correctness tests and --bench harness for P2 SYCL optimizations:
//   1. High-priority SYCL in-order queue creation & contention microbenchmark (MTP drafter cs_)
//   2. qsa_select Top-K prefill L2/GRF thrash guard (block_topk_reg vs block_topk_wide)
//   3. native_bf16 exact n_tok = 1, 2 specializations in bf16_f32_mmvf_rows & multi kernels

#include <sycl/sycl.hpp>
#include "strata/core/sycl_doorbell.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_failures = 0;

void check_cond(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++g_failures;
    }
}

uint32_t cpu_order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    uint32_t b = 0;
    std::memcpy(&b, &v, sizeof(b));
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

std::vector<int32_t> make_steps(int nq, int64_t n_kv_base, const k::QsaShapes& s, bool vary) {
    std::vector<int32_t> st((size_t)nq * k::kStepCount, 0);
    for (int i = 0; i < nq; ++i) {
        int32_t* p = st.data() + (size_t)i * k::kStepCount;
        const int64_t n_kv = vary ? (n_kv_base + i) : n_kv_base;
        p[k::kStepPos] = (int32_t)(n_kv - 1);
        p[k::kStepNKv] = (int32_t)n_kv;
        p[k::kStepNBid] = (int32_t)(n_kv / s.idx_block);
        p[k::kStepWidth] = (int32_t)k::qsa_selection_width(n_kv, s);
    }
    return st;
}

std::vector<int32_t> cpu_topk_ref(const float* sc, int64_t n_kv, int64_t n_bid, int64_t width, int R) {
    if (n_kv <= width) {
        std::vector<int32_t> out((size_t)n_kv);
        std::iota(out.begin(), out.end(), 0);
        return out;
    }
    const int64_t nb = n_bid + 1;
    struct BlockKey {
        uint32_t key;
        int64_t b;
        int w;
    };
    std::vector<BlockKey> blocks;
    blocks.reserve((size_t)nb);
    for (int64_t b = 0; b < nb; ++b) {
        const int w = (b < n_bid) ? R : (int)(n_kv - n_bid * R);
        if (w <= 0) continue;
        blocks.push_back({cpu_order_key(sc[b]), b, w});
    }
    std::sort(blocks.begin(), blocks.end(), [](const BlockKey& a, const BlockKey& c) {
        if (a.key != c.key) return a.key > c.key;
        return a.b < c.b;
    });
    std::vector<int32_t> out;
    out.reserve((size_t)width);
    int64_t rem = width;
    for (const auto& bk : blocks) {
        if (rem <= 0) break;
        const int take = (int)std::min<int64_t>(rem, bk.w);
        for (int c = 0; c < take; ++c) {
            out.push_back((int32_t)(bk.b * R + c));
        }
        rem -= take;
    }
    std::sort(out.begin(), out.end());
    return out;
}

// ---- 1. High-Priority Queue Smoke Test ----

void test_priority_queue_smoke(sycl::queue& base_q) {
    bool got_high = false;
    sycl::queue q_high = strata::make_in_order_queue(base_q.get_context(), base_q.get_device(), true, &got_high);
    sycl::queue q_norm = strata::make_in_order_queue(base_q.get_context(), base_q.get_device(), false, nullptr);

#if defined(SYCL_EXT_ONEAPI_QUEUE_PRIORITY)
    const bool has_prop = q_high.has_property<sycl::ext::oneapi::property::queue::priority_high>();
    std::printf("  priority_queue: compiled=yes, got_high=%d, has_property=%d\n", (int)got_high, (int)has_prop);
    check_cond(got_high == has_prop, "priority_high property matches creation status");
#else
    std::printf("  priority_queue: compiled=no (fallback in_order)\n");
#endif
    check_cond(!q_norm.has_property<sycl::ext::oneapi::property::queue::priority_high>(),
               "normal queue must not have priority_high");

    constexpr size_t N = 4096;
    float* d_a = sycl::malloc_device<float>(N, q_high);
    float* d_b = sycl::malloc_device<float>(N, q_norm);
    check_cond(d_a != nullptr && d_b != nullptr, "priority queue device alloc");
    if (!d_a || !d_b) return;

    sycl::event e1 = q_norm.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        d_b[i] = (float)i[0] * 0.5f + 1.0f;
    });
    sycl::event e2 = q_high.submit([&](sycl::handler& cgh) {
        cgh.depends_on(e1);
        cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
            d_a[i] = d_b[i] * 2.0f - 0.25f;
        });
    });
    std::vector<float> h_a(N);
    q_high.memcpy(h_a.data(), d_a, N * sizeof(float), e2).wait();

    bool ok = true;
    for (size_t i = 0; i < N; ++i) {
        const float exp_v = ((float)i * 0.5f + 1.0f) * 2.0f - 0.25f;
        if (std::fabs(h_a[i] - exp_v) > 1e-5f) {
            ok = false;
            break;
        }
    }
    check_cond(ok, "high-priority queue execution & cross-queue event dependency");

    sycl::free(d_a, q_high);
    sycl::free(d_b, q_norm);
}

// ---- 2. QSA Top-K Parity Test (counted == false decode & counted == true prefill) ----

void test_qsa_topk_case(sycl::queue& q, const k::QsaShapes& s, int nq, int64_t n_blocks, bool counted, int mode,
                        std::mt19937& rng) {
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    // Keep n_kv inside n_blocks so n_bid < n_blocks and total blocks <= n_blocks
    const int64_t max_blocks = counted ? (n_blocks + 32) : n_blocks;
    const int64_t active_blocks = counted ? n_blocks : -1;
    const int64_t n_kv_base = n_blocks * s.idx_block - 1 - nq;
    const std::vector<int32_t> steps = make_steps(nq, n_kv_base, s, true);

    std::vector<float> sc((size_t)nq * (size_t)max_blocks, -999.0f);
    std::uniform_real_distribution<float> ur(-50.0f, 200.0f);
    for (int i = 0; i < nq; ++i) {
        const int32_t nb = steps[(size_t)i * k::kStepCount + k::kStepNBid];
        const bool tail = (steps[(size_t)i * k::kStepCount + k::kStepNKv] % s.idx_block) != 0;
        for (int64_t b = 0; b <= nb; ++b) {
            float v = (mode == 1) ? (float)(rng() % 7u) - 3.0f : ur(rng);
            sc[(size_t)i * (size_t)max_blocks + (size_t)b] = v;
        }
        if (tail && mode != 1) sc[(size_t)i * (size_t)max_blocks + (size_t)nb] += 1e9f;
    }

    float* d_sc = sycl::malloc_device<float>(sc.size(), q);
    int32_t* d_st = sycl::malloc_device<int32_t>(steps.size(), q);
    int32_t* d_reg = sycl::malloc_device<int32_t>((size_t)nq * (size_t)cap, q);
    int32_t* d_wide = sycl::malloc_device<int32_t>((size_t)nq * (size_t)cap, q);
    int32_t* d_def = sycl::malloc_device<int32_t>((size_t)nq * (size_t)cap, q);
    q.memcpy(d_sc, sc.data(), sc.size() * sizeof(float));
    q.memcpy(d_st, steps.data(), steps.size() * sizeof(int32_t));
    q.fill(d_reg, -1, (size_t)nq * (size_t)cap);
    q.fill(d_wide, -1, (size_t)nq * (size_t)cap);
    q.fill(d_def, -1, (size_t)nq * (size_t)cap);
    q.wait();

    // wide=0 (force reg when reach <= fit)
    setenv("STRATA_QSA_TOPK_PREFILL_WIDE", "0", 1);
    k::qsa_block_topk(d_sc, d_st, nq, max_blocks, cap, s, d_reg, &q, active_blocks);
    // wide=1 (force wide when counted)
    setenv("STRATA_QSA_TOPK_PREFILL_WIDE", "1", 1);
    k::qsa_block_topk(d_sc, d_st, nq, max_blocks, cap, s, d_wide, &q, active_blocks);
    // default threshold path
    unsetenv("STRATA_QSA_TOPK_PREFILL_WIDE");
    unsetenv("STRATA_QSA_TOPK_WIDE_MIN");
    k::qsa_block_topk(d_sc, d_st, nq, max_blocks, cap, s, d_def, &q, active_blocks);
    q.wait();

    std::vector<int32_t> h_reg((size_t)nq * (size_t)cap);
    std::vector<int32_t> h_wide((size_t)nq * (size_t)cap);
    std::vector<int32_t> h_def((size_t)nq * (size_t)cap);
    q.memcpy(h_reg.data(), d_reg, h_reg.size() * sizeof(int32_t));
    q.memcpy(h_wide.data(), d_wide, h_wide.size() * sizeof(int32_t));
    q.memcpy(h_def.data(), d_def, h_def.size() * sizeof(int32_t));
    q.wait();

    bool ok = true;
    for (int i = 0; i < nq && ok; ++i) {
        const int64_t n_kv = steps[(size_t)i * k::kStepCount + k::kStepNKv];
        const int64_t n_bid = steps[(size_t)i * k::kStepCount + k::kStepNBid];
        const int64_t width = steps[(size_t)i * k::kStepCount + k::kStepWidth];
        const int64_t count = std::min(n_kv, width);
        const std::vector<int32_t> ref =
            cpu_topk_ref(sc.data() + (size_t)i * (size_t)max_blocks, n_kv, n_bid, width, s.idx_block);
        if ((int64_t)ref.size() != count) {
            std::fprintf(stderr, "FAIL topk ref size mismatch\n");
            ok = false;
            break;
        }
        const size_t off = (size_t)i * (size_t)cap;
        for (int64_t j = 0; j < count; ++j) {
            if (h_reg[off + j] != ref[j] || h_wide[off + j] != ref[j] || h_def[off + j] != ref[j]) {
                std::fprintf(stderr,
                             "FAIL topk nq=%d n_blocks=%lld counted=%d mode=%d q=%d idx=%lld: ref=%d reg=%d wide=%d def=%d\n",
                             nq, (long long)n_blocks, (int)counted, mode, i, (long long)j, ref[j],
                             h_reg[off + j], h_wide[off + j], h_def[off + j]);
                ok = false;
                break;
            }
        }
        // Untouched tail beyond count should remain -1
        if (ok && count < cap && (h_reg[off + count] != -1 || h_wide[off + count] != -1 || h_def[off + count] != -1)) {
            std::fprintf(stderr, "FAIL topk wrote past count=%lld\n", (long long)count);
            ok = false;
        }
    }
    if (!ok) ++g_failures;

    sycl::free(d_sc, q);
    sycl::free(d_st, q);
    sycl::free(d_reg, q);
    sycl::free(d_wide, q);
    sycl::free(d_def, q);
}

void test_qsa_topk_parity(sycl::queue& q) {
    const k::QsaShapes s = k::qsa_real_shapes();
    std::mt19937 rng(20261114u);

    // Decode (counted == false): n_q = 1, 2, 4, 8 across small & large contexts (including <= width)
    for (int nq : {1, 2, 4, 8}) {
        for (int64_t nb : {(int64_t)256, (int64_t)2048}) {
            for (int mode : {0, 1}) {
                test_qsa_topk_case(q, s, nq, nb, false, mode, rng);
            }
        }
    }

    // Prefill (counted == true): n_q = 32, 128, 512, n_blocks = 256, 2048
    for (int nq : {32, 128, 512}) {
        for (int64_t nb : {(int64_t)256, (int64_t)2048}) {
            for (int mode : {0, 1}) {
                test_qsa_topk_case(q, s, nq, nb, true, mode, rng);
            }
        }
    }
}

// ---- 3. Native BF16 Exact-N_tok Parity Test ----

void cpu_bf16_mm(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy, int64_t n_in, int64_t n_out,
                 int n_tok) {
    for (int t = 0; t < n_tok; ++t) {
        const float* xt = x + (size_t)t * (size_t)ldx;
        float* yt = y + (size_t)t * (size_t)ldy;
        for (int64_t o = 0; o < n_out; ++o) {
            const uint16_t* row = w + (size_t)o * (size_t)n_in;
            double sum = 0.0;
            for (int64_t i = 0; i < n_in; ++i) {
                sum += (double)k::f32_from_bf16(row[i]) * (double)xt[i];
            }
            yt[o] = (float)sum;
        }
    }
}

void test_native_bf16_parity(sycl::queue& q) {
    struct Shape {
        int64_t rows;
        int64_t k_dim;
        int64_t ldx_pad;
        int64_t ldy_pad;
    };
    const Shape shapes[] = {
        {33, 66, 0, 0},       // small, n_out < 64 (multi fallback), unaligned tail rows, odd-pair k
        {64, 130, 6, 3},      // exact RPB boundary, padded strides, odd-pair k
        {65, 512, 0, 0},      // RPB tail row (65 % 4 == 1)
        {515, 2560, 16, 5},   // unaligned rows + large k + padded strides
        {2560, 2560, 0, 0},   // standard dense projection
    };

    std::mt19937 rng(424242u);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    float worst_abs = 0.0f;
    float worst_rel = 0.0f;

    for (const auto& sh : shapes) {
        const int max_tok = 8;
        const int64_t ldx = sh.k_dim + sh.ldx_pad;
        const int64_t ldy = sh.rows + sh.ldy_pad;

        std::vector<float> h_x((size_t)max_tok * (size_t)ldx, 0.0f);
        std::vector<uint16_t> h_w((size_t)sh.rows * (size_t)sh.k_dim);
        std::vector<uint16_t> h_w_aux((size_t)sh.k_dim);
        for (int t = 0; t < max_tok; ++t) {
            for (int64_t i = 0; i < sh.k_dim; ++i) {
                h_x[(size_t)t * (size_t)ldx + (size_t)i] = dist(rng);
            }
        }
        for (auto& v : h_w) v = k::bf16_from_f32(dist(rng) * 0.05f);
        for (auto& v : h_w_aux) v = k::bf16_from_f32(dist(rng) * 0.05f);

        float* d_x = sycl::malloc_device<float>(h_x.size(), q);
        uint16_t* d_w = sycl::malloc_device<uint16_t>(h_w.size(), q);
        uint16_t* d_w_aux = sycl::malloc_device<uint16_t>(h_w_aux.size(), q);
        float* d_y_rows = sycl::malloc_device<float>((size_t)max_tok * (size_t)ldy, q);
        float* d_y_multi = sycl::malloc_device<float>((size_t)max_tok * (size_t)ldy, q);
        float* d_y_aux = sycl::malloc_device<float>((size_t)max_tok * 4, q);
        q.memcpy(d_x, h_x.data(), h_x.size() * sizeof(float));
        q.memcpy(d_w, h_w.data(), h_w.size() * sizeof(uint16_t));
        q.memcpy(d_w_aux, h_w_aux.data(), h_w_aux.size() * sizeof(uint16_t));
        q.wait();

        for (int n_tok : {1, 2, 3, 4, 8}) {
            std::vector<float> h_ref((size_t)n_tok * (size_t)ldy, 0.0f);
            cpu_bf16_mm(h_x.data(), ldx, h_w.data(), h_ref.data(), ldy, sh.k_dim, sh.rows, n_tok);

            q.fill(d_y_rows, -123.0f, (size_t)n_tok * (size_t)ldy);
            q.fill(d_y_multi, -123.0f, (size_t)n_tok * (size_t)ldy);
            q.wait();

            setenv("STRATA_MMVF_ROWS", "1", 1);
            k::bf16_gemv_fp32_mmvf_multi(d_x, ldx, d_w, d_y_rows, ldy, sh.k_dim, sh.rows, n_tok, &q);
            setenv("STRATA_MMVF_ROWS", "0", 1);
            k::bf16_gemv_fp32_mmvf_multi(d_x, ldx, d_w, d_y_multi, ldy, sh.k_dim, sh.rows, n_tok, &q);
            q.wait();

            std::vector<float> h_rows((size_t)n_tok * (size_t)ldy);
            std::vector<float> h_multi((size_t)n_tok * (size_t)ldy);
            q.memcpy(h_rows.data(), d_y_rows, h_rows.size() * sizeof(float));
            q.memcpy(h_multi.data(), d_y_multi, h_multi.size() * sizeof(float));
            q.wait();

            for (int t = 0; t < n_tok; ++t) {
                for (int64_t o = 0; o < sh.rows; ++o) {
                    const size_t idx = (size_t)t * (size_t)ldy + (size_t)o;
                    const float ref_v = h_ref[idx];
                    const float r_v = h_rows[idx];
                    const float m_v = h_multi[idx];
                    const float abs_r = std::fabs(r_v - ref_v);
                    const float abs_m = std::fabs(m_v - ref_v);
                    const float rel_r = abs_r / std::max(std::fabs(ref_v), 1e-6f);
                    const float rel_m = abs_m / std::max(std::fabs(ref_v), 1e-6f);
                    worst_abs = std::max(worst_abs, std::max(abs_r, abs_m));
                    worst_rel = std::max(worst_rel, std::max(rel_r, rel_m));
                    if ((rel_r > 5e-3f && abs_r > 1e-4f) || (rel_m > 5e-3f && abs_m > 1e-4f) ||
                        std::memcmp(&r_v, &m_v, sizeof(float)) != 0) {
                        std::fprintf(stderr,
                                     "FAIL bf16 rows=%lld k=%lld n_tok=%d t=%d o=%lld: ref=%.6g rows=%.6g multi=%.6g\n",
                                     (long long)sh.rows, (long long)sh.k_dim, n_tok, t, (long long)o, ref_v, r_v, m_v);
                        ++g_failures;
                        break;
                    }
                }
            }

            // Also verify bf16_gemv_fp32_mmvf_multi_aux for n_tok >= 2 and rows >= 64
            if (n_tok >= 2 && sh.rows >= 64) {
                std::vector<float> h_ref_aux((size_t)n_tok * 4, 0.0f);
                cpu_bf16_mm(h_x.data(), ldx, h_w_aux.data(), h_ref_aux.data(), 4, sh.k_dim, 1, n_tok);
                q.fill(d_y_rows, 0.0f, (size_t)n_tok * (size_t)ldy);
                q.fill(d_y_aux, 0.0f, (size_t)n_tok * 4).wait();
                setenv("STRATA_MMVF_ROWS", "1", 1);
                const bool ran = k::bf16_gemv_fp32_mmvf_multi_aux(
                    d_x, ldx, d_w, d_y_rows, ldy, sh.k_dim, sh.rows, n_tok, d_w_aux, d_y_aux, 4, &q);
                q.wait();
                check_cond(ran, "bf16_gemv_fp32_mmvf_multi_aux ran");
                std::vector<float> h_aux((size_t)n_tok * 4);
                q.memcpy(h_aux.data(), d_y_aux, h_aux.size() * sizeof(float)).wait();
                for (int t = 0; t < n_tok; ++t) {
                    const float abs_a = std::fabs(h_aux[(size_t)t * 4] - h_ref_aux[(size_t)t * 4]);
                    const float rel_a = abs_a / std::max(std::fabs(h_ref_aux[(size_t)t * 4]), 1e-6f);
                    if (rel_a > 5e-3f && abs_a > 1e-4f) {
                        std::fprintf(stderr, "FAIL bf16 aux rows=%lld k=%lld n_tok=%d t=%d\n",
                                     (long long)sh.rows, (long long)sh.k_dim, n_tok, t);
                        ++g_failures;
                    }
                }
            }
        }

        sycl::free(d_x, q);
        sycl::free(d_w, q);
        sycl::free(d_w_aux, q);
        sycl::free(d_y_rows, q);
        sycl::free(d_y_multi, q);
        sycl::free(d_y_aux, q);
    }
    unsetenv("STRATA_MMVF_ROWS");
    std::printf("  native_bf16 parity: max_abs_err=%.3e, max_rel_err=%.3e\n", worst_abs, worst_rel);
}

// ---- 4. Benchmarks (--bench) ----

template <typename Fn>
double bench_us(sycl::queue& q, Fn&& fn, int warmup = 5, int iters = 30) {
    for (int i = 0; i < warmup; ++i) fn();
    q.wait();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    q.wait();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / (double)iters;
}

void bench_qsa_topk(sycl::queue& q) {
    std::printf("\n=== Benchmark: qsa_block_topk (reg vs wide) ===\n");
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    std::mt19937 rng(777u);
    std::uniform_real_distribution<float> ur(0.0f, 100.0f);

    struct Case {
        int nq;
        int64_t n_blocks;
        bool counted;
    };
    const Case cases[] = {
        {1, 2048, false},    // decode
        {4, 2048, false},    // decode window
        {1, 16384, false},   // 64K decode
        {4, 16384, false},   // 64K decode window
        {64, 2048, true},    // prefill 8K
        {128, 2048, true},   // prefill 8K
        {256, 2048, true},   // prefill 8K
        {64, 16384, true},   // prefill 64K
        {128, 16384, true},  // prefill 64K
        {256, 16384, true},  // prefill 64K
    };

    for (const auto& c : cases) {
        const int64_t max_blocks = c.n_blocks + 32;
        const int64_t active_blocks = c.counted ? c.n_blocks : -1;
        const std::vector<int32_t> steps = make_steps(c.nq, c.n_blocks * s.idx_block - 1, s, false);
        std::vector<float> sc((size_t)c.nq * (size_t)max_blocks);
        for (auto& v : sc) v = ur(rng);

        float* d_sc = sycl::malloc_device<float>(sc.size(), q);
        int32_t* d_st = sycl::malloc_device<int32_t>(steps.size(), q);
        int32_t* d_ids = sycl::malloc_device<int32_t>((size_t)c.nq * (size_t)cap, q);
        q.memcpy(d_sc, sc.data(), sc.size() * sizeof(float));
        q.memcpy(d_st, steps.data(), steps.size() * sizeof(int32_t));
        q.wait();

        setenv("STRATA_QSA_TOPK_PREFILL_WIDE", "0", 1);
        const double us_reg = bench_us(q, [&] {
            k::qsa_block_topk(d_sc, d_st, c.nq, max_blocks, cap, s, d_ids, &q, active_blocks);
        });
        setenv("STRATA_QSA_TOPK_PREFILL_WIDE", "1", 1);
        const double us_wide = bench_us(q, [&] {
            k::qsa_block_topk(d_sc, d_st, c.nq, max_blocks, cap, s, d_ids, &q, active_blocks);
        });
        unsetenv("STRATA_QSA_TOPK_PREFILL_WIDE");
        const double us_def = bench_us(q, [&] {
            k::qsa_block_topk(d_sc, d_st, c.nq, max_blocks, cap, s, d_ids, &q, active_blocks);
        });

        std::printf("  %-7s nq=%3d nb=%5lld: reg=%8.2f us | wide=%8.2f us | default=%8.2f us (wide speedup=%.2fx)\n",
                    c.counted ? "prefill" : "decode", c.nq, (long long)c.n_blocks, us_reg, us_wide, us_def,
                    us_reg / us_wide);

        sycl::free(d_sc, q);
        sycl::free(d_st, q);
        sycl::free(d_ids, q);
    }
}

void bench_native_bf16(sycl::queue& q) {
    std::printf("\n=== Benchmark: bf16_f32_mmvf_rows & multi (n_tok = 1, 2, 4, 8) ===\n");
    struct Shape {
        int64_t rows;
        int64_t k_dim;
    };
    const Shape shapes[] = {
        {4096, 2560},
        {16384, 2560},
    };
    std::mt19937 rng(888u);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (const auto& sh : shapes) {
        std::vector<float> h_x(8 * (size_t)sh.k_dim);
        std::vector<uint16_t> h_w((size_t)sh.rows * (size_t)sh.k_dim);
        for (auto& v : h_x) v = dist(rng);
        for (auto& v : h_w) v = k::bf16_from_f32(dist(rng) * 0.05f);

        float* d_x = sycl::malloc_device<float>(h_x.size(), q);
        uint16_t* d_w = sycl::malloc_device<uint16_t>(h_w.size(), q);
        float* d_y = sycl::malloc_device<float>(8 * (size_t)sh.rows, q);
        q.memcpy(d_x, h_x.data(), h_x.size() * sizeof(float));
        q.memcpy(d_w, h_w.data(), h_w.size() * sizeof(uint16_t));
        q.wait();

        for (int n_tok : {1, 2, 4, 8}) {
            const double bytes = (double)sh.rows * (double)sh.k_dim * 2.0 +
                                 (double)n_tok * (double)sh.k_dim * 4.0 +
                                 (double)n_tok * (double)sh.rows * 4.0;

            setenv("STRATA_MMVF_ROWS", "1", 1);
            const double us_rows = bench_us(q, [&] {
                k::bf16_gemv_fp32_mmvf_multi(d_x, sh.k_dim, d_w, d_y, sh.rows, sh.k_dim, sh.rows, n_tok, &q);
            });
            setenv("STRATA_MMVF_ROWS", "0", 1);
            const double us_multi = bench_us(q, [&] {
                k::bf16_gemv_fp32_mmvf_multi(d_x, sh.k_dim, d_w, d_y, sh.rows, sh.k_dim, sh.rows, n_tok, &q);
            });
            unsetenv("STRATA_MMVF_ROWS");

            const double gbs_rows = (bytes / 1e9) / (us_rows * 1e-6);
            const double gbs_multi = (bytes / 1e9) / (us_multi * 1e-6);
            std::printf("  [%5lld x %4lld] n_tok=%d: rows=%8.2f us (%6.2f GB/s) | multi=%8.2f us (%6.2f GB/s)\n",
                        (long long)sh.rows, (long long)sh.k_dim, n_tok, us_rows, gbs_rows, us_multi, gbs_multi);
        }

        sycl::free(d_x, q);
        sycl::free(d_w, q);
        sycl::free(d_y, q);
    }
}

void bench_priority_queue_contention(sycl::queue& base_q) {
    std::printf("\n=== Benchmark: SYCL Priority Queue Contention (high vs normal) ===\n");
    bool got_high = false;
    sycl::queue q_bg(base_q.get_context(), base_q.get_device(), sycl::property::queue::in_order{});
    sycl::queue q_norm = strata::make_in_order_queue(base_q.get_context(), base_q.get_device(), false, nullptr);
    sycl::queue q_high = strata::make_in_order_queue(base_q.get_context(), base_q.get_device(), true, &got_high);

    constexpr size_t BG_N = 1 << 20;
    constexpr size_t FG_N = 4096;
    float* d_bg = sycl::malloc_device<float>(BG_N, base_q);
    float* d_fg = sycl::malloc_device<float>(FG_N, base_q);
    q_bg.fill(d_bg, 1.0f, BG_N);
    q_bg.fill(d_fg, 1.0f, FG_N);
    q_bg.wait();

    auto measure_under_load = [&](sycl::queue& q_fg) {
        constexpr int iters = 20;
        double sum_us = 0.0;
        for (int it = 0; it < iters; ++it) {
            // Submit background load
            for (int r = 0; r < 6; ++r) {
                q_bg.parallel_for(sycl::range<1>(BG_N), [=](sycl::id<1> i) {
                    float v = d_bg[i];
                    for (int s = 0; s < 32; ++s) v = sycl::fma(v, 1.0001f, 0.0001f);
                    d_bg[i] = v;
                });
            }
            const auto t0 = std::chrono::steady_clock::now();
            q_fg.parallel_for(sycl::range<1>(FG_N), [=](sycl::id<1> i) {
                d_fg[i] = d_fg[i] * 1.001f + 0.1f;
            });
            q_fg.wait();
            const auto t1 = std::chrono::steady_clock::now();
            q_bg.wait();
            sum_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        return sum_us / (double)iters;
    };

    // Warmup
    (void)measure_under_load(q_norm);
    const double us_norm = measure_under_load(q_norm);
    const double us_high = measure_under_load(q_high);
    std::printf("  got_high=%d: normal_fg=%.2f us | high_prio_fg=%.2f us (ratio=%.2fx)\n",
                (int)got_high, us_norm, us_high, us_norm / us_high);

    sycl::free(d_bg, base_q);
    sycl::free(d_fg, base_q);
}

}  // namespace

int main(int argc, char** argv) {
    bool do_bench = false;
    bool bench_only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) do_bench = true;
        if (std::strcmp(argv[i], "--bench-only") == 0) {
            do_bench = true;
            bench_only = true;
        }
    }

    sycl::queue q;
    try {
        q = sycl::queue(sycl::gpu_selector_v, sycl::property::queue::in_order());
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "sycl_p2_kernels_bench: no SYCL GPU device (%s)\n", e.what());
        return 77;
    }

    const auto dev = q.get_device();
    std::printf("Device: %s (SLM %zu B, CUs %u)\n",
                dev.get_info<sycl::info::device::name>().c_str(),
                (size_t)dev.get_info<sycl::info::device::local_mem_size>(),
                dev.get_info<sycl::info::device::max_compute_units>());

    if (!bench_only) {
        std::printf("Running P2 correctness tests...\n");
        test_priority_queue_smoke(q);
        test_qsa_topk_parity(q);
        test_native_bf16_parity(q);
        std::printf("sycl_p2_kernels_bench: %s (failures: %d)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    }

    if (do_bench && g_failures == 0) {
        bench_priority_queue_contention(q);
        bench_qsa_topk(q);
        bench_native_bf16(q);
    }

    return g_failures ? 1 : 0;
}
