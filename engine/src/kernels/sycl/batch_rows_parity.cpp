// src/kernels/sycl/batch_rows_parity.cpp - the batch window's one-launch-for-all-rows kernels (#2234, STRATA_BATCH_ATTN_ROWS /
// STRATA_BATCH_GDN_GROUPS) against the per-row launches they replace, bitwise (GPU, synthetic, no model).
//
//   1. qsa_block_scores_rows   == qsa_block_scores, one call per row, each row with its OWN indexer keys;
//   2. qsa_decode_attn_batch_rows == qsa_decode_attn_batch, one call per row, each row over its OWN K/V pools
//      (int8 codes + scales, and fp16);
//   3. gdn_step_norm_groups    == gdn_step_norm_multi, one call per slot group, in the window (all outputs, the state
//      untouched) and in the 1-row commit (the state written after `keep` tokens).
// Every case compares with memcmp, not a tolerance. Usage: batch_rows_parity
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
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
    T* d = sycl::malloc_device<T>(h.size() + 16, q);
    q.memcpy(d, h.data(), h.size() * sizeof(T)).wait();
    return d;
}

template <typename T>
std::vector<T> down(const T* d, size_t n, sycl::queue& q) {
    std::vector<T> h(n);
    q.memcpy(h.data(), d, n * sizeof(T)).wait();
    return h;
}

uint16_t f2h(float f) { return k::f16_from_f32(f); }

int bad = 0;
void report(const char* what, bool same) {
    std::printf("  %-64s %s\n", what, same ? "bitwise" : "*** DIFFERS ***");
    if (!same) ++bad;
}

// ---- 1. the block scores ----
void scores_case(int rows, sycl::queue& q) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t max_blocks = 3000;
    std::mt19937 rng(11 + rows);
    std::normal_distribution<float> nd(0.f, 1.f);
    k::QsaScoreRows sr;
    sr.n = rows;
    std::vector<const float*> pooled(rows), dead(rows);
    std::vector<float> qv((size_t) rows * s.idx_n_head * s.idx_dim);
    for (auto& x : qv) x = nd(rng);
    std::vector<int32_t> steps((size_t) rows * k::kStepCount);
    for (int r = 0; r < rows; ++r) {
        std::vector<float> p((size_t) max_blocks * s.idx_dim), d((size_t) s.idx_dim);
        for (auto& x : p) x = nd(rng);
        for (auto& x : d) x = nd(rng);
        pooled[r] = up(p, q);
        dead[r] = up(d, q);
        sr.pooled[r] = pooled[r];
        sr.dead[r] = dead[r];
        const int64_t pos = 700 + 997 * r + (r % 3) * 5;   // different lengths, and an incomplete block at the tail
        steps[(size_t) r * k::kStepCount + k::kStepPos] = (int32_t) pos;
        steps[(size_t) r * k::kStepCount + k::kStepNKv] = (int32_t) (pos + 1);
        steps[(size_t) r * k::kStepCount + k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
        steps[(size_t) r * k::kStepCount + k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
    }
    const float* d_q = up(qv, q);
    const int32_t* d_steps = up(steps, q);
    const size_t n = (size_t) rows * max_blocks;
    float* d_rows = sycl::malloc_device<float>(n + 16, q);
    float* d_ref = sycl::malloc_device<float>(n + 16, q);
    q.memset(d_rows, 0, n * sizeof(float)).wait();
    q.memset(d_ref, 0, n * sizeof(float)).wait();

    k::qsa_block_scores_rows(sr, d_q, d_steps, rows, max_blocks, s, d_rows, &q);
    for (int r = 0; r < rows; ++r)
        k::qsa_block_scores(pooled[r], dead[r], d_q + (size_t) r * s.idx_n_head * s.idx_dim, d_steps + (size_t) r * k::kStepCount, 1,
                            max_blocks, s, d_ref + (size_t) r * max_blocks, &q);
    q.wait();
    const std::vector<float> a = down(d_rows, n, q), b = down(d_ref, n, q);
    char name[96];
    std::snprintf(name, sizeof name, "qsa_block_scores_rows, %d rows (own keys each)", rows);
    report(name, std::memcmp(a.data(), b.data(), n * sizeof(float)) == 0);

    // the window selects with one top-k launch over all rows (row-independent) where the per-row path ran one per row
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    const size_t cap_n = (size_t) rows * cap;
    int32_t* i_rows = sycl::malloc_device<int32_t>(cap_n + 16, q);
    int32_t* i_ref = sycl::malloc_device<int32_t>(cap_n + 16, q);
    q.memset(i_rows, 0, cap_n * sizeof(int32_t)).wait();
    q.memset(i_ref, 0, cap_n * sizeof(int32_t)).wait();

    k::qsa_block_topk(d_ref, d_steps, rows, max_blocks, cap, s, i_rows, &q);
    for (int r = 0; r < rows; ++r)
        k::qsa_block_topk(d_ref + (size_t) r * max_blocks, d_steps + (size_t) r * k::kStepCount, 1, max_blocks, cap, s,
                          i_ref + (size_t) r * cap, &q);
    q.wait();
    const std::vector<int32_t> ta = down(i_rows, cap_n, q), tb = down(i_ref, cap_n, q);
    bool tsame = true;
    for (int r = 0; r < rows; ++r) {
        const int64_t w = steps[(size_t) r * k::kStepCount + k::kStepWidth];
        tsame = tsame && std::memcmp(ta.data() + (size_t) r * cap, tb.data() + (size_t) r * cap, (size_t) w * sizeof(int32_t)) == 0;
    }
    std::snprintf(name, sizeof name, "qsa_block_topk over %d rows == one call per row", rows);
    report(name, tsame);

    sycl::free(d_rows, q); sycl::free(d_ref, q);
    sycl::free(i_rows, q); sycl::free(i_ref, q);
    sycl::free(const_cast<float*>(d_q), q);
    sycl::free(const_cast<int32_t*>(d_steps), q);
    for (int r = 0; r < rows; ++r) {
        sycl::free(const_cast<float*>(pooled[r]), q);
        sycl::free(const_cast<float*>(dead[r]), q);
    }
}

// ---- 2. the decode attention ----
void attn_case(int rows, bool int8, sycl::queue& q) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t HD = s.head_dim, NKV = s.n_head_kv, NH = s.n_head, PS = s.page_size;
    const int64_t ctx_max = 5000, pages = (ctx_max + PS - 1) / PS, prow = pages * NKV * PS;
    const int64_t cap = k::qsa_selection_width(ctx_max, s);
    std::mt19937 rng(23 + rows + (int8 ? 1 : 0));
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> sc(0.005f, 0.03f);
    k::QsaAttnPoolsRows pr;
    pr.n = rows;
    std::vector<float> qv((size_t) rows * NH * HD);
    for (auto& x : qv) x = nd(rng) * 2.0f;
    std::vector<int32_t> ids((size_t) rows * cap, 0), steps((size_t) rows * k::kStepCount, 0);
    std::vector<void*> to_free;
    for (int r = 0; r < rows; ++r) {
        k::QsaAttnPools& pl = pr.p[r];
        if (int8) {
            std::vector<int8_t> kq((size_t) prow * HD), vq((size_t) prow * HD);
            std::vector<uint16_t> ks((size_t) prow * (HD / 64)), vs(ks.size());
            for (auto& x : kq) x = (int8_t) code(rng);
            for (auto& x : vq) x = (int8_t) code(rng);
            for (auto& x : ks) x = f2h(sc(rng));
            for (auto& x : vs) x = f2h(sc(rng));
            pl.k_q = up(kq, q); pl.v_q = up(vq, q); pl.k_scale = up(ks, q); pl.v_scale = up(vs, q);
            to_free.push_back(const_cast<int8_t*>(pl.k_q));
            to_free.push_back(const_cast<int8_t*>(pl.v_q));
            to_free.push_back(const_cast<uint16_t*>(pl.k_scale));
            to_free.push_back(const_cast<uint16_t*>(pl.v_scale));
        } else {
            std::vector<uint16_t> kh((size_t) prow * HD), vh(kh.size());
            for (auto& x : kh) x = f2h(nd(rng) * 1.5f);
            for (auto& x : vh) x = f2h(nd(rng));
            pl.k_pool = up(kh, q); pl.v_pool = up(vh, q);
            to_free.push_back(const_cast<uint16_t*>(pl.k_pool));
            to_free.push_back(const_cast<uint16_t*>(pl.v_pool));
        }
        std::vector<int32_t> table(pages);
        for (int64_t i = 0; i < pages; ++i) table[i] = (int32_t) i;
        std::shuffle(table.begin(), table.end(), rng);   // each row's pool has its own page order
        pl.page_table = up(table, q);
        to_free.push_back(const_cast<int32_t*>(pl.page_table));
        const int64_t nkv = 900 + 641 * r + (r % 2) * 3;
        const int64_t w = k::qsa_selection_width(nkv, s);
        std::vector<int32_t> all((size_t) nkv);
        for (int64_t c = 0; c < nkv; ++c) all[c] = (int32_t) c;
        std::shuffle(all.begin(), all.end(), rng);
        std::sort(all.begin(), all.begin() + w);
        std::copy(all.begin(), all.begin() + w, ids.begin() + (size_t) r * cap);
        steps[(size_t) r * k::kStepCount + k::kStepPos] = (int32_t) (nkv - 1);
        steps[(size_t) r * k::kStepCount + k::kStepNKv] = (int32_t) nkv;
        steps[(size_t) r * k::kStepCount + k::kStepNBid] = (int32_t) (nkv / s.idx_block);
        steps[(size_t) r * k::kStepCount + k::kStepWidth] = (int32_t) w;
    }
    const float* d_q = up(qv, q);
    const int32_t* d_ids = up(ids, q);
    const int32_t* d_steps = up(steps, q);
    const int64_t scr = (int64_t) k::qsa_decode_attn_scratch_floats(cap, s);
    float* scratch = sycl::malloc_device<float>((size_t) rows * scr + 16, q);
    const size_t n = (size_t) rows * NH * HD;
    float* d_rows = sycl::malloc_device<float>(n + 16, q);
    float* d_ref = sycl::malloc_device<float>(n + 16, q);
    q.memset(d_rows, 0, n * sizeof(float)).wait();
    q.memset(d_ref, 0, n * sizeof(float)).wait();

    k::qsa_decode_attn_batch_rows(d_q, pr, d_ids, d_steps, cap, s, scratch, d_rows, rows, &q);
    for (int r = 0; r < rows; ++r)
        k::qsa_decode_attn_batch(d_q + (size_t) r * NH * HD, pr.p[r], d_ids + (size_t) r * cap, d_steps + (size_t) r * k::kStepCount, cap,
                                 s, scratch + (size_t) r * scr, d_ref + (size_t) r * NH * HD, 1, &q);
    q.wait();
    const std::vector<float> a = down(d_rows, n, q), b = down(d_ref, n, q);
    char name[96];
    std::snprintf(name, sizeof name, "qsa_decode_attn_batch_rows, %d rows, %s pools (own pools each)", rows, int8 ? "int8" : "fp16");
    report(name, std::memcmp(a.data(), b.data(), n * sizeof(float)) == 0);

    sycl::free(scratch, q); sycl::free(d_rows, q); sycl::free(d_ref, q);
    sycl::free(const_cast<float*>(d_q), q);
    sycl::free(const_cast<int32_t*>(d_ids), q);
    sycl::free(const_cast<int32_t*>(d_steps), q);
    for (void* p : to_free) sycl::free(p, q);
}

// ---- 3. the GDN recurrence ----
void gdn_case(const std::vector<int>& ntok, bool commit, sycl::queue& q) {
    const int S = 128, h_k = 16, h_v = 48;   // the artifact's geometry
    const int qk = S * h_k, C = 2 * qk + S * h_v, vd = S * h_v;
    const float eps = 1e-6f;
    int T = 0;
    for (int n : ntok) T += n;
    const int G = (int) ntok.size();
    std::mt19937 rng(31 + T * 7 + G + (commit ? 1 : 0));
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hv((size_t) T * C), gate((size_t) T * h_v), beta((size_t) T * h_v), z((size_t) T * vd), gamma(S);
    for (auto& v : hv) v = nd(rng) * 0.1f;
    for (auto& v : gate) v = -(0.5f + 3.0f * (float) (rng() % 100) / 100.0f);
    for (auto& v : beta) v = (float) (rng() % 100) / 100.0f;
    for (auto& v : z) v = nd(rng);
    for (auto& v : gamma) v = 1.0f + 0.1f * nd(rng);
    const size_t st_n = (size_t) S * h_v * S;
    std::vector<std::vector<float>> st0(G);
    std::vector<float*> st_g(G), st_r(G);
    std::vector<int32_t*> keep(G);
    k::GdnGroups gg;
    gg.n = G;
    int first = 0;
    for (int i = 0; i < G; ++i) {
        st0[i].resize(st_n);
        for (auto& v : st0[i]) v = nd(rng) * 0.1f;
        st_g[i] = up(st0[i], q);
        st_r[i] = up(st0[i], q);
        const int32_t kp = commit ? ntok[i] : 0;   // a commit group keeps all of its rows here
        keep[i] = up(std::vector<int32_t>{kp}, q);
        gg.state[i] = st_g[i];
        gg.keep[i] = commit ? keep[i] : nullptr;
        gg.first[i] = first;
        gg.ntok[i] = ntok[i];
        first += ntok[i];
    }
    const float* d_h = up(hv, q);
    const float* d_gate = up(gate, q);
    const float* d_beta = up(beta, q);
    const float* d_z = up(z, q);
    const float* d_gm = up(gamma, q);
    float* d_y = sycl::malloc_device<float>((size_t) T * vd + 16, q);
    float* d_yr = sycl::malloc_device<float>((size_t) T * vd + 16, q);
    q.memset(d_y, 0, (size_t) T * vd * sizeof(float)).wait();
    q.memset(d_yr, 0, (size_t) T * vd * sizeof(float)).wait();

    k::gdn_step_norm_groups(gg, d_h, C, d_gate, d_beta, d_z, d_gm, eps, d_y, h_k, h_v, commit, &q);
    for (int i = 0; i < G; ++i) {
        const int f = gg.first[i];
        k::gdn_step_norm_multi(st_r[i], d_h + (size_t) f * C, C, d_gate + (size_t) f * h_v, d_beta + (size_t) f * h_v,
                               d_z + (size_t) f * vd, d_gm, eps, d_yr + (size_t) f * vd, h_k, h_v, ntok[i],
                               commit ? keep[i] : nullptr, &q, 0);
    }
    q.wait();
    bool same = true;
    for (int i = 0; i < G; ++i) {
        const std::vector<float> a = down(st_g[i], st_n, q), b = down(st_r[i], st_n, q);
        same = same && std::memcmp(a.data(), b.data(), st_n * sizeof(float)) == 0;
        if (!commit) same = same && std::memcmp(a.data(), st0[i].data(), st_n * sizeof(float)) == 0;
    }
    if (!commit) {
        const std::vector<float> a = down(d_y, (size_t) T * vd, q), b = down(d_yr, (size_t) T * vd, q);
        same = same && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
    }
    char name[96], rows[64] = "";
    for (int i = 0; i < G && i < 8; ++i) std::snprintf(rows + std::strlen(rows), sizeof rows - std::strlen(rows), "%s%d", i ? "," : "", ntok[i]);
    std::snprintf(name, sizeof name, "gdn_step_norm_groups %s, groups of %s rows", commit ? "commit" : "window", rows);
    report(name, same);

    sycl::free(d_y, q); sycl::free(d_yr, q);
    sycl::free(const_cast<float*>(d_h), q);
    sycl::free(const_cast<float*>(d_gate), q);
    sycl::free(const_cast<float*>(d_beta), q);
    sycl::free(const_cast<float*>(d_z), q);
    sycl::free(const_cast<float*>(d_gm), q);
    for (int i = 0; i < G; ++i) {
        sycl::free(st_g[i], q);
        sycl::free(st_r[i], q);
        sycl::free(keep[i], q);
    }
}
}  // namespace

int main() try {
    auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.empty()) {
        std::puts("SKIP: no GPU device");
        return 77;
    }
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    std::printf("batch rows parity (#2234 kernels against the per-row launches)\n");
    for (int r : {1, 2, 4, 8}) scores_case(r, q);
    for (int r : {1, 3, 8}) {
        attn_case(r, true, q);
        attn_case(r, false, q);
    }
    gdn_case({1}, false, q);
    gdn_case({1, 1, 1, 1}, false, q);
    gdn_case({2, 2, 2, 2}, false, q);
    gdn_case({1, 3, 2, 1, 1}, false, q);
    gdn_case({1, 1, 1, 1, 1, 1, 1, 1}, false, q);
    gdn_case({1}, true, q);
    gdn_case({1, 1, 1, 1}, true, q);
    gdn_case({1, 1, 1, 1, 1, 1, 1, 1}, true, q);
    std::printf("%s\n", bad ? "batch_rows_parity: FAILED" : "batch_rows_parity: ok");
    return bad ? 1 : 0;
} catch (const sycl::exception& e) {
    std::fprintf(stderr, "SYCL exception: %s\n", e.what());
    return 1;
}
