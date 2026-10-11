// src/kernels/sycl/qsa_select.dp.cpp - Pure SYCL 2020 migration of qsa_select
#include <sycl/sycl.hpp>
#include "strata/core/emulate.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {

constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4;
constexpr int SCORE_WARPS = 8;
constexpr int TOPK_T = 256;
constexpr int MQ = 8;
constexpr int TK_T = 1024;
constexpr int TK_PER = 33;

inline uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = sycl::bit_cast<unsigned int>(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

void block_scores_kernel(const float* __restrict__ pooled, const float* __restrict__ dead,
                         const float* __restrict__ q_idx, const int32_t* __restrict__ steps, int64_t max_blocks,
                         float* __restrict__ out, sycl::nd_item<2> item) {
    auto sg = item.get_sub_group();
    const int64_t qi = item.get_group(0);
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    const int64_t b = (int64_t)item.get_group(1) * SCORE_WARPS + (item.get_local_id(1) >> 5);
    if (b > n_bid || b >= max_blocks) return;
    const int lane = item.get_local_id(1) & 31;
    const float* key = (b == n_bid) ? dead : pooled + b * IDX_DIM;
    const sycl::float4 k4 = *reinterpret_cast<const sycl::float4*>(key + lane * 4);
    const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const sycl::float4 q4 = *reinterpret_cast<const sycl::float4*>(q + h * IDX_DIM);
        float d = k4.x() * q4.x() + k4.y() * q4.y() + k4.z() * q4.z() + k4.w() * q4.w();
        d = sycl::reduce_over_group(sg, d, sycl::plus<float>());
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (b == n_bid && n_kv % R != 0) score += 1e9f;
        out[qi * max_blocks + b] = score;
    }
}

void block_scores_multi_kernel(const float* __restrict__ pooled, const float* __restrict__ dead,
                               const float* __restrict__ q_idx, const int32_t* __restrict__ steps, int nq,
                               int64_t max_blocks, float* __restrict__ out, float* qs, int64_t* s_nkv, int64_t* s_nbid,
                               sycl::nd_item<1> item) {
    auto sg = item.get_sub_group();
    const int lid = item.get_local_id(0);
    const int lrange = item.get_local_range(0);

    for (int i = lid; i < nq * IDX_HEADS * IDX_DIM; i += lrange)
        qs[i] = q_idx[i];
    if (lid < nq) {
        s_nkv[lid] = steps[lid * kStepCount + kStepNKv];
        s_nbid[lid] = steps[lid * kStepCount + kStepNBid];
    }
    item.barrier(sycl::access::fence_space::local_space);
    int64_t top = 0;
    for (int q = 0; q < nq; ++q)
        top = s_nbid[q] > top ? s_nbid[q] : top;
    const int lane = lid & 31;
    const int64_t wstride = (int64_t)item.get_group_range(0) * SCORE_WARPS;
    for (int64_t b = (int64_t)item.get_group(0) * SCORE_WARPS + (lid >> 5); b <= top && b < max_blocks; b += wstride) {
        const sycl::float4 kp = *reinterpret_cast<const sycl::float4*>(pooled + b * IDX_DIM + lane * 4);
        const sycl::float4 kd = *reinterpret_cast<const sycl::float4*>(dead + lane * 4);
        for (int qi = 0; qi < nq; ++qi) {
            const int64_t n_bid = s_nbid[qi];
            if (b > n_bid) continue;
            const sycl::float4 k4 = (b == n_bid) ? kd : kp;
            const float* q = qs + qi * IDX_HEADS * IDX_DIM + lane * 4;
            float score = 0.0f;
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const sycl::float4 q4 = *reinterpret_cast<const sycl::float4*>(q + h * IDX_DIM);
                float d = k4.x() * q4.x() + k4.y() * q4.y() + k4.z() * q4.z() + k4.w() * q4.w();
                d = sycl::reduce_over_group(sg, d, sycl::plus<float>());
                score += d > 0.0f ? d : 0.0f;
            }
            if (lane == 0) {
                if (b == n_bid && s_nkv[qi] % R != 0) score += 1e9f;
                out[qi * max_blocks + b] = score;
            }
        }
    }
}

void block_topk_kernel(const float* __restrict__ scores, const int32_t* __restrict__ steps, int64_t max_blocks,
                       int64_t cap, int32_t* __restrict__ ids, int* hist, int* s_a, int* s_b, int& s_digit,
                       int& s_above, sycl::nd_item<1> item) {
    const int64_t qi = item.get_group(0);
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = item.get_local_id(0);
    if (n_kv <= width) {
        for (int64_t j = t; j < n_kv; j += TOPK_T)
            out[j] = (int32_t)j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;
    const int64_t per = (nb + TOPK_T - 1) / TOPK_T;
    const int64_t b0 = (int64_t)t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int)(n_kv - n_bid * R); };

    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 256; i += TOPK_T)
            hist[i] = 0;
        item.barrier(sycl::access::fence_space::local_space);
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int64_t b = b0; b < b1; ++b) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask)) {
                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                 sycl::access::address_space::local_space>(hist[(k >> shift) & 255])
                    .fetch_add(w);
            }
        }
        item.barrier(sycl::access::fence_space::local_space);
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[d] >= width) break;
                cum += hist[d];
            }
            s_digit = d;
            s_above = cum;
        }
        item.barrier(sycl::access::fence_space::local_space);
        prefix |= (uint32_t)s_digit << shift;
        above = s_above;
        item.barrier(sycl::access::fence_space::local_space);
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;
    int gt = 0, eq = 0;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr)
            gt += w;
        else if (k == thr)
            eq += w;
    }
    s_a[t] = gt;
    s_b[t] = eq;
    item.barrier(sycl::access::fence_space::local_space);
    if (t == 0) {
        int ag = 0, ae = 0;
        for (int i = 0; i < TOPK_T; ++i) {
            const int g = s_a[i], e = s_b[i];
            s_a[i] = ag;
            s_b[i] = ae;
            ag += g;
            ae += e;
        }
    }
    item.barrier(sycl::access::fence_space::local_space);
    const int64_t eq_before = s_b[t];
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int)my_eq;
    item.barrier(sycl::access::fence_space::local_space);
    s_a[t] = sel;
    item.barrier(sycl::access::fence_space::local_space);
    if (t == 0) {
        int a = 0;
        for (int i = 0; i < TOPK_T; ++i) {
            const int c = s_a[i];
            s_a[i] = a;
            a += c;
        }
    }
    item.barrier(sycl::access::fence_space::local_space);
    int64_t wpos = s_a[t];
    int64_t eq_left = my_eq;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) {
            for (int c = 0; c < w; ++c)
                out[wpos++] = (int32_t)(b * R + c);
        } else if (k == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left)
                out[wpos++] = (int32_t)(b * R + c);
        }
    }
}

inline int block_excl_scan(sycl::nd_item<1> item, int v, int* s_warp, int& total) {
    auto sg = item.get_sub_group();
    const int lane = item.get_local_id(0) & 31, warp = item.get_local_id(0) >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = sycl::shift_group_right(sg, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s_warp[warp] = x;
    item.barrier(sycl::access::fence_space::local_space);
    if (warp == 0) {
        int w = s_warp[lane];
        int z = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = sycl::shift_group_right(sg, z, o);
            if (lane >= o) z += y;
        }
        s_warp[lane] = z - w;
        if (lane == 31) s_warp[32] = z;
    }
    item.barrier(sycl::access::fence_space::local_space);
    const int r = s_warp[warp] + x - v;
    total = s_warp[32];
    item.barrier(sycl::access::fence_space::local_space);
    return r;
}

template <int PER>
void block_topk_reg_kernel(const float* __restrict__ scores, const int32_t* __restrict__ steps, int64_t max_blocks,
                           int64_t cap, int32_t* __restrict__ ids, int hist[32][256], int* s_warp, int& s_digit,
                           int& s_above, sycl::nd_item<1> item) {
    const int64_t qi = item.get_group(0);
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = item.get_local_id(0), lane = t & 31, warp = t >> 5;
    if (n_kv <= width) {
        for (int64_t j = t; j < n_kv; j += TK_T)
            out[j] = (int32_t)j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;
    const int64_t per = (nb + TK_T - 1) / TK_T;
    const int64_t b0 = (int64_t)t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    uint32_t key[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j)
        key[j] = (b0 + j < b1) ? order_key(sc[b0 + j]) : 0u;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int)(n_kv - n_bid * R); };
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
#pragma unroll
        for (int d = 0; d < 8; ++d)
            hist[warp][d * 32 + lane] = 0;
        item.barrier(sycl::access::fence_space::local_space);
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t b = b0 + j;
            if (b >= b1) break;
            const int w = weight(b);
            if (w == 0) continue;
            if ((key[j] & hi_mask) == (prefix & hi_mask)) {
                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                 sycl::access::address_space::local_space>(hist[warp][(key[j] >> shift) & 255])
                    .fetch_add(w);
            }
        }
        item.barrier(sycl::access::fence_space::local_space);
        if (t < 256) {
            int s = 0;
            for (int w2 = 0; w2 < TK_T / 32; ++w2)
                s += hist[w2][t];
            hist[0][t] = s;
        }
        item.barrier(sycl::access::fence_space::local_space);
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[0][d] >= width) break;
                cum += hist[0][d];
            }
            s_digit = d;
            s_above = cum;
        }
        item.barrier(sycl::access::fence_space::local_space);
        prefix |= (uint32_t)s_digit << shift;
        above = s_above;
        item.barrier(sycl::access::fence_space::local_space);
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;
    int gt = 0, eq = 0;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) break;
        const int w = weight(b);
        if (w == 0) continue;
        if (key[j] > thr)
            gt += w;
        else if (key[j] == thr)
            eq += w;
    }
    int tot;
    const int eq_before = block_excl_scan(item, eq, s_warp, tot);
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int)my_eq;
    int64_t wpos = block_excl_scan(item, sel, s_warp, tot);
    int64_t eq_left = my_eq;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) break;
        const int w = weight(b);
        if (w == 0) continue;
        if (key[j] > thr) {
            for (int c = 0; c < w; ++c)
                out[wpos++] = (int32_t)(b * R + c);
        } else if (key[j] == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left)
                out[wpos++] = (int32_t)(b * R + c);
        }
    }
}

void block_topk_wide_kernel(const float* __restrict__ scores, const int32_t* __restrict__ steps, int64_t max_blocks,
                            int64_t cap, int32_t* __restrict__ ids,
                            int (*hist)[256], int* s_gt, int* s_eq, int& s_digit, int& s_above,
                            sycl::nd_item<1> item) {
    auto sg = item.get_sub_group();
    const int64_t qi = item.get_group(0);
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = item.get_local_id(0), lane = t & 31, warp = t >> 5;
    if (n_kv <= width) {
        for (int64_t j = t; j < n_kv; j += TK_T) out[j] = (int32_t)j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;
    const int64_t per = (nb + TK_T - 1) / TK_T;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int)(n_kv - n_bid * R); };
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = lane; i < 256; i += 32) hist[warp][i] = 0;
        item.barrier(sycl::access::fence_space::local_space);
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int64_t b = t; b < nb; b += TK_T) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask)) {
                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                 sycl::access::address_space::local_space>(hist[warp][(k >> shift) & 255])
                    .fetch_add(w);
            }
        }
        item.barrier(sycl::access::fence_space::local_space);
        if (t < 256) {
            int sum = 0;
            for (int w2 = 0; w2 < TK_T / 32; ++w2) sum += hist[w2][t];
            hist[0][t] = sum;
        }
        item.barrier(sycl::access::fence_space::local_space);
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[0][d] >= width) break;
                cum += hist[0][d];
            }
            s_digit = d;
            s_above = cum;
        }
        item.barrier(sycl::access::fence_space::local_space);
        prefix |= (uint32_t)s_digit << shift;
        above = s_above;
        item.barrier(sycl::access::fence_space::local_space);
    }
    const uint32_t thr = prefix;
    const int eq_budget = (int)(width - above);
    const int64_t w0 = (int64_t)warp * 32 * per;
    int gt = 0, eq = 0;
    for (int64_t b = w0 + lane; b < nb && b < w0 + 32 * per; b += 32) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    gt = sycl::reduce_over_group(sg, gt, sycl::plus<int>());
    eq = sycl::reduce_over_group(sg, eq, sycl::plus<int>());
    if (lane == 0) { s_gt[warp] = gt; s_eq[warp] = eq; }
    item.barrier(sycl::access::fence_space::local_space);
    if (t == 0) {
        int eb = 0, sb = 0;
        for (int w2 = 0; w2 < TK_T / 32; ++w2) {
            const int g2 = s_gt[w2], e2 = s_eq[w2];
            const int take = eq_budget - eb < 0 ? 0 : (eq_budget - eb > e2 ? e2 : eq_budget - eb);
            s_gt[w2] = sb;
            s_eq[w2] = eb;
            sb += g2 + take;
            eb += e2;
        }
    }
    item.barrier(sycl::access::fence_space::local_space);
    int run_sel = s_gt[warp], run_eq = s_eq[warp];
    for (int64_t r0 = w0; r0 < nb && r0 < w0 + 32 * per; r0 += 32) {
        const int64_t b = r0 + lane;
        const int w = b < nb ? weight(b) : 0;
        const uint32_t k = w ? order_key(sc[b]) : 0u;
        const int my_gt = (w && k > thr) ? w : 0, my_eq = (w && k == thr) ? w : 0;
        if (!sycl::any_of_group(sg, (my_gt | my_eq) != 0)) continue;
        int pe = my_eq;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = sycl::shift_group_right(sg, pe, o);
            if (lane >= o) pe += y;
        }
        const int left = eq_budget - (run_eq + pe - my_eq);
        const int take = left < 0 ? 0 : (left > my_eq ? my_eq : left);
        const int my_sel = my_gt + take;
        int ps = my_sel;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = sycl::shift_group_right(sg, ps, o);
            if (lane >= o) ps += y;
        }
        int32_t* dst = out + run_sel + ps - my_sel;
        for (int c = 0; c < my_sel; ++c) dst[c] = (int32_t)(b * R + c);
        run_eq += sycl::select_from_group(sg, pe, 31);
        run_sel += sycl::select_from_group(sg, ps, 31);
    }
}

}  // namespace

void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
    if (nq <= 0) return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535) {
        std::fprintf(stderr, "qsa_block_scores: unsupported indexer geometry\n");
        std::exit(1);
    }
    static sycl::queue default_queue{sycl::property::queue::in_order()};
    sycl::queue* q_ptr = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    static const bool multi = [] {
        const char* v = std::getenv("STRATA_SCORES_MULTI");
        return v == nullptr || std::atoi(v) != 0;
    }();
    if (multi && nq <= MQ && active_blocks <= 0) {
        q_ptr->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> qs_acc(sycl::range<1>(MQ * IDX_HEADS * IDX_DIM), cgh);
            sycl::local_accessor<int64_t, 1> s_nkv_acc(sycl::range<1>(MQ), cgh);
            sycl::local_accessor<int64_t, 1> s_nbid_acc(sycl::range<1>(MQ), cgh);

            cgh.parallel_for(sycl::nd_range<1>(256 * SCORE_WARPS * 32, SCORE_WARPS * 32),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                                 block_scores_multi_kernel(
                                     pooled, dead, q_idx, steps, (int)nq, max_blocks, scores,
                                     qs_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                                     s_nkv_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                                     s_nbid_acc.get_multi_ptr<sycl::access::decorated::no>().get(), item);
                             });
        });
        return;
    }
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    const unsigned grid_x = (unsigned)((reach + SCORE_WARPS - 1) / SCORE_WARPS);

    q_ptr->parallel_for(
        sycl::nd_range<2>(sycl::range<2>(nq, grid_x * SCORE_WARPS * 32), sycl::range<2>(1, SCORE_WARPS * 32)),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
            block_scores_kernel(pooled, dead, q_idx, steps, max_blocks, scores, item);
        });
}

bool qsa_block_scores_tc(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                         int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
    (void)pooled;
    (void)dead;
    (void)q_idx;
    (void)steps;
    (void)nq;
    (void)max_blocks;
    (void)s;
    (void)scores;
    (void)stream;
    (void)active_blocks;
    return false;
}

void qsa_block_topk_ref(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                        const QsaShapes& s, int32_t* ids, void* stream) {
    if (nq <= 0) return;
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    static sycl::queue default_queue{sycl::property::queue::in_order()};
    sycl::queue* q_ptr = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    q_ptr->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> hist_acc(sycl::range<1>(256), cgh);
        sycl::local_accessor<int, 1> s_a_acc(sycl::range<1>(TOPK_T), cgh);
        sycl::local_accessor<int, 1> s_b_acc(sycl::range<1>(TOPK_T), cgh);
        sycl::local_accessor<int, 0> s_digit_acc(cgh);
        sycl::local_accessor<int, 0> s_above_acc(cgh);

        cgh.parallel_for(sycl::nd_range<1>(nq * TOPK_T, TOPK_T), [=](sycl::nd_item<1> item) {
            block_topk_kernel(
                scores, steps, max_blocks, cap, ids, hist_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                s_a_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                s_b_acc.get_multi_ptr<sycl::access::decorated::no>().get(), s_digit_acc, s_above_acc, item);
        });
    });
}

bool qsa_block_topk_cluster(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                            const QsaShapes& s, int32_t* ids, void* stream) {
    (void)scores;
    (void)steps;
    (void)nq;
    (void)max_blocks;
    (void)cap;
    (void)s;
    (void)ids;
    (void)stream;
    return false;
}

void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream, int64_t active_blocks) {
    static const bool old = std::getenv("STRATA_TOPK_OLD") != nullptr;
    if (nq <= 0) return;
#if !defined(__HIPCC__)
    static const bool cluster = [] {
        const char* v = std::getenv("STRATA_QSA_CLUSTER");
        return !v || std::atoi(v) != 0;
    }();
    if (cluster && !old && nq <= 16 && qsa_block_topk_cluster(scores, steps, nq, max_blocks, cap, s, ids, stream))
        return;
#endif
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    const int64_t fit = (int64_t)TK_T * TK_PER;
    if (old) {
        qsa_block_topk_ref(scores, steps, nq, max_blocks, cap, s, ids, stream);
        return;
    }
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    static sycl::queue default_queue{sycl::property::queue::in_order()};
    sycl::queue* q_ptr = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    if (reach > fit) {
        q_ptr->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<int[TK_T / 32][256], 0> hist_acc(cgh);
            sycl::local_accessor<int, 1> s_gt_acc(sycl::range<1>(TK_T / 32), cgh);
            sycl::local_accessor<int, 1> s_eq_acc(sycl::range<1>(TK_T / 32), cgh);
            sycl::local_accessor<int, 0> s_digit_acc(cgh);
            sycl::local_accessor<int, 0> s_above_acc(cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(nq * TK_T, TK_T), [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                    block_topk_wide_kernel(scores, steps, max_blocks, cap, ids, hist_acc,
                                           s_gt_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                                           s_eq_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                                           s_digit_acc, s_above_acc, item);
                });
        });
    } else {
        q_ptr->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<int[TK_T / 32][256], 0> hist_acc(cgh);
            sycl::local_accessor<int, 1> s_warp_acc(sycl::range<1>(33), cgh);
            sycl::local_accessor<int, 0> s_digit_acc(cgh);
            sycl::local_accessor<int, 0> s_above_acc(cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(nq * TK_T, TK_T), [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                    block_topk_reg_kernel<TK_PER>(scores, steps, max_blocks, cap, ids, hist_acc,
                                                  s_warp_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                                                  s_digit_acc, s_above_acc, item);
                });
        });
    }
}

}  // namespace strata::kernels
