// src/kernels/sycl/sampler.dp.cpp - Pure SYCL 2020 migration.
//
// The sampler chain in llama.cpp's order:
//     penalties -> top_k -> top_p -> min_p -> temperature -> pick
//
#include <sycl/sycl.hpp>
#include "strata/kernels/sampler.hpp"
#include "strata/core/coupled_draft.hpp"
#include "strata/core/emulate.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace strata::kernels {
namespace {

inline sycl::queue& get_default_in_order_queue() {
    static sycl::queue q(sycl::property::queue::in_order{});
    return q;
}

// Philox 4x32-10, counter-based generator.
inline uint32_t philox4x32_round(uint32_t &c0, uint32_t &c1,
                                 uint32_t &c2, uint32_t &c3,
                                 uint32_t k0, uint32_t k1) {
    const uint32_t hi0 = sycl::mul_hi<unsigned>(0x9E3779B9u, c0);
    const uint32_t hi1 = sycl::mul_hi<unsigned>(0xBB67AE85u, c2);
    const uint32_t lo0 = 0x9E3779B9u * c0;
    const uint32_t lo1 = 0xBB67AE85u * c2;
    const uint32_t n0 = hi1 ^ c1 ^ k0;
    const uint32_t n1 = lo1;
    const uint32_t n2 = hi0 ^ c3 ^ k1;
    const uint32_t n3 = lo0;
    c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    return 0;
}

inline float philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (int i = 0; i < 10; ++i) {
        philox4x32_round(c0, c1, c2, c3, (uint32_t) i, 0u);
    }
    // 24 bits of mantissa, so the value is uniform in [0,1) with no rounding to 1.0
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}

inline int history_count(const int *__restrict__ h, int n, int v) {
    int c = 0;
    for (int i = 0; i < n; ++i) if (h[i] == v) ++c;
    return c;
}

inline float apply_penalties(float logit, int count, const SamplerParams &p) {
    if (count <= 0) return logit;
    if (logit <= 0.0f) logit *= p.penalty_repeat;
    else               logit /= p.penalty_repeat;
    logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
    return logit;
}

void sampler_greedy_kernel(sycl::nd_item<1> item,
                           const float* __restrict__ logits, int n_vocab,
                           const int* __restrict__ history, int history_len, const SamplerParams p,
                           int pmin, int plen, int* __restrict__ out,
                           unsigned int *penal_bits, float *sv, int *si) {
    const int t = item.get_group(0);
    const float* l = logits + (size_t) t * n_vocab;
    (void) pmin;
    const int* hrow = history ? history + (size_t) t * history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = plen < history_len ? plen : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }

    const int bits_words = (int) ((n_vocab + 31) / 32);
    const bool use_bits = penal_bits != nullptr && hrow != nullptr && hlen > 0 && bits_words > 0;
    if (use_bits) {
        for (int w = item.get_local_id(0); w < bits_words; w += item.get_local_range(0))
            penal_bits[w] = 0u;
        sycl::group_barrier(item.get_group());
        for (int i = item.get_local_id(0); i < hlen; i += item.get_local_range(0)) {
            if (hrow[i] >= 0 && hrow[i] < n_vocab) {
                sycl::atomic_ref<unsigned int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::work_group,
                                 sycl::access::address_space::local_space>(
                    penal_bits[hrow[i] >> 5]).fetch_or(1u << (hrow[i] & 31));
            }
        }
        sycl::group_barrier(item.get_group());
    }
    auto hit_count = [&](int v) -> int {
        if (!use_bits || !(penal_bits[v >> 5] & (1u << (v & 31)))) return 0;
        return history_count(hrow, hlen, v);
    };

    float bv = sycl::bit_cast<float, int>(0xff800000); // -inf
    int best = n_vocab;
    for (int v = item.get_local_id(0); v < n_vocab; v += item.get_local_range(0)) {
        const float s = apply_penalties(l[v], hit_count(v), p);
        if (s > bv) { bv = s; best = v; }
    }
    auto sg = item.get_sub_group();
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = sycl::shift_group_left(sg, bv, off);
        const int oi = sycl::shift_group_left(sg, best, off);
        if (ov > bv || (ov == bv && oi < best)) { bv = ov; best = oi; }
    }

    const int warp = (int)(item.get_local_id(0) >> 5),
              lane = (int)(item.get_local_id(0) & 31);
    if (lane == 0) { sv[warp] = bv; si[warp] = best; }
    sycl::group_barrier(item.get_group());
    if (warp == 0) {
        const int nw = (int)((item.get_local_range(0) + 31) >> 5);
        float wv = lane < nw ? sv[lane] : sycl::bit_cast<float, int>(0xff800000);
        int wi = lane < nw ? si[lane] : n_vocab;
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = sycl::shift_group_left(sg, wv, off);
            const int oi = sycl::shift_group_left(sg, wi, off);
            if (ov > wv || (ov == wv && oi < wi)) { wv = ov; wi = oi; }
        }
        if (lane == 0) out[t] = (wi < n_vocab) ? wi : 0;
    }
}

void sampler_kernel(sycl::nd_item<1> item,
                    const float *__restrict__ logits, int n_vocab, int n_tokens,
                    const int *__restrict__ history, int history_len,
                    const SamplerParams p, int *__restrict__ out,
                    unsigned int *penal_bits, int *sel_ids, float *sel_logit,
                    float *sv, int *si) {
    const int t = item.get_group(0);
    if (t >= n_tokens) return;
    const float* l = logits + (size_t) t * n_vocab;

    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;

    const int* hrow = history ? history + (size_t) t * history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;
    }

    const int bits_words = (int) ((n_vocab + 31) / 32);
    const bool use_bits = penal_bits != nullptr && hrow != nullptr && hlen > 0 && bits_words > 0;
    if (use_bits) {
        for (int w = item.get_local_id(0); w < bits_words; w += item.get_local_range(0))
            penal_bits[w] = 0u;
        sycl::group_barrier(item.get_group());
        for (int i = item.get_local_id(0); i < hlen; i += item.get_local_range(0)) {
            if (hrow[i] >= 0 && hrow[i] < n_vocab) {
                sycl::atomic_ref<unsigned int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::work_group,
                                 sycl::access::address_space::local_space>(
                    penal_bits[hrow[i] >> 5]).fetch_or(1u << (hrow[i] & 31));
            }
        }
        sycl::group_barrier(item.get_group());
    }
    auto hit_count = [&](int v) -> int {
        if (!use_bits || !(penal_bits[v >> 5] & (1u << (v & 31)))) return 0;
        return history_count(hrow, hlen, v);
    };

    const int KMAX = 64;
    int k = (p.top_k > 0 && p.top_k < KMAX) ? p.top_k : KMAX;
    if (k > n_vocab) k = n_vocab;

    auto sg = item.get_sub_group();
    for (int i = 0; i < k; ++i) {
        float bv = sycl::bit_cast<float, int>(0xff800000); // -inf
        int best = n_vocab;
        for (int v = item.get_local_id(0); v < n_vocab; v += item.get_local_range(0)) {
            bool taken = false;
            for (int j = 0; j < i; ++j) if (sel_ids[j] == v) { taken = true; break; }
            if (taken) continue;
            const float s = apply_penalties(l[v], hit_count(v), p);
            if (s > bv) { bv = s; best = v; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = sycl::shift_group_left(sg, bv, off);
            const int oi = sycl::shift_group_left(sg, best, off);
            if (ov > bv || (ov == bv && oi < best)) { bv = ov; best = oi; }
        }
        const int warp = (int)(item.get_local_id(0) >> 5),
                  lane = (int)(item.get_local_id(0) & 31);
        if (lane == 0) { sv[warp] = bv; si[warp] = best; }
        sycl::group_barrier(item.get_group());
        if (warp == 0) {
            const int nw = (int)((item.get_local_range(0) + 31) >> 5);
            float wv = lane < nw ? sv[lane] : sycl::bit_cast<float, int>(0xff800000);
            int wi = lane < nw ? si[lane] : n_vocab;
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = sycl::shift_group_left(sg, wv, off);
                const int oi = sycl::shift_group_left(sg, wi, off);
                if (ov > wv || (ov == wv && oi < wi)) { wv = ov; wi = oi; }
            }
            if (lane == 0) { sel_ids[i] = (wi < n_vocab) ? wi : 0; sel_logit[i] = wv; }
        }
        sycl::group_barrier(item.get_group());
    }

    int n_keep = k;
    float mx = sel_logit[0];
    for (int i = 1; i < k; ++i) mx = sycl::fmax(mx, sel_logit[i]);
    if (p.top_p < 1.0f) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += sycl::exp((double)sel_logit[i] - (double)mx);
        double cum = 0.0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += sycl::exp((double)sel_logit[i] - (double)mx) / sum;
            if (cum >= (double) p.top_p) { cut = i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
    }
    if (p.min_p > 0.0f) {
        const float thresh = sel_logit[0] + sycl::log((float)(p.min_p));
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[i] < thresh) { n_keep = i; break; }
    }
    auto scaled = [&](int i, float *sel_logit) { return sel_logit[i] * inv_t; };
    float smx = scaled(0, sel_logit);
    for (int i = 1; i < n_keep; ++i)
        smx = sycl::fmax(smx, scaled(i, sel_logit));
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i)
        sum += sycl::exp((double)scaled(i, sel_logit) - (double)smx);
    const float u = philox_uniform(p.seed, p.counter + (uint64_t) t);
    double cum = 0.0;
    int pick = sel_ids[n_keep - 1];
    for (int i = 0; i < n_keep; ++i) {
        cum += sycl::exp((double)scaled(i, sel_logit) - (double)smx) / sum;
        if ((double) u < cum) { pick = sel_ids[i]; break; }
    }
    if (item.get_local_id(0) == 0) out[t] = pick;
}

constexpr int kSelMax = 64;
[[maybe_unused]] constexpr unsigned kFullMask = 0xFFFFFFFFu;

inline int sampled_k(int top_k, int n_vocab) {
    int k = (top_k > 0 && top_k < kSelMax) ? top_k : kSelMax;
    return k > n_vocab ? n_vocab : k;
}

inline void take_first(float &bv, int &bi, float ov, int oi) {
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
}

inline void warp_first(sycl::sub_group sg, float &bv, int &bi) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = sycl::permute_group_by_xor(sg, bv, off);
        const int oi = sycl::permute_group_by_xor(sg, bi, off);
        take_first(bv, bi, ov, oi);
    }
}

template <bool kProb = false>
inline void sampled_tail_warp(sycl::sub_group sg, const int* sel_ids, const float* sel_logit, int k, const SamplerParams& p, int t,
                              int* __restrict__ out, double* ex, float* prob_out = nullptr) {
    const int lane = (int)sg.get_local_linear_id();
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    int n_keep = k;
    float mx = sel_logit[0];
    for (int i = 1; i < k; ++i) mx = sycl::fmax(mx, sel_logit[i]);
    if (p.top_p < 1.0f) {
        for (int i = lane; i < k; i += 32)
            ex[i] = sycl::exp((double)sel_logit[i] - (double)mx);
        sycl::group_barrier(sg);
        double sum = 0.0;
        if (lane == 0)
            for (int i = 0; i < k; ++i) sum += ex[i];
        sum = sycl::select_from_group(sg, sum, 0);
        sycl::group_barrier(sg);
        for (int i = lane; i < k; i += 32) ex[i] = ex[i] / sum;
        sycl::group_barrier(sg);
        int cut = k;
        if (lane == 0) {
            double cum = 0.0;
            for (int i = 0; i < k; ++i) {
                cum += ex[i];
                if (cum >= (double) p.top_p) { cut = i + 1; break; }
            }
        }
        cut = sycl::select_from_group(sg, cut, 0);
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
        sycl::group_barrier(sg);
    }
    if (p.min_p > 0.0f) {
        const float thresh = sel_logit[0] + sycl::log((float)(p.min_p));
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[i] < thresh) { n_keep = i; break; }
    }
    float smx = sel_logit[0] * inv_t;
    for (int i = 1; i < n_keep; ++i)
        smx = sycl::fmax(smx, sel_logit[i] * inv_t);
    for (int i = lane; i < n_keep; i += 32)
        ex[i] = sycl::exp((double)(sel_logit[i] * inv_t) - (double)smx);
    sycl::group_barrier(sg);
    double sum = 0.0;
    if (lane == 0)
        for (int i = 0; i < n_keep; ++i) sum += ex[i];
    sum = sycl::select_from_group(sg, sum, 0);
    sycl::group_barrier(sg);
    for (int i = lane; i < n_keep; i += 32) ex[i] = ex[i] / sum;
    sycl::group_barrier(sg);
    if (lane == 0) {
        const float u = philox_uniform(p.seed, p.counter + (uint64_t) t);
        double cum = 0.0;
        int pi = n_keep > 0 ? n_keep - 1 : 0;
        int pick = sel_ids[pi];
        for (int i = 0; i < n_keep; ++i) {
            cum += ex[i];
            if ((double) u < cum) { pick = sel_ids[i]; pi = i; break; }
        }
        out[t] = pick;
        if constexpr (kProb) *prob_out = n_keep > 0 ? (float) ex[pi] : 1.0f;
    }
}

void sampler_one_block_kernel(sycl::nd_item<1> item,
                              const float *__restrict__ logits, int n_vocab,
                              const int *__restrict__ history, int history_len,
                              const SamplerParams p, int *__restrict__ out,
                              unsigned int *penal_bits, int *sel_ids,
                              float *sel_logit, double *ex, float *sv,
                              int *si) {
    const int t = item.get_group(0);
    const float* l = logits + (size_t) t * n_vocab;

    const int* hrow = history ? history + (size_t) t * history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;
    }
    const int bits_words = (int) ((n_vocab + 31) / 32);
    const bool use_bits = penal_bits != nullptr && hrow != nullptr && hlen > 0 && bits_words > 0;
    if (use_bits) {
        for (int w = item.get_local_id(0); w < bits_words; w += item.get_local_range(0))
            penal_bits[w] = 0u;
        sycl::group_barrier(item.get_group());
        for (int i = item.get_local_id(0); i < hlen; i += item.get_local_range(0)) {
            if (hrow[i] >= 0 && hrow[i] < n_vocab) {
                sycl::atomic_ref<unsigned int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::work_group,
                                 sycl::access::address_space::local_space>(
                    penal_bits[hrow[i] >> 5]).fetch_or(1u << (hrow[i] & 31));
            }
        }
        sycl::group_barrier(item.get_group());
    }
    auto hit_count = [&](int v) -> int {
        if (!use_bits || !(penal_bits[v >> 5] & (1u << (v & 31)))) return 0;
        return history_count(hrow, hlen, v);
    };

    const int k = sampled_k(p.top_k, n_vocab);
    const int warp = (int)(item.get_local_id(0) >> 5),
              lane = (int)(item.get_local_id(0) & 31);
    float prev_v = sycl::bit_cast<float>(0x7f800000);
    int prev_i = -1;
    auto sg = item.get_sub_group();

    for (int i = 0; i < k; ++i) {
        float bv = sycl::bit_cast<float, int>(0xff800000);
        int best = n_vocab;
        for (int v = item.get_local_id(0); v < n_vocab; v += item.get_local_range(0)) {
            const float s = apply_penalties(l[v], hit_count(v), p);
            if ((s < prev_v || (s == prev_v && v > prev_i)) && s > bv) { bv = s; best = v; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = sycl::shift_group_left(sg, bv, off);
            const int oi = sycl::shift_group_left(sg, best, off);
            if (ov > bv || (ov == bv && oi < best)) { bv = ov; best = oi; }
        }
        if (lane == 0) { sv[warp] = bv; si[warp] = best; }
        sycl::group_barrier(item.get_group());
        if (warp == 0) {
            const int nw = (int)((item.get_local_range(0) + 31) >> 5);
            float wv = lane < nw ? sv[lane] : sycl::bit_cast<float, int>(0xff800000);
            int wi = lane < nw ? si[lane] : n_vocab;
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = sycl::shift_group_left(sg, wv, off);
                const int oi = sycl::shift_group_left(sg, wi, off);
                if (ov > wv || (ov == wv && oi < wi)) { wv = ov; wi = oi; }
            }
            if (lane == 0) { sel_ids[i] = (wi < n_vocab) ? wi : 0; sel_logit[i] = wv; }
        }
        sycl::group_barrier(item.get_group());
        prev_v = sel_logit[i];
        prev_i = sel_ids[i];
    }
    if (warp != 0) return;
    sampled_tail_warp(sg, sel_ids, sel_logit, k, p, t, out, ex);
}

constexpr int kSplitPerLane = 32;
constexpr int kSplitWarpSpan = 32 * kSplitPerLane;
constexpr int kSplitWarps = 4;
constexpr int kSplitBlockSpan = kSplitWarps * kSplitWarpSpan;
constexpr int kSplitMaxBlocks = 64;
constexpr int kSplitMaxRows = 64;

template <typename Sink>
inline void warp_merge_lists(sycl::sub_group sg, const sycl::int2 *lists, int nl,
                             int stride, int k, int n_vocab,
                             Sink &&sink) {
    const int lane = (int)sg.get_local_linear_id();
    float hv[2];
    int hi[2], pos[2];
#pragma unroll
    for (int m = 0; m < 2; ++m) {
        const int L = lane + 32 * m;
        pos[m] = 0;
        hv[m] = sycl::bit_cast<float, int>(0xff800000);
        hi[m] = n_vocab;
        if (L < nl) {
            const sycl::int2 c = lists[(size_t)L * stride];
            hi[m] = c.x();
            hv[m] = sycl::bit_cast<float>(c.y());
        }
    }
    for (int i = 0; i < k; ++i) {
        float bv = hv[0];
        int bi = hi[0];
        take_first(bv, bi, hv[1], hi[1]);
        warp_first(sg, bv, bi);
        sink(i, bv, bi);
        if (bi < n_vocab) {
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                if (hi[m] != bi) continue;
                if (++pos[m] < k) {
                    const sycl::int2 c = lists[(size_t)(lane + 32 * m) * stride + pos[m]];
                    hi[m] = c.x();
                    hv[m] = sycl::bit_cast<float>(c.y());
                } else {
                    hi[m] = n_vocab;
                    hv[m] = sycl::bit_cast<float, int>(0xff800000);
                }
            }
        }
    }
}

void sampler_split_part_kernel(
    sycl::nd_item<2> item,
    const float *__restrict__ logits, int n_vocab,
    const int *__restrict__ history, int history_len, const SamplerParams p,
    int k, int n_blocks, sycl::int2 *__restrict__ cand, unsigned int *bits,
    sycl::int2 wl[4 /*kSplitWarps*/][64 /*kSelMax*/]) {
    const int t = (int)item.get_group(0);
    const float* l = logits + (size_t) t * n_vocab;
    const int lid = (int)item.get_local_id(1);
    const int warp = lid >> 5;
    const int lane = lid & 31;
    const int b = (int)item.get_group(1);
    const int blo = b * kSplitBlockSpan;

    const int* hrow = history ? history + (size_t) t * history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;
    }

    const bool use_bits = hrow != nullptr && hlen > 0;
    if (use_bits) {
        for (int w = lid; w < kSplitBlockSpan / 32; w += (int)item.get_local_range(1))
            bits[w] = 0u;
        sycl::group_barrier(item.get_group());
        for (int i = lid; i < hlen; i += (int)item.get_local_range(1)) {
            const int h = hrow[i];
            if (h >= 0 && h < n_vocab && h >= blo && h - blo < kSplitBlockSpan) {
                sycl::atomic_ref<unsigned int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::work_group,
                                 sycl::access::address_space::local_space>(
                    bits[(h - blo) >> 5]).fetch_or(1u << ((h - blo) & 31));
            }
        }
        sycl::group_barrier(item.get_group());
    }

    const int lo = blo + warp * kSplitWarpSpan;
    float s[kSplitPerLane];
#pragma unroll
    for (int j = 0; j < kSplitPerLane; ++j) {
        const int v = lo + 32 * j + lane;
        s[j] = v < n_vocab ? l[v] : sycl::bit_cast<float, int>(0xff800000);
    }
    if (use_bits) {
#pragma unroll
        for (int j = 0; j < kSplitPerLane; ++j) {
            const int v = lo + 32 * j + lane, bv_off = v - blo;
            if (v < n_vocab && (bits[bv_off >> 5] & (1u << (bv_off & 31))))
                s[j] = apply_penalties(s[j], history_count(hrow, hlen, v), p);
        }
    }

    float prev_v = sycl::bit_cast<float>(0x7f800000);
    int prev_i = -1;
    int i = 0;
    auto sg = item.get_sub_group();
    for (; i < k; ++i) {
        float b0 = sycl::bit_cast<float, int>(0xff800000),
              b1 = sycl::bit_cast<float, int>(0xff800000);
        int i0 = n_vocab, i1 = n_vocab;
#pragma unroll
        for (int j = 0; j < kSplitPerLane; j += 2) {
            const int v0 = lo + 32 * j + lane, v1 = v0 + 32;
            const float x0 = s[j], x1 = s[j + 1];
            if ((x0 < prev_v || (x0 == prev_v && v0 > prev_i)) && x0 > b0) { b0 = x0; i0 = v0; }
            if ((x1 < prev_v || (x1 == prev_v && v1 > prev_i)) && x1 > b1) { b1 = x1; i1 = v1; }
        }
        take_first(b0, i0, b1, i1);
        warp_first(sg, b0, i0);
        if (i0 >= n_vocab) break;
        if (lane == 0) wl[warp][i] = sycl::int2(i0, sycl::bit_cast<int>(b0));
        prev_v = b0;
        prev_i = i0;
    }
    for (int r = i + lane; r < k; r += 32)
        wl[warp][r] = sycl::int2(n_vocab, (int)0xff800000u);
    sycl::group_barrier(item.get_group());
    if (warp == 0) {
        sycl::int2 *dst = cand + ((size_t)t * n_blocks + b) * k;
        warp_merge_lists(sg, &wl[0][0], kSplitWarps, kSelMax, k, n_vocab, [=](int r, float v, int id) {
            if (lane == 0) dst[r] = sycl::int2(id, sycl::bit_cast<int>(v));
        });
    }
}

void sampler_split_merge_kernel(sycl::nd_item<1> item,
                                const sycl::int2 *__restrict__ cand,
                                int n_blocks, int n_vocab,
                                const SamplerParams p, int k,
                                int *__restrict__ out, sycl::int2 *lists,
                                int *sel_ids, float *sel_logit, double *ex) {
    const int t = (int)item.get_group(0);
    auto sg = item.get_sub_group();
    const int lane = (int)item.get_local_id(0);

    const sycl::int2 *src = cand + (size_t)t * n_blocks * k;
    for (int e = lane; e < n_blocks * k; e += 32) lists[e] = src[e];
    sycl::group_barrier(sg);
    warp_merge_lists(sg, lists, n_blocks, k, k, n_vocab, [=](int i, float v, int id) {
        if (lane == 0) { sel_ids[i] = id < n_vocab ? id : 0; sel_logit[i] = v; }
    });
    sycl::group_barrier(sg);
    sampled_tail_warp(sg, sel_ids, sel_logit, k, p, t, out, ex);
}

void coupled_stage_kernel(sycl::nd_item<1> item,
                          const SamplerParams* __restrict__ mp, const int* __restrict__ mh,
                          SamplerParams* __restrict__ dp, int* __restrict__ ring, int cap) {
    const volatile int *s = (const volatile int *)mp;
    int* d = (int*) dp;
    for (int i = item.get_local_id(0);
         i < (int)(sizeof(SamplerParams) / sizeof(int));
         i += item.get_local_range(0)) d[i] = s[i];
    const int h = strata::core::coupled_hist_len(((const volatile SamplerParams*) mp)->penalty_last_n, cap);
    const volatile int* vh = (const volatile int*) mh;
    for (int i = cap - h + (int)item.get_local_id(0); i < cap;
         i += item.get_local_range(0)) ring[i] = vh[i];
}

void coupled_penalize_kernel(sycl::nd_item<1> item,
                             float* __restrict__ logits, int nv, const int* __restrict__ id_to_sub,
                             int id_vocab, const SamplerParams* __restrict__ dp, const int* __restrict__ ring,
                             int cap, int j, unsigned int *seen) {
    const SamplerParams p = *dp;
    const int h = strata::core::coupled_hist_len(p.penalty_last_n, cap);
    if (h <= 0) return;
    const int* hrow = ring + strata::core::coupled_hist_start(cap, j, h);
    const int words = (nv + 31) / 32;
    for (int w = item.get_local_id(0); w < words; w += item.get_local_range(0)) seen[w] = 0u;
    sycl::group_barrier(item.get_group());
    for (int i = item.get_local_id(0); i < h; i += item.get_local_range(0)) {
        const int v = hrow[i];
        if (v < 0 || v >= id_vocab) continue;
        const int s = id_to_sub != nullptr ? id_to_sub[v] : v;
        if (s < 0 || s >= nv) continue;
        const unsigned bit = 1u << (s & 31);
        auto ref = sycl::atomic_ref<unsigned int, sycl::memory_order::relaxed,
                                    sycl::memory_scope::work_group,
                                    sycl::access::address_space::local_space>(seen[s >> 5]);
        if (ref.fetch_or(bit) & bit) continue;
        logits[s] = apply_penalties(logits[s], history_count(hrow, h, v), p);
    }
}

void coupled_merge_kernel(
    sycl::nd_item<1> item,
    const sycl::int2 *__restrict__ cand, int n_blocks, int nv, int kpart,
    const SamplerParams *__restrict__ dp, const int *__restrict__ step_rec,
    const int *__restrict__ sub_to_id, int *__restrict__ ring, int cap, int j,
    int *__restrict__ out_id, float *__restrict__ out_prob, sycl::int2 *lists,
    int *sel_ids, float *sel_logit, double *ex, int *pick, float *prob) {
    auto sg = item.get_sub_group();
    const int lane = (int)item.get_local_id(0);
    SamplerParams p = *dp;
    p.counter = strata::core::coupled_draft_counter((int64_t) step_rec[0]);
    const int k = sampled_k(p.top_k, nv);

    for (int e = lane; e < n_blocks * kpart; e += 32) lists[e] = cand[e];
    sycl::group_barrier(sg);
    warp_merge_lists(sg, lists, n_blocks, kpart, k, nv, [=](int i, float v, int id) {
        if (lane == 0) { sel_ids[i] = id < nv ? id : 0; sel_logit[i] = v; }
    });
    sycl::group_barrier(sg);
    if (p.greedy || p.temperature <= 0.0f) {
        if (lane == 0) { pick[0] = sel_ids[0]; prob[0] = 1.0f; }
    } else {
        sampled_tail_warp<true>(sg, sel_ids, sel_logit, k, p, 0, pick, ex, prob);
    }
    sycl::group_barrier(sg);
    if (lane == 0) {
        const int s = pick[0];
        const int id = sub_to_id != nullptr ? sub_to_id[s] : s;
        *out_id = id;
        *out_prob = prob[0];
        ring[cap + j] = id;
    }
}

enum class SampledPath { Split, OneBlock, Old };

bool env_flag(const char* name) {
    const char* e = std::getenv(name);
    return e != nullptr && *e != '\0' && std::strcmp(e, "0") != 0;
}

SampledPath sampled_path() {
    static const SampledPath path = env_flag("STRATA_OLD_SAMPLER")         ? SampledPath::Old
                                    : env_flag("STRATA_SAMPLER_ONE_BLOCK") ? SampledPath::OneBlock
                                                                           : SampledPath::Split;
    return path;
}

bool stream_capturing(void *stream) {
    (void)stream;
    return false;
}

sycl::int2 *split_scratch(sycl::queue* q, size_t entries) {
    struct Slot {
        void* queue_ptr;
        sycl::int2 *ptr;
        size_t entries;
        size_t failed;
    };
    static std::mutex mu;
    static std::vector<Slot> slots;
    static std::vector<sycl::int2 *> retired;
    std::lock_guard<std::mutex> lock(mu);
    Slot* slot = nullptr;
    for (Slot& s : slots)
        if (s.queue_ptr == q) slot = &s;
    if (slot == nullptr) {
        slots.push_back({q, nullptr, 0, 0});
        slot = &slots.back();
    }
    if (slot->entries >= entries) return slot->ptr;
    if (slot->failed != 0 && entries >= slot->failed) return nullptr;
    constexpr size_t kCap = (size_t) kSplitMaxRows * kSplitMaxBlocks * kSelMax;
    size_t want = 2 * slot->entries < kCap ? 2 * slot->entries : kCap;
    if (want < entries) want = entries;
    sycl::int2 *ptr = sycl::malloc_device<sycl::int2>(want, *q);
    if (!ptr) {
        want = entries;
        ptr = sycl::malloc_device<sycl::int2>(want, *q);
        if (!ptr) {
            slot->failed = entries;
            return nullptr;
        }
    }
    if (slot->ptr != nullptr) retired.push_back(slot->ptr);
    slot->ptr = ptr;
    slot->entries = want;
    return ptr;
}

}  // namespace

bool sample_greedy_cluster(const float *logits, int n_tokens, int n_vocab, int *out, void *stream) {
    (void)logits; (void)n_tokens; (void)n_vocab; (void)out; (void)stream;
    return false;
}

void sample_tokens(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                   const SamplerParams& p, int* out, void* stream) {
    if (n_tokens <= 0 || n_vocab <= 0) return;
    if (p.penalty_last_n > 0 && (history == nullptr || history_len <= 0)) {
        std::fprintf(stderr, "sample_tokens: penalty_last_n %d needs a history (got %p, len %d)\n",
                     p.penalty_last_n, (const void*) history, history_len);
        std::exit(1);
    }
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &get_default_in_order_queue();

    const unsigned shmem_bytes =
        (history != nullptr && history_len > 0 && p.penalty_last_n > 0)
            ? (unsigned)((n_vocab + 31) / 32) * sizeof(unsigned)
            : 0;
    const size_t shmem_words = (shmem_bytes + sizeof(unsigned) - 1) / sizeof(unsigned);

    if (p.greedy || p.temperature <= 0.0f) {
        const int gthreads = 1024;
        q->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<unsigned int, 1> penal_acc(sycl::range<1>(shmem_words > 0 ? shmem_words : 1), cgh);
            sycl::local_accessor<float, 1> sv_acc(sycl::range<1>(32), cgh);
            sycl::local_accessor<int, 1> si_acc(sycl::range<1>(32), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>((size_t)n_tokens * gthreads), sycl::range<1>(gthreads)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                    unsigned int* penal_ptr = shmem_words > 0 ? penal_acc.get_multi_ptr<sycl::access::decorated::no>().get() : nullptr;
                    sampler_greedy_kernel(
                        item, logits, n_vocab, history, history_len, p,
                        p.penalty_last_n, p.penalty_last_n, out,
                        penal_ptr,
                        sv_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                        si_acc.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    } else if (sampled_path() == SampledPath::Old) {
        q->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<unsigned int, 1> penal_acc(sycl::range<1>(shmem_words > 0 ? shmem_words : 1), cgh);
            sycl::local_accessor<int, 1> sel_ids_acc(sycl::range<1>(64), cgh);
            sycl::local_accessor<float, 1> sel_logit_acc(sycl::range<1>(64), cgh);
            sycl::local_accessor<float, 1> sv_acc(sycl::range<1>(32), cgh);
            sycl::local_accessor<int, 1> si_acc(sycl::range<1>(32), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>((size_t)n_tokens * 1024), sycl::range<1>(1024)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                    unsigned int* penal_ptr = shmem_words > 0 ? penal_acc.get_multi_ptr<sycl::access::decorated::no>().get() : nullptr;
                    sampler_kernel(
                        item, logits, n_vocab, n_tokens, history, history_len, p,
                        out, penal_ptr,
                        sel_ids_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                        sel_logit_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                        sv_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                        si_acc.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    } else {
        const int k = sampled_k(p.top_k, n_vocab);
        const int n_blocks = (n_vocab + kSplitBlockSpan - 1) / kSplitBlockSpan;
        sycl::int2 *scratch = nullptr;
        if (sampled_path() == SampledPath::Split && n_blocks <= kSplitMaxBlocks && n_tokens <= kSplitMaxRows &&
            !stream_capturing(stream))
            scratch = split_scratch(q, (size_t) (n_tokens > 16 ? n_tokens : 16) * n_blocks * kSelMax);

        if (scratch != nullptr) {
            q->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<unsigned int, 1> bits_acc(sycl::range<1>(kSplitBlockSpan / 32), cgh);
                sycl::local_accessor<sycl::int2[kSplitWarps][kSelMax], 0> wl_acc(cgh);

                cgh.parallel_for(
                    sycl::nd_range<2>(
                        sycl::range<2>((size_t)n_tokens, (size_t)n_blocks * (kSplitWarps * 32)),
                        sycl::range<2>(1, kSplitWarps * 32)),
                    [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
                        sampler_split_part_kernel(
                            item, logits, n_vocab, history, history_len, p, k,
                            n_blocks, scratch,
                            bits_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                            wl_acc);
                    });
            });

            q->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<sycl::int2, 1> lists_acc(sycl::range<1>(kSplitMaxBlocks * kSelMax), cgh);
                sycl::local_accessor<int, 1> sel_ids_acc(sycl::range<1>(kSelMax), cgh);
                sycl::local_accessor<float, 1> sel_logit_acc(sycl::range<1>(kSelMax), cgh);
                sycl::local_accessor<double, 1> ex_acc(sycl::range<1>(kSelMax), cgh);

                cgh.parallel_for(
                    sycl::nd_range<1>(sycl::range<1>((size_t)n_tokens * 32), sycl::range<1>(32)),
                    [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                        sampler_split_merge_kernel(
                            item, scratch, n_blocks, n_vocab, p, k, out,
                            lists_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                            sel_ids_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                            sel_logit_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                            ex_acc.get_multi_ptr<sycl::access::decorated::no>().get());
                    });
            });
        } else {
            q->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<unsigned int, 1> penal_acc(sycl::range<1>(shmem_words > 0 ? shmem_words : 1), cgh);
                sycl::local_accessor<int, 1> sel_ids_acc(sycl::range<1>(kSelMax), cgh);
                sycl::local_accessor<float, 1> sel_logit_acc(sycl::range<1>(kSelMax), cgh);
                sycl::local_accessor<double, 1> ex_acc(sycl::range<1>(kSelMax), cgh);
                sycl::local_accessor<float, 1> sv_acc(sycl::range<1>(32), cgh);
                sycl::local_accessor<int, 1> si_acc(sycl::range<1>(32), cgh);

                cgh.parallel_for(
                    sycl::nd_range<1>(sycl::range<1>((size_t)n_tokens * 1024), sycl::range<1>(1024)),
                    [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                        unsigned int* penal_ptr = shmem_words > 0 ? penal_acc.get_multi_ptr<sycl::access::decorated::no>().get() : nullptr;
                        sampler_one_block_kernel(
                            item, logits, n_vocab, history, history_len, p, out,
                            penal_ptr,
                            sel_ids_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                            sel_logit_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                            ex_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                            sv_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                            si_acc.get_multi_ptr<sycl::access::decorated::no>().get());
                    });
            });
        }
    }
    if (stream == nullptr) {
        q->wait();
    }
}

namespace {
int coupled_blocks(int nv) { return (nv + kSplitBlockSpan - 1) / kSplitBlockSpan; }
int coupled_kpart(int nv) { return nv < kSelMax ? nv : kSelMax; }
void coupled_check([[maybe_unused]] const char* what) {}
}  // namespace

size_t coupled_draft_scratch_bytes(int nv) {
    if (nv <= 0 || coupled_blocks(nv) > kSplitMaxBlocks) return 0;
    return (size_t)coupled_blocks(nv) * (size_t)kSelMax * sizeof(sycl::int2);
}

void coupled_draft_stage(const SamplerParams* mapped_params, const int32_t* mapped_hist, SamplerParams* params,
                         int32_t* ring, int cap, void* stream) {
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &get_default_in_order_queue();
    q->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(256), sycl::range<1>(256)),
                         [=](sycl::nd_item<1> item) {
                             coupled_stage_kernel(item, mapped_params, mapped_hist, params, ring, cap);
                         });
    });
    if (stream == nullptr) q->wait();
    coupled_check("coupled_draft_stage");
}

void coupled_draft_sample(float* logits, int nv, const int32_t* sub_to_id, const int32_t* id_to_sub, int id_vocab,
                          const SamplerParams* params, int32_t* ring, int cap, int j, const int32_t* step_rec,
                          void* scratch, int32_t* out_id, float* out_prob, void* stream) {
    sycl::queue* s = stream ? static_cast<sycl::queue*>(stream) : &get_default_in_order_queue();
    const int n_blocks = coupled_blocks(nv), kpart = coupled_kpart(nv);
    if (nv <= 0 || n_blocks > kSplitMaxBlocks || scratch == nullptr) {
        std::fprintf(stderr, "coupled_draft_sample: %d logits need scratch and at most %d blocks\n", nv, kSplitMaxBlocks);
        std::exit(1);
    }

    const size_t seen_words = (nv + 31) / 32;
    s->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<unsigned int, 1> seen_acc(sycl::range<1>(seen_words > 0 ? seen_words : 1), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(1024), sycl::range<1>(1024)),
            [=](sycl::nd_item<1> item) {
                coupled_penalize_kernel(
                    item, logits, nv, id_to_sub, id_vocab, params, ring, cap, j,
                    seen_acc.get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });
    coupled_check("coupled_penalize");

    s->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<unsigned int, 1> bits_acc(sycl::range<1>(kSplitBlockSpan / 32), cgh);
        sycl::local_accessor<sycl::int2[kSplitWarps][kSelMax], 0> wl_acc(cgh);

        cgh.parallel_for(
            sycl::nd_range<2>(
                sycl::range<2>(1, (size_t)n_blocks * (kSplitWarps * 32)),
                sycl::range<2>(1, kSplitWarps * 32)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
                sampler_split_part_kernel(
                    item, logits, nv, nullptr, 0, SamplerParams{}, kpart, n_blocks,
                    (sycl::int2 *)scratch,
                    bits_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                    wl_acc);
            });
    });
    coupled_check("coupled_draft split part");

    s->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<sycl::int2, 1> lists_acc(sycl::range<1>(kSplitMaxBlocks * kSelMax), cgh);
        sycl::local_accessor<int, 1> sel_ids_acc(sycl::range<1>(kSelMax), cgh);
        sycl::local_accessor<float, 1> sel_logit_acc(sycl::range<1>(kSelMax), cgh);
        sycl::local_accessor<double, 1> ex_acc(sycl::range<1>(kSelMax), cgh);
        sycl::local_accessor<int, 1> pick_acc(sycl::range<1>(1), cgh);
        sycl::local_accessor<float, 1> prob_acc(sycl::range<1>(1), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(32), sycl::range<1>(32)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                coupled_merge_kernel(
                    item, (const sycl::int2 *)scratch, n_blocks, nv, kpart,
                    params, step_rec, sub_to_id, ring, cap, j, out_id,
                    out_prob,
                    lists_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                    sel_ids_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                    sel_logit_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                    ex_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                    pick_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                    prob_acc.get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });
    coupled_check("coupled_draft merge");
    if (stream == nullptr) s->wait();
}

}  // namespace strata::kernels
