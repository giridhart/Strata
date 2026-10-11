// Adapted from topk-moe.cu/common.cuh in llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d; finite F32, 512-expert/10-output path.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include <sycl/sycl.hpp>
#include "strata/kernels/native_router.hpp"
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};

inline float warp_sum(float value, const sycl::sub_group& sg) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1)
        value += sycl::permute_group_by_xor(sg, value, mask);
    return value;
}

inline float warp_max(float value, const sycl::sub_group& sg) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1)
        value = sycl::fmax(value, sycl::permute_group_by_xor(sg, value, mask));
    return value;
}

inline void route_token(const float* logits, int32_t* ids, float* weights, size_t tok, int lane,
                        const sycl::sub_group& sg) {
    logits += tok * 512;
    ids += tok * 10;
    weights += tok * 10;

    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i)
        values[i] = logits[lane + i * 32];

    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i)
        maximum = sycl::fmax(maximum, values[i]);
    maximum = warp_max(maximum, sg);

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = sycl::native::exp(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum(sum, sg);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (sycl::isnan(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < 16; ++i) {
            if (values[i] > best) {
                best = values[i];
                expert = lane + i * 32;
            }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = sycl::permute_group_by_xor(sg, best, mask);
            const int other_id = sycl::permute_group_by_xor(sg, expert, mask);
            if (other > best || (other == best && other_id < expert)) {
                best = other;
                expert = other_id;
            }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            // Deliberately accumulate by WINNING EXPERT lane, not output rank.
            // Multiple selected experts in one lane add in selection order.
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = sycl::fmax(warp_sum(selected_sum, sg), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse_selected_sum;
}

// One subgroup (32 lanes) per token.  This replaces the legacy one thread per token scanning
// all 512 experts serially: lanes coalesced-read candidates f = lane, lane + 32, ... and reduce
// across the subgroup via shift_group_left.  Bitwise identical to reference semantics.
void route_resident_k(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights,
                      const int32_t* __restrict__ res, int n_tok, float margin, int lo, int hi,
                      unsigned long long* __restrict__ stats, const sycl::nd_item<1>& item) {
    const int t = item.get_group(0);
    if (t >= n_tok) return;
    const int lane = (int)item.get_local_id(0);
    const sycl::sub_group sg = item.get_sub_group();
    const float* l = logits + (size_t)t * 512;
    int32_t* id = ids + (size_t)t * 10;
    int32_t my[10];
    for (int r = 0; r < 10; ++r) my[r] = id[r];
    int swaps = 0, tail = 0, before = 0;
    if (lane == 0) {
        for (int r = 0; r < 10; ++r) before += (res[my[r]] < 0);
    }
    for (int r = lo; r <= hi && r < 10; ++r) {
        const int e = my[r];
        if (res[e] >= 0) continue;  // already resident: uniform across subgroup
        if (lane == 0) ++tail;
        float bv = -INFINITY;
        int bi = -1;
        for (int f = lane; f < 512; f += 32) {
            if (res[f] < 0) continue;
            bool used = false;
            for (int q = 0; q < 10; ++q) used |= (my[q] == f);
            if (used) continue;
            if (l[f] > bv) {
                bv = l[f];
                bi = f;
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = sycl::shift_group_left(sg, bv, off);
            const int oi = sycl::shift_group_left(sg, bi, off);
            if (ov > bv || (ov == bv && oi >= 0 && (bi < 0 || oi < bi))) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0 && bi >= 0 && l[e] - bv <= margin) {
            my[r] = bi;
            ++swaps;
        }
        my[r] = sycl::select_from_group(sg, my[r], 0);  // all lanes keep same picked set
    }
    int after = 0;
    if (lane == 0) {
        for (int r = 0; r < 10; ++r) after += (res[my[r]] < 0);
    }
    const int sw = sycl::select_from_group(sg, swaps, 0);
    if (sw) {
        float m = -INFINITY;
        for (int r = 0; r < 10; ++r) m = sycl::fmax(m, l[my[r]]);
        float ex[10], sum = 0.0f;
        for (int r = 0; r < 10; ++r) {
            ex[r] = sycl::exp(l[my[r]] - m);
            sum += ex[r];
        }
        for (int r = 0; r < 10; ++r) weights[(size_t)t * 10 + r] = ex[r] / sum;
    }
    if (lane == 0) {
        for (int r = 0; r < 10; ++r) id[r] = my[r];
        if (stats) {
            auto a_tail = sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                           sycl::access::address_space::global_space>(stats[0]);
            auto a_swaps = sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>(stats[1]);
            auto a_before = sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>(stats[2]);
            auto a_after = sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>(stats[3]);
            a_tail.fetch_add((unsigned long long)tail);
            a_swaps.fetch_add((unsigned long long)sw);
            a_before.fetch_add((unsigned long long)before);
            a_after.fetch_add((unsigned long long)after);
        }
    }
}

bool valid(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}

bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}  // namespace

void native_router_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_router_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream) {
    if (!stream || !valid(logits, 512 * 4) || !valid(ids, 10 * 4) || !valid(weights, 10 * 4) ||
        overlap(logits, 512 * 4, ids, 10 * 4) || overlap(logits, 512 * 4, weights, 10 * 4) ||
        overlap(ids, 10 * 4, weights, 10 * 4))
        throw std::invalid_argument("native router requires a stream, aligned spans, and disjoint outputs");

    static sycl::queue default_queue{sycl::property::queue::in_order()};
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(32), sycl::range<1>(32)),
                    [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                        route_token(logits, ids, weights, 0, (int)item.get_local_id(0), item.get_sub_group());
                    });
}

void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream) {
    if (!stream || n_tok < 1 || !valid(logits, (size_t)n_tok * 512 * 4) || !valid(ids, (size_t)n_tok * 10 * 4) ||
        !valid(weights, (size_t)n_tok * 10 * 4))
        throw std::invalid_argument("native router (multi) requires a stream and aligned [n,512]/[n,10] buffers");

    static sycl::queue default_queue{sycl::property::queue::in_order()};
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    constexpr size_t warps_per_wg = 8;
    constexpr size_t wg_size = warps_per_wg * 32;
    const size_t total_warps = ((size_t)n_tok + warps_per_wg - 1) / warps_per_wg * warps_per_wg;
    const size_t total_threads = total_warps * 32;

    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(total_threads), sycl::range<1>(wg_size)),
                    [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                        const size_t global_tid = item.get_global_id(0);
                        const size_t tok = global_tid / 32;
                        if (tok >= (size_t)n_tok) return;
                        const int lane = (int)(global_tid % 32);
                        route_token(logits, ids, weights, tok, lane, item.get_sub_group());
                    });
}

void native_route_resident(const float* logits, int32_t* ids, float* weights, const int32_t* res_layer, int n_tok,
                           float margin, int rank_lo, int rank_hi, unsigned long long* stats, void* stream) {
    if (!stream || n_tok < 1 || !res_layer)
        throw std::invalid_argument("native_route_resident: bad arguments");

    static sycl::queue default_queue{sycl::property::queue::in_order()};
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    const size_t wg = 32;
    const size_t global = (size_t)n_tok * 32;
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
                    [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                        route_resident_k(logits, ids, weights, res_layer, n_tok, margin, rank_lo, rank_hi, stats,
                                         item);
                    });
}
}  // namespace strata::kernels
