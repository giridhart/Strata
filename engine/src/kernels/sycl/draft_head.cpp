// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/sycl/draft_head.cpp - see include/strata/kernels/draft_head.hpp.
#include "strata/kernels/draft_head.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {

static sycl::queue& default_queue() {
    static sycl::queue q{sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};
    return q;
}

constexpr int kTopWG = 256, kTopPer = 16, kTopChunk = kTopWG * kTopPer;   // topk_stage's work-group and chunk

// One stage of topk: each work-group the k largest of its kTopChunk of the `n` values (their indices `ids_in[i]`, or
// i without), into out_v / out_i at k a group.  Each work-item holds kTopPer of them; the group's reductions pick one
// a turn, so no local memory and no barrier of its own.
sycl::event topk_stage(sycl::queue& q, const float* v, const int32_t* ids_in, int64_t n, int k, float* out_v,
                       int32_t* out_i) {
    const int64_t groups = (n + kTopChunk - 1) / kTopChunk;
    return q.parallel_for(sycl::nd_range<1>((size_t) groups * kTopWG, kTopWG), [=](sycl::nd_item<1> it) {
        constexpr float kNone = -std::numeric_limits<float>::infinity();
        const int64_t g = (int64_t) it.get_group(0), base = g * kTopChunk;
        const int t = (int) it.get_local_id(0);
        float val[kTopPer];
        int32_t id[kTopPer];
#pragma unroll
        for (int i = 0; i < kTopPer; ++i) {
            const int64_t at = base + (int64_t) i * kTopWG + t;
            val[i] = at < n ? v[at] : kNone;
            id[i] = at < n ? (ids_in != nullptr ? ids_in[at] : (int32_t) at) : -1;
        }
        for (int j = 0; j < k; ++j) {
            float best = kNone;
            int slot = -1;
#pragma unroll
            for (int i = 0; i < kTopPer; ++i)
                if (id[i] >= 0 && (slot < 0 || val[i] > best)) { best = val[i]; slot = i; }
            const float top = sycl::reduce_over_group(it.get_group(), best, sycl::maximum<float>());
            // the lowest work-item holding it gives it up, so one value leaves the chunk each turn
            const int who = sycl::reduce_over_group(it.get_group(), slot >= 0 && best == top ? t : kTopWG,
                                                    sycl::minimum<int>());
            if (t == who) {
                out_v[g * k + j] = top;
                out_i[g * k + j] = id[slot];
                id[slot] = -1;
            } else if (who == kTopWG && t == 0) {   // fewer than k values in the chunk
                out_v[g * k + j] = kNone;
                out_i[g * k + j] = -1;
            }
        }
    });
}

}  // namespace

std::size_t int2_row_bytes(int n_in) {
    return (std::size_t) n_in / 4 + (std::size_t) (n_in / kInt2Group) * 4;
}

void int2_pack_rows(const float* rows, int64_t n_rows, int n_in, uint8_t* out) {
    if (n_in <= 0 || n_in % kInt2Group != 0) throw std::invalid_argument("int2_pack_rows: n_in is not a multiple of 128");
    const std::size_t rb = int2_row_bytes(n_in);
    for (int64_t r = 0; r < n_rows; ++r) {
        const float* x = rows + r * (int64_t) n_in;
        uint8_t* codes = out + (std::size_t) r * rb;
        auto* scales = reinterpret_cast<sycl::half*>(codes + n_in / 4);
        std::fill(codes, codes + n_in / 4, (uint8_t) 0);
        for (int g = 0; g < n_in / kInt2Group; ++g) {
            const float* xg = x + (std::size_t) g * kInt2Group;
            const auto [lo, hi] = std::minmax_element(xg, xg + kInt2Group);
            const sycl::half m = (sycl::half) *lo, d = (sycl::half) ((*hi - *lo) / 3.0f);
            sycl::half* md = scales + (std::size_t) g * 2;
            md[0] = m;
            md[1] = d;
            const float fm = (float) m, fd = (float) d;
            for (int i = 0; i < kInt2Group; ++i) {
                const int q = fd > 0.0f ? std::clamp((int) std::lround((xg[i] - fm) / fd), 0, 3) : 0;
                const int at = g * kInt2Group + i;
                codes[at / 4] |= (uint8_t) (q << (2 * (at % 4)));
            }
        }
    }
}

void int2_gemv(const uint8_t* w, const float* x, float* y, int n_in, int64_t n_out, void* stream) {
    if (n_out <= 0) return;
    if (n_in <= 0 || n_in % kInt2Group != 0) throw std::invalid_argument("int2_gemv: n_in is not a multiple of 128");
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    constexpr int WG = 256, kRows = 32;
    const std::size_t rb = int2_row_bytes(n_in);
    const int words = n_in / 16;
    const int64_t groups = (n_out + kRows - 1) / kRows;
    sycl::event e = q->parallel_for(sycl::nd_range<1>((size_t) groups * WG, WG), [=](sycl::nd_item<1> it) {
        const sycl::sub_group sg = it.get_sub_group();
        const int lanes = (int) sg.get_local_range()[0], lane = (int) sg.get_local_id()[0];
        const int n_sg = (int) sg.get_group_range()[0], sgi = (int) sg.get_group_id()[0];
        const int64_t r0 = (int64_t) it.get_group(0) * kRows;
        for (int rr = sgi; rr < kRows; rr += n_sg) {
            const int64_t r = r0 + rr;
            if (r >= n_out) break;   // the same for the whole sub-group
            const uint8_t* row = w + (std::size_t) r * rb;
            const auto* code = reinterpret_cast<const uint32_t*>(row);
            const auto* sc = reinterpret_cast<const sycl::half*>(row + n_in / 4);
            float acc = 0.0f;
            for (int wd = lane; wd < words; wd += lanes) {
                const uint32_t c = code[wd];
                const sycl::float4* xw = reinterpret_cast<const sycl::float4*>(x + (std::size_t) wd * 16);
                float dot = 0.0f, sum = 0.0f;
#pragma unroll
                for (int j4 = 0; j4 < 4; ++j4) {
                    const sycl::float4 xv = xw[j4];
                    const uint32_t cb = c >> (8 * j4);
                    dot = sycl::fma((float) (cb & 3u), xv.x(), dot);
                    dot = sycl::fma((float) ((cb >> 2) & 3u), xv.y(), dot);
                    dot = sycl::fma((float) ((cb >> 4) & 3u), xv.z(), dot);
                    dot = sycl::fma((float) ((cb >> 6) & 3u), xv.w(), dot);
                    sum += (xv.x() + xv.y()) + (xv.z() + xv.w());
                }
                const sycl::half* md = sc + (std::size_t) (wd / (kInt2Group / 16)) * 2;   // the group's minimum and step
                acc = sycl::fma((float) md[1], dot, sycl::fma((float) md[0], sum, acc));
            }
            const float total = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (lane == 0) y[r] = total;
        }
    });
    if (!stream) e.wait();
}

std::size_t topk_scratch_bytes(int64_t n, int k) {
    const int64_t c = ((n + kTopChunk - 1) / kTopChunk) * k;
    return (std::size_t) (2 * c) * (sizeof(float) + sizeof(int32_t));
}

void topk(const float* v, int64_t n, int k, int32_t* ids, void* scratch, void* stream) {
    if (k < 1 || k > kTopKMax || n < 1) throw std::invalid_argument("topk: k is 1 to 64 and n positive");
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    const int64_t cap = ((n + kTopChunk - 1) / kTopChunk) * k;
    auto* fv = static_cast<float*>(scratch);
    auto* fi = reinterpret_cast<int32_t*>(fv + 2 * cap);
    float* bv[2] = {fv, fv + cap};
    int32_t* bi[2] = {fi, fi + cap};
    const float* in_v = v;
    const int32_t* in_i = nullptr;
    int side = 0;
    sycl::event e;
    while (true) {
        const int64_t groups = (n + kTopChunk - 1) / kTopChunk;
        e = topk_stage(*q, in_v, in_i, n, k, bv[side], bi[side]);
        n = groups * k;
        in_v = bv[side];
        in_i = bi[side];
        side ^= 1;
        if (groups == 1) break;
    }
    e = q->memcpy(ids, in_i, (size_t) k * sizeof(int32_t));
    if (!stream) e.wait();
}

void int2_pad_from(const int32_t* pos, int32_t from, float* y, int64_t n, void* stream) {
    if (n <= 0 || pos == nullptr) return;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    sycl::event e = q->parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        if (*pos >= from) y[i] = -1.7e38f;
    });
    if (!stream) e.wait();
}

void scatter_values(const float* vals, const int32_t* ids, int k, float* y, void* stream) {
    if (k <= 0) return;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    sycl::event e = q->parallel_for(sycl::range<1>((size_t) k), [=](sycl::id<1> j) {
        const int32_t at = ids[j];
        if (at >= 0) y[at] = vals[j];
    });
    if (!stream) e.wait();
}

}  // namespace strata::kernels
