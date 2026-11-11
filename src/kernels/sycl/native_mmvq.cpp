// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{quantize.cu,vecdotq.cuh,mmvq.cu,common.cuh}
// and ggml/src/ggml-common.h. See docs/native-mmvq.md for exact scope.
//
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
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include "strata/core/sycl_doorbell.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace strata::kernels {
sycl::queue& default_queue() {
    static sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    return q;
}
sycl::queue& native_mmvq_default_queue() {
    return default_queue();
}
namespace {

inline int dp4a(int a, int b, int c) {
    auto va = sycl::vec<int, 1>(a).as<sycl::vec<int8_t, 4>>().convert<int>();
    auto vb = sycl::vec<int, 1>(b).as<sycl::vec<int8_t, 4>>().convert<int>();
    return c + va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
}

// Q3_K's 110-byte stride gives alternate blocks only two-byte alignment.
// Preserve the pinned helper's pair of 16-bit loads and little-endian combine.
template <bool STREAMING = false>
inline int load_int_b2(const void* ptr, int i32) {
    const auto* x = static_cast<const uint16_t*>(ptr);
    if constexpr (STREAMING) {
        auto props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::intel::experimental::read_hint<
                sycl::ext::intel::experimental::cache_control<
                    sycl::ext::intel::experimental::cache_mode::streaming,
                    sycl::ext::oneapi::experimental::cache_level::L3>>};
        sycl::ext::oneapi::experimental::annotated_ptr<const uint16_t, decltype(props)> ax(x, props);
        int value = ax[2 * i32] << 0;
        value |= ax[2 * i32 + 1] << 16;
        return value;
    } else {
        int value = x[2 * i32] << 0;
        value |= x[2 * i32 + 1] << 16;
        return value;
    }
}

template <bool STREAMING = false>
inline int load_int_stream(const int* ptr) {
    if constexpr (STREAMING) {
        auto props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::intel::experimental::read_hint<
                sycl::ext::intel::experimental::cache_control<
                    sycl::ext::intel::experimental::cache_mode::streaming,
                    sycl::ext::oneapi::experimental::cache_level::L3>>};
        sycl::ext::oneapi::experimental::annotated_ptr<const int, decltype(props)> ap(ptr, props);
        return *ap;
    } else {
        return *ptr;
    }
}

// SYCL port: a 4-int vector load from a 4-byte aligned address (a Q8_1 block is 36 bytes, its qs sit at offset 4): one dword-vector
// message instead of four scalar loads.
template <bool STREAMING = false>
inline sycl::int4 ld_i4_a4(const int* p) {
    if constexpr (STREAMING) {
        return sycl::int4{load_int_stream<true>(p + 0),
                          load_int_stream<true>(p + 1),
                          load_int_stream<true>(p + 2),
                          load_int_stream<true>(p + 3)};
    } else {
        sycl::int4 v;
        v.load(0, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p));
        return v;
    }
}

// Q6_K: per-byte (v - 32) of four 6-bit values (bytes 0..63): no byte can saturate, so this is the saturating char4 subtract's
// result exactly, in four integer ops (SWAR: no borrow crosses a byte) instead of the per-byte vector builtin.
inline int q6_sub32(int x) {
    const uint32_t u = (uint32_t)x;
    return (int)(((u | 0x80808080u) - 0x20202020u) ^ (~u & 0x80808080u));
}

inline sycl::int4 load16_a2(const void* p) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const sycl::int4* q = reinterpret_cast<const sycl::int4*>(a & ~uintptr_t(15));
    const int k = int(a >> 2) & 3;
    const bool half = (a & 2) != 0;
    const sycl::int4 lo = q[0], hi = (a & 15) ? q[1] : lo;
    const uint32_t d[8] = {(uint32_t)lo.x(), (uint32_t)lo.y(), (uint32_t)lo.z(), (uint32_t)lo.w(),
                           (uint32_t)hi.x(), (uint32_t)hi.y(), (uint32_t)hi.z(), (uint32_t)hi.w()};
    uint32_t w[5];
#pragma unroll
    for (int i = 0; i < 5; ++i)
        w[i] = k == 0 ? d[i] : k == 1 ? d[i + 1] : k == 2 ? d[i + 2] : d[(i + 3) & 7];
    uint32_t o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = half ? (w[i] >> 16) | (w[i + 1] << 16) : w[i];
    return sycl::int4((int)o[0], (int)o[1], (int)o[2], (int)o[3]);
}

inline unsigned int byte_permute(unsigned int a, unsigned int b, unsigned int s) {
    const uint64_t v = ((uint64_t)b << 32) | a;
    return (unsigned int)((((v >> ((s & 0x7) * 8)) & 0xff)) | ((((v >> (((s >> 4) & 0x7) * 8)) & 0xff) << 8)) |
                          ((((v >> (((s >> 8) & 0x7) * 8)) & 0xff) << 16)) |
                          ((((v >> (((s >> 12) & 0x7) * 8)) & 0xff) << 24)));
}

inline int vsubss4(int a, int b) {
    auto va = sycl::vec<int, 1>(a).as<sycl::vec<int8_t, 4>>();
    auto vb = sycl::vec<int, 1>(b).as<sycl::vec<int8_t, 4>>();
    auto vr = sycl::sub_sat(va, vb);
    return vr.as<sycl::vec<int, 1>>()[0];
}

constexpr int QK = 256;
constexpr int Q8K = 32;
constexpr int QI = 32;
constexpr int VDR = 2;
constexpr int WARPS = 4;
constexpr int WARP = 32;
constexpr int QUANT_THREADS = 256;

struct Q5KBlock {
    sycl::half2 dm;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};
struct Q20Block {
    sycl::half d;
    uint8_t qs[16];
};
struct Q3KBlock {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    sycl::half d;
};
struct IQ4XSBlock {
    sycl::half d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};
struct Q4KBlock {
    sycl::half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
};
struct Q6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    sycl::half d;
};
struct Q40Block {
    sycl::half d;
    uint8_t qs[16];
};
struct Q50Block {
    sycl::half d;
    uint8_t qh[4];
    uint8_t qs[16];
};
struct Q80Block {
    sycl::half d;
    int8_t qs[32];
};
struct IQ4NLBlock {
    sycl::half d;
    uint8_t qs[16];
};
static_assert(sizeof(Q5KBlock) == 176 && alignof(Q5KBlock) == 4);
static_assert(sizeof(Q81Block) == 36 && alignof(Q81Block) == 4);
static_assert(sizeof(Q20Block) == 18 && alignof(Q20Block) == 2 && offsetof(Q20Block, qs) == 2);
static_assert(sizeof(Q3KBlock) == 110 && alignof(Q3KBlock) == 2 && offsetof(Q3KBlock, qs) == 32 &&
              offsetof(Q3KBlock, scales) == 96 && offsetof(Q3KBlock, d) == 108);
static_assert(sizeof(IQ4XSBlock) == 136 && alignof(IQ4XSBlock) == 2 && offsetof(IQ4XSBlock, scales_h) == 2 &&
              offsetof(IQ4XSBlock, scales_l) == 4 && offsetof(IQ4XSBlock, qs) == 8);
static_assert(offsetof(Q5KBlock, scales) == 4 && offsetof(Q5KBlock, qh) == 16 && offsetof(Q5KBlock, qs) == 48 &&
              offsetof(Q81Block, qs) == 4);
static_assert(sizeof(Q4KBlock) == 144 && alignof(Q4KBlock) == 4 && offsetof(Q4KBlock, scales) == 4 &&
              offsetof(Q4KBlock, qs) == 16);
static_assert(sizeof(Q6KBlock) == 210 && alignof(Q6KBlock) == 2 && offsetof(Q6KBlock, qh) == 128 &&
              offsetof(Q6KBlock, scales) == 192 && offsetof(Q6KBlock, d) == 208);
static_assert(sizeof(Q40Block) == 18 && alignof(Q40Block) == 2 && offsetof(Q40Block, qs) == 2);
static_assert(sizeof(Q50Block) == 22 && alignof(Q50Block) == 2 && offsetof(Q50Block, qh) == 2 &&
              offsetof(Q50Block, qs) == 6);
static_assert(sizeof(Q80Block) == 34 && alignof(Q80Block) == 2 && offsetof(Q80Block, qs) == 2);
static_assert(sizeof(IQ4NLBlock) == 18 && alignof(IQ4NLBlock) == 2 && offsetof(IQ4NLBlock, qs) == 2);

inline float warp_sum(float x) {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    return sycl::reduce_over_group(sg, x, sycl::plus<float>());
}

inline float warp_max(float x) {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    return sycl::reduce_over_group(sg, x, sycl::maximum<float>());
}

void native_quantize_q8_1_kernel(const float* __restrict__ x, Q81Block* __restrict__ y, int n_in) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = int(item_ct1.get_group(2)) * QUANT_THREADS + int(item_ct1.get_local_id(2));
    if (i >= n_in) return;  // n_in is a multiple of 32: only whole warps return.
    const float xi = x[i];
    const float amax = warp_max(sycl::fabs(xi));
    const float sum = warp_sum(xi);
    const float d = strata::kernels::q8_1_finite(amax / 127.0f);
    const int8_t q = strata::kernels::q8_1_quant(xi, d, amax);
    y[i / Q8K].qs[i % Q8K] = q;
    if (i % Q8K == 0) y[i / Q8K].ds = strata::kernels::q8_1_ds(d, sum);
}

void native_swiglu_quantize_q8_1_kernel(const float* __restrict__ gate,
                                        const float* __restrict__ up,
                                        Q81Block* __restrict__ y, int n_in) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = int(item_ct1.get_group(2)) * QUANT_THREADS + int(item_ct1.get_local_id(2));
    if (i >= n_in) return;  // n_in is a multiple of 32: only whole warps return.
    const float gi = gate[i];
    const float xi = (gi / (1.0f + sycl::native::exp(-gi))) * up[i];
    const float amax = warp_max(sycl::fabs(xi));
    const float sum = warp_sum(xi);
    const float d = strata::kernels::q8_1_finite(amax / 127.0f);
    const int8_t q = strata::kernels::q8_1_quant(xi, d, amax);
    y[i / Q8K].qs[i % Q8K] = q;
    if (i % Q8K == 0) y[i / Q8K].ds = strata::kernels::q8_1_ds(d, sum);
}

// Exact pinned vec_dot_q5_K_q8_1_impl_vmmq expression and integer dot order.
inline float q5_q8_dot_impl(const int* __restrict__ vl, const int* __restrict__ vh, const int* __restrict__ u,
                            const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m, const sycl::half2& dm5,
                            const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0f0f0f0f;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0f0f0f0f;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        /*
        DPCT1064:406: Migrated __dp4a call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        const int dot1 = dp4a(v0i, u[2 * i], dp4a(v1i, u[2 * i + 1], 0));
        /*
        DPCT1064:407: Migrated __dp4a call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        const int dot2 = dp4a(0x01010101, u[2 * i], dp4a(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const sycl::float2 dm5f = dm5.template convert<float, sycl::rounding_mode::automatic>();
    return dm5f.x() * sumf_d - dm5f.y() * sumf_m;
}

/*
DPCT1110:50: The total declared local variable size in device function q5_q8_dot
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
inline float q5_q8_dot(const Q5KBlock* __restrict__ bq5, const Q81Block* __restrict__ bq8, int iqs) {
    int vl[2];
    int vh[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = bq8i->ds[0];
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q5_q8_dot_impl(vl, vh, u, sc, m, bq5->dm, d8);
}

// The generic ncols=1 oracle uses 4 warps and 1 row (or 4 rows for small K),
// eight weight blocks per K iteration, warp-ascending shared sum, then XOR tree.
template <bool SmallK>

void native_q5_k_mmvq_kernel(const Q5KBlock* __restrict__ w, const Q81Block* __restrict__ x, float* __restrict__ y,
                             int n_in, int n_out, float partial[3 /*WARPS - 1*/][(SmallK ? 4 : 1)][32 /*WARP*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = VDR * WARPS * WARP / QI;
    const int tid = WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / QK;
    float tmp[ROWS] = {};
    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK / Q8K);
        const int kqs = VDR * (tid % (QI / VDR));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            // The source assumes allocator padding for partial row groups. This
            // guard preserves every valid row's math without an out-of-bounds read.
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q5_q8_dot(w + block, x + kby, kqs);
            }
        }
    }

    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] = tmp[i];
    }
    /*
    DPCT1065:408: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l)
            tmp[i] += partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (int(item_ct1.get_local_id(2)) == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Exact pinned vec_dot_q2_0_q8_1: each thread handles one 32-element chunk.
// The weight block is only 2-byte aligned, so qs is intentionally loaded as
// int16_t, unlike the naturally 4-byte aligned activation codes.
inline float q2_q8_dot(const Q20Block* __restrict__ w, const Q81Block* __restrict__ x, int iqs) {
    const float d2 = w->d;
    const int16_t* qs = reinterpret_cast<const int16_t*>(w->qs) + iqs * 4;
    const Q81Block* chunk = x + iqs;
    const int* q8 = reinterpret_cast<const int*>(chunk->qs);
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = q8[j * 2];
        const int v = q8[j * 2 + 1];
        const int qe = byte_permute(0x020100ff, 0x020100ff, q >> 0);
        const int qo = byte_permute(0x020100ff, 0x020100ff, q >> 2);
        const int qx = byte_permute(qe, qo, 0x5140);
        const int qy = byte_permute(qe, qo, 0x7362);
        /*
        DPCT1064:409: Migrated __dp4a call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        sumi = dp4a(u, qx, sumi);
        sumi = dp4a(v, qy, sumi);
    }
    const float d8 = chunk->ds[0];
    return d2 * d8 * sumi;
}

// Q2_0 generic MMVQ: QK=64, QI=2, VDR=1, 64 blocks per iteration.
// Preserve the same outer accumulation and cross-warp reduction as the oracle.
template <bool SmallK>

void native_q2_0_mmvq_kernel(const Q20Block* __restrict__ w, const Q81Block* __restrict__ x, float* __restrict__ y,
                             int n_in, int n_out, float partial[3 /*WARPS - 1*/][(SmallK ? 4 : 1)][32 /*WARP*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 2;
    const int tid = WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / 64;
    float tmp[ROWS] = {};
    for (int kbx = tid / 2; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 2;
        const int kqs = tid % 2;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q2_q8_dot(w + block, x + kby, kqs);
            }
        }
    }

    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] = tmp[i];
    }
    /*
    DPCT1065:410: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l)
            tmp[i] += partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (int(item_ct1.get_local_id(2)) == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

inline float q3_q8_dot_impl(int vl, int vh, const int* __restrict__ u, const uint8_t* __restrict__ scales,
                            int scale_offset, float d3, const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int isc = scale_offset + 2 * i;
        const int isc_low = isc % 8;
        const int sc_shift_low = 4 * (isc / 8);
        const int sc_low = (scales[isc_low] >> sc_shift_low) & 0xf;
        const int isc_high = isc % 4;
        const int sc_shift_high = 2 * (isc / 4);
        const int sc_high = ((scales[8 + isc_high] >> sc_shift_high) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi = vsubss4(vil, vih);
        sumf += d8[i] * (dp4a(vi, u[i], 0) * sc);
    }
    return d3 * sumf;
}

inline float q3_q8_dot(const Q3KBlock* __restrict__ w, const Q81Block* __restrict__ x, int iqs) {
    const int bq8_offset = 4 * (iqs / 8);
    const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
    const float d = w->d;
    const int vl = load_int_b2(w->qs, iqs);
    const int vh = ~load_int_b2(w->hmask, iqs % 8) >> bq8_offset;
    int u[4];
    float d8[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        u[i] = reinterpret_cast<const int*>(x[bq8_offset + i].qs)[iqs % 8];
        d8[i] = x[bq8_offset + i].ds[0];
    }
    return q3_q8_dot_impl(vl, vh, u, w->scales, scale_offset, d, d8);
}

// Q3_K generic MMVQ: QK=256, QI=16, VDR=1, eight blocks per iteration.
template <bool SmallK>

void native_q3_k_mmvq_kernel(const Q3KBlock* __restrict__ w, const Q81Block* __restrict__ x, float* __restrict__ y,
                             int n_in, int n_out, float partial[3 /*WARPS - 1*/][(SmallK ? 4 : 1)][32 /*WARP*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 16;
    const int tid = WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 16; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = tid % 16;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q3_q8_dot(w + block, x + kby, kqs);
            }
        }
    }

    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] = tmp[i];
    }
    /*
    DPCT1065:411: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l)
            tmp[i] += partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (int(item_ct1.get_local_id(2)) == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// The pinned nonlinear IQ4 codebook and its CUDA two-stage byte lookup. The
// explicit alignment satisfies the four 32-bit table loads; values are unchanged.

static constexpr int8_t iq4nl_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
inline sycl::int2 iq4_table_lookup(int q4) {
    const uint32_t* table32 = reinterpret_cast<const uint32_t*>(iq4nl_values);
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = 0x32103210 | ((q4 & 0x88888888) >> 1);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = byte_permute(table32[0], table32[1], q4 >> shift);
        const uint32_t high = byte_permute(table32[2], table32[3], q4 >> shift);
        tmp[i] = byte_permute(low, high, low_high_selection_indices >> shift);
    }
    return sycl::int2(byte_permute(tmp[0], tmp[1], 0x6420), byte_permute(tmp[0], tmp[1], 0x7531));
}

// Exact pinned vec_dot_iq4_xs_q8_1: a lane consumes one 32-element subblock,
// computes integer dot products, applies signed scale in the integer domain,
// then multiplies the two half scales and integer sum in the original order.
inline float iq4_xs_q8_dot(const IQ4XSBlock* __restrict__ w, const Q81Block* __restrict__ x, int iqs) {
    const sycl::int4 q4v = ld_i4_a4(reinterpret_cast<const int*>(w->qs) + iqs);
    const sycl::int4 u0v = ld_i4_a4(reinterpret_cast<const int*>(x[iqs / 4].qs));
    const sycl::int4 u1v = ld_i4_a4(reinterpret_cast<const int*>(x[iqs / 4].qs) + 4);
    const int q4[4] = {q4v.x(), q4v.y(), q4v.z(), q4v.w()};
    const int u0[4] = {u0v.x(), u0v.y(), u0v.z(), u0v.w()};
    const int u1[4] = {u1v.x(), u1v.y(), u1v.z(), u1v.w()};
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const sycl::int2 v = iq4_table_lookup(q4[j]);
        sumi = dp4a(v.x(), u0[j], sumi);
        sumi = dp4a(v.y(), u1[j], sumi);
    }
    const int ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) | (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d =
        sycl::vec<sycl::half, 1>(w->d).convert<float, sycl::rounding_mode::automatic>()[0] * x[iqs / 4].ds[0];
    return d * sumi;
}

// IQ4_XS generic MMVQ: QK=256, QI=32, VDR=4,16 blocks per iteration.
template <bool SmallK>

void native_iq4_xs_mmvq_kernel(const IQ4XSBlock* __restrict__ w, const Q81Block* __restrict__ x, float* __restrict__ y,
                               int n_in, int n_out,

                               float partial[3 /*WARPS - 1*/][(SmallK ? 4 : 1)][32 /*WARP*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = 4 * WARPS * WARP / 32;
    const int tid = WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 8; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = 4 * (tid % 8);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += iq4_xs_q8_dot(w + block, x + kby, kqs);
            }
        }
    }

    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] = tmp[i];
    }
    /*
    DPCT1065:412: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l)
            tmp[i] += partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (int(item_ct1.get_local_id(2)) == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Exact pinned vec_dot_q4_K_q8_1_impl_vmmq expression and integer dot order.
inline float q4_q8_dot_impl(const int* __restrict__ v, const int* __restrict__ u, const uint8_t* __restrict__ sc,
                            const uint8_t* __restrict__ m, const sycl::half2& dm4, const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (v[1] >> (4 * i)) & 0x0f0f0f0f;
        /*
        DPCT1064:413: Migrated __dp4a call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        const int dot1 = dp4a(v1i, u[2 * i + 1], dp4a(v0i, u[2 * i], 0));
        /*
        DPCT1064:414: Migrated __dp4a call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        const int dot2 = dp4a(0x01010101, u[2 * i + 1], dp4a(0x01010101, u[2 * i], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const sycl::float2 dm4f = dm4.template convert<float, sycl::rounding_mode::automatic>();
    return dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}

inline float q4_q8_dot(const Q4KBlock* __restrict__ bq4, const Q81Block* __restrict__ bq8, int iqs) {
    int v[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = ql[0];
    v[1] = ql[4];

    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4->scales);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = bq8i->ds[0];
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q4_q8_dot_impl(v, u, sc, m, bq4->dm, d8);
}

// The generic ncols=1 oracle uses 4 warps and 1 row (or 4 rows for small K),
// eight weight blocks per K iteration, warp-ascending shared sum, then XOR tree.
template <bool SmallK>

void native_q4_k_mmvq_kernel(const Q4KBlock* __restrict__ w, const Q81Block* __restrict__ x, float* __restrict__ y,
                             int n_in, int n_out, float partial[3 /*WARPS - 1*/][(SmallK ? 4 : 1)][32 /*WARP*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = VDR * WARPS * WARP / QI;
    const int tid = WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / QK;
    float tmp[ROWS] = {};
    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK / Q8K);
        const int kqs = VDR * (tid % (QI / VDR));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            // The source assumes allocator padding for partial row groups. This
            // guard preserves every valid row's math without an out-of-bounds read.
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q4_q8_dot(w + block, x + kby, kqs);
            }
        }
    }

    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] = tmp[i];
    }
    /*
    DPCT1065:415: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l)
            tmp[i] += partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (int(item_ct1.get_local_id(2)) == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Exact pinned vec_dot_q6_K_q8_1: keep signed per-16-element scales,
// signed-byte subtraction, DP4A order, and the float accumulation sequence.
inline float q6_q8_dot_impl(int vl, int vh, const int* __restrict__ u, const int8_t* __restrict__ scales, float d,
                            const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = q6_sub32(vil | vih);
        sumf += d8[i] * (dp4a(vi, u[i], 0) * sc);
    }
    return d * sumf;
}

inline float q6_q8_dot(const Q6KBlock* __restrict__ w, const Q81Block* __restrict__ x, int iqs) {
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int vl = load_int_b2(w->ql, iqs);
    const int vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
    int u[2];
    float d8[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        u[i] = reinterpret_cast<const int*>(x[bq8_offset + 2 * i].qs)[iqs % 8];
        d8[i] = x[bq8_offset + 2 * i].ds[0];
    }
    return q6_q8_dot_impl(vl, vh, u, w->scales + scale_offset, w->d, d8);
}

// Q6_K generic MMVQ: QK=256, QI=32, VDR=1, four blocks per iteration.
template <bool SmallK>

void native_q6_k_mmvq_kernel(const Q6KBlock* __restrict__ w, const Q81Block* __restrict__ x, float* __restrict__ y,
                             int n_in, int n_out, float partial[3 /*WARPS - 1*/][(SmallK ? 4 : 1)][32 /*WARP*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 32;
    const int tid = WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 32; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = tid % 32;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q6_q8_dot(w + block, x + kby, kqs);
            }
        }
    }

    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] = tmp[i];
    }
    /*
    DPCT1065:417: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l)
            tmp[i] += partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (int(item_ct1.get_local_id(2)) == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// The four 32-element formats use native two-byte loads and VDR=2. The affine
// Q4_0/Q5_0 correction consumes the original-input sum stored in Q8_1, exactly
// as the pinned CUDA dot does; a signed-integer code substitution would differ.
template <bool STREAMING = false>
inline float small_q8_dot(const Q40Block* __restrict__ w, const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2<STREAMING>(w->qs, iqs + i);
        const int vi0 = (v >> 0) & 0x0f0f0f0f;
        const int vi1 = (v >> 4) & 0x0f0f0f0f;
        sumi = dp4a(vi0, reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
        sumi = dp4a(vi1, reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
    }
    const sycl::float2 ds = (x->ds).template convert<float, sycl::rounding_mode::automatic>();
    const float d = w->d;
    return d * (sumi * ds.x() - 4 * ds.y());
}

template <bool STREAMING = false>
inline float small_q8_dot(const Q50Block* __restrict__ w, const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl = load_int_b2<STREAMING>(w->qs, iqs + i);
        const int vh = load_int_b2<STREAMING>(w->qh, 0) >> (4 * (iqs + i));
        int vi0 = (vl >> 0) & 0x0f0f0f0f;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = dp4a(vi0, reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
        int vi1 = (vl >> 4) & 0x0f0f0f0f;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = dp4a(vi1, reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
    }
    const sycl::float2 ds = (x->ds).template convert<float, sycl::rounding_mode::automatic>();
    const float d = w->d;
    return d * (sumi * ds.x() - 8 * ds.y());
}

template <bool STREAMING = false>
inline float small_q8_dot(const Q80Block* __restrict__ w, const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2<STREAMING>(w->qs, iqs + i);
        const int u = reinterpret_cast<const int*>(x->qs)[iqs + i];
        sumi = dp4a(v, u, sumi);
    }
    const float d0 = w->d;
    const float d1 = x->ds[0];
    return d0 * d1 * float(sumi);
}

template <bool STREAMING = false>
inline float small_q8_dot(const IQ4NLBlock* __restrict__ w, const Q81Block* __restrict__ x, int iqs) {
    const int* q8 = reinterpret_cast<const int*>(x->qs) + iqs;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const sycl::int2 v = iq4_table_lookup(load_int_b2<STREAMING>(w->qs, iqs + i));
        sumi = dp4a(v.x(), q8[i], sumi);
        sumi = dp4a(v.y(), q8[i + 4], sumi);
    }
    const float d = sycl::vec<sycl::half, 1>(w->d).convert<float, sycl::rounding_mode::automatic>()[0] * x->ds[0];
    return d * sumi;
}

// QI=4 for Q4_0/Q5_0/IQ4_NL and QI=8 for Q8_0. With VDR=2 this preserves
// the pinned 64/32-block iteration and 2048/1024-element small-K thresholds.
template <typename Weight, int Qi, bool SmallK>

void native_small_mmvq_kernel(const Weight* __restrict__ w, const Q81Block* __restrict__ x, float* __restrict__ y,
                              int n_in, int n_out,

                              float partial[3 /*WARPS - 1*/][(SmallK ? 4 : 1)][32 /*WARP*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = 2 * WARPS * WARP / Qi;
    const int tid = WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / 32;
    float tmp[ROWS] = {};
    for (int kbx = tid / (Qi / 2); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kqs = 2 * (tid % (Qi / 2));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += small_q8_dot(w + block, x + kbx, kqs);
            }
        }
    }

    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] = tmp[i];
    }
    /*
    DPCT1065:418: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l)
            tmp[i] += partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (int(item_ct1.get_local_id(2)) == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// ============================ plan v0.3 P3: ncols = 2..8 (speculative verify, small batches)
// ============================
//
// One generic kernel for every format, parameterized by the format's iteration traits below, which are
// transcribed from the ncols = 1 kernels above (same thread-to-block mapping, same blocks per iteration, same
// small-K rule). Column j reads activation blocks x + j * (n_in / 32) and writes y + j * n_out. Each (column,
// row) value is accumulated over kbx in the same order, summed across warps in the same order and reduced with
// the same warp tree as the ncols = 1 kernel, so every column is BITWISE equal to a single-column call on that
// column (checked by bench/micro/native_mmvq_multi.cpp). The ncols = 1 kernels are untouched.
constexpr int MAX_NCOLS = 8;

// Each format splits its dot product into `load` (everything that depends only on the weight block: codes,
// unpacked scales, block scale) and `apply` (the activation loads and the original *_impl expression). `load` runs
// once per (row, block) and `apply` once per column, so adding columns adds only activation work. `apply` calls
// the same impl functions, in the same order, with the same values as the ncols = 1 dot, which is what keeps
// every column bitwise equal to it.
struct Q5KTraits {
    using Block = Q5KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W {
        int vl[2], vh[2];
        uint16_t aux[2];
        sycl::half2 dm;
        int bq8_offset;
    };
    template <bool STREAMING = false>
    static W load(const Block* __restrict__ bq5, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
        r.vl[0] = load_int_stream<STREAMING>(ql + 0);
        r.vl[1] = load_int_stream<STREAMING>(ql + 4);
        r.vh[0] = load_int_stream<STREAMING>(qh + 0) >> r.bq8_offset;
        r.vh[1] = load_int_stream<STREAMING>(qh + 4) >> r.bq8_offset;
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
        const int j = r.bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = bq5->dm;
        return r;
    }
    template <bool STREAMING = false>
    static float apply(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = bq8i->ds[0];
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            u[2 * i] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
        return q5_q8_dot_impl(r.vl, r.vh, u, sc, sc + 2, r.dm, d8);
    }
};
struct Q4KTraits {
    using Block = Q4KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W {
        int v[2];
        uint16_t aux[2];
        sycl::half2 dm;
        int bq8_offset;
    };
    template <bool STREAMING = false>
    static W load(const Block* __restrict__ bq4, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        r.v[0] = load_int_stream<STREAMING>(ql + 0);
        r.v[1] = load_int_stream<STREAMING>(ql + 4);
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4->scales);
        const int j = r.bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = bq4->dm;
        return r;
    }
    template <bool STREAMING = false>
    static float apply(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = bq8i->ds[0];
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            u[2 * i] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
        return q4_q8_dot_impl(r.v, u, sc, sc + 2, r.dm, d8);
    }
};
struct Q20Traits {
    using Block = Q20Block;
    static constexpr int DIV = 64, T = 2, KBY = 2, BPI = WARPS * WARP / 2;
    static int kqs(int tid) { return tid % 2; }
    struct W {
        int qx[4], qy[4];
        float d2;
    };
    template <bool STREAMING = false>
    static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.d2 = w->d;
        const int16_t* qs = reinterpret_cast<const int16_t*>(w->qs) + iqs * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int q;
            if constexpr (STREAMING) {
                auto props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::intel::experimental::read_hint<
                        sycl::ext::intel::experimental::cache_control<
                            sycl::ext::intel::experimental::cache_mode::streaming,
                            sycl::ext::oneapi::experimental::cache_level::L3>>};
                sycl::ext::oneapi::experimental::annotated_ptr<const int16_t, decltype(props)> aqs(qs, props);
                q = aqs[j];
            } else {
                q = qs[j];
            }
            const int qe = byte_permute(0x020100ff, 0x020100ff, q >> 0);
            const int qo = byte_permute(0x020100ff, 0x020100ff, q >> 2);
            r.qx[j] = byte_permute(qe, qo, 0x5140);
            r.qy[j] = byte_permute(qe, qo, 0x7362);
        }
        return r;
    }
    template <bool STREAMING = false>
    static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        const Q81Block* chunk = x + iqs;
        const int* q8 = reinterpret_cast<const int*>(chunk->qs);
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = dp4a(q8[j * 2], r.qx[j], sumi);
            sumi = dp4a(q8[j * 2 + 1], r.qy[j], sumi);
        }
        const float d8 = chunk->ds[0];
        return r.d2 * d8 * sumi;
    }
};
struct Q3KTraits {
    using Block = Q3KBlock;
    static constexpr int DIV = 256, T = 16, KBY = 8, BPI = WARPS * WARP / 16;
    static int kqs(int tid) { return tid % 16; }
    struct W {
        int vl, vh;
        float d;
        const uint8_t* scales;
        int scale_offset, bq8_offset;
    };
    template <bool STREAMING = false>
    static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 8);
        r.scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
        r.d = w->d;
        r.vl = load_int_b2<STREAMING>(w->qs, iqs);
        r.vh = ~load_int_b2<STREAMING>(w->hmask, iqs % 8) >> r.bq8_offset;
        r.scales = w->scales;
        return r;
    }
    template <bool STREAMING = false>
    static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int u[4];
        float d8[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            u[i] = reinterpret_cast<const int*>(x[r.bq8_offset + i].qs)[iqs % 8];
            d8[i] = x[r.bq8_offset + i].ds[0];
        }
        return q3_q8_dot_impl(r.vl, r.vh, u, r.scales, r.scale_offset, r.d, d8);
    }
};
struct Q6KTraits {
    using Block = Q6KBlock;
    static constexpr int DIV = 256, T = 32, KBY = 8, BPI = WARPS * WARP / 32;
    static int kqs(int tid) { return tid % 32; }
    struct W {
        int vl, vh;
        float d;
        const int8_t* scales;
        int bq8_offset;
    };
    template <bool STREAMING = false>
    static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        r.vl = load_int_b2<STREAMING>(w->ql, iqs);
        r.vh = load_int_b2<STREAMING>(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
        r.scales = w->scales + scale_offset;
        r.d = w->d;
        return r;
    }
    template <bool STREAMING = false>
    static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int u[2];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            u[i] = reinterpret_cast<const int*>(x[r.bq8_offset + 2 * i].qs)[iqs % 8];
            d8[i] = x[r.bq8_offset + 2 * i].ds[0];
        }
        return q6_q8_dot_impl(r.vl, r.vh, u, r.scales, r.d, d8);
    }
};
struct IQ4XSTraits {
    using Block = IQ4XSBlock;
    static constexpr int DIV = 256, T = 8, KBY = 8, BPI = 4 * WARPS * WARP / 32;
    static int kqs(int tid) { return 4 * (tid % 8); }
    struct W {
        sycl::int2 v[4];
        int ls;
        float dw;
    };
    template <bool STREAMING = false>
    static W load(const Block* __restrict__ w, int iqs) {
        W r;
        const sycl::int4 q4 = ld_i4_a4<STREAMING>(reinterpret_cast<const int*>(w->qs) + iqs);
        const int qv[4] = {q4.x(), q4.y(), q4.z(), q4.w()};
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = iq4_table_lookup(qv[j]);
        r.ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) | (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = sycl::vec<sycl::half, 1>(w->d).convert<float, sycl::rounding_mode::automatic>()[0];
        return r;
    }
    template <bool STREAMING = false>
    static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        const sycl::int4 u0 = ld_i4_a4(reinterpret_cast<const int*>(x[iqs / 4].qs));
        const sycl::int4 u1 = ld_i4_a4(reinterpret_cast<const int*>(x[iqs / 4].qs) + 4);
        const int u0v[4] = {u0.x(), u0.y(), u0.z(), u0.w()};
        const int u1v[4] = {u1.x(), u1.y(), u1.z(), u1.w()};
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = dp4a(r.v[j].x(), u0v[j], sumi);
            sumi = dp4a(r.v[j].y(), u1v[j], sumi);
        }
        sumi *= r.ls - 32;
        const float d = r.dw * x[iqs / 4].ds[0];
        return d * sumi;
    }
};
// The four 32-element formats: `load` keeps the block pointer (their decode is a few integer ops) and `apply` is
// the unchanged small_q8_dot. Their per-column cost is small; the win above is for the K and IQ formats.
template <typename Weight, int Qi>
struct SmallTraits {
    using Block = Weight;
    static constexpr int DIV = 32, T = Qi / 2, KBY = 1, BPI = 2 * WARPS * WARP / Qi;
    static int kqs(int tid) { return 2 * (tid % (Qi / 2)); }
    struct W {
        const Weight* w;
    };
    template <bool STREAMING = false>
    static W load(const Block* __restrict__ w, int) { return W{w}; }
    template <bool STREAMING = false>
    static float apply(const W& r, const Q81Block* __restrict__ x, int k) { return small_q8_dot<STREAMING>(r.w, x, k); }
};

// NW warps per block and ROWS rows per block. The EXACT layout (NW = 4, ROWS = 1, or 4 for small K) is the
// ncols = 1 layout and keeps every column bitwise equal to a single-column call. The UPSTREAM layout is
// llama.cpp's generic multi-column table (ncols 2-4: 4 warps; 5-8: 2 warps; always 2 rows per block): faster,
// equal to ncols = 1 only to float rounding (the cross-warp reduction groups partial sums differently).
bool g_multi_exact =
    true;  // until the upstream layout is timed on an idle GPU (plan rule: default only what is measured)
// B6 STRATA_B6_MMVQ_ROWS (lab, default 0 = off): see native_mmvq_set_b6_rows
int g_b6_rows = [] {
    const char* v = std::getenv("STRATA_B6_MMVQ_ROWS");
    return v ? std::atoi(v) : 0;
}();

template <typename F, int NCOLS, int NW, int ROWS, bool STREAMING = false>
/*
DPCT1110:51: The total declared local variable size in device function
native_mmvq_multi_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/

void native_mmvq_multi_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                              float* __restrict__ y, int n_in, int n_out,
                              float partial[NW - 1 > 0 ? NW - 1 : 1][NCOLS][ROWS][32 /*WARP*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int BPI = F::BPI * NW / WARPS;  // blocks per iteration scale with the warp count
    const int tid = WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = n_in / Q8K;  // Q8_1 blocks per activation column
    float tmp[NCOLS][ROWS] = {};
    for (int kbx = tid / F::T; kbx < blocks_per_row; kbx += BPI) {
        const int kby = kbx * F::KBY;
        const int kqs = F::kqs(tid);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                /*
                DPCT1084:209: The function call "Q5KTraits::load" has multiple
                migration results in different template instantiations that
                could not be unified. You may need to adjust the code.
                */
                const typename F::W wv = F::template load<STREAMING>(w + block, kqs);  // once per (row, block)
#pragma unroll
                for (int j = 0; j < NCOLS; ++j)  // then per column
                    /*
                    DPCT1084:210: The function call "Q5KTraits::apply" has
                    multiple migration results in different template
                    instantiations that could not be unified. You may need to
                    adjust the code.
                    */
                    tmp[j][i] += F::template apply<STREAMING>(wv, x + std::size_t(j) * x_stride + kby, kqs);
            }
        }
    }

    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < ROWS; ++i)
                partial[item_ct1.get_local_id(1) - 1][j][i][item_ct1.get_local_id(2)] = tmp[j][i];
    }
    /*
    DPCT1065:419: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
#pragma unroll
            for (int l = 0; l < NW - 1; ++l)
                tmp[j][i] += partial[l][j][i][item_ct1.get_local_id(2)];
            tmp[j][i] = warp_sum(tmp[j][i]);
            if (int(item_ct1.get_local_id(2)) == i && row0 + i < n_out)
                y[std::size_t(j) * n_out + row0 + i] = tmp[j][i];
        }
    }
}

// MODEL-UPDATE: the dense Q8_0 projections of Unsloth's files (all of attention, SSM and the shared expert are Q8_0
// there) take one row per block where that was measured faster - the same per-row sums and reduction as two rows per
// block (each row its own partials), so every output is the same bits.
// On Intel Xe2 (IS_XE2), n_out >= 2048 benefits from 1-row blocks (+occupancy on wide projections).
template <typename F>
bool q8_0_rows1(int n_in, int n_out) {
    if constexpr (!std::is_same_v<F, SmallTraits<Q80Block, 8>>) {
        (void)n_in;
        (void)n_out;
        return false;
    } else {
        static const int env = [] {
            const char* v = std::getenv("STRATA_NO_MMVQ_ROWS2");
            return v == nullptr || v[0] == '\0' ? -1 : v[0] != '0';
        }();
        if (env == 0) return false;
        return n_out >= 2048;
    }
}

// ---- B6 STRATA_B6_MMVQ_ROWS (lab, opt-in): the exact multi-column layout with each column's activation fragment
// loaded ONCE per weight block and reused across ROWS rows of the block.
template <typename F>
struct B6X {
    static constexpr bool ok = false;
};
template <>
struct B6X<Q6KTraits> {
    static constexpr bool ok = true;
    struct X {
        int u[2];
        float d8[2];
    };
    struct W {
        int vl, vh, sc[2];
        float d;
    };
    static X xload(const Q81Block* x, int iqs) {
        X r;
        const int off = 4 * (iqs / 16) + (iqs % 16) / 8;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            r.u[i] = reinterpret_cast<const int*>(x[off + 2 * i].qs)[iqs % 8];
            r.d8[i] = x[off + 2 * i].ds[0];
        }
        return r;
    }
    template <bool STREAMING = false>
    static W wload(const Q6KBlock* __restrict__ w, int iqs) {
        const Q6KTraits::W t = Q6KTraits::load<STREAMING>(w, iqs);
        W r;
        r.vl = t.vl;
        r.vh = t.vh;
        r.sc[0] = t.scales[0];
        r.sc[1] = t.scales[4];
        r.d = t.d;
        return r;
    }
    static float dot(const W& w, const X& x) {
        float sumf = 0.0f;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int vil = (w.vl >> (4 * i)) & 0x0f0f0f0f;
            const int vih = ((w.vh >> (4 * i)) << 4) & 0x30303030;
            const int vi = q6_sub32(vil | vih);
            sumf += x.d8[i] * (dp4a(vi, x.u[i], 0) * w.sc[i]);
        }
        return w.d * sumf;
    }
};
template <typename T, int KIND>
struct B6XK {
    static constexpr bool ok = true;
    struct X {
        int u[4];
        float d8[2];
    };
    using W = typename T::W;
    static X xload(const Q81Block* bq8, int iqs) {
        X r;
        const int off = 2 * ((iqs / 2) / 4);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + off + i;
            r.d8[i] = bq8i->ds[0];
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            r.u[2 * i] = q8[0];
            r.u[2 * i + 1] = q8[4];
        }
        return r;
    }
    template <bool STREAMING = false>
    static W wload(const typename T::Block* __restrict__ w, int iqs) { return T::template load<STREAMING>(w, iqs); }
    static float dot(const W& w, const X& x) {
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(w.aux);
        if constexpr (KIND == 4)
            return q4_q8_dot_impl(w.v, x.u, sc, sc + 2, w.dm, x.d8);
        else
            return q5_q8_dot_impl(w.vl, w.vh, x.u, sc, sc + 2, w.dm, x.d8);
    }
};
template <>
struct B6X<Q4KTraits> : B6XK<Q4KTraits, 4> {};
template <>
struct B6X<Q5KTraits> : B6XK<Q5KTraits, 5> {};
template <>
struct B6X<IQ4XSTraits> {
    static constexpr bool ok = true;
    struct X {
        int u0[4], u1[4];
        float dx;
    };
    using W = IQ4XSTraits::W;
    static X xload(const Q81Block* x, int iqs) {
        X r;
        const sycl::int4 u0 = ld_i4_a4(reinterpret_cast<const int*>(x[iqs / 4].qs));
        const sycl::int4 u1 = ld_i4_a4(reinterpret_cast<const int*>(x[iqs / 4].qs) + 4);
        r.u0[0] = u0.x();
        r.u0[1] = u0.y();
        r.u0[2] = u0.z();
        r.u0[3] = u0.w();
        r.u1[0] = u1.x();
        r.u1[1] = u1.y();
        r.u1[2] = u1.z();
        r.u1[3] = u1.w();
        r.dx = x[iqs / 4].ds[0];
        return r;
    }
    template <bool STREAMING = false>
    static W wload(const IQ4XSBlock* __restrict__ w, int iqs) { return IQ4XSTraits::load<STREAMING>(w, iqs); }
    static float dot(const W& w, const X& x) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = dp4a(w.v[j].x(), x.u0[j], sumi);
            sumi = dp4a(w.v[j].y(), x.u1[j], sumi);
        }
        sumi *= w.ls - 32;
        const float d = w.dw * x.dx;
        return d * sumi;
    }
};

template <typename F, int NCOLS, int ROWS, bool STREAMING = false>
void b6_mmvq_rows_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x, float* __restrict__ y,
                         int n_in, int n_out, float partial[WARPS - 1][NCOLS][ROWS][WARP]) {
    using XB = B6X<F>;
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int tid = WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = n_in / Q8K;
    const int kqs = F::kqs(tid);
    float tmp[NCOLS][ROWS] = {};
    for (int kbx = tid / F::T; kbx < blocks_per_row; kbx += F::BPI) {
        const int kby = kbx * F::KBY;
        typename XB::X xv[NCOLS];
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) xv[j] = XB::xload(x + std::size_t(j) * x_stride + kby, kqs);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const typename XB::W wv = XB::template wload<STREAMING>(w + std::size_t(row0 + i) * blocks_per_row + kbx, kqs);
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) tmp[j][i] += XB::dot(wv, xv[j]);
            }
        }
    }
    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < ROWS; ++i)
                partial[item_ct1.get_local_id(1) - 1][j][i][item_ct1.get_local_id(2)] = tmp[j][i];
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
#pragma unroll
            for (int l = 0; l < WARPS - 1; ++l) tmp[j][i] += partial[l][j][i][item_ct1.get_local_id(2)];
            tmp[j][i] = warp_sum(tmp[j][i]);
            if (int(item_ct1.get_local_id(2)) == i && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = tmp[j][i];
        }
    }
}

inline bool mmvq_large_grf(sycl::queue* s) {
    static const bool v = [s] {
        const char* e = std::getenv("STRATA_MMVQ_LARGE_GRF");
        if (e != nullptr) return std::atoi(e) != 0;
        const auto gen = strata::intel_gpu_gen(s->get_device());
        return gen == strata::IntelGpuGen::Alchemist || gen == strata::IntelGpuGen::Battlemage;
    }();
    return v;
}

static bool g_mmvq_decode_256grf = [] {
    const char* d = std::getenv("STRATA_MMVQ_DECODE_256GRF");
    return d != nullptr && std::atoi(d) != 0;
}();

inline bool mmvq_decode_large_grf(sycl::queue* /*s*/) {
    return g_mmvq_decode_256grf;
}

inline int mmvq_sg() {
    static const int v = std::getenv("STRATA_MMVQ_SG") ? std::atoi(std::getenv("STRATA_MMVQ_SG")) : 32;
    return v;
}
inline bool mmvq_sg16(bool iq4xs, int n_out) {
    const int v = mmvq_sg();
    return v == 16 || ((v & 1) && v < 4 && iq4xs) || ((v & 2) && v < 4 && n_out <= 1024);
}

template <typename F, int NCOLS, int ROWS>
struct B6RowsGrf256Functor {
    const typename F::Block* w;
    const Q81Block* x;
    float* y;
    int n_in, n_out;
    sycl::local_accessor<float[WARPS - 1][NCOLS][ROWS][WARP], 0> partial_acc;
    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        return sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>,
                                                           sycl::ext::oneapi::experimental::sub_group_size<32>};
    }
    void operator()(sycl::nd_item<3>) const {
        b6_mmvq_rows_kernel<F, NCOLS, ROWS, true>(w, x, y, n_in, n_out, partial_acc);
    }
};

template <typename F, int NCOLS>
bool b6_launch(const typename F::Block* w, const Q81Block* x, float* y, int n_in, int n_out, sycl::queue* s) {
    if constexpr (!B6X<F>::ok) {
        return false;
    } else {
        if (g_b6_rows <= 0 || !g_multi_exact || n_in / F::DIV < 1) return false;
        const int rows = g_b6_rows == 1 ? 2 : g_b6_rows;
        const sycl::range<3> threads(1, WARPS, WARP);
        const bool large_grf = (NCOLS > 1) ? mmvq_large_grf(s) : mmvq_decode_large_grf(s);
        if (rows == 2) {
            const unsigned blocks = unsigned((n_out + 1) / 2);
            s->submit([&](sycl::handler& cgh) {
                sycl::local_accessor<float[WARPS - 1][NCOLS][2][WARP], 0> partial_acc(cgh);
                auto r = sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * threads, threads);
                if (large_grf) {
                    cgh.parallel_for(r, B6RowsGrf256Functor<F, NCOLS, 2>{w, x, y, n_in, n_out, partial_acc});
                } else {
                    cgh.parallel_for(r, [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
                        b6_mmvq_rows_kernel<F, NCOLS, 2>(w, x, y, n_in, n_out, partial_acc);
                    });
                }
            });
        } else if (rows == 8) {
            const unsigned blocks = unsigned((n_out + 7) / 8);
            s->submit([&](sycl::handler& cgh) {
                sycl::local_accessor<float[WARPS - 1][NCOLS][8][WARP], 0> partial_acc(cgh);
                auto r = sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * threads, threads);
                if (large_grf) {
                    cgh.parallel_for(r, B6RowsGrf256Functor<F, NCOLS, 8>{w, x, y, n_in, n_out, partial_acc});
                } else {
                    cgh.parallel_for(r, [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
                        b6_mmvq_rows_kernel<F, NCOLS, 8>(w, x, y, n_in, n_out, partial_acc);
                    });
                }
            });
        } else {
            const unsigned blocks = unsigned((n_out + 3) / 4);
            s->submit([&](sycl::handler& cgh) {
                sycl::local_accessor<float[WARPS - 1][NCOLS][4][WARP], 0> partial_acc(cgh);
                auto r = sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * threads, threads);
                if (large_grf) {
                    cgh.parallel_for(r, B6RowsGrf256Functor<F, NCOLS, 4>{w, x, y, n_in, n_out, partial_acc});
                } else {
                    cgh.parallel_for(r, [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
                        b6_mmvq_rows_kernel<F, NCOLS, 4>(w, x, y, n_in, n_out, partial_acc);
                    });
                }
            });
        }
        return true;
    }
}

template <typename F, int NCOLS, int NW, int ROWS>
struct NativeMmvqMultiGrf256Functor {
    const typename F::Block* w;
    const Q81Block* x;
    float* y;
    int n_in, n_out;
    sycl::local_accessor<float[NW - 1 > 0 ? NW - 1 : 1][NCOLS][ROWS][32 /*WARP*/], 0> partial_acc;
    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        return sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>,
                                                           sycl::ext::oneapi::experimental::sub_group_size<32>};
    }
    void operator()(sycl::nd_item<3>) const {
        native_mmvq_multi_kernel<F, NCOLS, NW, ROWS, true>(w, x, y, n_in, n_out, partial_acc);
    }
};

template <typename F, int NCOLS>
void launch_multi_n(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, sycl::queue* s) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    if (b6_launch<F, NCOLS>(w, x, y, n_in, n_out, s)) return;
    const bool large_grf = (NCOLS > 1) ? mmvq_large_grf(s) : mmvq_decode_large_grf(s);
    if (!g_multi_exact) {
        constexpr int NW = NCOLS <= 4 ? 4 : 2;
        const unsigned blocks = unsigned((std::size_t(n_out) + 1) / 2);
        s->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float[NW - 1 > 0 ? NW - 1 : 1][NCOLS][2][32 /*WARP*/], 0> partial_acc_ct1(cgh);
            auto r = sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, NW, WARP),
                                       sycl::range<3>(1, NW, WARP));
            if (large_grf) {
                cgh.parallel_for(r, NativeMmvqMultiGrf256Functor<F, NCOLS, NW, 2>{w, x, y, n_in, n_out, partial_acc_ct1});
            } else {
                cgh.parallel_for(r, [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                    native_mmvq_multi_kernel<F, NCOLS, NW, 2>(w, x, y, n_in, n_out, partial_acc_ct1);
                });
            }
        });
        return;
    }
    const sycl::range<3> threads(1, WARPS, WARP);
    // #783 PR-h (stuchapin909): two rows per block for every K, not only small K - each row keeps its own partial sums and
    // reduction, so a row's result does not depend on its neighbour; STRATA_NO_MMVQ_ROWS2=1 keeps one row for large K
    static const bool rows1 = [] {
        const char* v = std::getenv("STRATA_NO_MMVQ_ROWS2");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    if ((rows1 || q8_0_rows1<F>(n_in, n_out)) && n_in / F::DIV >= F::BPI) {
        s->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float[WARPS - 1 > 0 ? WARPS - 1 : 1][NCOLS][1][32 /*WARP*/], 0> partial_acc_ct1(cgh);
            auto r = sycl::nd_range<3>(sycl::range<3>(1, 1, unsigned(n_out)) * threads, threads);
            if (large_grf) {
                cgh.parallel_for(r, NativeMmvqMultiGrf256Functor<F, NCOLS, WARPS, 1>{w, x, y, n_in, n_out, partial_acc_ct1});
            } else {
                cgh.parallel_for(r, [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                    native_mmvq_multi_kernel<F, NCOLS, WARPS, 1>(w, x, y, n_in, n_out, partial_acc_ct1);
                });
            }
        });
        return;
    }
    if (n_in / F::DIV < F::BPI) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        s->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float[WARPS - 1 > 0 ? WARPS - 1 : 1][NCOLS][WARPS][32 /*WARP*/], 0> partial_acc_ct1(
                cgh);
            auto r = sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * threads, threads);
            if (large_grf) {
                cgh.parallel_for(r, NativeMmvqMultiGrf256Functor<F, NCOLS, WARPS, WARPS>{w, x, y, n_in, n_out, partial_acc_ct1});
            } else {
                cgh.parallel_for(r, [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                    native_mmvq_multi_kernel<F, NCOLS, WARPS, WARPS>(w, x, y, n_in, n_out,
                                                                     partial_acc_ct1);
                });
            }
        });
        return;
    }
    constexpr int ROWS = 2;
    const unsigned blocks = unsigned((std::size_t(n_out) + ROWS - 1) / ROWS);
    s->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float[WARPS - 1 > 0 ? WARPS - 1 : 1][NCOLS][ROWS][32 /*WARP*/], 0> partial_acc_ct1(cgh);
        auto r = sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * threads, threads);
        if (large_grf) {
            cgh.parallel_for(r, NativeMmvqMultiGrf256Functor<F, NCOLS, WARPS, ROWS>{w, x, y, n_in, n_out, partial_acc_ct1});
        } else {
            cgh.parallel_for(r, [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                native_mmvq_multi_kernel<F, NCOLS, WARPS, ROWS>(w, x, y, n_in, n_out, partial_acc_ct1);
            });
        }
    });
}

template <typename F>
void launch_multi(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    sycl::queue* s = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    switch (ncols) {
        case 2:
            launch_multi_n<F, 2>(weights, x_q8_1, y, n_in, n_out, s);
            break;
        case 3:
            launch_multi_n<F, 3>(weights, x_q8_1, y, n_in, n_out, s);
            break;
        case 4:
            launch_multi_n<F, 4>(weights, x_q8_1, y, n_in, n_out, s);
            break;
        case 5:
            launch_multi_n<F, 5>(weights, x_q8_1, y, n_in, n_out, s);
            break;
        case 6:
            launch_multi_n<F, 6>(weights, x_q8_1, y, n_in, n_out, s);
            break;
        case 7:
            launch_multi_n<F, 7>(weights, x_q8_1, y, n_in, n_out, s);
            break;
        case 8:
            launch_multi_n<F, 8>(weights, x_q8_1, y, n_in, n_out, s);
            break;
        default:
            throw std::invalid_argument("native MMVQ multi-column launch requires 2 <= ncols <= 8");
    }
}

void validate_shape(int n_in, int ncols, int block_elems = Q8K) {
    if (n_in <= 0 || n_in % block_elems != 0) {
        throw std::invalid_argument("native MMVQ requires n_in > 0 and divisible by its block element count");
    }
    if (ncols < 1 || ncols > MAX_NCOLS) throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 8");
}
void validate_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % 4 != 0) {
        throw std::invalid_argument("native MMVQ requires non-null 4-byte aligned device pointers");
    }
}
void validate_stream(void* /*stream*/) {}
void launch_check() {}

template <typename Weight, int Qi>
void small_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 32);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<SmallTraits<Weight, Qi>>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Weight*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    sycl::queue* s = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    const sycl::range<3> threads(1, WARPS, WARP);
    if (n_in / 32 < 2 * WARPS * WARP / Qi) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        /*
        DPCT1049:55: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            s->submit([&](sycl::handler& cgh) {
                /*
                DPCT1101:838: 'WARPS - 1' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                /*
                DPCT1101:839: 'WARP' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<float[3 /*WARPS - 1*/][4][32 /*WARP*/], 0> partial_acc_ct1(cgh);

                cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * threads, threads),
                                 [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                                     native_small_mmvq_kernel<Weight, Qi, true>(w, x, y, n_in, n_out, partial_acc_ct1);
                                 });
            });
        }
    } else {
        /*
        DPCT1049:56: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */

        s->submit([&](sycl::handler& cgh) {
            /*
            DPCT1101:840: 'WARPS - 1' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            /*
            DPCT1101:841: 'WARP' expression was replaced with a value. Modify
            the code to use the original expression, provided in comments, if
            it is correct.
            */
            sycl::local_accessor<float[3 /*WARPS - 1*/][1][32 /*WARP*/], 0> partial_acc_ct1(cgh);

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, unsigned(n_out)) * threads, threads),
                             [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                                 native_small_mmvq_kernel<Weight, Qi, false>(w, x, y, n_in, n_out, partial_acc_ct1);
                             });
        });
    }
    launch_check();
}

template <typename Weight, int Qi>
void small_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out, int ncols,
               void* stream) {
    validate_shape(n_in, ncols, 32);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    small_mmvq<Weight, Qi>(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

inline bool q6k_a2() {
    static const bool v = std::getenv("STRATA_MMVQ_A2") == nullptr || std::atoi(std::getenv("STRATA_MMVQ_A2")) != 0;
    return v;
}

template <int NCOLS, int RPW, int SG = WARP, int BS = (int)sizeof(Q6KBlock), bool A2 = false>
void native_mmvq_q6k_wide_kernel(const Q6KBlock* __restrict__ w, const Q81Block* __restrict__ x, float* __restrict__ y,
                                 int n_in, int n_out) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = int(item.get_local_id(2)), warp = int(item.get_local_id(1));
    constexpr int NWG = WARPS * WARP / SG;
    const int row0 = (int(item.get_group(2)) * NWG + warp) * RPW;
    if (row0 >= n_out) return;
    const int blocks_per_row = n_in / QK, x_stride = n_in / Q8K;
    const int g = lane & 7, sub = lane >> 3;
    const int vh_shift = 2 * ((g & 3) >> 1);
    const int qh4_idx = 2 * (g >> 2) + (g & 1);
    const int scale_offset = 8 * (g >> 2) + (g & 3);
    const int bq8_offset = 4 * (g >> 2) + ((g & 3) >> 1);
    const int u_int4 = g & 1;
    float acc[RPW][NCOLS] = {};
    for (int kbx = sub; kbx < blocks_per_row; kbx += SG / 8) {
        const int kby = kbx * (QK / Q8K);
        sycl::int4 ql4[RPW], qh4[RPW];
        float dsc0[RPW], dsc1[RPW];
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const Q6KBlock* b = reinterpret_cast<const Q6KBlock*>(
                reinterpret_cast<const uint8_t*>(w) +
                (std::size_t(row0 + (row0 + r < n_out ? r : 0)) * blocks_per_row + kbx) * BS);
            if constexpr (A2) {
                ql4[r] = load16_a2(b->ql + 16 * g);
                qh4[r] = load16_a2(b->qh + 16 * qh4_idx);
            } else {
                ql4[r] = reinterpret_cast<const sycl::int4*>(b->ql)[g];
                qh4[r] = reinterpret_cast<const sycl::int4*>(b->qh)[qh4_idx];
            }
            const float d = b->d;
            dsc0[r] = d * (float)b->scales[scale_offset];
            dsc1[r] = d * (float)b->scales[scale_offset + 4];
        }
        sycl::int4 u0[NCOLS], u1[NCOLS];
        float ds0[NCOLS], ds1[NCOLS];
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            const Q81Block* xb = x + std::size_t(j) * x_stride + kby + bq8_offset;
            if constexpr (A2) {
                u0[j] = ld_i4_a4(reinterpret_cast<const int*>(xb[0].qs) + 4 * u_int4);
                u1[j] = ld_i4_a4(reinterpret_cast<const int*>(xb[2].qs) + 4 * u_int4);
            } else {
                u0[j] = reinterpret_cast<const sycl::int4*>(xb[0].qs)[u_int4];
                u1[j] = reinterpret_cast<const sycl::int4*>(xb[2].qs)[u_int4];
            }
            ds0[j] = xb[0].ds[0];
            ds1[j] = xb[2].ds[0];
        }
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int vl[4] = {ql4[r].x(), ql4[r].y(), ql4[r].z(), ql4[r].w()};
            const int vh[4] = {qh4[r].x() >> vh_shift, qh4[r].y() >> vh_shift, qh4[r].z() >> vh_shift,
                               qh4[r].w() >> vh_shift};
            int vi0[4], vi1[4];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                vi0[p] = q6_sub32((vl[p] & 0x0f0f0f0f) | ((vh[p] << 4) & 0x30303030));
                vi1[p] = q6_sub32(((vl[p] >> 4) & 0x0f0f0f0f) | (((vh[p] >> 4) << 4) & 0x30303030));
            }
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                int s0 = 0, s1 = 0;
                s0 = dp4a(vi0[0], u0[j].x(), s0);
                s0 = dp4a(vi0[1], u0[j].y(), s0);
                s0 = dp4a(vi0[2], u0[j].z(), s0);
                s0 = dp4a(vi0[3], u0[j].w(), s0);
                s1 = dp4a(vi1[0], u1[j].x(), s1);
                s1 = dp4a(vi1[1], u1[j].y(), s1);
                s1 = dp4a(vi1[2], u1[j].z(), s1);
                s1 = dp4a(vi1[3], u1[j].w(), s1);
                acc[r][j] += ds0[j] * dsc0[r] * (float)s0 + ds1[j] * dsc1[r] * (float)s1;
            }
        }
    }
    auto sg = item.get_sub_group();
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            float v = acc[r][j];
#pragma unroll
            for (int o = SG / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
            if (lane == 0 && row0 + r < n_out) y[std::size_t(j) * n_out + row0 + r] = v;
        }
}

inline int q6k_wide_rpw() {
    static const int v = std::getenv("STRATA_MMVQ_WIDE_RPW") ? std::atoi(std::getenv("STRATA_MMVQ_WIDE_RPW")) : 1;
    return v;
}

template <int NCOLS>
struct Q6kWideA2Grf256Functor {
    const Q6KBlock* w;
    const Q81Block* x;
    float* y;
    int n_in, n_out;
    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        return sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>,
                                                           sycl::ext::oneapi::experimental::sub_group_size<32>};
    }
    void operator()(sycl::nd_item<3>) const {
        native_mmvq_q6k_wide_kernel<NCOLS, 1, WARP, (int)sizeof(Q6KBlock), true>(w, x, y, n_in, n_out);
    }
};

template <int NCOLS>
void launch_q6k_wide(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, sycl::queue* s) {
    const auto* w = static_cast<const Q6KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const int rpw = q6k_wide_rpw();
    if (rpw == 1 && q6k_a2()) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        const bool use_256 = (NCOLS > 1) ? mmvq_large_grf(s) : mmvq_decode_large_grf(s);
        if (use_256) {
            s->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, WARPS, WARP),
                                              sycl::range<3>(1, WARPS, WARP)),
                            Q6kWideA2Grf256Functor<NCOLS>{w, x, y, n_in, n_out});
            return;
        }
        if (mmvq_sg16(false, n_out)) {
            constexpr int NWG = WARPS * WARP / 16;
            const unsigned blocks16 = unsigned((std::size_t(n_out) + NWG - 1) / NWG);
            s->parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, blocks16) * sycl::range<3>(1, NWG, 16), sycl::range<3>(1, NWG, 16)),
                [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(16)]] {
                    native_mmvq_q6k_wide_kernel<NCOLS, 1, 16, (int)sizeof(Q6KBlock), true>(w, x, y, n_in, n_out);
                });
            return;
        }
        s->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, WARPS, WARP),
                                          sycl::range<3>(1, WARPS, WARP)),
                        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
                            native_mmvq_q6k_wide_kernel<NCOLS, 1, WARP, (int)sizeof(Q6KBlock), true>(w, x, y, n_in,
                                                                                                     n_out);
                        });
        return;
    }
#define STRATA_Q6W(RP)                                                                                             \
    if (rpw == RP) {                                                                                               \
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS * RP - 1) / (WARPS * RP));                    \
        if (q6k_a2()) {                                                                                            \
            s->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, WARPS, WARP),       \
                                              sycl::range<3>(1, WARPS, WARP)),                                     \
                            [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {                              \
                                native_mmvq_q6k_wide_kernel<NCOLS, RP, WARP, (int)sizeof(Q6KBlock), true>(         \
                                    w, x, y, n_in, n_out);                                                         \
                            });                                                                                    \
            return;                                                                                                \
        }                                                                                                          \
        s->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, WARPS, WARP),           \
                                          sycl::range<3>(1, WARPS, WARP)),                                         \
                        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {                                  \
                            native_mmvq_q6k_wide_kernel<NCOLS, RP>(w, x, y, n_in, n_out);                          \
                        });                                                                                        \
        return;                                                                                                    \
    }
    STRATA_Q6W(1) STRATA_Q6W(2) STRATA_Q6W(4)
#undef STRATA_Q6W
    const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
    s->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, WARPS, WARP),
                                      sycl::range<3>(1, WARPS, WARP)),
                    [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
                        native_mmvq_q6k_wide_kernel<NCOLS, 1>(w, x, y, n_in, n_out);
                    });
}

template <int NCOLS>
void launch_q6k_wide224(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, sycl::queue* s) {
    const auto* w = static_cast<const Q6KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
    s->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, WARPS, WARP),
                                      sycl::range<3>(1, WARPS, WARP)),
                    [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
                        native_mmvq_q6k_wide_kernel<NCOLS, 1, WARP, 224>(w, x, y, n_in, n_out);
                    });
}

bool g_q6k_wide = std::getenv("STRATA_MMVQ_WIDE") != nullptr && std::atoi(std::getenv("STRATA_MMVQ_WIDE")) != 0;

// SYCL port: KREG small-batch kernels for Q4_K, Q5_K, Q6_K and IQ4_XS (STRATA_MMVQ_KREG=1, default off)
constexpr int KREG_SG = 16;
inline bool kreg_env() {   // latched before graph capture
    static const bool v = [] {
        const char* e = std::getenv("STRATA_MMVQ_KREG");
        const bool on = e && e[0] == '1';
        if (on) std::fprintf(stderr, "strata: STRATA_MMVQ_KREG=1: Q4_K/Q5_K/Q6_K/IQ4_XS mmvq at 1-8 columns runs the KREG kernels\n");
        return on;
    }();
    return v;
}
static int g_kreg_override = -1;   // native_mmvq_set_kreg (tests): -1 = the STRATA_MMVQ_KREG latch
inline bool kreg_on() { return g_kreg_override >= 0 ? g_kreg_override != 0 : kreg_env(); }

constexpr int kXe2KqGrid = 2, kXe2KqTile = 4, kXe2KqHead = 8, kXe2KqPair = 16;
constexpr int kXe2KqAll = kXe2KqGrid | kXe2KqTile | kXe2KqHead | kXe2KqPair;
constexpr int kXe2KqDefault = kXe2KqGrid | kXe2KqTile | kXe2KqPair;   // what "1" turns on
inline int xe2_kq_env() {   // latched before graph capture
    static const int v = [] {
        const char* e = std::getenv("STRATA_XE2_KQ");
        if (e == nullptr || e[0] == '\0') return 0;
        const int n = std::atoi(e);
        const int m = n == 1 ? kXe2KqDefault : (n & kXe2KqAll);
        if (m != 0)
            std::fprintf(stderr, "strata: STRATA_XE2_KQ=%s: dense decode launches for Q5_K / Q6_K / Q8_0 (parts %d; 0 turns them off)\n", e, m);
        return m;
    }();
    return v;
}
static int g_xe2_kq_override = -1;   // native_mmvq_set_xe2_kq (tests): -1 = the STRATA_XE2_KQ latch
inline bool xe2_kq(int part) { return ((g_xe2_kq_override >= 0 ? g_xe2_kq_override : xe2_kq_env()) & part) != 0; }
static std::atomic<int> g_xe2_kq_seen{0};
inline void xe2_kq_note(int part) { g_xe2_kq_seen.fetch_or(part, std::memory_order_relaxed); }

inline int kreg_groups_cap(int ncols, int nsg, int grf, sycl::queue* s) {
    static const int cu = int(s->get_device().get_info<sycl::info::device::max_compute_units>());
    static const int cores = std::max(1, cu >= 64 ? cu / 8 : cu);   // Intel reports XVEs (8 a core)
    static const std::pair<int, int> w = [] {
        int a = -1, b = -1;
        if (const char* e = std::getenv("STRATA_KREG_WAVES")) std::sscanf(e, "%d,%d", &a, &b);
        return std::make_pair(a, b);
    }();
    const int per_core = (grf == 256 ? 32 : 64) / nsg;
    int waves = ncols == 1 ? w.first : w.second;
    if (waves < 0) waves = ncols >= 3 && per_core >= 2 ? 1 : 0;
    if (waves <= 0) return 1 << 30;
    return std::max(1, waves * cores * std::max(1, per_core));
}

template <int O, int N, int P>
inline void kreg_tsum(float (&v)[P], const sycl::sub_group& sg, int lane) {
    if constexpr (O > 0) {
        constexpr int H = N / 2;
        const bool up = (lane & O) != 0;
#pragma unroll
        for (int i = 0; i < H; ++i) {
            const float lo = v[i], hi = v[i + H];
            const float send = up ? lo : hi;
            const float keep = up ? hi : lo;
            v[i] = keep + sycl::permute_group_by_xor(sg, send, O);
        }
        kreg_tsum<O / 2, H, P>(v, sg, lane);
    }
}

inline sycl::int4 ld_q8_16(const Q81Block* b, int half) {
    return ld_i4_a4(reinterpret_cast<const int*>(b->qs) + 4 * half);
}

inline int dp4a4(const sycl::int4 a, const sycl::int4 b, int acc) {
    acc = dp4a(a.x(), b.x(), acc); acc = dp4a(a.y(), b.y(), acc);
    acc = dp4a(a.z(), b.z(), acc); return dp4a(a.w(), b.w(), acc);
}

struct KregActK { sycl::int4 ua, uc; float da, dc, sa, sc; };
inline KregActK kreg_act_k(const Q81Block* xb, int g) {
    const int sb = 2 * (g >> 1), half = g & 1;
    const sycl::int4 ones(0x01010101, 0x01010101, 0x01010101, 0x01010101);
    KregActK v;
    v.ua = ld_q8_16(xb + sb, half); v.uc = ld_q8_16(xb + sb + 1, half);
    v.da = (float) xb[sb].ds[0]; v.dc = (float) xb[sb + 1].ds[0];
    v.sa = v.da * (float) dp4a4(ones, v.ua, 0);
    v.sc = v.dc * (float) dp4a4(ones, v.uc, 0);
    return v;
}

struct KregWK { sycl::int4 lo, hi; float dl, dh, ml, mh; };
inline void kreg_scales_k(const sycl::int4 hdr, int g, KregWK& r) {
    const int sb = 2 * (g >> 1), sh = 8 * (sb & 3);
    const uint32_t A = ((uint32_t) hdr.y() >> sh) & 0xFFFF, B = ((uint32_t) hdr.z() >> sh) & 0xFFFF,
                   C = ((uint32_t) hdr.w() >> sh) & 0xFFFF;
    const uint32_t ls0 = A & 63, ls1 = (A >> 8) & 63, lm0 = B & 63, lm1 = (B >> 8) & 63;
    const uint32_t hs0 = (C & 0xF) | (((A >> 6) & 3) << 4), hs1 = ((C >> 8) & 0xF) | (((A >> 14) & 3) << 4);
    const uint32_t hm0 = ((C >> 4) & 0xF) | (((B >> 6) & 3) << 4), hm1 = ((C >> 12) & 0xF) | (((B >> 14) & 3) << 4);
    const bool lo4 = sb < 4;
    const uint32_t s0 = lo4 ? ls0 : hs0, s1 = lo4 ? ls1 : hs1, m0 = lo4 ? lm0 : hm0, m1 = lo4 ? lm1 : hm1;
    const sycl::half2 dmh = sycl::bit_cast<sycl::half2>(hdr.x());
    const float d = (float) dmh.x(), dmin = (float) dmh.y();
    r.dl = d * (float) s0; r.ml = dmin * (float) m0; r.dh = d * (float) s1; r.mh = dmin * (float) m1;
}

inline float kreg_dot_k(const KregWK& r, const KregActK& v) {
    return r.dl * (v.da * (float) dp4a4(r.lo, v.ua, 0)) - r.ml * v.sa +
           r.dh * (v.dc * (float) dp4a4(r.hi, v.uc, 0)) - r.mh * v.sc;
}

static constexpr int8_t kIq4nlTable[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

struct KregQ4K {
    static constexpr int kTableWords = 0;
    static uint32_t table_entry(int) { return 0; }
    using Block = Q4KBlock; using A = KregActK; using W = KregWK;
    static A act(const Q81Block* xb, int g) { return kreg_act_k(xb, g); }
    static W load(const Block* b, int g) {
        W r;
        const sycl::int4 hdr = *reinterpret_cast<const sycl::int4*>(b);
        const sycl::int4 q = reinterpret_cast<const sycl::int4*>(b->qs)[g];
        r.lo = sycl::int4(q.x() & 0x0f0f0f0f, q.y() & 0x0f0f0f0f, q.z() & 0x0f0f0f0f, q.w() & 0x0f0f0f0f);
        r.hi = sycl::int4((q.x() >> 4) & 0x0f0f0f0f, (q.y() >> 4) & 0x0f0f0f0f, (q.z() >> 4) & 0x0f0f0f0f, (q.w() >> 4) & 0x0f0f0f0f);
        kreg_scales_k(hdr, g, r);
        return r;
    }
    static float dot(const W& r, const A& v) { return kreg_dot_k(r, v); }
};

struct KregQ5K {
    static constexpr int kTableWords = 0;
    static uint32_t table_entry(int) { return 0; }
    using Block = Q5KBlock; using A = KregActK; using W = KregWK;
    static A act(const Q81Block* xb, int g) { return kreg_act_k(xb, g); }
    static W load(const Block* b, int g) {
        W r;
        const int sb = 2 * (g >> 1), half = g & 1;
        const sycl::int4 hdr = *reinterpret_cast<const sycl::int4*>(b);
        const sycl::int4 q = reinterpret_cast<const sycl::int4*>(b->qs)[g];
        const sycl::int4 h = reinterpret_cast<const sycl::int4*>(b->qh)[half];
        auto hb = [](int v, int bit) { return ((v >> bit) & 0x01010101) << 4; };
        r.lo = sycl::int4((q.x() & 0x0f0f0f0f) | hb(h.x(), sb), (q.y() & 0x0f0f0f0f) | hb(h.y(), sb),
                          (q.z() & 0x0f0f0f0f) | hb(h.z(), sb), (q.w() & 0x0f0f0f0f) | hb(h.w(), sb));
        r.hi = sycl::int4(((q.x() >> 4) & 0x0f0f0f0f) | hb(h.x(), sb + 1), ((q.y() >> 4) & 0x0f0f0f0f) | hb(h.y(), sb + 1),
                          ((q.z() >> 4) & 0x0f0f0f0f) | hb(h.z(), sb + 1), ((q.w() >> 4) & 0x0f0f0f0f) | hb(h.w(), sb + 1));
        kreg_scales_k(hdr, g, r);
        return r;
    }
    static float dot(const W& r, const A& v) { return kreg_dot_k(r, v); }
};

struct KregQ6K {
    static constexpr int kTableWords = 0;
    static uint32_t table_entry(int) { return 0; }
    using Block = Q6KBlock;
    struct A { sycl::int4 u0, u1; float ds0, ds1; };
    struct W { int vi0[4], vi1[4]; float dsc0, dsc1; };
    static A act(const Q81Block* xb, int g) {
        const int bq8_offset = 4 * (g >> 2) + ((g & 3) >> 1);
        A a;
        a.u0 = ld_q8_16(xb + bq8_offset, g & 1); a.u1 = ld_q8_16(xb + bq8_offset + 2, g & 1);
        a.ds0 = (float) xb[bq8_offset].ds[0]; a.ds1 = (float) xb[bq8_offset + 2].ds[0];
        return a;
    }
    static W load(const Block* b, int g) {
        W r;
        const int vh_shift = 2 * ((g & 3) >> 1);
        const int qh4_idx = 2 * (g >> 2) + (g & 1);
        const int scale_offset = 8 * (g >> 2) + (g & 3);
        const sycl::int4 ql4 = load16_a2(b->ql + 16 * g), qh4 = load16_a2(b->qh + 16 * qh4_idx);
        const float d = (float) b->d;
        r.dsc0 = d * (float) b->scales[scale_offset]; r.dsc1 = d * (float) b->scales[scale_offset + 4];
        const int vl[4] = {ql4.x(), ql4.y(), ql4.z(), ql4.w()};
        const int vh[4] = {qh4.x() >> vh_shift, qh4.y() >> vh_shift, qh4.z() >> vh_shift, qh4.w() >> vh_shift};
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            r.vi0[p] = (int) ((uint32_t((vl[p] & 0x0f0f0f0f) | ((vh[p] << 4) & 0x30303030)) + 0x60606060u) ^ 0x80808080u);
            r.vi1[p] = (int) ((uint32_t(((vl[p] >> 4) & 0x0f0f0f0f) | (((vh[p] >> 4) << 4) & 0x30303030)) + 0x60606060u) ^ 0x80808080u);
        }
        return r;
    }
    static float dot(const W& r, const A& a) {
        int s0 = 0, s1 = 0;
        s0 = dp4a(r.vi0[0], a.u0.x(), s0); s0 = dp4a(r.vi0[1], a.u0.y(), s0);
        s0 = dp4a(r.vi0[2], a.u0.z(), s0); s0 = dp4a(r.vi0[3], a.u0.w(), s0);
        s1 = dp4a(r.vi1[0], a.u1.x(), s1); s1 = dp4a(r.vi1[1], a.u1.y(), s1);
        s1 = dp4a(r.vi1[2], a.u1.z(), s1); s1 = dp4a(r.vi1[3], a.u1.w(), s1);
        return a.ds0 * r.dsc0 * (float) s0 + a.ds1 * r.dsc1 * (float) s1;
    }
};

struct KregIQ4XS {
    using Block = IQ4XSBlock;
    static constexpr int kTableWords = 256;
    static uint32_t table_entry(int i) {
        return (uint32_t) (uint8_t) kIq4nlTable[i & 15] | ((uint32_t) (uint8_t) kIq4nlTable[i >> 4] << 16);
    }
    struct A { sycl::int4 u0, u1; float ds; };
    struct W { sycl::int4 lo, hi; float d; };
    static A act(const Q81Block* xb, int g) {
        A a; a.u0 = ld_q8_16(xb + g, 0); a.u1 = ld_q8_16(xb + g, 1); a.ds = (float) xb[g].ds[0]; return a;
    }
    static W load(const Block* b, int g, const uint32_t* tl) {
        W r;
        const sycl::int2 hdr = *reinterpret_cast<const sycl::int2*>(b);
        const sycl::int2* q2 = reinterpret_cast<const sycl::int2*>(b->qs + 16 * g);
        const sycl::int2 qa = q2[0], qb = q2[1];
        const uint32_t v[4] = {(uint32_t) qa.x(), (uint32_t) qa.y(), (uint32_t) qb.x(), (uint32_t) qb.y()};
        int lo[4], hi[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t P = tl[v[i] & 0xFF] | (tl[(v[i] >> 8) & 0xFF] << 8);
            const uint32_t Q = tl[(v[i] >> 16) & 0xFF] | (tl[v[i] >> 24] << 8);
            lo[i] = (int) ((P & 0xFFFFu) | (Q << 16));
            hi[i] = (int) ((P >> 16) | (Q & 0xFFFF0000u));
        }
        r.lo = sycl::int4(lo[0], lo[1], lo[2], lo[3]); r.hi = sycl::int4(hi[0], hi[1], hi[2], hi[3]);
        const uint32_t h0 = (uint32_t) hdr.x(), sl = (uint32_t) hdr.y();
        const int ls = (int) (((sl >> (8 * (g >> 1) + 4 * (g & 1))) & 0x0f) | (((h0 >> (16 + 2 * g)) & 0x03) << 4));
        r.d = (float) sycl::bit_cast<sycl::half>((uint16_t) (h0 & 0xFFFF)) * (float) (ls - 32);
        return r;
    }
    static float dot(const W& r, const A& v) { return r.d * v.ds * (float) dp4a4(r.hi, v.u1, dp4a4(r.lo, v.u0, 0)); }
};

template <int R, int NCOLS> constexpr int kreg_p() { return (R * NCOLS + KREG_SG - 1) / KREG_SG * KREG_SG; }

template <typename F, int NCOLS, int R>
void native_mmvq_kreg_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                             float* __restrict__ y, int n_in, int n_out, float* __restrict__ slm) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const sycl::sub_group sg = it.get_sub_group();
    constexpr int P0 = R * NCOLS, P = kreg_p<R, NCOLS>(), PL = P / KREG_SG;
    const int lid = int(it.get_local_id(0)), lane = int(sg.get_local_linear_id()), sgi = lid / KREG_SG;
    const int nsg = int(it.get_local_range(0)) / KREG_SG;
    const int bpr = n_in / QK, x_stride = n_in / Q8K;
    const bool live = (lid >> 3) < bpr;
    const int kb = live ? (lid >> 3) : 0, g = lid & 7;
    typename F::A a[NCOLS];
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        a[j] = F::act(x + std::size_t(j) * x_stride + kb * (QK / Q8K), g);
        if (!live) a[j] = typename F::A{};
    }
    const uint32_t* tbl = nullptr;
    if constexpr (F::kTableWords > 0) {
        uint32_t* tw = reinterpret_cast<uint32_t*>(slm + 2 * nsg * P);
        for (int i = lid; i < F::kTableWords; i += int(it.get_local_range(0))) tw[i] = F::table_entry(i);
        it.barrier(sycl::access::fence_space::local_space);
        tbl = tw;
    }
    const int ntiles = (n_out + R - 1) / R;
    int buf = 0;
    for (int t = int(it.get_group(0)); t < ntiles; t += int(it.get_group_range(0))) {
        const int row0 = t * R;
        const typename F::Block* wt = w + std::size_t(row0) * bpr + kb;
        const int rlast = n_out - 1 - row0;
        typename F::W wv[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if constexpr (F::kTableWords > 0) wv[r] = F::load(wt + (r < rlast ? r : rlast) * bpr, g, tbl);
            else wv[r] = F::load(wt + (r < rlast ? r : rlast) * bpr, g);
        }
        float v[P];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) v[r * NCOLS + j] = F::dot(wv[r], a[j]);
#pragma unroll
        for (int k = P0; k < P; ++k) v[k] = 0.0f;
        kreg_tsum<KREG_SG / 2, P, P>(v, sg, lane);
        float* b = slm + buf * nsg * P;
#pragma unroll
        for (int i = 0; i < PL; ++i) b[sgi * P + lane * PL + i] = v[i];
        it.barrier(sycl::access::fence_space::local_space);
        for (int k = lid; k < P0; k += int(it.get_local_range(0))) {
            float s = 0.0f;
            for (int q = 0; q < nsg; ++q) s += b[q * P + k];
            const int r = k / NCOLS, j = k % NCOLS;
            if (row0 + r < n_out) y[std::size_t(j) * n_out + row0 + r] = s;
        }
        buf ^= 1;
    }
}

template <typename F, int NCOLS, int R, int GRF>
void launch_kreg(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, sycl::queue* s) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const int L = ((n_in / QK) * 8 + KREG_SG - 1) / KREG_SG * KREG_SG;
    const int nsg = L / KREG_SG;
    constexpr int P = kreg_p<R, NCOLS>();
    const int tiles = (n_out + R - 1) / R;
    int groups = tiles;
    if constexpr (std::is_same_v<F, KregQ5K> || std::is_same_v<F, KregQ6K>) {
        if (xe2_kq(kXe2KqGrid)) groups = std::min(tiles, kreg_groups_cap(NCOLS, nsg, GRF, s));
        if (groups < tiles) xe2_kq_note(kXe2KqGrid);
    }
    s->submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>(2 * nsg * P + F::kTableWords), h);
        auto k = [=](sycl::nd_item<1>) [[sycl::reqd_sub_group_size(16)]] {
            native_mmvq_kreg_kernel<F, NCOLS, R>(w, x, y, n_in, n_out, slm.get_multi_ptr<sycl::access::decorated::no>().get());
        };
        const sycl::nd_range<1> nr(std::size_t(groups) * L, L);
        if constexpr (GRF == 256)
            h.parallel_for(nr, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, k);
        else
            h.parallel_for(nr, k);
    });
}

template <typename F, int NCOLS>
void kreg_dispatch(const void* w, const void* x, float* y, int n_in, int n_out, sycl::queue* s) {
    if constexpr (std::is_same_v<F, KregIQ4XS>) {
        if constexpr (NCOLS == 1) launch_kreg<F, 1, 8, 128>(w, x, y, n_in, n_out, s);
        else launch_kreg<F, NCOLS, 4, 128>(w, x, y, n_in, n_out, s);
    } else if constexpr (std::is_same_v<F, KregQ6K>) {
        if constexpr (NCOLS <= 3) launch_kreg<F, NCOLS, 4, 128>(w, x, y, n_in, n_out, s);
        else launch_kreg<F, NCOLS, 2, 128>(w, x, y, n_in, n_out, s);
    } else {
        if constexpr (NCOLS == 1) launch_kreg<F, 1, 4, 128>(w, x, y, n_in, n_out, s);
        else if constexpr (NCOLS == 2) {
            if (n_in <= 4096) launch_kreg<F, 2, 8, 256>(w, x, y, n_in, n_out, s);
            else launch_kreg<F, 2, 4, 128>(w, x, y, n_in, n_out, s);
        } else if constexpr (NCOLS <= 4) launch_kreg<F, NCOLS, 4, 128>(w, x, y, n_in, n_out, s);
        else if constexpr (std::is_same_v<F, KregQ5K>) {
            if (n_in <= 4096) launch_kreg<F, NCOLS, 8, 256>(w, x, y, n_in, n_out, s);
            else if (n_in <= 8192 && xe2_kq(kXe2KqTile)) {
                xe2_kq_note(kXe2KqTile);
                launch_kreg<F, NCOLS, 4, 256>(w, x, y, n_in, n_out, s);
            }
            else launch_kreg<F, NCOLS, 2, 128>(w, x, y, n_in, n_out, s);
        } else launch_kreg<F, NCOLS, 8, 256>(w, x, y, n_in, n_out, s);
    }
}

// KREG2 (IQ4_XS at 1-4 columns under STRATA_MMVQ_KREG=1, at 5-8 columns when STRATA_KREG2_R names them; STRATA_KREG2=0: off)
struct K2Raw { sycl::int2 hdr, qa, qb; };
inline uint32_t k2_code(int n) {
    const uint32_t w0 = 0xBFAD9881u, w1 = 0xF6EADDCFu, w2 = 0x26190D01u, w3 = 0x71594535u;
    const uint32_t wd = n < 8 ? (n < 4 ? w0 : w1) : (n < 12 ? w2 : w3);
    return (wd >> (8 * (n & 3))) & 0xFFu;
}
inline K2Raw k2_raw(const IQ4XSBlock* b, int g) {
    K2Raw r;
    r.hdr = *reinterpret_cast<const sycl::int2*>(b);
    const sycl::int2* q2 = reinterpret_cast<const sycl::int2*>(b->qs + 16 * g);
    r.qa = q2[0]; r.qb = q2[1];
    return r;
}
inline KregIQ4XS::W k2_decode(const K2Raw& raw, int g, const uint32_t* tl) {
    KregIQ4XS::W r;
    const uint32_t v[4] = {(uint32_t) raw.qa.x(), (uint32_t) raw.qa.y(), (uint32_t) raw.qb.x(), (uint32_t) raw.qb.y()};
    const uint32_t* tb = tl + 256;
    int lo[4], hi[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        lo[i] = (int) (tl[v[i] & 0xFF] | tb[(v[i] >> 8) & 0xFF]);
        hi[i] = (int) (tl[(v[i] >> 16) & 0xFF] | tb[v[i] >> 24]);
    }
    r.lo = sycl::int4(lo[0], lo[1], lo[2], lo[3]); r.hi = sycl::int4(hi[0], hi[1], hi[2], hi[3]);
    const uint32_t h0 = (uint32_t) raw.hdr.x(), sl = (uint32_t) raw.hdr.y();
    const int ls = (int) (((sl >> (8 * (g >> 1) + 4 * (g & 1))) & 0x0f) | (((h0 >> (16 + 2 * g)) & 0x03) << 4));
    r.d = (float) sycl::bit_cast<sycl::half>((uint16_t) (h0 & 0xFFFF)) * (float) (ls - 32);
    return r;
}
template <int R>
inline void k2_load_tile(K2Raw (&raw)[R], const IQ4XSBlock* __restrict__ w, int t, int bpr, int kb, int g,
                         int n_out) {
    const int row0 = t * R;
    const IQ4XSBlock* wt = w + std::size_t(row0) * bpr + kb;
    const int rlast = n_out - 1 - row0;
#pragma unroll
    for (int r = 0; r < R; ++r) raw[r] = k2_raw(wt + (r < rlast ? r : rlast) * bpr, g);
}
template <int NCOLS, int R, bool PF>
void kreg2_iq4xs_kernel(const IQ4XSBlock* __restrict__ w, const Q81Block* __restrict__ x, float* __restrict__ y,
                        int n_in, int n_out, float* __restrict__ slm) {
    using F = KregIQ4XS;
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const sycl::sub_group sg = it.get_sub_group();
    constexpr int P0 = R * NCOLS, P = kreg_p<R, NCOLS>(), PL = P / KREG_SG;
    const int lid = int(it.get_local_id(0)), lane = int(sg.get_local_linear_id()), sgi = lid / KREG_SG;
    const int L = int(it.get_local_range(0)), nsg = L / KREG_SG;
    const int bpr = n_in / QK, x_stride = n_in / Q8K;
    const bool live = (lid >> 3) < bpr;
    const int kb = live ? (lid >> 3) : 0, g = lid & 7;
    const int ntiles = (n_out + R - 1) / R, G = int(it.get_group_range(0));
    int t = int(it.get_group(0));
    K2Raw cur[R];
    k2_load_tile<R>(cur, w, t, bpr, kb, g, n_out);
    typename F::A a[NCOLS];
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        a[j] = F::act(x + std::size_t(j) * x_stride + kb * (QK / Q8K), g);
        if (!live) a[j] = typename F::A{};
        const sycl::int4 u0 = a[j].u0, u1 = a[j].u1;
        auto pl = [](int l, int h) { return (int) (((uint32_t) l & 0xFFFFu) | ((uint32_t) h << 16)); };
        auto ph = [](int l, int h) { return (int) (((uint32_t) l >> 16) | ((uint32_t) h & 0xFFFF0000u)); };
        a[j].u0 = sycl::int4(pl(u0.x(), u1.x()), pl(u0.y(), u1.y()), pl(u0.z(), u1.z()), pl(u0.w(), u1.w()));
        a[j].u1 = sycl::int4(ph(u0.x(), u1.x()), ph(u0.y(), u1.y()), ph(u0.z(), u1.z()), ph(u0.w(), u1.w()));
    }
    uint32_t* tw = reinterpret_cast<uint32_t*>(slm + 2 * nsg * P);
    for (int i = lid; i < 256; i += L) {
        const uint32_t te = k2_code(i & 15) | (k2_code(i >> 4) << 16);
        tw[i] = te;
        tw[256 + i] = te << 8;
    }
    it.barrier(sycl::access::fence_space::local_space);
    int buf = 0;
    for (; t < ntiles; t += G) {
        const int tn = t + G;
        K2Raw nxt[R];
        if constexpr (PF) k2_load_tile<R>(nxt, w, tn < ntiles ? tn : t, bpr, kb, g, n_out);
        const int row0 = t * R;
        float v[P];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const typename F::W wv = k2_decode(cur[r], g, tw);
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) v[r * NCOLS + j] = F::dot(wv, a[j]);
        }
#pragma unroll
        for (int k = P0; k < P; ++k) v[k] = 0.0f;
        kreg_tsum<KREG_SG / 2, P, P>(v, sg, lane);
        float* b = slm + buf * nsg * P;
#pragma unroll
        for (int i = 0; i < PL; ++i) b[sgi * P + lane * PL + i] = v[i];
        it.barrier(sycl::access::fence_space::local_space);
        for (int k = lid; k < P0; k += L) {
            float s = 0.0f;
            for (int q = 0; q < nsg; ++q) s += b[q * P + k];
            const int r = k / NCOLS, j = k % NCOLS;
            if (row0 + r < n_out) y[std::size_t(j) * n_out + row0 + r] = s;
        }
        buf ^= 1;
        if constexpr (PF) {
#pragma unroll
            for (int r = 0; r < R; ++r) cur[r] = nxt[r];
        } else if (tn < ntiles) {
            k2_load_tile<R>(cur, w, tn, bpr, kb, g, n_out);
        }
    }
}
template <int NCOLS, int R>
void launch_kreg2(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int G, sycl::queue* s) {
    constexpr bool PF = NCOLS > 1;
    const auto* w = static_cast<const IQ4XSBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const int L = ((n_in / QK) * 8 + KREG_SG - 1) / KREG_SG * KREG_SG;
    const int nsg = L / KREG_SG;
    constexpr int P = kreg_p<R, NCOLS>();
    s->submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>(2 * nsg * P + 2 * 256), h);
        auto k = [=](sycl::nd_item<1>) [[sycl::reqd_sub_group_size(16)]] {
            kreg2_iq4xs_kernel<NCOLS, R, PF>(w, x, y, n_in, n_out, slm.get_multi_ptr<sycl::access::decorated::no>().get());
        };
        h.parallel_for(sycl::nd_range<1>(std::size_t(G) * L, L), k);
    });
}
struct Kreg2Cfg {
    bool on = true;
    int r[9] = {0, 2, 4, 4, 4, 0, 0, 0, 0};
};
static char g_kreg_r_override[64] = "";
inline Kreg2Cfg kreg2_cfg() {
    Kreg2Cfg k;
    const char* e = std::getenv("STRATA_KREG2");
    k.on = !(e && e[0] == '0');
    const char* rl = g_kreg_r_override[0] ? g_kreg_r_override : std::getenv("STRATA_KREG2_R");
    if (rl) {
        for (int i = 1; i <= 8 && *rl; ++i) {
            const int v = std::atoi(rl);
            if (v == 2 || v == 4 || (v == 0 && i > 4 && (*rl == '0'))) k.r[i] = v;
            while (*rl && *rl != ',') ++rl;
            if (*rl == ',') ++rl;
        }
    }
    return k;
}
template <int NC>
bool k2_go(const Kreg2Cfg& c, const void* w, const void* x, float* y, int n_in, int n_out, sycl::queue* s) {
    static const int cores = std::max(1, int(s->get_device().get_info<sycl::info::device::max_compute_units>()) / 8);
    const int nsg = ((n_in / QK) * 8 + KREG_SG - 1) / KREG_SG;
    const int grid = std::max(1, 60 / nsg) * cores;
    const int R = c.r[NC];
    if (R == 0) return false;
    const int tiles = (n_out + R - 1) / R;
    if (NC == 1 ? tiles > grid : tiles <= grid) return false;
    const int G = tiles < grid ? tiles : grid;
    if (R == 2) launch_kreg2<NC, 2>(w, x, y, n_in, n_out, G, s);
    else launch_kreg2<NC, 4>(w, x, y, n_in, n_out, G, s);
    return true;
}
inline bool try_kreg2(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, sycl::queue* s) {
    const Kreg2Cfg c = kreg2_cfg();
    if (!c.on || g_kreg_override == 3) return false;
    switch (ncols) {
        case 1: return k2_go<1>(c, w, x, y, n_in, n_out, s);
        case 2: return k2_go<2>(c, w, x, y, n_in, n_out, s);
        case 3: return k2_go<3>(c, w, x, y, n_in, n_out, s);
        case 4: return k2_go<4>(c, w, x, y, n_in, n_out, s);
        case 5: return k2_go<5>(c, w, x, y, n_in, n_out, s);
        case 6: return k2_go<6>(c, w, x, y, n_in, n_out, s);
        case 7: return k2_go<7>(c, w, x, y, n_in, n_out, s);
        case 8: return k2_go<8>(c, w, x, y, n_in, n_out, s);
        default: return false;
    }
}
template <typename F>
bool try_kreg(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (!kreg_on()) return false;
    if (ncols < 1 || ncols > 8 || n_out <= 0 || n_in <= 0 || n_in % QK != 0 || n_in > 32768) return false;
    if (!weights || !x_q8_1 || !y) return false;
    const uintptr_t wa = reinterpret_cast<uintptr_t>(weights);
    if ((std::is_same_v<F, KregQ4K> || std::is_same_v<F, KregQ5K>) && (wa & 15u) != 0) return false;
    if (std::is_same_v<F, KregIQ4XS> && (wa & 7u) != 0) return false;
    if ((wa & 1u) != 0 || (reinterpret_cast<uintptr_t>(x_q8_1) & 3u) != 0) return false;
    sycl::queue* s = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    if constexpr (std::is_same_v<F, KregIQ4XS>)
        if (try_kreg2(weights, x_q8_1, y, n_in, n_out, ncols, s)) return true;
    switch (ncols) {
        case 1: kreg_dispatch<F, 1>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 2: kreg_dispatch<F, 2>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 3: kreg_dispatch<F, 3>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 4: kreg_dispatch<F, 4>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 5: kreg_dispatch<F, 5>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 6: kreg_dispatch<F, 6>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 7: kreg_dispatch<F, 7>(weights, x_q8_1, y, n_in, n_out, s); return true;
        default: kreg_dispatch<F, 8>(weights, x_q8_1, y, n_in, n_out, s); return true;
    }
}

// Wide32 Q8_0 for pair launches
struct Wide32Q8 {
    using Block = Q80Block;
    static constexpr int LPB = 2;
    struct W { sycl::int4 q; float d; int half; };
    static W load(const Block* b, int l) {
        W r; r.half = l; r.q = load16_a2(b->qs + 16 * l); r.d = (float) b->d; return r;
    }
    static float apply(const W& r, const Q81Block* xb) {
        return r.d * (float) xb->ds[0] * (float) dp4a4(r.q, ld_q8_16(xb, r.half), 0);
    }
};

inline uint32_t iq4_lut4(uint32_t q4) {
    constexpr uint32_t t0 = 0xBFAD9881u, t1 = 0xF6EADDCFu, t2 = 0x26190D01u, t3 = 0x71594535u;
    uint32_t r = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t n = (q4 >> (8 * i)) & 0xF;
        const uint32_t t = n < 4 ? t0 : n < 8 ? t1 : n < 12 ? t2 : t3;
        r |= ((t >> (8 * (n & 3))) & 0xFF) << (8 * i);
    }
    return r;
}

struct Wide32IQ4NL {
    using Block = IQ4NLBlock;
    static constexpr int LPB = 1;
    struct W { sycl::int4 lo, hi; float d; };
    static W load(const Block* b, int) {
        W r;
        const sycl::int4 q = load16_a2(b->qs);
        const uint32_t v[4] = {(uint32_t) q.x(), (uint32_t) q.y(), (uint32_t) q.z(), (uint32_t) q.w()};
        int lo[4], hi[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { lo[i] = (int) iq4_lut4(v[i] & 0x0f0f0f0f); hi[i] = (int) iq4_lut4((v[i] >> 4) & 0x0f0f0f0f); }
        r.lo = sycl::int4(lo[0], lo[1], lo[2], lo[3]); r.hi = sycl::int4(hi[0], hi[1], hi[2], hi[3]);
        r.d = (float) b->d;
        return r;
    }
    static float apply(const W& r, const Q81Block* xb) {
        return r.d * (float) xb->ds[0] * (float) dp4a4(r.hi, ld_q8_16(xb, 1), dp4a4(r.lo, ld_q8_16(xb, 0), 0));
    }
};

template <typename F, int NCOLS>
void native_mmvq_wide32_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                               float* __restrict__ y, int n_in, int n_out) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = int(item.get_local_id(2)), warp = int(item.get_local_id(1));
    const int row = int(item.get_group(2)) * WARPS + warp;
    if (row >= n_out) return;
    constexpr int BPS = WARP / F::LPB;
    const int blocks_per_row = n_in / 32;
    const int l = lane % F::LPB, sub = lane / F::LPB;
    const typename F::Block* wr = w + std::size_t(row) * blocks_per_row;
    float acc[NCOLS] = {};
    for (int kb = sub; kb < blocks_per_row; kb += BPS) {
        const typename F::W wv = F::load(wr + kb, l);
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) acc[j] += F::apply(wv, x + std::size_t(j) * blocks_per_row + kb);
    }
    auto sg = item.get_sub_group();
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        float v = acc[j];
#pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
        if (lane == 0) y[std::size_t(j) * n_out + row] = v;
    }
}

template <typename F, int NCOLS>
void launch_wide32(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, sycl::queue* s) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
    s->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, WARPS, WARP), sycl::range<3>(1, WARPS, WARP)),
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
            native_mmvq_wide32_kernel<F, NCOLS>(w, x, y, n_in, n_out);
        });
}

inline bool wide_32() {
    static const bool v = std::getenv("STRATA_MMVQ_WIDE_32") == nullptr || std::atoi(std::getenv("STRATA_MMVQ_WIDE_32")) != 0;
    return v;
}

template <typename F>
bool try_wide32(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (!wide_32() || ncols < 1 || ncols > 8 || n_in % 32 != 0) return false;
    sycl::queue* s = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    switch (ncols) {
        case 1: launch_wide32<F, 1>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 2: launch_wide32<F, 2>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 3: launch_wide32<F, 3>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 4: launch_wide32<F, 4>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 5: launch_wide32<F, 5>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 6: launch_wide32<F, 6>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 7: launch_wide32<F, 7>(weights, x_q8_1, y, n_in, n_out, s); return true;
        default: launch_wide32<F, 8>(weights, x_q8_1, y, n_in, n_out, s); return true;
    }
}

template <typename F, int NCOLS>
void native_mmvq_wide32_pair_kernel(const typename F::Block* w1, const typename F::Block* w2,
                                    const Q81Block* __restrict__ x, float* y1, float* y2, int n_in, int n_out) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = int(item.get_local_id(2)), warp = int(item.get_local_id(1));
    int row = int(item.get_group(2)) * WARPS + warp;
    if (row >= 2 * n_out) return;
    const typename F::Block* w = w1;
    float* y = y1;
    if (row >= n_out) { row -= n_out; w = w2; y = y2; }
    constexpr int BPS = WARP / F::LPB;
    const int blocks_per_row = n_in / 32;
    const int l = lane % F::LPB, sub = lane / F::LPB;
    const typename F::Block* wr = w + std::size_t(row) * blocks_per_row;
    float acc[NCOLS] = {};
    for (int kb = sub; kb < blocks_per_row; kb += BPS) {
        const typename F::W wv = F::load(wr + kb, l);
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) acc[j] += F::apply(wv, x + std::size_t(j) * blocks_per_row + kb);
    }
    auto sg = item.get_sub_group();
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        float v = acc[j];
#pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
        if (lane == 0) y[std::size_t(j) * n_out + row] = v;
    }
}

template <typename F, int NCOLS>
void launch_wide32_pair(const void* weights1, const void* weights2, const void* x_q8_1, float* y1, float* y2, int n_in,
                        int n_out, sycl::queue* s) {
    const auto* w1 = static_cast<const typename F::Block*>(weights1);
    const auto* w2 = static_cast<const typename F::Block*>(weights2);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const unsigned blocks = unsigned((std::size_t(2) * std::size_t(n_out) + WARPS - 1) / WARPS);
    s->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, WARPS, WARP), sycl::range<3>(1, WARPS, WARP)),
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
            native_mmvq_wide32_pair_kernel<F, NCOLS>(w1, w2, x, y1, y2, n_in, n_out);
        });
}

// ============================ STRATA_Q6_PLANAR: the Q6_K heads as 16-byte-aligned planes (SYCL port) ============================
struct Q6Planar {
    void* base;
    const uint8_t* ql;
    const uint8_t* qh;
    const uint8_t* sc;
    const sycl::half* d;
    int n_in, n_out;
};
std::unordered_map<const void*, Q6Planar>& q6_planar_registry() {
    static std::unordered_map<const void*, Q6Planar> registry;
    return registry;
}

template <int NCOLS>
void native_mmvq_q6k_planar_kernel(const uint8_t* __restrict__ qlp, const uint8_t* __restrict__ qhp,
                                   const uint8_t* __restrict__ scp, const sycl::half* __restrict__ dp,
                                   const Q81Block* __restrict__ x, float* __restrict__ y, int n_in, int n_out) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = int(item.get_local_id(2)), warp = int(item.get_local_id(1));
    const int row = int(item.get_group(2)) * WARPS + warp;
    if (row >= n_out) return;
    const int blocks_per_row = n_in / QK, x_stride = n_in / Q8K;
    const int g = lane & 7, sub = lane >> 3;
    const int vh_shift = 2 * ((g & 3) >> 1);
    const int qh4_idx = 2 * (g >> 2) + (g & 1);
    const int bq8_offset = 4 * (g >> 2) + ((g & 3) >> 1);
    const int u_int4 = g & 1;
    float acc[NCOLS] = {};
    for (int kbx = sub; kbx < blocks_per_row; kbx += WARP / 8) {
        const int kby = kbx * (QK / Q8K);
        const std::size_t bi = std::size_t(row) * blocks_per_row + kbx;
        const sycl::int4 ql4 = reinterpret_cast<const sycl::int4*>(qlp + bi * 128)[g];
        const sycl::int4 qh4 = reinterpret_cast<const sycl::int4*>(qhp + bi * 64)[qh4_idx];
        const uint16_t s2 = reinterpret_cast<const uint16_t*>(scp + bi * 16)[g];
        const float d = dp[bi];
        const float dsc0 = d * (float)(int8_t)(s2 & 0xff), dsc1 = d * (float)(int8_t)(s2 >> 8);
        sycl::int4 u0[NCOLS], u1[NCOLS];
        float ds0[NCOLS], ds1[NCOLS];
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            const Q81Block* xb = x + std::size_t(j) * x_stride + kby + bq8_offset;
            u0[j] = ld_i4_a4(reinterpret_cast<const int*>(xb[0].qs) + 4 * u_int4);
            u1[j] = ld_i4_a4(reinterpret_cast<const int*>(xb[2].qs) + 4 * u_int4);
            ds0[j] = xb[0].ds[0];
            ds1[j] = xb[2].ds[0];
        }
        const int vl[4] = {ql4.x(), ql4.y(), ql4.z(), ql4.w()};
        const int vh[4] = {qh4.x() >> vh_shift, qh4.y() >> vh_shift, qh4.z() >> vh_shift, qh4.w() >> vh_shift};
        int vi0[4], vi1[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            vi0[p] = q6_sub32((vl[p] & 0x0f0f0f0f) | ((vh[p] << 4) & 0x30303030));
            vi1[p] = q6_sub32(((vl[p] >> 4) & 0x0f0f0f0f) | (((vh[p] >> 4) << 4) & 0x30303030));
        }
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            int s0 = 0, s1 = 0;
            s0 = dp4a(vi0[0], u0[j].x(), s0);
            s0 = dp4a(vi0[1], u0[j].y(), s0);
            s0 = dp4a(vi0[2], u0[j].z(), s0);
            s0 = dp4a(vi0[3], u0[j].w(), s0);
            s1 = dp4a(vi1[0], u1[j].x(), s1);
            s1 = dp4a(vi1[1], u1[j].y(), s1);
            s1 = dp4a(vi1[2], u1[j].z(), s1);
            s1 = dp4a(vi1[3], u1[j].w(), s1);
            acc[j] += ds0[j] * dsc0 * (float)s0 + ds1[j] * dsc1 * (float)s1;
        }
    }
    auto sg = item.get_sub_group();
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        float v = acc[j];
#pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
        if (lane == 0) y[std::size_t(j) * n_out + row] = v;
    }
}

template <int NCOLS>
struct Q6kPlanarGrf256Functor {
    const uint8_t* ql;
    const uint8_t* qh;
    const uint8_t* sc;
    const sycl::half* d;
    const Q81Block* x;
    float* y;
    int n_in, n_out;
    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        return sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>,
                                                           sycl::ext::oneapi::experimental::sub_group_size<32>};
    }
    void operator()(sycl::nd_item<3>) const {
        native_mmvq_q6k_planar_kernel<NCOLS>(ql, qh, sc, d, x, y, n_in, n_out);
    }
};

template <int NCOLS>
void q6_planar_launch(const Q6Planar& w, const Q81Block* x, float* y, sycl::queue* s) {
    const unsigned blocks = unsigned((std::size_t(w.n_out) + WARPS - 1) / WARPS);
    const uint8_t *ql = w.ql, *qh = w.qh, *sc = w.sc;
    const sycl::half* d = w.d;
    const int n_in = w.n_in, n_out = w.n_out;
    const bool use_256 = (NCOLS > 1) ? (mmvq_large_grf(s) || (NCOLS >= 6 && xe2_kq(kXe2KqHead))) : mmvq_decode_large_grf(s);
    if (use_256) {
        if (NCOLS >= 6 && xe2_kq(kXe2KqHead)) xe2_kq_note(kXe2KqHead);
        s->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, WARPS, WARP),
                                          sycl::range<3>(1, WARPS, WARP)),
                        Q6kPlanarGrf256Functor<NCOLS>{ql, qh, sc, d, x, y, n_in, n_out});
        return;
    }
    s->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, WARPS, WARP),
                                      sycl::range<3>(1, WARPS, WARP)),
                    [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
                        native_mmvq_q6k_planar_kernel<NCOLS>(ql, qh, sc, d, x, y, n_in, n_out);
                    });
}

bool q6_planar_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    auto& registry = q6_planar_registry();
    if (registry.empty() || ncols < 1) return false;
    const auto found = registry.find(weights);
    if (found == registry.end() || found->second.n_in != n_in || found->second.n_out != n_out) return false;
    validate_shape(n_in, std::min(ncols, 8), 256);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    sycl::queue* s = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    const std::size_t x_col = std::size_t(n_in / Q8K);
    for (int c0 = 0; c0 < ncols; c0 += 8) {
        const int nc = std::min(8, ncols - c0);
        const auto* x = static_cast<const Q81Block*>(x_q8_1) + std::size_t(c0) * x_col;
        float* yy = y + std::size_t(c0) * n_out;
        switch (nc) {
            case 1:
                q6_planar_launch<1>(found->second, x, yy, s);
                break;
            case 2:
                q6_planar_launch<2>(found->second, x, yy, s);
                break;
            case 3:
                q6_planar_launch<3>(found->second, x, yy, s);
                break;
            case 4:
                q6_planar_launch<4>(found->second, x, yy, s);
                break;
            case 5:
                q6_planar_launch<5>(found->second, x, yy, s);
                break;
            case 6:
                q6_planar_launch<6>(found->second, x, yy, s);
                break;
            case 7:
                q6_planar_launch<7>(found->second, x, yy, s);
                break;
            default:
                q6_planar_launch<8>(found->second, x, yy, s);
                break;
        }
    }
    launch_check();
    return true;
}

struct Q6PlanarLayout {
    std::size_t o_qh, o_sc, o_d, total;
};
Q6PlanarLayout q6_planar_layout(std::size_t nb) {
    Q6PlanarLayout l;
    l.o_qh = (nb * 128 + 255) & ~std::size_t(255);
    l.o_sc = (l.o_qh + nb * 64 + 255) & ~std::size_t(255);
    l.o_d = (l.o_sc + nb * 16 + 255) & ~std::size_t(255);
    l.total = l.o_d + nb * 2;
    return l;
}
Q6Planar q6_planar_make(void* dev, std::size_t nb, int n_in, int n_out) {
    const Q6PlanarLayout l = q6_planar_layout(nb);
    auto* base = static_cast<uint8_t*>(dev);
    return Q6Planar{dev, base, base + l.o_qh, base + l.o_sc, reinterpret_cast<const sycl::half*>(base + l.o_d), n_in,
                    n_out};
}

}  // namespace

bool native_q8_0_packed_enabled() { return false; }
bool native_q8_0_packed_eligible(int n_in, int n_out) {
    return n_in > 0 && n_in % 32 == 0 && n_in / 32 >= 2 * WARPS * WARP / 8 && n_out >= 2048;
}
void native_q8_0_pack_host(const void* gguf_blocks, void* out, int n_in, int n_out) {
    const auto* blocks = static_cast<const Q80Block*>(gguf_blocks);
    auto* qs = static_cast<int8_t*>(out);
    auto* d = reinterpret_cast<sycl::half*>(qs + std::size_t(n_in) * n_out);
    const std::size_t count = std::size_t(n_in / 32) * n_out;
    for (std::size_t b = 0; b < count; ++b) {
        std::memcpy(qs + b * 32, blocks[b].qs, 32);
        std::memcpy(d + b, &blocks[b].d, sizeof(sycl::half));
    }
}
void native_q8_0_packed_register(const void*, const void*, int, int) {}
void native_q8_0_packed_unregister(const void*) {}

bool native_q6_k_packed_enabled() { return false; }
bool native_q6_k_pack(const void*, int, int, const char*) { return false; }

bool native_q6_k_planar_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_Q6_PLANAR");
        return !(v && v[0] == '0' && v[1] == '\0');
    }();
    return enabled;
}

bool native_q6_k_is_planar(const void* weights) { return q6_planar_registry().count(weights) != 0; }

void* native_q6_k_planar_adopt(void* weights, int n_in, int n_out, const char* what) try {
    if (!weights || n_in <= 0 || n_in % 256 || n_out <= 0 || !native_q6_k_planar_enabled()) return weights;
    if (q6_planar_registry().count(weights)) return weights;
    auto& q = default_queue();
    const std::size_t nb = std::size_t(n_in / 256) * std::size_t(n_out);
    const Q6PlanarLayout l = q6_planar_layout(nb);
    void* dev = nullptr;
    try {
        dev = sycl::malloc_device(l.total, q);
    } catch (...) {
        dev = nullptr;
    }
    if (dev == nullptr) {
        std::fprintf(
            stderr,
            "strata: STRATA_Q6_PLANAR: no VRAM for the %s's planar copy (+%.1f MiB), keeping the GGUF layout\n", what,
            l.total / 1048576.0);
        return weights;
    }
    const uint8_t* src = static_cast<const uint8_t*>(weights);
    uint8_t* base = static_cast<uint8_t*>(dev);
    const std::size_t o_qh = l.o_qh, o_sc = l.o_sc, o_d = l.o_d;
    q.parallel_for(sycl::range<1>(nb), [=](sycl::id<1> id) {
        const std::size_t b = id[0];
        const uint16_t* s16 = reinterpret_cast<const uint16_t*>(src + b * 210);
        uint16_t* qlo = reinterpret_cast<uint16_t*>(base + b * 128);
        for (int i = 0; i < 64; ++i) qlo[i] = s16[i];
        uint16_t* qho = reinterpret_cast<uint16_t*>(base + o_qh + b * 64);
        for (int i = 0; i < 32; ++i) qho[i] = s16[64 + i];
        const int8_t* sc = reinterpret_cast<const int8_t*>(src + b * 210 + 192);
        uint8_t* sco = base + o_sc + b * 16;
        for (int g = 0; g < 8; ++g) {
            const int so = 8 * (g >> 2) + (g & 3);
            sco[2 * g] = (uint8_t)sc[so];
            sco[2 * g + 1] = (uint8_t)sc[so + 4];
        }
        reinterpret_cast<uint16_t*>(base + o_d)[b] = s16[104];
    });
    q.wait();
    sycl::free(weights, q);
    q6_planar_registry()[dev] = q6_planar_make(dev, nb, n_in, n_out);
    std::fprintf(stderr, "strata: STRATA_Q6_PLANAR: the %s (%d x %d) is stored as 16-byte-aligned planes (same bytes)\n",
                 what, n_out, n_in);
    return dev;
} catch (sycl::exception const& exc) {
    std::fprintf(stderr, "strata: STRATA_Q6_PLANAR conversion failed: %s\n", exc.what());
    return weights;
}

void* native_q6_k_planar_gather(const void* planar, int n_in, const int32_t* rows_dev, int64_t n_sel,
                                const char* what) try {
    const auto found = q6_planar_registry().find(planar);
    if (found == q6_planar_registry().end() || found->second.n_in != n_in || n_sel <= 0) return nullptr;
    auto& q = default_queue();
    const int nbr = n_in / 256;
    const std::size_t nb = std::size_t(nbr) * std::size_t(n_sel);
    const Q6PlanarLayout l = q6_planar_layout(nb);
    void* dev = nullptr;
    try {
        dev = sycl::malloc_device(l.total, q);
    } catch (...) {
        dev = nullptr;
    }
    if (dev == nullptr) return nullptr;
    const Q6Planar src = found->second;
    uint8_t* base = static_cast<uint8_t*>(dev);
    const std::size_t o_qh = l.o_qh, o_sc = l.o_sc, o_d = l.o_d;
    q.parallel_for(sycl::range<1>(nb), [=](sycl::id<1> id) {
        const std::size_t b = id[0];
        const std::size_t i = b / nbr, j = b - i * nbr;
        const std::size_t sb = std::size_t(rows_dev[i]) * nbr + j;
        const sycl::int4* sq = reinterpret_cast<const sycl::int4*>(src.ql + sb * 128);
        sycl::int4* dq = reinterpret_cast<sycl::int4*>(base + b * 128);
        for (int k = 0; k < 8; ++k) dq[k] = sq[k];
        const sycl::int4* sh = reinterpret_cast<const sycl::int4*>(src.qh + sb * 64);
        sycl::int4* dh = reinterpret_cast<sycl::int4*>(base + o_qh + b * 64);
        for (int k = 0; k < 4; ++k) dh[k] = sh[k];
        *reinterpret_cast<sycl::int4*>(base + o_sc + b * 16) = *reinterpret_cast<const sycl::int4*>(src.sc + sb * 16);
        reinterpret_cast<uint16_t*>(base + o_d)[b] = reinterpret_cast<const uint16_t*>(src.d)[sb];
    });
    q.wait();
    q6_planar_registry()[dev] = q6_planar_make(dev, nb, n_in, (int)n_sel);
    std::fprintf(stderr, "strata: STRATA_Q6_PLANAR: the %s (%lld x %d) is gathered from the planar head\n", what,
                 (long long)n_sel, n_in);
    return dev;
} catch (sycl::exception const& exc) {
    std::fprintf(stderr, "strata: STRATA_Q6_PLANAR gather failed: %s\n", exc.what());
    return nullptr;
}

void native_q6_k_unpack(const void* weights) {
    q6_planar_registry().erase(weights);
}

void native_mmvq_set_multi_exact(bool exact) { g_multi_exact = exact; }
bool native_mmvq_multi_exact() { return g_multi_exact; }
void native_mmvq_set_b6_rows(int mode) { g_b6_rows = mode; }
int native_mmvq_b6_rows() { return g_b6_rows; }
void native_mmvq_set_q6k_wide(bool on) { g_q6k_wide = on; }
void native_mmvq_set_kreg(int on) { g_kreg_override = on; }
void native_mmvq_set_kreg_r(const char* s) {
    if (s) {
        std::strncpy(g_kreg_r_override, s, sizeof(g_kreg_r_override) - 1);
        g_kreg_r_override[sizeof(g_kreg_r_override) - 1] = '\0';
    } else {
        g_kreg_r_override[0] = '\0';
    }
}
void native_mmvq_set_xe2_kq(int parts) { g_xe2_kq_override = parts < 0 ? -1 : (parts == 1 ? kXe2KqDefault : (parts & kXe2KqAll)); }
int native_mmvq_xe2_kq_seen() { return g_xe2_kq_seen.exchange(0, std::memory_order_relaxed); }
void native_mmvq_set_decode_256grf(bool on) { g_mmvq_decode_256grf = on; }
bool native_mmvq_decode_256grf() { return g_mmvq_decode_256grf; }
void native_q6_k_mmvq_stride224(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                                void* stream) {
    sycl::queue* s = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    switch (ncols) {
        case 1:
            launch_q6k_wide224<1>(weights, x_q8_1, y, n_in, n_out, s);
            return;
        case 2:
            launch_q6k_wide224<2>(weights, x_q8_1, y, n_in, n_out, s);
            return;
        case 3:
            launch_q6k_wide224<3>(weights, x_q8_1, y, n_in, n_out, s);
            return;
        case 4:
            launch_q6k_wide224<4>(weights, x_q8_1, y, n_in, n_out, s);
            return;
        default:
            launch_q6k_wide224<6>(weights, x_q8_1, y, n_in, n_out, s);
            return;
    }
}

std::size_t native_q8_1_bytes(int n_in, int ncols) {
    validate_shape(n_in, ncols);
    return std::size_t(ncols) * std::size_t(n_in / Q8K) * sizeof(Q81Block);
}

void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    validate_pointer(x);
    validate_pointer(x_q8_1);
    validate_stream(stream);
    // Columns are contiguous and n_in is a multiple of 32, so ncols columns quantize as one vector of
    // ncols * n_in elements: every 32-element block stays inside one column.
    const int n_total = n_in * ncols;
    const unsigned blocks = unsigned((std::size_t(n_total) + QUANT_THREADS - 1) / QUANT_THREADS);
    (stream ? static_cast<sycl::queue*>(stream) : &default_queue())
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, QUANT_THREADS),
                                         sycl::range<3>(1, 1, QUANT_THREADS)),
                       [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                           native_quantize_q8_1_kernel(x, static_cast<Q81Block*>(x_q8_1), n_total);
                       });
    launch_check();
}

void native_swiglu_quantize_q8_1(const float* gate, const float* up, void* x_q8_1, int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    validate_pointer(gate);
    validate_pointer(up);
    validate_pointer(x_q8_1);
    validate_stream(stream);
    const int n_total = n_in * ncols;
    const unsigned blocks = unsigned((std::size_t(n_total) + QUANT_THREADS - 1) / QUANT_THREADS);
    (stream ? static_cast<sycl::queue*>(stream) : &default_queue())
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, QUANT_THREADS),
                                         sycl::range<3>(1, 1, QUANT_THREADS)),
                       [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                           native_swiglu_quantize_q8_1_kernel(gate, up, static_cast<Q81Block*>(x_q8_1), n_total);
                       });
    launch_check();
}

void native_q5_k_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, QK);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (try_kreg<KregQ5K>(weights, x_q8_1, y, n_in, n_out, ncols, stream)) { launch_check(); return; }
    if (ncols > 1) {
        launch_multi<Q5KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q5KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    sycl::queue* s = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    const sycl::range<3> threads(1, WARPS, WARP);
    if (n_in / QK < VDR * WARPS * WARP / QI) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        /*
        DPCT1049:57: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            s->submit([&](sycl::handler& cgh) {
                /*
                DPCT1101:842: 'WARPS - 1' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                /*
                DPCT1101:843: 'WARP' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<float[3 /*WARPS - 1*/][4][32 /*WARP*/], 0> partial_acc_ct1(cgh);

                cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * threads, threads),
                                 [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                                     native_q5_k_mmvq_kernel<true>(w, x, y, n_in, n_out, partial_acc_ct1);
                                 });
            });
        }
    } else {
        /*
        DPCT1049:58: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        s->submit([&](sycl::handler& cgh) {
            /*
            DPCT1101:844: 'WARPS - 1' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            /*
            DPCT1101:845: 'WARP' expression was replaced with a value. Modify
            the code to use the original expression, provided in comments, if
            it is correct.
            */
            sycl::local_accessor<float[3 /*WARPS - 1*/][1][32 /*WARP*/], 0> partial_acc_ct1(cgh);

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, unsigned(n_out)) * threads, threads),
                             [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                                 native_q5_k_mmvq_kernel<false>(w, x, y, n_in, n_out, partial_acc_ct1);
                             });
        });
    }
    launch_check();
}

void native_q5_k_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out, int ncols,
                     void* stream) {
    // Validate all outputs before enqueueing the first operation.
    validate_shape(n_in, ncols, QK);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q5_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q2_0_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 64);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<Q20Traits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q20Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    sycl::queue* s = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    const sycl::range<3> threads(1, WARPS, WARP);
    if (n_in / 64 < WARPS * WARP / 2) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        /*
        DPCT1049:59: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        s->submit([&](sycl::handler& cgh) {
            /*
            DPCT1101:846: 'WARPS - 1' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            /*
            DPCT1101:847: 'WARP' expression was replaced with a value. Modify
            the code to use the original expression, provided in comments, if it
            is correct.
            */
            sycl::local_accessor<float[3 /*WARPS - 1*/][4][32 /*WARP*/], 0> partial_acc_ct1(cgh);

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * threads, threads),
                             [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                                 native_q2_0_mmvq_kernel<true>(w, x, y, n_in, n_out, partial_acc_ct1);
                             });
        });
    } else {
        /*
        DPCT1049:60: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        s->submit([&](sycl::handler& cgh) {
            /*
            DPCT1101:848: 'WARPS - 1' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            /*
            DPCT1101:849: 'WARP' expression was replaced with a value. Modify
            the code to use the original expression, provided in comments, if it
            is correct.
            */
            sycl::local_accessor<float[3 /*WARPS - 1*/][1][32 /*WARP*/], 0> partial_acc_ct1(cgh);

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, unsigned(n_out)) * threads, threads),
                             [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                                 native_q2_0_mmvq_kernel<false>(w, x, y, n_in, n_out, partial_acc_ct1);
                             });
        });
    }
    launch_check();
}

void native_q2_0_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out, int ncols,
                     void* stream) {
    validate_shape(n_in, ncols, 64);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q2_0_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q3_k_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<Q3KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q3KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    sycl::queue* s = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    const sycl::range<3> threads(1, WARPS, WARP);
    if (n_in / 256 < WARPS * WARP / 16) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        /*
        DPCT1049:61: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        s->submit([&](sycl::handler& cgh) {
            /*
            DPCT1101:850: 'WARPS - 1' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            /*
            DPCT1101:851: 'WARP' expression was replaced with a value. Modify
            the code to use the original expression, provided in comments, if it
            is correct.
            */
            sycl::local_accessor<float[3 /*WARPS - 1*/][4][32 /*WARP*/], 0> partial_acc_ct1(cgh);

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * threads, threads),
                             [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                                 native_q3_k_mmvq_kernel<true>(w, x, y, n_in, n_out, partial_acc_ct1);
                             });
        });
    } else {
        /*
        DPCT1049:62: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        s->submit([&](sycl::handler& cgh) {
            /*
            DPCT1101:852: 'WARPS - 1' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            /*
            DPCT1101:853: 'WARP' expression was replaced with a value. Modify
            the code to use the original expression, provided in comments, if it
            is correct.
            */
            sycl::local_accessor<float[3 /*WARPS - 1*/][1][32 /*WARP*/], 0> partial_acc_ct1(cgh);

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, unsigned(n_out)) * threads, threads),
                             [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                                 native_q3_k_mmvq_kernel<false>(w, x, y, n_in, n_out, partial_acc_ct1);
                             });
        });
    }
    launch_check();
}

void native_q3_k_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out, int ncols,
                     void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q3_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_xs_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                        void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (try_kreg<KregIQ4XS>(weights, x_q8_1, y, n_in, n_out, ncols, stream)) { launch_check(); return; }
    if (ncols > 1) {
        launch_multi<IQ4XSTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const IQ4XSBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    sycl::queue* s = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    const sycl::range<3> threads(1, WARPS, WARP);
    if (n_in / 256 < 4 * WARPS * WARP / 32) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        /*
        DPCT1049:63: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            s->submit([&](sycl::handler& cgh) {
                /*
                DPCT1101:854: 'WARPS - 1' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                /*
                DPCT1101:855: 'WARP' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<float[3 /*WARPS - 1*/][4][32 /*WARP*/], 0> partial_acc_ct1(cgh);

                cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * threads, threads),
                                 [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                                     native_iq4_xs_mmvq_kernel<true>(w, x, y, n_in, n_out, partial_acc_ct1);
                                 });
            });
        }
    } else {
        /*
        DPCT1049:64: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */

        s->submit([&](sycl::handler& cgh) {
            /*
            DPCT1101:856: 'WARPS - 1' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            /*
            DPCT1101:857: 'WARP' expression was replaced with a value. Modify
            the code to use the original expression, provided in comments, if
            it is correct.
            */
            sycl::local_accessor<float[3 /*WARPS - 1*/][1][32 /*WARP*/], 0> partial_acc_ct1(cgh);

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, unsigned(n_out)) * threads, threads),
                             [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                                 native_iq4_xs_mmvq_kernel<false>(w, x, y, n_in, n_out, partial_acc_ct1);
                             });
        });
    }
    launch_check();
}

void native_iq4_xs_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out,
                       int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_iq4_xs_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_k_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (try_kreg<KregQ4K>(weights, x_q8_1, y, n_in, n_out, ncols, stream)) { launch_check(); return; }
    if (ncols > 1) {
        launch_multi<Q4KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q4KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    sycl::queue* s = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    const sycl::range<3> threads(1, WARPS, WARP);
    if (n_in / 256 < WARPS * WARP / 16) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        /*
        DPCT1049:65: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            s->submit([&](sycl::handler& cgh) {
                /*
                DPCT1101:858: 'WARPS - 1' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                /*
                DPCT1101:859: 'WARP' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<float[3 /*WARPS - 1*/][4][32 /*WARP*/], 0> partial_acc_ct1(cgh);

                cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * threads, threads),
                                 [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                                     native_q4_k_mmvq_kernel<true>(w, x, y, n_in, n_out, partial_acc_ct1);
                                 });
            });
        }
    } else {
        /*
        DPCT1049:66: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        s->submit([&](sycl::handler& cgh) {
            /*
            DPCT1101:860: 'WARPS - 1' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            /*
            DPCT1101:861: 'WARP' expression was replaced with a value. Modify
            the code to use the original expression, provided in comments, if
            it is correct.
            */
            sycl::local_accessor<float[3 /*WARPS - 1*/][1][32 /*WARP*/], 0> partial_acc_ct1(cgh);

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, unsigned(n_out)) * threads, threads),
                             [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                                 native_q4_k_mmvq_kernel<false>(w, x, y, n_in, n_out, partial_acc_ct1);
                             });
        });
    }
    launch_check();
}

void native_q4_k_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out, int ncols,
                     void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q4_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q6_k_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (q6_planar_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream)) return;
    if (try_kreg<KregQ6K>(weights, x_q8_1, y, n_in, n_out, ncols, stream)) { launch_check(); return; }
    static const int wide_min =
        std::getenv("STRATA_MMVQ_WIDE_MIN") ? std::atoi(std::getenv("STRATA_MMVQ_WIDE_MIN")) : q6k_a2() ? 0 : 4096;
    static const int wide_maxc =
        std::getenv("STRATA_MMVQ_WIDE_MAXC") ? std::atoi(std::getenv("STRATA_MMVQ_WIDE_MAXC")) : q6k_a2() ? 8 : 4;
    if (g_q6k_wide && g_multi_exact && g_b6_rows <= 0 && ncols >= 1 && ncols <= wide_maxc && n_in > 0 &&
        n_in % 256 == 0 && n_out >= wide_min && n_out > 0) {
        validate_shape(n_in, ncols, 256);
        validate_pointer(weights);
        validate_pointer(x_q8_1);
        validate_pointer(y);
        validate_stream(stream);
        sycl::queue* s = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
        switch (ncols) {
            case 1:
                launch_q6k_wide<1>(weights, x_q8_1, y, n_in, n_out, s);
                return;
            case 2:
                launch_q6k_wide<2>(weights, x_q8_1, y, n_in, n_out, s);
                return;
            case 3:
                launch_q6k_wide<3>(weights, x_q8_1, y, n_in, n_out, s);
                return;
            case 4:
                launch_q6k_wide<4>(weights, x_q8_1, y, n_in, n_out, s);
                return;
            case 5:
                launch_q6k_wide<5>(weights, x_q8_1, y, n_in, n_out, s);
                return;
            case 6:
                launch_q6k_wide<6>(weights, x_q8_1, y, n_in, n_out, s);
                return;
            case 7:
                launch_q6k_wide<7>(weights, x_q8_1, y, n_in, n_out, s);
                return;
            default:
                launch_q6k_wide<8>(weights, x_q8_1, y, n_in, n_out, s);
                return;
        }
    }
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<Q6KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q6KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    sycl::queue* s = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    const sycl::range<3> threads(1, WARPS, WARP);
    if (n_in / 256 < WARPS * WARP / 32) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        /*
        DPCT1049:67: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        s->submit([&](sycl::handler& cgh) {
            /*
            DPCT1101:862: 'WARPS - 1' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            /*
            DPCT1101:863: 'WARP' expression was replaced with a value. Modify
            the code to use the original expression, provided in comments, if it
            is correct.
            */
            sycl::local_accessor<float[3 /*WARPS - 1*/][4][32 /*WARP*/], 0> partial_acc_ct1(cgh);

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * threads, threads),
                             [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                                 native_q6_k_mmvq_kernel<true>(w, x, y, n_in, n_out, partial_acc_ct1);
                             });
        });
    } else {
        /*
        DPCT1049:68: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        s->submit([&](sycl::handler& cgh) {
            /*
            DPCT1101:864: 'WARPS - 1' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            /*
            DPCT1101:865: 'WARP' expression was replaced with a value. Modify
            the code to use the original expression, provided in comments, if it
            is correct.
            */
            sycl::local_accessor<float[3 /*WARPS - 1*/][1][32 /*WARP*/], 0> partial_acc_ct1(cgh);

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, unsigned(n_out)) * threads, threads),
                             [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                                 native_q6_k_mmvq_kernel<false>(w, x, y, n_in, n_out, partial_acc_ct1);
                             });
        });
    }
    launch_check();
}

void native_q6_k_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out, int ncols,
                     void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q6_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_0_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<Q40Block, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_0_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out, int ncols,
                     void* stream) {
    small_f32<Q40Block, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q5_0_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<Q50Block, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q5_0_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out, int ncols,
                     void* stream) {
    small_f32<Q50Block, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

bool native_mmvq_pair(int ggml_type, const void* w1, const void* w2, const void* x_q8_1, float* y1, float* y2,
                      int n_in, int n_out, int ncols, void* stream) {
    if (ggml_type == 8 && xe2_kq(kXe2KqPair) && ncols >= 1 && ncols <= 8 && n_out > 0 && n_in > 0 &&
        n_in % 32 == 0 && w1 && w2 && x_q8_1 && y1 && y2 && y1 != y2 && stream) {
        sycl::queue* s = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
        switch (ncols) {
            case 1: launch_wide32_pair<Wide32Q8, 1>(w1, w2, x_q8_1, y1, y2, n_in, n_out, s); break;
            case 2: launch_wide32_pair<Wide32Q8, 2>(w1, w2, x_q8_1, y1, y2, n_in, n_out, s); break;
            case 3: launch_wide32_pair<Wide32Q8, 3>(w1, w2, x_q8_1, y1, y2, n_in, n_out, s); break;
            case 4: launch_wide32_pair<Wide32Q8, 4>(w1, w2, x_q8_1, y1, y2, n_in, n_out, s); break;
            case 5: launch_wide32_pair<Wide32Q8, 5>(w1, w2, x_q8_1, y1, y2, n_in, n_out, s); break;
            case 6: launch_wide32_pair<Wide32Q8, 6>(w1, w2, x_q8_1, y1, y2, n_in, n_out, s); break;
            case 7: launch_wide32_pair<Wide32Q8, 7>(w1, w2, x_q8_1, y1, y2, n_in, n_out, s); break;
            default: launch_wide32_pair<Wide32Q8, 8>(w1, w2, x_q8_1, y1, y2, n_in, n_out, s); break;
        }
        xe2_kq_note(kXe2KqPair);
        launch_check();
        return true;
    }
    return false;
}

void native_q8_0_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (try_wide32<Wide32Q8>(weights, x_q8_1, y, n_in, n_out, ncols, stream)) { launch_check(); return; }
    small_mmvq<Q80Block, 8>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q8_0_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out, int ncols,
                     void* stream) {
    small_f32<Q80Block, 8>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_nl_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                        void* stream) {
    if (try_wide32<Wide32IQ4NL>(weights, x_q8_1, y, n_in, n_out, ncols, stream)) { launch_check(); return; }
    small_mmvq<IQ4NLBlock, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_nl_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out,
                       int ncols, void* stream) {
    small_f32<IQ4NLBlock, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

bool native_mmvq_supported(int ggml_type) noexcept {
    return ggml_type == 2 || ggml_type == 6 || ggml_type == 7 || ggml_type == 8 || ggml_type == 11 || ggml_type == 12 ||
           ggml_type == 13 || ggml_type == 14 || ggml_type == 20 || ggml_type == 23 || ggml_type == 42 ||
           ggml_type == 16 || ggml_type == 17 || ggml_type == 18 || ggml_type == 21 || ggml_type == 22 ||
           ggml_type == 29;
}

std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out) {
    int block_elems, block_bytes;
    switch (ggml_type) {
        case 2:
            block_elems = 32;
            block_bytes = 18;
            break;
        case 6:
            block_elems = 32;
            block_bytes = 22;
            break;
        case 7:
            block_elems = 32;
            block_bytes = 24;
            break;
        case 8:
            block_elems = 32;
            block_bytes = 34;
            break;
        case 20:
            block_elems = 32;
            block_bytes = 18;
            break;
        case 11:
            block_elems = 256;
            block_bytes = 110;
            break;
        case 12:
            block_elems = 256;
            block_bytes = 144;
            break;
        case 13:
            block_elems = 256;
            block_bytes = 176;
            break;
        case 14:
            block_elems = 256;
            block_bytes = 210;
            break;
        case 23:
            block_elems = 256;
            block_bytes = 136;
            break;
        case 42:
            block_elems = 64;
            block_bytes = 18;
            break;
        case 16:
        case 17:
        case 18:
        case 21:
        case 22:
        case 29:
            block_elems = 256;
            block_bytes = (int)iq_row_bytes(ggml_type, 256);
            break;
        default:
            throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
    validate_shape(n_in, 1, block_elems);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    const std::size_t row_bytes = std::size_t(n_in / block_elems) * block_bytes;
    /*
    DPCT1007:424: Migration of std::numeric_limits::max is not supported.
    */
    if (row_bytes > std::numeric_limits<std::size_t>::max() / std::size_t(n_out)) {
        throw std::length_error("native MMVQ weight byte count overflows size_t");
    }
    return row_bytes * std::size_t(n_out);
}

void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                 void* stream) {
    if (ncols > MAX_NCOLS) {
        const std::size_t col_blocks = std::size_t(n_in / Q8K);
        for (int j = 0; j < ncols; j += MAX_NCOLS) {
            const int nj = (ncols - j < MAX_NCOLS) ? (ncols - j) : MAX_NCOLS;
            native_mmvq(ggml_type, weights, static_cast<const Q81Block*>(x_q8_1) + std::size_t(j) * col_blocks,
                        y + std::size_t(j) * std::size_t(n_out), n_in, n_out, nj, stream);
        }
        return;
    }
    switch (ggml_type) {
        case 2:
            native_q4_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream);
            break;
        case 6:
            native_q5_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream);
            break;
        case 7:
            iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream);
            break;
        case 8:
            native_q8_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream);
            break;
        case 20:
            native_iq4_nl_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream);
            break;
        case 11:
            native_q3_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream);
            break;
        case 12:
            native_q4_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream);
            break;
        case 13:
            native_q5_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream);
            break;
        case 14:
            native_q6_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream);
            break;
        case 23:
            native_iq4_xs_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream);
            break;
        case 42:
            native_q2_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream);
            break;
        case 16:
        case 17:
        case 18:
        case 21:
        case 22:
        case 29:
            iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream);
            break;
        default:
            throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
}

std::size_t native_q8_1_il_bytes(int n_in, int ncols) { return native_q8_1_bytes(n_in, ncols) * 2; }
void native_quantize_q8_1_il(const float*, void*, void*, int, int, void*) {
    throw std::invalid_argument("native_quantize_q8_1_il is CUDA only");
}
void native_q8_1_interleave(const void*, void*, int, int, void*) {
    throw std::invalid_argument("native_q8_1_interleave is CUDA only");
}
bool native_mmvq_il_supported(int, int, int) { return false; }
int native_mmvq_il_rows_for(int, int, int, int) { return 0; }
void native_mmvq_il(int ggml_type, const void* weights, const void* x_q8_1, const void*, float* y, int n_in, int n_out,
                    int ncols, void* stream) {
    native_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream);
}
void native_mmvq_il_tune(int) {}

}  // namespace strata::kernels
