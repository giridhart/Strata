// src/kernels/cuda/fused_gr.cu - see include/strata/kernels/fused_gr.hpp.
#include <sycl/sycl.hpp>
#include "strata/core/emulate.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace strata::kernels {
namespace {

static sycl::queue& default_queue() {
    static sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    return q;
}

constexpr int N = 2560;    // n_embd
constexpr int HC = 4;      // streams
constexpr int D = N * HC;  // 10240
constexpr int LR = 320;    // hc_lr
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int DOWN_BLOCKS = LR / WARPS;  // 40 blocks of 8 rows; one more for the inject rows
constexpr int UP_COLS = 32;              // columns d per `up` block (x 4 streams = 128 rows)
constexpr int UP_BLOCKS = N / UP_COLS;   // 80

inline float warp_sum(float v) {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    return sycl::reduce_over_group(sg, v, sycl::plus<float>());
}
inline float sigmoidf_(float x) { return 1.0f / (1.0f + sycl::native::exp(-x)); }

// 8 bf16 packed in a uint4 against 8 floats.
inline float dot8(const sycl::uint4 w, const float* x) {
    float acc = 0.0f;
    const uint32_t v[4] = {w.x(), w.y(), w.z(), w.w()};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        acc = sycl::fma(sycl::bit_cast<float>(v[j] << 16), (float)(x[2 * j]), acc);
        acc = sycl::fma(sycl::bit_cast<float>(v[j] & 0xffff0000u), (float)(x[2 * j + 1]), acc);
    }
    return acc;
}

/*
DPCT1110:14: The total declared local variable size in device function
gr_down_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
void gr_down_kernel(FusedGrArgs a, float* xn, float part[8 /*WARPS*/][4 /*HC*/], float* s_rs) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c)
        gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float)HC) : 0.0f;
    // 1. R' * w_norm into shared memory, and the per-stream sums of squares of R'.
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t * 4; i < D; i += THREADS * 4) {
        const int c = i / N, d = i - c * N;
        sycl::float4 r = *reinterpret_cast<const sycl::float4*>(a.R + i);
        if (a.apply) {
            const sycl::float4 b = *reinterpret_cast<const sycl::float4*>(a.bo_prev + d);
            r.x() = sycl::fma((float)(b.x()), gw[c], r.x());
            r.y() = sycl::fma((float)(b.y()), gw[c], r.y());
            r.z() = sycl::fma((float)(b.z()), gw[c], r.z());
            r.w() = sycl::fma((float)(b.w()), gw[c], r.w());
        }
        const sycl::float4 g = *reinterpret_cast<const sycl::float4*>(a.w_norm + i);
        float sq = r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + r.w() * r.w();
#pragma unroll
        for (int cc = 0; cc < HC; ++cc)
            if (cc == c) ss[cc] += sq;
        *reinterpret_cast<sycl::float4*>(xn + i) =
            sycl::float4(r.x() * g.x(), r.y() * g.y(), r.z() * g.z(), r.w() * g.w());
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    /*
    DPCT1065:246: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (t < HC) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w)
            s += part[w][t];
        s_rs[t] = sycl::rsqrt(s / (float)N + a.eps);
        if (item_ct1.get_group(2) == 0) a.rs[t] = s_rs[t];
    }
    /*
    DPCT1065:247: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int i = t; i < D; i += THREADS)
        xn[i] *= s_rs[i / N];
    /*
    DPCT1065:248: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // 2. one warp per output row: 10240 bf16 = 1280 chunks of 8, 40 per lane.
    const bool inject_block = item_ct1.get_group(2) == DOWN_BLOCKS;
    const int row = inject_block ? warp : item_ct1.get_group(2) * WARPS + warp;
    if (inject_block && (a.w_inject == nullptr || warp >= HC)) return;
    const uint16_t* wrow = (inject_block ? a.w_inject : a.w_down) + (size_t)row * D;
    const sycl::uint4* w4 = reinterpret_cast<const sycl::uint4*>(wrow);
    float acc = 0.0f;
#pragma unroll 4
    /*
    DPCT1098:249: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    for (int j = lane; j < D / 8; j += 32)
        acc += dot8(*(w4 + j), xn + j * 8);
    acc = warp_sum(acc);
    if (lane != 0) return;
    if (inject_block) {
        a.inject_out[row] = acc;
    } else {
        const float x = acc / (float)HC;
        a.lo[row] = x / (1.0f + sycl::native::exp(-x));
    }
}

void gr_up_kernel(FusedGrArgs a, float* lo, float g[4 /*HC*/][32 /*UP_COLS*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int d0 = item_ct1.get_group(2) * UP_COLS;
    for (int k = t; k < LR; k += THREADS)
        lo[k] = a.lo[k];
    /*
    DPCT1065:250: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // 128 rows (4 streams x 32 columns), 16 per warp: 320 bf16 = 40 chunks of 8.
    for (int r = warp; r < HC * UP_COLS; r += WARPS) {
        const int c = r / UP_COLS, dd = r - c * UP_COLS, i = c * N + d0 + dd;
        const sycl::uint4* w4 = reinterpret_cast<const sycl::uint4*>(a.w_up + (size_t)i * LR);
        /*
        DPCT1098:252: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        float acc = dot8(*(w4 + lane), lo + lane * 8);
        /*
        DPCT1098:253: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        if (lane < LR / 8 - 32) acc += dot8(*(w4 + 32 + lane), lo + (32 + lane) * 8);
        acc = warp_sum(acc);
        if (lane == 0) {
            float rv = a.R[i];
            if (a.apply) {
                rv = sycl::fma((float)(a.bo_prev[d0 + dd]), 2.0f * sigmoidf_(a.inj_prev[c] / (float)HC), rv);
                a.R_out[i] = rv;  // this block owns column d0+dd of every stream
            }
            const float x = rv * a.w_norm[i] * a.rs[c];
            g[c][dd] = x * sigmoidf_(acc);
        }
    }
    /*
    DPCT1065:251: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (t < UP_COLS) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c)
            s += g[c][t];
        a.mixed[d0 + t] = s / (float)HC;
    }
}

// ================================ plan v0.3 P6: T tokens, one weight read ================================
struct GrMulti {
    FusedGrArgs a[kFusedGrMaxT];
    float* xn;
    int T;
};

// Step 1 of `gr_down_kernel`, one block per token, same threads and reduction order: rs[t] and xn[t] to global.
/*
DPCT1110:15: The total declared local variable size in device function
gr_norm_multi_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void gr_norm_multi_kernel(GrMulti m, float part[8 /*WARPS*/][4 /*HC*/], float* s_rs) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const FusedGrArgs& a = m.a[item_ct1.get_group(2)];
    float* xn = m.xn + (size_t)item_ct1.get_group(2) * D;
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c)
        gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float)HC) : 0.0f;
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t * 4; i < D; i += THREADS * 4) {
        const int c = i / N, d = i - c * N;
        sycl::float4 r = *reinterpret_cast<const sycl::float4*>(a.R + i);
        if (a.apply) {
            const sycl::float4 b = *reinterpret_cast<const sycl::float4*>(a.bo_prev + d);
            r.x() = sycl::fma((float)(b.x()), gw[c], r.x());
            r.y() = sycl::fma((float)(b.y()), gw[c], r.y());
            r.z() = sycl::fma((float)(b.z()), gw[c], r.z());
            r.w() = sycl::fma((float)(b.w()), gw[c], r.w());
        }
        const sycl::float4 g = *reinterpret_cast<const sycl::float4*>(a.w_norm + i);
        float sq = r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + r.w() * r.w();
#pragma unroll
        for (int cc = 0; cc < HC; ++cc)
            if (cc == c) ss[cc] += sq;
        *reinterpret_cast<sycl::float4*>(xn + i) =
            sycl::float4(r.x() * g.x(), r.y() * g.y(), r.z() * g.z(), r.w() * g.w());
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    /*
    DPCT1065:254: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (t < HC) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w)
            s += part[w][t];
        s_rs[t] = sycl::rsqrt(s / (float)N + a.eps);
        a.rs[t] = s_rs[t];
    }
    /*
    DPCT1065:255: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int i = t; i < D; i += THREADS)
        xn[i] *= s_rs[i / N];
}

// Step 2 of `gr_down_kernel` for T tokens.  One warp per row (so each lane accumulates the same chunks in the
// same order as the single-token kernel); per tile the lane's weight chunks are loaded BEFORE the activation
// tile is staged, so the DRAM and L2 traffic are in flight together.
//
// TILEV = xn floats per token staged at a time.  The tile only changes the staging granularity: the lane's chunk
// order (lane + 32*q within the tile, tiles ascending) is strictly increasing for either value, so the results are
// bitwise identical to `gr_down_kernel` for both.  2560 stages 320 chunks of 8 per tile (10 per lane); cards whose
// opt-in below 8 * 2560 * 4 B slices the tokens - sm_75 (64 KiB) carries 6 tokens of it - run TILEV 1280 instead,
// which fits all eight tokens in one 40 KiB launch and stages 160 chunks of 8 (5 per lane, half the registers held
// for the weight prefetch); smaller tiles raise how many blocks share an SM (the 41-block grid), e.g. three blocks
// of four tokens instead of one on sm_75.
template <int TILEV>
/*
DPCT1110:16: The total declared local variable size in device function
gr_down_multi_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void gr_down_multi_kernel(GrMulti m, uint8_t* smem_raw) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int TQ = TILEV / 8 / 32;  // uint4 weight chunks per lane per tile
    auto tile = (float*)smem_raw;       // [T][TILEV]
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const bool inject_block = item_ct1.get_group(2) == DOWN_BLOCKS;
    const int row = inject_block ? warp : item_ct1.get_group(2) * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t)(active ? row : 0) * D;
    const sycl::uint4* w4 = reinterpret_cast<const sycl::uint4*>(wrow);
    float acc[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k)
        acc[k] = 0.0f;
    for (int base = 0; base < D; base += TILEV) {
        sycl::uint4 wv[TQ];
        if (active) {
#pragma unroll
            /*
            DPCT1098:256: The '*' expression is used instead of the __ldg call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            for (int q = 0; q < TQ; ++q)
                wv[q] = *(w4 + base / 8 + lane + 32 * q);
        }
        /*
        DPCT1118:17: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:257: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();  // the previous tile is consumed
        const sycl::float4* src4 = reinterpret_cast<const sycl::float4*>(m.xn);
        sycl::float4* tile4 = reinterpret_cast<sycl::float4*>(tile);
        for (int i = t; i < T * (TILEV / 4); i += THREADS) {
            const int k = i / (TILEV / 4), off = i - k * (TILEV / 4);
            tile4[i] = src4[((size_t)k * D + base) / 4 + off];
        }
        /*
        DPCT1118:18: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:258: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (!active) continue;
#pragma unroll
        for (int q = 0; q < TQ; ++q) {
            const int j = lane + 32 * q;
#pragma unroll
            for (int k = 0; k < kFusedGrMaxT; ++k)
                if (k < T) acc[k] += dot8(wv[q], tile + k * TILEV + j * 8);
        }
    }
    if (!active) return;
    float s[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k)
        s[k] = k < T ? warp_sum(acc[k]) : 0.0f;
    // lane k writes token k (every lane holds every sum after the xor reduction)
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) {
        if (k >= T || lane != k) continue;
        if (inject_block) {
            m.a[k].inject_out[row] = s[k];
        } else {
            const float x = s[k] / (float)HC;
            m.a[k].lo[row] = x / (1.0f + sycl::native::exp(-x));
        }
    }
}

constexpr int UPM_COLS = 16;              // columns per block (x 4 streams = 64 rows, 8 per warp)
constexpr int UPM_BLOCKS = N / UPM_COLS;  // 160

// `gr_up_kernel` for T tokens: each row of w_up read once; the T dots reduced by xor so every lane holds every
// sum, and lane k runs token k's epilogue - the T epilogues in parallel instead of one after another.
/*
DPCT1110:19: The total declared local variable size in device function
gr_up_multi_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void gr_up_multi_kernel(GrMulti m, float lo[8 /*kFusedGrMaxT*/][320 /*LR*/],
                        float g[8 /*kFusedGrMaxT*/][4 /*HC*/][16 /*UPM_COLS*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const int d0 = item_ct1.get_group(2) * UPM_COLS;
    for (int i = t; i < T * LR; i += THREADS)
        lo[i / LR][i % LR] = m.a[i / LR].lo[i % LR];
    /*
    DPCT1065:259: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int r = warp; r < HC * UPM_COLS; r += WARPS) {
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const sycl::uint4* w4 = reinterpret_cast<const sycl::uint4*>(m.a[0].w_up + (size_t)i * LR);
        /*
        DPCT1098:261: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const sycl::uint4 wa = *(w4 + lane);
        /*
        DPCT1098:262: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const sycl::uint4 wb = lane < LR / 8 - 32 ? *(w4 + 32 + lane) : sycl::uint4(0, 0, 0, 0);
        // the epilogue inputs of this lane's token, fetched while the dots run
        float rv = 0.0f, wn = 0.0f, rsc = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            rsc = a.rs[c];
            apply = a.apply;
            if (apply) {
                bo = a.bo_prev[d0 + dd];
                ip = a.inj_prev[c];
            }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            if (k >= T) break;
            float acc = dot8(wa, lo[k] + lane * 8);
            if (lane < LR / 8 - 32) acc += dot8(wb, lo[k] + (32 + lane) * 8);
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = sycl::fma(bo, 2.0f * sigmoidf_(ip / (float)HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsc;
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    /*
    DPCT1065:260: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c)
            s += g[k][c][col];
        m.a[k].mixed[d0 + col] = s / (float)HC;
    }
}

// ================================ hc read v3 (opt-in: STRATA_GR_V3=1) - two kernels, stream-split ================
// The norm kernel runs on only T blocks (~16 us of pure latency per call) and `down` on 41 blocks (~280 GB/s).
// v3: `down` is split over (row group, stream[, column half]) = 164 or 328 blocks; each block stages its slice of
// R' * w_norm for the T tokens, reduces that slice's sum of squares itself, and writes UNSCALED partial dots.  The
// rms scale is per stream, so  w_down . xn = sum_c rs[c] * (w_down[:, c] . (R'[c] * w_norm[c]))  - `up` applies it
// in its prologue (lo, inject, rs).  Same maths, ANOTHER SUMMATION ORDER: not bitwise the default kernels, hence
// opt-in.  Dynamic shared memory is T * (N / S) floats: S (1 or 2 column halves) is the smallest that fits the
// card's opt-in limit at kFusedGrMaxT tokens (Ampere 99 KB: S = 1; Turing / HIP 64 KB: S = 2); a card where
// neither fits keeps the default kernels.
constexpr int PR = LR + HC;                       // partial rows per (token, stream): 320 down + 4 inject
[[maybe_unused]] constexpr int TQ3 = N / 8 / 32;  // uint4 weight chunks per lane in one stream's slice (10)

template <int S>
/*
DPCT1110:20: The total declared local variable size in device function
gr_down_v3_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void gr_down_v3_kernel(GrMulti m, float* __restrict__ part, float* __restrict__ ssg, uint8_t* smem_raw,
                       float red[8 /*WARPS*/][8 /*kFusedGrMaxT*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto xs = (float*)smem_raw;  // [T][N / S]
    constexpr int R2 = 1;        // down rows per warp

    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int T = m.T;
    // S = 2: each stream's 2560 columns in two halves (blockIdx.y = stream * S + half): twice the blocks
    constexpr int SL = N / S, TQS = SL / 8 / 32;
    const int rg = item_ct1.get_group(2), c = item_ct1.get_group(1) / S,
              h = item_ct1.get_group(1) - (item_ct1.get_group(1) / S) * S;
    constexpr int NDB = LR / (WARPS * R2);  // down row blocks per stream; block NDB = the inject rows
    const bool inject_block = rg == NDB;
    // warp w owns rows row0 + w * R2 + r (r < R2); the inject block: warps 0-3, one row each
    const int row0 = inject_block ? warp : (rg * WARPS + warp) * R2;
    const int nrows = inject_block ? ((m.a[0].w_inject != nullptr && warp < HC) ? 1 : 0) : R2;
    const bool active = nrows > 0;
    const uint16_t* wbase = inject_block ? m.a[0].w_inject : m.a[0].w_down;
    sycl::uint4 wv[R2][TQS];
#pragma unroll
    for (int r = 0; r < R2; ++r) {
        if (r >= nrows) break;
        const sycl::uint4* w4 =
            reinterpret_cast<const sycl::uint4*>(wbase + (size_t)(row0 + r) * D + (size_t)c * N + (size_t)h * SL);
#pragma unroll
        /*
        DPCT1098:263: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        for (int q = 0; q < TQS; ++q)
            wv[r][q] = *(w4 + lane + 32 * q);
    }
    float ssp[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) {
        ssp[k] = 0.0f;
        if (k >= T) continue;
        const FusedGrArgs& a = m.a[k];
        const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float)HC) : 0.0f;
        const sycl::float4* R4 = reinterpret_cast<const sycl::float4*>(a.R + (size_t)c * N + (size_t)h * SL);
        const sycl::float4* G4 = reinterpret_cast<const sycl::float4*>(a.w_norm + (size_t)c * N + (size_t)h * SL);
        const sycl::float4* B4 = reinterpret_cast<const sycl::float4*>(a.bo_prev + (size_t)h * SL);
        sycl::float4* X4 = reinterpret_cast<sycl::float4*>(xs + (size_t)k * SL);
        for (int i = t; i < SL / 4; i += THREADS) {
            sycl::float4 r = R4[i];
            if (a.apply) {
                const sycl::float4 b = B4[i];
                r.x() = sycl::fma((float)(b.x()), (float)gw, r.x());
                r.y() = sycl::fma((float)(b.y()), (float)gw, r.y());
                r.z() = sycl::fma((float)(b.z()), (float)gw, r.z());
                r.w() = sycl::fma((float)(b.w()), (float)gw, r.w());
            }
            const sycl::float4 g = G4[i];
            ssp[k] += r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + r.w() * r.w();
            X4[i] = sycl::float4(r.x() * g.x(), r.y() * g.y(), r.z() * g.z(), r.w() * g.w());
        }
    }
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) {
        if (k >= T) break;
        const float v = warp_sum(ssp[k]);
        if (lane == 0) red[warp][k] = v;
    }
    /*
    DPCT1065:264: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (rg == 0 && t < T) {
        float sum = 0.0f;
        for (int w = 0; w < WARPS; ++w)
            sum += red[w][t];
        ssg[(t * HC + c) * S + h] = sum;
    }
    if (!active) return;
#pragma unroll
    for (int r = 0; r < R2; ++r) {
        if (r >= nrows) break;
        float acc[kFusedGrMaxT];
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k)
            acc[k] = 0.0f;
#pragma unroll
        for (int q = 0; q < TQS; ++q) {
            const int j = lane + 32 * q;
#pragma unroll
            for (int k = 0; k < kFusedGrMaxT; ++k)
                if (k < T) acc[k] += dot8(wv[r][q], xs + (size_t)k * SL + j * 8);
        }
        const int prow = inject_block ? LR + warp : row0 + r;
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            if (k >= T) break;
            const float v = warp_sum(acc[k]);
            if (lane == 0) part[(((size_t)k * HC + c) * S + h) * PR + prow] = v;
        }
    }
}

template <int S>
/*
DPCT1110:21: The total declared local variable size in device function
gr_up_v3_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
void gr_up_v3_kernel(GrMulti m, const float* __restrict__ part, const float* __restrict__ ssg,
                     float lo[8 /*kFusedGrMaxT*/][320 /*LR*/], float rsS[8 /*kFusedGrMaxT*/][4 /*HC*/],
                     float g[8 /*kFusedGrMaxT*/][4 /*HC*/][16 /*UPM_COLS*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const int d0 = item_ct1.get_group(2) * UPM_COLS;
    constexpr int RPW = HC * UPM_COLS / WARPS;  // 8 rows per warp
    if (t < T * HC) {
        const int k = t / HC, c = t - k * HC;
        float ss = 0.0f;
#pragma unroll
        for (int h = 0; h < S; ++h)
            ss += ssg[t * S + h];
        const float r = sycl::rsqrt(ss / (float)N + m.a[k].eps);
        rsS[k][c] = r;
        if (item_ct1.get_group(2) == 0) m.a[k].rs[c] = r;
    }
    /*
    DPCT1065:267: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int i = t; i < T * LR; i += THREADS) {
        const int k = i / LR, r = i - k * LR;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            float p = 0.0f;
#pragma unroll
            for (int h = 0; h < S; ++h)
                p += part[(((size_t)k * HC + c) * S + h) * PR + r];
            sum = sycl::fma(rsS[k][c], p, sum);
        }
        const float x = sum / (float)HC;
        lo[k][r] = x / (1.0f + sycl::native::exp(-x));
    }
    if (item_ct1.get_group(2) == 0 && t < T * HC) {
        const int k = t / HC, cc = t - k * HC;
        if (m.a[k].w_inject != nullptr) {
            float sum = 0.0f;
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                float p = 0.0f;
#pragma unroll
                for (int h = 0; h < S; ++h)
                    p += part[(((size_t)k * HC + c) * S + h) * PR + LR + cc];
                sum = sycl::fma(rsS[k][c], p, sum);
            }
            m.a[k].inject_out[cc] = sum;
        }
    }
    /*
    DPCT1065:268: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#pragma unroll
    for (int q = 0; q < RPW; ++q) {
        const int r = warp + q * WARPS;
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const sycl::uint4* w4 = reinterpret_cast<const sycl::uint4*>(m.a[0].w_up + (size_t)i * LR);
        /*
        DPCT1098:265: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const sycl::uint4 wa = *(w4 + lane);
        /*
        DPCT1098:266: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const sycl::uint4 wb = lane < LR / 8 - 32 ? *(w4 + 32 + lane) : sycl::uint4(0, 0, 0, 0);
        float rv = 0.0f, wn = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            apply = a.apply;
            if (apply) {
                bo = a.bo_prev[d0 + dd];
                ip = a.inj_prev[c];
            }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            if (k >= T) break;
            float acc = dot8(wa, lo[k] + lane * 8);
            if (lane < LR / 8 - 32) acc += dot8(wb, lo[k] + (32 + lane) * 8);
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = sycl::fma(bo, 2.0f * sigmoidf_(ip / (float)HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsS[lane][c];
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    /*
    DPCT1065:269: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c)
            sum += g[k][c][col];
        m.a[k].mixed[d0 + col] = sum / (float)HC;
    }
}
// ================================ #315: split / staged - the same arithmetic, split differently ==================
// The multi read's variants beside the plain read (0.1.31's default; not main's opt-in STRATA_GR_V3 read, which sums
// in another order).  Every variant computes each output with exactly the plain read's operations in its order (so
// bitwise the plain read and the single-token read); only who does what, and when, changes.  `fused_gr_check`
// compares them on the card before a verify window uses them (STRATA_HC_SPLIT below).
//  - the norm (split and staged): one block per token AND stream instead of one per token.  Each thread visits
//    exactly the elements, in the order, it visited for that stream in the plain read, so every sum of squares and rs
//    is the same; then it recomputes its R' * w_norm and writes it times rs (the plain read stored the product and
//    scaled it in place: the same two roundings).  On an RTX 4080 SUPER, the plain read's norm took 0.58 ms per
//    4-token round (96 launches of 1-4 blocks), this one 0.23 ms.
//  - the down projection, split: the plain read's kernel.
//  - the down projection, staged: the plain read's 8 rows per block and lane order, but the activations arrive by
//    cp.async in half-stream tiles, two in flight, so no thread waits on a chain of loads, and they sit in shared
//    memory as two planes (floats 0-3 and 4-7 of every 8), so a warp's reads hit no bank twice.  The next tile's
//    weights are loaded while the current one is used.  A tile is 160 = 5 x 32 chunks of 8, so a lane still
//    accumulates its chunks lane + 32 q in ascending order, as in the plain read with either tile.  Before sm_80
//    (and on HIP) the staging is a plain copy: the same bits, only not asynchronous.
constexpr int kHcPlain = 1, kHcSplit = 2, kHcStaged = 3;
constexpr int H_TILE = 1280;  // staged tile: half a stream = 160 chunks of 8, 5 per lane
constexpr int HQ = H_TILE / 8 / 32;
constexpr int N_HTILES = D / H_TILE;  // 8
static_assert(N % H_TILE == 0, "a staged tile never straddles two streams");
static_assert(H_TILE % 256 == 0, "a staged tile holds whole rounds of 32 chunks: the plain read's lane order");

/*
DPCT1110:22: The total declared local variable size in device function
gr_norm_split_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void gr_norm_split_kernel(GrMulti m, float* part, float& s_rs) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const FusedGrArgs& a = m.a[item_ct1.get_group(2)];
    const int c = item_ct1.get_group(1);
    float* xn = m.xn + (size_t)item_ct1.get_group(2) * D;
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float)HC) : 0.0f;
    float ss = 0.0f;
    for (int i = t * 4; i < D; i += THREADS * 4) {
        if (i / N != c) continue;  // another stream's element: another block's
        const int d = i - c * N;
        sycl::float4 r = *reinterpret_cast<const sycl::float4*>(a.R + i);
        if (a.apply) {
            const sycl::float4 b = *reinterpret_cast<const sycl::float4*>(a.bo_prev + d);
            r.x() = sycl::fma((float)(b.x()), (float)gw, r.x());
            r.y() = sycl::fma((float)(b.y()), (float)gw, r.y());
            r.z() = sycl::fma((float)(b.z()), (float)gw, r.z());
            r.w() = sycl::fma((float)(b.w()), (float)gw, r.w());
        }
        const float sq = r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + r.w() * r.w();
        ss += sq;
    }
    const float v = warp_sum(ss);
    if (lane == 0) part[warp] = v;
    /*
    DPCT1065:270: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (t == 0) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w)
            s += part[w];
        s_rs = sycl::rsqrt(s / (float)N + a.eps);
        a.rs[c] = s_rs;
    }
    /*
    DPCT1065:271: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const float rs = s_rs;
    for (int i = t * 4; i < D; i += THREADS * 4) {
        if (i / N != c) continue;
        const int d = i - c * N;
        sycl::float4 r = *reinterpret_cast<const sycl::float4*>(a.R + i);
        if (a.apply) {
            const sycl::float4 b = *reinterpret_cast<const sycl::float4*>(a.bo_prev + d);
            r.x() = sycl::fma((float)(b.x()), (float)gw, r.x());
            r.y() = sycl::fma((float)(b.y()), (float)gw, r.y());
            r.z() = sycl::fma((float)(b.z()), (float)gw, r.z());
            r.w() = sycl::fma((float)(b.w()), (float)gw, r.w());
        }
        const sycl::float4 g = *reinterpret_cast<const sycl::float4*>(a.w_norm + i);
        const float px = r.x() * g.x(), py = r.y() * g.y(), pz = r.z() * g.z(), pw = r.w() * g.w();
        *reinterpret_cast<sycl::float4*>(xn + i) = sycl::float4(px * rs, py * rs, pz * rs, pw * rs);
    }
}

// 143-amdk: gr_norm_split_kernel with the loads of a thread's (at most three) float4 issued together
// and the scaled product written once from registers. Eliminates the second global memory pass.
void gr_norm_split_fast_kernel(GrMulti m, float* part, float& s_rs) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const FusedGrArgs& a = m.a[item_ct1.get_group(2)];
    const int c = item_ct1.get_group(1);
    float* xn = m.xn + (size_t)item_ct1.get_group(2) * D;
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float)HC) : 0.0f;
    constexpr int KS = 3;
    int k0 = (c * N - 4 * t + THREADS * 4 - 1) / (THREADS * 4);
    if (k0 < 0) k0 = 0;
    sycl::float4 r[KS], g[KS];
    bool ok[KS];
#pragma unroll
    for (int j = 0; j < KS; ++j) {
        const int i = t * 4 + (k0 + j) * THREADS * 4;
        ok[j] = i < (c + 1) * N;
        if (ok[j]) {
            r[j] = *reinterpret_cast<const sycl::float4*>(a.R + i);
            g[j] = *reinterpret_cast<const sycl::float4*>(a.w_norm + i);
            if (a.apply) {
                const sycl::float4 b = *reinterpret_cast<const sycl::float4*>(a.bo_prev + (i - c * N));
                r[j].x() = sycl::fma(b.x(), gw, r[j].x());
                r[j].y() = sycl::fma(b.y(), gw, r[j].y());
                r[j].z() = sycl::fma(b.z(), gw, r[j].z());
                r[j].w() = sycl::fma(b.w(), gw, r[j].w());
            }
        }
    }
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < KS; ++j) {
        if (ok[j]) {
            const float sq = r[j].x() * r[j].x() + r[j].y() * r[j].y() + r[j].z() * r[j].z() + r[j].w() * r[j].w();
            ss += sq;
            r[j] = sycl::float4(r[j].x() * g[j].x(), r[j].y() * g[j].y(), r[j].z() * g[j].z(), r[j].w() * g[j].w());
        }
    }
    const float v = warp_sum(ss);
    if (lane == 0) part[warp] = v;
    item_ct1.barrier();
    if (t == 0) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w)
            s += part[w];
        s_rs = sycl::rsqrt(s / (float)N + a.eps);
        a.rs[c] = s_rs;
    }
    item_ct1.barrier();
    const float rs = s_rs;
#pragma unroll
    for (int j = 0; j < KS; ++j) {
        if (ok[j]) {
            *reinterpret_cast<sycl::float4*>(xn + t * 4 + (k0 + j) * THREADS * 4) =
                sycl::float4(r[j].x() * rs, r[j].y() * rs, r[j].z() * rs, r[j].w() * rs);
        }
    }
}

inline void cp_async16(void* smem, const void* gmem) {
    *reinterpret_cast<sycl::float4*>(smem) = *reinterpret_cast<const sycl::float4*>(gmem);
}
inline void cp_async_commit() {}
inline void cp_async_wait1() {}
inline void cp_async_wait0() {}

// `dot8` with its 8 activations as two float4: the same eight fmaf in the same order
inline float dot8v(const sycl::uint4 w, const sycl::float4 x0, const sycl::float4 x1) {
    float acc = 0.0f;
    acc = sycl::fma(sycl::bit_cast<float>(w.x() << 16), (float)(x0.x()), acc);
    acc = sycl::fma(sycl::bit_cast<float>(w.x() & 0xffff0000u), (float)(x0.y()), acc);
    acc = sycl::fma(sycl::bit_cast<float>(w.y() << 16), (float)(x0.z()), acc);
    acc = sycl::fma(sycl::bit_cast<float>(w.y() & 0xffff0000u), (float)(x0.w()), acc);
    acc = sycl::fma(sycl::bit_cast<float>(w.z() << 16), (float)(x1.x()), acc);
    acc = sycl::fma(sycl::bit_cast<float>(w.z() & 0xffff0000u), (float)(x1.y()), acc);
    acc = sycl::fma(sycl::bit_cast<float>(w.w() << 16), (float)(x1.z()), acc);
    acc = sycl::fma(sycl::bit_cast<float>(w.w() & 0xffff0000u), (float)(x1.w()), acc);
    return acc;
}

// Stage tile `h` of every token into `buf`: [T][2 planes][160 chunks] float4, plane 0 = floats 0-3 of a chunk.
inline void stage_htile(const GrMulti& m, int T, int h, sycl::float4* buf, int t) {
    for (int i = t; i < T * (H_TILE / 4); i += THREADS) {
        const int k = i / (H_TILE / 4), s4 = i - k * (H_TILE / 4);  // s4: float4 of the tile, chunk s4/2, half s4&1
        const float* src = m.xn + (size_t)k * D + (size_t)h * H_TILE + (size_t)s4 * 4;
        cp_async16(buf + (size_t)k * (H_TILE / 4) + (s4 & 1) * (H_TILE / 8) + (s4 >> 1), src);
    }
}

/*
DPCT1110:27: The total declared local variable size in device function
gr_down_staged_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void gr_down_staged_kernel(GrMulti m, uint8_t* smem_raw) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto hbuf = (sycl::float4*)smem_raw;  // 2 buffers x [T][2][160] float4
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const bool inject_block = item_ct1.get_group(2) == DOWN_BLOCKS;
    const int row = inject_block ? warp : item_ct1.get_group(2) * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t)(active ? row : 0) * D;
    const sycl::uint4* w4 = reinterpret_cast<const sycl::uint4*>(wrow);
    const size_t buf_f4 = (size_t)T * (H_TILE / 4);  // float4 per buffer (the second one follows the first)
    float acc[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k)
        acc[k] = 0.0f;
    sycl::uint4 wv[HQ], wnext[HQ];
    if (active) {
#pragma unroll
        /*
        DPCT1098:272: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        for (int q = 0; q < HQ; ++q)
            wv[q] = *(w4 + lane + 32 * q);
    }
    stage_htile(m, T, 0, hbuf, t);
    cp_async_commit();
    stage_htile(m, T, 1, hbuf + buf_f4, t);
    cp_async_commit();
#pragma unroll 1
    for (int h = 0; h < N_HTILES; ++h) {
        if (active && h + 1 < N_HTILES) {
#pragma unroll
            /*
            DPCT1098:275: The '*' expression is used instead of the __ldg call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            for (int q = 0; q < HQ; ++q)
                wnext[q] = *(w4 + (h + 1) * (H_TILE / 8) + lane + 32 * q);
        }
        if (h + 1 < N_HTILES)
            cp_async_wait1();  // tile h has landed (h + 1 may still be on its way)
        else
            cp_async_wait0();
        /*
        DPCT1118:28: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:273: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const sycl::float4* cur = hbuf + (h & 1) * buf_f4;
        if (active) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) {
                const int j = lane + 32 * q;
#pragma unroll
                for (int k = 0; k < kFusedGrMaxT; ++k) {
                    if (k < T) {
                        const sycl::float4* pk = cur + (size_t)k * (H_TILE / 4);
                        acc[k] += dot8v(wv[q], pk[j], pk[H_TILE / 8 + j]);
                    }
                }
            }
        }
        /*
        DPCT1118:29: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:274: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();  // every warp is done with buffer h & 1
        if (h + 2 < N_HTILES) stage_htile(m, T, h + 2, hbuf + (h & 1) * buf_f4, t);
        cp_async_commit();  // an empty group at the end keeps the wait counts simple
        if (h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < HQ; ++q)
                wv[q] = wnext[q];
        }
    }
    if (!active) return;
    float s[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k)
        s[k] = k < T ? warp_sum(acc[k]) : 0.0f;
    // lane k writes token k (every lane holds every sum after the xor reduction)
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) {
        if (k >= T || lane != k) continue;
        if (inject_block) {
            m.a[k].inject_out[row] = s[k];
        } else {
            const float x = s[k] / (float)HC;
            m.a[k].lo[row] = x / (1.0f + sycl::native::exp(-x));
        }
    }
}

// The tokens a down kernel may carry in one launch on the current card, and the plain read's tile: the plain read
// stages n_tok * TILEV floats (1280 on sm_75, 2560 elsewhere), staged two tiles of n_tok * H_TILE.  The shared-memory
// opt-in is set here once per device (a per-DEVICE setting: a layer split runs these kernels on two cards).
int down_chunk(bool staged, int* tile_out) {
    static bool attr = false;
    static int chunk = 0;
    static int chunk_staged = 0;
    static int tile = 0;
    if (!attr) {
        int optin = (int)default_queue().get_device().get_info<sycl::info::device::local_mem_size>();
        optin = strata::smem_optin_of(optin);
        const bool small_tile = false;
        const int tv = small_tile ? 1280 : 2560;
        tile = tv;
        int want = (int)(kFusedGrMaxT * tv * sizeof(float));
        if (optin > 0 && want > optin) want = optin;
        chunk = want / (int)(tv * sizeof(float));
        if (chunk < 1) chunk = 1;
        if (chunk > kFusedGrMaxT) chunk = kFusedGrMaxT;
        const int per_tok_staged = 2 * 2 * 160 * (int)sizeof(sycl::float4);
        int c_staged = optin > 0 ? (optin / per_tok_staged) : kFusedGrMaxT;
        if (c_staged < 1) c_staged = 1;
        if (c_staged > kFusedGrMaxT) c_staged = kFusedGrMaxT;
        chunk_staged = c_staged;
        attr = true;
    }
    *tile_out = tile;
    return staged ? chunk_staged : chunk;
}

// The multi read as `variant` (kHcPlain, kHcSplit or kHcStaged): the norm, the down projection in launches of as
// many tokens as fit the card, the up projection; the profile's stamps after the norm and after the down projection.
// kHcPlain is the default read exactly as before (never main's opt-in STRATA_GR_V3 path, which fused_gr_read_multi
// takes first).
void launch_multi(const GrMulti& m, int variant, sycl::queue* st, unsigned long long* stamp_buf, int stamp_i0) {
    const int n_tok = m.T;
    if (variant >= kHcSplit)
        st->submit([&](sycl::handler& cgh) {
            /*
            DPCT1101:867: 'WARPS' expression was replaced with a value. Modify
            the code to use the original expression, provided in comments, if it
            is correct.
            */
            sycl::local_accessor<float, 1> part_acc_ct1(sycl::range<1>(8 /*WARPS*/), cgh);
            sycl::local_accessor<float, 0> s_rs_acc_ct1(cgh);

            const bool norm_fast = [] {
                const char* e = std::getenv("STRATA_GR_NORM_FAST");
                return e == nullptr || std::atoi(e) != 0;
            }();
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, HC, (unsigned)n_tok) * sycl::range<3>(1, 1, THREADS),
                                               sycl::range<3>(1, 1, THREADS)),
                             [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                                 if (norm_fast)
                                     gr_norm_split_fast_kernel(
                                         m, part_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(), s_rs_acc_ct1);
                                 else
                                     gr_norm_split_kernel(
                                         m, part_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(), s_rs_acc_ct1);
                             });
        });
    else
        st->submit([&](sycl::handler& cgh) {
            /*
            DPCT1101:868: 'WARPS' expression was replaced with a value. Modify
            the code to use the original expression, provided in comments, if it
            is correct.
            */
            /*
            DPCT1101:869: 'HC' expression was replaced with a value. Modify the
            code to use the original expression, provided in comments, if it is
            correct.
            */
            sycl::local_accessor<float[8 /*WARPS*/][4 /*HC*/], 0> part_acc_ct1(cgh);
            /*
            DPCT1101:870: 'HC' expression was replaced with a value. Modify the
            code to use the original expression, provided in comments, if it is
            correct.
            */
            sycl::local_accessor<float, 1> s_rs_acc_ct1(sycl::range<1>(4 /*HC*/), cgh);

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, n_tok) * sycl::range<3>(1, 1, THREADS),
                                               sycl::range<3>(1, 1, THREADS)),
                             [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                                 gr_norm_multi_kernel(m, part_acc_ct1,
                                                      s_rs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, (void*)st);
    const bool staged = variant >= kHcStaged;
    int tv = 2560;
    const int chunk_tok = down_chunk(staged, &tv);
    /*
    DPCT1083:30: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    const size_t per_tok = (staged ? (size_t)2 * H_TILE : (size_t)tv) * sizeof(float);
    for (int c0 = 0; c0 < n_tok; c0 += chunk_tok) {
        const int ct = n_tok - c0 < chunk_tok ? n_tok - c0 : chunk_tok;
        GrMulti c{};
        if (ct == n_tok) {
            c = m;
        } else {
            c.xn = m.xn + (size_t)c0 * D;
            c.T = ct;
            for (int k = 0; k < ct; ++k)
                c.a[k] = m.a[c0 + k];
        }
        const size_t smem = (size_t)ct * per_tok;
        if (staged)
            st->submit([&](sycl::handler& cgh) {
                sycl::local_accessor<uint8_t, 1> smem_acc(sycl::range<1>(smem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, DOWN_BLOCKS + 1) * sycl::range<3>(1, 1, THREADS),
                                      sycl::range<3>(1, 1, THREADS)),
                    [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                        gr_down_staged_kernel(c, smem_acc.get_multi_ptr<sycl::access::decorated::no>().get());
                    });
            });
        else if (tv == 1280)
            st->submit([&](sycl::handler& cgh) {
                sycl::local_accessor<uint8_t, 1> smem_acc(sycl::range<1>(smem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, DOWN_BLOCKS + 1) * sycl::range<3>(1, 1, THREADS),
                                      sycl::range<3>(1, 1, THREADS)),
                    [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                        gr_down_multi_kernel<1280>(c, smem_acc.get_multi_ptr<sycl::access::decorated::no>().get());
                    });
            });
        else
            st->submit([&](sycl::handler& cgh) {
                sycl::local_accessor<uint8_t, 1> smem_acc(sycl::range<1>(smem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, DOWN_BLOCKS + 1) * sycl::range<3>(1, 1, THREADS),
                                      sycl::range<3>(1, 1, THREADS)),
                    [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                        gr_down_multi_kernel<2560>(c, smem_acc.get_multi_ptr<sycl::access::decorated::no>().get());
                    });
            });
    }
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, (void*)st);
    st->submit([&](sycl::handler& cgh) {
        /*
        DPCT1101:871: 'kFusedGrMaxT' expression was replaced with a value.
        Modify the code to use the original expression, provided in comments, if
        it is correct.
        */
        /*
        DPCT1101:872: 'LR' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        sycl::local_accessor<float[8 /*kFusedGrMaxT*/][320 /*LR*/], 0> lo_acc_ct1(cgh);
        /*
        DPCT1101:873: 'kFusedGrMaxT' expression was replaced with a value.
        Modify the code to use the original expression, provided in comments, if
        it is correct.
        */
        /*
        DPCT1101:874: 'HC' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        /*
        DPCT1101:875: 'UPM_COLS' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float[8 /*kFusedGrMaxT*/][4 /*HC*/][16 /*UPM_COLS*/], 0> g_acc_ct1(cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, UPM_BLOCKS) * sycl::range<3>(1, 1, THREADS),
                                           sycl::range<3>(1, 1, THREADS)),
                         [=](sycl::nd_item<3> /*item_ct1*/)
                             [[sycl::reqd_sub_group_size(32)]] { gr_up_multi_kernel(m, lo_acc_ct1, g_acc_ct1); });
    });
}

/// STRATA_HC_SPLIT: unset or 2 = the newest the check accepts (staged), 1 = at most split, 0 = the plain read
int env_variant() {
    const char* e = std::getenv("STRATA_HC_SPLIT");
    if (e == nullptr || e[0] == '\0') return kHcStaged;
    if (e[0] == '0') return kHcPlain;
    if (e[0] == '1') return kHcSplit;
    return kHcStaged;
}

// per device: the variant `fused_gr_check` chose (0 = not checked yet)
std::atomic<int> g_variant[64];

}  // namespace

bool fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream,
                         unsigned long long* stamp_buf, int stamp_i0) {
    if (n_tok < 1 || n_tok > kFusedGrMaxT || xn_scratch == nullptr) {
        std::fprintf(stderr, "fused_gr_read_multi: invalid arguments\n");
        std::exit(1);
    }
    GrMulti m;
    for (int t = 0; t < n_tok; ++t) {
        m.a[t] = a[t];
        const FusedGrArgs& x = a[t];
        if (!x.R || !x.w_norm || !x.w_down || !x.w_up || !x.lo || !x.rs || !x.mixed || (x.w_inject && !x.inject_out) ||
            (x.apply && (!x.bo_prev || !x.inj_prev || !x.R_out)) || x.w_down != a[0].w_down || x.w_up != a[0].w_up ||
            x.w_inject != a[0].w_inject || x.w_norm != a[0].w_norm) {
            std::fprintf(stderr, "fused_gr_read_multi: invalid arguments for token %d\n", t);
            std::exit(1);
        }
    }
    m.xn = xn_scratch;
    m.T = n_tok;
    sycl::queue* st = stream ? (sycl::queue*)stream : &default_queue();
    // STRATA_GR_V3=1: the two-kernel read above (another summation order - opt-in)
    static const bool v3 = [] {
        const char* v = std::getenv("STRATA_GR_V3");
        return v != nullptr && std::atoi(v) != 0;
    }();
    static int split3[64] = {};  // per device: 0 = not decided yet, 1 / 2 = column halves S, -1 = does not fit
    int dev3 = 0;
    if (v3) {
        dev3 = 0;
        if (dev3 >= 0 && dev3 < 64 && split3[dev3] == 0) {
            int optin = 0;
            /*
            DPCT1019:281: local_mem_size in SYCL is not a complete equivalent of
            cudaDevAttrMaxSharedMemoryPerBlockOptin in CUDA. You may need to
            adjust the code.
            */
            optin = (int)default_queue().get_device().get_info<sycl::info::device::local_mem_size>();
            const int limit = optin > 0 ? optin : 48 * 1024;
            const int need1 = (int)(kFusedGrMaxT * N * sizeof(float)), need2 = need1 / 2;
            int split = -1;
            if (need1 <= limit &&
                /*
                DPCT1027:283: The call to cudaFuncSetAttribute was replaced with
                0 because SYCL currently does not support corresponding setting.
                */
                0 == 0)
                split = 1;
            else if (need2 <= limit)
                // #375 (kenh0u): the S = 2 split (a 64 KB opt-in card: Turing) disagrees with itself in gr_parity
                // (graph replay vs direct call) - such a card keeps the default read until that split is fixed
                std::fprintf(stderr,
                             "strata: STRATA_GR_V3=1 needs the two-half split on this card, which fails its "
                             "checks (#375): the default read is used\n");
            /*
            DPCT1026:282: The call to cudaGetLastError was removed because this
            functionality is redundant in SYCL.
            */
            ;  // drop any error the attempts left behind
            split3[dev3] = split;
        }
    }
    const int split = v3 && dev3 >= 0 && dev3 < 64 ? split3[dev3] : -1;
    if (split > 0) {  // 2 kernels; scratch = partials + sums of squares
        float* part = xn_scratch;
        float* ssg = xn_scratch + (size_t)n_tok * HC * 2 * PR;  // room for S = 2
        /*
        DPCT1083:31: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        const size_t sm = (size_t)n_tok * (N / split) * sizeof(float);
        if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, stream);
        if (split == 2)
            st->submit([&](sycl::handler& cgh) {
                sycl::local_accessor<uint8_t, 1> smem_acc(sycl::range<1>(sm), cgh);
                sycl::local_accessor<float[8 /*WARPS*/][8 /*kFusedGrMaxT*/], 0> red_acc_ct1(cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, HC * 2, LR / WARPS + 1) * sycl::range<3>(1, 1, THREADS),
                                      sycl::range<3>(1, 1, THREADS)),
                    [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                        gr_down_v3_kernel<2>(m, part, ssg, smem_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                                             red_acc_ct1);
                    });
            });
        else
            st->submit([&](sycl::handler& cgh) {
                sycl::local_accessor<uint8_t, 1> smem_acc(sycl::range<1>(sm), cgh);
                sycl::local_accessor<float[8 /*WARPS*/][8 /*kFusedGrMaxT*/], 0> red_acc_ct1(cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, HC, LR / WARPS + 1) * sycl::range<3>(1, 1, THREADS),
                                      sycl::range<3>(1, 1, THREADS)),
                    [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                        gr_down_v3_kernel<1>(m, part, ssg, smem_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                                             red_acc_ct1);
                    });
            });
        if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, stream);
        if (split == 2)
            st->submit([&](sycl::handler& cgh) {
                /*
                DPCT1101:880: 'kFusedGrMaxT' expression was replaced with a
                value. Modify the code to use the original expression, provided
                in comments, if it is correct.
                */
                /*
                DPCT1101:881: 'LR' expression was replaced with a value. Modify
                the code to use the original expression, provided in comments,
                if it is correct.
                */
                sycl::local_accessor<float[8 /*kFusedGrMaxT*/][320 /*LR*/], 0> lo_acc_ct1(cgh);
                /*
                DPCT1101:882: 'kFusedGrMaxT' expression was replaced with a
                value. Modify the code to use the original expression, provided
                in comments, if it is correct.
                */
                /*
                DPCT1101:883: 'HC' expression was replaced with a value. Modify
                the code to use the original expression, provided in comments,
                if it is correct.
                */
                sycl::local_accessor<float[8 /*kFusedGrMaxT*/][4 /*HC*/], 0> rsS_acc_ct1(cgh);
                /*
                DPCT1101:884: 'kFusedGrMaxT' expression was replaced with a
                value. Modify the code to use the original expression, provided
                in comments, if it is correct.
                */
                /*
                DPCT1101:885: 'HC' expression was replaced with a value. Modify
                the code to use the original expression, provided in comments,
                if it is correct.
                */
                /*
                DPCT1101:886: 'UPM_COLS' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<float[8 /*kFusedGrMaxT*/][4 /*HC*/][16 /*UPM_COLS*/], 0> g_acc_ct1(cgh);

                cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, UPM_BLOCKS) * sycl::range<3>(1, 1, THREADS),
                                                   sycl::range<3>(1, 1, THREADS)),
                                 [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                                     gr_up_v3_kernel<2>(m, part, ssg, lo_acc_ct1, rsS_acc_ct1, g_acc_ct1);
                                 });
            });
        else
            st->submit([&](sycl::handler& cgh) {
                /*
                DPCT1101:887: 'kFusedGrMaxT' expression was replaced with a
                value. Modify the code to use the original expression, provided
                in comments, if it is correct.
                */
                /*
                DPCT1101:888: 'LR' expression was replaced with a value. Modify
                the code to use the original expression, provided in comments,
                if it is correct.
                */
                sycl::local_accessor<float[8 /*kFusedGrMaxT*/][320 /*LR*/], 0> lo_acc_ct1(cgh);
                /*
                DPCT1101:889: 'kFusedGrMaxT' expression was replaced with a
                value. Modify the code to use the original expression, provided
                in comments, if it is correct.
                */
                /*
                DPCT1101:890: 'HC' expression was replaced with a value. Modify
                the code to use the original expression, provided in comments,
                if it is correct.
                */
                sycl::local_accessor<float[8 /*kFusedGrMaxT*/][4 /*HC*/], 0> rsS_acc_ct1(cgh);
                /*
                DPCT1101:891: 'kFusedGrMaxT' expression was replaced with a
                value. Modify the code to use the original expression, provided
                in comments, if it is correct.
                */
                /*
                DPCT1101:892: 'HC' expression was replaced with a value. Modify
                the code to use the original expression, provided in comments,
                if it is correct.
                */
                /*
                DPCT1101:893: 'UPM_COLS' expression was replaced with a value.
                Modify the code to use the original expression, provided in
                comments, if it is correct.
                */
                sycl::local_accessor<float[8 /*kFusedGrMaxT*/][4 /*HC*/][16 /*UPM_COLS*/], 0> g_acc_ct1(cgh);

                cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, UPM_BLOCKS) * sycl::range<3>(1, 1, THREADS),
                                                   sycl::range<3>(1, 1, THREADS)),
                                 [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                                     gr_up_v3_kernel<1>(m, part, ssg, lo_acc_ct1, rsS_acc_ct1, g_acc_ct1);
                                 });
            });
        return false;
    }
    // the default read (STRATA_GR_V3 unset): v1, or the bitwise-equal v2 / v3 this card's check accepted
    launch_multi(m, fused_gr_variant(), st, stamp_buf, stamp_i0);
    return false;
}

bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr) { return n_embd == N && hc == HC && hc_lr == LR; }

void fused_gr_read(const FusedGrArgs& a, void* stream) {
    if (!a.R || !a.w_norm || !a.w_down || !a.w_up || !a.lo || !a.rs || !a.mixed || (a.w_inject && !a.inject_out) ||
        (a.apply && (!a.bo_prev || !a.inj_prev || !a.R_out)) || (a.apply && a.inj_prev == a.inject_out)) {
        std::fprintf(stderr, "fused_gr_read: invalid arguments\n");
        std::exit(1);
    }
    sycl::queue* st = stream ? (sycl::queue*)stream : &default_queue();
    st->submit([&](sycl::handler& cgh) {
        /*
        DPCT1101:894: 'D' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        sycl::local_accessor<float, 1> xn_acc_ct1(sycl::range<1>(10240 /*D*/), cgh);
        /*
        DPCT1101:895: 'WARPS' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        /*
        DPCT1101:896: 'HC' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        sycl::local_accessor<float[8 /*WARPS*/][4 /*HC*/], 0> part_acc_ct1(cgh);
        /*
        DPCT1101:897: 'HC' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        sycl::local_accessor<float, 1> s_rs_acc_ct1(sycl::range<1>(4 /*HC*/), cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, DOWN_BLOCKS + 1) * sycl::range<3>(1, 1, THREADS),
                                           sycl::range<3>(1, 1, THREADS)),
                         [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                             gr_down_kernel(a, xn_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
                                            part_acc_ct1,
                                            s_rs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
    });
    st->submit([&](sycl::handler& cgh) {
        /*
        DPCT1101:898: 'LR' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        sycl::local_accessor<float, 1> lo_acc_ct1(sycl::range<1>(320 /*LR*/), cgh);
        /*
        DPCT1101:899: 'HC' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        /*
        DPCT1101:900: 'UP_COLS' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float[4 /*HC*/][32 /*UP_COLS*/], 0> g_acc_ct1(cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, UP_BLOCKS) * sycl::range<3>(1, 1, THREADS),
                                           sycl::range<3>(1, 1, THREADS)),
                         [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                             gr_up_kernel(a, lo_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(), g_acc_ct1);
                         });
    });
}

namespace {

/// The plain read, split, staged and (for one token) the single-token read on random bf16 weights and random inputs,
/// 1..8 tokens, with and without the pending write; every output of split and staged compared with the plain read's
/// bit for bit (and the plain read's with the single-token read's).  `why[v]` gets the first difference of variant
/// v; false if the check itself could not run.
bool fused_gr_selftest(bool ok_variant[4], std::string why[4]) try {
    constexpr int TM = kFusedGrMaxT;
    constexpr int NV = 4;  // sets: 0 = plain, 1 = split, 2 = staged, 3 = single-token
    std::mt19937 rng(20260930u);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint16_t> h_down((size_t)LR * D), h_up((size_t)D * LR), h_inj((size_t)HC * D);
    std::vector<float> h_norm(D), h_R((size_t)TM * D), h_bo((size_t)TM * N), h_ip((size_t)TM * HC);
    for (auto& w : h_down)
        w = bf16_from_f32(0.02f * nd(rng));
    for (auto& w : h_up)
        w = bf16_from_f32(0.05f * nd(rng));
    for (auto& w : h_inj)
        w = bf16_from_f32(0.02f * nd(rng));
    for (auto& w : h_norm)
        w = 1.0f + 0.1f * nd(rng);
    for (auto& x : h_R)
        x = nd(rng);
    for (auto& x : h_bo)
        x = 0.5f * nd(rng);
    for (auto& x : h_ip)
        x = 2.0f * nd(rng);
    for (int v = 0; v < 4; ++v) {
        ok_variant[v] = v == 1;
        why[v].clear();
    }

    const size_t n_set = (size_t)TM * (D + D + LR + HC + HC + N);  // R_out, xn, lo, rs, inject, mixed (floats)
    const size_t bytes = h_down.size() * 2 + h_up.size() * 2 + h_inj.size() * 2 +
                         (h_norm.size() + h_R.size() + h_bo.size() + h_ip.size() + NV * n_set) * 4 + 32 * 256;
    uint8_t* base = static_cast<uint8_t*>(sycl::malloc_device(bytes, default_queue()));
    if (!base) {
        why[0] = "no room for the check (" + std::to_string(bytes >> 20) + " MiB)";
        return false;
    }
    size_t off = 0;
    auto take = [&](size_t b) {
        void* p = base + off;
        off += (b + 255) / 256 * 256;
        return p;
    };
    uint16_t* d_down = (uint16_t*)take(h_down.size() * 2);
    uint16_t* d_up = (uint16_t*)take(h_up.size() * 2);
    uint16_t* d_inj = (uint16_t*)take(h_inj.size() * 2);
    float* d_norm = (float*)take(h_norm.size() * 4);
    float* d_R = (float*)take(h_R.size() * 4);
    float* d_bo = (float*)take(h_bo.size() * 4);
    float* d_ip = (float*)take(h_ip.size() * 4);
    struct Set {
        float *R_out, *xn, *lo, *rs, *inj, *mixed;
    } set[NV];
    for (Set& x : set) {
        x.R_out = (float*)take((size_t)TM * D * 4);
        x.xn = (float*)take((size_t)TM * D * 4);
        x.lo = (float*)take((size_t)TM * LR * 4);
        x.rs = (float*)take((size_t)TM * HC * 4);
        x.inj = (float*)take((size_t)TM * HC * 4);
        x.mixed = (float*)take((size_t)TM * N * 4);
    }
    sycl::queue* st = &default_queue();
    bool ok = (off <= bytes);
    try {
        st->memcpy(d_down, h_down.data(), h_down.size() * 2);
        st->memcpy(d_up, h_up.data(), h_up.size() * 2);
        st->memcpy(d_inj, h_inj.data(), h_inj.size() * 2);
        st->memcpy(d_norm, h_norm.data(), h_norm.size() * 4);
        st->memcpy(d_R, h_R.data(), h_R.size() * 4);
        st->memcpy(d_bo, h_bo.data(), h_bo.size() * 4);
        st->memcpy(d_ip, h_ip.data(), h_ip.size() * 4);
        st->wait();
    } catch (...) {
        ok = false;
    }
    if (!ok) why[0] = "setting up the check failed";
    std::vector<float> h1, h2;
    // true when equal; false with the first difference in `w` (or a read-back failure in `ok`)
    auto same = [&](const float* d1, const float* d2, size_t n, const char* what, int T, int apply, std::string& w) {
        h1.resize(n);
        h2.resize(n);
        try {
            st->memcpy(h1.data(), d1, n * 4);
            st->memcpy(h2.data(), d2, n * 4);
            st->wait();
        } catch (...) {
            why[0] = "reading back the check failed";
            ok = false;
            return false;
        }
        for (size_t i = 0; i < n; ++i) {
            uint32_t u1, u2;
            std::memcpy(&u1, &h1[i], 4);
            std::memcpy(&u2, &h2[i], 4);
            if (u1 != u2) {
                char buf[192];
                std::snprintf(buf, sizeof buf, "%s differs for %d token(s)%s at %zu (%08x, not %08x)", what, T,
                              apply ? " with the pending write" : "", i, u2, u1);
                w = buf;
                return false;
            }
        }
        return true;
    };
    auto all_same = [&](const Set& s1, const Set& s2, int T, int apply, std::string& w) {
        return same(s1.lo, s2.lo, (size_t)T * LR, "lo", T, apply, w) &&
               same(s1.rs, s2.rs, (size_t)T * HC, "rs", T, apply, w) &&
               same(s1.inj, s2.inj, (size_t)T * HC, "inject", T, apply, w) &&
               same(s1.mixed, s2.mixed, (size_t)T * N, "mixed", T, apply, w) &&
               same(s1.R_out, s2.R_out, (size_t)T * D, "R", T, apply, w);
    };
    ok_variant[2] = ok_variant[3] = ok;
    bool single_ok = ok;
    for (int apply = 0; apply < 2 && ok; ++apply) {
        for (int T = 1; T <= TM && ok; ++T) {
            FusedGrArgs a[NV][TM];
            for (int v = 0; v < NV; ++v) {
                const size_t span = (size_t)((uint8_t*)(set[v].mixed + (size_t)TM * N) - (uint8_t*)set[v].R_out);
                st->memset(set[v].R_out, 0xFF, span);
                for (int k = 0; k < T; ++k) {
                    FusedGrArgs& x = a[v][k];
                    x.R = d_R + (size_t)k * D;
                    x.R_out = set[v].R_out + (size_t)k * D;
                    x.apply = apply != 0;
                    x.bo_prev = d_bo + (size_t)k * N;
                    x.inj_prev = d_ip + (size_t)k * HC;
                    x.w_norm = d_norm;
                    x.w_down = d_down;
                    x.w_up = d_up;
                    x.w_inject = d_inj;
                    x.eps = 1e-6f;
                    x.lo = set[v].lo + (size_t)k * LR;
                    x.rs = set[v].rs + (size_t)k * HC;
                    x.inject_out = set[v].inj + (size_t)k * HC;
                    x.mixed = set[v].mixed + (size_t)k * N;
                }
            }
            for (int v = 0; v < 3; ++v) {
                GrMulti m;
                for (int k = 0; k < T; ++k)
                    m.a[k] = a[v][k];
                m.xn = set[v].xn;
                m.T = T;
                launch_multi(m, v + 1, st, nullptr, 0);
            }
            if (T == 1) fused_gr_read(a[3][0], st);
            try {
                st->wait();
            } catch (...) {
                why[0] = "a kernel of the check failed";
                ok = false;
                break;
            }
            if (ok_variant[2] && !all_same(set[0], set[1], T, apply, why[2])) ok_variant[2] = false;
            if (ok && ok_variant[3] && !all_same(set[0], set[2], T, apply, why[3])) ok_variant[3] = false;
            if (ok && T == 1 && single_ok && !all_same(set[3], set[0], T, apply, why[1])) single_ok = false;
        }
    }
    if (!single_ok && ok) {  // the plain read itself disagrees with the single-token read
        why[2] = why[3] = "the plain read differs from the single-token read: " + why[1];
        ok_variant[2] = ok_variant[3] = false;
    }
    st->wait();
    sycl::free(base, default_queue());
    if (!ok) ok_variant[2] = ok_variant[3] = false;
    return ok;
} catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

}  // namespace

int fused_gr_variant() {
    int dev = 0;
    const int v = dev >= 0 && dev < 64 ? g_variant[dev].load() : 0;
    if (v > 0) return v;
    // not checked on this card: the plain read, unless STRATA_HC_SPLIT names a variant (a test such as gr_parity)
    const char* e = std::getenv("STRATA_HC_SPLIT");
    return e != nullptr && (e[0] == '1' || e[0] == '2') ? env_variant() : kHcPlain;
}

void fused_gr_check() {
    int dev = 0;
    if (dev < 0 || dev >= 64 || g_variant[dev].load() > 0) return;
    const int want = env_variant();
    if (want == kHcPlain) {
        g_variant[dev].store(kHcPlain);
        std::fprintf(stderr, "strata hc: CUDA%d: the hyper-connection read runs as the plain one (STRATA_HC_SPLIT=0)\n",
                     dev);
        return;
    }
    bool okv[4];
    std::string why[4];
    const bool ran = fused_gr_selftest(okv, why);
    int use = kHcPlain;
    if (want >= kHcStaged && okv[kHcStaged])
        use = kHcStaged;
    else if (okv[kHcSplit])
        use = kHcSplit;
    g_variant[dev].store(use);
    if (!ran)
        std::fprintf(stderr, "strata hc: CUDA%d: the check of split/staged could not run (%s)\n", dev, why[0].c_str());
    static const char* const name[4] = {"", "plain", "split", "staged"};
    for (int v = kHcStaged; v >= kHcSplit; --v)
        if (v <= want && !okv[v] && ran)
            std::fprintf(stderr,
                         "strata hc: CUDA%d: the %s read differs from the plain read on this card - not used: "
                         "%s\n",
                         dev, name[v], why[v].c_str());
    static const char* const what[4] = {
        "", "the plain read (the norm per token, the down projection on 41 blocks)",
        "split (the norm per token and stream, then the plain read's down projection)",
        "staged (the norm per token and stream, the down projection's activations staged ahead by cp.async)"};
    std::fprintf(stderr, "strata hc: CUDA%d: the hyper-connection read runs as %s%s\n", dev, what[use],
                 use >= kHcSplit ? "; checked bit for bit against the plain read on this card (STRATA_HC_SPLIT=1 or 0 "
                                   "for the earlier ones)"
                                 : "");
}

void fused_gr_set_fast(int) {}

}  // namespace strata::kernels
