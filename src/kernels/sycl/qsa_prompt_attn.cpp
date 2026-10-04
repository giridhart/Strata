// src/kernels/sycl/qsa_prompt_attn.dp.cpp - Pure SYCL 2020 migration of qsa_prompt_attn
#include <sycl/sycl.hpp>
#include "strata/core/emulate.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>

namespace strata::kernels {
namespace {

constexpr int HD = 256;           // head_dim
constexpr int G = 12;             // query heads per KV head
#ifndef D1_CH
#define D1_CH 32
#endif
constexpr int CH = D1_CH;         // cells per chunk
constexpr int THREADS = 128;      // 4 warps: scores by cell (8 each), p.v by dimension (64 each = one int8 scale group)
constexpr int QS = HD + 8;        // q row stride in halves (bank-conflict-free fragment loads)

inline void mma16816(const sycl::sub_group& sg, float *c, const uint32_t *a, const uint32_t *b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    const int lane = sg.get_local_linear_id();
    const short row_load_offset = 4 * (lane >> 2);
    const short col_load_offset = 8 * (lane % 4);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t recv_a[4], recv_b[4];
        recv_a[0] = sycl::select_from_group(sg, a[0], row_load_offset + i);
        recv_a[1] = sycl::select_from_group(sg, a[2], row_load_offset + i);
        recv_a[2] = sycl::select_from_group(sg, a[1], row_load_offset + i);
        recv_a[3] = sycl::select_from_group(sg, a[3], row_load_offset + i);

        recv_b[0] = sycl::select_from_group(sg, b[0], col_load_offset + i);
        recv_b[1] = sycl::select_from_group(sg, b[1], col_load_offset + i);
        recv_b[2] = sycl::select_from_group(sg, b[0], col_load_offset + 4 + i);
        recv_b[3] = sycl::select_from_group(sg, b[1], col_load_offset + 4 + i);

        const auto* ra = reinterpret_cast<const sycl::half*>(recv_a);
        const auto* rb = reinterpret_cast<const sycl::half*>(recv_b);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            c[0] += static_cast<float>(ra[j]) * static_cast<float>(rb[j]);
            c[1] += static_cast<float>(ra[j]) * static_cast<float>(rb[j + 4]);
            c[2] += static_cast<float>(ra[j + 4]) * static_cast<float>(rb[j]);
            c[3] += static_cast<float>(ra[j + 4]) * static_cast<float>(rb[j + 4]);
        }
    }
#endif
}

// Two int8 codes (low byte first) as an exact half2: 1024 + (c + 128) built in the mantissa, minus 1152.
inline uint32_t i8x2_to_h2(uint32_t x) {
    uint32_t y = ((x & 0xffu) | ((x & 0xff00u) << 8)) ^ 0x00800080u;
    y |= 0x64006400u;
    sycl::half2 h = *reinterpret_cast<sycl::half2 *>(&y);
    h = h - sycl::half2(sycl::half(1152.0f), sycl::half(1152.0f));
    return *reinterpret_cast<uint32_t*>(&h);
}

inline uint32_t pack_h2(float lo_k, float hi_k) { // element k in the low half
    sycl::half2 h = sycl::float2(lo_k, hi_k).convert<sycl::half, sycl::rounding_mode::rte>();
    return *reinterpret_cast<uint32_t*>(&h);
}

// KV_MODE 1: int8 codes + fp16 scale per 64 values. KV_MODE 0: fp16 values (scales 1).
// KV_MODE 3 (hybrid K8V4): K as mode 1, V as mode 0 - the row's q4_0 blocks are dequantized to fp16 at
// gather, so everything downstream of the load is the mode-0 V path; the caller un-rotates the output.
template <int KV_MODE>
struct Smem {
    using KElem = typename std::conditional<KV_MODE == 0, sycl::half, int8_t>::type;
    using VElem = typename std::conditional<KV_MODE == 1, int8_t, sycl::half>::type;
    static constexpr int KROW = KV_MODE == 0 ? HD + 8 : HD + 16;   // elements; 16-byte aligned rows, banks spread
    static constexpr int VROW = KV_MODE == 1 ? HD + 16 : HD + 8;
    sycl::half qh[16][QS];
    sycl::half ql[16][QS];
    KElem k[CH][KROW];
    VElem v[CH][VROW];
    float ks[CH][4];
    float vs[CH][4];
    float s[16][CH + 1];
    float qmax[THREADS / 32];
    float alpha[16];
    float lsum[16];
    float mrow[16];
    long long row[CH];
};

template <int KV_MODE>
void prompt_attn_kernel(const float *__restrict__ q, QsaAttnPools p,
                        const int32_t *__restrict__ ids,
                        const int32_t *__restrict__ steps, int n_kv_heads,
                        int page_size, float scale_log2,
                        float *__restrict__ attn, int cap,
                        uint8_t *local_mem, sycl::nd_item<2> item) {
    auto sg = item.get_sub_group();
    Smem<KV_MODE>& S = *reinterpret_cast<Smem<KV_MODE>*>(local_mem);
    const int kvh = item.get_group(0);
    const int qi = item.get_group(1);
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    const int n = *(steps + (size_t)qi * kStepCount + kStepWidth);
    const int t = item.get_local_id(1), lane = t & 31, warp = t >> 5;
    const int gid = lane >> 2, tig = lane & 3;

    // q: 12 heads + 4 zero rows, scaled by a power of two that puts its largest value near 2^14 (exact, and the
    // lo halves stay out of FP16's subnormal range), then split into hi + lo halves
    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS)
        qm = sycl::fmax(qm, sycl::fabs(q[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        qm = sycl::fmax(qm, sycl::permute_group_by_xor(sg, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    item.barrier(sycl::access::fence_space::local_space);
    qm = sycl::fmax(sycl::fmax(S.qmax[0], S.qmax[1]),
                    sycl::fmax(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) sycl::frexp(qm, &qe);
    const float qup = sycl::ldexp(1.0f, 14 - qe),
                qdown = sycl::ldexp(scale_log2, qe - 14);
    for (int i = t; i < 16 * HD; i += THREADS) {
        const int h = i / HD, d = i % HD;
        const float x = h < G ? q[(size_t) h * HD + d] * qup : 0.0f;
        const sycl::half hi =
            sycl::vec<float, 1>(x)
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
        S.qh[h][d] = hi;
        S.ql[h][d] =
            sycl::vec<float, 1>(
                x - sycl::vec<sycl::half, 1>(hi)
                        .convert<float, sycl::rounding_mode::automatic>()[0])
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
    }
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }

    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;

    for (int c0 = 0; c0 < n; c0 += CH) {
        const int nh = sycl::min(CH, n - c0);
        if (t < CH) {
            long long r = -1;
            if (t < nh) {
                const int cell = ids[c0 + t];
                const long long page = (long long) p.page_table[cell / page_size];
                r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
            }
            S.row[t] = r;
        }
        item.barrier(sycl::access::fence_space::local_space);

        // gather the chunk's K and V rows (16-byte pieces; K8V4's V as q4_0 blocks dequantized to fp16)
        // and their scales
        {
            constexpr int KPIECES = HD * (int) sizeof(typename Smem<KV_MODE>::KElem) / 16;   // per K row
            for (int i = t; i < CH * KPIECES; i += THREADS) {
                const int c = i / KPIECES, pc = i % KPIECES;
                const long long r = S.row[c];
                sycl::uint4 kx = sycl::uint4(0, 0, 0, 0);
                if (r >= 0) {
                    if constexpr (KV_MODE == 0)
                        kx = *(reinterpret_cast<const sycl::uint4 *>(p.k_pool + r * HD) + pc);
                    else   // modes 1 and 3: the K side is INT8
                        kx = *(reinterpret_cast<const sycl::uint4 *>(p.k_q + r * HD) + pc);
                }
                *reinterpret_cast<sycl::uint4 *>(
                    reinterpret_cast<unsigned char *>(&S.k[c][0]) + pc * 16) = kx;
            }
            if constexpr (KV_MODE == 3) {   // V: dequantize the row's q4_0 blocks straight into the fp16 V row
                constexpr int BLKS = HD / QK4_0;
                constexpr int BYTES = BLKS * (int) sizeof(block_q4_0);
                for (int i = t; i < CH * BLKS; i += THREADS) {
                    const int c = i / BLKS, b = i % BLKS;
                    const long long r = S.row[c];
#pragma unroll
                    for (int j = 0; j < QK4_0; ++j)
                        S.v[c][b * QK4_0 + j] = sycl::half(0);
                    if (r >= 0) {
                        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + r * BYTES) + b;
                        const float d = static_cast<float>(sycl::bit_cast<sycl::half, unsigned short>(blk->d));
#pragma unroll
                        for (int j = 0; j < QK4_0 / 2; ++j) {
                            S.v[c][b * QK4_0 + j] =
                                sycl::vec<float, 1>(
                                    (float)((int)(blk->qs[j] & 0x0F) - 8) * d)
                                    .convert<sycl::half,
                                             sycl::rounding_mode::rte>()[0];
                            S.v[c][b * QK4_0 + j + QK4_0 / 2] =
                                sycl::vec<float, 1>(
                                    (float)((int)(blk->qs[j] >> 4) - 8) * d)
                                    .convert<sycl::half,
                                             sycl::rounding_mode::rte>()[0];
                        }
                    }
                }
            } else {
                constexpr int VPIECES = HD * (int) sizeof(typename Smem<KV_MODE>::VElem) / 16;   // per V row
                for (int i = t; i < CH * VPIECES; i += THREADS) {
                    const int c = i / VPIECES, pc = i % VPIECES;
                    const long long r = S.row[c];
                    sycl::uint4 vx = sycl::uint4(0, 0, 0, 0);
                    if (r >= 0) {
                        if constexpr (KV_MODE == 1)
                            vx = *(reinterpret_cast<const sycl::uint4 *>(p.v_q + r * HD) + pc);
                        else
                            vx = *(reinterpret_cast<const sycl::uint4 *>(p.v_pool + r * HD) + pc);
                    }
                    *reinterpret_cast<sycl::uint4 *>(
                        reinterpret_cast<unsigned char *>(&S.v[c][0]) + pc * 16) = vx;
                }
            }
            for (int i = t; i < CH * 4; i += THREADS) {
                const int c = i / 4, g = i % 4;
                const long long r = S.row[c];
                float a = 0.0f, b = 0.0f;
                if (r >= 0) {
                    if constexpr (KV_MODE == 1) {
                        a = static_cast<float>(sycl::bit_cast<sycl::half, unsigned short>(p.k_scale[r * (HD / KV_Q8_GROUP) + g]));
                        b = static_cast<float>(sycl::bit_cast<sycl::half, unsigned short>(p.v_scale[r * (HD / KV_Q8_GROUP) + g]));
                    } else if constexpr (KV_MODE == 3) {   // K as int8, V dequantized to fp16 (scale 1)
                        a = static_cast<float>(sycl::bit_cast<sycl::half, unsigned short>(p.k_scale[r * (HD / KV_Q8_GROUP) + g]));
                        b = 1.0f;
                    } else {
                        a = b = 1.0f;
                    }
                }
                S.ks[c][g] = a;
                S.vs[c][g] = b;
            }
        }
        item.barrier(sycl::access::fence_space::local_space);

        // scores: warp w takes cells 8w..8w+7 (one n-tile) over all 256 dims, per 64-dim scale group
#pragma unroll
        for (int nt = 0; nt < CH / 32; ++nt) {
            const int cb = (warp + 4 * nt) * 8;
            float sc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                float tg[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const int k0 = (g * 4 + kk) * 16;
                    uint32_t ah[4], al[4], b[2];
                    ah[0] = *reinterpret_cast<const uint32_t*>(&S.qh[gid][k0 + 2 * tig]);
                    ah[1] = *reinterpret_cast<const uint32_t*>(&S.qh[gid + 8][k0 + 2 * tig]);
                    ah[2] = *reinterpret_cast<const uint32_t*>(&S.qh[gid][k0 + 2 * tig + 8]);
                    ah[3] = *reinterpret_cast<const uint32_t*>(&S.qh[gid + 8][k0 + 2 * tig + 8]);
                    al[0] = *reinterpret_cast<const uint32_t*>(&S.ql[gid][k0 + 2 * tig]);
                    al[1] = *reinterpret_cast<const uint32_t*>(&S.ql[gid + 8][k0 + 2 * tig]);
                    al[2] = *reinterpret_cast<const uint32_t*>(&S.ql[gid][k0 + 2 * tig + 8]);
                    al[3] = *reinterpret_cast<const uint32_t*>(&S.ql[gid + 8][k0 + 2 * tig + 8]);
                    if constexpr (KV_MODE != 0) {   // modes 1 and 3: the K side is INT8 codes
                        b[0] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(&S.k[cb + gid][k0 + 2 * tig]));
                        b[1] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(&S.k[cb + gid][k0 + 2 * tig + 8]));
                    } else {
                        b[0] = *reinterpret_cast<const uint32_t*>(&S.k[cb + gid][k0 + 2 * tig]);
                        b[1] = *reinterpret_cast<const uint32_t*>(&S.k[cb + gid][k0 + 2 * tig + 8]);
                    }
                    mma16816(sg, tg, ah, b);
#ifndef D1_NO_QLO
                    mma16816(sg, tg, al, b);
#endif
                }
                const float s0 = S.ks[cb + 2 * tig][g], s1 = S.ks[cb + 2 * tig + 1][g];
                sc[0] = sycl::fma(tg[0], s0, sc[0]);
                sc[1] = sycl::fma(tg[1], s1, sc[1]);
                sc[2] = sycl::fma(tg[2], s0, sc[2]);
                sc[3] = sycl::fma(tg[3], s1, sc[3]);
            }
            const int c = cb + 2 * tig;
            S.s[gid][c] = c < nh ? sc[0] * qdown : -INFINITY;
            S.s[gid][c + 1] = c + 1 < nh ? sc[1] * qdown : -INFINITY;
            S.s[gid + 8][c] = c < nh ? sc[2] * qdown : -INFINITY;
            S.s[gid + 8][c + 1] = c + 1 < nh ? sc[3] * qdown : -INFINITY;
        }
        item.barrier(sycl::access::fence_space::local_space);

        // online softmax: row t/8, 4 cells per thread, 8 threads per row (lanes 8r..8r+7 of a warp)
        {
            constexpr int PER = CH / 8;
            const int r = t >> 3, sub = t & 7;
            float x[PER], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                x[j] = S.s[r][sub * PER + j]; mx = sycl::fmax(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1)
                mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = sycl::fmax(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : sycl::exp2(x[j] - m_new);
                S.s[r][sub * PER + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1)
                sum += sycl::permute_group_by_xor(sg, sum, o);
            sycl::group_barrier(sg);
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : sycl::exp2(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = sycl::fma(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        item.barrier(sycl::access::fence_space::local_space);

        // p.v: warp w owns dims [64w, 64w+64), which is int8 scale group w. The scale is folded into p relative to
        // the chunk's largest, times 2^14 (p' <= 2^14: its lo half stays out of FP16's subnormal range); the chunk's
        // sum is then added to the running one in FP32 with the factor taken back out
        {
            float vmax = 0.0f;
#pragma unroll
            for (int c = lane; c < CH; c += 32)
                vmax = sycl::fmax(vmax, S.vs[c][warp]);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                vmax = sycl::fmax(vmax, sycl::permute_group_by_xor(sg, vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
            float tmp[8][4];
#pragma unroll
            for (int j = 0; j < 8; ++j) tmp[j][0] = tmp[j][1] = tmp[j][2] = tmp[j][3] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < CH / 16; ++ks) {
                const int cA = ks * 16 + 2 * tig, cB = cA + 8;
                const float w0 = S.vs[cA][warp] * vup, w1 = S.vs[cA + 1][warp] * vup, w2 = S.vs[cB][warp] * vup,
                            w3 = S.vs[cB + 1][warp] * vup;
                const float p00 = S.s[gid][cA] * w0, p01 = S.s[gid][cA + 1] * w1;
                const float p10 = S.s[gid + 8][cA] * w0, p11 = S.s[gid + 8][cA + 1] * w1;
                const float p02 = S.s[gid][cB] * w2, p03 = S.s[gid][cB + 1] * w3;
                const float p12 = S.s[gid + 8][cB] * w2, p13 = S.s[gid + 8][cB + 1] * w3;
                uint32_t ah[4], al[4];
                ah[0] = pack_h2(p00, p01);
                ah[1] = pack_h2(p10, p11);
                ah[2] = pack_h2(p02, p03);
                ah[3] = pack_h2(p12, p13);
                {
                    const sycl::half2 *h = reinterpret_cast<const sycl::half2 *>(ah);
                    sycl::float2 f;
                    f = h[0].template convert<float, sycl::rounding_mode::automatic>();
                    al[0] = pack_h2(p00 - f.x(), p01 - f.y());
                    f = h[1].template convert<float, sycl::rounding_mode::automatic>();
                    al[1] = pack_h2(p10 - f.x(), p11 - f.y());
                    f = h[2].template convert<float, sycl::rounding_mode::automatic>();
                    al[2] = pack_h2(p02 - f.x(), p03 - f.y());
                    f = h[3].template convert<float, sycl::rounding_mode::automatic>();
                    al[3] = pack_h2(p12 - f.x(), p13 - f.y());
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int d = warp * 64 + j * 8 + gid;
                    uint32_t b[2];
                    if constexpr (KV_MODE == 1) {
                        const uint32_t x0 = (uint8_t) S.v[cA][d] | ((uint32_t) (uint8_t) S.v[cA + 1][d] << 8);
                        const uint32_t x1 = (uint8_t) S.v[cB][d] | ((uint32_t) (uint8_t) S.v[cB + 1][d] << 8);
                        b[0] = i8x2_to_h2(x0);
                        b[1] = i8x2_to_h2(x1);
                    } else {
                        const sycl::half2 h0 = sycl::half2(S.v[cA][d], S.v[cA + 1][d]);
                        const sycl::half2 h1 = sycl::half2(S.v[cB][d], S.v[cB + 1][d]);
                        b[0] = *reinterpret_cast<const uint32_t*>(&h0);
                        b[1] = *reinterpret_cast<const uint32_t*>(&h1);
                    }
                    mma16816(sg, tmp[j], ah, b);
#ifndef D1_NO_PLO
                    mma16816(sg, tmp[j], al, b);
#endif
                }
            }
            const float a0 = S.alpha[gid], a1 = S.alpha[gid + 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j][0] = sycl::fma(acc[j][0], a0, tmp[j][0] * vdown);
                acc[j][1] = sycl::fma(acc[j][1], a0, tmp[j][1] * vdown);
                acc[j][2] = sycl::fma(acc[j][2], a1, tmp[j][2] * vdown);
                acc[j][3] = sycl::fma(acc[j][3], a1, tmp[j][3] * vdown);
            }
        }
    }
    item.barrier(sycl::access::fence_space::local_space);
    const float l0 = S.lsum[gid], l1 = S.lsum[gid + 8];
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = warp * 64 + j * 8 + 2 * tig;
        *reinterpret_cast<sycl::float2 *>(attn + (size_t)gid * HD + d) =
            sycl::float2(acc[j][0] * i0, acc[j][1] * i0);
        if (gid + 8 < G)
            *reinterpret_cast<sycl::float2 *>(attn + (size_t)(gid + 8) * HD + d) =
                sycl::float2(acc[j][2] * i1, acc[j][3] * i1);
    }
}

// ---- v2 (int8 KV): warp w owns dims [64w, 64w+64) for both q.k and p.v, which is also int8 scale group w. So a
// warp needs only its own 64-byte slice of each K and V row: it gathers it itself with cp.async into its own
// double-buffered stage while it computes the previous chunk, and q stays in registers. Only the q.k partial sums
// (one per dim group) cross warps, and they are added in a fixed order: the result is deterministic.
constexpr int CH2 = 32;

struct Smem2 {
    int8_t kv[2][4][2][CH2][64];   // stage, warp, K/V, cell, 64 dims in 16-byte pieces XOR-swizzled by the cell
    float sc[2][4][2][CH2];        // stage, warp, K/V scale of the cell for the warp's group
    float part[4][16][CH2 + 1];    // q.k per dim group
    float p[16][CH2 + 1];
    float qmax[4];
    float alpha[16];
    float lsum[16];
    float mrow[16];
};

inline int swz(int cell, int byte) { // byte offset of (cell, byte) in a stage slice
    return cell * 64 + ((((byte >> 4) ^ (cell >> 1)) & 3) << 4) + (byte & 15);
}
inline void cp_async16(void *smem, const void *gmem, bool valid) {
    if (valid) *reinterpret_cast<sycl::float4*>(smem) = *reinterpret_cast<const sycl::float4*>(gmem);
}
inline void cp_async_commit() {}
inline void cp_async_wait1() {}

void prompt_attn_i8_kernel(const float *__restrict__ q, QsaAttnPools p,
                           const int32_t *__restrict__ ids,
                           const int32_t *__restrict__ steps, int n_kv_heads,
                           int page_size, float scale_log2,
                           float *__restrict__ attn, int cap,
                           uint8_t *local_mem, sycl::nd_item<2> item) {
    auto sg = item.get_sub_group();
    Smem2& S = *reinterpret_cast<Smem2*>(local_mem);
    const int kvh = item.get_group(0);
    const int qi = item.get_group(1);
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    const int n = *(steps + (size_t)qi * kStepCount + kStepWidth);
    const int t = item.get_local_id(1), lane = t & 31, warp = t >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    const int dim0 = warp * 64;

    // q: the power-of-two prescale over all 12 heads (as v1), then this warp's 64 dims as hi/lo A fragments
    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS)
        qm = sycl::fmax(qm, sycl::fabs(q[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        qm = sycl::fmax(qm, sycl::permute_group_by_xor(sg, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }
    item.barrier(sycl::access::fence_space::local_space);
    qm = sycl::fmax(sycl::fmax(S.qmax[0], S.qmax[1]),
                    sycl::fmax(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) sycl::frexp(qm, &qe);
    const float qup = sycl::ldexp(1.0f, 14 - qe),
                qdown = sycl::ldexp(scale_log2, qe - 14);
    uint32_t qh[4][4], ql[4][4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = gid + (r & 1) * 8, col = dim0 + kk * 16 + 2 * tig + (r >> 1) * 8;
            sycl::float2 x = sycl::float2(0.f, 0.f);
            if (row < G) x = *reinterpret_cast<const sycl::float2 *>(q + (size_t)row * HD + col);
            x.x() *= qup;
            x.y() *= qup;
            const sycl::half2 hi = sycl::float2(x.x(), x.y()).convert<sycl::half, sycl::rounding_mode::rte>();
            const sycl::float2 hf = hi.template convert<float, sycl::rounding_mode::automatic>();
            const sycl::half2 lo = sycl::float2(x.x() - hf.x(), x.y() - hf.y()).convert<sycl::half, sycl::rounding_mode::rte>();
            qh[kk][r] = *reinterpret_cast<const uint32_t*>(&hi);
            ql[kk][r] = *reinterpret_cast<const uint32_t*>(&lo);
        }
    }

    // the chunk pipeline: cells two chunks ahead, their pool rows one chunk ahead, the data (cp.async) one ahead
    const int n_chunks = (n + CH2 - 1) / CH2;
    auto cell_of = [&](int c) -> int {
        return c < n ? *(ids + c) : -1;
    };
    auto row_of = [&](int cell) -> long long {
        if (cell < 0) return -1;
        const long long page = (long long)*(p.page_table + cell / page_size);
        return (page * n_kv_heads + kvh) * page_size + (cell % page_size);
    };
    auto issue = [&](long long r, int st, float& ksr, float& vsr) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = lane + 32 * j, cell = idx >> 2, pc = idx & 3;
            const long long rr = sycl::select_from_group(sg, r, cell);
            const bool ok = rr >= 0;
            const size_t off = ok ? (size_t) rr * HD + dim0 + pc * 16 : 0;
            cp_async16(&S.kv[st][warp][0][0][0] + swz(cell, pc * 16), p.k_q + off, ok);
            cp_async16(&S.kv[st][warp][1][0][0] + swz(cell, pc * 16), p.v_q + off, ok);
        }
        ksr = r >= 0 ? static_cast<float>(sycl::bit_cast<sycl::half, unsigned short>(*(p.k_scale + r * (HD / KV_Q8_GROUP) + warp))) : 0.0f;
        vsr = r >= 0 ? static_cast<float>(sycl::bit_cast<sycl::half, unsigned short>(*(p.v_scale + r * (HD / KV_Q8_GROUP) + warp))) : 0.0f;
    };
    float ksn, vsn;
    issue(row_of(cell_of(lane)), 0, ksn, vsn);
    cp_async_commit();
    S.sc[0][warp][0][lane] = ksn;
    S.sc[0][warp][1][lane] = vsn;
    long long r_next = row_of(cell_of(CH2 + lane));
    int cell_next2 = cell_of(2 * CH2 + lane);

    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;

    for (int ci = 0; ci < n_chunks; ++ci) {
        const int st = ci & 1, c0 = ci * CH2;
        const bool more = ci + 1 < n_chunks;
        if (more) issue(r_next, st ^ 1, ksn, vsn);
        cp_async_commit();
        r_next = row_of(cell_next2);
        cell_next2 = cell_of((ci + 3) * CH2 + lane);
        cp_async_wait1();
        sycl::group_barrier(sg);
        const int8_t* K = &S.kv[st][warp][0][0][0];
        const int8_t* V = &S.kv[st][warp][1][0][0];
        // q.k over this warp's 64 dims, times the cell's K scale for this group
#pragma unroll
        for (int nt = 0; nt < CH2 / 8; ++nt) {
            float tg[4] = {0.f, 0.f, 0.f, 0.f};
            const int cell = nt * 8 + gid;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                uint32_t b[2];
                b[0] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(K + swz(cell, kk * 16 + 2 * tig)));
                b[1] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(K + swz(cell, kk * 16 + 2 * tig + 8)));
                mma16816(sg, tg, qh[kk], b);
                mma16816(sg, tg, ql[kk], b);
            }
            const int c = nt * 8 + 2 * tig;
            const float s0 = S.sc[st][warp][0][c], s1 = S.sc[st][warp][0][c + 1];
            S.part[warp][gid][c] = tg[0] * s0;
            S.part[warp][gid][c + 1] = tg[1] * s1;
            S.part[warp][gid + 8][c] = tg[2] * s0;
            S.part[warp][gid + 8][c + 1] = tg[3] * s1;
        }
        item.barrier(sycl::access::fence_space::local_space);
        // online softmax over the four groups' sum (fixed order): row t/8, 4 cells per thread
        {
            const int r = t >> 3, sub = t & 7;
            float x[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                x[j] = c0 + c < n ? (((S.part[0][r][c] + S.part[1][r][c]) + S.part[2][r][c]) + S.part[3][r][c]) * qdown
                                  : -INFINITY;
                mx = sycl::fmax(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1)
                mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = sycl::fmax(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : sycl::exp2(x[j] - m_new);
                S.p[r][sub * 4 + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1)
                sum += sycl::permute_group_by_xor(sg, sum, o);
            sycl::group_barrier(sg);
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : sycl::exp2(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = sycl::fma(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        item.barrier(sycl::access::fence_space::local_space);
        // p.v over this warp's 64 dims (as v1)
        {
            float vmax = S.sc[st][warp][1][lane];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                vmax = sycl::fmax(vmax, sycl::permute_group_by_xor(sg, vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
            float tmp[8][4];
#pragma unroll
            for (int j = 0; j < 8; ++j) tmp[j][0] = tmp[j][1] = tmp[j][2] = tmp[j][3] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < CH2 / 16; ++ks) {
                const int cA = ks * 16 + 2 * tig, cB = cA + 8;
                const float w0 = S.sc[st][warp][1][cA] * vup, w1 = S.sc[st][warp][1][cA + 1] * vup,
                            w2 = S.sc[st][warp][1][cB] * vup, w3 = S.sc[st][warp][1][cB + 1] * vup;
                const float p00 = S.p[gid][cA] * w0, p01 = S.p[gid][cA + 1] * w1;
                const float p10 = S.p[gid + 8][cA] * w0, p11 = S.p[gid + 8][cA + 1] * w1;
                const float p02 = S.p[gid][cB] * w2, p03 = S.p[gid][cB + 1] * w3;
                const float p12 = S.p[gid + 8][cB] * w2, p13 = S.p[gid + 8][cB + 1] * w3;
                uint32_t ah[4], al[4];
                ah[0] = pack_h2(p00, p01);
                ah[1] = pack_h2(p10, p11);
                ah[2] = pack_h2(p02, p03);
                ah[3] = pack_h2(p12, p13);
                {
                    const sycl::half2 *h = reinterpret_cast<const sycl::half2 *>(ah);
                    sycl::float2 f;
                    f = h[0].template convert<float, sycl::rounding_mode::automatic>();
                    al[0] = pack_h2(p00 - f.x(), p01 - f.y());
                    f = h[1].template convert<float, sycl::rounding_mode::automatic>();
                    al[1] = pack_h2(p10 - f.x(), p11 - f.y());
                    f = h[2].template convert<float, sycl::rounding_mode::automatic>();
                    al[2] = pack_h2(p02 - f.x(), p03 - f.y());
                    f = h[3].template convert<float, sycl::rounding_mode::automatic>();
                    al[3] = pack_h2(p12 - f.x(), p13 - f.y());
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int d = j * 8 + gid;
                    const uint32_t x0 = (uint8_t) V[swz(cA, d)] | ((uint32_t) (uint8_t) V[swz(cA + 1, d)] << 8);
                    const uint32_t x1 = (uint8_t) V[swz(cB, d)] | ((uint32_t) (uint8_t) V[swz(cB + 1, d)] << 8);
                    uint32_t b[2];
                    b[0] = i8x2_to_h2(x0);
                    b[1] = i8x2_to_h2(x1);
                    mma16816(sg, tmp[j], ah, b);
                    mma16816(sg, tmp[j], al, b);
                }
            }
            const float a0 = S.alpha[gid], a1 = S.alpha[gid + 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j][0] = sycl::fma(acc[j][0], a0, tmp[j][0] * vdown);
                acc[j][1] = sycl::fma(acc[j][1], a0, tmp[j][1] * vdown);
                acc[j][2] = sycl::fma(acc[j][2], a1, tmp[j][2] * vdown);
                acc[j][3] = sycl::fma(acc[j][3], a1, tmp[j][3] * vdown);
            }
        }
        if (more) {
            S.sc[st ^ 1][warp][0][lane] = ksn;
            S.sc[st ^ 1][warp][1][lane] = vsn;
        }
        sycl::group_barrier(sg);
    }
    item.barrier(sycl::access::fence_space::local_space);
    const float l0 = S.lsum[gid], l1 = S.lsum[gid + 8];
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = dim0 + j * 8 + 2 * tig;
        *reinterpret_cast<sycl::float2 *>(attn + (size_t)gid * HD + d) =
            sycl::float2(acc[j][0] * i0, acc[j][1] * i0);
        if (gid + 8 < G)
            *reinterpret_cast<sycl::float2 *>(attn + (size_t)(gid + 8) * HD + d) =
                sycl::float2(acc[j][2] * i1, acc[j][3] * i1);
    }
}

bool launch_i8(const float *q, const QsaAttnPools &pools, const int32_t *ids,
               const int32_t *steps, int64_t cap, const QsaShapes &s,
               float *attn, int64_t n_q, sycl::queue *q_ptr) {
    if (!q_ptr->get_device().has(sycl::aspect::fp16)) return false;
    const size_t bytes = sizeof(Smem2);
    if (q_ptr->get_device().get_info<sycl::info::device::local_mem_size>() < bytes) return false;

    const float scale_log2 = 1.4426950408889634f / std::sqrt((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        auto q_sub = q + q0 * s.n_head * HD;
        auto ids_sub = ids + q0 * cap;
        auto steps_sub = steps + q0 * kStepCount;
        auto attn_sub = attn + q0 * s.n_head * HD;

        q_ptr->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> local_acc(sycl::range<1>(bytes), cgh);

            cgh.parallel_for(
                sycl::nd_range<2>(
                    sycl::range<2>(s.n_head_kv, nb * THREADS),
                    sycl::range<2>(1, THREADS)),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
                    prompt_attn_i8_kernel(
                        q_sub, pools, ids_sub, steps_sub, (int)s.n_head_kv,
                        (int)s.page_size, scale_log2, attn_sub, (int)cap,
                        local_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                        item);
                });
        });
    }
    return true;
}

template <int KV_MODE>
bool launch(const float *q, const QsaAttnPools &pools, const int32_t *ids,
            const int32_t *steps, int64_t cap, const QsaShapes &s, float *attn,
            int64_t n_q, sycl::queue *q_ptr) {
    if (!q_ptr->get_device().has(sycl::aspect::fp16)) return false;
    const size_t bytes = sizeof(Smem<KV_MODE>);
    if (q_ptr->get_device().get_info<sycl::info::device::local_mem_size>() < bytes) return false;

    const float scale_log2 = 1.4426950408889634f / std::sqrt((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        auto q_sub = q + q0 * s.n_head * HD;
        auto ids_sub = ids + q0 * cap;
        auto steps_sub = steps + q0 * kStepCount;
        auto attn_sub = attn + q0 * s.n_head * HD;

        q_ptr->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> local_acc(sycl::range<1>(bytes), cgh);

            cgh.parallel_for(
                sycl::nd_range<2>(
                    sycl::range<2>(s.n_head_kv, nb * THREADS),
                    sycl::range<2>(1, THREADS)),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
                    prompt_attn_kernel<KV_MODE>(
                        q_sub, pools, ids_sub, steps_sub, (int)s.n_head_kv,
                        (int)s.page_size, scale_log2, attn_sub, (int)cap,
                        local_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                        item);
                });
        });
    }
    return true;
}

inline int get_device_cc(const sycl::device& dev) {
    const char* w = std::getenv("STRATA_QSA_WARP");
    if (w && (!std::strcmp(w, "1") || !std::strcmp(w, "attn"))) return 75;
    
    std::string ver = dev.get_info<sycl::info::device::version>();
    int major = 0, minor = 0;
    size_t i = 0;
    while (i < ver.size() && !std::isdigit(static_cast<unsigned char>(ver[i]))) ++i;
    if (i < ver.size()) {
        major = std::atoi(&ver[i]);
        while (i < ver.size() && std::isdigit(static_cast<unsigned char>(ver[i]))) ++i;
        if (i < ver.size() && ver[i] == '.') {
            minor = std::atoi(&ver[i + 1]);
        }
    }
    return 10 * strata::cc_major_of(major) + strata::cc_minor_of(minor);
}

#if defined(__HIPCC__)
#if defined(__gfx1200__) || defined(__gfx1201__)
#define STRATA_PA_WMMA 1
#else
#define STRATA_PA_WMMA 0
#endif
typedef _Float16 wh8 __attribute__((ext_vector_type(8)));
typedef float wf8 __attribute__((ext_vector_type(8)));
constexpr int WCH = 32;
constexpr int WVS = 80;
struct alignas(16) SmemW {
    float part[4][16][WCH + 1];
    float p[16][WCH + 1];
    float vs[4][WCH];
    int valid[WCH];
    uint8_t v[4][WCH][WVS];
    float qmax[4];
    float alpha[16];
    float lsum[16];
    float mrow[16];
};

__device__ __forceinline__ wf8 wmma_f16(wh8 a, wh8 b, wf8 c) {
#if STRATA_PA_WMMA
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#else
    __builtin_trap();
    return c;
#endif
}

__device__ __forceinline__ wh8 i8x8_to_h8(uint2 x) {
    wh8 h;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        h[i] = (_Float16) (int) (int8_t) (x.x >> (8 * i));
        h[4 + i] = (_Float16) (int) (int8_t) (x.y >> (8 * i));
    }
    return h;
}

__global__ void __launch_bounds__(THREADS) prompt_attn_wmma_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                   const int32_t* __restrict__ ids,
                                                                   const int32_t* __restrict__ steps, int n_kv_heads,
                                                                   int page_size, float scale_log2,
                                                                   float* __restrict__ attn, int cap) {
#if STRATA_PA_WMMA
    __shared__ SmemW S;
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    const int n = __ldg(steps + (size_t) qi * kStepCount + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int col = lane & 15, half = lane >> 4;
    const int dim0 = warp * 64;

    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) qm = fmaxf(qm, fabsf(q[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale_log2, qe - 14);
    wh8 qh[4], ql[4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        float x[8];
        if (col < G) {
            const float4* src = reinterpret_cast<const float4*>(q + (size_t) col * HD + dim0 + kk * 16 + half * 8);
            const float4 a = src[0], b = src[1];
            x[0] = a.x; x[1] = a.y; x[2] = a.z; x[3] = a.w; x[4] = b.x; x[5] = b.y; x[6] = b.z; x[7] = b.w;
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) x[i] = 0.0f;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float v = x[i] * qup;
            const _Float16 hi = (_Float16) v;
            qh[kk][i] = hi;
            ql[kk][i] = (_Float16) (v - (float) hi);
        }
    }

    wf8 acc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j] = wf8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int n_chunks = (n + WCH - 1) / WCH;
    for (int ci = 0; ci < n_chunks; ++ci) {
        const int c0 = ci * WCH;
        long long myrow = -1;
        if (c0 + lane < n) {
            const int cell = __ldg(ids + c0 + lane);
            const long long page = (long long) __ldg(p.page_table + cell / page_size);
            if (page >= 0) myrow = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        {
            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0, v2 = v0, v3 = v0;
            float vsc = 0.0f;
            if (myrow >= 0) {
                const uint4* src = reinterpret_cast<const uint4*>(p.v_q + myrow * HD + dim0);
                v0 = __ldg(src); v1 = __ldg(src + 1); v2 = __ldg(src + 2); v3 = __ldg(src + 3);
                vsc = __half2float(__ushort_as_half(__ldg(p.v_scale + myrow * (HD / KV_Q8_GROUP) + warp)));
            }
            uint4* dst = reinterpret_cast<uint4*>(&S.v[warp][lane][0]);
            dst[0] = v0; dst[1] = v1; dst[2] = v2; dst[3] = v3;
            S.vs[warp][lane] = vsc;
            if (warp == 0) S.valid[lane] = myrow >= 0;
        }
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const long long rr = __shfl(myrow, nt * 16 + col);
            wf8 s = wf8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            float ksc = 0.0f;
            if (rr >= 0) ksc = __half2float(__ushort_as_half(__ldg(p.k_scale + rr * (HD / KV_Q8_GROUP) + warp)));
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                uint2 kx = make_uint2(0, 0);
                if (rr >= 0) kx = __ldg(reinterpret_cast<const uint2*>(p.k_q + rr * HD + dim0 + kk * 16 + half * 8));
                const wh8 b = i8x8_to_h8(kx);
                s = wmma_f16(qh[kk], b, s);
                s = wmma_f16(ql[kk], b, s);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) S.part[warp][half * 8 + i][nt * 16 + col] = s[i] * ksc;
        }
        __syncthreads();
        {
            const int r = t >> 3, sub = t & 7;
            float x[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                x[j] = S.valid[c] ? (((S.part[0][r][c] + S.part[1][r][c]) + S.part[2][r][c]) + S.part[3][r][c]) * qdown
                                  : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.p[r][sub * 4 + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();
        {
            float vmax = S.vs[warp][lane];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
            wf8 tmp[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) tmp[j] = wf8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                const int cb = ks * 16 + half * 8;
                wh8 ah, al;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float pv = S.p[col][cb + i] * (S.vs[warp][cb + i] * vup);
                    const _Float16 hi = (_Float16) pv;
                    ah[i] = hi;
                    al[i] = (_Float16) (pv - (float) hi);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int d = j * 16 + col;
                    wh8 b;
#pragma unroll
                    for (int i = 0; i < 8; ++i) b[i] = (_Float16) (int) (int8_t) S.v[warp][cb + i][d];
                    tmp[j] = wmma_f16(ah, b, tmp[j]);
                    tmp[j] = wmma_f16(al, b, tmp[j]);
                }
            }
            float a[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) a[i] = S.alpha[half * 8 + i];
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[j][i] = fmaf(acc[j][i], a[i], tmp[j][i] * vdown);
        }
    }
    float inv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float l = S.lsum[half * 8 + i];
        inv[i] = l > 0.0f ? 1.0f / l : 0.0f;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = half * 8 + i;
            if (row < G) attn[(size_t) row * HD + dim0 + j * 16 + col] = acc[j][i] * inv[i];
        }
#else
    __builtin_trap();
#endif
}

bool hip_wmma_usable() {
    static const bool want = [] {
        const char* e = std::getenv("STRATA_HIP_WMMA");
        return e != nullptr && e[0] == '1';
    }();
    if (!want) return false;
    static int arch[64] = {};
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) { (void) hipGetLastError(); return false; }
    if (arch[dev] == 0) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, dev) != hipSuccess) { (void) hipGetLastError(); return false; }
        arch[dev] = std::strncmp(prop.gcnArchName, "gfx12", 5) == 0 ? 1 : 2;
        static bool told = false;
        if (!told) {
            told = true;
            std::fprintf(stderr, "strata: STRATA_HIP_WMMA: the prompt attention on matrix cores %s (%s)\n",
                         arch[dev] == 1 ? "on" : "unavailable", prop.gcnArchName);
        }
    }
    return arch[dev] == 1;
}

bool launch_wmma(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                 const QsaShapes& s, float* attn, int64_t n_q, cudaStream_t st) {
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        prompt_attn_wmma_kernel<<<dim3((unsigned) nb, (unsigned) s.n_head_kv), THREADS, 0, st>>>(
            q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount, (int) s.n_head_kv,
            (int) s.page_size, scale_log2, attn + q0 * s.n_head * HD, (int) cap);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_prompt_attn_batch (wmma): %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}
#endif

}  // namespace

bool qsa_prompt_attn_batch(const float *q, const QsaAttnPools &pools,
                           const int32_t *ids, const int32_t *steps,
                           int64_t cap, const QsaShapes &s, float *attn,
                           int64_t n_q, void *stream) {
    if (n_q <= 0) return true;
    static sycl::queue default_queue{sycl::property::queue::in_order()};
    sycl::queue* q_ptr = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    bool turing = false;
    {
        const int cc = get_device_cc(q_ptr->get_device());
        if (cc < 75) return false;
        turing = cc < 80;
    }
#if defined(__HIPCC__)
    (void) turing;
    if (pools.k_q != nullptr && pools.v_q != nullptr && pools.k_scale != nullptr && pools.v_scale != nullptr &&
        pools.k_q4 == nullptr && pools.v_q4 == nullptr && s.head_dim == HD && s.n_head == (int64_t) G * s.n_head_kv &&
        cap > 0 && ids && steps && pools.page_table && hip_wmma_usable())
        return launch_wmma(q, pools, ids, steps, cap, s, attn, n_q, (cudaStream_t) stream);
    return false;
#endif
    if (pools.k_q4 != nullptr || s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !ids ||
        !steps || !pools.page_table)
        return false;

    if (pools.k_q != nullptr && pools.v_q4 != nullptr) {   // hybrid K8V4: int8 K + dequantized-q4 V
        if (!pools.k_scale) return false;
        return launch<3>(q, pools, ids, steps, cap, s, attn, n_q, q_ptr);
    }
    if (pools.k_q != nullptr) {
        if (!pools.v_q || !pools.k_scale || !pools.v_scale) return false;
        static const bool v1 = std::getenv("STRATA_PROMPT_ATTN_V1") != nullptr;
        if (v1 || turing) return launch<1>(q, pools, ids, steps, cap, s, attn, n_q, q_ptr);
        return launch_i8(q, pools, ids, steps, cap, s, attn, n_q, q_ptr);
    }
    if (!pools.k_pool || !pools.v_pool) return false;
    return launch<0>(q, pools, ids, steps, cap, s, attn, n_q, q_ptr);
}

}  // namespace strata::kernels
