// src/kernels/sycl/qsa_prompt_attn.cpp - Pure SYCL 2020 Hardware XMX Systolic Attention Kernel
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/matrix/matrix-unified.hpp>
#include "strata/core/emulate.hpp"
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/qsa_prompt_attn_esimd.hpp"
#include "strata/kernels/xmx_gemm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace strata::kernels {
namespace {

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace ix = sycl::ext::intel::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
using sycl::half;

constexpr int HD = 256;           // head_dim
constexpr int G = 12;             // query heads per KV head (+ 4 zero rows: two 8-row tiles)
constexpr int R = 16;             // rows of q, p and the output
constexpr int NG = 4;             // 64-dim groups
constexpr int SGS = 16;           // sub-group size
constexpr int NSG = 2 * NG;       // sub-groups: a group's dims x a tile of 8 rows
constexpr int WG = NSG * SGS;     // work-items (128)
constexpr int QS = HD + 8;        // row strides in halves (16-byte aligned rows)
constexpr int KR = HD + 8;
[[maybe_unused]] constexpr int TM = 8, TN = 16, TK = 16;
[[maybe_unused]] constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

// Lazy rescaling (FlashAttention-3): a row's softmax reference moves only when a score exceeds it by more than
// 2^TAU, so p <= 2^TAU, and p' = p * vscale / (largest V scale so far) * 2^(14 - TAU) stays within FP16. Most chunks
// then change no reference and leave the output unscaled.
[[maybe_unused]] constexpr float TAU = 8.0f;
[[maybe_unused]] constexpr float PSCALE = 16384.0f / 256.0f;   // 2^(14 - TAU)

// a local accessor's element `off` (row-major) as the multi_ptr joint_matrix loads from and stores to
template <typename A>
inline auto at(const A& a, size_t off) {
    return a.template get_multi_ptr<sycl::access::decorated::no>() + off;
}

// how a pool holds K or V: FP16 values (scales 1), INT8 codes with an FP16 scale per 64 values, or Q4_0 blocks (codes
// 0-15 minus 8, exact in FP16, with a signed FP16 scale per 32 values; kv_q4.hpp)
enum KvFmt { kF16 = 0, kI8 = 1, kQ4 = 2 };

template <int KF, int VF, int CH>   // K's and V's KvFmt
struct PromptAttn {
    static constexpr int KG = KF == kQ4 ? 2 * NG : NG;   // K's scale groups (q.k partials), V's
    static constexpr int VG = VF == kQ4 ? 2 * NG : NG;
    static constexpr int Q4ROW = (HD / QK4_0) * (int)sizeof(block_q4_0);   // bytes of a Q4_0 row (one head, one cell)

    // Q4_0 block `b` of a row into FP16 row `dst` (its 32 values) and its scale: element j is the low nibble of byte j
    // (j < 16), else the high nibble of byte j - 16, minus 8. The block's 18 bytes are 2-byte aligned
    static float q4_block(const uint8_t* row, int b, half* dst) {
        // five 4-byte loads from the 4-byte boundary at or below the block (a row starts 16-byte aligned, a block 2)
        const int start = b * (int)sizeof(block_q4_0), delta = start & 3;
        const auto* w4 = reinterpret_cast<const uint32_t*>(row + (start - delta));
        uint32_t u[5];
        for (int i = 0; i < 5; ++i) u[i] = w4[i];
        sycl::vec<uint32_t, 4> cw;
        uint16_t dbits;
        if (delta == 0) {   // d in bytes 0-1, the codes from byte 2
            dbits = (uint16_t)(u[0] & 0xFFFF);
            for (int i = 0; i < 4; ++i) cw[i] = (u[i] >> 16) | (u[i + 1] << 16);
        } else {            // d in bytes 2-3, the codes from byte 4
            dbits = (uint16_t)(u[0] >> 16);
            for (int i = 0; i < 4; ++i) cw[i] = u[i + 1];
        }
        // 16-byte aligned vector load / unpack for Q4_0 KV cache blocks directly into local memory
        const sycl::vec<uint16_t, 8> v = cw.template as<sycl::vec<uint16_t, 8>>();
        // bytes 2i and 2i + 1 of the codes are the low and high bytes of v[i]: their nibbles in element order
        const sycl::vec<uint16_t, 8> lo = v & (uint16_t)0x0F0F, hi = (v >> 4) & (uint16_t)0x0F0F;
        const auto l8 = lo.template as<sycl::vec<int8_t, 16>>() - (int8_t)8;
        const auto h8 = hi.template as<sycl::vec<int8_t, 16>>() - (int8_t)8;
        auto* d = reinterpret_cast<sycl::vec<half, 16>*>(dst);
        d[0] = l8.template convert<half>();
        d[1] = h8.template convert<half>();
        return f32_from_f16(dbits);
    }

    // x as FP16 hi + lo: hi rounded, lo the rest. hi goes back to FP32 through integer operations: written as
    // x - (float) half(x), the compiler took the round trip for exact and lo came out zero (the products kept
    // FP16's 11 bits)
    static void split(float x, half& hi, half& lo) {
        hi = half(x);
        lo = half(x - f32_from_f16(sycl::bit_cast<uint16_t>(hi)));
    }

    static constexpr int KV_HALVES = 2 * CH * KR;   // FP16 K and V rows; q (hi + lo) shares them at the start
    static_assert(KV_HALVES >= 2 * R * QS, "q's staging must fit in the K/V rows");
    const float* qv;
    QsaAttnPools p;
    const int32_t* ids;
    const int32_t* steps;
    int64_t cap, n_kv, page_size;
    float scale_log2;
    float* attn;
    sycl::local_accessor<half, 1> kvq;      // kh [CH][KR], vh [CH][KR]; at the start qh [R][QS], ql [R][QS]
    sycl::local_accessor<float, 2> ks, vs;  // [CH][KG], [CH][VG]
    sycl::local_accessor<long long, 1> rows;
    sycl::local_accessor<float, 3> part;    // [NG][R][CH]: q.k per 64-dim group; part[0] then holds p
    sycl::local_accessor<half, 3> pp;       // [2 VG][R][CH]: p' hi (2v) / lo (2v + 1) per V group
    sycl::local_accessor<float, 1> mrow, lsum, alpha;

    void operator()(sycl::nd_item<2> it) const {
        const auto grp = it.get_group();
        const auto sg = it.get_sub_group();
        const int64_t qi = (int64_t)it.get_group(0), kvh = (int64_t)it.get_group(1);
        const int t = (int)it.get_local_id(1), s = (int)sg.get_group_linear_id();
        const int lane = (int)sg.get_local_linear_id();
        const int g = s / 2, mt = s % 2;   // this sub-group's dim group and row tile
        const int64_t n_head = n_kv * G;
        const float* qp = qv + qi * n_head * HD + kvh * G * HD;
        float* out = attn + qi * n_head * HD + kvh * G * HD;
        const int32_t* sel = ids + qi * cap;
        const int n = steps[qi * kStepCount + kStepWidth];
        const size_t KH = 0, VH = (size_t)CH * KR, QH = 0, QL = (size_t)R * QS;

        // q: 12 heads + 4 zero rows, times 2^(14 - e) for its largest |value| < 2^e, as FP16 hi + lo
        float qm = 0.0f;
        for (int i = t; i < G * HD; i += WG) qm = sycl::fmax(qm, sycl::fabs(qp[i]));
        qm = sycl::reduce_over_group(grp, qm, sycl::maximum<float>());
        const int qe = qm > 0.0f ? (int)((sycl::bit_cast<uint32_t>(qm) >> 23) & 0xFF) - 126 : 0;   // qm < 2^qe
        const float qup = sycl::ldexp(1.0f, 14 - qe), qdown = sycl::ldexp(scale_log2, qe - 14);
        for (int i = t; i < R * HD; i += WG) {
            const int r = i / HD, d = i % HD;
            half hi, lo;
            split(r < G ? qp[r * HD + d] * qup : 0.0f, hi, lo);
            kvq[QH + (size_t)r * QS + d] = hi;
            kvq[QL + (size_t)r * QS + d] = lo;
        }
        if (t < R) { mrow[t] = NEG_INF; lsum[t] = 0.0f; }
        sycl::group_barrier(grp);
        // this sub-group's q: its 8 rows x 4 steps of 16 of its 64 dims, hi and lo, kept in registers
        mx::joint_matrix<sycl::sub_group, half, mx::use::a, TM, TK, mx::layout::row_major> qa[4], qb[4];
        for (int kk = 0; kk < 4; ++kk) {
            const size_t off = (size_t)mt * TM * QS + (size_t)g * 64 + (size_t)kk * TK;
            mx::joint_matrix_load(sg, qa[kk], at(kvq, QH + off), QS);
            mx::joint_matrix_load(sg, qb[kk], at(kvq, QL + off), QS);
        }
        // this sub-group's 8 output rows x its 64 dims, in units of `unit` (the largest V scale so far / PSCALE)
        mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> acc[4];
        for (int nt = 0; nt < 4; ++nt) mx::joint_matrix_fill(sg, acc[nt], 0.0f);
        float unit = 0.0f;

        for (int c0 = 0; c0 < n; c0 += CH) {
            const int nh = sycl::min(CH, n - c0);
            if (t < CH) {
                long long r = -1;
                if (t < nh) {
                    const int cell = sel[c0 + t];
                    const long long page = (long long)p.page_table[cell / page_size];
                    if (page >= 0) r = (page * n_kv + kvh) * page_size + cell % page_size;
                }
                rows[t] = r;
            }
            sycl::group_barrier(grp);

            // gather K and V as FP16 directly into local memory with 16-byte aligned vector loads (sycl::vec<uint16_t, 8>)
            for (int i = t; i < CH * (HD / 16); i += WG) {
                const int c = i / (HD / 16), pc = i % (HD / 16);
                const long long r = rows[c];
                sycl::vec<half, 16> kx(half(0.0f)), vx(half(0.0f));
                if (r >= 0) {
                    const int64_t off = r * HD + (int64_t)pc * 16;
                    if constexpr (KF == kI8) {
                        const auto raw = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(p.k_q + off);
                        kx = raw.template as<sycl::vec<int8_t, 16>>().template convert<half>();
                    } else if constexpr (KF == kF16) {
                        const auto raw0 = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(p.k_pool + off);
                        const auto raw1 = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(p.k_pool + off + 8);
                        auto* kxp = reinterpret_cast<sycl::vec<uint16_t, 8>*>(&kx);
                        kxp[0] = raw0;
                        kxp[1] = raw1;
                    }
                    if constexpr (VF == kI8) {
                        const auto raw = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(p.v_q + off);
                        vx = raw.template as<sycl::vec<int8_t, 16>>().template convert<half>();
                    } else if constexpr (VF == kF16) {
                        const auto raw0 = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(p.v_pool + off);
                        const auto raw1 = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(p.v_pool + off + 8);
                        auto* vxp = reinterpret_cast<sycl::vec<uint16_t, 8>*>(&vx);
                        vxp[0] = raw0;
                        vxp[1] = raw1;
                    }
                }
                if constexpr (KF != kQ4)
                    *reinterpret_cast<sycl::vec<half, 16>*>(&kvq[KH + (size_t)c * KR + (size_t)pc * 16]) = kx;
                if constexpr (VF != kQ4)
                    *reinterpret_cast<sycl::vec<half, 16>*>(&kvq[VH + (size_t)c * KR + (size_t)pc * 16]) = vx;
            }

            // Q4_0: one work-item a block (32 values and the scale): CH x 8 = 128 blocks, one each
            if constexpr (KF == kQ4 || VF == kQ4)
                for (int i = t; i < CH * (HD / QK4_0); i += WG) {
                    const int c = i / (HD / QK4_0), b = i % (HD / QK4_0);
                    const long long r = rows[c];
                    half* kd = &kvq[KH + (size_t)c * KR + (size_t)b * QK4_0];
                    half* vd = &kvq[VH + (size_t)c * KR + (size_t)b * QK4_0];
                    if constexpr (KF == kQ4) {
                        float a = 0.0f;
                        if (r >= 0) a = q4_block(p.k_q4 + r * Q4ROW, b, kd);
                        else for (int j = 0; j < QK4_0; ++j) kd[j] = half(0.0f);
                        ks[c][b] = a;
                    }
                    if constexpr (VF == kQ4) {
                        float a = 0.0f;
                        if (r >= 0) a = q4_block(p.v_q4 + r * Q4ROW, b, vd);
                        else for (int j = 0; j < QK4_0; ++j) vd[j] = half(0.0f);
                        vs[c][b] = a;
                    }
                }
            for (int i = t; i < CH * 2 * NG; i += WG) {
                const int c = i / (2 * NG), gg = i % (2 * NG);
                const long long r = rows[c];
                if (KF != kQ4 && gg < KG) {   // Q4_0's scales come with its blocks
                    float a = 0.0f;
                    if (r >= 0) a = KF == kI8 ? f32_from_f16(p.k_scale[r * (HD / KV_Q8_GROUP) + gg]) : 1.0f;
                    ks[c][gg] = a;
                }
                if (VF != kQ4 && gg < VG) {
                    float b = 0.0f;
                    if (r >= 0) b = VF == kI8 ? f32_from_f16(p.v_scale[r * (HD / KV_Q8_GROUP) + gg]) : 1.0f;
                    vs[c][gg] = b;
                }
            }
            sycl::group_barrier(grp);

            // q.k of this sub-group's 8 rows and 64 dims: CH/16 cell tiles, 4 steps of 16 dims, hi and lo.
            for (int nt = 0; nt < CH / TN; ++nt) {
                constexpr int KPG = KG / NG, KK = 4 / KPG;   // this sub-group's K groups, the 16-dim steps of each
                auto product = [&](auto& c, int kp) {
                    mx::joint_matrix_fill(sg, c, 0.0f);
                    for (int k2 = 0; k2 < KK; ++k2) {
                        const int kk = kp * KK + k2;
                        mx::joint_matrix<sycl::sub_group, half, mx::use::b, TK, TN, mx::layout::col_major> b;
                        const size_t off = KH + (size_t)nt * TN * KR + (size_t)g * 64 + (size_t)kk * TK;
                        mx::joint_matrix_load(sg, b, at(kvq, off), KR);
                        mx::joint_matrix_mad(sg, c, qa[kk], b, c);
                        mx::joint_matrix_mad(sg, c, qb[kk], b, c);
                    }
                };
                mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> c0;
                product(c0, 0);
                if constexpr (KPG == 2) {
                    mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> c1;
                    product(c1, 1);
                    const size_t cell = (size_t)nt * TN + lane, k0 = (size_t)2 * g;
                    const float s0 = ks[cell][k0], s1 = ks[cell][k0 + 1];
                    float hi[TM];
                    int i = 0;
                    mx::joint_matrix_apply(sg, c1, [&](float& x) { hi[i++] = x; });
                    i = 0;
                    mx::joint_matrix_apply(sg, c0, [&](float& x) { x = x * s0 + hi[i] * s1; ++i; });
                }
                mx::joint_matrix_store(sg, c0, at(part, (g * R + mt * TM) * CH + nt * TN), CH, mx::layout::row_major);
            }
            sycl::group_barrier(grp);

            // online softmax: row t/8, CH/8 cells each, 8 neighbouring lanes a row; the groups' partials in order
            {
                constexpr int PER = CH / 8;
                const int r = t / 8, sub = t % 8;
                float x[PER], mxv = NEG_INF;
                for (int j = 0; j < PER; ++j) {
                    const int c = sub * PER + j;
                    float sc;   // the groups' partials in order (Q4_0's already scaled)
                    if constexpr (KF == kQ4)
                        sc = ((part[0][r][c] + part[1][r][c]) + part[2][r][c]) + part[3][r][c];
                    else
                        sc = (((part[0][r][c] * ks[c][0] + part[1][r][c] * ks[c][1]) + part[2][r][c] * ks[c][2]) +
                              part[3][r][c] * ks[c][3]);
                    x[j] = c < nh && rows[c] >= 0 ? sc * qdown : NEG_INF;
                    mxv = sycl::fmax(mxv, x[j]);
                }
                for (int o = 1; o < 8; o <<= 1) mxv = sycl::fmax(mxv, sycl::permute_group_by_xor(sg, mxv, o));
                const float m_old = mrow[r];
                // FlashAttention-3 style lazy rescaling: denominator update avoided if within TAU
                const float m_new = m_old == NEG_INF || mxv > m_old + TAU ? mxv : m_old;
                float sum = 0.0f;
                for (int j = 0; j < PER; ++j) {
                    const float e = x[j] == NEG_INF ? 0.0f : sycl::exp2(x[j] - m_new);
                    part[0][r][sub * PER + j] = e;   // this work-item's own cells
                    sum += e;
                }
                for (int o = 1; o < 8; o <<= 1) sum += sycl::permute_group_by_xor(sg, sum, o);
                if (sub == 0) {
                    const float a = m_old == m_new ? 1.0f : m_old == NEG_INF ? 0.0f : sycl::exp2(m_old - m_new);
                    alpha[r] = a;
                    lsum[r] = sycl::fma(lsum[r], a, sum);
                    mrow[r] = m_new;
                }
            }
            sycl::group_barrier(grp);

            // p.v of this sub-group's 8 rows and 64 dims: p' = p * vscale / (largest |V scale| so far) * PSCALE, hi +
            // lo, one p' per V scale group (two of 32 dims for Q4_0, whose scales are signed)
            {
                constexpr int VPG = VG / NG;   // this sub-group's V groups
                float vmax = 0.0f;
                for (int c = lane; c < CH; c += SGS)
                    for (int vp = 0; vp < VPG; ++vp) vmax = sycl::fmax(vmax, sycl::fabs(vs[c][g * VPG + vp]));
                vmax = sycl::reduce_over_group(sg, vmax, sycl::maximum<float>());
                // the output so far into the new units when the largest V scale grows
                const float vdown = sycl::fmax(vmax * (1.0f / PSCALE), unit);
                const float ratio = vdown > unit && unit > 0.0f ? unit / vdown : 1.0f;
                unit = vdown;
                const float vup = vdown > 0.0f ? 1.0f / vdown : 0.0f;
                for (int vp = 0; vp < VPG; ++vp) {
                    const int vg = g * VPG + vp;
                    const float w = vs[lane][vg] * vup;   // CH == SGS: lane `lane` owns cell `lane` of the 8 rows
                    for (int k = 0; k < TM; ++k) {
                        const int r = mt * TM + k;
                        half hi, lo;
                        split(part[0][r][lane] * w, hi, lo);
                        pp[(size_t)2 * vg][r][lane] = hi;
                        pp[(size_t)2 * vg + 1][r][lane] = lo;
                    }
                }
                float f[TM];
                bool same = true;
                for (int i = 0; i < TM; ++i) {
                    f[i] = alpha[mt * TM + i] * ratio;
                    same = same && f[i] == 1.0f;
                }
                if (!same)
                    for (int nt = 0; nt < 4; ++nt) {
                        int i = 0;
                        mx::joint_matrix_apply(sg, acc[nt], [&](float& x) { x *= f[i++]; });
                    }
                sycl::group_barrier(sg);
#pragma unroll
                for (int kc = 0; kc < CH / TK; ++kc)
#pragma unroll
                    for (int vp = 0; vp < VPG; ++vp) {
                        const int vg = g * VPG + vp;
                        mx::joint_matrix<sycl::sub_group, half, mx::use::a, TM, TK, mx::layout::row_major> ph, pl;
                        mx::joint_matrix_load(sg, ph, at(pp, (2 * vg * R + mt * TM) * CH + kc * TK), CH);
                        mx::joint_matrix_load(sg, pl, at(pp, ((2 * vg + 1) * R + mt * TM) * CH + kc * TK), CH);
#pragma unroll
                        for (int n2 = 0; n2 < 4 / VPG; ++n2) {   // the 16-dim output tiles of this V group
                            const int nt = vp * (4 / VPG) + n2;
                            mx::joint_matrix<sycl::sub_group, half, mx::use::b, TK, TN, mx::layout::row_major> b;
                            const size_t off = VH + (size_t)kc * TK * KR + (size_t)g * 64 + (size_t)nt * TN;
                            mx::joint_matrix_load(sg, b, at(kvq, off), KR);
                            mx::joint_matrix_mad(sg, acc[nt], ph, b, acc[nt]);
                            mx::joint_matrix_mad(sg, acc[nt], pl, b, acc[nt]);
                        }
                    }
            }
        }
        sycl::group_barrier(grp);
        for (int nt = 0; nt < 4; ++nt)
            ix::joint_matrix_apply(sg, acc[nt], [&](float& x, size_t row, size_t col) {
                const int r = mt * TM + (int)row;
                if (r < G) {
                    const float l = lsum[r];
                    out[r * HD + g * 64 + nt * TN + (int)col] = l > 0.0f ? x * unit / l : 0.0f;
                }
            });
    }

    auto get(syclex::properties_tag) const {
        return syclex::properties{sycl::ext::intel::experimental::grf_size<256>, syclex::sub_group_size<SGS>};
    }
};

// the work-group's local memory, as `launch` allocates it
template <int KF, int VF, int CH>
constexpr size_t local_bytes() {
    using K = PromptAttn<KF, VF, CH>;
    return (size_t)K::KV_HALVES * 2 + (size_t)CH * (K::KG + K::VG) * 4 + (size_t)CH * 8 + (size_t)NG * R * CH * 4 +
           (size_t)2 * K::VG * R * CH * 2 + (size_t)3 * R * 4;
}

template <int KF, int VF, int CH>
sycl::event launch(sycl::queue& q, const float* qv, QsaAttnPools p, const int32_t* ids, const int32_t* steps,
                   int64_t cap, int64_t n_kv, int64_t page_size, float* attn, int64_t n_q) {
    const float scale_log2 = 1.4426950408889634f / std::sqrt((float)HD);
    return q.submit([&](sycl::handler& h) {
        using K = PromptAttn<KF, VF, CH>;
        K k{qv, p, ids, steps, cap, n_kv, page_size, scale_log2, attn,
            sycl::local_accessor<half, 1>(sycl::range<1>(K::KV_HALVES), h),
            sycl::local_accessor<float, 2>(sycl::range<2>(CH, K::KG), h),
            sycl::local_accessor<float, 2>(sycl::range<2>(CH, K::VG), h),
            sycl::local_accessor<long long, 1>(sycl::range<1>(CH), h),
            sycl::local_accessor<float, 3>(sycl::range<3>(NG, R, CH), h),
            sycl::local_accessor<half, 3>(sycl::range<3>((size_t)2 * K::VG, R, CH), h),
            sycl::local_accessor<float, 1>(sycl::range<1>(R), h),
            sycl::local_accessor<float, 1>(sycl::range<1>(R), h),
            sycl::local_accessor<float, 1>(sycl::range<1>(R), h)};
        h.parallel_for(sycl::nd_range<2>({(size_t)n_q, (size_t)(n_kv * WG)}, {1, WG}), k);
    });
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
        h[i] = (_Float16)(int)(int8_t)(x.x >> (8 * i));
        h[4 + i] = (_Float16)(int)(int8_t)(x.y >> (8 * i));
    }
    return h;
}

__global__ void __launch_bounds__(WG)
    prompt_attn_wmma_kernel(const float* __restrict__ q, QsaAttnPools p, const int32_t* __restrict__ ids,
                            const int32_t* __restrict__ steps, int n_kv_heads, int page_size, float scale_log2,
                            float* __restrict__ attn, int cap) {
#if STRATA_PA_WMMA
    __shared__ SmemW S;
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int n_head = n_kv_heads * G;
    q += (size_t)qi * n_head * HD + (size_t)kvh * G * HD;
    attn += (size_t)qi * n_head * HD + (size_t)kvh * G * HD;
    ids += (size_t)qi * cap;
    const int n = __ldg(steps + (size_t)qi * kStepCount + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int col = lane & 15, half = lane >> 4;
    const int dim0 = warp * 64;

    float qm = 0.0f;
    for (int i = t; i < G * HD; i += WG)
        qm = fmaxf(qm, fabsf(q[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    if (t < 16) {
        S.mrow[t] = -INFINITY;
        S.lsum[t] = 0.0f;
    }
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
            const float4* src = reinterpret_cast<const float4*>(q + (size_t)col * HD + dim0 + kk * 16 + half * 8);
            const float4 a = src[0], b = src[1];
            x[0] = a.x;
            x[1] = a.y;
            x[2] = a.z;
            x[3] = a.w;
            x[4] = b.x;
            x[5] = b.y;
            x[6] = b.z;
            x[7] = b.w;
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i)
                x[i] = 0.0f;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float v = x[i] * qup;
            const _Float16 hi = (_Float16)v;
            qh[kk][i] = hi;
            ql[kk][i] = (_Float16)(v - (float)hi);
        }
    }

    wf8 acc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
        acc[j] = wf8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int n_chunks = (n + WCH - 1) / WCH;
    for (int ci = 0; ci < n_chunks; ++ci) {
        const int c0 = ci * WCH;
        long long myrow = -1;
        if (c0 + lane < n) {
            const int cell = __ldg(ids + c0 + lane);
            if (cell >= 0) {
                const long long page = (long long)__ldg(p.page_table + cell / page_size);
                if (page >= 0) myrow = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
            }
        }
        {
            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0, v2 = v0, v3 = v0;
            float vsc = 0.0f;
            if (myrow >= 0) {
                const uint4* src = reinterpret_cast<const uint4*>(p.v_q + myrow * HD + dim0);
                v0 = __ldg(src);
                v1 = __ldg(src + 1);
                v2 = __ldg(src + 2);
                v3 = __ldg(src + 3);
                vsc = __half2float(__ushort_as_half(__ldg(p.v_scale + myrow * (HD / KV_Q8_GROUP) + warp)));
            }
            uint4* dst = reinterpret_cast<uint4*>(&S.v[warp][lane][0]);
            dst[0] = v0;
            dst[1] = v1;
            dst[2] = v2;
            dst[3] = v3;
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
            for (int i = 0; i < 8; ++i)
                S.part[warp][half * 8 + i][nt * 16 + col] = s[i] * ksc;
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
            for (int o = 1; o < 8; o <<= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
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
            for (int o = 1; o < 8; o <<= 1)
                sum += __shfl_xor_sync(0xffffffffu, sum, o);
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
            for (int o = 16; o > 0; o >>= 1)
                vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
            wf8 tmp[4];
#pragma unroll
            for (int j = 0; j < 4; ++j)
                tmp[j] = wf8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                const int cb = ks * 16 + half * 8;
                wh8 ah, al;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float pv = S.p[col][cb + i] * (S.vs[warp][cb + i] * vup);
                    const _Float16 hi = (_Float16)pv;
                    ah[i] = hi;
                    al[i] = (_Float16)(pv - (float)hi);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int d = j * 16 + col;
                    wh8 b;
#pragma unroll
                    for (int i = 0; i < 8; ++i)
                        b[i] = (_Float16)(int)(int8_t)S.v[warp][cb + i][d];
                    tmp[j] = wmma_f16(ah, b, tmp[j]);
                    tmp[j] = wmma_f16(al, b, tmp[j]);
                }
            }
            float a[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                a[i] = S.alpha[half * 8 + i];
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    acc[j][i] = fmaf(acc[j][i], a[i], tmp[j][i] * vdown);
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
            if (row < G) attn[(size_t)row * HD + dim0 + j * 16 + col] = acc[j][i] * inv[i];
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
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) {
        (void)hipGetLastError();
        return false;
    }
    if (arch[dev] == 0) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, dev) != hipSuccess) {
            (void)hipGetLastError();
            return false;
        }
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
    const float scale_log2 = 1.4426950408889634f / sqrtf((float)HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        prompt_attn_wmma_kernel<<<dim3((unsigned)nb, (unsigned)s.n_head_kv), WG, 0, st>>>(
            q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount, (int)s.n_head_kv, (int)s.page_size,
            scale_log2, attn + q0 * s.n_head * HD, (int)cap);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_prompt_attn_batch (wmma): %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}
#endif

constexpr int CHUNK = 16;
static_assert(CHUNK == 16, "the p split gives each lane of a 16-wide sub-group one cell");

int g_pa_imma = -1;

}  // namespace

bool qsa_decode_attn_split(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, uint64_t scratch_floats, float* attn,
                           int64_t n_q, void* stream) {
    (void)q;
    (void)pools;
    (void)ids;
    (void)steps;
    (void)cap;
    (void)s;
    (void)scratch;
    (void)scratch_floats;
    (void)attn;
    (void)n_q;
    (void)stream;
    return false;
}

void qsa_prompt_attn_set_imma(int on) { g_pa_imma = on < 0 ? -1 : (on ? 1 : 0); }

bool qsa_prompt_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return true;
    static sycl::queue default_queue{sycl::property::queue::in_order()};
    sycl::queue* q_ptr = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

#if defined(__HIPCC__)
    if (pools.k_q != nullptr && pools.v_q != nullptr && pools.k_scale != nullptr && pools.v_scale != nullptr &&
        pools.k_q4 == nullptr && pools.v_q4 == nullptr && s.head_dim == HD && s.n_head == (int64_t)G * s.n_head_kv &&
        cap > 0 && ids && steps && pools.page_table && hip_wmma_usable())
        return launch_wmma(q, pools, ids, steps, cap, s, attn, n_q, (cudaStream_t)stream);
    return false;
#endif

    // Fast-path: Intel Battlemage Xe2 int8 DPAS (ESIMD) prompt attention
    if (qsa_prompt_attn_esimd(q, pools, ids, steps, cap, s, attn, n_q, stream)) {
        return true;
    }

    if (s.head_dim != HD || s.n_head != (int64_t)G * s.n_head_kv || cap <= 0 || !ids || !steps || !pools.page_table)
        return false;

    if (!xmx_available(XmxType::f16))
        return false;

    const size_t slm = q_ptr->get_device().get_info<sycl::info::device::local_mem_size>();
    static const bool q4_on = [] {
        const char* v = std::getenv("STRATA_PROMPT_ATTN_Q4");
        return v == nullptr || v[0] != '0';
    }();

    const int64_t nkv = s.n_head_kv, ps = s.page_size;
    sycl::event e;
    if (pools.k_q4 != nullptr) {   // --kv q4_0
        if (!q4_on || !pools.v_q4 || slm < local_bytes<kQ4, kQ4, CHUNK>()) return false;
        e = launch<kQ4, kQ4, CHUNK>(*q_ptr, q, pools, ids, steps, cap, nkv, ps, attn, n_q);
    } else if (pools.k_q != nullptr) {
        // --kv k8v4 (INT8 K, Q4_0 V) stays on the FP32 kernel: this one ran it at 0.84x (B70, 32K cells, 2,048 queries;
        // Q4_0 V doubles the p' planes of each chunk)
        if (pools.v_q4 != nullptr) return false;
        if (!pools.v_q || !pools.k_scale || !pools.v_scale || slm < local_bytes<kI8, kI8, CHUNK>()) return false;
        e = launch<kI8, kI8, CHUNK>(*q_ptr, q, pools, ids, steps, cap, nkv, ps, attn, n_q);
    } else {
        if (!pools.k_pool || !pools.v_pool || slm < local_bytes<kF16, kF16, CHUNK>()) return false;
        e = launch<kF16, kF16, CHUNK>(*q_ptr, q, pools, ids, steps, cap, nkv, ps, attn, n_q);
    }
    if (!stream) e.wait();
    return true;
}

}  // namespace strata::kernels
