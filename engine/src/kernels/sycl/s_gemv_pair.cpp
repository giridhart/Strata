// src/kernels/sycl/s_gemv_pair.cpp - Pure SYCL 2020 gate/up dual GEMV + SwiGLU in ONE kernel for Intel Xe2.
#include <sycl/sycl.hpp>
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/s_gemv.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int Q8K_BLOCK_BYTES = 292;
constexpr int Q8K_BLOCK_ELEMS = 256;
constexpr int Q8_0_BLOCK_BYTES = 34;
constexpr int Q8_0_BLOCK_ELEMS = 32;

// Side kinds: 0 = S2 row (Q8_0 act), 1 = SideQ8K (S4/S8 with Q8_K act), 2 = SideQ80 (S4/S8 with Q8_0 act)
enum { SideS2 = 0, SideQ8K = 1, SideQ80 = 2 };

constexpr signed char kIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

template <int CODE_BITS>
inline float decode_tbl(int code, int bias, int codebook, const signed char* tbl) {
    if (codebook == (int)Codebook::Iq4Nl) return (float)tbl[code & 0x0F];
    return (float)(code + bias);
}

inline float q8k_at(const uint8_t* __restrict__ x, long long i) {
    const uint8_t* blk = x + (i / Q8K_BLOCK_ELEMS) * Q8K_BLOCK_BYTES;
    const float d = *(const float*)blk;
    const int8_t q = ((const int8_t*)(blk + 4))[i % Q8K_BLOCK_ELEMS];
    return d * (float)q;
}

inline float q8_0_at(const uint8_t* __restrict__ x, long long i) {
    const uint8_t* blk = x + (i / Q8_0_BLOCK_ELEMS) * Q8_0_BLOCK_BYTES;
    const float d = f32_from_f16(*(const uint16_t*)blk);
    const int8_t q = ((const int8_t*)(blk + 2))[i % Q8_0_BLOCK_ELEMS];
    return d * (float)q;
}

struct SideArgs {
    int kind;
    int code_bits;
    int bias;
    int codebook;
    int group_shift;
    int has_offset;
    const uint8_t* codes;
    const float* scales;
    const float* off;
    const uint8_t* x80;
    const uint8_t* x8k;
};

template <int CODE_BITS, bool Q8K, int SUB_SIZE>
inline float row_s_q8(const uint8_t* __restrict__ x, const uint8_t* __restrict__ codes,
                      const float* __restrict__ scales, const float* __restrict__ offset,
                      long long n_in, long long o, int lane, int bias, int codebook,
                      int group_shift, int has_offset, const signed char* s_iq4nl,
                      const sycl::sub_group& sg) {
    constexpr int PER_BYTE = 8 / CODE_BITS;
    const long long n_groups = n_in >> group_shift;
    const long long codes_per_row = n_in / PER_BYTE;
    const uint8_t* c = codes + o * codes_per_row;
    const float* s = scales + o * n_groups;
    const float* off = has_offset ? offset + o * n_groups : nullptr;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    float acc8 = 0.0f, acc9 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;
    float acc12 = 0.0f, acc13 = 0.0f, acc14 = 0.0f, acc15 = 0.0f;
    constexpr int QE = 16;
    constexpr int QW = 4 * CODE_BITS / 8;
    constexpr unsigned MASK = (1u << CODE_BITS) - 1u;
    long long i = (long long)lane * QE;
    for (; i + QE <= n_in; i += (long long)SUB_SIZE * QE) {
        const long long g = i >> group_shift;
        const float d = s[g];
        const float b = off ? off[g] : 0.0f;
        const uint8_t* cp = c + i / PER_BYTE;
        unsigned v, v2, v3, v4;
        if constexpr (QW == 1) {
            v = cp[0]; v2 = cp[1]; v3 = cp[2]; v4 = cp[3];
        } else if constexpr (QW == 2) {
            v = *(const uint16_t*)cp;
            v2 = *(const uint16_t*)(cp + 2);
            v3 = *(const uint16_t*)(cp + 4);
            v4 = *(const uint16_t*)(cp + 6);
        } else {
            v = *(const uint32_t*)cp;
            v2 = *(const uint32_t*)(cp + 4);
            v3 = *(const uint32_t*)(cp + 8);
            v4 = *(const uint32_t*)(cp + 12);
        }
        constexpr int blk_elems = Q8K ? Q8K_BLOCK_ELEMS : Q8_0_BLOCK_ELEMS;
        constexpr int blk_bytes = Q8K ? Q8K_BLOCK_BYTES : Q8_0_BLOCK_BYTES;
        const uint8_t* xb = x + (i / blk_elems) * blk_bytes;
        const int xi = (int)(i % blk_elems);
        const float xd = Q8K ? *(const float*)xb : f32_from_f16(*(const uint16_t*)xb);
        const int8_t* xq = (const int8_t*)(xb + (Q8K ? 4 : 2)) + xi;

        const float w0 = decode_tbl<CODE_BITS>((int)(v & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w1 = decode_tbl<CODE_BITS>((int)((v >> CODE_BITS) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w2 = decode_tbl<CODE_BITS>((int)((v >> (2 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w3 = decode_tbl<CODE_BITS>((int)((v >> (3 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w4 = decode_tbl<CODE_BITS>((int)(v2 & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w5 = decode_tbl<CODE_BITS>((int)((v2 >> CODE_BITS) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w6 = decode_tbl<CODE_BITS>((int)((v2 >> (2 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w7 = decode_tbl<CODE_BITS>((int)((v2 >> (3 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b;
        acc0 += w0 * (xd * (float)xq[0]);
        acc1 += w1 * (xd * (float)xq[1]);
        acc2 += w2 * (xd * (float)xq[2]);
        acc3 += w3 * (xd * (float)xq[3]);
        acc4 += w4 * (xd * (float)xq[4]);
        acc5 += w5 * (xd * (float)xq[5]);
        acc6 += w6 * (xd * (float)xq[6]);
        acc7 += w7 * (xd * (float)xq[7]);
        const float w8 = decode_tbl<CODE_BITS>((int)(v3 & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w9 = decode_tbl<CODE_BITS>((int)((v3 >> CODE_BITS) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w10 = decode_tbl<CODE_BITS>((int)((v3 >> (2 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w11 = decode_tbl<CODE_BITS>((int)((v3 >> (3 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w12 = decode_tbl<CODE_BITS>((int)(v4 & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w13 = decode_tbl<CODE_BITS>((int)((v4 >> CODE_BITS) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w14 = decode_tbl<CODE_BITS>((int)((v4 >> (2 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w15 = decode_tbl<CODE_BITS>((int)((v4 >> (3 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b;
        acc8 += w8 * (xd * (float)xq[8]);
        acc9 += w9 * (xd * (float)xq[9]);
        acc10 += w10 * (xd * (float)xq[10]);
        acc11 += w11 * (xd * (float)xq[11]);
        acc12 += w12 * (xd * (float)xq[12]);
        acc13 += w13 * (xd * (float)xq[13]);
        acc14 += w14 * (xd * (float)xq[14]);
        acc15 += w15 * (xd * (float)xq[15]);
    }
    for (; i < n_in; i += (long long)SUB_SIZE * QE) {
        for (int k = 0; k < QE && i + k < n_in; ++k) {
            const long long e = i + k;
            const long long g = e >> group_shift;
            const int code = (c[e / PER_BYTE] >> ((int)(e % PER_BYTE) * CODE_BITS)) & MASK;
            acc0 += (decode_tbl<CODE_BITS>(code, bias, codebook, s_iq4nl) * s[g] + (off ? off[g] : 0.0f)) *
                    (Q8K ? q8k_at(x, e) : q8_0_at(x, e));
        }
    }
    float acc = (((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7))) +
                (((acc8 + acc9) + (acc10 + acc11)) + ((acc12 + acc13) + (acc14 + acc15)));
    return sycl::reduce_over_group(sg, acc, sycl::plus<float>());
}

template <int SUB_SIZE>
inline float row_s2_q8(const uint8_t* __restrict__ act, const uint8_t* __restrict__ codes,
                       const float* __restrict__ scales, long long n_in, long long o,
                       int lane, const sycl::sub_group& sg) {
    const long long n_quads = n_in / 4;
    const uint8_t* c = codes + o * n_quads;
    const float* s = scales + o * (n_in / 64);

    float a0 = 0.0f, a1 = 0.0f;
    for (long long q = lane; q < n_quads; q += SUB_SIZE) {
        const uint8_t byte = c[q];
        const float d = s[q >> 4];
        const long long ablk = (q * 4) / 32;
        const uint8_t* blk = act + ablk * 34;
        const uint16_t dbits = (uint16_t)(blk[0] | (blk[1] << 8));
        const float dx = f32_from_f16(dbits);
        const int8_t* xq = reinterpret_cast<const int8_t*>(blk + 2);
        const int off = (int)((q * 4) % 32);

        const int c0 = (int)(byte & 3) - 1;
        const int c1 = (int)((byte >> 2) & 3) - 1;
        const int c2 = (int)((byte >> 4) & 3) - 1;
        const int c3 = (int)((byte >> 6) & 3) - 1;
        const int idot0 = c0 * (int)xq[off + 0] + c1 * (int)xq[off + 1];
        const int idot1 = c2 * (int)xq[off + 2] + c3 * (int)xq[off + 3];
        const float scale = d * dx;
        a0 += (float)idot0 * scale;
        a1 += (float)idot1 * scale;
    }
    float acc_sum = a0 + a1;
    return sycl::reduce_over_group(sg, acc_sum, sycl::plus<float>());
}

template <int SUB_SIZE>
inline float side_row(int kind, int cb, const uint8_t* x80, const uint8_t* x8k,
                      const uint8_t* codes, const float* scales, const float* off,
                      long long n_in, long long o, int lane, int bias, int codebook,
                      int group_shift, int has_offset, const signed char* s_iq4nl,
                      const sycl::sub_group& sg) {
    if (kind == SideS2) {
        return row_s2_q8<SUB_SIZE>(x80, codes, scales, n_in, o, lane, sg);
    }
    const uint8_t* x = (kind == SideQ8K) ? x8k : x80;
    if (cb == 4) {
        if (kind == SideQ8K) {
            return row_s_q8<4, true, SUB_SIZE>(x, codes, scales, off, n_in, o, lane, bias, codebook,
                                               group_shift, has_offset, s_iq4nl, sg);
        } else {
            return row_s_q8<4, false, SUB_SIZE>(x, codes, scales, off, n_in, o, lane, bias, codebook,
                                                group_shift, has_offset, s_iq4nl, sg);
        }
    } else {
        if (kind == SideQ8K) {
            return row_s_q8<8, true, SUB_SIZE>(x, codes, scales, off, n_in, o, lane, bias, codebook,
                                               group_shift, has_offset, s_iq4nl, sg);
        } else {
            return row_s_q8<8, false, SUB_SIZE>(x, codes, scales, off, n_in, o, lane, bias, codebook,
                                                group_shift, has_offset, s_iq4nl, sg);
        }
    }
}

inline float swilu_eval(float g, float u, int swilu_kind) {
    if (swilu_kind == 0) {
        const double x = (double)g;
        return (float)(x / (1.0 + sycl::exp(-x))) * u;
    } else {
        return (g / (1.0f + sycl::native::exp(-g))) * u;
    }
}

template <int SUB_SIZE>
void pair_kernel(sycl::nd_item<1> item, SideArgs g, SideArgs u, float* __restrict__ out,
                 long long n_in, long long n_out, int swilu_kind, signed char* s_iq4nl) {
    if (item.get_local_id(0) < 16) {
        s_iq4nl[item.get_local_id(0)] = kIq4Nl[item.get_local_id(0)];
    }
    item.barrier(sycl::access::fence_space::local_space);

    const auto sg = item.get_sub_group();
    const int sgs_per_wg = (int)(item.get_local_range(0) / SUB_SIZE);
    const long long o = (long long)item.get_group(0) * sgs_per_wg + (item.get_local_id(0) / SUB_SIZE);
    if (o >= n_out) return;

    const int lane = sg.get_local_linear_id();
    const float ag = side_row<SUB_SIZE>(g.kind, g.code_bits, g.x80, g.x8k, g.codes, g.scales, g.off,
                                        n_in, o, lane, g.bias, g.codebook, g.group_shift, g.has_offset,
                                        s_iq4nl, sg);
    const float au = side_row<SUB_SIZE>(u.kind, u.code_bits, u.x80, u.x8k, u.codes, u.scales, u.off,
                                        n_in, o, lane, u.bias, u.codebook, u.group_shift, u.has_offset,
                                        s_iq4nl, sg);

    if (sg.leader()) {
        out[o] = swilu_eval(ag, au, swilu_kind);
    }
}

bool side_of(const SForm& f, int64_t n_in, const uint8_t* x80, const uint8_t* x8k, const uint8_t* codes,
             const float* scales, const float* off, SideArgs* out) {
    if (f.code_bits == 2) {
        if (!x80 || (n_in % 32 != 0) || (n_in % 64 != 0)) return false;
        out->kind = SideS2;
    } else if (f.code_bits == 4 || f.code_bits == 8) {
        if (f.group_elems <= 0 || (f.group_elems & (f.group_elems - 1)) != 0 || (f.group_elems % 16 != 0))
            return false;
        if (n_in % f.group_elems != 0) return false;
        if (f.act_kind == 1) {
            if (!x8k || (n_in % 256 != 0)) return false;
            out->kind = SideQ8K;
        } else {
            if (!x80 || (n_in % 32 != 0)) return false;
            out->kind = SideQ80;
        }
    } else {
        return false;
    }
    if (f.has_offset && off == nullptr) return false;
    out->code_bits = f.code_bits;
    out->bias = f.code_bias;
    out->codebook = (int)f.codebook;
    out->group_shift = 0;
    while ((1 << out->group_shift) < f.group_elems) ++out->group_shift;
    if ((1 << out->group_shift) != f.group_elems) return false;
    out->has_offset = f.has_offset ? 1 : 0;
    out->codes = codes;
    out->scales = scales;
    out->off = off;
    out->x80 = x80;
    out->x8k = x8k;
    return true;
}

}  // namespace

bool s_gemv_pair_silu(const uint8_t* x_q8_0, const uint8_t* x_q8k, const SForm& form_g, const uint8_t* codes_g,
                      const float* scales_g, const float* off_g, const SForm& form_u, const uint8_t* codes_u,
                      const float* scales_u, const float* off_u, float* out, int64_t n_in, int64_t n_out,
                      int threads_per_row, int swilu_kind, void* stream) {
    if (n_in <= 0 || n_out <= 0) return false;
    if (threads_per_row > 0 && threads_per_row != 16 && threads_per_row != 32) return false;

    SideArgs g, u;
    if (!side_of(form_g, n_in, x_q8_0, x_q8k, codes_g, scales_g, off_g, &g)) return false;
    if (!side_of(form_u, n_in, x_q8_0, x_q8k, codes_u, scales_u, off_u, &u)) return false;

    static sycl::queue default_queue{sycl::default_selector_v, sycl::property::queue::in_order()};
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    constexpr int SG_SIZE = strata::sycl_arch::SUB_GROUP_SIZE;
    constexpr int THREADS = 256;
    constexpr int sgs_per_wg = THREADS / SG_SIZE;
    const unsigned blocks = (unsigned)((n_out + sgs_per_wg - 1) / sgs_per_wg);

    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<signed char, 1> s_iq4nl_acc(sycl::range<1>(16), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(blocks * THREADS), sycl::range<1>(THREADS)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                pair_kernel<SG_SIZE>(item, g, u, out, n_in, n_out, swilu_kind,
                                    s_iq4nl_acc.get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });

    if (!stream) default_queue.wait();
    return true;
}

}  // namespace strata::kernels
