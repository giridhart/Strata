// src/kernels/cuda/dequant_bf16.cu - see include/strata/kernels/dequant_bf16.hpp.
//
// Arithmetic transcribed from ggml/src/ggml-quants.c at the pinned llama.cpp (MIT License, Copyright (c) 2023-2026
// The ggml authors): dequantize_row_q2_0/q4_0/q5_0/q8_0/q3_K/q4_K/q5_K/q6_K/iq4_nl/iq4_xs.
#include <sycl/sycl.hpp>
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

static sycl::queue& default_queue() {
    static sycl::queue q{sycl::property::queue::in_order()};
    return q;
}

inline float h2f(const uint8_t* p) {
    return sycl::vec<sycl::half, 1>(sycl::bit_cast<sycl::half, unsigned short>((uint16_t)(p[0] | (p[1] << 8))))
        .convert<float, sycl::rounding_mode::automatic>()[0];
}

inline uint16_t f2bf(float f) {
    uint32_t u = sycl::bit_cast<unsigned int>(f);
    // a NaN (a NaN scale in the block) stays a quiet NaN, as in ggml_compute_fp32_to_bf16 and `bf16_from_f32`:
    // the rounding add below would carry it into -0 or inf
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 64u);
    u += 0x7fffu + ((u >> 16) & 1u);  // round to nearest even
    return (uint16_t)(u >> 16);
}

inline void put(uint16_t* o, int i, float v) { o[i] = f2bf(v); }

struct H16 {
    uint16_t v;
};

inline void put(H16* o, int i, float v) {
    o[i].v = sycl::bit_cast<unsigned short, sycl::half>(
        sycl::vec<float, 1>(v).convert<sycl::half, sycl::rounding_mode::rte>()[0]);
}

inline void put(float* o, int i, float v) { o[i] = v; }

static constexpr int8_t kv_iq4nl_table[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

inline void scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// One 32-element group `g` (row-major over the whole slice); `out` points at that group's 32 outputs.
template <int TYPE, typename T>
inline void group32(const uint8_t* row_blocks, int gi_in_row, T* out) {
    if constexpr (TYPE == 42) {  // Q2_0: 64 per block of 18 B
        const uint8_t* b = row_blocks + (size_t)(gi_in_row / 2) * 18;
        const float d = h2f(b);
        const int e0 = (gi_in_row % 2) * 32;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const int e = e0 + j;
            const int q = (b[2 + e / 4] >> ((e % 4) * 2)) & 3;
            put(out, j, (float)(q - 1) * d);
        }
    } else if constexpr (TYPE == 2) {  // Q4_0
        const uint8_t* b = row_blocks + (size_t)gi_in_row * 18;
        const float d = h2f(b);
        for (int j = 0; j < 16; ++j) {
            put(out, j, (float)((b[2 + j] & 0x0F) - 8) * d);
            put(out, j + 16, (float)((b[2 + j] >> 4) - 8) * d);
        }
    } else if constexpr (TYPE == 6) {  // Q5_0
        const uint8_t* b = row_blocks + (size_t)gi_in_row * 22;
        const float d = h2f(b);
        const uint32_t qh = (uint32_t)b[2] | ((uint32_t)b[3] << 8) | ((uint32_t)b[4] << 16) | ((uint32_t)b[5] << 24);
        for (int j = 0; j < 16; ++j) {
            const int xh0 = ((qh >> j) << 4) & 0x10;
            const int xh1 = (qh >> (j + 12)) & 0x10;
            put(out, j, (float)(((b[6 + j] & 0x0F) | xh0) - 16) * d);
            put(out, j + 16, (float)(((b[6 + j] >> 4) | xh1) - 16) * d);
        }
    } else if constexpr (TYPE == 7) {  // Q5_1
        const uint8_t* b = row_blocks + (size_t)gi_in_row * 24;
        const float d = h2f(b), m = h2f(b + 2);
        const uint32_t qh = (uint32_t)b[4] | ((uint32_t)b[5] << 8) | ((uint32_t)b[6] << 16) | ((uint32_t)b[7] << 24);
        for (int j = 0; j < 16; ++j) {
            const int xh0 = ((qh >> j) & 1) << 4;
            const int xh1 = ((qh >> (j + 16)) & 1) << 4;
            put(out, j, (float)((b[8 + j] & 15) | xh0) * d + m);
            put(out, j + 16, (float)((b[8 + j] >> 4) | xh1) * d + m);
        }
    } else if constexpr (TYPE == 8) {  // Q8_0
        const uint8_t* b = row_blocks + (size_t)gi_in_row * 34;
        const float d = h2f(b);
        for (int j = 0; j < 32; ++j)
            put(out, j, (float)(int8_t)b[2 + j] * d);
    } else if constexpr (TYPE == 20) {  // IQ4_NL
        const uint8_t* b = row_blocks + (size_t)gi_in_row * 18;
        const float d = h2f(b);
        for (int j = 0; j < 16; ++j) {
            put(out, j, d * (float)kv_iq4nl_table[b[2 + j] & 0xf]);
            put(out, j + 16, d * (float)kv_iq4nl_table[b[2 + j] >> 4]);
        }
    } else if constexpr (TYPE == 11) {  // Q3_K: hmask[32] qs[64] scales[12] d
        const uint8_t* b = row_blocks + (size_t)(gi_in_row / 8) * 110;
        const int gi = gi_in_row % 8, n = gi / 4, jj = gi % 4;
        const uint8_t* hm = b;
        const uint8_t* q = b + 32 + n * 32;
        const uint8_t* sc = b + 96;
        const float d_all = h2f(b + 108);
        uint32_t aux[4];
        for (int i = 0; i < 3; ++i) {
            aux[i] = (uint32_t)sc[i * 4] | ((uint32_t)sc[i * 4 + 1] << 8) | ((uint32_t)sc[i * 4 + 2] << 16) |
                     ((uint32_t)sc[i * 4 + 3] << 24);
        }
        const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu, tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
        const int shift = 2 * jj;
        const uint8_t m = (uint8_t)(1u << (n * 4 + jj));
        for (int t = 0; t < 32; ++t) {
            const int is = n * 8 + jj * 2 + (t >= 16 ? 1 : 0);
            const float dl = d_all * (float)(scales[is] - 32);
            put(out, t, dl * (float)((int)((q[t] >> shift) & 3) - ((hm[t] & m) ? 0 : 4)));
        }
    } else if constexpr (TYPE == 12) {  // Q4_K: d dmin scales[12] qs[128]
        const uint8_t* b = row_blocks + (size_t)(gi_in_row / 8) * 144;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        scale_min_k4(gi, b + 4, sc, m);
        const float d1 = d * (float)sc, m1 = dmin * (float)m;
        const uint8_t* q = b + 16 + 32 * j64;
        for (int l = 0; l < 32; ++l)
            put(out, l, d1 * (float)(hi ? (q[l] >> 4) : (q[l] & 0xF)) - m1);
    } else if constexpr (TYPE == 13) {  // Q5_K: d dmin scales[12] qh[32] qs[128]
        const uint8_t* b = row_blocks + (size_t)(gi_in_row / 8) * 176;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        scale_min_k4(gi, b + 4, sc, m);
        const float d1 = d * (float)sc, m1 = dmin * (float)m;
        const uint8_t* qh = b + 16;
        const uint8_t* ql = b + 48 + 32 * j64;
        const uint8_t u = (uint8_t)(1u << (2 * j64 + hi));
        for (int l = 0; l < 32; ++l) {
            const int nib = hi ? (ql[l] >> 4) : (ql[l] & 0xF);
            put(out, l, d1 * (float)(nib + ((qh[l] & u) ? 16 : 0)) - m1);
        }
    } else if constexpr (TYPE == 14) {  // Q6_K: ql[128] qh[64] scales[16] d
        const uint8_t* b = row_blocks + (size_t)(gi_in_row / 8) * 210;
        const int gi = gi_in_row % 8, n = gi / 4, qu = gi % 4;
        const uint8_t* ql = b + 64 * n;
        const uint8_t* qh = b + 128 + 32 * n;
        const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192) + 8 * n;
        const float d = h2f(b + 208);
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            int q;
            if (qu == 0)
                q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
            else if (qu == 1)
                q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            else if (qu == 2)
                q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
            else
                q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
            put(out, l, d * (float)sc[is + 2 * qu] * (float)(q - 32));
        }
    } else if constexpr (TYPE == 23) {  // IQ4_XS: d scales_h scales_l[4] qs[128]
        const uint8_t* b = row_blocks + (size_t)(gi_in_row / 8) * 136;
        const int ib = gi_in_row % 8;
        const float d = h2f(b);
        const uint16_t scales_h = (uint16_t)(b[2] | (b[3] << 8));
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (float)(ls - 32);
        const uint8_t* qs = b + 8 + 16 * ib;
        for (int j = 0; j < 16; ++j) {
            put(out, j, dl * (float)kv_iq4nl_table[qs[j] & 0xf]);
            put(out, j + 16, dl * (float)kv_iq4nl_table[qs[j] >> 4]);
        }
    }
}

template <int TYPE, typename T>
void dequant_kernel(const uint8_t* __restrict__ blocks, int64_t row_bytes, int64_t row0, int64_t rows,
                    int64_t groups_per_row, T* __restrict__ out, sycl::nd_item<1> item) {
    const int64_t g = (int64_t)item.get_global_id(0);
    if (g >= rows * groups_per_row) return;
    const int64_t r = g / groups_per_row, gi = g % groups_per_row;
    group32<TYPE>(blocks + (row0 + r) * row_bytes, (int)gi, out + r * groups_per_row * 32 + gi * 32);
}

// Q8_0 with four threads per block, 8 values each: a subgroup of 32 reads 8 consecutive blocks and writes 512
// consecutive bytes (a thread per block wrote 2-byte values 64 bytes apart: scattered DRAM stores)
template <typename T>
void dequant_q8_0_kernel(const uint8_t* __restrict__ blocks, int64_t row_bytes, int64_t row0, int64_t rows,
                         int64_t groups_per_row, T* __restrict__ out, const sycl::nd_item<1>& item) {
    const int64_t i = (int64_t)item.get_global_id(0);
    if (i >= rows * groups_per_row * 4) return;
    const int64_t g = i >> 2, r = g / groups_per_row, gi = g % groups_per_row;
    const int piece = (int)(i & 3);
    const uint8_t* b = blocks + (row0 + r) * row_bytes + gi * 34;
    const float d = h2f(b);
    const uint16_t* q = (const uint16_t*)(b + 2 + 8 * piece);  // blocks are 2-byte aligned
    alignas(16) T v[8];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint16_t w = q[k];
        put(v, 2 * k, (float)(int8_t)(w & 0xFF) * d);
        put(v, 2 * k + 1, (float)(int8_t)(w >> 8) * d);
    }
    T* dst = out + g * 32 + 8 * piece;
    if constexpr (sizeof(T) == 2) {
        *reinterpret_cast<sycl::vec<uint32_t, 4>*>(dst) = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(v);
    } else if constexpr (sizeof(T) == 4) {
        reinterpret_cast<sycl::vec<float, 4>*>(dst)[0] = reinterpret_cast<const sycl::vec<float, 4>*>(v)[0];
        reinterpret_cast<sycl::vec<float, 4>*>(dst)[1] = reinterpret_cast<const sycl::vec<float, 4>*>(v)[1];
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) dst[k] = v[k];
    }
}

bool geometry(int type, int& block_elems, int& block_bytes) {
    switch (type) {
        case 2:
            block_elems = 32;
            block_bytes = 18;
            return true;
        case 6:
            block_elems = 32;
            block_bytes = 22;
            return true;
        case 7:
            block_elems = 32;
            block_bytes = 24;
            return true;
        case 8:
            block_elems = 32;
            block_bytes = 34;
            return true;
        case 20:
            block_elems = 32;
            block_bytes = 18;
            return true;
        case 11:
            block_elems = 256;
            block_bytes = 110;
            return true;
        case 12:
            block_elems = 256;
            block_bytes = 144;
            return true;
        case 13:
            block_elems = 256;
            block_bytes = 176;
            return true;
        case 14:
            block_elems = 256;
            block_bytes = 210;
            return true;
        case 23:
            block_elems = 256;
            block_bytes = 136;
            return true;
        case 42:
            block_elems = 64;
            block_bytes = 18;
            return true;
        default:
            return false;
    }
}

template <typename T>
void launch(int type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, T* out, void* stream) {
    int be = 0, bb = 0;
    if (!geometry(type, be, bb) || cols % be != 0 || rows <= 0) {
        std::fprintf(stderr, "dequant: unsupported type %d or shape %lld x %lld\n", type, (long long)rows,
                     (long long)cols);
        std::exit(1);
    }
    const int64_t row_bytes = cols / be * bb, gpr = cols / 32, total = rows * gpr;
    constexpr size_t THREADS = 256;
    const size_t grid = (size_t)((total + THREADS - 1) / THREADS);
    const uint8_t* p = (const uint8_t*)blocks;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();

    if (type == 8 && (reinterpret_cast<uintptr_t>(blocks) & 1) == 0 &&
        (reinterpret_cast<uintptr_t>(out) & 15) == 0) {
        const size_t total4 = (size_t)rows * gpr * 4;
        const size_t grid4 = (total4 + THREADS - 1) / THREADS;
        q->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(grid4 * THREADS), sycl::range<1>(THREADS)),
                [=](sycl::nd_item<1> item) {
                    dequant_q8_0_kernel<T>(p, row_bytes, row0, rows, gpr, out, item);
                });
        });
        return;
    }

#define STRATA_DQ(TY)                                                                                             \
    case TY: {                                                                                                    \
        q->submit([&](sycl::handler& cgh) {                                                                       \
            cgh.parallel_for(                                                                                     \
                sycl::nd_range<1>(sycl::range<1>(grid * THREADS), sycl::range<1>(THREADS)),                       \
                [=](sycl::nd_item<1> item) { dequant_kernel<TY, T>(p, row_bytes, row0, rows, gpr, out, item); }); \
        });                                                                                                       \
    } break

    switch (type) {
        STRATA_DQ(2);
        STRATA_DQ(6);
        STRATA_DQ(7);
        STRATA_DQ(8);
        STRATA_DQ(11);
        STRATA_DQ(12);
        STRATA_DQ(13);
        STRATA_DQ(14);
        STRATA_DQ(20);
        STRATA_DQ(23);
        STRATA_DQ(42);
    }
#undef STRATA_DQ
}

}  // namespace

bool dequant_bf16_supported(int ggml_type) noexcept {
    int a, b;
    return geometry(ggml_type, a, b);
}

namespace {
// plan v0.3 P6: the i-quant formats (llama.cpp's dequantizers, iq_kernels.cu)
bool iq_only(int t) { return t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 29; }
}  // namespace

void dequant_bf16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out,
                  void* stream) {
    launch<uint16_t>(ggml_type, blocks, row0, rows, cols, out, stream);
}

void dequant_f16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out,
                 void* stream) {
    if (iq_only(ggml_type)) {
        iq_dequant_f16(ggml_type, (const uint8_t*)blocks + (size_t)row0 * iq_row_bytes(ggml_type, cols), rows * cols,
                       out, stream);
        return;
    }
    launch<H16>(ggml_type, blocks, row0, rows, cols, reinterpret_cast<H16*>(out), stream);
}

void dequant_f32(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, float* out,
                 void* stream) {
    if (iq_only(ggml_type)) {
        iq_dequant_f32(ggml_type, (const uint8_t*)blocks + (size_t)row0 * iq_row_bytes(ggml_type, cols), rows * cols,
                       out, stream);
        return;
    }
    launch<float>(ggml_type, blocks, row0, rows, cols, out, stream);
}

}  // namespace strata::kernels
