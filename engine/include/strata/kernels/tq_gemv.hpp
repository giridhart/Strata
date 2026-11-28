// include/strata/kernels/tq_gemv.hpp - Trellis Quantization (TQ) decoders, FWHT Hadamard rotations, and GEMV for Intel Xe2.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include "strata/kernels/tq_lut.hpp"

namespace strata::kernels {

constexpr int QK_TQ2_T = 128;
constexpr int QK_TQK = 128;
constexpr uint32_t TQ_STATE_BITS = 15;
constexpr uint32_t TQ_STATE_MASK = ((1u << TQ_STATE_BITS) - 1u);

#pragma pack(push, 1)
struct block_tq2_t {
    uint16_t d;               // fp16 scale
    uint8_t  qs[QK_TQ2_T / 4]; // 32 bytes (one trellis byte per 4 weights)
};
static_assert(sizeof(block_tq2_t) == 34, "wrong block_tq2_t size");

struct block_tqk6 {
    uint16_t d;               // fp16 scale
    uint8_t  qs[4 * 6];       // 24 bytes (32 steps x 6 bits)
};
static_assert(sizeof(block_tqk6) == 26, "wrong block_tqk6 size");

struct block_tqk7 {
    uint16_t d;               // fp16 scale
    uint8_t  qs[4 * 7];       // 28 bytes (32 steps x 7 bits)
};
static_assert(sizeof(block_tqk7) == 30, "wrong block_tqk7 size");
#pragma pack(pop)

template <int TY> struct TqInfo;
template <> struct TqInfo<144> { using block_t = block_tq2_t; static constexpr int K = 8; };
template <> struct TqInfo<145> { using block_t = block_tqk6;  static constexpr int K = 6; };
template <> struct TqInfo<146> { using block_t = block_tqk7;  static constexpr int K = 7; };

inline bool is_tq_type(int type) noexcept {
    return type == 144 || type == 145 || type == 146;
}

constexpr size_t tq_block_size(int type) noexcept {
    switch (type) {
        case 144: return sizeof(block_tq2_t); // 34
        case 145: return sizeof(block_tqk6);  // 26
        case 146: return sizeof(block_tqk7);  // 30
        default:  return 0;
    }
}

constexpr int tq_kbits(int type) noexcept {
    switch (type) {
        case 144: return 8;
        case 145: return 6;
        case 146: return 7;
        default:  return 0;
    }
}

/// Bit-exact FP16 to FP32 conversion portable across host and device.
inline float tq_half_to_float(uint16_t h) noexcept {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu, man = h & 0x3ffu, bits;
    if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (man << 13);
    } else if (exp != 0) {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    } else if (man == 0) {
        bits = sign;
    } else {
        const float m = (float)man * (1.0f / 16777216.0f);
        return sign ? -m : m;
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

/// Unpacks the 4 circular trellis states for step group l in [0, 7] (steps 4l..4l+3).
template <int TY>
inline void tq_states4(const uint8_t* __restrict__ qs, int l, uint32_t s[4]) noexcept {
    const uint16_t* qs16 = reinterpret_cast<const uint16_t*>(qs);
    if constexpr (TY == 144) {
        const uint32_t w01 = qs16[2 * l + 0], w23 = qs16[2 * l + 1];
        const uint32_t prev = qs[(4 * l + 31) & 31];
        const uint32_t b0 = w01 & 0xFF, b1 = w01 >> 8, b2 = w23 & 0xFF, b3 = w23 >> 8;
        s[0] = (prev << 8) | b0;
        s[1] = (b0 << 8) | b1;
        s[2] = (b1 << 8) | b2;
        s[3] = (b2 << 8) | b3;
    } else {
        constexpr int K = TqInfo<TY>::K, nwords = 2 * K;
        const int off3 = (28 - 4 * l) * K, wb = off3 >> 4, r = off3 & 15;
        const int wi1 = wb + 1 >= nwords ? wb + 1 - nwords : wb + 1;
        const int wi2 = wb + 2 >= nwords ? wb + 2 - nwords : wb + 2;
        const int wi3 = wb + 3 >= nwords ? wb + 3 - nwords : wb + 3;
        const uint64_t v = static_cast<uint64_t>(qs16[wb]) |
                           (static_cast<uint64_t>(qs16[wi1]) << 16) |
                           (static_cast<uint64_t>(qs16[wi2]) << 32) |
                           (static_cast<uint64_t>(qs16[wi3]) << 48);
        for (int i = 0; i < 4; ++i) {
            s[i] = static_cast<uint32_t>((v >> (r + (3 - i) * K)) & 0xFFFFu);
        }
    }
}

// -----------------------------------------------------------------------------
// Host-callable kernel entry points
// -----------------------------------------------------------------------------

/// In-place or out-of-place Fast Walsh-Hadamard Transform (FWHT128 / FWHT256) in SLM with sign modulation.
void hadamard_rows(const float* in, float* out, int64_t rows, int64_t cols, int block,
                   const float* signs, void* stream = nullptr);

/// Dequantize Trellis Quantization blocks (144, 145, 146) to FP32 using the FP16 codebook (tq_lut_f16).
void dequantize_tq(int type, const void* vx, float* y, int64_t n, void* stream = nullptr);

/// Native Xe2 SIMD16 subgroup Trellis GEMV (w * x -> y), supporting SLM codebook staging or global memory lookup.
void tq_gemv(int type, const void* w, const float* x, float* y, int64_t n_out, int64_t n_in,
             void* stream = nullptr);

void tq_gemv_ext(int type, const void* w, const float* x, float* y, int64_t n_out, int64_t n_in,
                 bool stage_slm, void* stream = nullptr);

}  // namespace strata::kernels
