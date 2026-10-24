// include/strata/kernels/numerics_fp8_fp4.hpp - Fast E4M3 / E2M1 FP8 / FP4 Numerics & Block Quantization for Intel Xe2 SYCL 2020.
//
// Ports the DeepSeek-V4.1 low-bit numerics and block quantization/dequantization routines
// (src/dsv41/ds4_numerics.cuh and src/dsv41/ds4_kernels.cu) to pure SYCL 2020.
//
// Optimized for Intel Xe2 execution units (16-wide SIMD / SIMD16 sub-groups):
//   1. Sub-group register reductions (sycl::reduce_over_group)
//   2. Sub-group data permutes / shuffles (sycl::permute_group_by_xor)
//   3. Half-precision bit-trick decode for E2M1 FP4 via e2m1_val14
//   4. Power-of-two UE8M0 block scaling for exact multiplier restoration
//   5. Multi-block instruction interleaving for maximum DRAM bandwidth saturation
#pragma once

#include <sycl/sycl.hpp>
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/kv_stream.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace strata::kernels {

// ============================================================================
// Low-bit Numerics & Bit Manipulations
// ============================================================================

/// 2^ceil(log2(x)) for a positive normal float x (the reference fast_round_scale).
inline float pow2_ceil(float x) {
    const uint32_t b = sycl::bit_cast<uint32_t>(x);
    const int e = (int)((b >> 23) & 0xffu) - 127 + ((b & 0x7fffffu) != 0u ? 1 : 0);
    return sycl::bit_cast<float>((uint32_t)(e + 127) << 23);
}

/// A UE8M0 scale byte's value (the byte is the F32 exponent: 2^(b - 127); 0 -> 0.0f).
inline float ue8m0_val(uint32_t b) {
    return sycl::bit_cast<float>(b << 23);
}

/// Extract the UE8M0 scale byte from a power-of-two scale float.
inline uint8_t ue8m0_scale_byte(float s) {
    return static_cast<uint8_t>(sycl::bit_cast<uint32_t>(s) >> 23);
}

/// v rounded to the nearest E4M3 value (ties to even), saturated at +-448.
inline float e4m3_round(float v) {
    const float a = sycl::fabs(v);
    if (!(a == a)) return v; // NaN
    float r;
    if (a >= 448.0f) {
        r = 448.0f;
    } else {
        const int e = (int)((sycl::bit_cast<uint32_t>(a) >> 23) & 0xffu) - 127; // binade
        const float step = (e < -6) ? 0.001953125f : sycl::bit_cast<float>((uint32_t)(e - 3 + 127) << 23);
        r = (a == 0.0f) ? 0.0f : sycl::rint(a / step) * step;
        r = sycl::fmin(r, 448.0f);
    }
    return sycl::copysign(r, v);
}

/// OCP FP8 E4M3 round-to-nearest-even.
inline float fp8_rt(float v) {
    return e4m3_round(v);
}

/// The E4M3 code of v / 2^k for a quotient that is an E4M3 value (|v / 2^k| <= 448).
inline uint32_t e4m3_code(float v, int k = 0) {
    const uint32_t b = sycl::bit_cast<uint32_t>(v), sg = (b >> 24) & 0x80u, a = b & 0x7fffffffu;
    if (a == 0u) return sg;
    const int e = (int)(a >> 23) - 127 - k; // the quotient's binade
    if (e >= -6) return sg | ((uint32_t)(e + 7) << 3) | ((a >> 20) & 7u);
    return sg | (((a & 0x7fffffu) | 0x800000u) >> (14 - e)); // subnormal: m * 2^-9
}

/// An E4M3 code's value decoded to exact F32.
inline float e4m3_value(uint8_t q) {
    const uint32_t e = (q >> 3) & 15u, m = q & 7u;
    float v;
    if (e == 15u && m == 7u) v = sycl::bit_cast<float>(0x7fc00000u); // NaN
    else if (e == 0u) v = (float)m * 0.001953125f; // m * 2^-9
    else v = sycl::bit_cast<float>(((e + 120u) << 23) | (m << 20)); // (1 + m / 8) * 2^(e - 7)
    return (q & 0x80u) ? -v : v;
}

/// E2M1 magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (nearest, ties to even).
inline float e2m1_rt(float v) {
    const float a = sycl::fabs(v);
    float r;
    if (a <= 0.25f) r = 0.0f;
    else if (a < 0.75f) r = 0.5f;
    else if (a <= 1.25f) r = 1.0f;
    else if (a < 1.75f) r = 1.5f;
    else if (a <= 2.5f) r = 2.0f;
    else if (a < 3.5f) r = 3.0f;
    else if (a <= 5.0f) r = 4.0f;
    else r = 6.0f;
    return sycl::copysign(r, v);
}

/// The E2M1 4-bit code of a value e2m1_rt returned.
/// Bit 3: sign bit; bits 2..0: magnitude index (0..7 for 0, 0.5, 1, 1.5, 2, 3, 4, 6).
inline uint32_t e2m1_code(float q) {
    const float a = sycl::fabs(q);
    const uint32_t m = (a <= 1.5f) ? static_cast<uint32_t>(a * 2.0f) :
                       (a <= 2.0f) ? 4u :
                       (a <= 3.0f) ? 5u :
                       (a <= 4.0f) ? 6u : 7u;
    return m | ((sycl::bit_cast<uint32_t>(q) >> 31) << 3);
}

/// An E2M1 code's value times 2^-14 via half-precision FP16 bit trick.
/// The code's sign, exponent, and mantissa map directly to FP16 bits:
///   bit 3 (sign) -> bit 15
///   bits 2..0 (mag) -> bits 11..9
/// Value is decoded by multiplying the result with (16384.0f * scale).
inline float e2m1_val14(uint32_t c) {
    const uint16_t h_bits = static_cast<uint16_t>(((c & 8u) << 12) | ((c & 7u) << 9));
    const sycl::half h = sycl::bit_cast<sycl::half>(h_bits);
    return static_cast<float>(h);
}

// ============================================================================
// Buffer Size Calculation Helpers
// ============================================================================

/// Bytes required for packed FP8 E4M3 row/buffer: N codes + N/block_size scales.
inline constexpr size_t fp8_packed_bytes(size_t num_elements, int block_size = 32) {
    const size_t num_blocks = (num_elements + block_size - 1) / block_size;
    return num_elements + num_blocks;
}

/// Bytes required for packed FP4 E2M1 row/buffer: N/2 codes + N/block_size scales.
inline constexpr size_t fp4_packed_bytes(size_t num_elements, int block_size = 32) {
    const size_t num_blocks = (num_elements + block_size - 1) / block_size;
    return (num_elements + 1) / 2 + num_blocks;
}

// ============================================================================
// Vectorized Intel Xe2 SIMD16 Kernels
// ============================================================================

/// Quantize F32 to FP8 E4M3 with block size 32 (split buffers: codes and scales).
template <int BlocksPerSg = 4>
inline sycl::event quantize_fp8_e4m3(const float* in, uint8_t* out_codes, uint8_t* out_scales,
                                    size_t num_elements, sycl::queue& q) {
    if (num_elements == 0) return q.submit([&](sycl::handler&) {});
    constexpr int WG_SIZE = 128;
    constexpr int SG_SIZE = 16;
    constexpr int BLK_SIZE = 32;
    const size_t num_blocks = (num_elements + BLK_SIZE - 1) / BLK_SIZE;
    constexpr size_t SGS_PER_WG = WG_SIZE / SG_SIZE;
    constexpr size_t BLOCKS_PER_WG = SGS_PER_WG * BlocksPerSg;
    const size_t num_wgs = (num_blocks + BLOCKS_PER_WG - 1) / BLOCKS_PER_WG;

    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_wgs * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                auto sg = item.get_sub_group();
                const size_t sg_in_wg = item.get_local_id(0) / SG_SIZE;
                const size_t base_blk = (size_t)item.get_group(0) * BLOCKS_PER_WG + sg_in_wg * BlocksPerSg;
                const unsigned lane = sg.get_local_linear_id();

                #pragma unroll
                for (int b = 0; b < BlocksPerSg; ++b) {
                    const size_t block_idx = base_blk + b;
                    if (block_idx >= num_blocks) break;

                    const size_t base_idx = block_idx * BLK_SIZE + 2 * lane;
                    float v0 = 0.0f;
                    float v1 = 0.0f;
                    if (base_idx + 1 < num_elements) {
                        v0 = in[base_idx];
                        v1 = in[base_idx + 1];
                    } else if (base_idx < num_elements) {
                        v0 = in[base_idx];
                    }

                    // Sub-group max reduction across the 32 elements in the block
                    const float local_max = sycl::fmax(sycl::fabs(v0), sycl::fabs(v1));
                    const float block_max = sycl::reduce_over_group(sg, local_max, sycl::maximum<float>());

                    // Power-of-two scale calculation
                    const float s = pow2_ceil(sycl::fmax(block_max, 1e-4f) * (1.0f / 448.0f));
                    const float inv_s = 1.0f / s;

                    // Quantize and code
                    const float q0 = fp8_rt(sycl::clamp(v0 * inv_s, -448.0f, 448.0f));
                    const float q1 = fp8_rt(sycl::clamp(v1 * inv_s, -448.0f, 448.0f));
                    const uint8_t c0 = static_cast<uint8_t>(e4m3_code(q0));
                    const uint8_t c1 = static_cast<uint8_t>(e4m3_code(q1));

                    if (base_idx + 1 < num_elements) {
                        out_codes[base_idx] = c0;
                        out_codes[base_idx + 1] = c1;
                    } else if (base_idx < num_elements) {
                        out_codes[base_idx] = c0;
                    }

                    if (lane == 0) {
                        out_scales[block_idx] = ue8m0_scale_byte(s);
                    }
                }
            });
    });
}

/// Quantize F32 to FP8 E4M3 with block size 32 (unified packed buffer).
template <int BlocksPerSg = 4>
inline sycl::event quantize_fp8_e4m3(const float* in, uint8_t* out_packed,
                                    size_t num_elements, sycl::queue& q) {
    uint8_t* out_codes = out_packed;
    uint8_t* out_scales = out_packed + num_elements;
    return quantize_fp8_e4m3<BlocksPerSg>(in, out_codes, out_scales, num_elements, q);
}

/// Dequantize FP8 E4M3 with block size 32 to F32 or FP16 (split buffers).
template <typename OutT = float, int BlocksPerSg = 4>
inline sycl::event dequantize_fp8_e4m3(const uint8_t* in_codes, const uint8_t* in_scales,
                                      OutT* out, size_t num_elements, sycl::queue& q) {
    if (num_elements == 0) return q.submit([&](sycl::handler&) {});
    constexpr int WG_SIZE = 128;
    constexpr int SG_SIZE = 16;
    constexpr int BLK_SIZE = 32;
    const size_t num_blocks = (num_elements + BLK_SIZE - 1) / BLK_SIZE;
    constexpr size_t SGS_PER_WG = WG_SIZE / SG_SIZE;
    constexpr size_t BLOCKS_PER_WG = SGS_PER_WG * BlocksPerSg;
    const size_t num_wgs = (num_blocks + BLOCKS_PER_WG - 1) / BLOCKS_PER_WG;

    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_wgs * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                auto sg = item.get_sub_group();
                const size_t sg_in_wg = item.get_local_id(0) / SG_SIZE;
                const size_t base_blk = (size_t)item.get_group(0) * BLOCKS_PER_WG + sg_in_wg * BlocksPerSg;
                const unsigned lane = sg.get_local_linear_id();

                #pragma unroll
                for (int b = 0; b < BlocksPerSg; ++b) {
                    const size_t block_idx = base_blk + b;
                    if (block_idx >= num_blocks) break;

                    const size_t base_idx = block_idx * BLK_SIZE + 2 * lane;
                    uint8_t sc_byte = in_scales[block_idx];
                    sc_byte = sycl::group_broadcast(sg, sc_byte, 0);
                    const float s = ue8m0_val(sc_byte);

                    if (base_idx + 1 < num_elements) {
                        const uint8_t c0 = in_codes[base_idx];
                        const uint8_t c1 = in_codes[base_idx + 1];
                        const float v0 = e4m3_value(c0) * s;
                        const float v1 = e4m3_value(c1) * s;
                        out[base_idx] = static_cast<OutT>(v0);
                        out[base_idx + 1] = static_cast<OutT>(v1);
                    } else if (base_idx < num_elements) {
                        const uint8_t c0 = in_codes[base_idx];
                        const float v0 = e4m3_value(c0) * s;
                        out[base_idx] = static_cast<OutT>(v0);
                    }
                }
            });
    });
}

/// Dequantize FP8 E4M3 with block size 32 to F32 or FP16 (unified packed buffer).
template <typename OutT = float, int BlocksPerSg = 4>
inline sycl::event dequantize_fp8_e4m3(const uint8_t* in_packed, OutT* out,
                                      size_t num_elements, sycl::queue& q) {
    const uint8_t* in_codes = in_packed;
    const uint8_t* in_scales = in_packed + num_elements;
    return dequantize_fp8_e4m3<OutT, BlocksPerSg>(in_codes, in_scales, out, num_elements, q);
}

/// Quantize F32 to FP4 E2M1 with block size 16 or 32 (split buffers: codes and scales).
template <int BlockSize = 32, int BlocksPerSg = 4>
inline sycl::event quantize_fp4_e2m1(const float* in, uint8_t* out_packed, uint8_t* out_scales,
                                    size_t num_elements, sycl::queue& q) {
    static_assert(BlockSize == 16 || BlockSize == 32, "BlockSize must be 16 or 32");
    if (num_elements == 0) return q.submit([&](sycl::handler&) {});
    constexpr int WG_SIZE = 128;
    constexpr int SG_SIZE = 16;
    const size_t num_blocks = (num_elements + BlockSize - 1) / BlockSize;
    constexpr size_t SGS_PER_WG = WG_SIZE / SG_SIZE;
    constexpr size_t BLOCKS_PER_WG = SGS_PER_WG * BlocksPerSg;
    const size_t num_wgs = (num_blocks + BLOCKS_PER_WG - 1) / BLOCKS_PER_WG;

    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_wgs * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                auto sg = item.get_sub_group();
                const size_t sg_in_wg = item.get_local_id(0) / SG_SIZE;
                const size_t base_blk = (size_t)item.get_group(0) * BLOCKS_PER_WG + sg_in_wg * BlocksPerSg;
                const unsigned lane = sg.get_local_linear_id();

                #pragma unroll
                for (int b = 0; b < BlocksPerSg; ++b) {
                    const size_t block_idx = base_blk + b;
                    if (block_idx >= num_blocks) break;

                    if constexpr (BlockSize == 32) {
                        const size_t base_idx = block_idx * 32 + 2 * lane;
                        float v0 = 0.0f;
                        float v1 = 0.0f;
                        if (base_idx + 1 < num_elements) {
                            v0 = in[base_idx];
                            v1 = in[base_idx + 1];
                        } else if (base_idx < num_elements) {
                            v0 = in[base_idx];
                        }

                        const float local_max = sycl::fmax(sycl::fabs(v0), sycl::fabs(v1));
                        const float block_max = sycl::reduce_over_group(sg, local_max, sycl::maximum<float>());

                        const float s = pow2_ceil(sycl::fmax(block_max, 6.0f * 1.1754943508222875e-38f) * (1.0f / 6.0f));
                        const float inv_s = 1.0f / s;

                        const float q0 = e2m1_rt(sycl::clamp(v0 * inv_s, -6.0f, 6.0f));
                        const float q1 = e2m1_rt(sycl::clamp(v1 * inv_s, -6.0f, 6.0f));
                        const uint32_t c0 = e2m1_code(q0);
                        const uint32_t c1 = e2m1_code(q1);

                        const uint8_t packed_byte = static_cast<uint8_t>(c0 | (c1 << 4));
                        const size_t byte_idx = block_idx * 16 + lane;
                        if (byte_idx < (num_elements + 1) / 2) {
                            out_packed[byte_idx] = packed_byte;
                        }

                        if (lane == 0) {
                            out_scales[block_idx] = ue8m0_scale_byte(s);
                        }
                    } else {
                        // BlockSize == 16: 1 block per sub-group
                        const size_t base_idx = block_idx * 16 + lane;
                        float v = (base_idx < num_elements) ? in[base_idx] : 0.0f;

                        const float block_max = sycl::reduce_over_group(sg, sycl::fabs(v), sycl::maximum<float>());
                        const float s = pow2_ceil(sycl::fmax(block_max, 6.0f * 1.1754943508222875e-38f) * (1.0f / 6.0f));
                        const float inv_s = 1.0f / s;

                        const float q_val = e2m1_rt(sycl::clamp(v * inv_s, -6.0f, 6.0f));
                        const uint32_t c = e2m1_code(q_val);

                        const uint32_t c_hi = sycl::permute_group_by_xor(sg, c, 1);
                        if ((lane & 1) == 0) {
                            const uint8_t packed_byte = static_cast<uint8_t>(c | (c_hi << 4));
                            const size_t byte_idx = block_idx * 8 + (lane >> 1);
                            if (byte_idx < (num_elements + 1) / 2) {
                                out_packed[byte_idx] = packed_byte;
                            }
                        }

                        if (lane == 0) {
                            out_scales[block_idx] = ue8m0_scale_byte(s);
                        }
                    }
                }
            });
    });
}

/// Quantize F32 to FP4 E2M1 (unified packed buffer).
template <int BlockSize = 32, int BlocksPerSg = 4>
inline sycl::event quantize_fp4_e2m1(const float* in, uint8_t* out_packed,
                                    size_t num_elements, sycl::queue& q) {
    uint8_t* out_codes = out_packed;
    uint8_t* out_scales = out_packed + (num_elements + 1) / 2;
    return quantize_fp4_e2m1<BlockSize, BlocksPerSg>(in, out_codes, out_scales, num_elements, q);
}

/// Dequantize FP4 E2M1 with block size 16 or 32 to F32 or FP16 (split buffers).
template <int BlockSize = 32, typename OutT = float, int BlocksPerSg = 4>
inline sycl::event dequantize_fp4_e2m1(const uint8_t* in_packed, const uint8_t* in_scales,
                                      OutT* out, size_t num_elements, sycl::queue& q) {
    static_assert(BlockSize == 16 || BlockSize == 32, "BlockSize must be 16 or 32");
    if (num_elements == 0) return q.submit([&](sycl::handler&) {});
    constexpr int WG_SIZE = 128;
    constexpr int SG_SIZE = 16;
    const size_t num_blocks = (num_elements + BlockSize - 1) / BlockSize;
    constexpr size_t SGS_PER_WG = WG_SIZE / SG_SIZE;
    constexpr size_t BLOCKS_PER_WG = SGS_PER_WG * BlocksPerSg;
    const size_t num_wgs = (num_blocks + BLOCKS_PER_WG - 1) / BLOCKS_PER_WG;

    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_wgs * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                auto sg = item.get_sub_group();
                const size_t sg_in_wg = item.get_local_id(0) / SG_SIZE;
                const size_t base_blk = (size_t)item.get_group(0) * BLOCKS_PER_WG + sg_in_wg * BlocksPerSg;
                const unsigned lane = sg.get_local_linear_id();

                #pragma unroll
                for (int b = 0; b < BlocksPerSg; ++b) {
                    const size_t block_idx = base_blk + b;
                    if (block_idx >= num_blocks) break;

                    uint8_t sc_byte = in_scales[block_idx];
                    sc_byte = sycl::group_broadcast(sg, sc_byte, 0);
                    const float scale_factor = 16384.0f * ue8m0_val(sc_byte);

                    if constexpr (BlockSize == 32) {
                        const size_t byte_idx = block_idx * 16 + lane;
                        const uint8_t packed_byte = (byte_idx < (num_elements + 1) / 2) ? in_packed[byte_idx] : (uint8_t)0;
                        const uint32_t c0 = packed_byte & 0x0fu;
                        const uint32_t c1 = (packed_byte >> 4) & 0x0fu;

                        const float v0 = e2m1_val14(c0) * scale_factor;
                        const float v1 = e2m1_val14(c1) * scale_factor;

                        const size_t base_idx = block_idx * 32 + 2 * lane;
                        if (base_idx + 1 < num_elements) {
                            out[base_idx] = static_cast<OutT>(v0);
                            out[base_idx + 1] = static_cast<OutT>(v1);
                        } else if (base_idx < num_elements) {
                            out[base_idx] = static_cast<OutT>(v0);
                        }
                    } else {
                        // BlockSize == 16: 8 packed bytes per block of 16 elements
                        const size_t byte_idx = block_idx * 8 + (lane >> 1);
                        const uint8_t packed_byte = (byte_idx < (num_elements + 1) / 2) ? in_packed[byte_idx] : (uint8_t)0;
                        const uint32_t c = (lane & 1) ? ((packed_byte >> 4) & 0x0fu) : (packed_byte & 0x0fu);
                        const float v = e2m1_val14(c) * scale_factor;

                        const size_t base_idx = block_idx * 16 + lane;
                        if (base_idx < num_elements) {
                            out[base_idx] = static_cast<OutT>(v);
                        }
                    }
                }
            });
    });
}

/// Dequantize FP4 E2M1 with block size 16 or 32 (unified packed buffer).
template <int BlockSize = 32, typename OutT = float, int BlocksPerSg = 4>
inline sycl::event dequantize_fp4_e2m1(const uint8_t* in_packed, OutT* out,
                                      size_t num_elements, sycl::queue& q) {
    const uint8_t* in_codes = in_packed;
    const uint8_t* in_scales = in_packed + (num_elements + 1) / 2;
    return dequantize_fp4_e2m1<BlockSize, OutT, BlocksPerSg>(in_codes, in_scales, out, num_elements, q);
}

// ============================================================================
// Runtime Configuration Helpers (Item D)
// ============================================================================

namespace detail {
inline sycl::queue& default_numerics_queue() {
    static sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    return q;
}

inline std::atomic<int>& lowbit_kv_override() {
    static std::atomic<int> s_override{-1};
    return s_override;
}
}  // namespace detail

/// Returns active low-bit KV storage format:
///   0 = default (FP16 / INT8 / Q4_0),
///   1 = FP8 E4M3 (1.031 bytes/element),
///   2 = FP4 E2M1 (0.531 bytes/element).
/// Programmatic override via set_lowbit_kv_format takes precedence,
/// followed by STRATA_KV_FP4 and STRATA_KV_FP8 environment variables.
inline int lowbit_kv_format() {
    int val = detail::lowbit_kv_override().load(std::memory_order_relaxed);
    if (val != -1) return val;
    const char* e_fp4 = std::getenv("STRATA_KV_FP4");
    if (e_fp4 != nullptr && std::atoi(e_fp4) != 0) return 2;
    const char* e_fp8 = std::getenv("STRATA_KV_FP8");
    if (e_fp8 != nullptr && std::atoi(e_fp8) != 0) return 1;
    return 0;
}

/// Checks whether low-bit KV cache compression is currently active (format != 0).
inline bool lowbit_kv_enabled() {
    return lowbit_kv_format() != 0;
}

/// Programmatic override for low-bit KV cache format.
/// Pass 1 for FP8 E4M3, 2 for FP4 E2M1, 0 to disable, or -1 to revert to environment variables.
inline void set_lowbit_kv_format(int fmt) {
    detail::lowbit_kv_override().store(fmt, std::memory_order_relaxed);
}

// ============================================================================
// Engine Paged KV Cache Appending Kernels (Intel Xe2 SIMD16)
// ============================================================================

inline void kv_append_fp8_kernel(
    uint8_t* __restrict__ k_fp8, uint8_t* __restrict__ v_fp8,
    const int32_t* __restrict__ table, const int32_t* __restrict__ step,
    const float* __restrict__ kcur, const float* __restrict__ vcur,
    int kv_heads, int head_dim, int page_size,
    KvHostPools host, int planes,
    sycl::nd_item<1> item) {
    auto sg = item.get_sub_group();
    constexpr int SG_SIZE = 16;
    constexpr int BLK_SIZE = 32;
    const int blocks_per_head = (head_dim + BLK_SIZE - 1) / BLK_SIZE;
    const int total_blocks = planes * kv_heads * blocks_per_head;

    const size_t blk_idx = item.get_global_id(0) / SG_SIZE;
    if (blk_idx >= (size_t)total_blocks) return;

    const unsigned lane = sg.get_local_linear_id();

    const int plane = (int)(blk_idx / (kv_heads * blocks_per_head));
    const int rem = (int)(blk_idx % (kv_heads * blocks_per_head));
    const int h = rem / blocks_per_head;
    const int b = rem % blocks_per_head;
    const bool is_v = (plane == 1);

    const float* in_vec = (is_v ? vcur : kcur) + (size_t)h * head_dim;
    uint8_t* out_pool = is_v ? v_fp8 : k_fp8;

    long long row = -1;
    if (table != nullptr && step != nullptr) {
        const long long pos = (long long)*(step + kStepPos);
        const long long page = (long long)table[pos / page_size];
        if (page >= 0) {
            row = (page * kv_heads + h) * page_size + (pos % page_size);
        }
    } else {
        row = h;
    }

    const size_t base_elem = (size_t)b * BLK_SIZE + 2 * lane;
    float v0 = 0.0f;
    float v1 = 0.0f;
    if (base_elem + 1 < (size_t)head_dim) {
        v0 = in_vec[base_elem];
        v1 = in_vec[base_elem + 1];
    } else if (base_elem < (size_t)head_dim) {
        v0 = in_vec[base_elem];
    }

    const float local_max = sycl::fmax(sycl::fabs(v0), sycl::fabs(v1));
    const float block_max = sycl::reduce_over_group(sg, local_max, sycl::maximum<float>());

    const float s = pow2_ceil(sycl::fmax(block_max, 1e-4f) * (1.0f / 448.0f));
    const float inv_s = 1.0f / s;

    const float q0 = fp8_rt(sycl::clamp(v0 * inv_s, -448.0f, 448.0f));
    const float q1 = fp8_rt(sycl::clamp(v1 * inv_s, -448.0f, 448.0f));
    const uint8_t c0 = static_cast<uint8_t>(e4m3_code(q0));
    const uint8_t c1 = static_cast<uint8_t>(e4m3_code(q1));
    const uint8_t sc_byte = ue8m0_scale_byte(s);

    const size_t row_bytes = fp8_packed_bytes(head_dim, BLK_SIZE);

    if (row >= 0 && out_pool != nullptr) {
        uint8_t* row_base = out_pool + row * row_bytes;
        if (base_elem + 1 < (size_t)head_dim) {
            row_base[base_elem] = c0;
            row_base[base_elem + 1] = c1;
        } else if (base_elem < (size_t)head_dim) {
            row_base[base_elem] = c0;
        }
        if (lane == 0) {
            row_base[head_dim + b] = sc_byte;
        }
    }

    if (host.present() && step != nullptr) {
        uint8_t* host_pool = reinterpret_cast<uint8_t*>(is_v ? (host.v_pool ? (void*)host.v_pool : (void*)host.v_q4)
                                                              : (host.k_pool ? (void*)host.k_pool : (void*)host.k_q4));
        if (host_pool != nullptr) {
            const long long pos = (long long)*(step + kStepPos);
            const long long host_row = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
            uint8_t* host_base = host_pool + host_row * row_bytes;
            if (base_elem + 1 < (size_t)head_dim) {
                host_base[base_elem] = c0;
                host_base[base_elem + 1] = c1;
            } else if (base_elem < (size_t)head_dim) {
                host_base[base_elem] = c0;
            }
            if (lane == 0) {
                host_base[head_dim + b] = sc_byte;
            }
        }
    }
}

inline void kv_append_fp4_kernel(
    uint8_t* __restrict__ k_fp4, uint8_t* __restrict__ v_fp4,
    const int32_t* __restrict__ table, const int32_t* __restrict__ step,
    const float* __restrict__ kcur, const float* __restrict__ vcur,
    int kv_heads, int head_dim, int page_size,
    KvHostPools host, int planes,
    sycl::nd_item<1> item) {
    auto sg = item.get_sub_group();
    constexpr int SG_SIZE = 16;
    constexpr int BLK_SIZE = 32;
    const int blocks_per_head = (head_dim + BLK_SIZE - 1) / BLK_SIZE;
    const int total_blocks = planes * kv_heads * blocks_per_head;

    const size_t blk_idx = item.get_global_id(0) / SG_SIZE;
    if (blk_idx >= (size_t)total_blocks) return;

    const unsigned lane = sg.get_local_linear_id();

    const int plane = (int)(blk_idx / (kv_heads * blocks_per_head));
    const int rem = (int)(blk_idx % (kv_heads * blocks_per_head));
    const int h = rem / blocks_per_head;
    const int b = rem % blocks_per_head;
    const bool is_v = (plane == 1);

    const float* in_vec = (is_v ? vcur : kcur) + (size_t)h * head_dim;
    uint8_t* out_pool = is_v ? v_fp4 : k_fp4;

    long long row = -1;
    if (table != nullptr && step != nullptr) {
        const long long pos = (long long)*(step + kStepPos);
        const long long page = (long long)table[pos / page_size];
        if (page >= 0) {
            row = (page * kv_heads + h) * page_size + (pos % page_size);
        }
    } else {
        row = h;
    }

    const size_t base_elem = (size_t)b * BLK_SIZE + 2 * lane;
    float v0 = 0.0f;
    float v1 = 0.0f;
    if (base_elem + 1 < (size_t)head_dim) {
        v0 = in_vec[base_elem];
        v1 = in_vec[base_elem + 1];
    } else if (base_elem < (size_t)head_dim) {
        v0 = in_vec[base_elem];
    }

    const float local_max = sycl::fmax(sycl::fabs(v0), sycl::fabs(v1));
    const float block_max = sycl::reduce_over_group(sg, local_max, sycl::maximum<float>());

    const float s = pow2_ceil(sycl::fmax(block_max, 6.0f * 1.1754943508222875e-38f) * (1.0f / 6.0f));
    const float inv_s = 1.0f / s;

    const float q0 = e2m1_rt(sycl::clamp(v0 * inv_s, -6.0f, 6.0f));
    const float q1 = e2m1_rt(sycl::clamp(v1 * inv_s, -6.0f, 6.0f));
    const uint32_t c0 = e2m1_code(q0);
    const uint32_t c1 = e2m1_code(q1);
    const uint8_t packed_byte = static_cast<uint8_t>(c0 | (c1 << 4));
    const uint8_t sc_byte = ue8m0_scale_byte(s);

    const size_t row_bytes = fp4_packed_bytes(head_dim, BLK_SIZE);
    const size_t code_bytes = (head_dim + 1) / 2;

    if (row >= 0 && out_pool != nullptr) {
        uint8_t* row_base = out_pool + row * row_bytes;
        const size_t byte_idx = (size_t)b * 16 + lane;
        if (byte_idx < code_bytes) {
            row_base[byte_idx] = packed_byte;
        }
        if (lane == 0) {
            row_base[code_bytes + b] = sc_byte;
        }
    }

    if (host.present() && step != nullptr) {
        uint8_t* host_pool = reinterpret_cast<uint8_t*>(is_v ? (host.v_pool ? (void*)host.v_pool : (void*)host.v_q4)
                                                              : (host.k_pool ? (void*)host.k_pool : (void*)host.k_q4));
        if (host_pool != nullptr) {
            const long long pos = (long long)*(step + kStepPos);
            const long long host_row = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
            uint8_t* host_base = host_pool + host_row * row_bytes;
            const size_t byte_idx = (size_t)b * 16 + lane;
            if (byte_idx < code_bytes) {
                host_base[byte_idx] = packed_byte;
            }
            if (lane == 0) {
                host_base[code_bytes + b] = sc_byte;
            }
        }
    }
}

// ============================================================================
// Engine APIs: kv_append_fp8_step & kv_append_fp4_step
// ============================================================================

/// Appends K and V activation rows into paged FP8 E4M3 KV cache (1.031 bytes/element).
inline void kv_append_fp8_step(
    uint8_t* k_fp8, uint8_t* v_fp8,
    const int32_t* page_table, const int32_t* step,
    const float* kcur, const float* vcur,
    const QsaShapes& s,
    sycl::queue& q,
    const KvHostPools* host = nullptr) {
    if (s.head_dim <= 0 || s.n_head_kv <= 0) return;
    const int planes = (v_fp8 == nullptr || vcur == nullptr) ? 1 : 2;
    constexpr int SG_SIZE = 16;
    constexpr int WG_SIZE = 128;
    constexpr int SGS_PER_WG = WG_SIZE / SG_SIZE;
    const int blocks_per_head = ((int)s.head_dim + 31) / 32;
    const int total_blocks = planes * (int)s.n_head_kv * blocks_per_head;
    const int num_wgs = (total_blocks + SGS_PER_WG - 1) / SGS_PER_WG;

    KvHostPools host_pools = host ? *host : KvHostPools{};
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_wgs * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                kv_append_fp8_kernel(k_fp8, v_fp8, page_table, step, kcur, vcur,
                                     (int)s.n_head_kv, (int)s.head_dim, (int)s.page_size,
                                     host_pools, planes, item);
            });
    });
}

inline void kv_append_fp8_step(
    uint8_t* k_fp8, uint8_t* v_fp8,
    const int32_t* page_table, const int32_t* step,
    const float* kcur, const float* vcur,
    const QsaShapes& s,
    void* stream = nullptr,
    const KvHostPools* host = nullptr) {
    sycl::queue& q = stream ? *static_cast<sycl::queue*>(stream) : detail::default_numerics_queue();
    kv_append_fp8_step(k_fp8, v_fp8, page_table, step, kcur, vcur, s, q, host);
}

inline void kv_append_fp8_step(
    uint8_t* k_fp8, uint8_t* v_fp8,
    const int32_t* page_table, const int32_t* step,
    const float* kcur, const float* vcur,
    const QsaShapes& s,
    sycl::queue* stream,
    const KvHostPools* host = nullptr) {
    kv_append_fp8_step(k_fp8, v_fp8, page_table, step, kcur, vcur, s, static_cast<void*>(stream), host);
}

template <typename KPtr, typename VPtr,
          std::enable_if_t<!std::is_same_v<KPtr, const float*>, int> = 0>
inline void kv_append_fp8_step(
    KPtr k_fp8, VPtr v_fp8,
    const int32_t* page_table, const int32_t* step,
    const float* kcur, const float* vcur,
    const QsaShapes& s,
    void* stream = nullptr,
    const KvHostPools* host = nullptr) {
    kv_append_fp8_step(reinterpret_cast<uint8_t*>(k_fp8), reinterpret_cast<uint8_t*>(v_fp8),
                       page_table, step, kcur, vcur, s, stream, host);
}

template <typename KPtr, typename VPtr,
          std::enable_if_t<!std::is_same_v<KPtr, const float*>, int> = 0>
inline void kv_append_fp8_step(
    KPtr k_fp8, VPtr v_fp8,
    const int32_t* page_table, const int32_t* step,
    const float* kcur, const float* vcur,
    const QsaShapes& s,
    sycl::queue& q,
    const KvHostPools* host = nullptr) {
    kv_append_fp8_step(reinterpret_cast<uint8_t*>(k_fp8), reinterpret_cast<uint8_t*>(v_fp8),
                       page_table, step, kcur, vcur, s, q, host);
}

template <typename KPtr, typename VPtr,
          std::enable_if_t<!std::is_same_v<KPtr, const float*>, int> = 0>
inline void kv_append_fp8_step(
    KPtr k_fp8, VPtr v_fp8,
    const int32_t* page_table, const int32_t* step,
    const float* kcur, const float* vcur,
    const QsaShapes& s,
    sycl::queue* stream,
    const KvHostPools* host = nullptr) {
    kv_append_fp8_step(reinterpret_cast<uint8_t*>(k_fp8), reinterpret_cast<uint8_t*>(v_fp8),
                       page_table, step, kcur, vcur, s, static_cast<void*>(stream), host);
}

/// Direct activation row compression helper into FP8 E4M3 packed buffer.
inline sycl::event kv_append_fp8_step(
    const float* kcur, const float* vcur,
    uint8_t* k_out, uint8_t* v_out,
    size_t num_elements,
    sycl::queue& q) {
    sycl::event e1, e2;
    if (kcur && k_out && num_elements > 0) {
        e1 = quantize_fp8_e4m3(kcur, k_out, num_elements, q);
    }
    if (vcur && v_out && num_elements > 0) {
        e2 = quantize_fp8_e4m3(vcur, v_out, num_elements, q);
    }
    return (vcur && v_out) ? e2 : e1;
}

inline void kv_append_fp8_step(
    const float* kcur, const float* vcur,
    uint8_t* k_out, uint8_t* v_out,
    size_t num_elements,
    void* stream = nullptr) {
    sycl::queue& q = stream ? *static_cast<sycl::queue*>(stream) : detail::default_numerics_queue();
    kv_append_fp8_step(kcur, vcur, k_out, v_out, num_elements, q);
}

inline void kv_append_fp8_step(
    const float* kcur, const float* vcur,
    uint8_t* k_out, uint8_t* v_out,
    size_t num_elements,
    sycl::queue* stream) {
    kv_append_fp8_step(kcur, vcur, k_out, v_out, num_elements, static_cast<void*>(stream));
}

/// Appends K and V activation rows into paged FP4 E2M1 KV cache (0.531 bytes/element).
inline void kv_append_fp4_step(
    uint8_t* k_fp4, uint8_t* v_fp4,
    const int32_t* page_table, const int32_t* step,
    const float* kcur, const float* vcur,
    const QsaShapes& s,
    sycl::queue& q,
    const KvHostPools* host = nullptr) {
    if (s.head_dim <= 0 || s.n_head_kv <= 0) return;
    const int planes = (v_fp4 == nullptr || vcur == nullptr) ? 1 : 2;
    constexpr int SG_SIZE = 16;
    constexpr int WG_SIZE = 128;
    constexpr int SGS_PER_WG = WG_SIZE / SG_SIZE;
    const int blocks_per_head = ((int)s.head_dim + 31) / 32;
    const int total_blocks = planes * (int)s.n_head_kv * blocks_per_head;
    const int num_wgs = (total_blocks + SGS_PER_WG - 1) / SGS_PER_WG;

    KvHostPools host_pools = host ? *host : KvHostPools{};
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_wgs * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                kv_append_fp4_kernel(k_fp4, v_fp4, page_table, step, kcur, vcur,
                                     (int)s.n_head_kv, (int)s.head_dim, (int)s.page_size,
                                     host_pools, planes, item);
            });
    });
}

inline void kv_append_fp4_step(
    uint8_t* k_fp4, uint8_t* v_fp4,
    const int32_t* page_table, const int32_t* step,
    const float* kcur, const float* vcur,
    const QsaShapes& s,
    void* stream = nullptr,
    const KvHostPools* host = nullptr) {
    sycl::queue& q = stream ? *static_cast<sycl::queue*>(stream) : detail::default_numerics_queue();
    kv_append_fp4_step(k_fp4, v_fp4, page_table, step, kcur, vcur, s, q, host);
}

inline void kv_append_fp4_step(
    uint8_t* k_fp4, uint8_t* v_fp4,
    const int32_t* page_table, const int32_t* step,
    const float* kcur, const float* vcur,
    const QsaShapes& s,
    sycl::queue* stream,
    const KvHostPools* host = nullptr) {
    kv_append_fp4_step(k_fp4, v_fp4, page_table, step, kcur, vcur, s, static_cast<void*>(stream), host);
}

template <typename KPtr, typename VPtr,
          std::enable_if_t<!std::is_same_v<KPtr, const float*>, int> = 0>
inline void kv_append_fp4_step(
    KPtr k_fp4, VPtr v_fp4,
    const int32_t* page_table, const int32_t* step,
    const float* kcur, const float* vcur,
    const QsaShapes& s,
    void* stream = nullptr,
    const KvHostPools* host = nullptr) {
    kv_append_fp4_step(reinterpret_cast<uint8_t*>(k_fp4), reinterpret_cast<uint8_t*>(v_fp4),
                       page_table, step, kcur, vcur, s, stream, host);
}

template <typename KPtr, typename VPtr,
          std::enable_if_t<!std::is_same_v<KPtr, const float*>, int> = 0>
inline void kv_append_fp4_step(
    KPtr k_fp4, VPtr v_fp4,
    const int32_t* page_table, const int32_t* step,
    const float* kcur, const float* vcur,
    const QsaShapes& s,
    sycl::queue& q,
    const KvHostPools* host = nullptr) {
    kv_append_fp4_step(reinterpret_cast<uint8_t*>(k_fp4), reinterpret_cast<uint8_t*>(v_fp4),
                       page_table, step, kcur, vcur, s, q, host);
}

template <typename KPtr, typename VPtr,
          std::enable_if_t<!std::is_same_v<KPtr, const float*>, int> = 0>
inline void kv_append_fp4_step(
    KPtr k_fp4, VPtr v_fp4,
    const int32_t* page_table, const int32_t* step,
    const float* kcur, const float* vcur,
    const QsaShapes& s,
    sycl::queue* stream,
    const KvHostPools* host = nullptr) {
    kv_append_fp4_step(reinterpret_cast<uint8_t*>(k_fp4), reinterpret_cast<uint8_t*>(v_fp4),
                       page_table, step, kcur, vcur, s, static_cast<void*>(stream), host);
}

/// Direct activation row compression helper into FP4 E2M1 packed buffer.
inline sycl::event kv_append_fp4_step(
    const float* kcur, const float* vcur,
    uint8_t* k_out, uint8_t* v_out,
    size_t num_elements,
    sycl::queue& q) {
    sycl::event e1, e2;
    if (kcur && k_out && num_elements > 0) {
        e1 = quantize_fp4_e2m1<32>(kcur, k_out, num_elements, q);
    }
    if (vcur && v_out && num_elements > 0) {
        e2 = quantize_fp4_e2m1<32>(vcur, v_out, num_elements, q);
    }
    return (vcur && v_out) ? e2 : e1;
}

inline void kv_append_fp4_step(
    const float* kcur, const float* vcur,
    uint8_t* k_out, uint8_t* v_out,
    size_t num_elements,
    void* stream = nullptr) {
    sycl::queue& q = stream ? *static_cast<sycl::queue*>(stream) : detail::default_numerics_queue();
    kv_append_fp4_step(kcur, vcur, k_out, v_out, num_elements, q);
}

inline void kv_append_fp4_step(
    const float* kcur, const float* vcur,
    uint8_t* k_out, uint8_t* v_out,
    size_t num_elements,
    sycl::queue* stream) {
    kv_append_fp4_step(kcur, vcur, k_out, v_out, num_elements, static_cast<void*>(stream));
}

} // namespace strata::kernels
