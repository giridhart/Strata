// include/strata/kernels/finish.hpp - Fused Finish kernel (RMSNorm + RoPE + Activation Quantization) for SYCL 2020 on Intel Xe2.
//
// Ports the DeepSeek-V4.1 finish kernel pattern (src/dsv41/ds4_kernels.cu lines 135-245) to pure SYCL 2020.
// Fuses:
//   1. RMSNorm: vector load into registers, workgroup sum-of-squares reduction, rsqrt(mean(x^2) + eps) * w
//   2. RoPE: rotates adjacent complex pairs (re, im) using sub-group shuffles (sycl::permute_group_by_xor)
//   3. Activation Quantization / Output Writeback: FP8 E4M3 (group-32 scale), BF16, or FP16 directly from registers.
#pragma once

#include <sycl/sycl.hpp>
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace strata::kernels {

constexpr int kFinMax = 32;       // Maximum values per thread: len <= 32 * 128 = 4096 (or 32 * 256 = 8192)
constexpr int kFinThreads = 128;   // Work-group size tuned for Intel Xe2 sub-groups (8x SIMD16 or 4x SIMD32)

// Quantization format constants matching DeepSeek-V4.1 cache format
constexpr int kKvBf16 = 0;
constexpr int kKvFp8 = 1;

inline constexpr size_t finish_kv_row_bytes(int fmt, int d) {
    return fmt == kKvBf16 ? (size_t)d * 2
                          : ((size_t)d + (size_t)d / 32 + 15) / 16 * 16;
}

// 2^ceil(log2(x)) for positive normal x (reference fast_round_scale)
inline float pow2_ceil(float x) {
    const uint32_t b = sycl::bit_cast<uint32_t>(x);
    const int e = (int)((b >> 23) & 0xffu) - 127 + ((b & 0x7fffffu) != 0u ? 1 : 0);
    return sycl::bit_cast<float>((uint32_t)(e + 127) << 23);
}

// v rounded to the nearest E4M3 value (ties to even), saturated at +-448
inline float e4m3_round(float v) {
    const float a = sycl::fabs(v);
    if (!(a == a)) return v;
    float r;
    if (a >= 448.0f) {
        r = 448.0f;
    } else {
        const int e = (int)((sycl::bit_cast<uint32_t>(a) >> 23) & 0xffu) - 127;
        const float step = e < -6 ? 0.001953125f : sycl::bit_cast<float>((uint32_t)(e - 3 + 127) << 23);
        r = (a == 0.0f) ? 0.0f : sycl::rint(a / step) * step;
        r = sycl::fmin(r, 448.0f);
    }
    return v < 0.0f ? -r : r;
}

inline float fp8_rt(float v) { return e4m3_round(v); }

// E4M3 code of v / 2^k for a quotient that is an E4M3 value
inline uint32_t e4m3_code(float v, int k = 0) {
    const uint32_t b = sycl::bit_cast<uint32_t>(v), sg = (b >> 24) & 0x80u, a = b & 0x7fffffffu;
    if (a == 0u) return sg;
    const int e = (int)(a >> 23) - 127 - k;
    if (e >= -6) return sg | ((uint32_t)(e + 7) << 3) | ((a >> 20) & 7u);
    return sg | (((a & 0x7fffffu) | 0x800000u) >> (14 - e));
}

// E4M3 code value decoded to F32
inline float e4m3_value(uint8_t q) {
    const uint32_t e = (q >> 3) & 15u, m = q & 7u;
    float v;
    if (e == 15u && m == 7u) v = sycl::bit_cast<float>(0x7fc00000u);
    else if (e == 0u) v = (float)m * 0.001953125f;
    else v = sycl::bit_cast<float>(((e + 120u) << 23) | (m << 20));
    return (q & 0x80u) ? -v : v;
}

struct FinishArgs {
    const float* w = nullptr;
    float eps = 1e-6f;
    int rd = 0;
    int pos = 0;
    int inverse = 0;
    int vpp = 1;
    int pos_step = 1;
    const float* freqs = nullptr; // rd / 2 values
    int quant = 0;                // 0: none/BF16, 1: FP8 E4M3
    uint16_t* dst = nullptr;      // BF16 destination
    long long dst_stride = 0;
    uint8_t* pdst = nullptr;      // Packed FP8 destination
    long long pdst_stride = 0;
};

template <bool kPack>
inline void launch_finish_kernel(const float* x, int n_vec, int len, long long stride,
                                 FinishArgs f, sycl::queue& q) {
    if (n_vec <= 0 || len <= 0) return;
    if (len > kFinMax * kFinThreads) {
        std::fprintf(stderr, "finish: len %d exceeds maximum %d\n", len, kFinMax * kFinThreads);
        return;
    }

    const sycl::range<1> global_range(n_vec * kFinThreads);
    const sycl::range<1> local_range(kFinThreads);

    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> slm_red(sycl::range<1>(32), cgh);
        sycl::local_accessor<float, 1> slm_tot(sycl::range<1>(1), cgh);
        sycl::local_accessor<float, 1> slm_max(sycl::range<1>(32), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(global_range, local_range),
            [=](sycl::nd_item<1> item) {
                const int v = item.get_group(0);
                const float* in_vec = x + (long long)v * stride;
                float* out_vec = const_cast<float*>(in_vec);

                auto sg = item.get_sub_group();
                const int tid = item.get_local_id(0);
                const unsigned sg_size = sg.get_local_range()[0];
                const int sg_id = tid / sg_size;
                const int lane = sg.get_local_linear_id();

                float r[kFinMax];
                const int nj = (len + kFinThreads - 1) / kFinThreads;

                // 1. RMSNorm: Load into registers & accumulate sum-of-squares
                float ss = 0.0f;
#pragma unroll
                for (int j = 0; j < kFinMax; ++j) {
                    if (j >= nj) break;
                    const int e = tid + j * kFinThreads;
                    r[j] = (e < len) ? in_vec[e] : 0.0f;
                    ss += r[j] * r[j];
                }

                // Workgroup reduction for RMSNorm
                ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
                if (lane == 0) {
                    slm_red[sg_id] = ss;
                }
                sycl::group_barrier(item.get_group());

                if (sg_id == 0) {
                    const unsigned num_sgs = kFinThreads / sg_size;
                    float block_ss = ((unsigned)lane < num_sgs) ? slm_red[lane] : 0.0f;
                    block_ss = sycl::reduce_over_group(sg, block_ss, sycl::plus<float>());
                    if (lane == 0) {
                        slm_tot[0] = block_ss;
                    }
                }
                sycl::group_barrier(item.get_group());

                const float inv = sycl::rsqrt(slm_tot[0] / (float)len + f.eps);

#pragma unroll
                for (int j = 0; j < kFinMax; ++j) {
                    if (j >= nj) break;
                    const int e = tid + j * kFinThreads;
                    if (e < len) {
                        r[j] = (f.w != nullptr) ? (r[j] * inv * f.w[e]) : (r[j] * inv);
                    }
                }

                // 2. RoPE: Rotates adjacent complex pairs (re, im) using sub-group shuffles
                if (f.rd > 0) {
                    const int r0 = len - f.rd;
                    const int pos = f.pos + (int)(v / f.vpp) * f.pos_step;
#pragma unroll
                    for (int j = 0; j < kFinMax; ++j) {
                        if (j >= nj) break;
                        const int e = tid + j * kFinThreads;
                        const float other = sycl::permute_group_by_xor(sg, r[j], 1);
                        if (e >= r0 && e < len) {
                            const int i = (e - r0) >> 1;
                            const float ang = (float)pos * f.freqs[i];
                            const float c = sycl::cos(ang);
                            const float sn = f.inverse ? -sycl::sin(ang) : sycl::sin(ang);
                            if (((e - r0) & 1) == 0) {
                                r[j] = r[j] * c - other * sn;
                            } else {
                                r[j] = other * sn + r[j] * c;
                            }
                        }
                    }
                }

                // 3. Quantization & Output writeback
                if constexpr (kPack) {
                    // DeepSeek-V4.1 FP8 E4M3 packed row format (codes + group scales)
                    uint8_t* pr = f.pdst + (long long)v * f.pdst_stride;
                    constexpr int g = 32; // group size 32
#pragma unroll
                    for (int j = 0; j < kFinMax; ++j) {
                        if (j >= nj) break;
                        const int e = tid + j * kFinThreads;
                        float a = (e < len) ? sycl::fabs(r[j]) : 0.0f;

                        if (sg_size == 32) {
                            for (int o = 16; o > 0; o >>= 1) {
                                a = sycl::fmax(a, sycl::permute_group_by_xor(sg, a, o));
                            }
                        } else {
                            for (int o = 8; o > 0; o >>= 1) {
                                a = sycl::fmax(a, sycl::permute_group_by_xor(sg, a, o));
                            }
                            if (lane == 0) slm_max[sg_id] = a;
                            sycl::group_barrier(item.get_group());
                            a = sycl::fmax(a, slm_max[sg_id ^ 1]);
                        }

                        float q = 0.0f, s = 1.0f;
                        if (e < len) {
                            s = pow2_ceil(sycl::fmax(a, 1e-4f) * (1.0f / 448.0f));
                            q = fp8_rt(sycl::clamp(r[j] / s, -448.0f, 448.0f));
                            r[j] = q * s;
                            pr[e] = (uint8_t)e4m3_code(q);
                            if (e % g == 0) {
                                pr[len + e / g] = (uint8_t)(sycl::bit_cast<uint32_t>(s) >> 23);
                            }
                        }
                    }
                } else if (f.quant == 1) {
                    // FP8 in-place rounding
                    [[maybe_unused]] constexpr int g = 32;
#pragma unroll
                    for (int j = 0; j < kFinMax; ++j) {
                        if (j >= nj) break;
                        const int e = tid + j * kFinThreads;
                        float a = (e < len) ? sycl::fabs(r[j]) : 0.0f;
                        if (sg_size == 32) {
                            for (int o = 16; o > 0; o >>= 1) {
                                a = sycl::fmax(a, sycl::permute_group_by_xor(sg, a, o));
                            }
                        } else {
                            for (int o = 8; o > 0; o >>= 1) {
                                a = sycl::fmax(a, sycl::permute_group_by_xor(sg, a, o));
                            }
                            if (lane == 0) slm_max[sg_id] = a;
                            sycl::group_barrier(item.get_group());
                            a = sycl::fmax(a, slm_max[sg_id ^ 1]);
                        }
                        if (e < len) {
                            const float s = pow2_ceil(sycl::fmax(a, 1e-4f) * (1.0f / 448.0f));
                            r[j] = fp8_rt(sycl::clamp(r[j] / s, -448.0f, 448.0f)) * s;
                        }
                    }
                }

                // Writeback to dst (BF16) or out_vec (F32)
                uint16_t* d = (f.dst != nullptr) ? (f.dst + (long long)v * f.dst_stride) : nullptr;
#pragma unroll
                for (int j = 0; j < kFinMax; ++j) {
                    if (j >= nj) break;
                    const int e = tid + j * kFinThreads;
                    if (e >= len) continue;
                    if (d != nullptr) {
                        d[e] = bf16_from_f32(r[j]);
                    } else if constexpr (!kPack) {
                        out_vec[e] = r[j];
                    }
                }
            });
    });
}

inline void finish(const float* x, int n_vec, int len, long long stride,
                   const FinishArgs& f, sycl::queue* stream = nullptr) {
    static sycl::queue default_queue{sycl::default_selector_v, sycl::property::queue::in_order()};
    sycl::queue& q = stream ? *stream : default_queue;

    if (f.pdst != nullptr) {
        launch_finish_kernel<true>(x, n_vec, len, stride, f, q);
    } else {
        launch_finish_kernel<false>(x, n_vec, len, stride, f, q);
    }
}

}  // namespace strata::kernels
