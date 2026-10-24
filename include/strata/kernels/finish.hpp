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
#include "strata/kernels/numerics_fp8_fp4.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

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

struct FinishArgs {
    const float* w = nullptr;
    float eps = 1e-6f;
    int rd = 0;
    int pos = 0;
    int inverse = 0;
    int vpp = 1;
    int pos_step = 1;
    const float* freqs = nullptr; // rd / 2 values
    float theta = 10000.0f;       // Base frequency if freqs == nullptr
    int quant = 0;                // 0: none/BF16, 1: FP8 E4M3
    float* out = nullptr;         // F32 destination (if null, in-place on x)
    long long out_stride = 0;     // F32 destination stride (if 0, uses stride)
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
                float* out_vec = (f.out != nullptr)
                    ? (f.out + (long long)v * (f.out_stride != 0 ? f.out_stride : stride))
                    : const_cast<float*>(in_vec);

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
                            const float freq = (f.freqs != nullptr)
                                ? f.freqs[i]
                                : sycl::pow(f.theta, -2.0f * (float)i / (float)f.rd);
                            const float ang = (float)pos * freq;
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
                   const FinishArgs& f, void* stream = nullptr) {
    static sycl::queue default_queue{sycl::default_selector_v, sycl::property::queue::in_order()};
    sycl::queue& q = stream ? *static_cast<sycl::queue*>(stream) : default_queue;

    if (f.pdst != nullptr) {
        launch_finish_kernel<true>(x, n_vec, len, stride, f, q);
    } else {
        launch_finish_kernel<false>(x, n_vec, len, stride, f, q);
    }
}

inline void finish(const float* x, int n_vec, int len, long long stride,
                   const FinishArgs& f, sycl::queue* stream) {
    finish(x, n_vec, len, stride, f, static_cast<void*>(stream));
}

inline void finish(const float* x, int n_vec, int len, long long stride,
                   const FinishArgs& f, sycl::queue& q) {
    finish(x, n_vec, len, stride, f, static_cast<void*>(&q));
}

namespace detail {
inline std::atomic<int>& fused_finish_override() {
    static std::atomic<int> s_override{-1};
    return s_override;
}
}  // namespace detail

inline bool fused_finish_enabled() {
    int val = detail::fused_finish_override().load(std::memory_order_relaxed);
    if (val != -1) return val != 0;
    static const bool env = [] {
        const char* e = std::getenv("STRATA_FUSED_FINISH");
        return (e != nullptr && std::atoi(e) != 0);
    }();
    return env;
}

inline void set_fused_finish_enabled(bool enabled) {
    detail::fused_finish_override().store(enabled ? 1 : 0, std::memory_order_relaxed);
}

inline bool fused_finish_usable(int len) {
    return len > 0 && len <= (kFinMax * kFinThreads);
}

// ----------------------------------------------------------------------------
// Engine API: finish_norm_rope & finish_norm_rope_quant
// ----------------------------------------------------------------------------

// Fuses RMSNorm and RoPE in-place (out == nullptr or out == x) or into output buffer out.
inline void finish_norm_rope(
    const float* x,
    float* out,
    int n_vec,
    int len,
    long long stride,
    const float* w = nullptr,
    float eps = 1e-6f,
    int rd = 0,
    const float* freqs = nullptr,
    int pos = 0,
    int inverse = 0,
    int vpp = 1,
    int pos_step = 1,
    long long out_stride = 0,
    void* stream = nullptr) {
    FinishArgs args;
    args.w = w;
    args.eps = eps;
    args.rd = rd;
    args.pos = pos;
    args.inverse = inverse;
    args.vpp = vpp;
    args.pos_step = pos_step;
    args.freqs = freqs;
    args.quant = 0;
    args.out = out;
    args.out_stride = out_stride != 0 ? out_stride : stride;
    args.dst = nullptr;
    args.pdst = nullptr;
    finish(x, n_vec, len, stride, args, stream);
}

inline void finish_norm_rope(
    const float* x,
    float* out,
    int n_vec,
    int len,
    long long stride,
    const float* w,
    float eps,
    int rd,
    const float* freqs,
    int pos,
    int inverse,
    int vpp,
    int pos_step,
    long long out_stride,
    sycl::queue& q) {
    finish_norm_rope(x, out, n_vec, len, stride, w, eps, rd, freqs, pos, inverse, vpp, pos_step, out_stride, static_cast<void*>(&q));
}

// Fuses RMSNorm and RoPE with direct BF16 destination writeback from registers.
inline void finish_norm_rope(
    const float* x,
    uint16_t* dst,
    int n_vec,
    int len,
    long long stride,
    long long dst_stride,
    const float* w = nullptr,
    float eps = 1e-6f,
    int rd = 0,
    const float* freqs = nullptr,
    int pos = 0,
    int inverse = 0,
    int vpp = 1,
    int pos_step = 1,
    void* stream = nullptr) {
    FinishArgs args;
    args.w = w;
    args.eps = eps;
    args.rd = rd;
    args.pos = pos;
    args.inverse = inverse;
    args.vpp = vpp;
    args.pos_step = pos_step;
    args.freqs = freqs;
    args.quant = 0;
    args.out = nullptr;
    args.dst = dst;
    args.dst_stride = dst_stride;
    args.pdst = nullptr;
    finish(x, n_vec, len, stride, args, stream);
}

inline void finish_norm_rope(
    const float* x,
    uint16_t* dst,
    int n_vec,
    int len,
    long long stride,
    long long dst_stride,
    const float* w,
    float eps,
    int rd,
    const float* freqs,
    int pos,
    int inverse,
    int vpp,
    int pos_step,
    sycl::queue& q) {
    finish_norm_rope(x, dst, n_vec, len, stride, dst_stride, w, eps, rd, freqs, pos, inverse, vpp, pos_step, static_cast<void*>(&q));
}

inline void finish_norm_rope_bf16(
    const float* x,
    uint16_t* dst,
    int n_vec,
    int len,
    long long stride,
    long long dst_stride,
    const float* w = nullptr,
    float eps = 1e-6f,
    int rd = 0,
    const float* freqs = nullptr,
    int pos = 0,
    int inverse = 0,
    int vpp = 1,
    int pos_step = 1,
    void* stream = nullptr) {
    finish_norm_rope(x, dst, n_vec, len, stride, dst_stride, w, eps, rd, freqs, pos, inverse, vpp, pos_step, stream);
}

// Fuses RMSNorm, RoPE, and FP8 E4M3 packed quantization (codes + group scales) into pdst.
inline void finish_norm_rope_quant(
    const float* x,
    uint8_t* pdst,
    int n_vec,
    int len,
    long long stride,
    long long pdst_stride,
    const float* w = nullptr,
    float eps = 1e-6f,
    int rd = 0,
    const float* freqs = nullptr,
    int pos = 0,
    int inverse = 0,
    int vpp = 1,
    int pos_step = 1,
    void* stream = nullptr) {
    FinishArgs args;
    args.w = w;
    args.eps = eps;
    args.rd = rd;
    args.pos = pos;
    args.inverse = inverse;
    args.vpp = vpp;
    args.pos_step = pos_step;
    args.freqs = freqs;
    args.quant = 1;
    args.out = nullptr;
    args.dst = nullptr;
    args.pdst = pdst;
    args.pdst_stride = pdst_stride;
    finish(x, n_vec, len, stride, args, stream);
}

inline void finish_norm_rope_quant(
    const float* x,
    uint8_t* pdst,
    int n_vec,
    int len,
    long long stride,
    long long pdst_stride,
    const float* w,
    float eps,
    int rd,
    const float* freqs,
    int pos,
    int inverse,
    int vpp,
    int pos_step,
    sycl::queue& q) {
    finish_norm_rope_quant(x, pdst, n_vec, len, stride, pdst_stride, w, eps, rd, freqs, pos, inverse, vpp, pos_step, static_cast<void*>(&q));
}

}  // namespace strata::kernels
