// src/kernels/sycl/quantize_act.cpp - P2.S2: quantize an activation the way ggml converts src1.
//
// WHY THIS EXISTS AT ALL.  `ggml_mul_mat` converts the ACTIVATION (src1) to the weight's `vec_dot_type`
// before the dot product - the rule established over rounds 119-136 and written into `ref/quant.py` as
// `apply_src1`.  The CPU expert path already honours it (`bench/micro/cpu_s2.cpp` quantizes to int8 and uses
// `vpdpbusd`), and its P0.T2 parity passes at 1.461e-06.  The GPU kernels added in rounds 166-177 take FP16
// activations instead.  **Left that way, the two paths would compute different numbers for different experts
// of the SAME token** - not a tolerance question, an inconsistency inside one forward pass.
//
// Q2_0 IS THE CASE THAT MATTERS: the routed experts are all Q2_0, `Q2_0 -> Q8_0`, and that is 31.64 GiB of the
// 38 GiB pack.  Q8_K (for the K-quants and IQ4_XS) is the other half of the table and is not here yet.
//
// THE THREE SUBTLETIES, each of which was paid for once already and is transcribed rather than recalled:
//
//   1. `d32 = amax / 127.0f` is computed in FP32, and the QUANTIZED INTEGERS divide by **d32**, not by the
//      fp16-rounded value that gets stored.  Conflating the two shifts every quant near a rounding boundary -
//      measured against ggml's own bytes, 18 of 80 blocks differed, and because Q2_0 experts convert to Q8_0
//      the error landed in all 48 MoE blocks and made the end-to-end KL WORSE.
//   2. The reference divides in FLOAT64 (`blk.astype(np.float64) / float(np.float32(d32))`) and rounds with
//      `rint` - half to even.  This kernel therefore divides in double too: an FP32 division followed by
//      `rintf` differs from the reference wherever the f32 quotient rounds across a .5 boundary.
//   3. The DEQUANTIZED value is `q * d16`, the fp16 scale, not `q * d32`.  The block stores fp16 and that is
//      what a reader multiplies by.
#include <sycl/sycl.hpp>
#include <cstdint>
#include <algorithm>
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int QK8_0 = 32;
constexpr int QK_K = 256;
constexpr int Q8K_BYTES = 292;

/// ggml's `nearest_int` (ggml-quants.c L621), transcribed rather than replaced.
///
/// The magic number is 1.5 * 2^23: adding it forces the mantissa's integer part into the low bits, and the
/// mask/subtract recover it.  The reason it is transcribed and not written as `rintf` is that the two differ
/// on exact ties - and that is the whole point of the constant, so a "cleaner" rewrite would silently change
/// which way ties go.
inline int nearest_int_dev(float fval) {
    const float val = fval + 12582912.0f;
    int i;
    memcpy(&i, &val, 4);
    return (i & 0x007fffff) - 0x00400000;
}

/// `block_q8_0` for one 32-element block, ggml's bytes: `d32 = amax/127` in fp32, `d` stored fp16, the
/// QUANTIZED INTEGERS divided by `d32` (not by the stored fp16), and `rint` in DOUBLE.
inline void quantize_q8_0_block(const float* __restrict__ xb, uint8_t* __restrict__ out) {
    float amax = 0.0f;
    for (int i = 0; i < QK8_0; ++i)
        amax = sycl::fmax(amax, sycl::fabs(xb[i]));
    if (amax == 0.0f) {
        const uint16_t zb = f16_from_f32(0.0f);
        out[0] = (uint8_t)(zb & 0xFF);
        out[1] = (uint8_t)(zb >> 8);
        for (int i = 0; i < QK8_0; ++i)
            out[2 + i] = 0;
        return;
    }
    const float d32 = amax / 127.0f;
    const uint16_t d16bits = f16_from_f32(d32);
    out[0] = (uint8_t)(d16bits & 0xFF);
    out[1] = (uint8_t)(d16bits >> 8);

    for (int i = 0; i < QK8_0; ++i) {
        double q = sycl::rint((double)xb[i] / (double)d32);
        if (q > 127.0) q = 127.0;
        if (q < -128.0) q = -128.0;
        out[2 + i] = (uint8_t)(int8_t)q;
    }
}

/// `block_q8_0` bytes PLUS the fp32 scale and the CPU's rounding rule (round half away from zero).
inline void quantize_q8_0_scaled_block(const float* __restrict__ xb, uint8_t* __restrict__ out,
                                       float* __restrict__ scale_out) {
    float amax = 0.0f;
    for (int i = 0; i < QK8_0; ++i)
        amax = sycl::fmax(amax, sycl::fabs(xb[i]));
    const float s = amax > 0.f ? amax / 127.f : 0.f;
    const float inv = s > 0.f ? 1.f / s : 0.f;
    *scale_out = s;

    const uint16_t d16bits = f16_from_f32(s);
    out[0] = (uint8_t)(d16bits & 0xFF);
    out[1] = (uint8_t)(d16bits >> 8);
    for (int i = 0; i < QK8_0; ++i) {
        const float t = xb[i] * inv;
        const float r = t + (t >= 0.f ? 0.5f : -0.5f);
        int v = (int)r;
        v = v < -127 ? -127 : (v > 127 ? 127 : v);
        out[2 + i] = (uint8_t)(int8_t)v;
    }
}

/// `block_q8_K` for one 256-element block.
inline void quantize_q8_K_block(const float* __restrict__ xb, uint8_t* __restrict__ out) {
    float* d = (float*)out;
    int8_t* qs = (int8_t*)(out + 4);
    int16_t* bsums = (int16_t*)(out + 4 + QK_K);

    float max = 0.0f, amax = 0.0f;
    for (int j = 0; j < QK_K; ++j) {
        const float ax = sycl::fabs(xb[j]);
        if (ax > amax) {
            amax = ax;
            max = xb[j];
        }
    }
    if (amax == 0.0f) {
        *d = 0.0f;
        for (int j = 0; j < QK_K; ++j)
            qs[j] = 0;
        for (int j = 0; j < QK_K / 16; ++j)
            bsums[j] = 0;
        return;
    }
    volatile double vd_max = max;
    volatile double vd_scale_num = -127.0;
    const float iscale = (float)(vd_scale_num / vd_max);
    volatile float v_iscale = iscale;
    for (int j = 0; j < QK_K; ++j) {
        volatile float prod = v_iscale * xb[j];
        const int v = nearest_int_dev(prod);
        qs[j] = (int8_t)sycl::min(127, v);
    }
    for (int j = 0; j < QK_K / 16; ++j) {
        int sum = 0;
        for (int ii = 0; ii < 16; ++ii)
            sum += qs[j * 16 + ii];
        bsums[j] = (int16_t)sum;
    }
    volatile float v_d = (float)(vd_max / vd_scale_num);
    *d = v_d;
}

void quantize_q8_0_kernel(sycl::nd_item<1> item, const float* __restrict__ x, uint8_t* __restrict__ blocks,
                          long long n_blocks) {
    const long long b = (long long)item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    if (b >= n_blocks) return;
    quantize_q8_0_block(x + b * QK8_0, blocks + b * 34);
}

void quantize_q8_0_scaled_kernel(sycl::nd_item<1> item, const float* __restrict__ x, uint8_t* __restrict__ blocks,
                                 float* __restrict__ scales, long long n_blocks) {
    const long long b = (long long)item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    if (b >= n_blocks) return;
    quantize_q8_0_scaled_block(x + b * QK8_0, blocks + b * 34, scales + b);
}

void dequant_q8_0_kernel(sycl::nd_item<1> item, const uint8_t* __restrict__ blocks, float* __restrict__ x,
                         long long n_blocks) {
    const long long b = (long long)item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    if (b >= n_blocks) return;
    const uint8_t* blk = blocks + b * 34;
    const uint16_t dbits = (uint16_t)(blk[0] | (blk[1] << 8));
    const float d = f32_from_f16(dbits);
    float* out = x + b * QK8_0;
    for (int i = 0; i < QK8_0; ++i)
        out[i] = (float)(int8_t)blk[2 + i] * d;
}

void quantize_q8_K_kernel(sycl::nd_item<1> item, const float* __restrict__ x, uint8_t* __restrict__ blocks,
                          long long n_blocks) {
    const long long b = (long long)item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    if (b >= n_blocks) return;
    quantize_q8_K_block(x + b * QK_K, blocks + b * Q8K_BYTES);
}

void dequant_q8_K_kernel(sycl::nd_item<1> item, const uint8_t* __restrict__ blocks, float* __restrict__ x,
                         long long n_blocks) {
    const long long b = (long long)item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    if (b >= n_blocks) return;
    const uint8_t* blk = blocks + b * Q8K_BYTES;
    float d;
    memcpy(&d, blk, 4);
    const int8_t* qs = (const int8_t*)(blk + 4);
    float* out = x + b * QK_K;
    for (int i = 0; i < QK_K; ++i)
        out[i] = (float)qs[i] * d;
}

// ===================== fused silu(gate) * up + quantize =====================

template <int KIND>
inline float swilu_apply(float g, float u) {
    if constexpr (KIND == 0) {
        const double x = (double)g;
        return (float)(x / (1.0 + sycl::exp(-x))) * u;
    } else {
        return (g / (1.0f + sycl::native::exp(-g))) * u;
    }
}

template <int KIND, bool SCALED>
void swilu_quantize_q8_0_launch(const float* gate, const float* up, float* gate_out, int64_t n,
                                uint8_t* blocks, float* scales, void* stream) {
    static sycl::queue default_queue{sycl::default_selector_v, sycl::property::queue::in_order()};
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    const long long nb = n / QK8_0;
    constexpr int SG_SIZE = strata::sycl_arch::SUB_GROUP_SIZE;
    constexpr int THREADS = 64;
    constexpr int SGS_PER_WG = THREADS / SG_SIZE;
    const unsigned blocks_wg = (unsigned)((nb + SGS_PER_WG - 1) / SGS_PER_WG);

    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> p(sycl::range<1>(SGS_PER_WG * 32), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(blocks_wg * THREADS), sycl::range<1>(THREADS)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const int sg_idx = item.get_local_id(0) / SG_SIZE;
                const int lane = item.get_local_id(0) % SG_SIZE;
                const long long b = (long long)item.get_group(0) * SGS_PER_WG + sg_idx;
                float* p_ptr = p.get_multi_ptr<sycl::access::decorated::no>().get();
                float* p_block = p_ptr + sg_idx * 32;

                if (b < nb) {
                    const long long i0 = b * QK8_0;
                    constexpr int pair_per_lane = 32 / SG_SIZE;
                    for (int k = 0; k < pair_per_lane; ++k) {
                        const int idx = lane + k * SG_SIZE;
                        const float v = swilu_apply<KIND>(gate[i0 + idx], up[i0 + idx]);
                        gate_out[i0 + idx] = v;
                        p_block[idx] = v;
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);
                if (b < nb && lane == 0) {
                    if constexpr (SCALED) {
                        quantize_q8_0_scaled_block(p_block, blocks + b * 34, scales + b);
                    } else {
                        quantize_q8_0_block(p_block, blocks + b * 34);
                    }
                }
            });
    });

    if (!stream) default_queue.wait();
}

template <int KIND>
void swilu_quantize_q8_K_launch(const float* gate, const float* up, float* gate_out, int64_t n,
                                uint8_t* blocks, void* stream) {
    static sycl::queue default_queue{sycl::default_selector_v, sycl::property::queue::in_order()};
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    const long long nb = n / QK_K;
    constexpr int SG_SIZE = strata::sycl_arch::SUB_GROUP_SIZE;
    constexpr int THREADS = 64;
    const unsigned blocks_wg = (unsigned)nb;

    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> p(sycl::range<1>(256), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(blocks_wg * THREADS), sycl::range<1>(THREADS)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const long long b = item.get_group(0);
                const int lid = item.get_local_id(0);
                float* p_ptr = p.get_multi_ptr<sycl::access::decorated::no>().get();
                if (b < nb) {
                    const long long i0 = b * QK_K;
                    for (int j = lid; j < QK_K; j += THREADS) {
                        const float v = swilu_apply<KIND>(gate[i0 + j], up[i0 + j]);
                        gate_out[i0 + j] = v;
                        p_ptr[j] = v;
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);
                if (b < nb && lid == 0) {
                    quantize_q8_K_block(p_ptr, blocks + b * Q8K_BYTES);
                }
            });
    });

    if (!stream) default_queue.wait();
}

}  // namespace

void quantize_q8_0(const float* x, uint8_t* blocks, int64_t n, void* stream) {
    static sycl::queue default_queue{sycl::default_selector_v, sycl::property::queue::in_order()};
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    if (n <= 0) return;
    if (n % QK8_0 != 0) {
        std::fprintf(stderr, "quantize_q8_0: n %lld is not a multiple of %d\n", (long long)n, QK8_0);
        std::exit(1);
    }
    const long long nb = n / QK8_0;
    const int threads = 128;
    const unsigned grid = (unsigned)((nb + threads - 1) / threads);
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>((grid) * (threads)), sycl::range<1>(threads)),
                    [=](sycl::nd_item<1> item) { quantize_q8_0_kernel(item, x, blocks, nb); });
    if (!stream) default_queue.wait();
}

void quantize_q8_0_scaled(const float* x, uint8_t* blocks, float* scales, int64_t n, void* stream) {
    static sycl::queue default_queue{sycl::default_selector_v, sycl::property::queue::in_order()};
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    if (n <= 0) return;
    if (n % QK8_0 != 0) {
        std::fprintf(stderr, "quantize_q8_0_scaled: n %lld is not a multiple of %d\n", (long long)n, QK8_0);
        std::exit(1);
    }
    if (scales == nullptr) {
        std::fprintf(stderr, "quantize_q8_0_scaled: scales is null\n");
        std::exit(1);
    }
    const long long nb = n / QK8_0;
    const int threads = 128;
    const unsigned grid = (unsigned)((nb + threads - 1) / threads);
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>((grid) * (threads)), sycl::range<1>(threads)),
                    [=](sycl::nd_item<1> item) { quantize_q8_0_scaled_kernel(item, x, blocks, scales, nb); });
    if (!stream) default_queue.wait();
}

void dequant_q8_0(const uint8_t* blocks, float* x, int64_t n, void* stream) {
    static sycl::queue default_queue{sycl::default_selector_v, sycl::property::queue::in_order()};
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    if (n <= 0) return;
    const long long nb = n / QK8_0;
    const int threads = 128;
    const unsigned grid = (unsigned)((nb + threads - 1) / threads);
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>((grid) * (threads)), sycl::range<1>(threads)),
                    [=](sycl::nd_item<1> item) { dequant_q8_0_kernel(item, blocks, x, nb); });
    if (!stream) default_queue.wait();
}

void quantize_q8_K(const float* x, uint8_t* blocks, int64_t n, void* stream) {
    static sycl::queue default_queue{sycl::default_selector_v, sycl::property::queue::in_order()};
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    if (n <= 0) return;
    if (n % QK_K != 0) {
        std::fprintf(stderr, "quantize_q8_K: n %lld is not a multiple of %d\n", (long long)n, QK_K);
        std::exit(1);
    }
    const long long nb = n / QK_K;
    const int threads = 64;
    const unsigned grid = (unsigned)((nb + threads - 1) / threads);
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>((grid) * (threads)), sycl::range<1>(threads)),
                    [=](sycl::nd_item<1> item) { quantize_q8_K_kernel(item, x, blocks, nb); });
    if (!stream) default_queue.wait();
}

void dequant_q8_K(const uint8_t* blocks, float* x, int64_t n, void* stream) {
    static sycl::queue default_queue{sycl::default_selector_v, sycl::property::queue::in_order()};
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    if (n <= 0) return;
    const long long nb = n / QK_K;
    const int threads = 64;
    const unsigned grid = (unsigned)((nb + threads - 1) / threads);
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>((grid) * (threads)), sycl::range<1>(threads)),
                    [=](sycl::nd_item<1> item) { dequant_q8_K_kernel(item, blocks, x, nb); });
    if (!stream) default_queue.wait();
}

void swilu_quantize_q8_0(const float* gate, const float* up, float* gate_out, int64_t n, int swilu_kind,
                         uint8_t* blocks, void* stream) {
    if (n <= 0) return;
    if (n % QK8_0 != 0) {
        std::fprintf(stderr, "swilu_quantize_q8_0: n %lld is not a multiple of %d\n", (long long)n, QK8_0);
        std::exit(1);
    }
    switch (swilu_kind) {
        case 0: swilu_quantize_q8_0_launch<0, false>(gate, up, gate_out, n, blocks, nullptr, stream); break;
        case 1: swilu_quantize_q8_0_launch<1, false>(gate, up, gate_out, n, blocks, nullptr, stream); break;
        default: swilu_quantize_q8_0_launch<2, false>(gate, up, gate_out, n, blocks, nullptr, stream); break;
    }
}

void swilu_quantize_q8_0_scaled(const float* gate, const float* up, float* gate_out, int64_t n, int swilu_kind,
                                uint8_t* blocks, float* scales, void* stream) {
    if (n <= 0) return;
    if (n % QK8_0 != 0) {
        std::fprintf(stderr, "swilu_quantize_q8_0_scaled: n %lld is not a multiple of %d\n", (long long)n, QK8_0);
        std::exit(1);
    }
    if (scales == nullptr) {
        std::fprintf(stderr, "swilu_quantize_q8_0_scaled: scales is null\n");
        std::exit(1);
    }
    switch (swilu_kind) {
        case 0: swilu_quantize_q8_0_launch<0, true>(gate, up, gate_out, n, blocks, scales, stream); break;
        case 1: swilu_quantize_q8_0_launch<1, true>(gate, up, gate_out, n, blocks, scales, stream); break;
        default: swilu_quantize_q8_0_launch<2, true>(gate, up, gate_out, n, blocks, scales, stream); break;
    }
}

void swilu_quantize_q8_K(const float* gate, const float* up, float* gate_out, int64_t n, int swilu_kind,
                         uint8_t* blocks, void* stream) {
    if (n <= 0) return;
    if (n % QK_K != 0) {
        std::fprintf(stderr, "swilu_quantize_q8_K: n %lld is not a multiple of %d\n", (long long)n, QK_K);
        std::exit(1);
    }
    switch (swilu_kind) {
        case 0: swilu_quantize_q8_K_launch<0>(gate, up, gate_out, n, blocks, stream); break;
        case 1: swilu_quantize_q8_K_launch<1>(gate, up, gate_out, n, blocks, stream); break;
        default: swilu_quantize_q8_K_launch<2>(gate, up, gate_out, n, blocks, stream); break;
    }
}

void quantize_act_images(const float* x, int64_t n, uint8_t* q8_0_blocks, uint8_t* q8_K_blocks,
                         uint16_t* bf16, void* stream) {
    if (n <= 0) return;
    if (q8_0_blocks == nullptr && q8_K_blocks == nullptr && bf16 == nullptr) return;
    if (n % QK8_0 != 0 || (q8_K_blocks != nullptr && n % QK_K != 0)) {
        std::fprintf(stderr, "quantize_act_images: n %lld must be a multiple of %d (and %d for Q8_K)\n",
                     (long long)n, QK8_0, QK_K);
        std::exit(1);
    }

    static sycl::queue default_queue{sycl::default_selector_v, sycl::property::queue::in_order()};
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    constexpr int SG_SIZE = strata::sycl_arch::SUB_GROUP_SIZE;
    constexpr int THREADS = 64;
    const unsigned chunks = (unsigned)((n + 255) / 256);

    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> s_x(sycl::range<1>(256), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(chunks * THREADS), sycl::range<1>(THREADS)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const long long gid = item.get_group(0);
                const int lid = item.get_local_id(0);
                const long long elem_idx = gid * 256 + lid * 4;

                // 128-bit vector load of x
                sycl::vec<float, 4> v(0.0f);
                if (elem_idx + 3 < n) {
                    v = *reinterpret_cast<const sycl::vec<float, 4>*>(x + elem_idx);
                } else if (elem_idx < n) {
                    for (int j = 0; j < 4 && elem_idx + j < n; ++j) {
                        v[j] = x[elem_idx + j];
                    }
                }

                // bf16 image
                if (bf16 != nullptr) {
                    if (elem_idx + 3 < n) {
                        sycl::vec<uint16_t, 4> out_b16;
                        out_b16[0] = bf16_from_f32(v[0]);
                        out_b16[1] = bf16_from_f32(v[1]);
                        out_b16[2] = bf16_from_f32(v[2]);
                        out_b16[3] = bf16_from_f32(v[3]);
                        *reinterpret_cast<sycl::vec<uint16_t, 4>*>(bf16 + elem_idx) = out_b16;
                    } else if (elem_idx < n) {
                        for (int j = 0; j < 4 && elem_idx + j < n; ++j) {
                            bf16[elem_idx + j] = bf16_from_f32(v[j]);
                        }
                    }
                }

                // q8_0 and q8_K images stage through SLM
                if (q8_0_blocks != nullptr || q8_K_blocks != nullptr) {
                    float* s_x_ptr = s_x.get_multi_ptr<sycl::access::decorated::no>().get();
                    s_x_ptr[lid * 4 + 0] = v[0];
                    s_x_ptr[lid * 4 + 1] = v[1];
                    s_x_ptr[lid * 4 + 2] = v[2];
                    s_x_ptr[lid * 4 + 3] = v[3];
                    item.barrier(sycl::access::fence_space::local_space);

                    if (q8_0_blocks != nullptr && lid < 8) {
                        const long long b = gid * 8 + lid;
                        if (b < n / QK8_0) {
                            quantize_q8_0_block(s_x_ptr + lid * 32, q8_0_blocks + b * 34);
                        }
                    }

                    if (q8_K_blocks != nullptr && lid == 0) {
                        const long long k = gid;
                        if (k < n / QK_K) {
                            quantize_q8_K_block(s_x_ptr, q8_K_blocks + k * Q8K_BYTES);
                        }
                    }
                }
            });
    });

    if (!stream) default_queue.wait();
}

}  // namespace strata::kernels
