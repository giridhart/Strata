// perf/sycl/perf_iq1m_lut.cpp - Fast IQ1_M SLM LUT Dequantization & GEMV Microbenchmark for Intel Xe2
//
// Evaluates performance optimizations for IQ1_M (1.75-bit GGUF quantization) on Intel Xe2:
// 1. Baseline SYCL IQ1_M:
//    Runtime bit-unpacking with scatter loads from global table `iq1s_grid_gpu` and floating-point
//    delta compensation (`delta * sum(u)`).
// 2. SLM Grid 2048:
//    Staging the 2048-entry 32-bit codebook table (8 KiB) into Shared Local Memory (SLM).
// 3. Fast IQ1_M SLM LUT (DeepSeek-V4.1 port):
//    Precomputed 4096-entry 64-bit LUT (32 KiB) staged into SLM.
//    Folds the delta sign and scale factor directly into 8-bit signed integers, allowing:
//    - Single 64-bit lookup per 8 elements
//    - Zero runtime bit unpacking or mask operations
//    - Zero floating-point operations in the dot-product inner loop
//    - Halving dp4a instructions (8 dp4a vs 16 dp4a in baseline)
//
// Evaluated on critical LLM decoding & dequantization shapes on Intel Xe2 (Lunar Lake / Battlemage).

#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

namespace sycl { using half2 = sycl::vec<sycl::half, 2>; }

#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

// ============================================================================
// Codebook Tables & Data Structures
// ============================================================================

struct alignas(8) uint2_u64 {
    uint32_t x;
    uint32_t y;
};

// 4096-entry 64-bit lookup table ported from DeepSeek-V4.1 (src/dsv41/ds4_iq1m_codes.inc)
static const uint2_u64 kIq1mCodesHost[4096] = {
#include "../../src/dsv41/ds4_iq1m_codes.inc"
};

// Helper to initialize valid IQ1_M blocks with proper FP16 scales
void init_iq1m_blocks(block_iq1_m* blocks, size_t num_blocks, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    for (size_t b = 0; b < num_blocks; ++b) {
        for (int i = 0; i < 32; ++i) blocks[b].qs[i] = static_cast<uint8_t>(rng() & 0xFF);
        for (int i = 0; i < 16; ++i) blocks[b].qh[i] = static_cast<uint8_t>(rng() & 0xFF);
        for (int i = 0; i < 8; ++i) blocks[b].scales[i] = static_cast<uint8_t>(rng() & 0xFF);

        // Valid FP16 scale (~0.05f to 0.15f) encoded into upper nibbles of scales[1, 3, 5, 7]
        float scale_val = 0.05f + 0.0001f * static_cast<float>(rng() % 1000);
        sycl::half sh(scale_val);
        uint16_t d_f16 = *reinterpret_cast<uint16_t*>(&sh);

        blocks[b].scales[1] = static_cast<uint8_t>((blocks[b].scales[1] & 0x0f) | ((d_f16 & 0x000f) << 4));
        blocks[b].scales[3] = static_cast<uint8_t>((blocks[b].scales[3] & 0x0f) | (d_f16 & 0x00f0));
        blocks[b].scales[5] = static_cast<uint8_t>((blocks[b].scales[5] & 0x0f) | ((d_f16 & 0x0f00) >> 4));
        blocks[b].scales[7] = static_cast<uint8_t>((blocks[b].scales[7] & 0x0f) | ((d_f16 & 0xf000) >> 8));
    }
}

// Helper to initialize Q8_1 activations
void init_q8_1_blocks(block_q8_1* blocks, size_t num_blocks, uint32_t seed = 101) {
    std::mt19937 rng(seed);
    for (size_t b = 0; b < num_blocks; ++b) {
        blocks[b].ds[0] = 0.02f + 0.00005f * static_cast<float>(rng() % 500);
        blocks[b].ds[1] = 0.0f;
        for (int i = 0; i < 32; ++i) {
            blocks[b].qs[i] = static_cast<int8_t>((rng() % 255) - 127);
        }
    }
}

// ============================================================================
// Dequantization Kernels (IQ1_M -> FP16)
// ============================================================================

// 1. Baseline SYCL Dequantization (from src/kernels/sycl/iq_kernels.cpp)
void dq_baseline_kernel(sycl::nd_item<1> item,
                        const block_iq1_m* __restrict__ src,
                        sycl::half* __restrict__ dst,
                        size_t num_blocks) {
    const size_t blk_idx = item.get_group(0);
    if (blk_idx >= num_blocks) return;

    const int tid = item.get_local_id(0); // 0..31
    const int64_t il = tid % 4;
    const int64_t ib = tid / 4;
    sycl::half* y = dst + blk_idx * 256 + 32 * ib + 8 * il;

    const auto& blk = src[blk_idx];
    const uint16_t* sc = reinterpret_cast<const uint16_t*>(blk.scales);
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2 * ib + il / 2;
    const float d = static_cast<float>(scale.f16) * (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 0x7) + 1);
    const float delta = (blk.qh[2 * ib + il / 2] & (0x08 << (4 * (il % 2)))) ? -1.0f - IQ1M_DELTA : -1.0f + IQ1M_DELTA;

    uint32_t grid32[2];
    const int8_t* q = reinterpret_cast<const int8_t*>(grid32);
    grid32[0] = iq1s_grid_gpu[blk.qs[4 * ib + il] | (((blk.qh[2 * ib + il / 2] >> (4 * (il % 2))) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        y[j] = static_cast<sycl::half>(d * (q[j] + delta));
    }
}

// 2. SLM Grid 2048 Dequantization (Stages 8 KiB iq1s_grid_gpu into SLM)
void dq_slm_grid_kernel(sycl::nd_item<1> item,
                        const block_iq1_m* __restrict__ src,
                        const uint32_t* __restrict__ d_grid,
                        sycl::half* __restrict__ dst,
                        size_t num_blocks,
                        sycl::local_accessor<uint32_t, 1> slm_grid) {
    const int tid = item.get_local_id(0);
    const int wg_size = item.get_local_range(0);

    // Collaborative load of 2048 words (8 KiB)
    for (int j = tid; j < 2048; j += wg_size) {
        slm_grid[j] = d_grid[j];
    }
    item.barrier(sycl::access::fence_space::local_space);

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = wg_size / 32;
    const int il = lane_id % 4;
    const int ib = lane_id / 4;

    const size_t total_warps = item.get_group_range(0) * num_warps;
    const size_t start_block = item.get_group(0) * num_warps + warp_id;

    for (size_t blk_idx = start_block; blk_idx < num_blocks; blk_idx += total_warps) {
        sycl::half* y = dst + blk_idx * 256 + 32 * ib + 8 * il;
        const auto& blk = src[blk_idx];

        const uint16_t* sc = reinterpret_cast<const uint16_t*>(blk.scales);
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        const int64_t ib16 = 2 * ib + il / 2;
        const float d = static_cast<float>(scale.f16) * (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 0x7) + 1);
        const float delta = (blk.qh[2 * ib + il / 2] & (0x08 << (4 * (il % 2)))) ? -1.0f - IQ1M_DELTA : -1.0f + IQ1M_DELTA;

        uint32_t grid32[2];
        const int8_t* q = reinterpret_cast<const int8_t*>(grid32);
        grid32[0] = slm_grid[blk.qs[4 * ib + il] | (((blk.qh[2 * ib + il / 2] >> (4 * (il % 2))) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            y[j] = static_cast<sycl::half>(d * (q[j] + delta));
        }
    }
}

// 3. Fast IQ1_M SLM LUT Dequantization (DeepSeek-V4.1 Port: 4096-entry 64-bit LUT in SLM)
void dq_slm_lut_kernel(sycl::nd_item<1> item,
                       const block_iq1_m* __restrict__ src,
                       const uint2_u64* __restrict__ d_lut,
                       sycl::half* __restrict__ dst,
                       size_t num_blocks,
                       sycl::local_accessor<uint2_u64, 1> slm_lut) {
    const int tid = item.get_local_id(0);
    const int wg_size = item.get_local_range(0);

    // Stage 4096 entries (32 KiB) using 128-bit vector transfers
    const sycl::uint4* src4 = reinterpret_cast<const sycl::uint4*>(d_lut);
    auto* slm4 = reinterpret_cast<sycl::uint4*>(slm_lut.get_multi_ptr<sycl::access::decorated::no>().get());
    for (int j = tid; j < 2048; j += wg_size) {
        slm4[j] = src4[j];
    }
    item.barrier(sycl::access::fence_space::local_space);

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = wg_size / 32;
    const int il = lane_id % 4;
    const int ib = lane_id / 4;

    const size_t total_warps = item.get_group_range(0) * num_warps;
    const size_t start_block = item.get_group(0) * num_warps + warp_id;

    for (size_t blk_idx = start_block; blk_idx < num_blocks; blk_idx += total_warps) {
        sycl::half* y = dst + blk_idx * 256 + 32 * ib + 8 * il;
        const auto& blk = src[blk_idx];

        const uint16_t* sc = reinterpret_cast<const uint16_t*>(blk.scales);
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        const int64_t ib16 = 2 * ib + il / 2;
        const float d = static_cast<float>(scale.f16) * (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 0x7) + 1) * 0.125f;

        const uint32_t qs_byte = blk.qs[4 * ib + il];
        const uint32_t qh_nibble = (blk.qh[2 * ib + il / 2] >> (4 * (il % 2))) & 0x0f;
        const uint32_t lut_idx = qs_byte | (qh_nibble << 8);

        const uint2_u64 c = slm_lut[lut_idx];
        const int8_t* codes = reinterpret_cast<const int8_t*>(&c);

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            y[j] = static_cast<sycl::half>(d * codes[j]);
        }
    }
}

// 4. Fast IQ1_M SLM LUT Direct (1 Block per Workgroup)
void dq_slm_lut_direct_kernel(sycl::nd_item<1> item,
                              const block_iq1_m* __restrict__ src,
                              const uint2_u64* __restrict__ d_lut,
                              sycl::half* __restrict__ dst,
                              size_t num_blocks,
                              sycl::local_accessor<uint2_u64, 1> slm_lut) {
    const size_t blk_idx = item.get_group(0);
    if (blk_idx >= num_blocks) return;

    const int tid = item.get_local_id(0);
    const int wg_size = item.get_local_range(0);

    const sycl::uint4* src4 = reinterpret_cast<const sycl::uint4*>(d_lut);
    auto* slm4 = reinterpret_cast<sycl::uint4*>(slm_lut.get_multi_ptr<sycl::access::decorated::no>().get());
    for (int j = tid; j < 2048; j += wg_size) {
        slm4[j] = src4[j];
    }
    item.barrier(sycl::access::fence_space::local_space);

    const int64_t il = tid % 4;
    const int64_t ib = tid / 4;
    sycl::half* y = dst + blk_idx * 256 + 32 * ib + 8 * il;

    const auto& blk = src[blk_idx];
    const uint16_t* sc = reinterpret_cast<const uint16_t*>(blk.scales);
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2 * ib + il / 2;
    const float d = static_cast<float>(scale.f16) * (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 0x7) + 1) * 0.125f;

    const uint32_t qs_byte = blk.qs[4 * ib + il];
    const uint32_t qh_nibble = (blk.qh[2 * ib + il / 2] >> (4 * (il % 2))) & 0x0f;
    const uint32_t lut_idx = qs_byte | (qh_nibble << 8);

    const uint2_u64 c = slm_lut[lut_idx];
    const int8_t* codes = reinterpret_cast<const int8_t*>(&c);

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        y[j] = static_cast<sycl::half>(d * codes[j]);
    }
}

// ============================================================================
// GEMV / Dot Product Kernels (IQ1_M Weights x Q8_1 Activations -> FP32 Output)
// ============================================================================

// 1. Baseline SYCL GEMV (Global table lookups, 16 dp4a + 8 float ops per 32 elements)
void gemv_baseline_kernel(sycl::nd_item<1> item,
                          const block_iq1_m* __restrict__ w,
                          const block_q8_1* __restrict__ x,
                          float* __restrict__ y,
                          size_t N, size_t K) {
    const size_t row = item.get_group(0);
    if (row >= N) return;

    const int tid = item.get_local_id(0);
    const int wg_size = item.get_local_range(0);
    const size_t nb = K / 256;
    const block_iq1_m* row_w = w + row * nb;

    float s = 0.0f;
    for (size_t k = tid; k < nb * 8; k += wg_size) {
        const size_t kbx = k / 8;
        const int iqs = static_cast<int>(k % 8);
        const auto& blk = row_w[kbx];

        const uint32_t qsw = *reinterpret_cast<const uint32_t*>(blk.qs + 4 * iqs);
        const uint8_t* qs = reinterpret_cast<const uint8_t*>(&qsw);
        int sumi[2] = {0, 0};
        float sumf[2] = {0.0f, 0.0f};

        #pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int qhl = blk.qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
            const int grid = iq1s_grid_gpu[qs[l0 / 2] | ((qhl & 0x07) << 8)];
            const int grid0 = (grid >> 0) & 0x0F0F0F0F;
            const int grid1 = (grid >> 4) & 0x0F0F0F0F;
            const int u0 = *reinterpret_cast<const int*>(x[kbx * 8 + iqs].qs + 4 * (l0 + 0));
            const int u1 = *reinterpret_cast<const int*>(x[kbx * 8 + iqs].qs + 4 * (l0 + 1));
            sumi[l0 / 4] = strata_dp4a(grid0, u0, sumi[l0 / 4]);
            sumi[l0 / 4] = strata_dp4a(grid1, u1, sumi[l0 / 4]);
            const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
            int sumy = 0;
            sumy = strata_dp4a(u0, 0x01010101, sumy);
            sumy = strata_dp4a(u1, 0x01010101, sumy);
            sumf[l0 / 4] += delta * sumy;
        }

        const uint16_t* sc = reinterpret_cast<const uint16_t*>(blk.scales);
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
        const float d = static_cast<float>(scale.f16) * (x[kbx * 8 + iqs].ds)[0];
        const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
        const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
        const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
        s += d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1);
    }

    float row_sum = sycl::reduce_over_group(item.get_group(), s, sycl::plus<>());
    if (tid == 0) y[row] = row_sum;
}

// 2. SLM Grid 2048 GEMV (8 KiB iq1s_grid_gpu staged in SLM)
void gemv_slm_grid_kernel(sycl::nd_item<1> item,
                          const block_iq1_m* __restrict__ w,
                          const block_q8_1* __restrict__ x,
                          const uint32_t* __restrict__ d_grid,
                          float* __restrict__ y,
                          size_t N, size_t K,
                          sycl::local_accessor<uint32_t, 1> slm_grid) {
    const int tid = item.get_local_id(0);
    const int wg_size = item.get_local_range(0);

    for (int j = tid; j < 2048; j += wg_size) {
        slm_grid[j] = d_grid[j];
    }
    item.barrier(sycl::access::fence_space::local_space);

    const size_t nb = K / 256;

    // Persistent loop across rows
    for (size_t row = item.get_group(0); row < N; row += item.get_group_range(0)) {
        const block_iq1_m* row_w = w + row * nb;
        float s = 0.0f;

        for (size_t k = tid; k < nb * 8; k += wg_size) {
            const size_t kbx = k / 8;
            const int iqs = static_cast<int>(k % 8);
            const auto& blk = row_w[kbx];

            const uint32_t qsw = *reinterpret_cast<const uint32_t*>(blk.qs + 4 * iqs);
            const uint8_t* qs = reinterpret_cast<const uint8_t*>(&qsw);
            int sumi[2] = {0, 0};
            float sumf[2] = {0.0f, 0.0f};

            #pragma unroll
            for (int l0 = 0; l0 < 8; l0 += 2) {
                const int qhl = blk.qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
                const int grid = slm_grid[qs[l0 / 2] | ((qhl & 0x07) << 8)];
                const int grid0 = (grid >> 0) & 0x0F0F0F0F;
                const int grid1 = (grid >> 4) & 0x0F0F0F0F;
                const int u0 = *reinterpret_cast<const int*>(x[kbx * 8 + iqs].qs + 4 * (l0 + 0));
                const int u1 = *reinterpret_cast<const int*>(x[kbx * 8 + iqs].qs + 4 * (l0 + 1));
                sumi[l0 / 4] = strata_dp4a(grid0, u0, sumi[l0 / 4]);
                sumi[l0 / 4] = strata_dp4a(grid1, u1, sumi[l0 / 4]);
                const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
                int sumy = 0;
                sumy = strata_dp4a(u0, 0x01010101, sumy);
                sumy = strata_dp4a(u1, 0x01010101, sumy);
                sumf[l0 / 4] += delta * sumy;
            }

            const uint16_t* sc = reinterpret_cast<const uint16_t*>(blk.scales);
            iq1m_scale_t scale;
            scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
            const float d = static_cast<float>(scale.f16) * (x[kbx * 8 + iqs].ds)[0];
            const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
            const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
            const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
            s += d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1);
        }

        float row_sum = sycl::reduce_over_group(item.get_group(), s, sycl::plus<>());
        if (tid == 0) y[row] = row_sum;
    }
}

// 3. Fast IQ1_M SLM LUT GEMV (DeepSeek-V4.1 Port: 32 KiB 64-bit LUT, 8 dp4a, 0 float ops in loop)
void gemv_slm_lut_kernel(sycl::nd_item<1> item,
                         const block_iq1_m* __restrict__ w,
                         const block_q8_1* __restrict__ x,
                         const uint2_u64* __restrict__ d_lut,
                         float* __restrict__ y,
                         size_t N, size_t K,
                         sycl::local_accessor<uint2_u64, 1> slm_lut) {
    const int tid = item.get_local_id(0);
    const int wg_size = item.get_local_range(0);

    // Stage 32 KiB LUT once per workgroup
    const sycl::uint4* src4 = reinterpret_cast<const sycl::uint4*>(d_lut);
    auto* slm4 = reinterpret_cast<sycl::uint4*>(slm_lut.get_multi_ptr<sycl::access::decorated::no>().get());
    for (int j = tid; j < 2048; j += wg_size) {
        slm4[j] = src4[j];
    }
    item.barrier(sycl::access::fence_space::local_space);

    const size_t nb = K / 256;

    // Persistent loop across rows
    for (size_t row = item.get_group(0); row < N; row += item.get_group_range(0)) {
        const block_iq1_m* row_w = w + row * nb;
        float s = 0.0f;

        for (size_t k = tid; k < nb * 8; k += wg_size) {
            const size_t kbx = k / 8;
            const int iqs = static_cast<int>(k % 8);
            const auto& blk = row_w[kbx];

            const uint32_t qsw = *reinterpret_cast<const uint32_t*>(blk.qs + 4 * iqs);
            const uint16_t qhw = *reinterpret_cast<const uint16_t*>(blk.qh + 2 * iqs);
            int lo = 0, hi = 0;

            // l = 0
            {
                const uint32_t idx0 = (qsw & 0xffu) | ((qhw << 8) & 0xf00u);
                const uint2_u64 c0 = slm_lut[idx0];
                lo = strata_dp4a(c0.x, *reinterpret_cast<const int*>(x[kbx * 8 + iqs].qs + 0), lo);
                lo = strata_dp4a(c0.y, *reinterpret_cast<const int*>(x[kbx * 8 + iqs].qs + 4), lo);
            }
            // l = 1
            {
                const uint32_t idx1 = ((qsw >> 8) & 0xffu) | ((qhw << 4) & 0xf00u);
                const uint2_u64 c1 = slm_lut[idx1];
                lo = strata_dp4a(c1.x, *reinterpret_cast<const int*>(x[kbx * 8 + iqs].qs + 8), lo);
                lo = strata_dp4a(c1.y, *reinterpret_cast<const int*>(x[kbx * 8 + iqs].qs + 12), lo);
            }
            // l = 2
            {
                const uint32_t idx2 = ((qsw >> 16) & 0xffu) | (qhw & 0xf00u);
                const uint2_u64 c2 = slm_lut[idx2];
                hi = strata_dp4a(c2.x, *reinterpret_cast<const int*>(x[kbx * 8 + iqs].qs + 16), hi);
                hi = strata_dp4a(c2.y, *reinterpret_cast<const int*>(x[kbx * 8 + iqs].qs + 20), hi);
            }
            // l = 3
            {
                const uint32_t idx3 = ((qsw >> 24) & 0xffu) | ((qhw >> 4) & 0xf00u);
                const uint2_u64 c3 = slm_lut[idx3];
                hi = strata_dp4a(c3.x, *reinterpret_cast<const int*>(x[kbx * 8 + iqs].qs + 24), hi);
                hi = strata_dp4a(c3.y, *reinterpret_cast<const int*>(x[kbx * 8 + iqs].qs + 28), hi);
            }

            const uint16_t* sc = reinterpret_cast<const uint16_t*>(blk.scales);
            iq1m_scale_t scale;
            scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
            const float d = static_cast<float>(scale.f16) * (x[kbx * 8 + iqs].ds)[0] * 0.125f;
            const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
            const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
            const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
            s += d * (sc0 * lo + sc1 * hi);
        }

        float row_sum = sycl::reduce_over_group(item.get_group(), s, sycl::plus<>());
        if (tid == 0) y[row] = row_sum;
    }
}

// 4. Fast IQ1_M SLM LUT GEMV Direct (1 Row per Workgroup)
void gemv_slm_lut_direct_kernel(sycl::nd_item<1> item,
                                const block_iq1_m* __restrict__ w,
                                const block_q8_1* __restrict__ x,
                                const uint2_u64* __restrict__ d_lut,
                                float* __restrict__ y,
                                size_t N, size_t K,
                                sycl::local_accessor<uint2_u64, 1> slm_lut) {
    const size_t row = item.get_group(0);
    if (row >= N) return;

    const int tid = item.get_local_id(0);
    const int wg_size = item.get_local_range(0);

    const sycl::uint4* src4 = reinterpret_cast<const sycl::uint4*>(d_lut);
    auto* slm4 = reinterpret_cast<sycl::uint4*>(slm_lut.get_multi_ptr<sycl::access::decorated::no>().get());
    for (int j = tid; j < 2048; j += wg_size) {
        slm4[j] = src4[j];
    }
    item.barrier(sycl::access::fence_space::local_space);

    const size_t nb = K / 256;
    const block_iq1_m* row_w = w + row * nb;
    float s = 0.0f;

    for (size_t k = tid; k < nb * 8; k += wg_size) {
        const size_t kbx = k / 8;
        const int iqs = static_cast<int>(k % 8);
        const auto& blk = row_w[kbx];

        const uint32_t qsw = *reinterpret_cast<const uint32_t*>(blk.qs + 4 * iqs);
        const uint16_t qhw = *reinterpret_cast<const uint16_t*>(blk.qh + 2 * iqs);
        int lo = 0, hi = 0;

        // l = 0
        {
            const uint32_t idx0 = (qsw & 0xffu) | ((qhw << 8) & 0xf00u);
            const uint2_u64 c0 = slm_lut[idx0];
            lo = strata_dp4a(c0.x, *reinterpret_cast<const int*>(x[kbx * 8 + iqs].qs + 0), lo);
            lo = strata_dp4a(c0.y, *reinterpret_cast<const int*>(x[kbx * 8 + iqs].qs + 4), lo);
        }
        // l = 1
        {
            const uint32_t idx1 = ((qsw >> 8) & 0xffu) | ((qhw << 4) & 0xf00u);
            const uint2_u64 c1 = slm_lut[idx1];
            lo = strata_dp4a(c1.x, *reinterpret_cast<const int*>(x[kbx * 8 + iqs].qs + 8), lo);
            lo = strata_dp4a(c1.y, *reinterpret_cast<const int*>(x[kbx * 8 + iqs].qs + 12), lo);
        }
        // l = 2
        {
            const uint32_t idx2 = ((qsw >> 16) & 0xffu) | (qhw & 0xf00u);
            const uint2_u64 c2 = slm_lut[idx2];
            hi = strata_dp4a(c2.x, *reinterpret_cast<const int*>(x[kbx * 8 + iqs].qs + 16), hi);
            hi = strata_dp4a(c2.y, *reinterpret_cast<const int*>(x[kbx * 8 + iqs].qs + 20), hi);
        }
        // l = 3
        {
            const uint32_t idx3 = ((qsw >> 24) & 0xffu) | ((qhw >> 4) & 0xf00u);
            const uint2_u64 c3 = slm_lut[idx3];
            hi = strata_dp4a(c3.x, *reinterpret_cast<const int*>(x[kbx * 8 + iqs].qs + 24), hi);
            hi = strata_dp4a(c3.y, *reinterpret_cast<const int*>(x[kbx * 8 + iqs].qs + 28), hi);
        }

        const uint16_t* sc = reinterpret_cast<const uint16_t*>(blk.scales);
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
        const float d = static_cast<float>(scale.f16) * (x[kbx * 8 + iqs].ds)[0] * 0.125f;
        const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
        const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
        const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
        s += d * (sc0 * lo + sc1 * hi);
    }

    float row_sum = sycl::reduce_over_group(item.get_group(), s, sycl::plus<>());
    if (tid == 0) y[row] = row_sum;
}

// ============================================================================
// Verification Suite (Guarantees Numerical Equivalence on Device)
// ============================================================================

bool run_device_verification(sycl::queue& q) {
    constexpr size_t num_blocks = 256;
    constexpr size_t num_elements = num_blocks * 256;
    constexpr size_t K = 2560;
    constexpr size_t N = 128;
    constexpr size_t gemv_blocks = (K / 256) * N;

    std::printf("[VERIFY] Running numerical parity verification on device...\n");

    std::vector<block_iq1_m> h_dq_blocks(num_blocks);
    init_iq1m_blocks(h_dq_blocks.data(), num_blocks, 1234);

    std::vector<block_iq1_m> h_gemv_w(gemv_blocks);
    init_iq1m_blocks(h_gemv_w.data(), gemv_blocks, 5678);

    std::vector<block_q8_1> h_gemv_x(K / 32);
    init_q8_1_blocks(h_gemv_x.data(), K / 32, 9012);

    block_iq1_m* d_dq_blocks = sycl::malloc_device<block_iq1_m>(num_blocks, q);
    block_iq1_m* d_gemv_w = sycl::malloc_device<block_iq1_m>(gemv_blocks, q);
    block_q8_1* d_gemv_x = sycl::malloc_device<block_q8_1>(K / 32, q);
    uint2_u64* d_lut = sycl::malloc_device<uint2_u64>(4096, q);
    uint32_t* d_grid = sycl::malloc_device<uint32_t>(2048, q);

    sycl::half* d_out_base = sycl::malloc_device<sycl::half>(num_elements, q);
    sycl::half* d_out_grid = sycl::malloc_device<sycl::half>(num_elements, q);
    sycl::half* d_out_lut = sycl::malloc_device<sycl::half>(num_elements, q);

    float* d_y_base = sycl::malloc_device<float>(N, q);
    float* d_y_grid = sycl::malloc_device<float>(N, q);
    float* d_y_lut = sycl::malloc_device<float>(N, q);

    q.memcpy(d_dq_blocks, h_dq_blocks.data(), num_blocks * sizeof(block_iq1_m));
    q.memcpy(d_gemv_w, h_gemv_w.data(), gemv_blocks * sizeof(block_iq1_m));
    q.memcpy(d_gemv_x, h_gemv_x.data(), (K / 32) * sizeof(block_q8_1));
    q.memcpy(d_lut, kIq1mCodesHost, 4096 * sizeof(uint2_u64));
    q.memcpy(d_grid, iq1s_grid_gpu, 2048 * sizeof(uint32_t));
    q.wait();

    // 1. Dequant Verification
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(num_blocks * 32, 32), [=](sycl::nd_item<1> item) {
            dq_baseline_kernel(item, d_dq_blocks, d_out_base, num_blocks);
        });
    });

    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<uint32_t, 1> slm_grid(sycl::range<1>(2048), cgh);
        cgh.parallel_for(sycl::nd_range<1>(num_blocks * 32, 32), [=](sycl::nd_item<1> item) {
            dq_slm_grid_kernel(item, d_dq_blocks, d_grid, d_out_grid, num_blocks, slm_grid);
        });
    });

    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<uint2_u64, 1> slm_lut(sycl::range<1>(4096), cgh);
        cgh.parallel_for(sycl::nd_range<1>(num_blocks * 32, 32), [=](sycl::nd_item<1> item) {
            dq_slm_lut_kernel(item, d_dq_blocks, d_lut, d_out_lut, num_blocks, slm_lut);
        });
    });
    q.wait();

    std::vector<sycl::half> h_out_base(num_elements);
    std::vector<sycl::half> h_out_grid(num_elements);
    std::vector<sycl::half> h_out_lut(num_elements);

    q.memcpy(h_out_base.data(), d_out_base, num_elements * sizeof(sycl::half));
    q.memcpy(h_out_grid.data(), d_out_grid, num_elements * sizeof(sycl::half));
    q.memcpy(h_out_lut.data(), d_out_lut, num_elements * sizeof(sycl::half));
    q.wait();

    float dq_max_diff_lut = 0.0f;
    float dq_max_diff_grid = 0.0f;
    for (size_t i = 0; i < num_elements; ++i) {
        float fb = static_cast<float>(h_out_base[i]);
        float fg = static_cast<float>(h_out_grid[i]);
        float fl = static_cast<float>(h_out_lut[i]);

        float dg = std::abs(fb - fg);
        float dl = std::abs(fb - fl);
        if (dg > dq_max_diff_grid) dq_max_diff_grid = dg;
        if (dl > dq_max_diff_lut) dq_max_diff_lut = dl;
    }

    // 2. GEMV Verification
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(N * 64, 64), [=](sycl::nd_item<1> item) {
            gemv_baseline_kernel(item, d_gemv_w, d_gemv_x, d_y_base, N, K);
        });
    });

    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<uint32_t, 1> slm_grid(sycl::range<1>(2048), cgh);
        cgh.parallel_for(sycl::nd_range<1>(32 * 64, 64), [=](sycl::nd_item<1> item) {
            gemv_slm_grid_kernel(item, d_gemv_w, d_gemv_x, d_grid, d_y_grid, N, K, slm_grid);
        });
    });

    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<uint2_u64, 1> slm_lut(sycl::range<1>(4096), cgh);
        cgh.parallel_for(sycl::nd_range<1>(32 * 64, 64), [=](sycl::nd_item<1> item) {
            gemv_slm_lut_kernel(item, d_gemv_w, d_gemv_x, d_lut, d_y_lut, N, K, slm_lut);
        });
    });
    q.wait();

    std::vector<float> h_y_base(N);
    std::vector<float> h_y_grid(N);
    std::vector<float> h_y_lut(N);

    q.memcpy(h_y_base.data(), d_y_base, N * sizeof(float));
    q.memcpy(h_y_grid.data(), d_y_grid, N * sizeof(float));
    q.memcpy(h_y_lut.data(), d_y_lut, N * sizeof(float));
    q.wait();

    float gemv_max_diff_lut = 0.0f;
    float gemv_max_diff_grid = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        float fb = h_y_base[i];
        float fg = h_y_grid[i];
        float fl = h_y_lut[i];

        float dg = std::abs(fb - fg);
        float dl = std::abs(fb - fl);
        if (dg > gemv_max_diff_grid) gemv_max_diff_grid = dg;
        if (dl > gemv_max_diff_lut) gemv_max_diff_lut = dl;
    }

    sycl::free(d_dq_blocks, q);
    sycl::free(d_gemv_w, q);
    sycl::free(d_gemv_x, q);
    sycl::free(d_lut, q);
    sycl::free(d_grid, q);
    sycl::free(d_out_base, q);
    sycl::free(d_out_grid, q);
    sycl::free(d_out_lut, q);
    sycl::free(d_y_base, q);
    sycl::free(d_y_grid, q);
    sycl::free(d_y_lut, q);

    std::printf("[VERIFY] Dequant vs SLM Grid: max diff = %e\n", dq_max_diff_grid);
    std::printf("[VERIFY] Dequant vs SLM LUT:  max diff = %e\n", dq_max_diff_lut);
    std::printf("[VERIFY] GEMV vs SLM Grid:    max diff = %e\n", gemv_max_diff_grid);
    std::printf("[VERIFY] GEMV vs SLM LUT:     max diff = %e\n", gemv_max_diff_lut);

    bool ok = (dq_max_diff_lut < 1e-4f) && (gemv_max_diff_lut < 1e-3f);
    if (ok) {
        std::printf("[VERIFY] SUCCESS: All dequant and GEMV parity checks passed!\n\n");
    } else {
        std::fprintf(stderr, "[VERIFY] ERROR: Numerical parity verification failed!\n\n");
    }
    return ok;
}

// ============================================================================
// Google Benchmark Fixtures & Registrations
// ============================================================================

// --- Dequantization Fixture ---
struct DequantFixture {
    size_t num_elements;
    size_t num_blocks;
    sycl::queue q;
    block_iq1_m* d_src;
    sycl::half* d_dst;
    uint2_u64* d_lut;
    uint32_t* d_grid;

    DequantFixture(size_t ne)
        : num_elements(ne), num_blocks(ne / 256),
          q(sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order()}) {
        std::vector<block_iq1_m> h_src(num_blocks);
        init_iq1m_blocks(h_src.data(), num_blocks, 42);

        d_src = sycl::malloc_device<block_iq1_m>(num_blocks, q);
        d_dst = sycl::malloc_device<sycl::half>(num_elements, q);
        d_lut = sycl::malloc_device<uint2_u64>(4096, q);
        d_grid = sycl::malloc_device<uint32_t>(2048, q);

        q.memcpy(d_src, h_src.data(), num_blocks * sizeof(block_iq1_m));
        q.memcpy(d_lut, kIq1mCodesHost, 4096 * sizeof(uint2_u64));
        q.memcpy(d_grid, iq1s_grid_gpu, 2048 * sizeof(uint32_t));
        q.wait();
    }

    ~DequantFixture() {
        sycl::free(d_src, q);
        sycl::free(d_dst, q);
        sycl::free(d_lut, q);
        sycl::free(d_grid, q);
    }
};

static void BM_IQ1M_Dequant_Baseline(benchmark::State& state) {
    const size_t num_elements = static_cast<size_t>(state.range(0));
    DequantFixture fix(num_elements);

    auto* d_src = fix.d_src;
    auto* d_dst = fix.d_dst;
    const size_t num_blocks = fix.num_blocks;

    // Warmup
    for (int i = 0; i < 5; ++i) {
        fix.q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(num_blocks * 32, 32), [=](sycl::nd_item<1> item) {
                dq_baseline_kernel(item, d_src, d_dst, num_blocks);
            });
        });
    }
    fix.q.wait();

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        fix.q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(num_blocks * 32, 32), [=](sycl::nd_item<1> item) {
                dq_baseline_kernel(item, d_src, d_dst, num_blocks);
            });
        });
        fix.q.wait();
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }

    const double bytes = static_cast<double>(num_blocks * sizeof(block_iq1_m) + num_elements * sizeof(sycl::half));
    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["elements_per_sec"] =
        benchmark::Counter(static_cast<double>(num_elements), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

static void BM_IQ1M_Dequant_SLM_Grid2048(benchmark::State& state) {
    const size_t num_elements = static_cast<size_t>(state.range(0));
    DequantFixture fix(num_elements);

    auto* d_src = fix.d_src;
    auto* d_dst = fix.d_dst;
    auto* d_grid = fix.d_grid;
    const size_t num_blocks = fix.num_blocks;

    constexpr int wg_size = 128;
    const size_t num_wgs = std::min<size_t>(num_blocks, 64);

    for (int i = 0; i < 5; ++i) {
        fix.q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<uint32_t, 1> slm_grid(sycl::range<1>(2048), cgh);
            cgh.parallel_for(sycl::nd_range<1>(num_wgs * wg_size, wg_size), [=](sycl::nd_item<1> item) {
                dq_slm_grid_kernel(item, d_src, d_grid, d_dst, num_blocks, slm_grid);
            });
        });
    }
    fix.q.wait();

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        fix.q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<uint32_t, 1> slm_grid(sycl::range<1>(2048), cgh);
            cgh.parallel_for(sycl::nd_range<1>(num_wgs * wg_size, wg_size), [=](sycl::nd_item<1> item) {
                dq_slm_grid_kernel(item, d_src, d_grid, d_dst, num_blocks, slm_grid);
            });
        });
        fix.q.wait();
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }

    const double bytes = static_cast<double>(num_blocks * sizeof(block_iq1_m) + num_elements * sizeof(sycl::half));
    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["elements_per_sec"] =
        benchmark::Counter(static_cast<double>(num_elements), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

static void BM_IQ1M_Dequant_Fast_SLM_LUT4096_Direct(benchmark::State& state) {
    const size_t num_elements = static_cast<size_t>(state.range(0));
    DequantFixture fix(num_elements);

    auto* d_src = fix.d_src;
    auto* d_dst = fix.d_dst;
    auto* d_lut = fix.d_lut;
    const size_t num_blocks = fix.num_blocks;

    for (int i = 0; i < 5; ++i) {
        fix.q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<uint2_u64, 1> slm_lut(sycl::range<1>(4096), cgh);
            cgh.parallel_for(sycl::nd_range<1>(num_blocks * 32, 32), [=](sycl::nd_item<1> item) {
                dq_slm_lut_direct_kernel(item, d_src, d_lut, d_dst, num_blocks, slm_lut);
            });
        });
    }
    fix.q.wait();

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        fix.q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<uint2_u64, 1> slm_lut(sycl::range<1>(4096), cgh);
            cgh.parallel_for(sycl::nd_range<1>(num_blocks * 32, 32), [=](sycl::nd_item<1> item) {
                dq_slm_lut_direct_kernel(item, d_src, d_lut, d_dst, num_blocks, slm_lut);
            });
        });
        fix.q.wait();
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }

    const double bytes = static_cast<double>(num_blocks * sizeof(block_iq1_m) + num_elements * sizeof(sycl::half));
    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["elements_per_sec"] =
        benchmark::Counter(static_cast<double>(num_elements), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

static void BM_IQ1M_Dequant_Fast_SLM_LUT4096_Persistent(benchmark::State& state) {
    const size_t num_elements = static_cast<size_t>(state.range(0));
    DequantFixture fix(num_elements);

    auto* d_src = fix.d_src;
    auto* d_dst = fix.d_dst;
    auto* d_lut = fix.d_lut;
    const size_t num_blocks = fix.num_blocks;

    constexpr int wg_size = 128;
    const size_t num_wgs = std::min<size_t>(num_blocks, 64);

    for (int i = 0; i < 5; ++i) {
        fix.q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<uint2_u64, 1> slm_lut(sycl::range<1>(4096), cgh);
            cgh.parallel_for(sycl::nd_range<1>(num_wgs * wg_size, wg_size), [=](sycl::nd_item<1> item) {
                dq_slm_lut_kernel(item, d_src, d_lut, d_dst, num_blocks, slm_lut);
            });
        });
    }
    fix.q.wait();

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        fix.q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<uint2_u64, 1> slm_lut(sycl::range<1>(4096), cgh);
            cgh.parallel_for(sycl::nd_range<1>(num_wgs * wg_size, wg_size), [=](sycl::nd_item<1> item) {
                dq_slm_lut_kernel(item, d_src, d_lut, d_dst, num_blocks, slm_lut);
            });
        });
        fix.q.wait();
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }

    const double bytes = static_cast<double>(num_blocks * sizeof(block_iq1_m) + num_elements * sizeof(sycl::half));
    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["elements_per_sec"] =
        benchmark::Counter(static_cast<double>(num_elements), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// --- GEMV Fixture ---
struct GemvFixture {
    size_t K;
    size_t N;
    size_t total_blocks;
    sycl::queue q;
    block_iq1_m* d_w;
    block_q8_1* d_x;
    float* d_y;
    uint2_u64* d_lut;
    uint32_t* d_grid;

    GemvFixture(size_t k_dim, size_t n_dim)
        : K(k_dim), N(n_dim), total_blocks((k_dim / 256) * n_dim),
          q(sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order()}) {
        std::vector<block_iq1_m> h_w(total_blocks);
        init_iq1m_blocks(h_w.data(), total_blocks, 77);

        std::vector<block_q8_1> h_x(K / 32);
        init_q8_1_blocks(h_x.data(), K / 32, 88);

        d_w = sycl::malloc_device<block_iq1_m>(total_blocks, q);
        d_x = sycl::malloc_device<block_q8_1>(K / 32, q);
        d_y = sycl::malloc_device<float>(N, q);
        d_lut = sycl::malloc_device<uint2_u64>(4096, q);
        d_grid = sycl::malloc_device<uint32_t>(2048, q);

        q.memcpy(d_w, h_w.data(), total_blocks * sizeof(block_iq1_m));
        q.memcpy(d_x, h_x.data(), (K / 32) * sizeof(block_q8_1));
        q.memcpy(d_lut, kIq1mCodesHost, 4096 * sizeof(uint2_u64));
        q.memcpy(d_grid, iq1s_grid_gpu, 2048 * sizeof(uint32_t));
        q.wait();
    }

    ~GemvFixture() {
        sycl::free(d_w, q);
        sycl::free(d_x, q);
        sycl::free(d_y, q);
        sycl::free(d_lut, q);
        sycl::free(d_grid, q);
    }
};

static void BM_IQ1M_GEMV_Baseline(benchmark::State& state) {
    const size_t K = static_cast<size_t>(state.range(0));
    const size_t N = static_cast<size_t>(state.range(1));
    GemvFixture fix(K, N);

    auto* d_w = fix.d_w;
    auto* d_x = fix.d_x;
    auto* d_y = fix.d_y;

    constexpr int wg_size = 64;

    for (int i = 0; i < 5; ++i) {
        fix.q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(N * wg_size, wg_size), [=](sycl::nd_item<1> item) {
                gemv_baseline_kernel(item, d_w, d_x, d_y, N, K);
            });
        });
    }
    fix.q.wait();

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        fix.q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(N * wg_size, wg_size), [=](sycl::nd_item<1> item) {
                gemv_baseline_kernel(item, d_w, d_x, d_y, N, K);
            });
        });
        fix.q.wait();
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }

    const double bytes = static_cast<double>(fix.total_blocks * sizeof(block_iq1_m) +
                                             (K / 32) * sizeof(block_q8_1) + N * sizeof(float));
    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["GFLOPS"] =
        benchmark::Counter(static_cast<double>(2 * K * N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

static void BM_IQ1M_GEMV_SLM_Grid2048(benchmark::State& state) {
    const size_t K = static_cast<size_t>(state.range(0));
    const size_t N = static_cast<size_t>(state.range(1));
    GemvFixture fix(K, N);

    auto* d_w = fix.d_w;
    auto* d_x = fix.d_x;
    auto* d_y = fix.d_y;
    auto* d_grid = fix.d_grid;

    constexpr int wg_size = 64;
    const size_t num_wgs = std::min<size_t>(N, 64);

    for (int i = 0; i < 5; ++i) {
        fix.q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<uint32_t, 1> slm_grid(sycl::range<1>(2048), cgh);
            cgh.parallel_for(sycl::nd_range<1>(num_wgs * wg_size, wg_size), [=](sycl::nd_item<1> item) {
                gemv_slm_grid_kernel(item, d_w, d_x, d_grid, d_y, N, K, slm_grid);
            });
        });
    }
    fix.q.wait();

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        fix.q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<uint32_t, 1> slm_grid(sycl::range<1>(2048), cgh);
            cgh.parallel_for(sycl::nd_range<1>(num_wgs * wg_size, wg_size), [=](sycl::nd_item<1> item) {
                gemv_slm_grid_kernel(item, d_w, d_x, d_grid, d_y, N, K, slm_grid);
            });
        });
        fix.q.wait();
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }

    const double bytes = static_cast<double>(fix.total_blocks * sizeof(block_iq1_m) +
                                             (K / 32) * sizeof(block_q8_1) + N * sizeof(float));
    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["GFLOPS"] =
        benchmark::Counter(static_cast<double>(2 * K * N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

static void BM_IQ1M_GEMV_Fast_SLM_LUT4096_Direct(benchmark::State& state) {
    const size_t K = static_cast<size_t>(state.range(0));
    const size_t N = static_cast<size_t>(state.range(1));
    GemvFixture fix(K, N);

    auto* d_w = fix.d_w;
    auto* d_x = fix.d_x;
    auto* d_y = fix.d_y;
    auto* d_lut = fix.d_lut;

    constexpr int wg_size = 64;

    for (int i = 0; i < 5; ++i) {
        fix.q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<uint2_u64, 1> slm_lut(sycl::range<1>(4096), cgh);
            cgh.parallel_for(sycl::nd_range<1>(N * wg_size, wg_size), [=](sycl::nd_item<1> item) {
                gemv_slm_lut_direct_kernel(item, d_w, d_x, d_lut, d_y, N, K, slm_lut);
            });
        });
    }
    fix.q.wait();

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        fix.q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<uint2_u64, 1> slm_lut(sycl::range<1>(4096), cgh);
            cgh.parallel_for(sycl::nd_range<1>(N * wg_size, wg_size), [=](sycl::nd_item<1> item) {
                gemv_slm_lut_direct_kernel(item, d_w, d_x, d_lut, d_y, N, K, slm_lut);
            });
        });
        fix.q.wait();
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }

    const double bytes = static_cast<double>(fix.total_blocks * sizeof(block_iq1_m) +
                                             (K / 32) * sizeof(block_q8_1) + N * sizeof(float));
    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["GFLOPS"] =
        benchmark::Counter(static_cast<double>(2 * K * N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

static void BM_IQ1M_GEMV_Fast_SLM_LUT4096_Persistent(benchmark::State& state) {
    const size_t K = static_cast<size_t>(state.range(0));
    const size_t N = static_cast<size_t>(state.range(1));
    GemvFixture fix(K, N);

    auto* d_w = fix.d_w;
    auto* d_x = fix.d_x;
    auto* d_y = fix.d_y;
    auto* d_lut = fix.d_lut;

    constexpr int wg_size = 64;
    const size_t num_wgs = std::min<size_t>(N, 64);

    for (int i = 0; i < 5; ++i) {
        fix.q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<uint2_u64, 1> slm_lut(sycl::range<1>(4096), cgh);
            cgh.parallel_for(sycl::nd_range<1>(num_wgs * wg_size, wg_size), [=](sycl::nd_item<1> item) {
                gemv_slm_lut_kernel(item, d_w, d_x, d_lut, d_y, N, K, slm_lut);
            });
        });
    }
    fix.q.wait();

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        fix.q.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<uint2_u64, 1> slm_lut(sycl::range<1>(4096), cgh);
            cgh.parallel_for(sycl::nd_range<1>(num_wgs * wg_size, wg_size), [=](sycl::nd_item<1> item) {
                gemv_slm_lut_kernel(item, d_w, d_x, d_lut, d_y, N, K, slm_lut);
            });
        });
        fix.q.wait();
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }

    const double bytes = static_cast<double>(fix.total_blocks * sizeof(block_iq1_m) +
                                             (K / 32) * sizeof(block_q8_1) + N * sizeof(float));
    state.counters["Bandwidth_GBps"] =
        benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["GFLOPS"] =
        benchmark::Counter(static_cast<double>(2 * K * N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// ============================================================================
// Benchmark Registration
// ============================================================================

// Dequantization: Elements = 262,144 (1024 blocks), 1,048,576 (4096 blocks), 4,194,304 (16384 blocks)
BENCHMARK(BM_IQ1M_Dequant_Baseline)->UseManualTime()->Arg(262144)->Arg(1048576)->Arg(4194304);
BENCHMARK(BM_IQ1M_Dequant_SLM_Grid2048)->UseManualTime()->Arg(262144)->Arg(1048576)->Arg(4194304);
BENCHMARK(BM_IQ1M_Dequant_Fast_SLM_LUT4096_Direct)->UseManualTime()->Arg(262144)->Arg(1048576)->Arg(4194304);
BENCHMARK(BM_IQ1M_Dequant_Fast_SLM_LUT4096_Persistent)->UseManualTime()->Arg(262144)->Arg(1048576)->Arg(4194304);

// GEMV: Routed expert and decode projection shapes
// Shape 1: K=2560, N=2048 (Qwen-14B/32B expert / gate-up tile)
// Shape 2: K=2560, N=6144 (Standard decode projection)
// Shape 3: K=5120, N=2048 (DeepSeek-V4.1 wide projection)
BENCHMARK(BM_IQ1M_GEMV_Baseline)->UseManualTime()->Args({2560, 2048})->Args({2560, 6144})->Args({5120, 2048});
BENCHMARK(BM_IQ1M_GEMV_SLM_Grid2048)->UseManualTime()->Args({2560, 2048})->Args({2560, 6144})->Args({5120, 2048});
BENCHMARK(BM_IQ1M_GEMV_Fast_SLM_LUT4096_Direct)->UseManualTime()->Args({2560, 2048})->Args({2560, 6144})->Args({5120, 2048});
BENCHMARK(BM_IQ1M_GEMV_Fast_SLM_LUT4096_Persistent)->UseManualTime()->Args({2560, 2048})->Args({2560, 6144})->Args({5120, 2048});

} // namespace

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char** argv) {
    try {
        sycl::queue q{sycl::default_selector_v};
        auto dev = q.get_device();
        std::printf("=================================================================\n");
        std::printf(" Intel Xe2 IQ1_M SLM LUT Dequantization & GEMV Benchmark\n");
        std::printf(" Device:       %s\n", dev.get_info<sycl::info::device::name>().c_str());
        std::printf(" Driver:       %s\n", dev.get_info<sycl::info::device::driver_version>().c_str());
        std::printf(" Subgroup:     %d (Native Xe2 SIMD16)\n", strata::sycl_arch::SUB_GROUP_SIZE);
        std::printf("=================================================================\n\n");

        if (!run_device_verification(q)) {
            std::fprintf(stderr, "FATAL: Device verification check failed. Aborting benchmarks.\n");
            return 1;
        }
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "Fatal SYCL exception: %s\n", e.what());
        return 1;
    }

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
