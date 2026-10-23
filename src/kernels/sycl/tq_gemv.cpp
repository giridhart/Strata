// src/kernels/sycl/tq_gemv.cpp - Pure SYCL 2020 Trellis Quantization (TQ) decoders, FWHT, and GEMV for Intel Xe2.
#include <sycl/sycl.hpp>
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/tq_gemv.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

inline sycl::queue& default_queue() {
    static sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    return q;
}

constexpr int SG_SIZE = 16;
constexpr int WG_THREADS = 128;
constexpr int SGS_PER_WG = WG_THREADS / SG_SIZE; // 8 subgroups per workgroup

// -----------------------------------------------------------------------------
// Kernel: Fast Walsh-Hadamard Transform (hadamard_rows)
// -----------------------------------------------------------------------------
template <int BLOCK_SIZE>
void hadamard_rows_kernel(sycl::nd_item<1> item,
                          const float* __restrict__ in,
                          float* __restrict__ out,
                          int64_t rows, int64_t cols,
                          const float* __restrict__ signs,
                          float scale,
                          sycl::local_accessor<float, 1> hbuf) {
    const int tid = item.get_local_id(0);
    const int64_t num_blocks_per_row = cols / BLOCK_SIZE;
    const int64_t gid = item.get_group(0);
    const int64_t row = gid / num_blocks_per_row;
    const int64_t col_block = gid % num_blocks_per_row;
    if (row >= rows) return;

    const int64_t col = col_block * BLOCK_SIZE + tid;
    const int64_t idx = row * cols + col;

    float v = in[idx];
    if (signs != nullptr) {
        v *= signs[col];
    }
    hbuf[tid] = v;
    sycl::group_barrier(item.get_group());

#pragma unroll
    for (int len = 1; len < BLOCK_SIZE; len <<= 1) {
        if ((tid & len) == 0) {
            const float a = hbuf[tid];
            const float c = hbuf[tid + len];
            hbuf[tid] = a + c;
            hbuf[tid + len] = a - c;
        }
        sycl::group_barrier(item.get_group());
    }

    out[idx] = hbuf[tid] * scale;
}

// -----------------------------------------------------------------------------
// Kernel: TQ Dequantization (FP16 codebook)
// -----------------------------------------------------------------------------
template <int TY>
void dequantize_tq_kernel(sycl::nd_item<1> item,
                          const void* __restrict__ vx,
                          float* __restrict__ y,
                          int64_t n_blocks) {
    const int64_t global_lane = item.get_global_id(0);
    const int64_t blk_idx = global_lane / 8;
    const int l = static_cast<int>(global_lane % 8);
    if (blk_idx >= n_blocks) return;

    constexpr size_t bsize = tq_block_size(TY);
    const uint8_t* blk_ptr = static_cast<const uint8_t*>(vx) + blk_idx * bsize;
    const uint16_t d_raw = *reinterpret_cast<const uint16_t*>(blk_ptr);
    const float d = tq_half_to_float(d_raw);
    const uint8_t* qs = blk_ptr + 2;

    uint32_t s[4];
    tq_states4<TY>(qs, l, s);

    const int64_t base_out = blk_idx * 128 + l * 16;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t x = (s[j] & TQ_STATE_MASK) * 0x9E3779B1u;
        const uint32_t pa = x >> 21;
        const uint32_t pb = (x >> 10) & 2047u;

        y[base_out + j * 4 + 0] = d * tq_half_to_float(tq_lut_f16[2 * pa + 0]);
        y[base_out + j * 4 + 1] = d * tq_half_to_float(tq_lut_f16[2 * pa + 1]);
        y[base_out + j * 4 + 2] = d * tq_half_to_float(tq_lut_f16[2 * pb + 0]);
        y[base_out + j * 4 + 3] = d * tq_half_to_float(tq_lut_f16[2 * pb + 1]);
    }
}

// -----------------------------------------------------------------------------
// Kernel: Native Xe2 SIMD16 Subgroup TQ GEMV
// -----------------------------------------------------------------------------
template <int TY, bool STAGE_SLM>
void tq_gemv_subgroup_kernel(sycl::nd_item<1> item,
                             const void* __restrict__ w,
                             const float* __restrict__ x,
                             float* __restrict__ y,
                             int64_t n_out, int64_t n_in,
                             sycl::local_accessor<uint16_t, 1> s_lut) {
    const auto sg = item.get_sub_group();
    const int sg_id = static_cast<int>(item.get_local_id(0) / SG_SIZE);
    const int64_t row = static_cast<int64_t>(item.get_group(0)) * SGS_PER_WG + sg_id;
    const int s_lane = static_cast<int>(sg.get_local_linear_id());

    // Cooperative SLM staging of 4 KB codebook across workgroup
    if constexpr (STAGE_SLM) {
        const int tid = static_cast<int>(item.get_local_id(0));
#pragma unroll
        for (int i = tid; i < 2048; i += WG_THREADS) {
            s_lut[i] = tq_lut_i8[i];
        }
        sycl::group_barrier(item.get_group());
    }

    if (row >= n_out) return;

    constexpr size_t bsize = tq_block_size(TY);
    const int64_t n_blocks = n_in / 128;
    const uint8_t* row_w = static_cast<const uint8_t*>(w) + row * (n_blocks * bsize);

    float acc = 0.0f;
    const int blk_offset = s_lane < 8 ? 0 : 1;
    const int l = s_lane & 7;

    const int64_t n_block_pairs = (n_blocks + 1) / 2;
    for (int64_t bp = 0; bp < n_block_pairs; ++bp) {
        const int64_t b = bp * 2 + blk_offset;
        if (b < n_blocks) {
            const uint8_t* blk_ptr = row_w + b * bsize;
            const uint16_t d_raw = *reinterpret_cast<const uint16_t*>(blk_ptr);
            const float d = tq_half_to_float(d_raw) * TQ_LUT_I8_SCALE;
            const uint8_t* qs = blk_ptr + 2;

            uint32_t s[4];
            tq_states4<TY>(qs, l, s);

            float lane_sum = 0.0f;
            const int64_t col_base = b * 128 + l * 16;

#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t x_code = (s[j] & TQ_STATE_MASK) * 0x9E3779B1u;
                uint16_t pa, pb;
                if constexpr (STAGE_SLM) {
                    pa = s_lut[x_code >> 21];
                    pb = s_lut[(x_code >> 10) & 2047u];
                } else {
                    pa = tq_lut_i8[x_code >> 21];
                    pb = tq_lut_i8[(x_code >> 10) & 2047u];
                }

                const int8_t w0 = static_cast<int8_t>(pa & 0xFF);
                const int8_t w1 = static_cast<int8_t>(pa >> 8);
                const int8_t w2 = static_cast<int8_t>(pb & 0xFF);
                const int8_t w3 = static_cast<int8_t>(pb >> 8);

                const int64_t col = col_base + j * 4;
                sycl::vec<float, 4> xv = *reinterpret_cast<const sycl::vec<float, 4>*>(&x[col]);

                lane_sum += static_cast<float>(w0) * xv[0]
                          + static_cast<float>(w1) * xv[1]
                          + static_cast<float>(w2) * xv[2]
                          + static_cast<float>(w3) * xv[3];
            }
            acc += lane_sum * d;
        }
    }

    const float total = sycl::reduce_over_group(sg, acc, sycl::plus<>());
    if (sg.leader()) {
        y[row] = total;
    }
}

} // namespace

// -----------------------------------------------------------------------------
// Host API implementations
// -----------------------------------------------------------------------------

void hadamard_rows(const float* in, float* out, int64_t rows, int64_t cols, int block,
                   const float* signs, void* stream) {
    if (rows <= 0 || cols <= 0) return;
    if (block != 128 && block != 256) {
        std::fprintf(stderr, "hadamard_rows: unsupported block size %d (must be 128 or 256)\n", block);
        std::exit(1);
    }
    if (cols % block != 0) {
        std::fprintf(stderr, "hadamard_rows: cols (%lld) not a multiple of block (%d)\n",
                     static_cast<long long>(cols), block);
        std::exit(1);
    }

    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    const int64_t total_chunks = rows * (cols / block);
    const float scale = 1.0f / std::sqrt(static_cast<float>(block));

    if (block == 128) {
        q->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> hbuf(128, cgh);
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(total_chunks * 128), sycl::range<1>(128)),
                             [=](sycl::nd_item<1> item) {
                                 hadamard_rows_kernel<128>(item, in, out, rows, cols, signs, scale, hbuf);
                             });
        });
    } else {
        q->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> hbuf(256, cgh);
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(total_chunks * 256), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) {
                                 hadamard_rows_kernel<256>(item, in, out, rows, cols, signs, scale, hbuf);
                             });
        });
    }
}

void dequantize_tq(int type, const void* vx, float* y, int64_t n, void* stream) {
    if (n <= 0 || !is_tq_type(type)) return;
    if (n % 128 != 0) {
        std::fprintf(stderr, "dequantize_tq: n (%lld) not a multiple of 128\n", static_cast<long long>(n));
        std::exit(1);
    }

    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    const int64_t n_blocks = n / 128;
    const int64_t total_lanes = n_blocks * 8;
    constexpr int WG = 128;
    const int64_t grid = ((total_lanes + WG - 1) / WG) * WG;

    auto launch = [&](auto type_tag) {
        constexpr int TY = decltype(type_tag)::value;
        q->parallel_for(sycl::nd_range<1>(sycl::range<1>(grid), sycl::range<1>(WG)),
                        [=](sycl::nd_item<1> item) {
                            dequantize_tq_kernel<TY>(item, vx, y, n_blocks);
                        });
    };

    if (type == 144) launch(std::integral_constant<int, 144>{});
    else if (type == 145) launch(std::integral_constant<int, 145>{});
    else if (type == 146) launch(std::integral_constant<int, 146>{});
}

void tq_gemv_ext(int type, const void* w, const float* x, float* y, int64_t n_out, int64_t n_in,
                 bool stage_slm, void* stream) {
    if (n_out <= 0 || n_in <= 0 || !is_tq_type(type)) return;
    if (n_in % 128 != 0) {
        std::fprintf(stderr, "tq_gemv: n_in (%lld) not a multiple of 128\n", static_cast<long long>(n_in));
        std::exit(1);
    }

    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    const int64_t total_wgs = (n_out + SGS_PER_WG - 1) / SGS_PER_WG;
    const size_t global_size = static_cast<size_t>(total_wgs * WG_THREADS);

    auto dispatch = [&](auto ty_tag, auto slm_tag) {
        constexpr int TY = decltype(ty_tag)::value;
        constexpr bool SLM = decltype(slm_tag)::value;

        q->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<uint16_t, 1> s_lut(SLM ? 2048 : 0, cgh);
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(WG_THREADS)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                                 tq_gemv_subgroup_kernel<TY, SLM>(item, w, x, y, n_out, n_in, s_lut);
                             });
        });
    };

    if (stage_slm) {
        if (type == 144) dispatch(std::integral_constant<int, 144>{}, std::true_type{});
        else if (type == 145) dispatch(std::integral_constant<int, 145>{}, std::true_type{});
        else if (type == 146) dispatch(std::integral_constant<int, 146>{}, std::true_type{});
    } else {
        if (type == 144) dispatch(std::integral_constant<int, 144>{}, std::false_type{});
        else if (type == 145) dispatch(std::integral_constant<int, 145>{}, std::false_type{});
        else if (type == 146) dispatch(std::integral_constant<int, 146>{}, std::false_type{});
    }
}

void tq_gemv(int type, const void* w, const float* x, float* y, int64_t n_out, int64_t n_in,
             void* stream) {
    tq_gemv_ext(type, w, x, y, n_out, n_in, /*stage_slm=*/true, stream);
}

} // namespace strata::kernels
