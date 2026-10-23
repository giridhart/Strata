// src/kernels/sycl/bf16_gemv.dp.cpp - Pure SYCL 2020 BF16 GEMV kernels.
// Optimized for Intel Xe2 microarchitecture (Battlemage B580 and Lunar Lake) native SIMD16 execution.
#include <sycl/sycl.hpp>
#include <cstdint>
#include <algorithm>
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int SG_SIZE = strata::sycl_arch::SUB_GROUP_SIZE;

inline sycl::queue& default_queue() {
    static sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    return q;
}

/// One thread per output row, walking it contiguously. Kept as the naive reference the split version is
/// checked against, and used directly for the shapes where the output width is already large.
[[maybe_unused]] void bf16_gemv_naive_kernel(sycl::nd_item<1> item, const uint16_t* __restrict__ x,
                                             const uint16_t* __restrict__ w, float* __restrict__ y, long long n_in,
                                             long long n_out) {
    const long long o = (long long)item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    if (o >= n_out) return;
    const uint16_t* row = w + o * n_in;
    float acc = 0.0f;
    for (long long i = 0; i < n_in; ++i)
        acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);
    y[o] = acc;
}

/// One SUBGROUP per output row, lanes striding the reduction axis. On Xe2 (B580/Lunar Lake),
/// this executes natively in SIMD16 with single-cycle hardware group reduction.
template <int SUB_SIZE = SG_SIZE, bool STREAMING = false>
void bf16_gemv_subgroup_kernel(sycl::nd_item<1> item, const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                               float* __restrict__ y, long long n_in, long long n_out) {
    const auto sg = item.get_sub_group();
    const int sgs_per_block = (int)(item.get_local_range(0) / SUB_SIZE);
    const long long o = (long long)item.get_group(0) * sgs_per_block + (long long)(item.get_local_id(0) / SUB_SIZE);
    if (o >= n_out) return;
    const int lane = sg.get_local_linear_id();
    const uint16_t* row = w + o * n_in;
    float acc = 0.0f;
    if constexpr (STREAMING) {
        auto stream_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::intel::experimental::read_hint<
                sycl::ext::intel::experimental::cache_control<
                    sycl::ext::intel::experimental::cache_mode::streaming,
                    sycl::ext::oneapi::experimental::cache_level::L3>>};
        sycl::ext::oneapi::experimental::annotated_ptr<
            const uint16_t, decltype(stream_props)> row_stream(row, stream_props);
        for (long long i = lane; i < n_in; i += SUB_SIZE)
            acc += f32_from_bf16(x[i]) * f32_from_bf16(uint16_t(row_stream[i]));
    } else {
        for (long long i = lane; i < n_in; i += SUB_SIZE)
            acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);
    }
    acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[o] = acc;
}

/// Dual-weight GEMV from one read of x (llama.cpp `*_mul_mat_multi` equivalent).
/// One subgroup computes row `o` of both matrices simultaneously with dual accumulators over the SAME
/// lane-strided pass, loading `x[i]` once into registers.
/// On Xe2 (B580/Lunar Lake), this executes natively in SIMD16 with single-cycle hardware group reduction.
template <int SUB_SIZE = SG_SIZE, bool STREAMING = false>
void bf16_gemv_pair_subgroup_kernel(sycl::nd_item<1> item,
                                    const uint16_t* __restrict__ x,
                                    const uint16_t* __restrict__ w1, float* __restrict__ y1, long long n_out1,
                                    const uint16_t* __restrict__ w2, float* __restrict__ y2, long long n_out2,
                                    long long n_in) {
    const auto sg = item.get_sub_group();
    const int sgs_per_block = (int)(item.get_local_range(0) / SUB_SIZE);
    const long long o = (long long)item.get_group(0) * sgs_per_block + (long long)(item.get_local_id(0) / SUB_SIZE);
    const long long n_out = n_out1 > n_out2 ? n_out1 : n_out2;
    if (o >= n_out) return;

    const bool run1 = (w1 != nullptr && y1 != nullptr && o < n_out1);
    const bool run2 = (w2 != nullptr && y2 != nullptr && o < n_out2);
    if (!run1 && !run2) return;

    const int lane = sg.get_local_linear_id();
    const uint16_t* r1 = run1 ? (w1 + o * n_in) : nullptr;
    const uint16_t* r2 = run2 ? (w2 + o * n_in) : nullptr;
    float acc1 = 0.0f;
    float acc2 = 0.0f;

    if constexpr (STREAMING) {
        auto stream_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::intel::experimental::read_hint<
                sycl::ext::intel::experimental::cache_control<
                    sycl::ext::intel::experimental::cache_mode::streaming,
                    sycl::ext::oneapi::experimental::cache_level::L3>>};

        if (run1 && run2) {
            sycl::ext::oneapi::experimental::annotated_ptr<
                const uint16_t, decltype(stream_props)> r1_stream(r1, stream_props);
            sycl::ext::oneapi::experimental::annotated_ptr<
                const uint16_t, decltype(stream_props)> r2_stream(r2, stream_props);
            for (long long i = lane; i < n_in; i += SUB_SIZE) {
                const float xv = f32_from_bf16(x[i]);
                acc1 += xv * f32_from_bf16(uint16_t(r1_stream[i]));
                acc2 += xv * f32_from_bf16(uint16_t(r2_stream[i]));
            }
        } else if (run1) {
            sycl::ext::oneapi::experimental::annotated_ptr<
                const uint16_t, decltype(stream_props)> r1_stream(r1, stream_props);
            for (long long i = lane; i < n_in; i += SUB_SIZE) {
                acc1 += f32_from_bf16(x[i]) * f32_from_bf16(uint16_t(r1_stream[i]));
            }
        } else if (run2) {
            sycl::ext::oneapi::experimental::annotated_ptr<
                const uint16_t, decltype(stream_props)> r2_stream(r2, stream_props);
            for (long long i = lane; i < n_in; i += SUB_SIZE) {
                acc2 += f32_from_bf16(x[i]) * f32_from_bf16(uint16_t(r2_stream[i]));
            }
        }
    } else {
        if (run1 && run2) {
            for (long long i = lane; i < n_in; i += SUB_SIZE) {
                const float xv = f32_from_bf16(x[i]);
                acc1 += xv * f32_from_bf16(r1[i]);
                acc2 += xv * f32_from_bf16(r2[i]);
            }
        } else if (run1) {
            for (long long i = lane; i < n_in; i += SUB_SIZE) {
                acc1 += f32_from_bf16(x[i]) * f32_from_bf16(r1[i]);
            }
        } else if (run2) {
            for (long long i = lane; i < n_in; i += SUB_SIZE) {
                acc2 += f32_from_bf16(x[i]) * f32_from_bf16(r2[i]);
            }
        }
    }

    if (run1) {
        acc1 = sycl::reduce_over_group(sg, acc1, sycl::plus<float>());
        if (lane == 0) y1[o] = acc1;
    }
    if (run2) {
        acc2 = sycl::reduce_over_group(sg, acc2, sycl::plus<float>());
        if (lane == 0) y2[o] = acc2;
    }
}

/// TPR lanes cooperate on ONE row's reduction: `threads_per_row` threads each take a strided slice and the
/// block reduces through shared memory. On Xe2, we first reduce within each SIMD16 subgroup via
/// single-cycle hardware reduction, then reduce the subgroup sums in local memory.
template <int SUB_SIZE = SG_SIZE, bool STREAMING = false>
void bf16_gemv_split_kernel(sycl::nd_item<1> item, const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                            float* __restrict__ y, long long n_in, long long n_out, int tpr,
                            sycl::local_accessor<float, 1> scratch) {
    const long long o = item.get_group(0);
    if (o >= n_out) return;
    const auto sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const int t = item.get_local_id(0);  // 0 .. tpr-1
    const int sg_id = t / SUB_SIZE;
    const int num_sgs = (tpr + SUB_SIZE - 1) / SUB_SIZE;

    const uint16_t* row = w + o * n_in;
    float acc = 0.0f;
    if constexpr (STREAMING) {
        auto stream_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::intel::experimental::read_hint<
                sycl::ext::intel::experimental::cache_control<
                    sycl::ext::intel::experimental::cache_mode::streaming,
                    sycl::ext::oneapi::experimental::cache_level::L3>>};
        sycl::ext::oneapi::experimental::annotated_ptr<
            const uint16_t, decltype(stream_props)> row_stream(row, stream_props);
        for (long long i = t; i < n_in; i += tpr)
            acc += f32_from_bf16(x[i]) * f32_from_bf16(uint16_t(row_stream[i]));
    } else {
        for (long long i = t; i < n_in; i += tpr)
            acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);
    }

    // Subgroup reduction first (single-cycle hardware reduction on Xe2)
    acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) scratch[sg_id] = acc;
    item.barrier(sycl::access::fence_space::local_space);

    // First subgroup reduces the per-subgroup partials
    if (sg_id == 0) {
        float final_acc = 0.0f;
        for (int k = lane; k < num_sgs; k += SUB_SIZE)
            final_acc += scratch[k];
        final_acc = sycl::reduce_over_group(sg, final_acc, sycl::plus<float>());
        if (lane == 0) y[o] = final_acc;
    }
}

}  // namespace

void bf16_gemv(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, void* stream) {
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    if (n_in <= 0 || n_out <= 0) return;
    const auto prof = strata::sycl_arch::get_xe2_tuning_profile(q->get_device());
    if (n_out >= 64) {
        const int sgs = THREADS / SG_SIZE;
        const unsigned grid = (unsigned)((n_out + sgs - 1) / sgs);
        if (prof.enable_streaming_hints) {
            q->parallel_for(sycl::nd_range<1>(sycl::range<1>((grid) * (THREADS)), sycl::range<1>(THREADS)),
                            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                                bf16_gemv_subgroup_kernel<SG_SIZE, true>(item, x, w, y, n_in, n_out);
                            });
        } else {
            q->parallel_for(sycl::nd_range<1>(sycl::range<1>((grid) * (THREADS)), sycl::range<1>(THREADS)),
                            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                                bf16_gemv_subgroup_kernel<SG_SIZE, false>(item, x, w, y, n_in, n_out);
                            });
        }
        return;
    }
    // For narrow shapes (n_out < 64, e.g. ssm_alpha [2560, 48]), dispatch the SLM split reduction
    // with 128 threads per row to fully saturate Xe2 Vector Engines (160 engines).
    bf16_gemv_split(x, w, y, n_in, n_out, 128, stream);
}

void bf16_gemv_pair(const uint16_t* x, const uint16_t* w1, float* y1, int64_t n_out1,
                    const uint16_t* w2, float* y2, int64_t n_out2, int64_t n_in, void* stream) {
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    if (n_in <= 0 || (n_out1 <= 0 && n_out2 <= 0)) return;
    const long long n_out = n_out1 > n_out2 ? n_out1 : n_out2;
    const int sgs = THREADS / SG_SIZE;
    const unsigned grid = (unsigned)((n_out + sgs - 1) / sgs);
    const auto prof = strata::sycl_arch::get_xe2_tuning_profile(q->get_device());
    if (prof.enable_streaming_hints) {
        q->parallel_for(sycl::nd_range<1>(sycl::range<1>((grid) * (THREADS)), sycl::range<1>(THREADS)),
                        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                            bf16_gemv_pair_subgroup_kernel<SG_SIZE, true>(item, x, w1, y1, n_out1, w2, y2, n_out2, n_in);
                        });
    } else {
        q->parallel_for(sycl::nd_range<1>(sycl::range<1>((grid) * (THREADS)), sycl::range<1>(THREADS)),
                        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                            bf16_gemv_pair_subgroup_kernel<SG_SIZE, false>(item, x, w1, y1, n_out1, w2, y2, n_out2, n_in);
                        });
    }
}

void bf16_gemv_split(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, int threads_per_row,
                     void* stream) {
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    if (n_in <= 0 || n_out <= 0) return;
    const auto prof = strata::sycl_arch::get_xe2_tuning_profile(q->get_device());
    if (threads_per_row == 32 || threads_per_row == SG_SIZE) {
        const int sgs = THREADS / SG_SIZE;
        const unsigned grid = (unsigned)((n_out + sgs - 1) / sgs);
        if (prof.enable_streaming_hints) {
            q->parallel_for(sycl::nd_range<1>(sycl::range<1>((grid) * (THREADS)), sycl::range<1>(THREADS)),
                            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                                bf16_gemv_subgroup_kernel<SG_SIZE, true>(item, x, w, y, n_in, n_out);
                            });
        } else {
            q->parallel_for(sycl::nd_range<1>(sycl::range<1>((grid) * (THREADS)), sycl::range<1>(THREADS)),
                            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                                bf16_gemv_subgroup_kernel<SG_SIZE, false>(item, x, w, y, n_in, n_out);
                            });
        }
        return;
    }
    if (threads_per_row <= 0 || (threads_per_row & (threads_per_row - 1)) != 0) {
        std::fprintf(stderr,
                     "bf16_gemv_split: threads_per_row %d must be a power of two (32 selects the "
                     "warp-per-row path)\n",
                     threads_per_row);
        std::exit(1);
    }
    const unsigned grid = (unsigned)n_out;
    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> scratch_acc(sycl::range<1>(threads_per_row), cgh);
        if (prof.enable_streaming_hints) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((grid) * (threads_per_row)), sycl::range<1>(threads_per_row)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                                 bf16_gemv_split_kernel<SG_SIZE, true>(item, x, w, y, n_in, n_out, threads_per_row, scratch_acc);
                             });
        } else {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((grid) * (threads_per_row)), sycl::range<1>(threads_per_row)),
                             [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                                 bf16_gemv_split_kernel<SG_SIZE, false>(item, x, w, y, n_in, n_out, threads_per_row, scratch_acc);
                             });
        }
    });
}

}  // namespace strata::kernels
