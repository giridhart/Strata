// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{norm.cu,common.cuh}. Scope: contiguous weighted F32 RMSNorm.
// Optimized for Intel Xe2 microarchitecture (Battlemage B580 / Lunar Lake) native SIMD16 execution.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <sycl/sycl.hpp>
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/native_gr_norm.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

static sycl::queue default_queue{sycl::property::queue::in_order()};
constexpr int SG_SZ = strata::sycl_arch::SUB_GROUP_SIZE;

template<int BlockSize>
void weighted_rms_norm(sycl::nd_item<1> item,
                       const float* __restrict__ input,
                       const float* __restrict__ gamma,
                       float* __restrict__ output, int n_cols, float epsilon,
                       float *sums) {
    auto sg = item.get_sub_group();
    const int tid = item.get_local_id(0);
    const std::size_t row_offset = std::size_t(item.get_group(0)) * n_cols;
    input += row_offset;
    gamma += row_offset;
    output += row_offset;
    float partial = 0.0f;
    for (int col = tid; col < n_cols; col += BlockSize) {
        const float value = input[col];
        partial += value * value;
    }

    // Native hardware subgroup reduction
    partial = sycl::reduce_over_group(sg, partial, sycl::plus<float>());
    const int lane = sg.get_local_linear_id();
    const int sg_id = sg.get_group_linear_id();
    if (lane == 0) sums[sg_id] = partial;
    sycl::group_barrier(item.get_group());

    partial = 0.0f;
    constexpr int num_sgs = BlockSize / SG_SZ;
    for (int s = lane; s < num_sgs; s += SG_SZ) {
        partial += sums[s];
    }
    partial = sycl::reduce_over_group(sg, partial, sycl::plus<float>());

    const float mean = partial / n_cols;
    const float scale = sycl::rsqrt(mean + epsilon);
    for (int col = tid; col < n_cols; col += BlockSize) {
        output[col] = scale * input[col] * gamma[col];
    }
}

void check_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % alignof(float) != 0)
        throw std::invalid_argument("native GR RMSNorm requires non-null four-byte aligned pointers");
}

} // namespace

void native_gr_rms_norm_weighted(const float* input, const float* gamma, float* output,
                                 int n_cols, int n_rows, float epsilon, void* stream) {
    if (n_cols <= 0 || n_rows <= 0 || !std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native GR RMSNorm requires positive dimensions and finite nonnegative epsilon");
    check_pointer(input);
    check_pointer(gamma);
    check_pointer(output);
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;
    if (n_cols < 1024) {
        q->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> sums(sycl::range<1>(256 / SG_SZ), cgh);
            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>((size_t)n_rows * 256), sycl::range<1>(256)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SZ)]] {
                    weighted_rms_norm<256>(
                        item, input, gamma, output, n_cols, epsilon,
                        sums.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    } else {
        q->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> sums(sycl::range<1>(1024 / SG_SZ), cgh);
            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>((size_t)n_rows * 1024), sycl::range<1>(1024)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SZ)]] {
                    weighted_rms_norm<1024>(
                        item, input, gamma, output, n_cols, epsilon,
                        sums.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    }
    if (!stream) default_queue.wait();
}

} // namespace strata::kernels
