// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{norm.cu,common.cuh,unary.cu}.
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
#include "strata/kernels/native_qsa.hpp"
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};

inline float warp_sum(sycl::sub_group sg, float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += sycl::permute_group_by_xor(sg, value, offset);
    return value;
}

template<int BlockSize>
void norm(const float* input, const float* __restrict__ gamma, float* output,
          int n_cols, float epsilon, float *sums, sycl::nd_item<2> item) {
    auto sg = item.get_sub_group();
    const int tid = item.get_local_id(1);
    const std::size_t row_offset = std::size_t(item.get_group(0)) * n_cols;
    input += row_offset; output += row_offset;
    float partial = 0.0f;
    for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize) {
        const float x = input[col];
        partial += x * x;
    }

    partial = warp_sum(sg, partial);
    const int lane = tid % 32;
    if (lane == 0) sums[tid / 32] = partial;

    item.barrier(sycl::access::fence_space::local_space);
    partial = lane < BlockSize / 32 ? sums[lane] : 0.0f;
    partial = warp_sum(sg, partial);
    const float mean = partial / n_cols;
    const float scale = sycl::rsqrt(mean + epsilon);
    for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize)
        output[col] = scale * input[col] * gamma[col];
}

void gate(const float* attn, const float* __restrict__ q_full, float* output,
          int n_head, int head_dim, sycl::id<1> idx) {
    (void)n_head;
    const std::size_t i = idx[0];
    const std::size_t head = i / head_dim, channel = i % head_dim;
    const float raw = q_full[head * 2 * head_dim + head_dim + channel];
    const float sigmoid = 1.0f / (1.0f + sycl::native::exp(-raw));
    output[i] = attn[i] * sigmoid;
}

std::size_t elements(int cols, int rows) {
    if (cols <= 0 || rows <= 0 ||
        std::uint64_t(cols) * rows >
            std::uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("native QSA requires positive bounded dimensions");
    return std::size_t(cols) * rows;
}

bool valid(const void* ptr, std::size_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(ptr);
    return ptr && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}

bool overlap(const void* a, std::size_t an, const void* b, std::size_t bn) {
    const auto ap = reinterpret_cast<std::uintptr_t>(a), bp = reinterpret_cast<std::uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}

void buffers(const float* input, std::size_t in_bytes, const float* weight, std::size_t weight_bytes,
             float* output, void* stream) {
    if (!stream || !valid(input, in_bytes) || !valid(weight, weight_bytes) || !valid(output, in_bytes) ||
        overlap(input, in_bytes, weight, weight_bytes) || overlap(output, in_bytes, weight, weight_bytes) ||
        (input != output && overlap(input, in_bytes, output, in_bytes)))
        throw std::invalid_argument("native QSA requires a stream, aligned spans, and disjoint buffers or exact input/output alias");
}

} // namespace

void native_qsa_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_qsa_rms_norm_weighted(const float* input, const float* gamma, float* output,
                                  int n_cols, int n_rows, float epsilon, void* stream) {
    const auto count = elements(n_cols, n_rows);
    if (!std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native QSA requires finite nonnegative epsilon");
    buffers(input, count * 4, gamma, std::size_t(n_cols) * 4, output, stream);

    static sycl::queue default_queue{sycl::property::queue::in_order()};
    sycl::queue* q_ptr = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    if (n_cols < 1024) {
        q_ptr->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> sums_acc(sycl::range<1>(32), cgh);

            cgh.parallel_for(
                sycl::nd_range<2>(sycl::range<2>(n_rows, 256),
                                  sycl::range<2>(1, 256)),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
                    norm<256>(
                        input, gamma, output, n_cols, epsilon,
                        sums_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                        item);
                });
        });
    } else {
        q_ptr->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> sums_acc(sycl::range<1>(32), cgh);

            cgh.parallel_for(
                sycl::nd_range<2>(sycl::range<2>(n_rows, 1024),
                                  sycl::range<2>(1, 1024)),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
                    norm<1024>(
                        input, gamma, output, n_cols, epsilon,
                        sums_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                        item);
                });
        });
    }
}

void native_qsa_gate_apply(const float* attn, const float* q_full, float* output,
                           int n_head, int head_dim, void* stream) {
    const auto count = elements(head_dim, n_head);
    buffers(attn, count * 4, q_full, count * 8, output, stream);

    static sycl::queue default_queue{sycl::property::queue::in_order()};
    sycl::queue* q_ptr = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    q_ptr->parallel_for(
        sycl::range<1>(count),
        [=](sycl::id<1> idx) {
            gate(attn, q_full, output, n_head, head_dim, idx);
        });
}

} // namespace strata::kernels
