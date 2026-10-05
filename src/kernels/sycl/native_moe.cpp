// Arithmetic adapted from the MIT-licensed pinned ggml CUDA
// moe-weighted-reduction.cu at 3cf03257f219afbe7334045ff7c6a06ac68c627d.
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
#include "strata/core/sycl_arch.hpp"
#include <sycl/sycl.hpp>
#include "strata/kernels/native_moe.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int SG_SIZE = strata::sycl_arch::SUB_GROUP_SIZE;

inline sycl::queue& get_default_in_order_queue() {
    static sycl::queue q(sycl::property::queue::in_order{});
    return q;
}

std::atomic<bool> enabled{false};

template <int SUB_SIZE = SG_SIZE>
void combine_vec4(sycl::nd_item<2> item, const float* __restrict__ parts, const float* __restrict__ weights,
                  const float* __restrict__ shared, float* __restrict__ output, int64_t n_embd, int k) {
    const int64_t tk = item.get_group(0);
    parts += tk * k * n_embd;
    weights += tk * k;
    if (shared) shared += tk * n_embd;
    output += tk * n_embd;
    const int64_t col4 = (int64_t(item.get_group(1)) * item.get_local_range(1) + item.get_local_id(1)) * 4;
    if (col4 >= n_embd) return;

    float w_reg[16];
#pragma unroll
    for (int e = 0; e < k; ++e)
        w_reg[e] = weights[e];

    sycl::vec<float, 4> sum(0.0f);
#pragma unroll
    for (int expert = 0; expert < k; ++expert) {
        const auto p = *reinterpret_cast<const sycl::vec<float, 4>*>(parts + int64_t(expert) * n_embd + col4);
        sum += p * w_reg[expert];
    }
    if (shared) {
        sum += *reinterpret_cast<const sycl::vec<float, 4>*>(shared + col4);
    }
    *reinterpret_cast<sycl::vec<float, 4>*>(output + col4) = sum;
}

void combine(sycl::nd_item<2> item, const float* __restrict__ parts, const float* __restrict__ weights,
             const float* __restrict__ shared, float* __restrict__ output, int64_t n_embd, int k) {
    const int64_t tk = item.get_group(0);
    parts += tk * k * n_embd;
    weights += tk * k;
    if (shared) shared += tk * n_embd;
    output += tk * n_embd;
    const int64_t col = int64_t(item.get_group(1)) * item.get_local_range(1) + item.get_local_id(1);
    if (col >= n_embd) return;

    float w_reg[16];
#pragma unroll
    for (int e = 0; e < k; ++e)
        w_reg[e] = weights[e];

    float sum = 0.0f;
#pragma unroll
    for (int expert = 0; expert < k; ++expert) {
        sum += parts[int64_t(expert) * n_embd + col] * w_reg[expert];
    }
    if (shared) sum += shared[col];
    output[col] = sum;
}

bool valid_span(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % alignof(float) == 0 && bytes <= UINTPTR_MAX - address;
}

bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}

}  // namespace

void native_moe_combine_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_moe_combine_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_moe_combine(const float* parts, const float* weights, const float* shared, float* output, int64_t n_embd,
                        int64_t k, void* stream) {
    if (!stream || n_embd <= 0 || n_embd > std::numeric_limits<int>::max() || k < 1 || k > 15)
        throw std::invalid_argument("native MoE combine requires a stream, positive width and 1..15 experts");
    const size_t row_bytes = size_t(n_embd) * sizeof(float);
    const size_t part_bytes = row_bytes * size_t(k), weight_bytes = size_t(k) * sizeof(float);
    if (!valid_span(parts, part_bytes) || !valid_span(weights, weight_bytes) || !valid_span(output, row_bytes) ||
        (shared && !valid_span(shared, row_bytes)) || overlap(output, row_bytes, parts, part_bytes) ||
        overlap(output, row_bytes, weights, weight_bytes) || (shared && overlap(output, row_bytes, shared, row_bytes)))
        throw std::invalid_argument("native MoE combine requires aligned spans and disjoint output");

    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &get_default_in_order_queue();
    if (n_embd % 4 == 0 && (reinterpret_cast<uintptr_t>(parts) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(output) % 16 == 0) && (!shared || reinterpret_cast<uintptr_t>(shared) % 16 == 0)) {
        const size_t n_vec = (size_t)n_embd / 4;
        const size_t cols = (n_vec + 255) / 256 * 256;
        q->parallel_for(sycl::nd_range<2>(sycl::range<2>(1, cols), sycl::range<2>(1, 256)),
                        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                            combine_vec4<SG_SIZE>(item, parts, weights, shared, output, n_embd, int(k));
                        });
    } else {
        const size_t cols = ((size_t)n_embd + 255) / 256 * 256;
        q->parallel_for(sycl::nd_range<2>(sycl::range<2>(1, cols), sycl::range<2>(1, 256)),
                        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                            combine(item, parts, weights, shared, output, n_embd, int(k));
                        });
    }
    if (!stream) q->wait();
}

void native_moe_combine_multi(const float* parts, const float* weights, const float* shared, float* output,
                              int64_t n_embd, int64_t k, int n_tok, void* stream) {
    if (!stream || n_embd <= 0 || k < 1 || k > 15 || n_tok < 1)
        throw std::invalid_argument("native MoE combine (multi) requires a stream, width, 1..15 experts, tokens");

    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &get_default_in_order_queue();
    if (n_embd % 4 == 0 && (reinterpret_cast<uintptr_t>(parts) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(output) % 16 == 0) && (!shared || reinterpret_cast<uintptr_t>(shared) % 16 == 0)) {
        const size_t n_vec = (size_t)n_embd / 4;
        const size_t cols = (n_vec + 255) / 256 * 256;
        q->parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)n_tok, cols), sycl::range<2>(1, 256)),
                        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                            combine_vec4<SG_SIZE>(item, parts, weights, shared, output, n_embd, int(k));
                        });
    } else {
        const size_t cols = ((size_t)n_embd + 255) / 256 * 256;
        q->parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)n_tok, cols), sycl::range<2>(1, 256)),
                        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                            combine(item, parts, weights, shared, output, n_embd, int(k));
                        });
    }
    if (!stream) q->wait();
}

}  // namespace strata::kernels
