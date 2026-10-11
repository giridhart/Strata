// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{norm.cu,common.cuh,unary.cu,unary.cuh,ssm-conv.cu,scale.cu}.
// Compile with --use_fast_math, as the pinned backend does.
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
#include "strata/kernels/native_gdn_preprocess.hpp"

#include <sycl/sycl.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>

namespace strata::kernels {
namespace {
constexpr int S = 128;

static sycl::queue& default_queue() {
    static sycl::queue q{sycl::property::queue::in_order()};
    return q;
}

inline float warp_sum(sycl::sub_group sg, float value) {
    return sycl::reduce_over_group(sg, value, sycl::plus<float>());
}

inline float norm_sum(sycl::sub_group sg, sycl::group<1> grp, float value, sycl::local_accessor<float, 1> sums,
                      int local_id) {
    const int lane = local_id % 32;
    value = warp_sum(sg, value);
    if (lane == 0) sums[local_id / 32] = value;
    sycl::group_barrier(grp);
    value = lane < 8 ? sums[lane] : 0.0f;
    return warp_sum(sg, value);
}

inline float sigmoid(float value) { return 1.0f / (1.0f + sycl::native::exp(-value)); }

void conv_silu(sycl::nd_item<1> item, float* __restrict__ history, const float* __restrict__ input,
               const float* __restrict__ weights, float* __restrict__ raw_output, float* __restrict__ silu_output,
               int channels) {
    const int c = item.get_global_id(0);
    if (c >= channels) return;
    float values[4] = {history[c * 3], history[c * 3 + 1], history[c * 3 + 2], input[c]};
    float sum = 0.0f;
#pragma unroll
    for (int tap = 0; tap < 4; ++tap)
        sum += values[tap] * weights[c * 4 + tap];
    raw_output[c] = sum;
    silu_output[c] = sum / (1.0f + sycl::native::exp(-sum));
#pragma unroll
    for (int tap = 0; tap < 3; ++tap)
        history[c * 3 + tap] = values[tap + 1];
}

void l2_norm(sycl::nd_item<1> item, float* input, float epsilon, float scale_after,
             sycl::local_accessor<float, 1> sums) {
    const int col = item.get_local_id(0);
    input += (size_t)item.get_group(0) * S;
    const float value = col < S ? input[col] : 0.0f;
    float partial = 0.0f;
    if (col < S) partial += value * value;

    auto sg = item.get_sub_group();
    partial = norm_sum(sg, item.get_group(), partial, sums, col);
    const float scale = sycl::rsqrt(partial / S + epsilon);
    if (col < S) {
        const float normalized = scale * value;
        input[col] = sycl::fma(normalized, scale_after, 0.0f);
    }
}

void beta_sigmoid(sycl::nd_item<1> item, float* beta, int count) {
    const int i = item.get_global_id(0);
    if (i < count) beta[i] = sigmoid(beta[i]);
}

void gate_softplus(sycl::nd_item<1> item, const float* __restrict__ alpha, const float* __restrict__ dt,
                   const float* __restrict__ ssm_a, float* __restrict__ gate, int count) {
    const int i = item.get_global_id(0);
    if (i >= count) return;
    const float value = alpha[i] + dt[i];
    const float softplus = value > 20.0f ? value : sycl::log1p(sycl::native::exp(value));
    gate[i] = softplus * ssm_a[i];
}

void out_norm(sycl::nd_item<1> item, const float* __restrict__ input, const float* __restrict__ z,
              const float* __restrict__ gamma, float* __restrict__ output, float epsilon,
              sycl::local_accessor<float, 1> sums) {
    const int col = item.get_local_id(0);
    const size_t offset = (size_t)item.get_group(0) * S;
    const float value = col < S ? input[offset + col] : 0.0f;
    float partial = 0.0f;
    if (col < S) partial += value * value;

    auto sg = item.get_sub_group();
    partial = norm_sum(sg, item.get_group(), partial, sums, col);
    const float scale = sycl::rsqrt(partial / S + epsilon);
    if (col < S) {
        const float weighted = scale * value * gamma[col];
        output[offset + col] = weighted * sigmoid(z[offset + col]);
    }
}

struct Span {
    const void* pointer;
    size_t bytes;
};
void valid(Span span) {
    const auto address = reinterpret_cast<uintptr_t>(span.pointer);
    if (!span.pointer || address % sizeof(float) || span.bytes > UINTPTR_MAX - address)
        throw std::invalid_argument("native GDN preprocessing requires aligned nonnull valid spans");
}
void disjoint(Span a, Span b) {
    const auto ap = reinterpret_cast<uintptr_t>(a.pointer), bp = reinterpret_cast<uintptr_t>(b.pointer);
    if (ap < bp + b.bytes && bp < ap + a.bytes)
        throw std::invalid_argument("native GDN preprocessing requires disjoint writable spans");
}
void count_check(int64_t count) {
    if (count <= 0 || count > 65535)
        throw std::invalid_argument("native GDN preprocessing requires count in [1,65535]");
}
void norm_geometry(int64_t rows, int64_t cols, float epsilon) {
    count_check(rows);
    if (cols != S || !std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native GDN norm requires width 128 and finite nonnegative epsilon");
}
}  // namespace

void native_gdn_conv_silu(float* history, const float* input, const float* weights, float* raw_output,
                          float* silu_output, int64_t channels, int64_t d_conv, void* stream) {
    count_check(channels);
    if (d_conv != 4) throw std::invalid_argument("native GDN convolution requires four taps");
    const size_t bytes = size_t(channels) * sizeof(float);
    const Span writable[] = {{history, 3 * bytes}, {raw_output, bytes}, {silu_output, bytes}};
    const Span inputs[] = {{input, bytes}, {weights, 4 * bytes}};
    for (auto span : writable)
        valid(span);
    for (auto span : inputs)
        valid(span);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < i; ++j)
            disjoint(writable[i], writable[j]);
        for (auto span : inputs)
            disjoint(writable[i], span);
    }
    sycl::queue* q_ptr = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q_ptr->parallel_for(sycl::nd_range<1>(sycl::range<1>(((size_t)channels + 255) / 256 * 256), sycl::range<1>(256)),
                        [=](sycl::nd_item<1> item) {
                            conv_silu(item, history, input, weights, raw_output, silu_output, int(channels));
                        });
}

void native_gdn_l2_norm(float* input, int64_t rows, int64_t cols, float epsilon, void* stream) {
    norm_geometry(rows, cols, epsilon);
    valid({input, size_t(rows) * S * sizeof(float)});
    sycl::queue* q_ptr = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q_ptr->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> sums(sycl::range<1>(32), cgh);
        const float eps_S = epsilon / S;
        const float scale_after = 1.0f / sqrtf(float(S));
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)rows * 256), sycl::range<1>(256)),
                         [=](sycl::nd_item<1> item)
                             [[sycl::reqd_sub_group_size(32)]] { l2_norm(item, input, eps_S, scale_after, sums); });
    });
}

void native_gdn_beta_gate(float* beta, int64_t heads, void* stream) {
    count_check(heads);
    valid({beta, size_t(heads) * sizeof(float)});
    sycl::queue* q_ptr = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q_ptr->parallel_for(sycl::nd_range<1>(sycl::range<1>(((size_t)heads + 255) / 256 * 256), sycl::range<1>(256)),
                        [=](sycl::nd_item<1> item) { beta_sigmoid(item, beta, int(heads)); });
}

void native_gdn_gate(const float* alpha, const float* dt, const float* ssm_a, float* gate, int64_t heads,
                     void* stream) {
    count_check(heads);
    const size_t bytes = size_t(heads) * sizeof(float);
    const Span output{gate, bytes};
    valid(output);
    for (auto input : {Span{alpha, bytes}, Span{dt, bytes}, Span{ssm_a, bytes}}) {
        valid(input);
        disjoint(output, input);
    }
    sycl::queue* q_ptr = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q_ptr->parallel_for(sycl::nd_range<1>(sycl::range<1>(((size_t)heads + 255) / 256 * 256), sycl::range<1>(256)),
                        [=](sycl::nd_item<1> item) { gate_softplus(item, alpha, dt, ssm_a, gate, int(heads)); });
}

void native_gdn_out_norm(const float* output, const float* z, const float* gamma, float* destination, int64_t heads,
                         int64_t cols, float epsilon, void* stream) {
    norm_geometry(heads, cols, epsilon);
    const size_t bytes = size_t(heads) * S * sizeof(float);
    const Span writable{destination, bytes};
    valid(writable);
    for (auto input : {Span{output, bytes}, Span{z, bytes}, Span{gamma, S * sizeof(float)}}) {
        valid(input);
        disjoint(writable, input);
    }
    sycl::queue* q_ptr = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q_ptr->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> sums(sycl::range<1>(32), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)heads * 256), sycl::range<1>(256)),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                             out_norm(item, output, z, gamma, destination, epsilon, sums);
                         });
    });
}

}  // namespace strata::kernels
