// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{dsv4-hc.cu,scale.cu,unary.cu,unary.cuh}.
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
#include "strata/kernels/native_gr_postops.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

static sycl::queue default_queue{sycl::property::queue::in_order()};
constexpr int THREADS = 256;

inline float sigmoid(float x) { return 1.0f / (1.0f + sycl::native::exp(-x)); }

// ggml SCALE uses scale*x+bias, including its +0 bias. Retain this operation
// explicitly so compile-time zero does not change signed-zero behavior.
inline float scale_zero_bias(float x, float scale) { return sycl::fma(scale, x, 0.0f); }

void down_silu(sycl::nd_item<1> item, float* lo, int count, float scale) {
    const std::size_t i = item.get_global_id(0);
    if (i >= std::size_t(count)) return;
    const float x = scale_zero_bias(lo[i], scale);
    lo[i] = x / (1.0f + sycl::native::exp(-x));
}

template <bool Fused>
void pre_gated_multi(sycl::nd_item<2> item, const float* __restrict__ xn, float* __restrict__ gate,
                     float* __restrict__ mixed, int n_embd, int hc, float scale) {
    const std::size_t d = item.get_global_id(1);
    if (d >= std::size_t(n_embd)) return;
    const std::size_t t = item.get_group(0);
    xn += t * std::size_t(n_embd) * hc;
    gate += t * std::size_t(n_embd) * hc;
    mixed += t * std::size_t(n_embd);
    float sum = 0.0f;
    for (int c = 0; c < hc; ++c) {
        const std::size_t i = std::size_t(c) * n_embd + d;
        const float x = xn[i], w = sigmoid(gate[i]);
        const float product = x * w;
        gate[i] = product;
        if constexpr (Fused)
            sum = sycl::fma(x, w, sum);
        else
            sum = c == 0 ? product : sum + product;
    }
    if constexpr (Fused)
        mixed[d] = scale * sum;
    else
        mixed[d] = scale_zero_bias(sum, scale);
}

void post_multi_k(sycl::nd_item<2> item, const float* residual, const float* __restrict__ block_out,
                  const float* __restrict__ inject, float* output, int n_embd, int hc, float scale,
                  long long r_stride, long long b_stride, long long i_stride) {
    const std::size_t i = item.get_global_id(1);
    if (i >= std::size_t(n_embd) * hc) return;
    const long long t = item.get_group(0);
    residual += t * r_stride;
    output += t * r_stride;
    block_out += t * b_stride;
    inject += t * i_stride;
    const int c = int(i / n_embd), d = int(i % n_embd);
    const float weight = scale_zero_bias(sigmoid(scale_zero_bias(inject[c], scale)), 2.0f);
    // Exact residual/output alias is supported; no other thread reads residual[i].
    output[i] = sycl::fma(block_out[d], weight, residual[i]);
}

void check_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % alignof(float))
        throw std::invalid_argument("native GR postops require non-null four-byte aligned pointers");
}

void check_shape(int n, int hc) {
    if (n <= 0 || hc <= 0 || std::uint64_t(n) * hc > std::uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("native GR postops require positive bounded dimensions");
}

unsigned blocks(std::size_t n) { return unsigned((n + THREADS - 1) / THREADS); }

}  // namespace

void native_gr_down_silu(float* lo, int hc_lr, int hc, void* stream) {
    check_shape(hc_lr, hc);
    check_pointer(lo);
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;
    const float float_hc = 1.0f / float(hc);
    const size_t total_threads = (size_t)blocks(hc_lr) * THREADS;
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(total_threads), sycl::range<1>(THREADS)),
                    [=](sycl::nd_item<1> item) { down_silu(item, lo, hc_lr, float_hc); });
    if (!stream) default_queue.wait();
}

void native_gr_pre_gated_multi(const float* xn, float* gate, float* mixed, int n_embd, int hc, int n_tok,
                               bool fused_layer, void* stream) {
    if (n_tok <= 0)
        throw std::invalid_argument("native GR postops require positive n_tok");
    check_shape(n_embd, hc);
    check_pointer(xn);
    check_pointer(gate);
    check_pointer(mixed);
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;
    const float float_hc = 1.0f / float(hc);
    const size_t total_cols = (size_t)blocks(n_embd) * THREADS;
    if (fused_layer) {
        q->parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)n_tok, total_cols), sycl::range<2>(1, THREADS)),
                        [=](sycl::nd_item<2> item) {
                            pre_gated_multi<true>(item, xn, gate, mixed, n_embd, hc, float_hc);
                        });
    } else {
        q->parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)n_tok, total_cols), sycl::range<2>(1, THREADS)),
                        [=](sycl::nd_item<2> item) {
                            pre_gated_multi<false>(item, xn, gate, mixed, n_embd, hc, float_hc);
                        });
    }
    if (!stream) default_queue.wait();
}

void native_gr_pre_gated(const float* xn, float* gate, float* mixed, int n_embd, int hc, bool fused_layer,
                         void* stream) {
    native_gr_pre_gated_multi(xn, gate, mixed, n_embd, hc, 1, fused_layer, stream);
}

void native_gr_post_multi(const float* residual, const float* block_out, const float* inject, float* output,
                          int n_embd, int hc, int n_tok, long long r_stride, long long b_stride, long long i_stride,
                          void* stream) {
    if (n_tok <= 0)
        throw std::invalid_argument("native GR postops require positive n_tok");
    check_shape(n_embd, hc);
    check_pointer(residual);
    check_pointer(block_out);
    check_pointer(inject);
    check_pointer(output);
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;
    const float float_hc = 1.0f / float(hc);
    const size_t total_elements = (size_t)blocks(std::size_t(n_embd) * hc) * THREADS;
    q->parallel_for(
        sycl::nd_range<2>(sycl::range<2>((size_t)n_tok, total_elements), sycl::range<2>(1, THREADS)),
        [=](sycl::nd_item<2> item) {
            post_multi_k(item, residual, block_out, inject, output, n_embd, hc, float_hc,
                         r_stride, b_stride, i_stride);
        });
    if (!stream) default_queue.wait();
}

void native_gr_post_multi(const float* residual, const float* block_out, const float* inject, float* output,
                          int n_embd, int hc, int n_tok, void* stream) {
    const long long hc_dim = (long long)n_embd * hc;
    native_gr_post_multi(residual, block_out, inject, output, n_embd, hc, n_tok, hc_dim, (long long)n_embd,
                         (long long)hc, stream);
}

void native_gr_post(const float* residual, const float* block_out, const float* inject, float* output, int n_embd,
                    int hc, void* stream) {
    native_gr_post_multi(residual, block_out, inject, output, n_embd, hc, 1, stream);
}

}  // namespace strata::kernels
