// Arithmetic adapted from gated_delta_net.cu at pinned llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d. Only state addressing differs:
// each warp owns one Strata (head,column) and retains its four rows in registers.
// Compile with --use_fast_math to match the pinned CUDA implementation.
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
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/native_gdn.hpp"

#include <sycl/sycl.hpp>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
constexpr int S = 128;
constexpr int SG_SIZE = strata::sycl_arch::SUB_GROUP_SIZE;

static sycl::queue& default_queue() {
    static sycl::queue q{sycl::property::queue::in_order()};
    return q;
}

template <int SUB_SIZE = SG_SIZE>
void step(sycl::nd_item<2> item, float* __restrict__ state, const float* __restrict__ q, const float* __restrict__ k,
          const float* __restrict__ v, const float* __restrict__ gate, const float* __restrict__ beta,
          float* __restrict__ output, int h_k, int h_v, float scale) {
    constexpr int R_COUNT = S / SUB_SIZE;
    const int head = item.get_group(0);
    const int lane = item.get_local_id(1);
    const int col = item.get_group(1) * 4 + item.get_local_id(0);
    const int q_head = head % h_k;
    float s_shard[R_COUNT], k_reg[R_COUNT], q_reg[R_COUNT];
#pragma unroll
    for (int r = 0; r < R_COUNT; ++r) {
        const int i = r * SUB_SIZE + lane;
        s_shard[r] = state[(size_t(i) * h_v + head) * S + col];
        k_reg[r] = k[q_head * S + i];
        q_reg[r] = q[q_head * S + i];
    }
    const float g_val = sycl::native::exp(gate[head]);
    float kv_shard = 0.0f;
#pragma unroll
    for (int r = 0; r < R_COUNT; ++r)
        kv_shard += s_shard[r] * k_reg[r];
    auto sg = item.get_sub_group();
    const float kv_col = sycl::reduce_over_group(sg, kv_shard, sycl::plus<float>());
    const float delta_col = (v[head * S + col] - g_val * kv_col) * beta[head];
    float attn_partial = 0.0f;
#pragma unroll
    for (int r = 0; r < R_COUNT; ++r) {
        s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
        attn_partial += s_shard[r] * q_reg[r];
    }
    const float attn_col = sycl::reduce_over_group(sg, attn_partial, sycl::plus<float>());
    if (lane == 0) output[head * S + col] = attn_col * scale;
#pragma unroll
    for (int r = 0; r < R_COUNT; ++r) {
        const int i = r * SUB_SIZE + lane;
        state[(size_t(i) * h_v + head) * S + col] = s_shard[r];
    }
}

bool valid_span(const void* pointer, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    return pointer && address % sizeof(float) == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}  // namespace

void native_gdn_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_gdn_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_gdn_step(float* state, const float* q, const float* k, const float* v, const float* gate, const float* beta,
                     float* output, const GdnShapes& shape, void* stream) {
    if (shape.S != S || shape.h_k <= 0 || shape.h_v <= 0 || shape.h_v > 65535 || shape.h_v % shape.h_k != 0)
        throw std::invalid_argument("native GDN requires S=128 and positive divisible head counts <=65535");
    const size_t state_bytes = size_t(S) * S * size_t(shape.h_v) * sizeof(float);
    const size_t qk_bytes = size_t(S) * size_t(shape.h_k) * sizeof(float);
    const size_t output_bytes = size_t(S) * size_t(shape.h_v) * sizeof(float);
    const size_t head_bytes = size_t(shape.h_v) * sizeof(float);
    if (!valid_span(state, state_bytes) || !valid_span(output, output_bytes) ||
        overlap(state, state_bytes, output, output_bytes))
        throw std::invalid_argument("native GDN requires aligned, disjoint state and output spans");
    const void* inputs[] = {q, k, v, gate, beta};
    const size_t bytes[] = {qk_bytes, qk_bytes, output_bytes, head_bytes, head_bytes};
    for (int i = 0; i < 5; ++i) {
        if (!valid_span(inputs[i], bytes[i]) || overlap(state, state_bytes, inputs[i], bytes[i]) ||
            overlap(output, output_bytes, inputs[i], bytes[i]))
            throw std::invalid_argument("native GDN requires aligned input spans disjoint from state and output");
    }
    const float scale = 1.0f / sqrtf(float(S));
    sycl::queue* q_ptr = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q_ptr->parallel_for(sycl::nd_range<2>(sycl::range<2>(shape.h_v * 4, (S / 4) * SG_SIZE), sycl::range<2>(4, SG_SIZE)),
                        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                            step<SG_SIZE>(item, state, q, k, v, gate, beta, output, int(shape.h_k), int(shape.h_v),
                                          scale);
                        });
}
}  // namespace strata::kernels
