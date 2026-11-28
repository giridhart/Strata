// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{mmf.cuh,mma.cuh,unary.cu,binbcast.cu}.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <sycl/sycl.hpp>
#include "strata/kernels/native_qsa_score.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
constexpr int D = 128, HEADS = 4, R = 4;

void scalar_score_kernel(const float* __restrict__ pooled, const float* __restrict__ query,
                         const float* __restrict__ bias, const int32_t* __restrict__ step, int max_cells,
                         float* __restrict__ cells, const sycl::nd_item<1>& item) {
    const int n = step[kStepNKv], full = step[kStepNBid];
    if (n < 1 || n > max_cells || step[kStepPos] != n - 1 || full != n / R || step[kStepWidth] != (n < 2051 ? n : 2051))
        return;

    const int row = item.get_group(0);
    if (row > full) return;

    const int head = item.get_local_id(0);
    float h = 0.0f;
    if (head < HEADS) {
        float dot = 0.0f;
        const float* p_row = pooled + (size_t)row * D;
        const float* q_head = query + (size_t)head * D;
#pragma unroll
        for (int d = 0; d < D; ++d) {
            dot = sycl::fma(p_row[d], q_head[d], dot);
        }
        h = dot > 0.0f ? dot : 0.0f;
    }

    auto sg = item.get_sub_group();
    float h0 = sycl::select_from_group(sg, h, 0);
    float h1 = sycl::select_from_group(sg, h, 1);
    float h2 = sycl::select_from_group(sg, h, 2);
    float h3 = sycl::select_from_group(sg, h, 3);

    if (head == 0) {
        float sum = h0 + h1 + h2 + h3;
        if (bias) sum += bias[row];
        if (row == full && n % R) sum += 1e9f;
        for (int i = row * R; i < n && i < (row + 1) * R; ++i) {
            cells[i] = sum;
        }
    }
}

struct Span {
    const void* p;
    size_t n;
};
void validate(Span s) {
    const auto p = reinterpret_cast<uintptr_t>(s.p);
    if (!p || p % 4 || s.n > UINTPTR_MAX - p)
        throw std::invalid_argument("native QSA score requires aligned bounded spans");
}

bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<uintptr_t>(a.p), y = reinterpret_cast<uintptr_t>(b.p);
    return x < y + b.n && y < x + a.n;
}
}  // namespace

void native_qsa_score_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_score_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_qsa_score(const float* pooled, const float* query, const float* bias, const QsaShapes& s,
                      const int32_t* step, int64_t max_blocks, int64_t max_cells, float* cells, void* stream) {
    if (!stream || s.idx_dim != D || s.idx_n_head != HEADS || s.idx_block != R || s.idx_top_k != 2048 ||
        max_cells < 1 || max_cells > INT32_MAX - 3 || max_blocks != max_cells / R + 1)
        throw std::invalid_argument(
            "native QSA score requires 128dim/4heads/4cells/2048budget, exact capacities and explicit stream");
    const Span spans[] = {{pooled, size_t(max_blocks) * D * 4},
                          {query, HEADS * D * 4},
                          {step, kStepCount * 4},
                          {cells, size_t(max_cells) * 4},
                          {bias, bias ? size_t(max_blocks) * 4 : 0}};
    const int count = bias ? 5 : 4;
    for (int i = 0; i < count; ++i)
        validate(spans[i]);
    for (int i = 0; i < count; ++i)
        for (int j = i + 1; j < count; ++j)
            if (overlaps(spans[i], spans[j])) throw std::invalid_argument("native QSA score spans overlap");

    static sycl::queue default_queue{sycl::property::queue::in_order()};
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    q->parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)max_blocks * 32), sycl::range<1>(32)),
                    [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                        scalar_score_kernel(pooled, query, bias, step, int(max_cells), cells, item);
                    });
}

}  // namespace strata::kernels
