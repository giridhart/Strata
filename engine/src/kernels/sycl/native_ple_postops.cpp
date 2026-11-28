// Arithmetic adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// src/models/qwen4exp.cpp and ggml-cuda/{reduce_rows.cuh,sumrows.cu,unary.cu}.
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
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/ngram.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
constexpr int N = 2560, H = 4, D = N * H, HISTORY = 9;


void gate_kernel(const float* key, const float* query, float* gate, float scale, float* partials,
                 const sycl::nd_item<1>& item) {
    const int row = item.get_group(0);
    const int tid = item.get_local_id(0);
    const int bsize = item.get_local_range(0);
    auto sg = item.get_sub_group();

    float sum = 0.0f;
    for (int d = tid; d < N; d += bsize) {
        sum += key[row * N + d] * query[row * N + d];
    }
    sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());

    const int lane = sg.get_local_linear_id();
    const int sg_id = sg.get_group_linear_id();
    if (lane == 0) partials[sg_id] = sum;
    sycl::group_barrier(item.get_group());

    if (sg.get_group_linear_id() == 0) {
        const int sg_sz = sg.get_local_range()[0];
        const int num_sgs = bsize / sg_sz;
        float total = 0.0f;
        for (int s = lane; s < num_sgs; s += sg_sz) {
            total += partials[s];
        }
        total = sycl::reduce_over_group(sg, total, sycl::plus<float>());
        if (lane == 0) {
            const float s = sycl::fma(scale, total, 0.0f);  // ggml SCALE's zero bias
            const float mag = sycl::sqrt(sycl::fmax(sycl::fabs(s), 1e-6f));
            const float sign = float((s > 0.0f) - (s < 0.0f));
            gate[row] = 1.0f / (1.0f + sycl::native::exp(-sign * mag));
        }
    }
}

void broadcast_kernel(const float* value, const float* gate, float* gated, const sycl::nd_item<1>& item) {
    const size_t i = item.get_global_id(0);
    if (i < D) gated[i] = value[i % N] * gate[i / N];
}

void conv_residual_kernel(const float* history, const float* normalized, const uint16_t* weights, const float* hidden,
                          const float* gated, float* conv, float* result, const sycl::nd_item<1>& item) {
    const size_t c = item.get_global_id(0);
    if (c >= D) return;
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float x = k == 3 ? normalized[c] : history[c * HISTORY + 3 * k];
        const float w = sycl::vec<sycl::half, 1>(sycl::bit_cast<sycl::half, unsigned short>(weights[c * 4 + k]))
                            .convert<float, sycl::rounding_mode::automatic>()[0];
        const float term = x * w;
        sum = k == 0 ? term : sum + term;
    }
    const float activation = sum / (1.0f + sycl::native::exp(-sum));
    conv[c] = activation;
    // Exact hidden/result alias is safe: each thread owns one element.
    result[c] = hidden[c] + gated[c] + activation;
}


void broadcast_batch_kernel(const float* value, const float* gate, float* gated, int T, const sycl::nd_item<1>& item) {
    const size_t i = item.get_global_id(0);
    if (i >= size_t(T) * D) return;
    const size_t t = i / D, d = i % D;
    gated[i] = value[t * N + d % N] * gate[t * H + d / N];
}

// the dilated conv (taps 9, 6, 3 tokens back and this one) and the residual; a tap before the chunk reads the history
void conv_residual_batch_kernel(const float* history, const float* normalized, const uint16_t* weights, float* hidden,
                                const float* gated, int T, const sycl::nd_item<1>& item) {
    const size_t i = item.get_global_id(0);
    if (i >= size_t(T) * D) return;
    const int t = int(i / D), c = int(i % D);
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int p = t - 9 + 3 * k;  // the token this tap reads (k == 3: this one)
        const float x = p >= 0 ? normalized[size_t(p) * D + c] : history[size_t(c) * HISTORY + (9 + p)];
        const float wk = sycl::vec<sycl::half, 1>(sycl::bit_cast<sycl::half, unsigned short>(weights[c * 4 + k]))
                             .convert<float, sycl::rounding_mode::automatic>()[0];
        const float term = x * wk;
        sum = k == 0 ? term : sum + term;
    }
    const float activation = sum / (1.0f + sycl::native::exp(-sum));
    hidden[i] = hidden[i] + gated[i] + activation;
}

// the history after the chunk: the last nine normalized rows (older ones from the history when T < 9)
void history_batch_kernel(float* history, const float* normalized, int T, const sycl::nd_item<1>& item) {
    const size_t c = item.get_global_id(0);
    if (c >= D) return;
    float h[HISTORY];
#pragma unroll
    for (int r = 0; r < HISTORY; ++r) {
        const int p = T - HISTORY + r;
        h[r] = p >= 0 ? normalized[size_t(p) * D + c] : history[size_t(c) * HISTORY + (T + r)];
    }
#pragma unroll
    for (int r = 0; r < HISTORY; ++r)
        history[size_t(c) * HISTORY + r] = h[r];
}

void history_batch_snap_kernel(float* history, const float* normalized, int T, float* snap,
                               const sycl::nd_item<1>& item) {
    const size_t c = item.get_global_id(0);
    if (c >= D) return;
    float h0[HISTORY];
#pragma unroll
    for (int r = 0; r < HISTORY; ++r)
        h0[r] = history[size_t(c) * HISTORY + r];
    for (int t = 0; t < T; ++t) {
        float* out = snap + size_t(t) * HISTORY * D + size_t(c) * HISTORY;
#pragma unroll
        for (int r = 0; r < HISTORY; ++r) {
            const int p = r + t + 1 - HISTORY;
            out[r] = p >= 0 ? normalized[size_t(p) * D + c] : h0[r + t + 1];
        }
    }
#pragma unroll
    for (int r = 0; r < HISTORY; ++r) {
        const int p = r + T - HISTORY;
        history[size_t(c) * HISTORY + r] = p >= 0 ? normalized[size_t(p) * D + c] : h0[r + T];
    }
}

struct Span {
    const void* p;
    size_t bytes;
    size_t alignment;
};
bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<uintptr_t>(a.p), y = reinterpret_cast<uintptr_t>(b.p);
    return x < y + b.bytes && y < x + a.bytes;
}

void validate(Span span) {
    const auto p = reinterpret_cast<uintptr_t>(span.p);
    if (!p || p % span.alignment || p > std::numeric_limits<uintptr_t>::max() - span.bytes)
        throw std::invalid_argument("native PLE postops require nonnull aligned bounded spans");
}
}  // namespace

void native_ple_postops(const float* projected_key, const float* hidden, const float* value, const float* history,
                        const PleWeights& w, const NativePlePostopsBuffers& b, void* stream) {
    if (!stream) throw std::invalid_argument("native PLE postops require an explicit stream");
    const Span inputs[] = {{projected_key, D * 4, 4},     {hidden, D * 4, 4},          {value, N * 4, 4},
                           {history, HISTORY * D * 4, 4}, {w.norm_key, D * 4, 4},      {w.norm_query, D * 4, 4},
                           {w.norm_conv, D * 4, 4},       {w.conv1d_f16, 4 * D * 2, 2}};
    const Span outputs[] = {{b.key, D * 4, 4},        {b.query, D * 4, 4}, {b.gate, H * 4, 4},  {b.gated, D * 4, 4},
                            {b.normalized, D * 4, 4}, {b.conv, D * 4, 4},  {b.result, D * 4, 4}};
    for (const auto& span : inputs)
        validate(span);
    for (const auto& span : outputs)
        validate(span);
    for (size_t i = 0; i < 7; ++i) {
        for (size_t j = 0; j < 8; ++j)
            if (!(i == 6 && j == 1 && b.result == hidden) && overlaps(outputs[i], inputs[j]))
                throw std::invalid_argument("native PLE postops output overlaps an input or weight");
        for (size_t j = i + 1; j < 7; ++j)
            if (!(i == 1 && j == 4 && b.query == b.normalized) && overlaps(outputs[i], outputs[j]))
                throw std::invalid_argument("native PLE postops writable spans overlap");
    }

    native_gr_rms_norm_weighted(projected_key, w.norm_key, b.key, N, H, NG_RMS_EPS, stream);
    native_gr_rms_norm_weighted(hidden, w.norm_query, b.query, N, H, NG_RMS_EPS, stream);

    static sycl::queue default_queue{sycl::property::queue::in_order()};
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    const float scale = 1.0f / std::sqrt(float(N));

    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> partials_acc(sycl::range<1>(64), cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(H * 512), sycl::range<1>(512)),
                         [=](sycl::nd_item<1> item) {
                             gate_kernel(b.key, b.query, b.gate, scale,
                                         partials_acc.get_multi_ptr<sycl::access::decorated::no>().get(), item);
                         });
    });

    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(D), sycl::range<1>(256)),
                    [=](sycl::nd_item<1> item) { broadcast_kernel(value, b.gate, b.gated, item); });

    native_gr_rms_norm_weighted(b.gated, w.norm_conv, b.normalized, N, H, NG_RMS_EPS, stream);

    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(D), sycl::range<1>(256)), [=](sycl::nd_item<1> item) {
        conv_residual_kernel(history, b.normalized, w.conv1d_f16, hidden, b.gated, b.conv, b.result, item);
    });
}

void native_ple_postops_batch(float* key, float* hidden, const float* value, float* history, const PleWeights& w,
                              float* query_norm, float* gated, float* gate, int T, void* stream) {
    if (!stream || T <= 0 || !key || !hidden || !value || !history || !query_norm || !gated || !gate)
        throw std::invalid_argument("native PLE postops batch: null input or empty batch");

    static sycl::queue default_queue{sycl::property::queue::in_order()};
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    const size_t rows = size_t(T) * H;
    const size_t blocks = ((size_t(T) * D + 255) / 256) * 256;
    const float scale = 1.0f / std::sqrt(float(N));

    native_gr_rms_norm_weighted_multi(key, w.norm_key, key, N, H, T, NG_RMS_EPS, stream);
    native_gr_rms_norm_weighted_multi(hidden, w.norm_query, query_norm, N, H, T, NG_RMS_EPS, stream);

    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> partials_acc(sycl::range<1>(64), cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(rows * 512), sycl::range<1>(512)),
                         [=](sycl::nd_item<1> item) {
                             gate_kernel(key, query_norm, gate, scale,
                                         partials_acc.get_multi_ptr<sycl::access::decorated::no>().get(), item);
                         });
    });

    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks), sycl::range<1>(256)),
                    [=](sycl::nd_item<1> item) { broadcast_batch_kernel(value, gate, gated, T, item); });

    native_gr_rms_norm_weighted_multi(gated, w.norm_conv, query_norm, N, H, T, NG_RMS_EPS, stream);

    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks), sycl::range<1>(256)), [=](sycl::nd_item<1> item) {
        conv_residual_batch_kernel(history, query_norm, w.conv1d_f16, hidden, gated, T, item);
    });

    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(D), sycl::range<1>(256)),
                    [=](sycl::nd_item<1> item) { history_batch_kernel(history, query_norm, T, item); });
}

void native_ple_postops_batch_snap(float* key, float* hidden, const float* value, float* history, const PleWeights& w,
                                   float* query_norm, float* gated, float* gate, int T, float* snap, void* stream) {
    if (!stream || T <= 0 || !key || !hidden || !value || !history || !query_norm || !gated || !gate || !snap)
        throw std::invalid_argument("native PLE postops batch (snapshots): null input or empty batch");
    auto* q = static_cast<sycl::queue*>(stream);

    const unsigned rows = unsigned(T) * H;
    const unsigned blocks = unsigned((size_t(T) * D + 255) / 256) * 256;
    const float scale = 1.0f / std::sqrt(float(N));

    native_gr_rms_norm_weighted_multi(key, w.norm_key, key, N, H, T, NG_RMS_EPS, stream);
    native_gr_rms_norm_weighted_multi(hidden, w.norm_query, query_norm, N, H, T, NG_RMS_EPS, stream);

    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> partials_acc(sycl::range<1>(64), cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(rows * 512), sycl::range<1>(512)),
                         [=](sycl::nd_item<1> item) {
                             gate_kernel(key, query_norm, gate, scale,
                                         partials_acc.get_multi_ptr<sycl::access::decorated::no>().get(), item);
                         });
    });

    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks), sycl::range<1>(256)),
                    [=](sycl::nd_item<1> item) { broadcast_batch_kernel(value, gate, gated, T, item); });

    native_gr_rms_norm_weighted_multi(gated, w.norm_conv, query_norm, N, H, T, NG_RMS_EPS, stream);

    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks), sycl::range<1>(256)), [=](sycl::nd_item<1> item) {
        conv_residual_batch_kernel(history, query_norm, w.conv1d_f16, hidden, gated, T, item);
    });

    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(D), sycl::range<1>(256)),
                    [=](sycl::nd_item<1> item) { history_batch_snap_kernel(history, query_norm, T, snap, item); });
}

}  // namespace strata::kernels
