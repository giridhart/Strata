// Numerical contract: pinned ggml/src/ggml-cuda/rope.cu, rope_multi/rope_yarn.
// Text positions are equal across the four IMRoPE sections, so section routing
// reduces to the one position associated with each contiguous row.
// MIT License
//
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
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/mrope.hpp"
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
bool overlaps(const void* a, size_t an, const void* b, size_t bn) {
    auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x < an : x - y < bn;
}

// TAB (#280, STRATA_ROPE_TABLE=1): the angles from the session's float64 table.  The host launches <false> whenever
// no table applies - the default - so the default kernel is 0.1.31's code exactly (the table read is not in it;
// with it merely skipped at run time, the compiled default path changed its results).
template <bool TAB>
void apply(const float* x, float* out, int rows, int width, int n_rot, float theta_scale, float freq_scale,
           float corr_low, float corr_high, float ext_factor, float mscale, const int* positions, const int32_t* mtab,
           RopeTab rt, const sycl::nd_item<2>& item) {
    const int row = item.get_global_id(0);
    const int pair = item.get_global_id(1);
    if (row >= rows || pair >= width / 2) return;
    const size_t start = size_t(row) * width;
    if (pair >= n_rot / 2) {
        if (x != out) {
            out[start + 2 * pair] = x[start + 2 * pair];
            out[start + 2 * pair + 1] = x[start + 2 * pair + 1];
        }
        return;
    }
    float c, s;
    if (!(TAB && rope_tab_cs(rt, mrope_pos(mtab, positions[row], pair), pair, c, s))) {
        const float theta_extrap = mrope_pos(mtab, positions[row], pair) * sycl::pow(theta_scale, float(pair));
        rope_scaled_angle(theta_extrap, freq_scale, corr_low, corr_high, ext_factor, mscale, pair, c, s);
    }
    const float a = x[start + pair], b = x[start + pair + n_rot / 2];
    out[start + pair] = a * c - b * s;
    out[start + pair + n_rot / 2] = a * s + b * c;
}
}  // namespace

// one table per device (a layer split runs the rope kernels on several): set and read for the current device
namespace {
constexpr int kMropeDevices = 64;
std::atomic<const int32_t*> mrope_tab[kMropeDevices] = {};
inline int mrope_dev() { return 0; }
}  // namespace

void mrope_table_set(const int32_t* device_table) {
    mrope_tab[mrope_dev()].store(device_table, std::memory_order_relaxed);
}
const int32_t* mrope_table() { return mrope_tab[mrope_dev()].load(std::memory_order_relaxed); }

namespace {
// #280: the float64 angle table of each device, set at session init (before any graph is captured)
struct RopeReg {
    RopeTab tab;
    RopeScaling scaling;
};
RopeReg rope_tab[kMropeDevices] = {};
bool same_scaling(const RopeScaling& a, const RopeScaling& b) {
    return a.type == b.type && a.freq_base == b.freq_base && a.factor == b.factor &&
           a.freq_scale_in == b.freq_scale_in && a.orig_ctx == b.orig_ctx && a.ext_factor == b.ext_factor &&
           a.attn_factor == b.attn_factor && a.beta_fast == b.beta_fast && a.beta_slow == b.beta_slow;
}
// opt-in: STRATA_ROPE_TABLE=1 (the table's angles differ from the fast-math ones in the last bits, so outputs move)
bool rope_table_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_ROPE_TABLE");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}
}  // namespace

void rope_table_set(const float* cos_tab, const float* sin_tab, int max_pos, const RopeScaling& scaling) {
    rope_tab[mrope_dev()] = RopeReg{RopeTab{cos_tab, sin_tab, max_pos}, scaling};
}
void rope_table_release(const float* cos_tab) {
    // every device's entry that points at it (the caller may free it from another current device)
    for (RopeReg& r : rope_tab)
        if (cos_tab != nullptr && r.tab.cos == cos_tab) r = RopeReg{};
}
RopeTab rope_table_for(const RopeScaling& scaling) {
    if (!rope_table_enabled()) return {};
    const RopeReg& r = rope_tab[mrope_dev()];
    return r.tab.cos != nullptr && same_scaling(r.scaling, scaling) ? r.tab : RopeTab{};
}

namespace {
inline float warp_sum(sycl::sub_group sg, float value) {
    return sycl::reduce_over_group(sg, value, sycl::plus<float>());
}

template <bool TAB>
void norm_rope_kernel(const float* x, int in_stride, const float* __restrict__ gamma, float* out,
                      int n_cols, int n_rot, float epsilon, float theta_scale,
                      float freq_scale, float corr_low, float corr_high,
                      float ext_factor, float mscale, const int* positions,
                      const int32_t* mtab, RopeTab rt,
                      float* sums, sycl::nd_item<2> item) {
    auto sg = item.get_sub_group();
    const int tid = item.get_local_id(1);
    const int row = item.get_group(0);
    const float* in_row = x + size_t(row) * in_stride;
    float* out_row = out + size_t(row) * n_cols;

    float partial = 0.0f;
    for (std::size_t col = tid; col < std::size_t(n_cols); col += 256) {
        const float xv = in_row[col];
        partial += xv * xv;
    }

    partial = warp_sum(sg, partial);
    const int lane = tid % 32;
    if (lane == 0) sums[tid / 32] = partial;

    item.barrier(sycl::access::fence_space::local_space);
    partial = lane < 256 / 32 ? sums[lane] : 0.0f;
    partial = warp_sum(sg, partial);
    const float mean = partial / n_cols;
    const float scale = sycl::rsqrt(mean + epsilon);

    const int half_rot = n_rot / 2;
    if (tid >= n_rot && tid < n_cols) {
        out_row[tid] = scale * in_row[tid] * gamma[tid];
    } else if (tid < half_rot) {
        const int pair = tid;
        volatile float a_v = scale * in_row[pair] * gamma[pair];
        volatile float b_v = scale * in_row[pair + half_rot] * gamma[pair + half_rot];
        const float a = a_v;
        const float b = b_v;
        float c, s;
        if (!(TAB && rope_tab_cs(rt, mrope_pos(mtab, positions[row], pair), pair, c, s))) {
            const float theta_extrap = mrope_pos(mtab, positions[row], pair) * sycl::pow(theta_scale, float(pair));
            rope_scaled_angle(theta_extrap, freq_scale, corr_low, corr_high, ext_factor, mscale, pair, c, s);
        }
        out_row[pair] = a * c - b * s;
        out_row[pair + half_rot] = a * s + b * c;
    }
}
}  // namespace

void native_rope_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_rope_enabled() { return enabled.load(std::memory_order_relaxed); }

bool native_norm_rope_usable(int head_dim, int n_rot) {
    static const bool on = [] {
        const char* off = std::getenv("STRATA_NO_NORM_ROPE");
        if (off != nullptr && off[0] != '\0' && off[0] != '0') return false;
        return true;
    }();
    return on && (head_dim == 128 || head_dim == 256) && n_rot == 64;
}

void native_rope_apply(const float* x, float* out, int rows, int head_dim, int n_rot, const RopeScaling& scaling,
                       const int* positions, void* stream) {
    if (!x || !out || !positions || !stream || rows < 1 || rows > 65535 || (head_dim != 128 && head_dim != 256) ||
        n_rot != 64 || rope_scaling_invalid(scaling) != nullptr || reinterpret_cast<uintptr_t>(x) % 4 ||
        reinterpret_cast<uintptr_t>(out) % 4 || reinterpret_cast<uintptr_t>(positions) % 4) {
        throw std::invalid_argument(
            "native RoPE requires aligned F32 rows, width 128/256, rotation 64, valid base/scaling and explicit "
            "stream");
    }
    const size_t bytes = size_t(rows) * head_dim * sizeof(float);
    if ((x != out && overlaps(x, bytes, out, bytes)) || overlaps(positions, size_t(rows) * sizeof(int), out, bytes) ||
        overlaps(positions, size_t(rows) * sizeof(int), x, bytes)) {
        throw std::invalid_argument("native RoPE buffers partially overlap");
    }
    // Match pinned host-side float powf before device fast powf/trigonometry.
    const float theta_scale = powf((float)scaling.freq_base, -2.0f / n_rot);
    const RopeKernelArgs k = scaling.kernel_args(n_rot);  // none: the identity constants
    const RopeTab rt = rope_table_for(scaling);

    const size_t threads = (x == out && n_rot == 64 && native_norm_rope_usable(head_dim, n_rot)) ? 32 : 128;
    const size_t pair_blocks = threads == 32 ? 1 : (head_dim / 2 + 127) / 128;
    const sycl::nd_range<2> ndr(sycl::range<2>(rows, pair_blocks * threads), sycl::range<2>(1, threads));

    static sycl::queue default_queue{sycl::property::queue::in_order()};
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

    const int32_t* mtab = mrope_table();
    if (rt.cos != nullptr) {
        q->parallel_for(ndr, [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
            apply<true>(x, out, rows, head_dim, n_rot, theta_scale, k.freq_scale, k.corr_low, k.corr_high, k.ext_factor,
                        k.attn_factor, positions, mtab, rt, item);
        });
    } else {
        q->parallel_for(ndr, [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
            apply<false>(x, out, rows, head_dim, n_rot, theta_scale, k.freq_scale, k.corr_low, k.corr_high,
                         k.ext_factor, k.attn_factor, positions, mtab, rt, item);
        });
    }
}

void native_qsa_rms_norm_rope(const float* x, int in_stride, const float* gamma, float* out,
                              int rows, int head_dim, int n_rot, float epsilon,
                              const RopeScaling& scaling, const int* positions, void* stream) {
    if (!x || !gamma || !out || !positions || !stream || rows < 1 || rows > 65535 ||
        (head_dim != 128 && head_dim != 256) || in_stride < head_dim || n_rot != 64 ||
        !std::isfinite(epsilon) || epsilon < 0.0f ||
        rope_scaling_invalid(scaling) != nullptr ||
        reinterpret_cast<uintptr_t>(x) % 4 || reinterpret_cast<uintptr_t>(gamma) % 4 ||
        reinterpret_cast<uintptr_t>(out) % 4 || reinterpret_cast<uintptr_t>(positions) % 4) {
        throw std::invalid_argument("native_qsa_rms_norm_rope: invalid arguments");
    }
    const float theta_scale = powf((float)scaling.freq_base, -2.0f / n_rot);
    const RopeKernelArgs k = scaling.kernel_args(n_rot);
    const RopeTab rt = rope_table_for(scaling);

    static sycl::queue default_queue{sycl::property::queue::in_order()};
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;
    const int32_t* mtab = mrope_table();

    if (rt.cos != nullptr) {
        q->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> sums(sycl::range<1>(32), cgh);
            cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(rows, 256), sycl::range<2>(1, 256)),
                             [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
                                 norm_rope_kernel<true>(x, in_stride, gamma, out, head_dim, n_rot, epsilon,
                                                        theta_scale, k.freq_scale, k.corr_low, k.corr_high,
                                                        k.ext_factor, k.attn_factor, positions, mtab, rt,
                                                        sums.template get_multi_ptr<sycl::access::decorated::no>().get(),
                                                        item);
                             });
        });
    } else {
        q->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> sums(sycl::range<1>(32), cgh);
            cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(rows, 256), sycl::range<2>(1, 256)),
                             [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
                                 norm_rope_kernel<false>(x, in_stride, gamma, out, head_dim, n_rot, epsilon,
                                                         theta_scale, k.freq_scale, k.corr_low, k.corr_high,
                                                         k.ext_factor, k.attn_factor, positions, mtab, rt,
                                                         sums.template get_multi_ptr<sycl::access::decorated::no>().get(),
                                                         item);
                             });
        });
    }
}
}  // namespace strata::kernels
