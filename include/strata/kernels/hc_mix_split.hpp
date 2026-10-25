// include/strata/kernels/hc_mix_split.hpp - Fused Hyper-Connection Mix + Sinkhorn Split kernel for Intel Xe2.
//
// Ports the DeepSeek-V4.1 fused hyper-connection mix + split optimization (src/dsv41/ds4_kernels.cu lines 343-380)
// to pure SYCL 2020, engineered for Intel Xe2 SIMD16 execution units (Battlemage Arc B580 / Lunar Lake Xe2-LPG).
//
// Background:
//   In hyper-connection transformer architectures (DeepSeek-V4.1 / Qwen MoE), each layer computes:
//     1. hc_mix: evaluates mix projections for T tokens across n_mix streams:
//          mixes[t, m] = (h[t] . fn[m]) * rsqrt(mean(h[t]^2) + eps)
//     2. hc_split: normalizes and balances mixes via Sinkhorn iterations into pre-, post-, and combination routing matrices.
//
// Fusion Optimization:
//   Standard execution launches two separate kernels (hc_mix then hc_split), paying kernel launch latency (4-6 us per launch),
//   an intermediate global memory roundtrip, and stream synchronization overhead.
//   hc_mix_split fuses both operations into a single kernel dispatch:
//     - Dispatches grid (n_mix, T) workgroups.
//     - Each workgroup computes one mix coefficient and RMS norm scaling for token t.
//     - Workgroup leader uses a device-scoped atomic fetch-add on an atomic counter (cnt[t]).
//     - The final workgroup to finish token t (prev == n_mix - 1) directly invokes hc_split in-place on sub-group 0!
//     - Replaces 2 kernel launches per layer with 1, eliminating launch gaps and intermediate stalls.
#pragma once

#include <sycl/sycl.hpp>
#include "strata/core/sycl_arch.hpp"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

constexpr int kHcMixThreads = 128; // 8x SIMD16 sub-groups on Intel Xe2

/// Sinkhorn balancing and routing matrix computation body.
/// For hc == 4 (the standard 4-stream hyper-connection):
///   comb matrix is 4x4 = 16 elements, fitting within a single SIMD16 sub-group without SLM spills.
template <typename SubGroup>
inline void hc_split_body_sycl(const float* mx, int t, int lane, const float* scale,
                               const float* base, int hc, int iters, float eps,
                               float* pre, float* post, float* comb, const SubGroup& sg) {
    pre += static_cast<size_t>(t) * 8;
    post += static_cast<size_t>(t) * 8;
    comb += static_cast<size_t>(t) * 64;

    if (lane < hc) {
        pre[lane] = 1.0f / (1.0f + sycl::native::exp(-(mx[lane] * scale[0] + base[lane]))) + eps;
        post[lane] = 2.0f / (1.0f + sycl::native::exp(-(mx[hc + lane] * scale[1] + base[hc + lane])));
    }

    if (hc == 4) {
        // 4x4 routing grid: rows j = lane / 4, cols k = lane % 4
        const bool on = lane < 16;
        float c = on ? (mx[8 + lane] * scale[2] + base[8 + lane]) : -1e30f;

        auto row_sum = [&](float v) {
            v += sycl::permute_group_by_xor(sg, v, 1);
            return v + sycl::permute_group_by_xor(sg, v, 2);
        };
        auto col_sum = [&](float v) {
            v += sycl::permute_group_by_xor(sg, v, 4);
            return v + sycl::permute_group_by_xor(sg, v, 8);
        };

        float m = sycl::fmax(c, sycl::permute_group_by_xor(sg, c, 1));
        m = sycl::fmax(m, sycl::permute_group_by_xor(sg, m, 2));

        c = on ? sycl::native::exp(c - m) : 0.0f;
        c = (c / (row_sum(c) + eps)) + eps; // softmax over each row, + eps
        c = on ? c : 0.0f;
        c = c / (col_sum(c) + eps);         // each destination column over its sources

        for (int it = 1; it < iters; ++it) {
            c = c / (row_sum(c) + eps);
            c = c / (col_sum(c) + eps);
        }

        if (on) comb[lane] = c;
        return;
    }

    // Generic fallback for non-4 stream configurations (computed on lane 0)
    if (lane == 0) {
        float c[8][8];
        for (int j = 0; j < hc; ++j) {
            float m = -1e30f;
            for (int k = 0; k < hc; ++k) {
                c[j][k] = mx[2 * hc + j * hc + k] * scale[2] + base[2 * hc + j * hc + k];
                m = std::max(m, c[j][k]);
            }
            float s = 0.0f;
            for (int k = 0; k < hc; ++k) {
                c[j][k] = std::exp(c[j][k] - m);
                s += c[j][k];
            }
            for (int k = 0; k < hc; ++k) c[j][k] = c[j][k] / s + eps;
        }
        for (int k = 0; k < hc; ++k) {
            float s = 0.0f;
            for (int j = 0; j < hc; ++j) s += c[j][k];
            for (int j = 0; j < hc; ++j) c[j][k] /= (s + eps);
        }
        for (int it = 1; it < iters; ++it) {
            for (int j = 0; j < hc; ++j) {
                float s = 0.0f;
                for (int k = 0; k < hc; ++k) s += c[j][k];
                for (int k = 0; k < hc; ++k) c[j][k] /= (s + eps);
            }
            for (int k = 0; k < hc; ++k) {
                float s = 0.0f;
                for (int j = 0; j < hc; ++j) s += c[j][k];
                for (int j = 0; j < hc; ++j) c[j][k] /= (s + eps);
            }
        }
        for (int j = 0; j < hc; ++j) {
            for (int k = 0; k < hc; ++k) {
                comb[j * hc + k] = c[j][k];
            }
        }
    }
}

/// Baseline separate hc_mix kernel
inline void hc_mix(const float* h, const float* fn, int hcdim, int n_mix, int T, float eps,
                   float* mixes, sycl::queue& q) {
    if (T <= 0 || n_mix <= 0 || hcdim <= 0) return;

    sycl::range<2> global_size(T, n_mix * kHcMixThreads);
    sycl::range<2> local_size(1, kHcMixThreads);

    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> slm_acc(sycl::range<1>(kHcMixThreads / 16), cgh);
        sycl::local_accessor<float, 1> slm_ss(sycl::range<1>(kHcMixThreads / 16), cgh);

        cgh.parallel_for(sycl::nd_range<2>(global_size, local_size), [=](sycl::nd_item<2> item) {
            const int t = item.get_group(0);
            const int m = item.get_group(1);
            const int tid = item.get_local_id(1);

            const float* w = fn + static_cast<size_t>(m) * hcdim;
            const float* x = h + static_cast<size_t>(t) * hcdim;

            float acc = 0.0f;
            float ss = 0.0f;
            for (int k = tid * 4; k < hcdim; k += kHcMixThreads * 4) {
                const auto hv = *reinterpret_cast<const sycl::vec<float, 4>*>(x + k);
                const auto wv = *reinterpret_cast<const sycl::vec<float, 4>*>(w + k);
                acc += hv[0] * wv[0] + hv[1] * wv[1] + hv[2] * wv[2] + hv[3] * wv[3];
                ss += hv[0] * hv[0] + hv[1] * hv[1] + hv[2] * hv[2] + hv[3] * hv[3];
            }

            auto sg = item.get_sub_group();
            acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());

            const int lane = sg.get_local_linear_id();
            const int sg_id = sg.get_group_linear_id();
            if (lane == 0) {
                slm_acc[sg_id] = acc;
                slm_ss[sg_id] = ss;
            }
            sycl::group_barrier(item.get_group());

            if (sg_id == 0) {
                const int num_sgs = kHcMixThreads / sg.get_local_range()[0];
                float tot_acc = (lane < num_sgs) ? slm_acc[lane] : 0.0f;
                float tot_ss = (lane < num_sgs) ? slm_ss[lane] : 0.0f;
                tot_acc = sycl::reduce_over_group(sg, tot_acc, sycl::plus<float>());
                tot_ss = sycl::reduce_over_group(sg, tot_ss, sycl::plus<float>());

                if (lane == 0) {
                    mixes[static_cast<size_t>(t) * 32 + m] = tot_acc * sycl::rsqrt(tot_ss / static_cast<float>(hcdim) + eps);
                }
            }
        });
    });
}

/// Baseline separate hc_split kernel
inline void hc_split(const float* mixes, const float* scale, const float* base, int hc,
                     int iters, float eps, int T, float* pre, float* post, float* comb, sycl::queue& q) {
    if (T <= 0) return;

    sycl::range<1> global_size(T * 32);
    sycl::range<1> local_size(32);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
            const int t = item.get_group(0);
            const int lane = item.get_local_id(0);
            auto sg = item.get_sub_group();
            const float* mx = mixes + static_cast<size_t>(t) * 32;
            hc_split_body_sycl(mx, t, lane, scale, base, hc, iters, eps, pre, post, comb, sg);
        });
    });
}

/// Fused hc_mix_split kernel: Single launch combining mix dot-products, RMS scaling, and Sinkhorn routing matrix split.
inline void hc_mix_split(const float* h, const float* fn, int hcdim, int n_mix, int T, float eps,
                         float* mixes, uint32_t* cnt, const float* scale, const float* base, int hc,
                         int iters, float hc_eps, float* pre, float* post, float* comb, sycl::queue& q) {
    if (T <= 0 || n_mix <= 0 || hcdim <= 0) return;

    sycl::range<2> global_size(T, n_mix * kHcMixThreads);
    sycl::range<2> local_size(1, kHcMixThreads);

    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> slm_acc(sycl::range<1>(kHcMixThreads / 16), cgh);
        sycl::local_accessor<float, 1> slm_ss(sycl::range<1>(kHcMixThreads / 16), cgh);
        sycl::local_accessor<uint32_t, 1> slm_last(sycl::range<1>(1), cgh);
        sycl::local_accessor<float, 1> slm_mx(sycl::range<1>(32), cgh);

        cgh.parallel_for(sycl::nd_range<2>(global_size, local_size), [=](sycl::nd_item<2> item) {
            const int t = item.get_group(0);
            const int m = item.get_group(1);
            const int tid = item.get_local_id(1);

            const float* w = fn + static_cast<size_t>(m) * hcdim;
            const float* x = h + static_cast<size_t>(t) * hcdim;

            float acc = 0.0f;
            float ss = 0.0f;
            for (int k = tid * 4; k < hcdim; k += kHcMixThreads * 4) {
                const auto hv = *reinterpret_cast<const sycl::vec<float, 4>*>(x + k);
                const auto wv = *reinterpret_cast<const sycl::vec<float, 4>*>(w + k);
                acc += hv[0] * wv[0] + hv[1] * wv[1] + hv[2] * wv[2] + hv[3] * wv[3];
                ss += hv[0] * hv[0] + hv[1] * hv[1] + hv[2] * hv[2] + hv[3] * hv[3];
            }

            auto sg = item.get_sub_group();
            acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());

            const int lane = sg.get_local_linear_id();
            const int sg_id = sg.get_group_linear_id();
            if (lane == 0) {
                slm_acc[sg_id] = acc;
                slm_ss[sg_id] = ss;
            }
            sycl::group_barrier(item.get_group());

            if (sg_id == 0) {
                const int num_sgs = kHcMixThreads / sg.get_local_range()[0];
                float tot_acc = (lane < num_sgs) ? slm_acc[lane] : 0.0f;
                float tot_ss = (lane < num_sgs) ? slm_ss[lane] : 0.0f;
                tot_acc = sycl::reduce_over_group(sg, tot_acc, sycl::plus<float>());
                tot_ss = sycl::reduce_over_group(sg, tot_ss, sycl::plus<float>());

                if (lane == 0) {
                    mixes[static_cast<size_t>(t) * 32 + m] = tot_acc * sycl::rsqrt(tot_ss / static_cast<float>(hcdim) + eps);

                    // Atomic completion detection across the n_mix workgroups for token t
                    sycl::atomic_ref<uint32_t,
                                     sycl::memory_order::acq_rel,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> ref(cnt[t]);
                    const uint32_t finished = ref.fetch_add(1u);
                    slm_last[0] = (finished == static_cast<uint32_t>(n_mix - 1)) ? 1u : 0u;
                }
            }
            sycl::group_barrier(item.get_group());

            if (slm_last[0] == 0u) return;

            // The last workgroup to complete for token t runs the split in-place!
            if (tid < 32) {
                slm_mx[tid] = (tid < n_mix) ? mixes[static_cast<size_t>(t) * 32 + tid] : 0.0f;
            }
            sycl::group_barrier(item.get_group());

            if (sg_id == 0) {
                hc_split_body_sycl(&slm_mx[0], t, lane, scale, base, hc, iters, hc_eps, pre, post, comb, sg);
            }

            // Reset counter for future runs
            if (tid == 0) {
                cnt[t] = 0u;
            }
        });
    });
}

/// Stream / queue interop overload
inline void hc_mix_split(const float* h, const float* fn, int hcdim, int n_mix, int T, float eps,
                         float* mixes, uint32_t* cnt, const float* scale, const float* base, int hc,
                         int iters, float hc_eps, float* pre, float* post, float* comb, void* stream) {
    if (stream != nullptr) {
        auto* q = static_cast<sycl::queue*>(stream);
        hc_mix_split(h, fn, hcdim, n_mix, T, eps, mixes, cnt, scale, base, hc, iters, hc_eps, pre, post, comb, *q);
    } else {
        static sycl::queue default_q{sycl::default_selector_v, sycl::property::queue::in_order()};
        hc_mix_split(h, fn, hcdim, n_mix, T, eps, mixes, cnt, scale, base, hc, iters, hc_eps, pre, post, comb, default_q);
    }
}

/// Runtime configuration controls
inline bool hc_mix_split_enabled() {
    static const int env_val = []() {
        const char* e = std::getenv("STRATA_HC_MIX_SPLIT");
        return (e != nullptr && e[0] == '0') ? 0 : 1; // Default ON
    }();
    return env_val != 0;
}

} // namespace strata::kernels
