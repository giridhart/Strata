// src/kernels/sycl/router_top10.dp.cpp - Pure SYCL 2020 MoE router.
#include <sycl/sycl.hpp>
#include <cstdint>
#include <algorithm>
#include "strata/kernels/router_top10.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace strata::kernels {
namespace {

constexpr int RT_MAX_THREADS = 512;

inline sycl::queue& default_queue() {
    static sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    return q;
}

/// Does this queue's device run fp64 kernels? (Asked once per device; Intel Arc A- and B-series do not.)
bool device_has_fp64(const sycl::queue& q) {
    static std::mutex mu;
    static std::vector<std::pair<sycl::device, bool>> seen;
    const sycl::device d = q.get_device();
    std::lock_guard<std::mutex> lock(mu);
    for (const auto& [dev, has] : seen)
        if (dev == d) return has;
    const bool has = d.has(sycl::aspect::fp64);
    seen.emplace_back(d, has);
    return has;
}

/// Shared MoE top-k router kernel parameterized by accumulator type (`double` on FP64 devices, `float` + Kahan
/// compensated summation on devices without FP64 such as Intel Arc A/B-series so no `double` instruction appears).
template <typename AccumT>
void router_top10_kernel_impl(sycl::nd_item<1> item, const float* __restrict__ logits, int n_tokens, int n_expert, int k,
                              int* __restrict__ ids, float* __restrict__ weights, uint8_t* s_raw, float* s_red,
                              int* s_rid, AccumT& s_sum) {
    const size_t taken_bytes = ((size_t)n_expert + 15u) & ~(size_t)15u;
    unsigned char* s_taken = s_raw;
    AccumT* s_ex = (AccumT*)(s_raw + taken_bytes);

    const int t = item.get_group(0);
    if (t >= n_tokens) return;
    const int tid = item.get_local_id(0);
    const int nt = item.get_local_range(0);
    const float* l = logits + (size_t)t * n_expert;

    // ---- softmax over ALL experts, for stability: the max. A tree of `fmax` is EXACT and order-independent.
    float mx = -INFINITY;
    for (int e = tid; e < n_expert; e += nt)
        mx = sycl::fmax(mx, l[e]);
    mx = sycl::reduce_over_group(item.get_sub_group(), mx, sycl::maximum<float>());
    if ((tid & 31) == 0) s_red[tid >> 5] = mx;
    item.barrier(sycl::access::fence_space::local_space);
    if (tid < 32) {
        const int nw = (nt + 31) >> 5;
        float v = (tid < nw) ? s_red[tid] : -INFINITY;
        v = sycl::reduce_over_group(item.get_sub_group(), v, sycl::maximum<float>());
        if (tid == 0) s_red[0] = v;
    }
    item.barrier(sycl::access::fence_space::local_space);
    mx = s_red[0];

    // ---- exponentials, once each and in parallel.
    for (int e = tid; e < n_expert; e += nt) {
        if constexpr (std::is_same_v<AccumT, float>)
            s_ex[e] = sycl::exp(l[e] - mx);
        else
            s_ex[e] = sycl::exp((double)l[e] - (double)mx);
    }
    item.barrier(sycl::access::fence_space::local_space);

    // ---- ascending serial sum on one thread (Kahan compensated for float, bit-reproducible order).
    if (tid == 0) {
        AccumT sum = AccumT(0);
        if constexpr (std::is_same_v<AccumT, float>) {
            float c = 0.0f;
            for (int e = 0; e < n_expert; ++e) {
                const float y = s_ex[e] - c;
                const float u = sum + y;
                c = (u - sum) - y;
                sum = u;
            }
        } else {
            for (int e = 0; e < n_expert; ++e)
                sum += s_ex[e];
        }
        s_sum = sum;
    }
    item.barrier(sycl::access::fence_space::local_space);
    const float inv = (float)(AccumT(1) / s_sum);

    float* s_p = (float*)(s_ex + n_expert);
    for (int e = tid; e < n_expert; e += nt)
        s_p[e] = (float)(s_ex[e] * (AccumT)inv);
    item.barrier(sycl::access::fence_space::local_space);

    // ================================ THE SELECTION, IN k PASSES ================================
    for (int e = tid; e < n_expert; e += nt)
        s_taken[e] = 0;
    item.barrier(sycl::access::fence_space::local_space);

    for (int i = 0; i < k; ++i) {
        float bv = -INFINITY;
        int bi = n_expert;  // a sentinel that loses to every real index
        for (int e = tid; e < n_expert; e += nt) {
            if (s_taken[e]) continue;
            const float pe = s_p[e];
            if (pe > bv) {
                bv = pe;
                bi = e;
            }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = sycl::shift_group_left(item.get_sub_group(), bv, off);
            const int oi = sycl::shift_group_left(item.get_sub_group(), bi, off);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if ((tid & 31) == 0) {
            s_red[tid >> 5] = bv;
            s_rid[tid >> 5] = bi;
        }
        item.barrier(sycl::access::fence_space::local_space);
        if (tid < 32) {
            const int nw = (nt + 31) >> 5;
            float v = (tid < nw) ? s_red[tid] : -INFINITY;
            int ix = (tid < nw) ? s_rid[tid] : n_expert;
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = sycl::shift_group_left(item.get_sub_group(), v, off);
                const int oi = sycl::shift_group_left(item.get_sub_group(), ix, off);
                if (ov > v || (ov == v && oi < ix)) {
                    v = ov;
                    ix = oi;
                }
            }
            if (tid == 0 && ix < n_expert) {
                ids[(size_t)t * k + i] = ix;
                weights[(size_t)t * k + i] = v;
                s_taken[ix] = 1;
            }
        }
        item.barrier(sycl::access::fence_space::local_space);
    }
    item.barrier(sycl::access::fence_space::local_space);

    // ---- renormalise, with ggml's lower clamp. Order preserved.
    if (tid == 0) {
        constexpr AccumT kClamp = AccumT(6.103515625e-05);  // 2**-14
        AccumT s = AccumT(0);
        if constexpr (std::is_same_v<AccumT, float>) {
            float c = 0.0f;
            for (int i = 0; i < k; ++i) {
                const float y = weights[(size_t)t * k + i] - c;
                const float u = s + y;
                c = (u - s) - y;
                s = u;
            }
        } else {
            for (int i = 0; i < k; ++i)
                s += (AccumT)weights[(size_t)t * k + i];
        }
        const AccumT sc = sycl::fmax(s, kClamp);
        for (int i = 0; i < k; ++i) {
            weights[(size_t)t * k + i] = (float)((AccumT)weights[(size_t)t * k + i] / sc);
        }
    }
}

template <typename AccumT>
void launch_router_kernel(sycl::queue& q, const float* logits, int n_tokens, int n_expert, int k, int* ids,
                          float* weights) {
    int threads = n_expert < RT_MAX_THREADS ? n_expert : RT_MAX_THREADS;
    threads = (threads + 31) & ~31;  // at least one full warp, for the reductions

    const size_t taken_bytes = ((size_t)n_expert + 15u) & ~(size_t)15u;
    const size_t smem = taken_bytes + (size_t)n_expert * sizeof(AccumT) + (size_t)n_expert * sizeof(float);

    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<uint8_t, 1> s_raw_acc(sycl::range<1>(smem), cgh);
        sycl::local_accessor<float, 1> s_red_acc(sycl::range<1>(RT_MAX_THREADS / 32), cgh);
        sycl::local_accessor<int, 1> s_rid_acc(sycl::range<1>(RT_MAX_THREADS / 32), cgh);
        sycl::local_accessor<AccumT, 0> s_sum_acc(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(((unsigned)n_tokens) * (threads)), sycl::range<1>(threads)),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                             router_top10_kernel_impl<AccumT>(
                                 item, logits, n_tokens, n_expert, k, ids, weights,
                                 s_raw_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                                 s_red_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                                 s_rid_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                                 s_sum_acc);
                         });
    });
}

bool valid_router_args(int n_tokens, int n_expert, int k) {
    if (n_tokens <= 0 || n_expert <= 0 || k <= 0) return false;
    if (k > 64) {
        std::fprintf(stderr, "router_top10: k %d exceeds the kernel's 64\n", k);
        std::exit(1);
    }
    if (n_expert > RT_MAX_THREADS * 64) {
        std::fprintf(stderr, "router_top10: n_expert %d is past the kernel's %d\n", n_expert, RT_MAX_THREADS * 64);
        std::exit(1);
    }
    return true;
}

}  // namespace

bool router_top10_variant(const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights,
                          void* stream, int variant) {
    // variant 0 = FP64 (requires device aspect::fp64), variant 3 = FP32 (works on all devices including Arc A/B).
    if (!valid_router_args(n_tokens, n_expert, k)) return false;
    if (variant != 0 && variant != 3) return false;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    if (variant == 0 && !device_has_fp64(*q)) return false;
    if (variant == 3)
        launch_router_kernel<float>(*q, logits, n_tokens, n_expert, k, ids, weights);
    else
        launch_router_kernel<double>(*q, logits, n_tokens, n_expert, k, ids, weights);
    return true;
}

void router_top10(const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights, void* stream) {
    if (!valid_router_args(n_tokens, n_expert, k)) return;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    // A card without fp64 (Intel Arc A/B-series) refuses kernels using `double`; take the FP32 kernel there.
    if (!device_has_fp64(*q))
        launch_router_kernel<float>(*q, logits, n_tokens, n_expert, k, ids, weights);
    else
        launch_router_kernel<double>(*q, logits, n_tokens, n_expert, k, ids, weights);
}

}  // namespace strata::kernels
