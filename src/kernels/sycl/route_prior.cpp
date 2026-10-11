// src/kernels/sycl/route_prior.cpp - STRATA_ROUTE_PRIOR=lambda: cache-conditioned routing (opt-in).
//
// "Mixture of Cache-Conditional Experts" (arXiv 2412.00099), without training: when a token's 10 experts are picked,
// an expert already in VRAM gets a `lambda` bonus on its logit. The SELECTION uses the biased logits; the WEIGHTS stay
// the router's (softmax over the 512 unbiased logits), renormalised over the 10 picked like the native router.
#include <sycl/sycl.hpp>
#include "strata/kernels/route_prior.hpp"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>

namespace strata::kernels {
namespace {

static sycl::queue default_queue;

inline float wmax(float v, const sycl::sub_group& sg) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v = sycl::fmax(v, sycl::permute_group_by_xor(sg, v, m));
    return v;
}

// One subgroup (32 lanes) per token; 512 experts = 16 values per lane
void route_prior_k(const float* __restrict__ logits, const int32_t* __restrict__ res, float lambda,
                   int32_t* __restrict__ ids, float* __restrict__ weights, const sycl::nd_item<1>& item) {
    const float* lg = logits + (size_t)item.get_group(0) * 512;
    int32_t* id = ids + (size_t)item.get_group(0) * 10;
    float* w = weights + (size_t)item.get_group(0) * 10;
    const int lane = item.get_local_id(0);
    auto sg = item.get_sub_group();

    float p[16], s[16];
    float mx = -std::numeric_limits<float>::infinity();
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        p[i] = lg[lane + 32 * i];
        mx = sycl::fmax(mx, p[i]);
    }
    mx = wmax(mx, sg);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int e = lane + 32 * i;
        s[i] = p[i] + (res[e] >= 0 ? lambda : 0.0f);  // the selection score
        p[i] = sycl::exp(p[i] - mx);                  // router probability (unnormalised)
        sum += p[i];
    }
    (void)sum;
    float sel_p = 0.0f, sel_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = -std::numeric_limits<float>::max(), bp = 0.0f;
        int be = 1 << 30;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            if (s[i] > best) {
                best = s[i];
                be = lane + 32 * i;
                bp = p[i];
            }
        }
#pragma unroll
        for (int m = 16; m; m >>= 1) {
            const float ob = sycl::permute_group_by_xor(sg, best, m);
            const int oe = sycl::permute_group_by_xor(sg, be, m);
            const float op = sycl::permute_group_by_xor(sg, bp, m);
            if (ob > best || (ob == best && oe < be)) {
                best = ob;
                be = oe;
                bp = op;
            }
        }
        if ((be & 31) == lane) s[be >> 5] = -std::numeric_limits<float>::max();
        if (lane == rank) {
            id[rank] = be;
            sel_p = bp;
        }
        sel_sum += bp;  // the same on every lane
    }
    if (lane < 10) w[lane] = sel_p / sycl::fmax(sel_sum, 1e-30f);
}

}  // namespace

void route_prior_top10(const float* logits, const int32_t* res_layer, float lambda, int32_t* ids, float* weights,
                       int n_tok, void* stream) {
    if (n_tok <= 0) return;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)n_tok * 32), sycl::range<1>(32)),
                    [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                        route_prior_k(logits, res_layer, lambda, ids, weights, item);
                    });
    if (!stream) default_queue.wait();
}

}  // namespace strata::kernels
