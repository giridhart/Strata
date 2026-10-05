// src/kernels/sycl/fused_gdn.dp.cpp - Pure SYCL 2020 refactoring
// see include/strata/kernels/fused_gdn.hpp.
#include <sycl/sycl.hpp>
#include "strata/kernels/fused_gdn.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int S = 128;       // state size (rows = cols = 128)
constexpr int RG = 4;        // row groups
constexpr int RPG = S / RG;  // 32 rows per thread

static sycl::queue& default_queue() {
    static sycl::queue q{sycl::property::queue::in_order()};
    return q;
}

void gdn_step_norm_kernel(float* __restrict__ state, const float* __restrict__ q, const float* __restrict__ k,
                          const float* __restrict__ v, const float* __restrict__ gate, const float* __restrict__ beta,
                          const float* __restrict__ z, const float* __restrict__ gamma, float eps,
                          float* __restrict__ y, int h_k, int h_v, sycl::nd_item<2> item_ct1, float* sk, float* sq,
                          sycl::local_accessor<float, 2> red, float* wsum) {
    const int head = item_ct1.get_group(0);
    const int rg = item_ct1.get_local_id(0);   // 0..3
    const int col = item_ct1.get_local_id(1);  // 0..127
    const int tid = rg * S + col;
    const int qh = head % h_k;
    if (tid < S) {
        sk[tid] = k[qh * S + tid];
        sq[tid] = q[qh * S + tid];
    }
    float s[RPG];
    float* base = state + ((size_t)(rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t)h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r)
        s[r] = base[r * row_stride];
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const float g = sycl::native::exp(gate[head]);
    float kv = 0.0f;
#pragma unroll
    for (int r = 0; r < RPG; ++r)
        kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
    red[rg][col] = kv;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const float kv_col = red[0][col] + red[1][col] + red[2][col] + red[3][col];
    const float delta = (v[head * S + col] - g * kv_col) * beta[head];
    float o = 0.0f;
#pragma unroll
    for (int r = 0; r < RPG; ++r) {
        s[r] = sycl::fma((float)g, s[r], sk[rg * RPG + r] * delta);
        o = sycl::fma(s[r], sq[rg * RPG + r], o);
        base[r * row_stride] = s[r];
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);  // every thread has read red[] for kv_col
    red[rg][col] = o;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    float oc = 0.0f, sq_part = 0.0f;
    if (rg == 0) {
        oc = (red[0][col] + red[1][col] + red[2][col] + red[3][col]) * sycl::rsqrt((float)S);
        sq_part = oc * oc;
    }
    // RMS over the head's 128 outputs: warps of row group 0 are threads 0..127.
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    for (int o2 = 16; o2 > 0; o2 >>= 1) {
        sq_part += sycl::permute_group_by_xor(sg, sq_part, o2);
    }
    if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (rg == 0) {
        const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
        const float scale = sycl::rsqrt(ss / (float)S + eps);
        const float zz = z[head * S + col];
        y[head * S + col] = oc * scale * gamma[col] * (1.0f / (1.0f + sycl::native::exp(-zz)));
    }
}

void gdn_conv_l2_kernel(float* __restrict__ hist, const float* __restrict__ qkv, const float* __restrict__ w,
                        float* __restrict__ h, int qk_heads, float eps, sycl::nd_item<1> item_ct1, float* part) {
    const int c = item_ct1.get_global_id(0);
    const int local_id = item_ct1.get_local_id(0);
    const int group_id = item_ct1.get_group(0);
    const float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2], x = qkv[c];
    float sum = v0 * w[c * 4] + v1 * w[c * 4 + 1] + v2 * w[c * 4 + 2] + x * w[c * 4 + 3];
    hist[c * 3] = v1;
    hist[c * 3 + 1] = v2;
    hist[c * 3 + 2] = x;
    float y = sum / (1.0f + sycl::native::exp(-sum));
    if (group_id < qk_heads) {
        float sq = y * y;
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        for (int o = 16; o > 0; o >>= 1) {
            sq += sycl::permute_group_by_xor(sg, sq, o);
        }
        if ((local_id & 31) == 0) part[local_id >> 5] = sq;
        item_ct1.barrier(sycl::access::fence_space::local_space);
        const float ss = part[0] + part[1] + part[2] + part[3];
        y *= sycl::rsqrt(ss + eps);
    }
    h[c] = y;
}

void gdn_ab_kernel(const float* __restrict__ x, const uint16_t* __restrict__ wa, const uint16_t* __restrict__ wb,
                   const float* __restrict__ dt, const float* __restrict__ ssm_a, float* __restrict__ gate,
                   float* __restrict__ beta, int n, int h_v, sycl::nd_item<1> item_ct1) {
    const int row = item_ct1.get_group(0) * 8 + (item_ct1.get_local_id(0) >> 5);
    const int lane = item_ct1.get_local_id(0) & 31;
    if (row >= 2 * h_v) return;
    const bool is_beta = row >= h_v;
    const int r = is_beta ? row - h_v : row;
    const sycl::uint4* w4 = reinterpret_cast<const sycl::uint4*>((is_beta ? wb : wa) + (size_t)r * n);
    float acc = 0.0f;
    for (int j = lane; j < n / 8; j += 32) {
        const sycl::uint4 wv = *(w4 + j);
        const sycl::float4 xa = *reinterpret_cast<const sycl::float4*>(x + j * 8);
        const sycl::float4 xb = *reinterpret_cast<const sycl::float4*>(x + j * 8 + 4);
        acc = sycl::fma(sycl::bit_cast<float>(wv.x() << 16), (float)(xa.x()), acc);
        acc = sycl::fma(sycl::bit_cast<float>(wv.x() & 0xffff0000u), (float)(xa.y()), acc);
        acc = sycl::fma(sycl::bit_cast<float>(wv.y() << 16), (float)(xa.z()), acc);
        acc = sycl::fma(sycl::bit_cast<float>(wv.y() & 0xffff0000u), (float)(xa.w()), acc);
        acc = sycl::fma(sycl::bit_cast<float>(wv.z() << 16), (float)(xb.x()), acc);
        acc = sycl::fma(sycl::bit_cast<float>(wv.z() & 0xffff0000u), (float)(xb.y()), acc);
        acc = sycl::fma(sycl::bit_cast<float>(wv.w() << 16), (float)(xb.z()), acc);
        acc = sycl::fma(sycl::bit_cast<float>(wv.w() & 0xffff0000u), (float)(xb.w()), acc);
    }
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    for (int o = 16; o > 0; o >>= 1) {
        acc += sycl::permute_group_by_xor(sg, acc, o);
    }
    if (lane != 0) return;
    if (is_beta) {
        beta[r] = 1.0f / (1.0f + sycl::native::exp(-acc));
    } else {
        const float v = acc + dt[r];
        const float sp = v > 20.0f ? v : sycl::log1p(sycl::native::exp(v));
        gate[r] = sp * ssm_a[r];
    }
}

}  // namespace

void fused_gdn_conv_l2(float* history, const float* qkv, const float* conv_w, float* h, int channels, int qk_heads,
                       float eps, void* stream) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || qk_heads < 0 || qk_heads > channels / S) {
        std::fprintf(stderr, "fused_gdn_conv_l2: invalid arguments\n");
        std::exit(1);
    }
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> part_acc(sycl::range<1>(S / 32), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(channels), sycl::range<1>(S)),
                         [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                             gdn_conv_l2_kernel(history, qkv, conv_w, h, qk_heads, eps, item_ct1,
                                                part_acc.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
    });
}

void fused_gdn_ab(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, void* stream) {
    if (!x || !w_alpha || !w_beta || !dt || !ssm_a || !gate || !beta || n_embd % 8 != 0 || h_v <= 0) {
        std::fprintf(stderr, "fused_gdn_ab: invalid arguments\n");
        std::exit(1);
    }
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    const unsigned blocks = (unsigned)((2 * h_v + 7) / 8);
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * 256), sycl::range<1>(256)),
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                        gdn_ab_kernel(x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, item_ct1);
                    });
}

void fused_gdn_step_norm(float* state, const float* q, const float* k, const float* v, const float* gate,
                         const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k, int h_v,
                         void* stream) {
    if (!state || !q || !k || !v || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v <= 0 || h_v % h_k) {
        std::fprintf(stderr, "fused_gdn_step_norm: invalid arguments\n");
        std::exit(1);
    }
    sycl::queue* s_queue = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    s_queue->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> sk_acc(sycl::range<1>(S), cgh);
        sycl::local_accessor<float, 1> sq_acc(sycl::range<1>(S), cgh);
        sycl::local_accessor<float, 2> red_acc(sycl::range<2>(RG, S), cgh);
        sycl::local_accessor<float, 1> wsum_acc(sycl::range<1>(S * RG / 32), cgh);

        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(h_v * RG, S), sycl::range<2>(RG, S)),
                         [=](sycl::nd_item<2> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                             gdn_step_norm_kernel(state, q, k, v, gate, beta, z, gamma, eps, y, h_k, h_v, item_ct1,
                                                  sk_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                                                  sq_acc.get_multi_ptr<sycl::access::decorated::no>().get(), red_acc,
                                                  wsum_acc.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
    });
}

}  // namespace strata::kernels
