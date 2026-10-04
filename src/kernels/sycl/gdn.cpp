// src/kernels/cuda/gdn.cu - P2.S2: the gated delta-net's non-projection parts.
#include <sycl/sycl.hpp>
#include "strata/kernels/gdn.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int JTHREADS = 32;
constexpr int MAX_H = 1;

[[maybe_unused]] inline float softplus_f(float x) {
  return x > 20.0f ? x : sycl::log1p(sycl::native::exp(x));
}
inline float sigmoid_f(float x) {
  return 1.0f / (1.0f + sycl::native::exp(-x));
}

void gdn_step_kernel(sycl::nd_item<2> item, float* __restrict__ state, const float* __restrict__ q,
                                const float* __restrict__ k, const float* __restrict__ v,
                                const float* __restrict__ gate, const float* __restrict__ beta,
                                float* __restrict__ o, int S, int h_k, int h_v,
                                const sycl::local_accessor<float, 2>& ks,
                                const sycl::local_accessor<float, 2>& qs, float *dec_s,
                                float *beta_s) {
    const int h0 = item.get_group(0) * MAX_H;
    const int nh = sycl::min(MAX_H, h_v - h0);
    const int j = item.get_group(1) * JTHREADS + item.get_local_id(1);

    for (int hi = 0; hi < nh; ++hi) {
        const int h = h0 + hi;
        const int src = h % h_k;
        for (int i = item.get_local_id(1); i < S; i += JTHREADS) {
            ks[hi][i] = k[src * S + i];
            qs[hi][i] = q[src * S + i];
        }
        if (item.get_local_id(1) == 0) {
            dec_s[hi] = sycl::native::exp(gate[h]);
            beta_s[hi] = beta[h];
        }
    }
    item.barrier(sycl::access::fence_space::local_space);
    if (j >= S) return;

    for (int hi = 0; hi < nh; ++hi) {
        const int h = h0 + hi;
        const float dec = dec_s[hi], b = beta_s[hi];
        float* col = state + (size_t) h * S + j;
        const size_t stride = (size_t) h_v * S;

        float sk = 0.0f;
        for (int i = 0; i < S; ++i) {
            const float s = col[(size_t) i * stride] * dec;
            col[(size_t) i * stride] = s;
            sk += s * ks[hi][i];
        }
        const float d = (v[(size_t) h * S + j] - sk) * b;
        float dot = 0.0f;
        for (int i = 0; i < S; ++i) {
            const float s = col[(size_t) i * stride] + ks[hi][i] * d;
            col[(size_t) i * stride] = s;
            dot += s * qs[hi][i];
        }
        o[(size_t) h * S + j] = dot;
    }
}

void gdn_conv_kernel(sycl::nd_item<1> item, float* __restrict__ cs, const float* __restrict__ x,
                                const float* __restrict__ kW, float* __restrict__ out, int C, int dc) {
    const int c = item.get_global_id(0);
    if (c >= C) return;
    float* st = cs + (size_t) c * (dc - 1);
    const float* w = kW + (size_t) c * dc;
    float acc = 0.0f;
    for (int i = 0; i < dc - 1; ++i) acc += st[i] * w[i];
    acc += x[c] * w[dc - 1];
    out[c] = acc;
    for (int i = 0; i < dc - 2; ++i) st[i] = st[i + 1];
    st[dc - 2] = x[c];
}

void gdn_l2_kernel(sycl::nd_item<1> item, float* __restrict__ x, int cols, float eps, double &ssum) {
    const int row = item.get_group(0);
    float* p = x + (size_t) row * cols;
    double acc = 0.0;
    for (int i = item.get_local_id(0); i < cols;
         i += item.get_local_range(0)) acc += (double)p[i] * (double)p[i];
         
    for (int off = 16; off > 0; off >>= 1) acc += sycl::shift_group_left(
        item.get_sub_group(), acc, off);

    if (item.get_local_id(0) == 0) ssum = acc;
    item.barrier(sycl::access::fence_space::local_space);
    
    const float inv = (float)(1.0 / sycl::sqrt(ssum + (double)eps));
    for (int i = item.get_local_id(0); i < cols;
         i += item.get_local_range(0)) p[i] *= inv;
}

void gdn_out_norm_kernel(sycl::nd_item<1> item, const float* __restrict__ o, const float* __restrict__ z,
                                    const float* __restrict__ ssm_norm, float* __restrict__ y, int S,
                                    float eps, double &ssum) {
    const int h = item.get_group(0);
    const float* po = o + (size_t) h * S;
    const float* pz = z + (size_t) h * S;
    float* py = y + (size_t) h * S;
    double acc = 0.0;
    for (int i = item.get_local_id(0); i < S;
         i += item.get_local_range(0)) acc += (double)po[i] * (double)po[i];
         
    for (int off = 16; off > 0; off >>= 1) acc += sycl::shift_group_left(
        item.get_sub_group(), acc, off);

    if (item.get_local_id(0) == 0) ssum = acc;
    item.barrier(sycl::access::fence_space::local_space);
    
    const float inv = (float)(1.0 / sycl::sqrt(ssum / (double)S + (double)eps));
    for (int i = item.get_local_id(0); i < S;
         i += item.get_local_range(0))
        py[i] = po[i] * inv * ssm_norm[i] * sigmoid_f(pz[i]);
}

void gdn_beta_gate_kernel(sycl::nd_item<1> item, float* __restrict__ beta, int n) {
    const int i = item.get_global_id(0);
    if (i < n) beta[i] = sigmoid_f(beta[i]);
}

}  // namespace

void gdn_step(float *state, const float *q, const float *k, const float *v,
              const float *gate, const float *beta, float *o,
              const GdnShapes &s, void *stream) {
    if (s.S <= 0 || s.h_k <= 0 || s.h_v <= 0) return;
    if (s.S > 128) {
        std::fprintf(stderr, "gdn_step: S = %lld exceeds the staged 128 (`ks`/`qs` are [8][128])\n", (long long) s.S);
        std::exit(1);
    }
    const unsigned grid_x = (unsigned)((s.S + JTHREADS - 1) / JTHREADS);
    const unsigned grid_y = (unsigned)((s.h_v + MAX_H - 1) / MAX_H);
    
    static sycl::queue default_q;
    sycl::queue* q_ptr = stream ? reinterpret_cast<sycl::queue*>(stream) : &default_q;
    
    q_ptr->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 2> ks_acc(sycl::range<2>(MAX_H, 128), cgh);
        sycl::local_accessor<float, 2> qs_acc(sycl::range<2>(MAX_H, 128), cgh);
        sycl::local_accessor<float, 1> dec_s_acc(sycl::range<1>(MAX_H), cgh);
        sycl::local_accessor<float, 1> beta_s_acc(sycl::range<1>(MAX_H), cgh);

        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(grid_y, grid_x * JTHREADS), sycl::range<2>(1, JTHREADS)),
            [=](sycl::nd_item<2> item) {
                gdn_step_kernel(
                    item, state, q, k, v, gate, beta, o, (int)s.S, (int)s.h_k,
                    (int)s.h_v, ks_acc, qs_acc,
                    dec_s_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                    beta_s_acc.get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });
    if (stream == nullptr) q_ptr->wait();
}

void gdn_conv_step(float *conv_state, const float *x, const float *kW,
                   float *out, int64_t channels, int64_t d_conv,
                   void *stream) {
    if (channels <= 0 || d_conv < 1) return;
    const int blocks = (int) ((channels + 255) / 256);
    static sycl::queue default_q;
    sycl::queue* q_ptr = stream ? reinterpret_cast<sycl::queue*>(stream) : &default_q;
    
    q_ptr->parallel_for(sycl::nd_range<1>(blocks * 256, 256),
                   [=](sycl::nd_item<1> item) {
                       gdn_conv_kernel(item, conv_state, x, kW, out, (int)channels, (int)d_conv);
                   });
    if (stream == nullptr) q_ptr->wait();
}

void gdn_l2_norm(float *x, int64_t rows, int64_t cols, float eps,
                 void *stream) {
    if (rows <= 0 || cols <= 0) return;
    if (cols > 1024) {
        std::fprintf(stderr, "gdn_l2_norm: cols = %lld exceeds the 1024-wide warp reduction\n", (long long) cols);
        std::exit(1);
    }
    static sycl::queue default_q;
    sycl::queue* q_ptr = stream ? reinterpret_cast<sycl::queue*>(stream) : &default_q;

    q_ptr->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<double, 0> ssum_acc(cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(rows * 32, 32),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                gdn_l2_kernel(item, x, (int)cols, eps, ssum_acc);
            });
    });
    if (stream == nullptr) q_ptr->wait();
}

void gdn_beta_gate(float *beta, int64_t h_v, void *stream) {
    if (beta == nullptr || h_v <= 0) return;
    const int n = (int) h_v;
    static sycl::queue default_q;
    sycl::queue* q_ptr = stream ? reinterpret_cast<sycl::queue*>(stream) : &default_q;
    
    const int threads = 128;
    const int blocks = (n + threads - 1) / threads;
    
    q_ptr->parallel_for(sycl::nd_range<1>(blocks * threads, threads),
                   [=](sycl::nd_item<1> item) {
                       gdn_beta_gate_kernel(item, beta, n);
                   });
    if (stream == nullptr) q_ptr->wait();
}

void gdn_out_norm(const float *o, const float *z, const float *ssm_norm,
                  float *y, int64_t h_v, int64_t S, float eps,
                  void *stream) {
    if (h_v <= 0 || S <= 0) return;
    static sycl::queue default_q;
    sycl::queue* q_ptr = stream ? reinterpret_cast<sycl::queue*>(stream) : &default_q;

    q_ptr->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<double, 0> ssum_acc(cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(h_v * 32, 32),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                gdn_out_norm_kernel(item, o, z, ssm_norm, y, (int)S, eps, ssum_acc);
            });
    });
    if (stream == nullptr) q_ptr->wait();
}

}  // namespace strata::kernels
