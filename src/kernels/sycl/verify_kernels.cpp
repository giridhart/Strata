// src/kernels/cuda/verify_kernels.cu - see include/strata/kernels/verify_kernels.hpp.
//
// The per-token arithmetic of every kernel here is transcribed from its single-token original (fused_gdn.cu,
// elementwise.cu) with the same operation order, so a verify window reproduces plain decode bit for bit.
#include <sycl/sycl.hpp>
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/dp4a.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace strata::kernels {
namespace {

static sycl::queue& default_queue() {
    static sycl::queue q{sycl::property::queue::in_order()};
    return q;
}

constexpr int S = 128;  // GDN state size
constexpr int RG = 4;
constexpr int RPG = S / RG;  // 32

void gdn_conv_l2_multi_kernel(const float* __restrict__ hist, const float* __restrict__ qkv,
                              const float* __restrict__ w, float* __restrict__ h, int C, int qk_heads, float eps,
                              int t_begin, float* part, sycl::nd_item<2> item) {
    const int t = t_begin + item.get_group(0);
    const int qk_group = item.get_group(1);
    const int c = qk_group * S + item.get_local_id(1);
    const int local_id = item.get_local_id(1);

    // the window of token t: [hist0, hist1, hist2, x_0, ..., x_t], its last four entries
    float win[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = t + j;  // index into [hist(3) | x...]
        win[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t)(src - 3) * C + c];
    }
    const float v0 = win[0], v1 = win[1], v2 = win[2], x = qkv[(size_t)t * C + c];
    float sum = v0 * w[c * 4] + v1 * w[c * 4 + 1] + v2 * w[c * 4 + 2] + x * w[c * 4 + 3];
    float y = sum / (1.0f + sycl::native::exp(-sum));
    if (qk_group < qk_heads) {
        float sq = y * y;
        auto sg = item.get_sub_group();
        sq = sycl::reduce_over_group(sg, sq, sycl::plus<float>());
        if ((local_id & 31) == 0) part[local_id >> 5] = sq;
        item.barrier(sycl::access::fence_space::local_space);
        const float ss = part[0] + part[1] + part[2] + part[3];
        y *= sycl::rsqrt(ss + eps);
    }
    h[(size_t)t * C + c] = y;
}

void gdn_conv_commit_kernel(float* __restrict__ hist, const float* __restrict__ qkv, int C,
                            const int32_t* __restrict__ n_keep, sycl::nd_item<1> item) {
    const int c = item.get_global_id(0);
    if (c >= C) return;
    const int n = *n_keep;
    if (n <= 0) return;
    float seq[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = n + j;  // the last three of [hist(3) | x_0..x_{n-1}]
        seq[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t)(src - 3) * C + c];
    }
    hist[c * 3] = seq[0];
    hist[c * 3 + 1] = seq[1];
    hist[c * 3 + 2] = seq[2];
}

void gdn_conv_commit_layers_kernel(float* __restrict__ hist0, int64_t hist_stride, const float* __restrict__ qkv0,
                                   int64_t qkv_stride, int C, const int32_t* __restrict__ n_keep,
                                   sycl::nd_item<2> item) {
    const size_t layer = item.get_group(0);
    const int c = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
    if (c >= C) return;
    const int n = *n_keep;
    if (n <= 0) return;
    float* hist = hist0 + (size_t)layer * hist_stride;
    const float* qkv = qkv0 + (size_t)layer * qkv_stride;
    float seq[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = n + j;  // the last three of [hist(3) | x_0..x_{n-1}]
        seq[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t)(src - 3) * C + c];
    }
    hist[c * 3] = seq[0];
    hist[c * 3 + 1] = seq[1];
    hist[c * 3 + 2] = seq[2];
}

void gdn_ab_multi_kernel(const float* __restrict__ x, const uint16_t* __restrict__ wa, const uint16_t* __restrict__ wb,
                         const float* __restrict__ dt, const float* __restrict__ ssm_a, float* __restrict__ gate,
                         float* __restrict__ beta, int n, int h_v, int T, sycl::nd_item<1> item) {
    const int row = item.get_group(0) * 8 + (item.get_local_id(0) >> 5);
    const int lane = item.get_local_id(0) & 31;
    if (row >= 2 * h_v) return;
    const bool is_beta = row >= h_v;
    const int r = is_beta ? row - h_v : row;
    const sycl::uint4* w4 = reinterpret_cast<const sycl::uint4*>((is_beta ? wb : wa) + (size_t)r * n);
    float acc[kVerifyMaxT];
#pragma unroll
    for (int t = 0; t < kVerifyMaxT; ++t)
        acc[t] = 0.0f;
    for (int j = lane; j < n / 8; j += 32) {
        const sycl::uint4 wv = *(w4 + j);
#pragma unroll
        for (int t = 0; t < kVerifyMaxT; ++t) {
            if (t >= T) break;
            const float* xt = x + (size_t)t * n;
            const sycl::float4 xa = *reinterpret_cast<const sycl::float4*>(xt + j * 8);
            const sycl::float4 xb = *reinterpret_cast<const sycl::float4*>(xt + j * 8 + 4);
            float a = acc[t];
            a = sycl::fma(sycl::bit_cast<float>(wv.x() << 16), (float)(xa.x()), a);
            a = sycl::fma(sycl::bit_cast<float>(wv.x() & 0xffff0000u), (float)(xa.y()), a);
            a = sycl::fma(sycl::bit_cast<float>(wv.y() << 16), (float)(xa.z()), a);
            a = sycl::fma(sycl::bit_cast<float>(wv.y() & 0xffff0000u), (float)(xa.w()), a);
            a = sycl::fma(sycl::bit_cast<float>(wv.z() << 16), (float)(xb.x()), a);
            a = sycl::fma(sycl::bit_cast<float>(wv.z() & 0xffff0000u), (float)(xb.y()), a);
            a = sycl::fma(sycl::bit_cast<float>(wv.w() << 16), (float)(xb.z()), a);
            a = sycl::fma(sycl::bit_cast<float>(wv.w() & 0xffff0000u), (float)(xb.w()), a);
            acc[t] = a;
        }
    }
    auto sg = item.get_sub_group();
#pragma unroll
    for (int t = 0; t < kVerifyMaxT; ++t) {
        if (t >= T) break;
        float a = acc[t];
        a = sycl::reduce_over_group(sg, a, sycl::plus<float>());
        if (lane != 0) continue;
        if (is_beta) {
            beta[(size_t)t * h_v + r] = 1.0f / (1.0f + sycl::native::exp(-a));
        } else {
            const float v = a + dt[r];
            const float sp = v > 20.0f ? v : sycl::log1p(sycl::native::exp(v));
            gate[(size_t)t * h_v + r] = sp * ssm_a[r];
        }
    }
}

inline void gdn_step_commit_body(float* __restrict__ state, const float* __restrict__ hbuf, int C,
                                 const float* __restrict__ gate, const float* __restrict__ beta,
                                 int h_k, int h_v, int n,
                                 sycl::local_accessor<float, 1> sk,
                                 sycl::local_accessor<float, 2> red,
                                 int head, int col_block, int rg, int col_local,
                                 sycl::nd_item<3> item) {
    const int col = col_block * 32 + col_local;
    const int tid = rg * 32 + col_local;
    const int qh = head % h_k;
    const int qk = S * h_k;
    float s[RPG];
    float* base = state + ((size_t)(rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t)h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t)t * C;
        item.barrier(sycl::access::fence_space::local_space);
        sk[tid] = ht[qk + qh * S + tid];
        item.barrier(sycl::access::fence_space::local_space);
        const float g = sycl::native::exp(gate[(size_t)t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
        red[rg][col_local] = kv;
        item.barrier(sycl::access::fence_space::local_space);
        const float kv_col = red[0][col_local] + red[1][col_local] + red[2][col_local] + red[3][col_local];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t)t * h_v + head];
#pragma unroll
        for (int r = 0; r < RPG; ++r) s[r] = sycl::fma(g, s[r], sk[rg * RPG + r] * delta);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
}

void gdn_step_norm_multi_kernel(float* __restrict__ state, const float* __restrict__ hbuf, int C,
                                const float* __restrict__ gate, const float* __restrict__ beta,
                                const float* __restrict__ z, const float* __restrict__ gamma, float eps,
                                float* __restrict__ y, int h_k, int h_v, int T, const int32_t* __restrict__ n_keep,
                                int t_out_begin, sycl::local_accessor<float, 1> sk, sycl::local_accessor<float, 1> sq,
                                sycl::local_accessor<float, 2> red, sycl::local_accessor<float, 1> wsum,
                                sycl::nd_item<2> item) {
    const int head = item.get_group(0);
    const int rg = item.get_local_id(0);
    const int col = item.get_local_id(1);
    const int tid = rg * S + col;
    const int qh = head % h_k;
    const int qk = S * h_k;  // q at [0, qk), k at [qk, 2qk), v at [2qk, ...)
    const int value_dim = S * h_v;
    const int n = n_keep ? *n_keep : T;
    float s[RPG];
    float* base = state + ((size_t)(rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t)h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r)
        s[r] = base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t)t * C;
        item.barrier(sycl::access::fence_space::local_space);  // previous token done
        if (tid < S) {
            sk[tid] = ht[qk + qh * S + tid];
            sq[tid] = ht[qh * S + tid];
        }
        item.barrier(sycl::access::fence_space::local_space);
        const float g = sycl::native::exp(gate[(size_t)t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r)
            kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
        red[rg][col] = kv;
        item.barrier(sycl::access::fence_space::local_space);
        const float kv_col = red[0][col] + red[1][col] + red[2][col] + red[3][col];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t)t * h_v + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = sycl::fma((float)g, s[r], sk[rg * RPG + r] * delta);
            o = sycl::fma(s[r], sq[rg * RPG + r], o);
        }
        item.barrier(sycl::access::fence_space::local_space);
        red[rg][col] = o;
        item.barrier(sycl::access::fence_space::local_space);
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red[0][col] + red[1][col] + red[2][col] + red[3][col]) * sycl::rsqrt((float)S);
            sq_part = oc * oc;
        }
        if (t < t_out_begin) continue;  // a replayed token: its state update is needed, its output is not
        auto sg = item.get_sub_group();
        sq_part = sycl::reduce_over_group(sg, sq_part, sycl::plus<float>());
        if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
        item.barrier(sycl::access::fence_space::local_space);
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = sycl::rsqrt(ss / (float)S + eps);
            const float zz = z[(size_t)t * value_dim + head * S + col];
            y[(size_t)t * value_dim + head * S + col] =
                oc * scale * gamma[col] * (1.0f / (1.0f + sycl::native::exp(-zz)));
        }
    }
    if (n_keep != nullptr && n > 0) {
#pragma unroll
        for (int r = 0; r < RPG; ++r)
            base[r * row_stride] = s[r];
    }
}

void embedding_gather_dev_kernel(const uint8_t* __restrict__ codes, const float* __restrict__ scales,
                                 const float* __restrict__ offsets, const int32_t* __restrict__ tokens, int64_t n,
                                 int code_bits, int code_bias, int group_elems, unsigned long long row_codes,
                                 unsigned long long row_groups, float* __restrict__ out, sycl::nd_item<2> item) {
    const int t = item.get_group(0);
    const int64_t i = (int64_t)item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
    if (i >= n) return;
    const unsigned long long token = (unsigned long long)tokens[t];
    const uint8_t* c = codes + token * row_codes;
    const float* sc = scales + token * row_groups;
    const float* of = offsets ? offsets + token * row_groups : nullptr;
    const int per_byte = 8 / code_bits;
    const unsigned mask = (1u << code_bits) - 1u;
    const int code = (c[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
    const int64_t group = i / group_elems;
    const float product = (float)(code + code_bias) * sc[group];
    out[(size_t)t * n + i] = product + (of ? of[group] : 0.0f);
}

void broadcast_streams_kernel(const float* __restrict__ x, float* __restrict__ R, int64_t n, int hc,
                              sycl::nd_item<2> item) {
    const int t = item.get_group(0);
    const int64_t i = (int64_t)item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
    if (i >= n * hc) return;
    R[(size_t)t * n * hc + i] = x[(size_t)t * n + i % n];
}

void copy_indexed_kernel(float* __restrict__ dst, const float* __restrict__ src, int64_t stride,
                         const int32_t* __restrict__ index, int64_t n, sycl::nd_item<1> item) {
    const int idx = *index;
    if (idx < 0) return;
    for (int64_t i = item.get_global_id(0); i < n; i += item.get_global_range(0)) {
        dst[i] = src[(size_t)idx * stride + i];
    }
}

void fetch_blobs_kernel(const unsigned long long* __restrict__ src, const int32_t* __restrict__ n,
                        sycl::uint4* __restrict__ dst, long long per, sycl::nd_item<1> item) {
    const long long total = (long long)*n * per;
    for (long long i = item.get_global_id(0); i < total; i += item.get_global_range(0)) {
        const long long k = i / per, off = i - k * per;
        dst[i] = ((const sycl::uint4*)src[k])[off];
    }
}

void rebase_ptrs_kernel(unsigned long long* ptr, const int32_t* n, unsigned long long base, long long bytes,
                        sycl::nd_item<1> item) {
    const int k = item.get_local_id(0);
    if (k < *n) ptr[k] = base + (unsigned long long)k * (unsigned long long)bytes;
}

void add_streams_broadcast_kernel(const float* __restrict__ h, const float* __restrict__ e, float* __restrict__ R,
                                  int64_t n, int hc, sycl::nd_item<2> item) {
    const int t = item.get_group(0);
    const int64_t i = (int64_t)item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
    if (i >= n * hc) return;
    R[(size_t)t * n * hc + i] = h[(size_t)t * n * hc + i] + e[(size_t)t * n + i % n];
}

void ident_hits_kernel(const int32_t* __restrict__ ids, int n, int32_t* __restrict__ slot, int32_t* __restrict__ dst,
                       int32_t* __restrict__ count, sycl::nd_item<1> item) {
    const int i = item.get_local_id(0);
    if (i < n) {
        slot[i] = ids[i];
        dst[i] = i;
    }
    if (i == 0) *count = n;
}

template <typename E>
void gather_rows_kernel(const E* __restrict__ src, long long row_e, const int32_t* __restrict__ ids, long long n,
                        E* __restrict__ dst, sycl::nd_item<1> item) {
    const long long total = n * row_e;
    for (long long i = item.get_global_id(0); i < total; i += item.get_global_range(0)) {
        const long long r = i / row_e, o = i - r * row_e;
        dst[i] = src[(long long)ids[r] * row_e + o];
    }
}

void map_ids_kernel(int32_t* ids, const int32_t* __restrict__ table, int n, sycl::nd_item<1> item) {
    const int i = item.get_local_id(0);
    if (i < n) ids[i] = table[ids[i]];
}

void row_top_prob_kernel(const float* __restrict__ logits, int n_vocab, const int32_t* __restrict__ ids,
                         float* __restrict__ probs, sycl::local_accessor<float, 1> part, sycl::nd_item<2> item) {
    const int t = item.get_group(0);
    const int tid = item.get_local_id(1);
    const float* l = logits + (size_t)t * n_vocab;
    const float m = l[ids[t]];
    float s = 0.0f;
    for (int i = tid; i < n_vocab; i += 1024)
        s += sycl::native::exp(l[i] - m);
    auto sg = item.get_sub_group();
    s = sycl::reduce_over_group(sg, s, sycl::plus<float>());
    if ((tid & 31) == 0) part[tid >> 5] = s;
    item.barrier(sycl::access::fence_space::local_space);
    if (tid == 0) {
        float tot = 0.0f;
        for (int w = 0; w < 32; ++w)
            tot += part[w];
        probs[t] = 1.0f / tot;
    }
}

void mtp_select_kernel(const float* __restrict__ R_src, int64_t stride, const int32_t* __restrict__ ids,
                       const int32_t* __restrict__ row_dev, float* __restrict__ R_dst, int32_t* __restrict__ tok_dst,
                       int32_t* out, int j, const float* probs, float* out_p, sycl::nd_item<1> item) {
    const int row = *row_dev;
    for (int64_t i = item.get_global_id(0); i < stride; i += item.get_global_range(0)) {
        R_dst[i] = R_src[(size_t)row * stride + i];
    }
    if (item.get_global_id(0) == 0) {
        const int32_t tok = ids[row];
        *tok_dst = tok;
        if (probs != nullptr && out_p != nullptr) ((volatile float*)out_p)[j] = probs[row];
        sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
        if (out != nullptr) ((volatile int32_t*)out)[j] = tok;
    }
}

void dense_steps_kernel(const int32_t* __restrict__ cells, int n, int32_t* __restrict__ steps, sycl::nd_item<1> item) {
    const int i = item.get_local_id(0);
    if (i >= n) return;
    const int c = cells[i];
    steps[i * 4 + 0] = c;
    steps[i * 4 + 1] = c + 1;
    steps[i * 4 + 2] = (c + 1) / 4;
    steps[i * 4 + 3] = c + 1;
}

void window_ids_kernel(int32_t* steps, int window, int32_t* ids, long long stride, sycl::nd_item<2> item) {
    const int q = item.get_group(0);
    int32_t* st = steps + q * 4;
    const int n_kv = st[1];
    const int start = n_kv > window ? n_kv - window : 0;
    const int width = n_kv - start;
    for (int j = item.get_local_id(1); j < width; j += item.get_local_range(1)) {
        ids[q * stride + j] = start + j;
    }
    item.barrier(sycl::access::fence_space::local_space);
    if (item.get_local_id(1) == 0) st[3] = width;
}

void resident_plan_kernel(const int32_t* __restrict__ ids, int n, int k, const int32_t* __restrict__ res, int n_expert,
                          const uint8_t* cache_base, const unsigned long long* slot_off, long long blob,
                          int32_t* __restrict__ pl, long long capx, uint32_t* skip, uint32_t ring,
                          int32_t* s_ids, unsigned long long* s_ptr, int32_t* s_first, int32_t* s_cnt,
                          int32_t* s_gstart, int* s_bad, sycl::nd_item<1> item, uint32_t* plan_err = nullptr) {
    const int tid = item.get_local_id(0);
    if (tid == 0) *s_bad = 0;
    item.barrier(sycl::access::fence_space::local_space);

    int32_t eid = -1;
    if (tid < n) {
        eid = ids[tid];
        s_ids[tid] = eid;
        const int32_t slot = (eid >= 0 && eid < n_expert) ? res[eid] : -1;
        if (slot < 0) {
            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                             sycl::access::address_space::local_space>(*s_bad).fetch_or(1);
        } else {
            s_ptr[tid] = (unsigned long long)(cache_base + (slot_off ? (size_t)slot_off[slot] : (size_t)slot * (size_t)blob));
        }
    }
    item.barrier(sycl::access::fence_space::local_space);
    if (*s_bad) {
        if (tid == 0) {
            if (skip != nullptr) *skip = 0;
            if (plan_err != nullptr) *plan_err = 1;
        }
        return;
    }

    int first_j = tid;
    int rank_in_group = 0;
    int count_same = 0;
    if (tid < n) {
        for (int j = 0; j < n; ++j) {
            if (s_ids[j] == eid) {
                if (j < first_j) first_j = j;
                if (j < tid) ++rank_in_group;
                ++count_same;
            }
        }
        const bool is_first = (first_j == tid);
        s_first[tid] = is_first ? 1 : 0;
        s_cnt[tid] = is_first ? count_same : 0;
    }
    item.barrier(sycl::access::fence_space::local_space);

    int32_t* counts = pl;
    int32_t* start = pl + 4;
    int32_t* dst = start + capx + 1;
    int32_t* tok = dst + capx;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    unsigned long long* ptr = (unsigned long long*)(pl + ptr_off);
    int32_t* start2 = pl + ptr_off + 4 * capx;

    if (tid < n && first_j == tid) {
        int grp_idx = 0;
        int ent_start = 0;
        for (int j = 0; j < tid; ++j) {
            grp_idx += s_first[j];
            ent_start += s_cnt[j];
        }
        ptr[grp_idx] = s_ptr[tid];
        start[grp_idx] = ent_start;
        s_gstart[tid] = ent_start;
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (tid < n) {
        const int out_idx = s_gstart[first_j] + rank_in_group;
        dst[out_idx] = tid;
        tok[out_idx] = tid / k;
    }
    if (tid == 0) {
        int groups = 0;
        for (int j = 0; j < n; ++j) groups += s_first[j];
        start[groups] = n;
        start2[0] = n;
        counts[0] = groups;
        counts[1] = n;
        counts[2] = 0;
        if (skip != nullptr) {
            sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
            *skip = ring;
        }
    }
}

void wait_flag_ge_kernel(const volatile uint32_t* flag, uint32_t value) {
    while (*flag < value)
        strata_spin_pause();
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}

void wait_flag_ge_or_kernel(const volatile uint32_t* flag, uint32_t value, const volatile uint32_t* skip) {
    if (*skip == value) return;
    while (*flag < value)
        strata_spin_pause();
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}

void copy_i32_unless_kernel(int32_t* __restrict__ dst, const volatile int32_t* src, int n, const uint32_t* skip,
                            uint32_t value, sycl::nd_item<1> item) {
    if (*skip == value) return;
    for (int i = item.get_local_id(0); i < n; i += item.get_local_range(0)) {
        dst[i] = src[i];
    }
}

void copy_or_zero_kernel(sycl::float4* __restrict__ dst, const sycl::float4* src, long long n4, const uint32_t* skip,
                         uint32_t value, sycl::nd_item<1> item) {
    const bool zero = *skip == value;
    for (long long i = item.get_global_id(0); i < n4; i += item.get_global_range(0)) {
        dst[i] = zero ? sycl::float4(0.f, 0.f, 0.f, 0.f) : src[i];
    }
}

void gpu_stamp_kernel(unsigned long long* buf, int i) {
    unsigned long long t = 0;
#if defined(__HIPCC__)
    t = wall_clock64() * 10ull;
#endif
    buf[i] = t;
}

}  // namespace

void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap,
                 void* stream) {
    if (cap <= 0) return;
    if (blob_bytes % 16 != 0) {
        std::fprintf(stderr, "fetch_blobs: blob size must be a multiple of 16\n");
        std::exit(1);
    }
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q->submit([&](sycl::handler& cgh) {
        const long long blob_bytes_16 = (long long)(blob_bytes / 16);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(48 * 8 * 256), sycl::range<1>(256)),
                         [=](sycl::nd_item<1> item) {
                             fetch_blobs_kernel(src, n, reinterpret_cast<sycl::uint4*>(dst), blob_bytes_16, item);
                         });
    });
}

void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* stream) {
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(128), sycl::range<1>(128)), [=](sycl::nd_item<1> item) {
        rebase_ptrs_kernel(ptr, n, (unsigned long long)base, (long long)blob_bytes, item);
    });
}

void add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    const size_t grid_x = ((size_t)(n_embd * hc) + 255) / 256;
    q->parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)n_tok, grid_x * 256), sycl::range<2>(1, 256)),
                    [=](sycl::nd_item<2> item) { add_streams_broadcast_kernel(h, e, R, n_embd, hc, item); });
}

void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream) {
    if (n < 1 || n > 1024) {
        std::fprintf(stderr, "ident_hits: n out of range\n");
        std::exit(1);
    }
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(1024), sycl::range<1>(1024)),
                    [=](sycl::nd_item<1> item) { ident_hits_kernel(ids, n, slot, dst, count, item); });
}

void mtp_select(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
                int32_t* tok_dst, int32_t* out, int j, void* stream, const float* probs, float* out_p) {
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(256), sycl::range<1>(256)), [=](sycl::nd_item<1> item) {
        mtp_select_kernel(R_src, R_stride, ids, row_dev, R_dst, tok_dst, out, j, probs, out_p, item);
    });
}

void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream) {
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    if (row_bytes % 16 == 0) {
        q->submit([&](sycl::handler& cgh) {
            const long long row_bytes_16 = row_bytes / 16;
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(48 * 8 * 256), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) {
                                 gather_rows_kernel(reinterpret_cast<const sycl::uint4*>(src), row_bytes_16, ids, n,
                                                    reinterpret_cast<sycl::uint4*>(dst), item);
                             });
        });
    } else if (row_bytes % 4 == 0) {
        q->submit([&](sycl::handler& cgh) {
            const long long row_bytes_4 = row_bytes / 4;
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(48 * 8 * 256), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) {
                                 gather_rows_kernel(reinterpret_cast<const uint32_t*>(src), row_bytes_4, ids, n,
                                                    reinterpret_cast<uint32_t*>(dst), item);
                             });
        });
    } else {
        q->parallel_for(sycl::nd_range<1>(sycl::range<1>(48 * 8 * 256), sycl::range<1>(256)),
                        [=](sycl::nd_item<1> item) { gather_rows_kernel(src, row_bytes, ids, n, dst, item); });
    }
}

void map_ids(int32_t* ids, const int32_t* table, int n, void* stream) {
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(64), sycl::range<1>(64)),
                    [=](sycl::nd_item<1> item) { map_ids_kernel(ids, table, n, item); });
}

void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream) {
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(32), cgh);
        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)n_rows, 1024), sycl::range<2>(1, 1024)),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
                             row_top_prob_kernel(logits, n_vocab, ids, probs, part, item);
                         });
    });
}

void window_ids(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, void* stream) {
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q->parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)n, 8 * 256), sycl::range<2>(1, 256)),
                    [=](sycl::nd_item<2> item) { window_ids_kernel(steps, window, ids, (long long)ids_stride, item); });
}

void dense_steps(const int32_t* cells, int n, int32_t* steps, void* stream) {
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(64), sycl::range<1>(64)),
                    [=](sycl::nd_item<1> item) { dense_steps_kernel(cells, n, steps, item); });
}

void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, int n_tok, void* stream, int t_begin, bool commit) {
    (void)commit;
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_conv_l2_multi: invalid arguments\n");
        std::exit(1);
    }
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(S / 32), cgh);
        const size_t qk_groups = channels / S;
        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)n_tok, qk_groups * S), sycl::range<2>(1, S)),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
                             gdn_conv_l2_multi_kernel(history, qkv, conv_w, h, channels, qk_heads, eps, t_begin,
                                                      part.get_multi_ptr<sycl::access::decorated::no>().get(), item);
                         });
    });
}

void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream) {
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    const size_t blocks = ((size_t)channels + 255) / 256;
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * 256), sycl::range<1>(256)),
                    [=](sycl::nd_item<1> item) { gdn_conv_commit_kernel(history, qkv, channels, n_keep, item); });
}

bool gdn_commit_layers(float* state, int64_t state_stride, int64_t conv_off, const float* qkv, int64_t qkv_stride,
                       const float* h, int64_t h_stride, const float* gate, const float* beta, int64_t gb_stride,
                       int conv_channels, int h_k, int h_v, int n_max, int n_layers, const int32_t* n_keep, void* stream) {
    static const bool per_layer = [] {
        const char* sp = std::getenv("STRATA_GDN_SPLIT");
        const char* cs = std::getenv("STRATA_GDN_COMMIT_SPLIT");
        const char* dbg = std::getenv("STRATA_DBG_GDN");
        return (sp && sp[0] == '1') || (cs && cs[0] == '0') || (dbg && dbg[0] == '1');
    }();
    if (per_layer || n_layers <= 0 || n_layers > 65535 || n_max < 1 || n_max > kVerifyMaxT || n_keep == nullptr ||
        h_v <= 0 || h_k <= 0 || h_v % h_k != 0)
        return false;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    const size_t conv_blocks = ((size_t)conv_channels + 255) / 256;
    q->parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)n_layers, conv_blocks * 256), sycl::range<2>(1, 256)),
                    [=](sycl::nd_item<2> item) {
                        gdn_conv_commit_layers_kernel(state + conv_off, state_stride, qkv, qkv_stride, conv_channels,
                                                      n_keep, item);
                    });
    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> sk(sycl::range<1>(S), cgh);
        sycl::local_accessor<float, 2> red(sycl::range<2>(RG, 32), cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>((size_t)n_layers, 4 * RG, (size_t)h_v * 32),
                                           sycl::range<3>(1, RG, 32)),
                         [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
                             const int n = *n_keep;
                             if (n <= 0) return;
                             const size_t z = item.get_group(0);
                             const int col_block = item.get_group(1);
                             const int head = item.get_group(2);
                             const int rg = item.get_local_id(1);
                             const int col_local = item.get_local_id(2);
                             gdn_step_commit_body(state + z * state_stride, h + z * h_stride,
                                                  conv_channels, gate + z * gb_stride,
                                                  beta + z * gb_stride, h_k, h_v, n, sk, red, head, col_block,
                                                  rg, col_local, item);
                         });
    });
    return true;
}

void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream) {
    if (n_embd % 8 != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_ab_multi: invalid arguments\n");
        std::exit(1);
    }
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    const size_t blocks = ((size_t)(2 * h_v) + 7) / 8;
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * 256), sycl::range<1>(256)),
                    [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                        gdn_ab_multi_kernel(x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok, item);
                    });
}

void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
                         const int32_t* n_keep, void* stream, int t_out_begin, void* xq_out) {
    (void)xq_out;
    if (!state || !h || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v % h_k || n_tok < 1 ||
        n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_step_norm_multi: invalid arguments\n");
        std::exit(1);
    }
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    static const bool commit_split = [] {
        const char* e = std::getenv("STRATA_GDN_COMMIT_SPLIT");
        return !e || e[0] != '0';
    }();
    if (commit_split && n_keep != nullptr && t_out_begin >= n_tok) {
        q->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> sk(sycl::range<1>(S), cgh);
            sycl::local_accessor<float, 2> red(sycl::range<2>(RG, 32), cgh);

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 4 * RG, (size_t)h_v * 32),
                                               sycl::range<3>(1, RG, 32)),
                             [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
                                 const int n = *n_keep;
                                 if (n <= 0) return;
                                 const int col_block = item.get_group(1);
                                 const int head = item.get_group(2);
                                 const int rg = item.get_local_id(1);
                                 const int col_local = item.get_local_id(2);
                                 gdn_step_commit_body(state, h, conv_channels, gate, beta, h_k, h_v, n, sk, red, head,
                                                      col_block, rg, col_local, item);
                             });
        });
        return;
    }
    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> sk(sycl::range<1>(S), cgh);
        sycl::local_accessor<float, 1> sq(sycl::range<1>(S), cgh);
        sycl::local_accessor<float, 2> red(sycl::range<2>(RG, S), cgh);
        sycl::local_accessor<float, 1> wsum(sycl::range<1>(S * RG / 32), cgh);

        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)h_v * RG, S), sycl::range<2>(RG, S)),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
                             gdn_step_norm_multi_kernel(state, h, conv_channels, gate, beta, z, gamma, eps, y, h_k, h_v,
                                                        n_tok, n_keep, t_out_begin, sk, sq, red, wsum, item);
                         });
    });
}

void resident_plan(const int32_t* ids, int n_entries, int k, const int32_t* res_layer, int n_expert,
                   const uint8_t* cache_base, const unsigned long long* slot_off, long long blob, int32_t* plan,
                   long long capx, uint32_t* skip, uint32_t ring, void* stream, uint32_t* plan_err) {
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int32_t, 1> s_ids(sycl::range<1>(128), cgh);
        sycl::local_accessor<unsigned long long, 1> s_ptr(sycl::range<1>(128), cgh);
        sycl::local_accessor<int32_t, 1> s_first(sycl::range<1>(128), cgh);
        sycl::local_accessor<int32_t, 1> s_cnt(sycl::range<1>(128), cgh);
        sycl::local_accessor<int32_t, 1> s_gstart(sycl::range<1>(128), cgh);
        sycl::local_accessor<int, 0> s_bad(cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(128), sycl::range<1>(128)),
                         [=](sycl::nd_item<1> item) {
                             resident_plan_kernel(ids, n_entries, k, res_layer, n_expert, cache_base, slot_off, blob,
                                                  plan, capx, skip, ring,
                                                  s_ids.get_multi_ptr<sycl::access::decorated::no>().get(),
                                                  s_ptr.get_multi_ptr<sycl::access::decorated::no>().get(),
                                                  s_first.get_multi_ptr<sycl::access::decorated::no>().get(),
                                                  s_cnt.get_multi_ptr<sycl::access::decorated::no>().get(),
                                                  s_gstart.get_multi_ptr<sycl::access::decorated::no>().get(),
                                                  s_bad.get_multi_ptr<sycl::access::decorated::no>().get(),
                                                  item, plan_err);
                         });
    });
}

void wait_flag_ge_or(const uint32_t* flag, uint32_t value, const uint32_t* skip, void* stream) {
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)),
                    [=](sycl::nd_item<1>) { wait_flag_ge_or_kernel(flag, value, skip); });
}

void copy_i32_from_mapped_unless(int32_t* dst, const int32_t* src, long long n, const uint32_t* skip, uint32_t value,
                                 void* stream) {
    if (n <= 0) return;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(128), sycl::range<1>(128)), [=](sycl::nd_item<1> item) {
        copy_i32_unless_kernel(dst, (const volatile int32_t*)src, (int)n, skip, value, item);
    });
}

void copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value,
                              void* stream) {
    if (n <= 0) return;
    const long long n4 = n / 4;
    const int blocks = (int)((n4 + 255) / 256 < 64 ? (n4 + 255) / 256 : 64);
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * 256), sycl::range<1>(256)), [=](sycl::nd_item<1> item) {
        copy_or_zero_kernel(reinterpret_cast<sycl::float4*>(dst), reinterpret_cast<const sycl::float4*>(src), n4, skip,
                            value, item);
    });
}

void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream) {
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)),
                    [=](sycl::nd_item<1>) { wait_flag_ge_kernel(flag, value); });
}

void embedding_gather_dev(const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens,
                          int n_tok, int64_t n, int code_bits, int code_bias, int group_elems, uint64_t row_codes,
                          uint64_t row_groups, float* out, void* stream) {
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    const size_t grid_x = ((size_t)n + 255) / 256;
    q->parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)n_tok, grid_x * 256), sycl::range<2>(1, 256)),
                    [=](sycl::nd_item<2> item) {
                        embedding_gather_dev_kernel(codes, scales, offsets, tokens, n, code_bits, code_bias,
                                                    group_elems, row_codes, row_groups, out, item);
                    });
}

void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    const size_t grid_x = ((size_t)(n_embd * hc) + 255) / 256;
    q->parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)n_tok, grid_x * 256), sycl::range<2>(1, 256)),
                    [=](sycl::nd_item<2> item) { broadcast_streams_kernel(x, R, n_embd, hc, item); });
}

void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream) {
    const unsigned blocks = (unsigned)((n + 255) / 256 < 64 ? (n + 255) / 256 : 64);
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * 256), sycl::range<1>(256)),
                    [=](sycl::nd_item<1> item) { copy_indexed_kernel(dst, src, stride, index, n, item); });
}

void gpu_stamp(unsigned long long* buf, int i, void* stream) {
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)),
                    [=](sycl::nd_item<1>) { gpu_stamp_kernel(buf, i); });
}

void gpu_noop(void* stream) {
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)), [=](sycl::nd_item<1>) {});
}

void copy_rows_strided(float* dst, const float* src, int64_t rows, int64_t w, int64_t src_w, void* stream) {
    if (rows <= 0 || w <= 0) return;
    if ((w & 3) || (src_w & 3) || src_w < w || ((uintptr_t)dst & 15) || ((uintptr_t)src & 15)) {
        std::fprintf(stderr, "copy_rows_strided: widths must be multiples of 4 floats (src_w >= w), pointers 16-byte aligned\n");
        std::exit(1);
    }
    const long long w4 = w / 4, src_w4 = src_w / 4;
    const long long total = (long long)rows * w4;
    const unsigned blocks = (unsigned)((total + 255) / 256 < 256 ? (total + 255) / 256 : 256);
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * 256), sycl::range<1>(256)), [=](sycl::nd_item<1> item) {
        for (long long i = (long long)item.get_global_id(0); i < total; i += (long long)item.get_global_range(0)) {
            const long long r = i / w4, c = i - r * w4;
            reinterpret_cast<sycl::float4*>(dst)[i] = reinterpret_cast<const sycl::float4*>(src)[r * src_w4 + c];
        }
    });
}

void wait_copy_i32_from_mapped(const uint32_t* flag, uint32_t value, int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q->parallel_for(sycl::nd_range<1>(sycl::range<1>(128), sycl::range<1>(128)), [=](sycl::nd_item<1> item) {
        if (item.get_local_id(0) == 0) {
            while (*flag < value) {}
        }
        item.barrier(sycl::access::fence_space::local_space);
        for (int i = (int)item.get_local_id(0); i < n; i += (int)item.get_local_range(0)) dst[i] = src[i];
    });
}

void gdn_conv_ab_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels, int qk_heads,
                       float eps, int n_tok, int t_begin, bool commit, const float* x, const uint16_t* w_alpha,
                       const uint16_t* w_beta, const float* dt, const float* ssm_a, float* gate, float* beta, int n_embd,
                       int h_v, void* stream) {
    gdn_conv_l2_multi(history, qkv, conv_w, h, channels, qk_heads, eps, n_tok, stream, t_begin, commit);
    gdn_ab_multi(x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok, stream);
}

namespace {
constexpr int kArgMaxBlocks = 128;
constexpr int kArgThreads = 256;
constexpr int kTopBlocks = 8;

struct ArgRow {
    uint32_t counter;
    uint32_t pad[3];
    float v[kArgMaxBlocks];
    int32_t i[kArgMaxBlocks];
};

struct TopRow {
    uint32_t counter;
    uint32_t pad[3];
    float parts[32];
};

inline void arg_take(float ov, int oi, float& bv, int& bi) {
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
}

inline void arg_block(float& bv, int& bi, int none, sycl::nd_item<2> item, float* sv, int* si) {
    auto sg = item.get_sub_group();
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = sycl::shift_group_left(sg, bv, off);
        const int oi = sycl::shift_group_left(sg, bi, off);
        arg_take(ov, oi, bv, bi);
    }
    const int warp = (int)(item.get_local_id(1) >> 5);
    const int lane = (int)(item.get_local_id(1) & 31);
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
    item.barrier(sycl::access::fence_space::local_space);
    if (warp == 0) {
        const int nw = (int)(item.get_local_range(1) >> 5);
        bv = lane < nw ? sv[lane] : -std::numeric_limits<float>::infinity();
        bi = lane < nw ? si[lane] : none;
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = sycl::shift_group_left(sg, bv, off);
            const int oi = sycl::shift_group_left(sg, bi, off);
            arg_take(ov, oi, bv, bi);
        }
    }
}
}  // namespace

uint64_t argmax_rows_scratch_bytes(int n_rows) {
    return (uint64_t)n_rows * sizeof(ArgRow);
}

void argmax_rows(const float* logits, int n_rows, int n, void* scratch, int32_t* out, void* stream) {
    if (n_rows <= 0) return;
    const int nb = std::min(kArgMaxBlocks, std::max(1, (n + 4095) / 4096));
    const int per_block = (n + nb - 1) / nb;
    ArgRow* rows = static_cast<ArgRow*>(scratch);
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> sv(sycl::range<1>(32), cgh);
        sycl::local_accessor<int, 1> si(sycl::range<1>(32), cgh);
        sycl::local_accessor<bool, 0> last(cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t)n_rows, (size_t)nb * kArgThreads),
                              sycl::range<2>(1, kArgThreads)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
                const int row = (int)item.get_group(0);
                const int b = (int)item.get_group(1);
                const int tid = (int)item.get_local_id(1);
                float* pv = rows[row].v;
                int* pi = rows[row].i;
                const float* l = logits + (size_t)row * n;
                const int lo = b * per_block, hi = sycl::min(n, lo + per_block);
                float bv = -std::numeric_limits<float>::infinity();
                int bi = n;
                for (int v = lo + tid; v < hi; v += kArgThreads) {
                    const float s = l[v];
                    if (s > bv) { bv = s; bi = v; }
                }
                arg_block(bv, bi, n, item, sv.get_multi_ptr<sycl::access::decorated::no>().get(),
                          si.get_multi_ptr<sycl::access::decorated::no>().get());
                if (tid == 0) {
                    pv[b] = bv;
                    pi[b] = bi;
                    sycl::atomic_ref<uint32_t, sycl::memory_order::seq_cst, sycl::memory_scope::device> cnt(rows[row].counter);
                    last = (cnt.fetch_add(1u) == (uint32_t)(nb - 1));
                }
                item.barrier(sycl::access::fence_space::local_space);
                if (!last) return;
                bv = -std::numeric_limits<float>::infinity();
                bi = n;
                if (tid < nb) {
                    bv = pv[tid];
                    bi = pi[tid];
                }
                arg_block(bv, bi, n, item, sv.get_multi_ptr<sycl::access::decorated::no>().get(),
                          si.get_multi_ptr<sycl::access::decorated::no>().get());
                if (tid == 0) {
                    out[row] = bi < n ? bi : 0;
                    rows[row].counter = 0;
                }
            });
    });
}

uint64_t row_top_prob_scratch_bytes(int n_rows) {
    return (uint64_t)n_rows * sizeof(TopRow);
}

void row_top_prob_split(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* scratch,
                        void* stream) {
    if (n_rows <= 0) return;
    TopRow* rows = static_cast<TopRow*>(scratch);
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<bool, 0> last(cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t)n_rows, (size_t)kTopBlocks * 128),
                              sycl::range<2>(1, 128)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
                const int row = (int)item.get_group(0);
                const int b = (int)item.get_group(1);
                const int tid = (int)item.get_local_id(1);
                float* parts = rows[row].parts;
                const float* l = logits + (size_t)row * n_vocab;
                const float m = l[ids[row]];
                const int vt = b * 128 + tid;
                float s = 0.0f;
                for (int i = vt; i < n_vocab; i += 1024) s += sycl::native::exp(l[i] - m);
                auto sg = item.get_sub_group();
                s = sycl::reduce_over_group(sg, s, sycl::plus<float>());
                if ((tid & 31) == 0) parts[vt >> 5] = s;
                item.barrier(sycl::access::fence_space::local_space);
                if (tid == 0) {
                    sycl::atomic_ref<uint32_t, sycl::memory_order::seq_cst, sycl::memory_scope::device> cnt(rows[row].counter);
                    last = (cnt.fetch_add(1u) == (uint32_t)(kTopBlocks - 1));
                }
                item.barrier(sycl::access::fence_space::local_space);
                if (!last || tid != 0) return;
                float tot = 0.0f;
                for (int w = 0; w < 32; ++w) tot += parts[w];
                probs[row] = 1.0f / tot;
                rows[row].counter = 0;
            });
    });
}

void logits_nonfinite_rows(const float* logits, int n_rows, int n, uint32_t* flags, void* stream) {
    if (n_rows <= 0 || n <= 0) return;
    const int per_block = (n + kNonfiniteBlocks - 1) / kNonfiniteBlocks;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> s_bad(sycl::range<1>(8), cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t)n_rows, (size_t)kNonfiniteBlocks * 256),
                              sycl::range<2>(1, 256)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
                const int row = (int)item.get_group(0);
                const int b = (int)item.get_group(1);
                const int tid = (int)item.get_local_id(1);
                const float* l = logits + (size_t)row * n;
                const int lo = b * per_block, hi = sycl::min(n, lo + per_block);
                int bad = 0;
                for (int v = lo + tid; v < hi; v += 256) {
                    uint32_t u = sycl::bit_cast<uint32_t>(l[v]);
                    bad |= (u & 0x7f800000u) == 0x7f800000u;
                }
                auto sg = item.get_sub_group();
                bad = sycl::reduce_over_group(sg, bad, sycl::bit_or<int>());
                const int warp = tid >> 5;
                const int lane = tid & 31;
                if (lane == 0) s_bad[warp] = bad;
                item.barrier(sycl::access::fence_space::local_space);
                if (warp == 0) {
                    int wbad = lane < 8 ? s_bad[lane] : 0;
                    wbad = sycl::reduce_over_group(sg, wbad, sycl::bit_or<int>());
                    if (lane == 0) flags[row * kNonfiniteBlocks + b] = wbad ? 1u : 0u;
                }
            });
    });
}

bool multi_block_head_ops() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_MULTI_BLOCK_ARGMAX");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}

bool argmax_rows_wanted() {
    return multi_block_head_ops();
}

}  // namespace strata::kernels
