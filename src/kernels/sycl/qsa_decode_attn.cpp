// src/kernels/sycl/qsa_decode_attn.dp.cpp - Pure SYCL 2020 migration.
#include <sycl/sycl.hpp>
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace strata::kernels {
namespace {

inline sycl::queue& get_default_in_order_queue() {
    static sycl::queue q(sycl::property::queue::in_order{});
    return q;
}

constexpr int HD = 256;    // head_dim
constexpr int G = 12;      // query heads per KV head (24 / 2)
constexpr int CHUNK = 64;  // cells per block
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;

inline float warp_sum(sycl::sub_group sg, float v) {
    return sycl::reduce_over_group(sg, v, sycl::plus<float>());
}

inline float warp_max(sycl::sub_group sg, float v) {
    return sycl::reduce_over_group(sg, v, sycl::maximum<float>());
}

inline void load8_f16(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    const uint16_t* base = (value ? p.v_pool : p.k_pool) + row * HD + d0;
    const sycl::uint4 raw = *reinterpret_cast<const sycl::uint4*>(base);
    const sycl::half2* h2 = reinterpret_cast<const sycl::half2*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const sycl::float2 f = h2[j].template convert<float, sycl::rounding_mode::automatic>();
        out[2 * j] = f.x();
        out[2 * j + 1] = f.y();
    }
}

inline void load8_q8(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    const int8_t* codes = (value ? p.v_q : p.k_q) + row * HD + d0;
    const uint16_t sbits = (value ? p.v_scale : p.k_scale)[row * (HD / KV_Q8_GROUP) + d0 / KV_Q8_GROUP];
    const float sc = sycl::vec<sycl::half, 1>(sycl::bit_cast<sycl::half, unsigned short>(sbits))
                         .convert<float, sycl::rounding_mode::automatic>()[0];
    const sycl::uint2 raw = *reinterpret_cast<const sycl::uint2*>(codes);
    const int8_t* c = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
    for (int j = 0; j < 8; ++j)
        out[j] = (float)c[j] * sc;
}

inline void load8_q4(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
    const int b = d0 / QK4_0;
    const int rem = d0 % QK4_0;
    const block_q4_0* blk = reinterpret_cast<const block_q4_0*>((value ? p.v_q4 : p.k_q4) + row * bytes_per_head) + b;
    const float d = sycl::vec<sycl::half, 1>(sycl::bit_cast<sycl::half, unsigned short>(blk->d))
                        .convert<float, sycl::rounding_mode::automatic>()[0];
    const int j = (rem == 0 || rem == 16) ? 0 : 8;
    const uint8_t* bytes = blk->qs + j;
    if (rem < 16) {
#pragma unroll
        for (int k = 0; k < 8; ++k)
            out[k] = (float)((int)(bytes[k] & 0x0F) - 8) * d;
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k)
            out[k] = (float)((int)(bytes[k] >> 4) - 8) * d;
    }
}

template <int KV_MODE>
inline void load8(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    if constexpr (KV_MODE == 0)
        load8_f16(p, value, row, d0, out);
    else if constexpr (KV_MODE == 1)
        load8_q8(p, value, row, d0, out);
    else if constexpr (KV_MODE == 3) {
        if (value)
            load8_q4(p, value, row, d0, out);
        else
            load8_q8(p, value, row, d0, out);
    } else
        load8_q4(p, value, row, d0, out);
}

template <int KV_MODE, bool LANE_CELL = false>
void attn_chunk_kernel(sycl::nd_item<3> item, const float* __restrict__ q, QsaAttnPools p,
                       const int32_t* __restrict__ ids, const int32_t* __restrict__ step, int n_kv_heads, int page_size,
                       float scale, float* __restrict__ part_acc, float* __restrict__ part_m,
                       float* __restrict__ part_l, int n_chunks, float sq[G][HD], float sp[G][CHUNK], long long* srow,
                       int cap = 0, long long scratch_stride = 0) {
    const size_t q_idx = item.get_group(0);
    q += q_idx * (size_t)(n_kv_heads * G) * HD;
    ids += q_idx * (size_t)cap;
    step += q_idx * kStepCount;
    part_acc += q_idx * (size_t)scratch_stride;
    part_m += q_idx * (size_t)scratch_stride;
    part_l += q_idx * (size_t)scratch_stride;

    const int n_ids = *(step + kStepWidth);
    const int chunk = (int)item.get_group(2), kvh = (int)item.get_group(1);
    const int t = (int)item.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = sycl::min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) {
            part_m[slot * G + t] = -FLT_MAX;
            part_l[slot * G + t] = 0.0f;
        }
        return;
    }
    for (int i = t; i < G * HD; i += THREADS)
        sq[i / HD][i % HD] = q[(size_t)(kvh * G) * HD + i];
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long)p.page_table[cell / page_size];
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    sycl::group_barrier(item.get_group());
    auto sg = item.get_sub_group();

    if constexpr (LANE_CELL) {
        // S25: STRATA_ATTN_LANECELL=1 - thread t scores cell t % 64 for heads 3 (t / 64) .. +2 by itself.
        constexpr int HPT = G * CHUNK / THREADS;  // heads per thread (3)
        const int c = t % CHUNK, h0 = (t / CHUNK) * HPT;
        if (c >= n_here || srow[c] < 0) {
#pragma unroll
            for (int j = 0; j < HPT; ++j) sp[h0 + j][c] = -FLT_MAX;
        } else {
            float a[HPT][32];
#pragma unroll
            for (int l = 0; l < 32; ++l) {
                float k8[8];
                load8<KV_MODE>(p, false, srow[c], l * 8, k8);
#pragma unroll
                for (int j = 0; j < HPT; ++j) {
                    const sycl::float4 qa = *reinterpret_cast<const sycl::float4*>(&sq[h0 + j][l * 8]);
                    const sycl::float4 qb = *reinterpret_cast<const sycl::float4*>(&sq[h0 + j][l * 8 + 4]);
                    float s = k8[0] * qa.x() + k8[1] * qa.y() + k8[2] * qa.z() + k8[3] * qa.w() +
                              k8[4] * qb.x() + k8[5] * qb.y() + k8[6] * qb.z() + k8[7] * qb.w();
                    a[j][l] = s;
                }
            }
#pragma unroll
            for (int j = 0; j < HPT; ++j) {
#pragma unroll
                for (int o = 16; o > 0; o >>= 1)
#pragma unroll
                    for (int l = 0; l < o; ++l) a[j][l] = a[j][l] + a[j][l + o];
                sp[h0 + j][c] = a[j][0] * scale;
            }
        }
    } else {
        for (int c = warp; c < CHUNK; c += WARPS) {
            if (c >= n_here || srow[c] < 0) {
                if (lane < G) sp[lane][c] = -FLT_MAX;
                continue;
            }
            float k8[8];
            load8<KV_MODE>(p, false, srow[c], lane * 8, k8);
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const sycl::float4 qa = *reinterpret_cast<const sycl::float4*>(&sq[h][lane * 8]);
                const sycl::float4 qb = *reinterpret_cast<const sycl::float4*>(&sq[h][lane * 8 + 4]);
                float s = k8[0] * qa.x() + k8[1] * qa.y() + k8[2] * qa.z() + k8[3] * qa.w() + k8[4] * qb.x() +
                          k8[5] * qb.y() + k8[6] * qb.z() + k8[7] * qb.w();
                s = warp_sum(sg, s);
                if (lane == 0) sp[h][c] = s * scale;
            }
        }
    }
    sycl::group_barrier(item.get_group());

    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max(sg, sycl::fmax(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? sycl::native::exp(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? sycl::native::exp(b - m) : 0.0f;
        sp[h][lane] = ea;
        sp[h][lane + 32] = eb;
        const float l = warp_sum(sg, ea + eb);
        if (lane == 0) {
            part_m[slot * G + h] = m;
            part_l[slot * G + h] = l;
        }
    }
    sycl::group_barrier(item.get_group());

    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h)
        acc[h] = 0.0f;
    for (int c = 0; c < n_here; ++c) {
        if (srow[c] < 0) continue;
        float v;
        if constexpr (KV_MODE == 0) {
            v = sycl::vec<sycl::half, 1>(sycl::bit_cast<sycl::half, unsigned short>(p.v_pool[srow[c] * HD + t]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
        } else if constexpr (KV_MODE == 1) {
            const float sc = sycl::vec<sycl::half, 1>(sycl::bit_cast<sycl::half, unsigned short>(
                                                          p.v_scale[srow[c] * (HD / KV_Q8_GROUP) + t / KV_Q8_GROUP]))
                                 .convert<float, sycl::rounding_mode::automatic>()[0];
            v = (float)p.v_q[srow[c] * HD + t] * sc;
        } else {
            constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
            const int b = t / QK4_0;
            const int rem = t % QK4_0;
            const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + srow[c] * bytes_per_head) + b;
            const float d = sycl::vec<sycl::half, 1>(sycl::bit_cast<sycl::half, unsigned short>(blk->d))
                                .convert<float, sycl::rounding_mode::automatic>()[0];
            const int j = rem < 16 ? rem : (rem - 16);
            const uint8_t byte = blk->qs[j];
            const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
            v = (float)nibble * d;
        }
#pragma unroll
        for (int h = 0; h < G; ++h)
            acc[h] = sycl::fma(sp[h][c], v, acc[h]);
    }
#pragma unroll
    for (int h = 0; h < G; ++h)
        part_acc[((size_t)slot * G + h) * HD + t] = acc[h];
}

void attn_merge_kernel(sycl::nd_item<2> item, const float* __restrict__ part_acc, const float* __restrict__ part_m,
                       const float* __restrict__ part_l, int n_chunks, float* __restrict__ attn, int n_heads,
                       long long scratch_stride) {
    const size_t q_idx = item.get_group(0);
    part_acc += q_idx * (size_t)scratch_stride;
    part_m += q_idx * (size_t)scratch_stride;
    part_l += q_idx * (size_t)scratch_stride;
    attn += q_idx * (size_t)n_heads * HD;

    const int h = (int)item.get_group(1);  // global query head
    const int kvh = h / G, hl = h % G;
    const int d = (int)item.get_local_id(1);

    float M = -FLT_MAX;
    for (int c = 0; c < n_chunks; ++c)
        M = sycl::fmax(M, part_m[(kvh * n_chunks + c) * G + hl]);
    float L = 0.0f, acc = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const int slot = kvh * n_chunks + c;
        const float m = part_m[slot * G + hl];
        if (m == -FLT_MAX) continue;
        const float w = sycl::native::exp(m - M);
        L = sycl::fma(part_l[slot * G + hl], w, L);
        acc = sycl::fma(part_acc[((size_t)slot * G + hl) * HD + d], w, acc);
    }
    attn[(size_t)h * HD + d] = L > 0.0f ? acc / L : 0.0f;
}

template <int KV_MODE, bool LANE_CELL = false>
void launch_attn_chunk(sycl::queue* st, const float* q, const QsaAttnPools& pools, const int32_t* ids,
                       const int32_t* steps, int n_kv_heads, int page_size, float scale, float* part_acc, float* part_m,
                       float* part_l, int n_chunks, int cap, long long stride, int n_q) {
    sycl::range<3> global_range(n_q, n_kv_heads, (size_t)n_chunks * THREADS);
    sycl::range<3> local_range(1, 1, THREADS);
    st->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float[G][HD], 0> sq_acc(cgh);
        sycl::local_accessor<float[G][CHUNK], 0> sp_acc(cgh);
        sycl::local_accessor<long long, 1> srow_acc(sycl::range<1>(CHUNK), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(global_range, local_range), [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
                attn_chunk_kernel<KV_MODE, LANE_CELL>(item, q, pools, ids, steps, n_kv_heads, page_size, scale, part_acc, part_m,
                                           part_l, n_chunks, sq_acc, sp_acc,
                                           srow_acc.get_multi_ptr<sycl::access::decorated::no>().get(), cap, stride);
            });
    });
}

}  // namespace

void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    if (s.head_dim != HD || s.n_head != (int64_t)G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps ||
        !pools.page_table || n_q > 65535) {
        std::fprintf(stderr, "qsa_decode_attn_batch: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = pools.k_q4 != nullptr
                            ? 2
                            : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3 : (pools.k_q != nullptr ? 1 : 0));
    const int n_chunks = (int)((cap + CHUNK - 1) / CHUNK);
    const long long stride = (long long)qsa_decode_attn_scratch_floats(cap, s);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t)n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t)n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float)HD);
    sycl::queue* st = stream ? static_cast<sycl::queue*>(stream) : &get_default_in_order_queue();

    static const bool lane_cell = [] {
        const char* v = std::getenv("STRATA_ATTN_LANECELL");
        return v && v[0] == '1';
    }();

    if (lane_cell) {
        if (kv_mode == 3) {
            launch_attn_chunk<3, true>(st, q, pools, ids, steps, (int)s.n_head_kv, (int)s.page_size, scale, part_acc, part_m,
                                      part_l, n_chunks, (int)cap, stride, (int)n_q);
        } else if (kv_mode == 2) {
            launch_attn_chunk<2, true>(st, q, pools, ids, steps, (int)s.n_head_kv, (int)s.page_size, scale, part_acc, part_m,
                                      part_l, n_chunks, (int)cap, stride, (int)n_q);
        } else if (kv_mode == 1) {
            launch_attn_chunk<1, true>(st, q, pools, ids, steps, (int)s.n_head_kv, (int)s.page_size, scale, part_acc, part_m,
                                      part_l, n_chunks, (int)cap, stride, (int)n_q);
        } else {
            launch_attn_chunk<0, true>(st, q, pools, ids, steps, (int)s.n_head_kv, (int)s.page_size, scale, part_acc, part_m,
                                      part_l, n_chunks, (int)cap, stride, (int)n_q);
        }
    } else {
        if (kv_mode == 3) {
            launch_attn_chunk<3, false>(st, q, pools, ids, steps, (int)s.n_head_kv, (int)s.page_size, scale, part_acc, part_m,
                                       part_l, n_chunks, (int)cap, stride, (int)n_q);
        } else if (kv_mode == 2) {
            launch_attn_chunk<2, false>(st, q, pools, ids, steps, (int)s.n_head_kv, (int)s.page_size, scale, part_acc, part_m,
                                       part_l, n_chunks, (int)cap, stride, (int)n_q);
        } else if (kv_mode == 1) {
            launch_attn_chunk<1, false>(st, q, pools, ids, steps, (int)s.n_head_kv, (int)s.page_size, scale, part_acc, part_m,
                                       part_l, n_chunks, (int)cap, stride, (int)n_q);
        } else {
            launch_attn_chunk<0, false>(st, q, pools, ids, steps, (int)s.n_head_kv, (int)s.page_size, scale, part_acc, part_m,
                                       part_l, n_chunks, (int)cap, stride, (int)n_q);
        }
    }

    st->parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)n_q, (size_t)s.n_head * HD), sycl::range<2>(1, HD)),
                     [=](sycl::nd_item<2> item) {
                         attn_merge_kernel(item, part_acc, part_m, part_l, n_chunks, attn, (int)s.n_head, stride);
                     });

    if (!stream) st->wait();
}

uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    const int64_t chunks = (cap + CHUNK - 1) / CHUNK;
    return (uint64_t)chunks * (uint64_t)s.n_head * (HD + 2) + 64;
}

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream) {
    qsa_decode_attn_batch(q, pools, ids, step, cap, s, scratch, attn, 1, stream);
}

}  // namespace strata::kernels
