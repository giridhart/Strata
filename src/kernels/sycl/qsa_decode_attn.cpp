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
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
    return v;
}

inline float warp_max(sycl::sub_group sg, float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = sycl::fmax(v, sycl::permute_group_by_xor(sg, v, o));
    return v;
}

inline float reduce12(sycl::sub_group sg, const float (&part)[16], int lane) {
    float r8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const bool hi = (lane & 16) != 0;
        r8[i] = (hi ? part[i + 8] : part[i]) + sycl::permute_group_by_xor(sg, hi ? part[i] : part[i + 8], 16);
    }
    float r4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bool hi = (lane & 8) != 0;
        r4[i] = (hi ? r8[i + 4] : r8[i]) + sycl::permute_group_by_xor(sg, hi ? r8[i] : r8[i + 4], 8);
    }
    float r2[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool hi = (lane & 4) != 0;
        r2[i] = (hi ? r4[i + 2] : r4[i]) + sycl::permute_group_by_xor(sg, hi ? r4[i] : r4[i + 2], 4);
    }
    const bool hi2 = (lane & 2) != 0;
    const float r1 = (hi2 ? r2[1] : r2[0]) + sycl::permute_group_by_xor(sg, hi2 ? r2[0] : r2[1], 2);
    return r1 + sycl::permute_group_by_xor(sg, r1, 1);
}

inline int head_of_lane(int lane) {
    return ((lane >> 4) & 1) * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1);
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

template <int KV_MODE>
inline float load_v1(const QsaAttnPools& p, long long row, int d) {
    if constexpr (KV_MODE == 0) {
        return sycl::vec<sycl::half, 1>(sycl::bit_cast<sycl::half, unsigned short>(p.v_pool[row * HD + d]))
            .convert<float, sycl::rounding_mode::automatic>()[0];
    } else if constexpr (KV_MODE == 1) {
        const float sc = sycl::vec<sycl::half, 1>(sycl::bit_cast<sycl::half, unsigned short>(
                                                      p.v_scale[row * (HD / KV_Q8_GROUP) + d / KV_Q8_GROUP]))
                             .convert<float, sycl::rounding_mode::automatic>()[0];
        return (float)p.v_q[row * HD + d] * sc;
    } else {
        constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
        const int b = d / QK4_0;
        const int rem = d % QK4_0;
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + row * bytes_per_head) + b;
        const float dd = sycl::vec<sycl::half, 1>(sycl::bit_cast<sycl::half, unsigned short>(blk->d))
                             .convert<float, sycl::rounding_mode::automatic>()[0];
        const int j = rem < 16 ? rem : (rem - 16);
        const uint8_t byte = blk->qs[j];
        const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
        return (float)nibble * dd;
    }
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
        const float v = load_v1<KV_MODE>(p, srow[c], t);
#pragma unroll
        for (int h = 0; h < G; ++h)
            acc[h] = sycl::fma(sp[h][c], v, acc[h]);
    }
#pragma unroll
    for (int h = 0; h < G; ++h)
        part_acc[((size_t)slot * G + h) * HD + t] = acc[h];
}

template <int KV_MODE, int NC_ = 2>
void attn_chunk_kernel_pre75(sycl::nd_item<3> item, const float* __restrict__ q, QsaAttnPools p,
                             const int32_t* __restrict__ ids, const int32_t* __restrict__ step, int n_kv_heads,
                             int page_size, float scale, float* __restrict__ part_acc, float* __restrict__ part_m,
                             float* __restrict__ part_l, int n_chunks, float sq[G][HD], float sp[G][CHUNK],
                             long long* srow, int cap = 0, long long scratch_stride = 0) {
    const size_t q_idx = item.get_group(0);
    q += q_idx * (size_t)(n_kv_heads * G) * HD;
    ids += q_idx * (size_t)cap;
    step += q_idx * kStepCount;
    part_acc += q_idx * (size_t)scratch_stride;
    part_m += q_idx * (size_t)scratch_stride;
    part_l += q_idx * (size_t)scratch_stride;

    float (*const spt)[G] = reinterpret_cast<float (*)[G]>(&sq[0][0]);
    static_assert(CHUNK * G <= G * HD, "spt must fit in sq");

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

    constexpr int NC = NC_;
    const int hd = head_of_lane(lane);
    for (int i = 0; i < CHUNK / WARPS; i += NC) {
        int cc[NC];
        bool all_ok = true;
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            cc[j] = warp + WARPS * (i + j);
            all_ok = all_ok && cc[j] < n_here && srow[cc[j]] >= 0;
        }
        if (all_ok) {
            float kk[NC][8];
#pragma unroll
            for (int j = 0; j < NC; ++j) load8<KV_MODE>(p, false, srow[cc[j]], lane * 8, kk[j]);
            float pp[NC][16];
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const sycl::float4 qa = *reinterpret_cast<const sycl::float4*>(&sq[h][lane * 8]);
                const sycl::float4 qb = *reinterpret_cast<const sycl::float4*>(&sq[h][lane * 8 + 4]);
#pragma unroll
                for (int j = 0; j < NC; ++j)
                    pp[j][h] = kk[j][0] * qa.x() + kk[j][1] * qa.y() + kk[j][2] * qa.z() + kk[j][3] * qa.w() +
                               kk[j][4] * qb.x() + kk[j][5] * qb.y() + kk[j][6] * qb.z() + kk[j][7] * qb.w();
            }
#pragma unroll
            for (int j = 0; j < NC; ++j) {
#pragma unroll
                for (int h = G; h < 16; ++h) pp[j][h] = 0.0f;
                const float sj = reduce12(sg, pp[j], lane);
                if ((lane & 1) == 0 && hd < G) sp[hd][cc[j]] = sj * scale;
            }
        } else {
            for (int k = 0; k < NC; ++k) {
                const int c = cc[k];
                if (c >= n_here || srow[c] < 0) {
                    if (lane < G) sp[lane][c] = -FLT_MAX;
                    continue;
                }
                float k8[8];
                load8<KV_MODE>(p, false, srow[c], lane * 8, k8);
                float part[16];
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    const sycl::float4 qa = *reinterpret_cast<const sycl::float4*>(&sq[h][lane * 8]);
                    const sycl::float4 qb = *reinterpret_cast<const sycl::float4*>(&sq[h][lane * 8 + 4]);
                    part[h] = k8[0] * qa.x() + k8[1] * qa.y() + k8[2] * qa.z() + k8[3] * qa.w() +
                              k8[4] * qb.x() + k8[5] * qb.y() + k8[6] * qb.z() + k8[7] * qb.w();
                }
#pragma unroll
                for (int h = G; h < 16; ++h) part[h] = 0.0f;
                const float s = reduce12(sg, part, lane);
                if ((lane & 1) == 0 && hd < G) sp[hd][c] = s * scale;
            }
        }
    }
    sycl::group_barrier(item.get_group());

    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max(sg, sycl::fmax(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? sycl::native::exp(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? sycl::native::exp(b - m) : 0.0f;
        spt[lane][h] = ea;
        spt[lane + 32][h] = eb;
        const float l = warp_sum(sg, ea + eb);
        if (lane == 0) {
            part_m[slot * G + h] = m;
            part_l[slot * G + h] = l;
        }
    }
    sycl::group_barrier(item.get_group());

    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    const auto fold = [&](int c, float v) {
        const sycl::float4* w4 = reinterpret_cast<const sycl::float4*>(spt[c]);
        const sycl::float4 w0 = w4[0], w1 = w4[1], w2 = w4[2];
        acc[0] = sycl::fma(w0.x(), v, acc[0]);   acc[1] = sycl::fma(w0.y(), v, acc[1]);
        acc[2] = sycl::fma(w0.z(), v, acc[2]);   acc[3] = sycl::fma(w0.w(), v, acc[3]);
        acc[4] = sycl::fma(w1.x(), v, acc[4]);   acc[5] = sycl::fma(w1.y(), v, acc[5]);
        acc[6] = sycl::fma(w1.z(), v, acc[6]);   acc[7] = sycl::fma(w1.w(), v, acc[7]);
        acc[8] = sycl::fma(w2.x(), v, acc[8]);   acc[9] = sycl::fma(w2.y(), v, acc[9]);
        acc[10] = sycl::fma(w2.z(), v, acc[10]); acc[11] = sycl::fma(w2.w(), v, acc[11]);
    };
    int c = 0;
    for (; c + 4 <= n_here; c += 4) {
        const long long r0 = srow[c], r1 = srow[c + 1], r2 = srow[c + 2], r3 = srow[c + 3];
        if ((r0 | r1 | r2 | r3) < 0) break;
        const float v0 = load_v1<KV_MODE>(p, r0, t), v1 = load_v1<KV_MODE>(p, r1, t);
        const float v2 = load_v1<KV_MODE>(p, r2, t), v3 = load_v1<KV_MODE>(p, r3, t);
        fold(c, v0);
        fold(c + 1, v1);
        fold(c + 2, v2);
        fold(c + 3, v3);
    }
    for (; c < n_here; ++c) {
        if (srow[c] < 0) continue;
        fold(c, load_v1<KV_MODE>(p, srow[c], t));
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

constexpr int kMergeU = 16;

template <int U, bool TAIL>
inline void merge_max_block(const float* __restrict__ pm, int c0, int n_act, float& M) {
    float mm[U];
#pragma unroll
    for (int u = 0; u < U; ++u) mm[u] = (!TAIL || c0 + u < n_act) ? pm[(size_t)(c0 + u) * G] : -FLT_MAX;
#pragma unroll
    for (int u = 0; u < U; ++u)
        if (!TAIL || c0 + u < n_act) M = sycl::fmax(M, mm[u]);
}

template <int U, bool TAIL>
inline void merge_sum_block(const float* __restrict__ pm, const float* __restrict__ pl,
                            const float* __restrict__ pa, int c0, int n_act, float M, float& L,
                            float& acc) {
    float mm[U], ll[U], aa[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
        const bool in = !TAIL || c0 + u < n_act;
        mm[u] = in ? pm[(size_t)(c0 + u) * G] : -FLT_MAX;
        ll[u] = in ? pl[(size_t)(c0 + u) * G] : 0.0f;
        aa[u] = in ? pa[(size_t)(c0 + u) * G * HD] : 0.0f;
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
        const float w = sycl::native::exp(mm[u] - M);
        const float L_next = sycl::fma(ll[u], w, L);
        const float acc_next = sycl::fma(aa[u], w, acc);
        const bool live = !(mm[u] == -FLT_MAX);
        L = live ? L_next : L;
        acc = live ? acc_next : acc;
    }
}

template <int U>
void attn_merge_v2_kernel(sycl::nd_item<2> item, const float* __restrict__ part_acc,
                          const float* __restrict__ part_m, const float* __restrict__ part_l, int n_chunks,
                          const int32_t* __restrict__ steps, float* __restrict__ attn, int n_heads,
                          long long scratch_stride) {
    const size_t q_idx = item.get_group(0);
    part_acc += q_idx * (size_t)scratch_stride;
    part_m += q_idx * (size_t)scratch_stride;
    part_l += q_idx * (size_t)scratch_stride;
    attn += q_idx * (size_t)n_heads * HD;

    const int width = *(steps + q_idx * kStepCount + kStepWidth);
    const int n_act = width > 0 ? sycl::min(n_chunks, (width - 1) / CHUNK + 1) : 0;
    const int h = (int)item.get_group(1);
    const int kvh = h / G, hl = h % G;
    const int d = (int)item.get_local_id(1);
    const size_t base = (size_t)kvh * (size_t)n_chunks * G + (size_t)hl;
    const float* pm = part_m + base;
    const float* pl = part_l + base;
    const float* pa = part_acc + base * HD + (size_t)d;
    const int n_full = n_act - n_act % U;
    float M = -FLT_MAX;
    for (int c0 = 0; c0 < n_full; c0 += U) merge_max_block<U, false>(pm, c0, n_act, M);
    if (n_full < n_act) merge_max_block<U, true>(pm, n_full, n_act, M);
    float L = 0.0f, acc = 0.0f;
    for (int c0 = 0; c0 < n_full; c0 += U) merge_sum_block<U, false>(pm, pl, pa, c0, n_act, M, L, acc);
    if (n_full < n_act) merge_sum_block<U, true>(pm, pl, pa, n_full, n_act, M, L, acc);
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

template <int KV_MODE>
void launch_attn_chunk_pre75(sycl::queue* st, const float* q, const QsaAttnPools& pools, const int32_t* ids,
                             const int32_t* steps, int n_kv_heads, int page_size, float scale, float* part_acc,
                             float* part_m, float* part_l, int n_chunks, int cap, long long stride, int n_q) {
    sycl::range<3> global_range(n_q, n_kv_heads, (size_t)n_chunks * THREADS);
    sycl::range<3> local_range(1, 1, THREADS);
    st->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<sycl::float4, 1> sq_acc(sycl::range<1>(G * HD / 4), cgh);
        sycl::local_accessor<float[G][CHUNK], 0> sp_acc(cgh);
        sycl::local_accessor<long long, 1> srow_acc(sycl::range<1>(CHUNK), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(global_range, local_range), [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
                auto* sq = reinterpret_cast<float (*)[HD]>(sq_acc.get_multi_ptr<sycl::access::decorated::no>().get());
                attn_chunk_kernel_pre75<KV_MODE>(item, q, pools, ids, steps, n_kv_heads, page_size, scale, part_acc,
                                                 part_m, part_l, n_chunks, sq, sp_acc,
                                                 srow_acc.get_multi_ptr<sycl::access::decorated::no>().get(), cap, stride);
            });
    });
}

bool daily_attn() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_ATTN_DAILY");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

bool pre75_attn() {
    static const int env = [] {
        const char* e = std::getenv("STRATA_ATTN_PRE75");
        return e == nullptr ? -1 : e[0] == '0' ? 0 : e[0] == '1' ? 1 : -1;
    }();
    if (env == 0) return false;
    if (env == 1) return true;
    return daily_attn();
}

bool lane_cell_attn() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_ATTN_LANECELL");
        return v && v[0] == '1';
    }();
    return on;
}

int attn_kv_mode(const QsaAttnPools& p) {
    return p.k_q4 != nullptr ? 2 : (p.k_q != nullptr && p.v_q4 != nullptr ? 3 : (p.k_q != nullptr ? 1 : 0));
}

int attn_merge_v2() {
    static const int on = [] {
        int r = daily_attn() ? 1 : 0;
        const char* v = std::getenv("STRATA_ATTN_MERGE_V2");
        if (v != nullptr && *v != '\0') r = std::atoi(v) != 0 ? 1 : 0;
        return r;
    }();
    return on;
}

void launch_merge(int v2, const float* part_acc, const float* part_m, const float* part_l, int n_chunks,
                  const int32_t* steps, float* attn, long long stride, int n_head, int n_q, sycl::queue* st) {
    const sycl::nd_range<2> range(sycl::range<2>((size_t)n_q, (size_t)n_head * HD), sycl::range<2>(1, HD));
    if (v2) {
        st->parallel_for(range, [=](sycl::nd_item<2> item) {
            attn_merge_v2_kernel<kMergeU>(item, part_acc, part_m, part_l, n_chunks, steps, attn, n_head, stride);
        });
    } else {
        st->parallel_for(range, [=](sycl::nd_item<2> item) {
            attn_merge_kernel(item, part_acc, part_m, part_l, n_chunks, attn, n_head, stride);
        });
    }
}

void launch_chunks_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                         int64_t cap, const QsaShapes& s, float* scratch, int64_t n_q, sycl::queue* st, int kernel) {
    const int kv_mode = attn_kv_mode(pools);
    const int n_chunks = (int)((cap + CHUNK - 1) / CHUNK);
    const long long stride = (long long)qsa_decode_attn_scratch_floats(cap, s);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t)n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t)n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float)HD);
    const bool pre = kernel < 0 ? pre75_attn() : kernel == 1;
    const bool lane_cell = kernel < 0 && lane_cell_attn();

    if (lane_cell) {
        if (kv_mode == 3) launch_attn_chunk<3, true>(st, q, pools, ids, steps, (int)s.n_head_kv, (int)s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int)cap, stride, (int)n_q);
        else if (kv_mode == 2) launch_attn_chunk<2, true>(st, q, pools, ids, steps, (int)s.n_head_kv, (int)s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int)cap, stride, (int)n_q);
        else if (kv_mode == 1) launch_attn_chunk<1, true>(st, q, pools, ids, steps, (int)s.n_head_kv, (int)s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int)cap, stride, (int)n_q);
        else launch_attn_chunk<0, true>(st, q, pools, ids, steps, (int)s.n_head_kv, (int)s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int)cap, stride, (int)n_q);
    } else if (pre) {
        if (kv_mode == 3) launch_attn_chunk_pre75<3>(st, q, pools, ids, steps, (int)s.n_head_kv, (int)s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int)cap, stride, (int)n_q);
        else if (kv_mode == 2) launch_attn_chunk_pre75<2>(st, q, pools, ids, steps, (int)s.n_head_kv, (int)s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int)cap, stride, (int)n_q);
        else if (kv_mode == 1) launch_attn_chunk_pre75<1>(st, q, pools, ids, steps, (int)s.n_head_kv, (int)s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int)cap, stride, (int)n_q);
        else launch_attn_chunk_pre75<0>(st, q, pools, ids, steps, (int)s.n_head_kv, (int)s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int)cap, stride, (int)n_q);
    } else {
        if (kv_mode == 3) launch_attn_chunk<3, false>(st, q, pools, ids, steps, (int)s.n_head_kv, (int)s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int)cap, stride, (int)n_q);
        else if (kv_mode == 2) launch_attn_chunk<2, false>(st, q, pools, ids, steps, (int)s.n_head_kv, (int)s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int)cap, stride, (int)n_q);
        else if (kv_mode == 1) launch_attn_chunk<1, false>(st, q, pools, ids, steps, (int)s.n_head_kv, (int)s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int)cap, stride, (int)n_q);
        else launch_attn_chunk<0, false>(st, q, pools, ids, steps, (int)s.n_head_kv, (int)s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int)cap, stride, (int)n_q);
    }
}

void launch_merge_batch(const int32_t* steps, int64_t cap, const QsaShapes& s, const float* scratch, float* attn,
                        int64_t n_q, int v2, sycl::queue* st) {
    const int n_chunks = (int)((cap + CHUNK - 1) / CHUNK);
    const long long stride = (long long)qsa_decode_attn_scratch_floats(cap, s);
    const float* part_acc = scratch;
    const float* part_m = scratch + (size_t)n_chunks * s.n_head * HD;
    const float* part_l = part_m + (size_t)n_chunks * s.n_head;
    launch_merge(v2, part_acc, part_m, part_l, n_chunks, steps, attn, stride, (int)s.n_head, (int)n_q, st);
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
    sycl::queue* st = stream ? static_cast<sycl::queue*>(stream) : &get_default_in_order_queue();
    launch_chunks_batch(q, pools, ids, steps, cap, s, scratch, n_q, st, -1);
    launch_merge_batch(steps, cap, s, scratch, attn, n_q, attn_merge_v2(), st);
    if (!stream) st->wait();
}

uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    const int64_t chunks = (cap + CHUNK - 1) / CHUNK;
    return (uint64_t)chunks * (uint64_t)s.n_head * (HD + 2) + 64;
}

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream) {
    if (s.head_dim != HD || s.n_head != (int64_t)G * s.n_head_kv || cap <= 0 || !scratch || !ids || !step ||
        !pools.page_table) {
        std::fprintf(stderr, "qsa_decode_attn: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = attn_kv_mode(pools);
    if (kv_mode == 3 ? (!pools.k_scale || !pools.v_q4)
                     : (kv_mode == 2 ? (!pools.v_q4) : (kv_mode == 1 ? (!pools.v_q || !pools.k_scale || !pools.v_scale)
                                                                     : (!pools.k_pool || !pools.v_pool)))) {
        std::fprintf(stderr, "qsa_decode_attn: incomplete KV pools\n");
        std::exit(1);
    }
    qsa_decode_attn_batch(q, pools, ids, step, cap, s, scratch, attn, 1, stream);
}

void qsa_decode_attn_chunks_only(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                                 int64_t cap, const QsaShapes& s, float* scratch, int64_t n_q, int daily_chunk,
                                 void* stream) {
    if (n_q <= 0 || n_q > 65535 || s.head_dim != HD || s.n_head != (int64_t)G * s.n_head_kv || cap <= 0 || !scratch ||
        !q || !ids || !steps || !pools.page_table || (daily_chunk != 0 && daily_chunk != 1)) {
        std::fprintf(stderr, "qsa_decode_attn_chunks_only: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    sycl::queue* st = stream ? static_cast<sycl::queue*>(stream) : &get_default_in_order_queue();
    launch_chunks_batch(q, pools, ids, steps, cap, s, scratch, n_q, st, daily_chunk);
    if (!stream) st->wait();
}

void qsa_decode_attn_merge_only(const int32_t* steps, int64_t cap, const QsaShapes& s, const float* scratch,
                                float* attn, int64_t n_q, int v2, void* stream) {
    if (n_q <= 0 || n_q > 65535 || s.head_dim != HD || s.n_head != (int64_t)G * s.n_head_kv || cap <= 0 || !scratch ||
        (v2 != 0 && v2 != 1) || !steps || !attn) {
        std::fprintf(stderr, "qsa_decode_attn_merge_only: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    sycl::queue* st = stream ? static_cast<sycl::queue*>(stream) : &get_default_in_order_queue();
    launch_merge_batch(steps, cap, s, scratch, attn, n_q, v2, st);
    if (!stream) st->wait();
}

}  // namespace strata::kernels
