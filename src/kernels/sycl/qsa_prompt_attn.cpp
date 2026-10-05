// src/kernels/sycl/qsa_prompt_attn.cpp - Pure SYCL 2020 Hardware XMX Systolic Attention Kernel
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix-unified.hpp>
#include "strata/core/emulate.hpp"
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>

namespace strata::kernels {
namespace {

namespace jm = sycl::ext::oneapi::experimental::matrix;

constexpr int HD = 256;            // head_dim
constexpr int G = 12;              // query heads per KV head
constexpr int SG = 16;             // Xe2 native SIMD16 sub-group size
constexpr int NSG = 8;             // 8 sub-groups per workgroup (128 threads)
constexpr int THREADS = SG * NSG;  // 128 threads
constexpr int QS = HD + 8;         // q row stride in halves (bank-conflict-free)

// Local memory layout for CH cells per chunk; P tiles reuse the K^T region
template <int CH>
struct Layout {
    static constexpr int PS = CH + 8;                  // P / score row stride
    static constexpr int NT = CH / 16;                 // cell tiles
    static constexpr int UNITS = NT * 4;               // (cell tile, scale group) score units
    static constexpr size_t q = 2 * 16 * QS * 2;       // Q hi, Q lo (halves)
    static constexpr size_t kp = (size_t)CH * HD * 2;  // packed K^T, then the P tiles
    static constexpr size_t pv = (size_t)4 * 2 * 16 * PS * 2;
    static constexpr size_t kp_or_pv = kp > pv ? kp : pv;
    static constexpr size_t vp = (size_t)CH * HD * 2;                             // packed V
    static constexpr size_t parts = (size_t)(UNITS > 16 ? UNITS : 16) * 256 * 4;  // score units, then P.V tiles
    static constexpr size_t s = (size_t)16 * PS * 4;                              // probabilities
    static constexpr size_t acc = (size_t)16 * HD * 4;
    static constexpr size_t scal = (size_t)CH * 4 * 4 * 2;  // ks, vs
    static constexpr size_t misc = 80 * 4;
    static constexpr size_t row = (size_t)CH * 8;
    static constexpr size_t bytes = q + kp_or_pv + vp + parts + s + acc + scal + misc + row + 16 * 12;
};

template <typename T>
inline auto lptr(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
}

inline float half_as_float_exact(sycl::half h) {
    volatile uint16_t v = sycl::bit_cast<uint16_t>(h);
    uint16_t u = v;
    return static_cast<float>(sycl::bit_cast<sycl::half>(u));
}

inline bool has_xmx_matrix(const sycl::device& dev) {
    if (!dev.has(sycl::aspect::fp16)) return false;
    try {
        auto combos = dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
        if (!combos.empty()) return true;
    } catch (...) {
    }
    return false;
}

template <int KV_MODE, int CH>
void prompt_attn_xmx_kernel(const float* __restrict__ q, QsaAttnPools p, const int32_t* __restrict__ ids,
                            const int32_t* __restrict__ steps, int n_kv_heads, int page_size, float scale_log2,
                            float* __restrict__ attn, int cap, uint8_t* local_mem, sycl::nd_item<2> item) {
    using L = Layout<CH>;
    constexpr int PS = L::PS, UNITS = L::UNITS;
    auto sg = item.get_sub_group();
    uint8_t* at = local_mem;
    auto take = [&](size_t b) {
        uint8_t* r = at;
        at += (b + 15) & ~(size_t)15;
        return r;
    };
    sycl::half* qh = reinterpret_cast<sycl::half*>(take(L::q / 2));
    sycl::half* ql = reinterpret_cast<sycl::half*>(take(L::q / 2));
    sycl::half* kp = reinterpret_cast<sycl::half*>(take(L::kp_or_pv));
    sycl::half* ptile = kp;  // [g][hi/lo][16][PS], after the scores
    sycl::half* vp = reinterpret_cast<sycl::half*>(take(L::vp));
    float* parts = reinterpret_cast<float*>(take(L::parts));
    float* sp = reinterpret_cast<float*>(take(L::s));
    float* acc = reinterpret_cast<float*>(take(L::acc));
    float* ks = reinterpret_cast<float*>(take(L::scal / 2));
    float* vs = reinterpret_cast<float*>(take(L::scal / 2));
    float* misc = reinterpret_cast<float*>(take(L::misc));
    long long* rows = reinterpret_cast<long long*>(take(L::row));
    float* qmax = misc;        // [8]
    float* mrow = misc + 8;    // [16]
    float* lsum = misc + 24;   // [16]
    float* alpha = misc + 40;  // [16]
    float* vup = misc + 56;    // [4]
    float* vdown = misc + 60;  // [4]

    const int kvh = static_cast<int>(item.get_group(0));
    const int qi = static_cast<int>(item.get_group(1));
    const int n_head = n_kv_heads * G;
    q += (size_t)qi * n_head * HD + (size_t)kvh * G * HD;
    attn += (size_t)qi * n_head * HD + (size_t)kvh * G * HD;
    ids += (size_t)qi * cap;
    const int n = *(steps + (size_t)qi * kStepCount + kStepWidth);
    const int t = static_cast<int>(item.get_local_id(1));
    const int lane = static_cast<int>(sg.get_local_id()[0]);
    const int sgid = static_cast<int>(sg.get_group_id()[0]);

    // q: scaled so its largest value sits near 2^14 (exact), split into hi + lo halves
    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS)
        qm = sycl::fmax(qm, sycl::fabs(q[i]));
    qm = sycl::reduce_over_group(sg, qm, sycl::maximum<float>());
    if (lane == 0) qmax[sgid] = qm;
    item.barrier(sycl::access::fence_space::local_space);
    qm = 0.0f;
    for (int i = 0; i < NSG; ++i)
        qm = sycl::fmax(qm, qmax[i]);
    int qe = 0;
    if (qm > 0.0f) sycl::frexp(qm, &qe);
    const float qup = sycl::ldexp(1.0f, 14 - qe), qdown = sycl::ldexp(scale_log2, qe - 14);
    for (int i = t; i < 16 * HD; i += THREADS) {
        const int h = i / HD, d = i % HD;
        const float x = h < G ? q[(size_t)h * HD + d] * qup : 0.0f;
        const sycl::half hi = sycl::vec<float, 1>(x).convert<sycl::half, sycl::rounding_mode::rte>()[0];
        qh[h * QS + d] = hi;
        ql[h * QS + d] =
            sycl::vec<float, 1>(x - half_as_float_exact(hi)).convert<sycl::half, sycl::rounding_mode::rte>()[0];
    }
    for (int i = t; i < 16 * HD; i += THREADS)
        acc[i] = 0.0f;
    if (t < 16) {
        mrow[t] = -INFINITY;
        lsum[t] = 0.0f;
    }

    for (int c0 = 0; c0 < n; c0 += CH) {
        const int nh = sycl::min(CH, n - c0);
        if (t < CH) {
            long long r = -1;
            if (t < nh) {
                const int cell = ids[c0 + t];
                const long long page = static_cast<long long>(p.page_table[cell / page_size]);
                if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
            }
            rows[t] = r;
        }
        item.barrier(sycl::access::fence_space::local_space);

        // K rows -> packed K^T
        for (int i = t; i < CH * (HD / 8); i += THREADS) {
            const int c = i / (HD / 8), piece = i % (HD / 8);
            const long long r = rows[c];
            uint32_t w[4] = {0u, 0u, 0u, 0u};
            if (r >= 0) {
                if constexpr (KV_MODE == 0) {
                    const auto v = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(p.k_pool + r * HD + piece * 8);
                    for (int j = 0; j < 4; ++j)
                        w[j] = v[j];
                } else {
                    const auto v = *reinterpret_cast<const sycl::vec<int8_t, 8>*>(p.k_q + r * HD + piece * 8);
                    for (int j = 0; j < 4; ++j) {
                        const sycl::half2 h2(static_cast<float>(v[2 * j]), static_cast<float>(v[2 * j + 1]));
                        w[j] = sycl::bit_cast<uint32_t>(h2);
                    }
                }
            }
            uint32_t* dst = reinterpret_cast<uint32_t*>(kp);
            for (int j = 0; j < 4; ++j)
                dst[(size_t)(piece * 4 + j) * CH + c] = w[j];
        }

        // V rows -> packed V
        if constexpr (KV_MODE == 3) {
            constexpr int BLKS = HD / QK4_0;
            constexpr int BYTES = BLKS * static_cast<int>(sizeof(block_q4_0));
            for (int i = t; i < (CH / 2) * BLKS; i += THREADS) {
                const int cp = i / BLKS, b = i % BLKS;
                float v0[QK4_0], v1[QK4_0];
                for (int e = 0; e < 2; ++e) {
                    const long long r = rows[2 * cp + e];
                    float* v = e ? v1 : v0;
                    if (r < 0) {
                        for (int j = 0; j < QK4_0; ++j)
                            v[j] = 0.0f;
                        continue;
                    }
                    const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + r * BYTES) + b;
                    const float d = static_cast<float>(sycl::bit_cast<sycl::half, unsigned short>(blk->d));
                    for (int j = 0; j < QK4_0 / 2; ++j) {
                        v[j] = static_cast<float>(static_cast<int>(blk->qs[j] & 0x0F) - 8) * d;
                        v[j + QK4_0 / 2] = static_cast<float>(static_cast<int>(blk->qs[j] >> 4) - 8) * d;
                    }
                }
                uint32_t* dst = reinterpret_cast<uint32_t*>(vp) + (size_t)cp * HD + b * QK4_0;
                for (int j = 0; j < QK4_0; j += 8) {
                    sycl::vec<uint32_t, 8> o;
                    for (int u = 0; u < 8; ++u)
                        o[u] = sycl::bit_cast<uint32_t>(sycl::half2(v0[j + u], v1[j + u]));
                    *reinterpret_cast<sycl::vec<uint32_t, 8>*>(dst + j) = o;
                }
            }
        } else {
            for (int i = t; i < (CH / 2) * (HD / 8); i += THREADS) {
                const int cp = i / (HD / 8), piece = i % (HD / 8);
                sycl::half h0[8], h1[8];
                for (int e = 0; e < 2; ++e) {
                    const long long r = rows[2 * cp + e];
                    sycl::half* h = e ? h1 : h0;
                    if (r < 0) {
                        for (int j = 0; j < 8; ++j)
                            h[j] = sycl::half(0.0f);
                        continue;
                    }
                    if constexpr (KV_MODE == 1) {
                        const auto v = *reinterpret_cast<const sycl::vec<int8_t, 8>*>(p.v_q + r * HD + piece * 8);
                        for (int j = 0; j < 8; ++j)
                            h[j] = sycl::half(static_cast<float>(v[j]));
                    } else {
                        const auto v = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(p.v_pool + r * HD + piece * 8);
                        for (int j = 0; j < 8; ++j)
                            h[j] = sycl::bit_cast<sycl::half, unsigned short>(v[j]);
                    }
                }
                sycl::vec<uint32_t, 8> o;
                for (int j = 0; j < 8; ++j)
                    o[j] = sycl::bit_cast<uint32_t>(sycl::half2(h0[j], h1[j]));
                *reinterpret_cast<sycl::vec<uint32_t, 8>*>(reinterpret_cast<uint32_t*>(vp) + (size_t)cp * HD +
                                                           piece * 8) = o;
            }
        }
        for (int i = t; i < CH * 4; i += THREADS) {
            const int c = i / 4, g = i % 4;
            const long long r = rows[c];
            float a = 0.0f, b = 0.0f;
            if (r >= 0) {
                if constexpr (KV_MODE == 1) {
                    a = static_cast<float>(
                        sycl::bit_cast<sycl::half, unsigned short>(p.k_scale[r * (HD / KV_Q8_GROUP) + g]));
                    b = static_cast<float>(
                        sycl::bit_cast<sycl::half, unsigned short>(p.v_scale[r * (HD / KV_Q8_GROUP) + g]));
                } else if constexpr (KV_MODE == 3) {
                    a = static_cast<float>(
                        sycl::bit_cast<sycl::half, unsigned short>(p.k_scale[r * (HD / KV_Q8_GROUP) + g]));
                    b = 1.0f;
                } else {
                    a = b = 1.0f;
                }
            }
            ks[c * 4 + g] = a;
            vs[c * 4 + g] = b;
        }
        item.barrier(sycl::access::fence_space::local_space);

        // Scores: unit u = (cell tile u / 4, scale group u % 4)
        for (int u = sgid; u < UNITS; u += NSG) {
            const int nt = u / 4, g = u % 4;
            jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 16, 16> C;
            jm::joint_matrix_fill(sg, C, 0.0f);
            for (int kk = 0; kk < 4; ++kk) {
                const int k0 = (g * 4 + kk) * 16;
                jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::ext_intel_packed> B;
                jm::joint_matrix_load(sg, B, lptr(kp + (k0 >> 1) * (CH * 2) + nt * 16 * 2),
                                      static_cast<size_t>(CH * 2));
                jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 16, 16, jm::layout::row_major> A;
                jm::joint_matrix_load(sg, A, lptr(qh + k0), static_cast<size_t>(QS));
                jm::joint_matrix_mad(sg, C, A, B, C);
                jm::joint_matrix_load(sg, A, lptr(ql + k0), static_cast<size_t>(QS));
                jm::joint_matrix_mad(sg, C, A, B, C);
            }
            jm::joint_matrix_store(sg, C, lptr(parts + (size_t)u * 256), static_cast<size_t>(16),
                                   jm::layout::row_major);
        }

        if (t < 4) {
            float vmax = 0.0f;
            for (int c = 0; c < CH; ++c)
                vmax = sycl::fmax(vmax, sycl::fabs(vs[c * 4 + t]));
            vup[t] = vmax > 0.0f ? 16384.0f / vmax : 0.0f;
            vdown[t] = vmax * (1.0f / 16384.0f);
        }
        item.barrier(sycl::access::fence_space::local_space);

        // Online softmax
        {
            constexpr int PER = CH / 8;
            const int r = t >> 3, sub = t & 7;
            float x[PER], mx = -INFINITY;
            for (int j = 0; j < PER; ++j) {
                const int c = sub * PER + j, nt = c >> 4, cc = c & 15;
                float v = 0.0f;
                for (int g = 0; g < 4; ++g)
                    v += parts[(size_t)(nt * 4 + g) * 256 + r * 16 + cc] * ks[c * 4 + g];
                x[j] = c < nh ? v * qdown : -INFINITY;
                mx = sycl::fmax(mx, x[j]);
            }
            for (int o = 1; o < 8; o <<= 1)
                mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, o));
            const float m_old = mrow[r];
            const float m_new = sycl::fmax(m_old, mx);
            float sum = 0.0f;
            for (int j = 0; j < PER; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : sycl::exp2(x[j] - m_new);
                sp[r * PS + sub * PER + j] = e;
                sum += e;
            }
            for (int o = 1; o < 8; o <<= 1)
                sum += sycl::permute_group_by_xor(sg, sum, o);
            sycl::group_barrier(sg);
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : sycl::exp2(m_old - m_new);
                alpha[r] = a;
                lsum[r] = sycl::fma(lsum[r], a, sum);
                mrow[r] = m_new;
            }
        }
        item.barrier(sycl::access::fence_space::local_space);

        // P tiles into kp region
        constexpr int NGP = KV_MODE == 1 ? 4 : 1;
        for (int i = t; i < NGP * 16 * (CH / 2); i += THREADS) {
            const int g = i / (16 * (CH / 2)), rc = i % (16 * (CH / 2)), r = rc / (CH / 2), c = (rc % (CH / 2)) * 2;
            const float w0 = vs[c * 4 + g] * vup[g], w1 = vs[(c + 1) * 4 + g] * vup[g];
            const float p0 = sp[r * PS + c] * w0, p1 = sp[r * PS + c + 1] * w1;
            const sycl::half h0 = sycl::vec<float, 1>(p0).convert<sycl::half, sycl::rounding_mode::rte>()[0];
            const sycl::half h1 = sycl::vec<float, 1>(p1).convert<sycl::half, sycl::rounding_mode::rte>()[0];
            *reinterpret_cast<sycl::half2*>(ptile + ((size_t)(g * 2 + 0) * 16 + r) * PS + c) = sycl::half2(h0, h1);
            *reinterpret_cast<sycl::half2*>(ptile + ((size_t)(g * 2 + 1) * 16 + r) * PS + c) = sycl::half2(
                sycl::vec<float, 1>(p0 - half_as_float_exact(h0)).convert<sycl::half, sycl::rounding_mode::rte>()[0],
                sycl::vec<float, 1>(p1 - half_as_float_exact(h1)).convert<sycl::half, sycl::rounding_mode::rte>()[0]);
        }
        item.barrier(sycl::access::fence_space::local_space);

        // P.V matrix multiply
        {
            const int g = sgid >> 1, gp = KV_MODE == 1 ? g : 0;
            const sycl::half* ah = ptile + (size_t)(gp * 2 + 0) * 16 * PS;
            const sycl::half* al = ptile + (size_t)(gp * 2 + 1) * 16 * PS;
            float* tmp = parts + (size_t)sgid * 512;
            for (int dt = 0; dt < 2; ++dt) {
                const int d0 = sgid * 32 + dt * 16;
                jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 16, 16> C;
                jm::joint_matrix_fill(sg, C, 0.0f);
                for (int ks16 = 0; ks16 < CH; ks16 += 16) {
                    jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::ext_intel_packed> B;
                    jm::joint_matrix_load(sg, B, lptr(vp + (size_t)(ks16 >> 1) * (HD * 2) + d0 * 2),
                                          static_cast<size_t>(HD * 2));
                    jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 16, 16, jm::layout::row_major> A;
                    jm::joint_matrix_load(sg, A, lptr(ah + ks16), static_cast<size_t>(PS));
                    jm::joint_matrix_mad(sg, C, A, B, C);
                    jm::joint_matrix_load(sg, A, lptr(al + ks16), static_cast<size_t>(PS));
                    jm::joint_matrix_mad(sg, C, A, B, C);
                }
                jm::joint_matrix_store(sg, C, lptr(tmp + dt * 256), static_cast<size_t>(16), jm::layout::row_major);
            }
            sycl::group_barrier(sg);
            const float vd = vdown[g];
            for (int i = lane; i < 512; i += SG) {
                const int dt = i >> 8, rc = i & 255, r = rc >> 4, cc = rc & 15;
                float* a = acc + r * HD + sgid * 32 + dt * 16 + cc;
                *a = sycl::fma(*a, alpha[r], tmp[i] * vd);
            }
        }
    }
    item.barrier(sycl::access::fence_space::local_space);
    for (int i = t; i < G * HD; i += THREADS) {
        const int h = i / HD, d = i % HD;
        const float l = lsum[h];
        attn[(size_t)h * HD + d] = l > 0.0f ? acc[h * HD + d] / l : 0.0f;
    }
}

template <int KV_MODE, int CH>
bool launch_xmx(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                const QsaShapes& s, float* attn, int64_t n_q, sycl::queue* q_ptr) {
    const float scale_log2 = 1.4426950408889634f / std::sqrt(static_cast<float>(HD));
    constexpr size_t bytes = Layout<CH>::bytes;
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        const float* qq = q + q0 * s.n_head * HD;
        const int32_t* idq = ids + q0 * cap;
        const int32_t* stq = steps + q0 * kStepCount;
        float* aq = attn + q0 * s.n_head * HD;
        const int nkv = static_cast<int>(s.n_head_kv);
        const int ps = static_cast<int>(s.page_size);
        const int capi = static_cast<int>(cap);
        q_ptr->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(bytes), cgh);
            cgh.parallel_for(
                sycl::nd_range<2>(sycl::range<2>(static_cast<size_t>(nkv), static_cast<size_t>(nb * THREADS)),
                                  sycl::range<2>(1, THREADS)),
                [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                    prompt_attn_xmx_kernel<KV_MODE, CH>(qq, pools, idq, stq, nkv, ps, scale_log2, aq, capi,
                                                        slm.get_multi_ptr<sycl::access::decorated::no>().get(), it);
                });
        });
    }
    return true;
}

template <int CH>
bool dispatch_xmx(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                  const QsaShapes& s, float* attn, int64_t n_q, sycl::queue* q_ptr) {
    if (pools.k_q != nullptr && pools.v_q4 != nullptr) {
        if (!pools.k_scale) return false;
        return launch_xmx<3, CH>(q, pools, ids, steps, cap, s, attn, n_q, q_ptr);
    }
    if (pools.k_q != nullptr) {
        if (!pools.v_q || !pools.k_scale || !pools.v_scale) return false;
        return launch_xmx<1, CH>(q, pools, ids, steps, cap, s, attn, n_q, q_ptr);
    }
    if (!pools.k_pool || !pools.v_pool) return false;
    return launch_xmx<0, CH>(q, pools, ids, steps, cap, s, attn, n_q, q_ptr);
}

#if defined(__HIPCC__)
#if defined(__gfx1200__) || defined(__gfx1201__)
#define STRATA_PA_WMMA 1
#else
#define STRATA_PA_WMMA 0
#endif
typedef _Float16 wh8 __attribute__((ext_vector_type(8)));
typedef float wf8 __attribute__((ext_vector_type(8)));
constexpr int WCH = 32;
constexpr int WVS = 80;
struct alignas(16) SmemW {
    float part[4][16][WCH + 1];
    float p[16][WCH + 1];
    float vs[4][WCH];
    int valid[WCH];
    uint8_t v[4][WCH][WVS];
    float qmax[4];
    float alpha[16];
    float lsum[16];
    float mrow[16];
};

__device__ __forceinline__ wf8 wmma_f16(wh8 a, wh8 b, wf8 c) {
#if STRATA_PA_WMMA
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#else
    __builtin_trap();
    return c;
#endif
}

__device__ __forceinline__ wh8 i8x8_to_h8(uint2 x) {
    wh8 h;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        h[i] = (_Float16)(int)(int8_t)(x.x >> (8 * i));
        h[4 + i] = (_Float16)(int)(int8_t)(x.y >> (8 * i));
    }
    return h;
}

__global__ void __launch_bounds__(THREADS)
    prompt_attn_wmma_kernel(const float* __restrict__ q, QsaAttnPools p, const int32_t* __restrict__ ids,
                            const int32_t* __restrict__ steps, int n_kv_heads, int page_size, float scale_log2,
                            float* __restrict__ attn, int cap) {
#if STRATA_PA_WMMA
    __shared__ SmemW S;
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int n_head = n_kv_heads * G;
    q += (size_t)qi * n_head * HD + (size_t)kvh * G * HD;
    attn += (size_t)qi * n_head * HD + (size_t)kvh * G * HD;
    ids += (size_t)qi * cap;
    const int n = __ldg(steps + (size_t)qi * kStepCount + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int col = lane & 15, half = lane >> 4;
    const int dim0 = warp * 64;

    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS)
        qm = fmaxf(qm, fabsf(q[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    if (t < 16) {
        S.mrow[t] = -INFINITY;
        S.lsum[t] = 0.0f;
    }
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale_log2, qe - 14);
    wh8 qh[4], ql[4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        float x[8];
        if (col < G) {
            const float4* src = reinterpret_cast<const float4*>(q + (size_t)col * HD + dim0 + kk * 16 + half * 8);
            const float4 a = src[0], b = src[1];
            x[0] = a.x;
            x[1] = a.y;
            x[2] = a.z;
            x[3] = a.w;
            x[4] = b.x;
            x[5] = b.y;
            x[6] = b.z;
            x[7] = b.w;
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i)
                x[i] = 0.0f;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float v = x[i] * qup;
            const _Float16 hi = (_Float16)v;
            qh[kk][i] = hi;
            ql[kk][i] = (_Float16)(v - (float)hi);
        }
    }

    wf8 acc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
        acc[j] = wf8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int n_chunks = (n + WCH - 1) / WCH;
    for (int ci = 0; ci < n_chunks; ++ci) {
        const int c0 = ci * WCH;
        long long myrow = -1;
        if (c0 + lane < n) {
            const int cell = __ldg(ids + c0 + lane);
            const long long page = (long long)__ldg(p.page_table + cell / page_size);
            if (page >= 0) myrow = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        {
            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0, v2 = v0, v3 = v0;
            float vsc = 0.0f;
            if (myrow >= 0) {
                const uint4* src = reinterpret_cast<const uint4*>(p.v_q + myrow * HD + dim0);
                v0 = __ldg(src);
                v1 = __ldg(src + 1);
                v2 = __ldg(src + 2);
                v3 = __ldg(src + 3);
                vsc = __half2float(__ushort_as_half(__ldg(p.v_scale + myrow * (HD / KV_Q8_GROUP) + warp)));
            }
            uint4* dst = reinterpret_cast<uint4*>(&S.v[warp][lane][0]);
            dst[0] = v0;
            dst[1] = v1;
            dst[2] = v2;
            dst[3] = v3;
            S.vs[warp][lane] = vsc;
            if (warp == 0) S.valid[lane] = myrow >= 0;
        }
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const long long rr = __shfl(myrow, nt * 16 + col);
            wf8 s = wf8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            float ksc = 0.0f;
            if (rr >= 0) ksc = __half2float(__ushort_as_half(__ldg(p.k_scale + rr * (HD / KV_Q8_GROUP) + warp)));
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                uint2 kx = make_uint2(0, 0);
                if (rr >= 0) kx = __ldg(reinterpret_cast<const uint2*>(p.k_q + rr * HD + dim0 + kk * 16 + half * 8));
                const wh8 b = i8x8_to_h8(kx);
                s = wmma_f16(qh[kk], b, s);
                s = wmma_f16(ql[kk], b, s);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i)
                S.part[warp][half * 8 + i][nt * 16 + col] = s[i] * ksc;
        }
        __syncthreads();
        {
            const int r = t >> 3, sub = t & 7;
            float x[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                x[j] = S.valid[c] ? (((S.part[0][r][c] + S.part[1][r][c]) + S.part[2][r][c]) + S.part[3][r][c]) * qdown
                                  : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.p[r][sub * 4 + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1)
                sum += __shfl_xor_sync(0xffffffffu, sum, o);
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();
        {
            float vmax = S.vs[warp][lane];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
            wf8 tmp[4];
#pragma unroll
            for (int j = 0; j < 4; ++j)
                tmp[j] = wf8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                const int cb = ks * 16 + half * 8;
                wh8 ah, al;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float pv = S.p[col][cb + i] * (S.vs[warp][cb + i] * vup);
                    const _Float16 hi = (_Float16)pv;
                    ah[i] = hi;
                    al[i] = (_Float16)(pv - (float)hi);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int d = j * 16 + col;
                    wh8 b;
#pragma unroll
                    for (int i = 0; i < 8; ++i)
                        b[i] = (_Float16)(int)(int8_t)S.v[warp][cb + i][d];
                    tmp[j] = wmma_f16(ah, b, tmp[j]);
                    tmp[j] = wmma_f16(al, b, tmp[j]);
                }
            }
            float a[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                a[i] = S.alpha[half * 8 + i];
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    acc[j][i] = fmaf(acc[j][i], a[i], tmp[j][i] * vdown);
        }
    }
    float inv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float l = S.lsum[half * 8 + i];
        inv[i] = l > 0.0f ? 1.0f / l : 0.0f;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = half * 8 + i;
            if (row < G) attn[(size_t)row * HD + dim0 + j * 16 + col] = acc[j][i] * inv[i];
        }
#else
    __builtin_trap();
#endif
}

bool hip_wmma_usable() {
    static const bool want = [] {
        const char* e = std::getenv("STRATA_HIP_WMMA");
        return e != nullptr && e[0] == '1';
    }();
    if (!want) return false;
    static int arch[64] = {};
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) {
        (void)hipGetLastError();
        return false;
    }
    if (arch[dev] == 0) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, dev) != hipSuccess) {
            (void)hipGetLastError();
            return false;
        }
        arch[dev] = std::strncmp(prop.gcnArchName, "gfx12", 5) == 0 ? 1 : 2;
        static bool told = false;
        if (!told) {
            told = true;
            std::fprintf(stderr, "strata: STRATA_HIP_WMMA: the prompt attention on matrix cores %s (%s)\n",
                         arch[dev] == 1 ? "on" : "unavailable", prop.gcnArchName);
        }
    }
    return arch[dev] == 1;
}

bool launch_wmma(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                 const QsaShapes& s, float* attn, int64_t n_q, cudaStream_t st) {
    const float scale_log2 = 1.4426950408889634f / sqrtf((float)HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        prompt_attn_wmma_kernel<<<dim3((unsigned)nb, (unsigned)s.n_head_kv), THREADS, 0, st>>>(
            q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount, (int)s.n_head_kv, (int)s.page_size,
            scale_log2, attn + q0 * s.n_head * HD, (int)cap);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_prompt_attn_batch (wmma): %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}
#endif

}  // namespace

bool qsa_prompt_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return true;
    static sycl::queue default_queue{sycl::property::queue::in_order()};
    sycl::queue* q_ptr = stream ? static_cast<sycl::queue*>(stream) : &default_queue;

#if defined(__HIPCC__)
    if (pools.k_q != nullptr && pools.v_q != nullptr && pools.k_scale != nullptr && pools.v_scale != nullptr &&
        pools.k_q4 == nullptr && pools.v_q4 == nullptr && s.head_dim == HD && s.n_head == (int64_t)G * s.n_head_kv &&
        cap > 0 && ids && steps && pools.page_table && hip_wmma_usable())
        return launch_wmma(q, pools, ids, steps, cap, s, attn, n_q, (cudaStream_t)stream);
    return false;
#endif

    if (pools.k_q4 != nullptr || s.head_dim != HD || s.n_head != (int64_t)G * s.n_head_kv || cap <= 0 || !ids ||
        !steps || !pools.page_table)
        return false;

    if (!has_xmx_matrix(q_ptr->get_device())) return false;

    const size_t have = q_ptr->get_device().get_info<sycl::info::device::local_mem_size>();
    static const int chunk_override = [] {
        const char* v = std::getenv("STRATA_PROMPT_ATTN_CHUNK");
        return v ? std::atoi(v) : 0;
    }();
    const bool ch64 = (chunk_override == 64 || (chunk_override != 32 && have >= Layout<64>::bytes));
    if (ch64 && have >= Layout<64>::bytes) {
        return dispatch_xmx<64>(q, pools, ids, steps, cap, s, attn, n_q, q_ptr);
    } else if (have >= Layout<32>::bytes) {
        return dispatch_xmx<32>(q, pools, ids, steps, cap, s, attn, n_q, q_ptr);
    }

    return false;
}

}  // namespace strata::kernels
