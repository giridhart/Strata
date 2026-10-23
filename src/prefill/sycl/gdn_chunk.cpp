// src/prefill/sycl/gdn_chunk.cpp - Chunked GDN prompt recurrence (FlashQLA gdn-chunk) for Intel Xe2
#include "strata/prefill/gdn_chunk.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace strata::prefill {

namespace {

constexpr int GDN_S = 128;
constexpr int GDN_HK = 16;
constexpr int GDN_HV = 48;
constexpr int GDN_C = 10240;
constexpr int GDN_CK = 16;                                 // Tokens per chunk (matching Xe2 SIMD16 subgroup)
constexpr int GDN_VPK = GDN_HV / GDN_HK;                   // 3 value heads per key head

inline float sigm(float x) {
    return 1.0f / (1.0f + sycl::native::exp(-x));
}

inline uint16_t hf(float f) {
    return sycl::bit_cast<unsigned short, sycl::half>(
        sycl::vec<float, 1>(f).convert<sycl::half, sycl::rounding_mode::rte>()[0]);
}

// Subgroup SIMD16 reduction using sycl::permute_group_by_xor
inline float sg16_reduce_sum(sycl::sub_group sg, float v) {
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) {
        v += sycl::permute_group_by_xor(sg, v, o);
    }
    return v;
}

// Subgroup SIMD16 inclusive scan using sycl::select_from_group
inline float sg16_inclusive_scan(sycl::sub_group sg, float v) {
    const unsigned lane = sg.get_local_linear_id();
#pragma unroll
    for (int o = 1; o < GDN_CK; o <<= 1) {
        const float up = sycl::select_from_group(sg, v, lane >= (unsigned)o ? lane - (unsigned)o : 0u);
        if (lane >= (unsigned)o) v += up;
    }
    return v;
}

// Device workspace for triangular solve T, Gram matrix P, gate cumsum gamma, and beta
struct GdnChunkWorkspace {
    sycl::queue* q = nullptr;
    float* ws_T = nullptr;
    float* ws_P = nullptr;
    float* ws_gamma = nullptr;
    float* ws_beta = nullptr;
    size_t max_chunks = 0;

    void ensure(size_t want_chunks, sycl::queue& queue) {
        if (want_chunks <= max_chunks && q == &queue && ws_T != nullptr) return;
        release();
        q = &queue;
        max_chunks = want_chunks < 32 ? 32 : want_chunks;
        const size_t tp_elems = max_chunks * GDN_HV * (GDN_CK * GDN_CK);
        const size_t gb_elems = max_chunks * GDN_HV * GDN_CK;
        ws_T = sycl::malloc_device<float>(tp_elems, *q);
        ws_P = sycl::malloc_device<float>(tp_elems, *q);
        ws_gamma = sycl::malloc_device<float>(gb_elems, *q);
        ws_beta = sycl::malloc_device<float>(gb_elems, *q);
        if (!ws_T || !ws_P || !ws_gamma || !ws_beta) {
            std::fprintf(stderr, "gdn_chunk: malloc_device failed for workspace (%zu chunks)\n", max_chunks);
            std::exit(1);
        }
    }

    void release() {
        if (q) {
            if (ws_T) sycl::free(ws_T, *q);
            if (ws_P) sycl::free(ws_P, *q);
            if (ws_gamma) sycl::free(ws_gamma, *q);
            if (ws_beta) sycl::free(ws_beta, *q);
        }
        ws_T = ws_P = ws_gamma = ws_beta = nullptr;
        max_chunks = 0;
        q = nullptr;
    }

    ~GdnChunkWorkspace() {
        // Intentionally no-op on static/thread-local destruction to avoid
        // freeing device memory after SYCL runtime teardown.
    }
};

static GdnChunkWorkspace& get_workspace() {
    static thread_local GdnChunkWorkspace ws;
    return ws;
}

// ----------------------------------------------------------------------------
// Stage 1: Preparation Kernel (GdnChunkPrepFunctor)
// Computes intra-chunk Gram matrices (k_t.k_i, q_t.k_i), gate cumsum gamma,
// and solves T = (I + A)^-1 via in-place forward substitution using SIMD16 subgroups.
// Grid: (n_chunks, GDN_HK) workgroups, workgroup size = 48 (3 subgroups of 16).
// ----------------------------------------------------------------------------
class GdnChunkPrepFunctor {
public:
    const float* h;
    const float* gate;
    const float* beta;
    float* ws_T;
    float* ws_P;
    float* ws_gamma;
    float* ws_beta;
    int64_t T;
    int64_t n_chunks;

    // Local accessors
    sycl::local_accessor<float, 2> K_slm;           // [16][128]
    sycl::local_accessor<float, 2> Q_slm;           // [16][128]
    sycl::local_accessor<float, 2> kk_slm;          // [16][16] shared across value heads
    sycl::local_accessor<float, 2> qk_slm;          // [16][16] shared across value heads
    sycl::local_accessor<float, 2> A_slm;           // [3][256] per value head
    sycl::local_accessor<float, 2> sgam_slm;        // [3][16] per value head
    sycl::local_accessor<float, 2> sbeta_slm;       // [3][16] per value head

    void operator()(sycl::nd_item<2> item) const {
        auto sg = item.get_sub_group();
        const int ch = item.get_group(0);
        const int qh = item.get_group(1);
        const int tid = item.get_local_id(1);
        const int w = tid / 16;
        const int lane = tid % 16;
        const int vh = qh + w * GDN_HK;
        const int64_t t0 = (int64_t)ch * GDN_CK;

        // 1. Collaborative loading of K and Q tiles for the 16 tokens
        for (int idx = tid; idx < GDN_CK * GDN_S; idx += 48) {
            const int t = idx / GDN_S;
            const int d = idx % GDN_S;
            const bool valid = (t0 + t < T);
            const float* ht = h + (valid ? (t0 + t) * GDN_C : 0);
            K_slm[t][d] = valid ? ht[GDN_HK * GDN_S + qh * GDN_S + d] : 0.0f;
            Q_slm[t][d] = valid ? ht[qh * GDN_S + d] : 0.0f;
        }
        item.barrier(sycl::access::fence_space::local_space);

        // 2. Collaborative dot product computation for kk_slm and qk_slm
        // 48 threads compute 256 matrix entries (only i <= t needed)
        for (int p = tid; p < GDN_CK * GDN_CK; p += 48) {
            const int t = p / GDN_CK;
            const int i = p % GDN_CK;
            if (i <= t) {
                float dot_kk = 0.0f;
                float dot_qk = 0.0f;
#pragma unroll 8
                for (int d = 0; d < GDN_S; ++d) {
                    dot_kk += K_slm[t][d] * K_slm[i][d];
                    dot_qk += Q_slm[t][d] * K_slm[i][d];
                }
                kk_slm[t][i] = dot_kk;
                qk_slm[t][i] = dot_qk;
            } else {
                kk_slm[t][i] = 0.0f;
                qk_slm[t][i] = 0.0f;
            }
        }

        // 3. Load gate and beta, compute inclusive prefix sum gamma via SIMD16 subgroup scan
        const bool valid_lane = (t0 + lane < T);
        const float g_raw = valid_lane ? gate[(t0 + lane) * GDN_HV + vh] : 0.0f;
        const float b_raw = valid_lane ? beta[(t0 + lane) * GDN_HV + vh] : 0.0f;

        const float g_cum = sg16_inclusive_scan(sg, g_raw);
        sgam_slm[w][lane] = g_cum;
        sbeta_slm[w][lane] = b_raw;

        item.barrier(sycl::access::fence_space::local_space);

        // 4. Form A and P matrices for this value head
        const int t = lane;
        const float beta_t = sbeta_slm[w][t];
        const float gamma_t = sgam_slm[w][t];
        const size_t ws_base_tp = ((size_t)ch * GDN_HV + vh) * (GDN_CK * GDN_CK);

        for (int i = 0; i <= t; ++i) {
            const float gamma_i = sgam_slm[w][i];
            const float dec = sycl::native::exp(gamma_t - gamma_i);

            const float p_val = dec * qk_slm[t][i];
            ws_P[ws_base_tp + t * GDN_CK + i] = p_val;

            if (i < t) {
                const float a_val = beta_t * dec * kk_slm[t][i];
                A_slm[w][t * GDN_CK + i] = a_val;
            } else {
                A_slm[w][t * GDN_CK + i] = 0.0f;
            }
        }
        for (int i = t + 1; i < GDN_CK; ++i) {
            ws_P[ws_base_tp + t * GDN_CK + i] = 0.0f;
            A_slm[w][t * GDN_CK + i] = 0.0f;
        }

        item.barrier(sycl::access::fence_space::local_space);

        // 5. In-place forward-substitution triangular solve for T = (I + A)^-1
        // Lane `c = lane` solves column `c` of unit lower-triangular matrix (I + A)
        const int c = lane;
        float tc[GDN_CK];
#pragma unroll
        for (int row = 0; row < GDN_CK; ++row) {
            float v = (row == c) ? 1.0f : 0.0f;
#pragma unroll
            for (int k = 0; k < row; ++k) {
                v -= A_slm[w][row * GDN_CK + k] * tc[k];
            }
            tc[row] = v;
            ws_T[ws_base_tp + row * GDN_CK + c] = v;
        }

        // Store gamma and beta to workspace
        const size_t ws_base_gb = ((size_t)ch * GDN_HV + vh) * GDN_CK;
        ws_gamma[ws_base_gb + lane] = sgam_slm[w][lane];
        ws_beta[ws_base_gb + lane] = sbeta_slm[w][lane];
    }
};

// ----------------------------------------------------------------------------
// Stage 2: Scan Kernel (GdnChunkScanFunctor)
// Fully resident [128 x 128] state recurrence: one workgroup of 128 threads per head,
// each thread holding all 128 state rows for its column strictly in registers.
// Fuses RMSNorm and sigmoid gating directly when z is provided.
// Grid: GDN_HV (48) workgroups, workgroup size = 128 (8 subgroups of 16).
// ----------------------------------------------------------------------------
class GdnChunkScanFunctor {
public:
    float* state;
    const float* h;
    const float* ws_T;
    const float* ws_P;
    const float* ws_gamma;
    const float* ws_beta;
    const float* z;
    const float* gamma_norm;
    float eps;
    float* oc;
    uint16_t* y16;
    int64_t ld16;
    int64_t T;
    int64_t n_chunks;

    // Shared local memory accessors
    sycl::local_accessor<float, 2> K_tile;           // [16][128]
    sycl::local_accessor<float, 2> Q_tile;           // [16][128]
    sycl::local_accessor<float, 2> T_tile;           // [16][16]
    sycl::local_accessor<float, 2> P_tile;           // [16][16]
    sycl::local_accessor<float, 1> gamma_tile;       // [16]
    sycl::local_accessor<float, 1> beta_tile;        // [16]
    sycl::local_accessor<float, 2> sn;               // [16][8]

    void operator()(sycl::nd_item<1> item) const {
        auto sg = item.get_sub_group();
        const int vh = item.get_group(0);               // 0..47 (value head)
        const int qh = vh % GDN_HK;
        const int col = item.get_local_id(0);           // 0..127 (column index)
        const int sg_id = col / 16;
        const int lane = col % 16;
        const size_t rs = (size_t)GDN_HV * GDN_S;

        // Persistent 128 rows of state for column `col` held strictly in registers
        float W[GDN_S];
#pragma unroll
        for (int r = 0; r < GDN_S; ++r) {
            W[r] = state[(size_t)r * rs + (size_t)vh * GDN_S + col];
        }

        const float g_col = (gamma_norm != nullptr) ? gamma_norm[col] : 1.0f;

        // Loop over 16-token chunks
        for (int64_t ch = 0; ch < n_chunks; ++ch) {
            const int64_t t0 = ch * GDN_CK;
            if (t0 >= T) break;

            // 1. Collaborative load of K_tile and Q_tile (16 floats per thread)
            for (int i = 0; i < 16; ++i) {
                const int idx = col * 16 + i;
                const int tok = idx / GDN_S;
                const int d = idx % GDN_S;
                const bool valid = (t0 + tok < T);
                const float* ht = h + (valid ? (t0 + tok) * GDN_C : 0);
                K_tile[tok][d] = valid ? ht[GDN_HK * GDN_S + qh * GDN_S + d] : 0.0f;
                Q_tile[tok][d] = valid ? ht[qh * GDN_S + d] : 0.0f;
            }

            // 2. Collaborative load of T_tile and P_tile (2 floats per thread)
            for (int k = 0; k < 2; ++k) {
                const int tok = (col / 16) + k * 8;
                const int c_sub = col % 16;
                const size_t tp_idx = ((size_t)ch * GDN_HV + vh) * (GDN_CK * GDN_CK) + tok * GDN_CK + c_sub;
                T_tile[tok][c_sub] = ws_T[tp_idx];
                P_tile[tok][c_sub] = ws_P[tp_idx];
            }

            // 3. Load gamma and beta for the chunk
            if (col < GDN_CK) {
                const size_t gb_idx = ((size_t)ch * GDN_HV + vh) * GDN_CK + col;
                gamma_tile[col] = ws_gamma[gb_idx];
                beta_tile[col] = ws_beta[gb_idx];
            }

            item.barrier(sycl::access::fence_space::local_space);

            // 4. Compute u_t and qw_t for all 16 tokens entirely in thread-private registers
            float u[GDN_CK];
            float qw[GDN_CK];
#pragma unroll
            for (int tok = 0; tok < GDN_CK; ++tok) {
                float u_val = 0.0f;
                float qw_val = 0.0f;
#pragma unroll
                for (int r = 0; r < GDN_S; ++r) {
                    u_val += K_tile[tok][r] * W[r];
                    qw_val += Q_tile[tok][r] * W[r];
                }
                u[tok] = u_val;
                qw[tok] = qw_val;
            }

            // 5. Form b_t = beta_t * (v_t - exp(gamma_t) * u_t)
            float b[GDN_CK];
#pragma unroll
            for (int tok = 0; tok < GDN_CK; ++tok) {
                const bool valid = (t0 + tok < T);
                const float v_val = valid ? h[(t0 + tok) * GDN_C + 2 * GDN_HK * GDN_S + vh * GDN_S + col] : 0.0f;
                const float exp_gamma = sycl::native::exp(gamma_tile[tok]);
                b[tok] = beta_tile[tok] * (v_val - exp_gamma * u[tok]);
            }

            // 6. Triangular solve delta = T * b in thread-private registers
            float delta[GDN_CK];
#pragma unroll
            for (int tok = 0; tok < GDN_CK; ++tok) {
                float d = 0.0f;
#pragma unroll
                for (int s = 0; s <= tok; ++s) {
                    d += T_tile[tok][s] * b[s];
                }
                delta[tok] = d;
            }

            // 7. Output o_t, delta_tilde, and fused norm
            const float sigma = gamma_tile[GDN_CK - 1];
            float delta_tilde[GDN_CK];
            float oc_tok[GDN_CK];
#pragma unroll
            for (int tok = 0; tok < GDN_CK; ++tok) {
                delta_tilde[tok] = sycl::native::exp(sigma - gamma_tile[tok]) * delta[tok];

                float o = sycl::native::exp(gamma_tile[tok]) * qw[tok];
#pragma unroll
                for (int s = 0; s <= tok; ++s) {
                    o += P_tile[tok][s] * delta[s];
                }
                const float oc_val = o * sycl::rsqrt((float)GDN_S);
                oc_tok[tok] = oc_val;

                if (!z && t0 + tok < T) {
                    oc[((size_t)(t0 + tok) * GDN_HV + vh) * GDN_S + col] = oc_val;
                }
            }

            // Fused RMSNorm and gating if z is provided
            if (z) {
#pragma unroll
                for (int tok = 0; tok < GDN_CK; ++tok) {
                    const float sq = (t0 + tok < T) ? (oc_tok[tok] * oc_tok[tok]) : 0.0f;
                    const float sg_sq = sg16_reduce_sum(sg, sq);
                    if (lane == 0) {
                        sn[tok][sg_id] = sg_sq;
                    }
                }

                item.barrier(sycl::access::fence_space::local_space);

#pragma unroll
                for (int tok = 0; tok < GDN_CK; ++tok) {
                    if (t0 + tok < T) {
                        const float total_sq = sn[tok][0] + sn[tok][1] + sn[tok][2] + sn[tok][3] +
                                               sn[tok][4] + sn[tok][5] + sn[tok][6] + sn[tok][7];
                        const float rs_val = sycl::rsqrt(total_sq / (float)GDN_S + eps);
                        const size_t at = ((size_t)(t0 + tok) * GDN_HV + vh) * GDN_S + col;
                        const float v = oc_tok[tok] * rs_val * g_col * sigm(z[at]);
                        if (oc) oc[at] = v;
                        if (y16) y16[(size_t)(t0 + tok) * ld16 + (size_t)vh * GDN_S + col] = hf(v);
                    }
                }
            }

            // 8. State update in thread-private registers
            const float exp_sigma = sycl::native::exp(sigma);
            for (int r = 0; r < GDN_S; ++r) {
                float delta_W = 0.0f;
#pragma unroll
                for (int tok = 0; tok < GDN_CK; ++tok) {
                    delta_W += K_tile[tok][r] * delta_tilde[tok];
                }
                W[r] = exp_sigma * W[r] + delta_W;
            }

            // Synchronize before any thread overwrites SLM for the next chunk
            item.barrier(sycl::access::fence_space::local_space);
        }

        // Write final resident state back to global memory
#pragma unroll
        for (int r = 0; r < GDN_S; ++r) {
            state[(size_t)r * rs + (size_t)vh * GDN_S + col] = W[r];
        }
    }
};

}  // namespace

bool gdn_chunk_available(void* stream) {
    const char* env = std::getenv("STRATA_GDN_CHUNKED");
    if (env && std::atoi(env) == 0) return false;

    if (!stream) return false;
    auto* q = static_cast<sycl::queue*>(stream);
    try {
        const auto device = q->get_device();
        if (!device.is_gpu()) return false;
        const auto subgroups = device.get_info<sycl::info::device::sub_group_sizes>();
        const bool sg16 = std::find(subgroups.begin(), subgroups.end(), size_t(16)) != subgroups.end();
        return sg16;
    } catch (...) {
        return false;
    }
}

void gdn_chunk_recurrence(float* state, const float* h, const float* gate, const float* beta, float* oc,
                          int64_t T, void* stream) {
    if (T <= 0 || !stream) return;
    auto* q = static_cast<sycl::queue*>(stream);
    const int64_t n_chunks = (T + GDN_CK - 1) / GDN_CK;

    auto& ws = get_workspace();
    ws.ensure((size_t)n_chunks, *q);

    // 1. Launch preparation kernel
    {
        const auto prep_global = sycl::range<2>((size_t)n_chunks, (size_t)GDN_HK * 48);
        const auto prep_local = sycl::range<2>(1, 48);
        q->submit([&](sycl::handler& cgh) {
            GdnChunkPrepFunctor functor{
                h, gate, beta, ws.ws_T, ws.ws_P, ws.ws_gamma, ws.ws_beta, T, n_chunks,
                sycl::local_accessor<float, 2>(sycl::range<2>(GDN_CK, GDN_S), cgh),
                sycl::local_accessor<float, 2>(sycl::range<2>(GDN_CK, GDN_S), cgh),
                sycl::local_accessor<float, 2>(sycl::range<2>(GDN_CK, GDN_CK), cgh),
                sycl::local_accessor<float, 2>(sycl::range<2>(GDN_CK, GDN_CK), cgh),
                sycl::local_accessor<float, 2>(sycl::range<2>(GDN_VPK, GDN_CK * GDN_CK), cgh),
                sycl::local_accessor<float, 2>(sycl::range<2>(GDN_VPK, GDN_CK), cgh),
                sycl::local_accessor<float, 2>(sycl::range<2>(GDN_VPK, GDN_CK), cgh)
            };
            cgh.parallel_for<class GdnChunkPrepKernel>(
                sycl::nd_range<2>(prep_global, prep_local),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                    functor(item);
                });
        });
    }

    // 2. Launch resident scan kernel (un-normed recurrence)
    {
        const auto scan_global = sycl::range<1>((size_t)GDN_HV * 128);
        const auto scan_local = sycl::range<1>(128);
        q->submit([&](sycl::handler& cgh) {
            GdnChunkScanFunctor functor{
                state, h, ws.ws_T, ws.ws_P, ws.ws_gamma, ws.ws_beta,
                nullptr, nullptr, 1e-6f, oc, nullptr, 0, T, n_chunks,
                sycl::local_accessor<float, 2>(sycl::range<2>(GDN_CK, GDN_S), cgh),
                sycl::local_accessor<float, 2>(sycl::range<2>(GDN_CK, GDN_S), cgh),
                sycl::local_accessor<float, 2>(sycl::range<2>(GDN_CK, GDN_CK), cgh),
                sycl::local_accessor<float, 2>(sycl::range<2>(GDN_CK, GDN_CK), cgh),
                sycl::local_accessor<float, 1>(sycl::range<1>(GDN_CK), cgh),
                sycl::local_accessor<float, 1>(sycl::range<1>(GDN_CK), cgh),
                sycl::local_accessor<float, 2>(sycl::range<2>(GDN_CK, 8), cgh)
            };
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wignored-attributes"
#endif
            cgh.parallel_for<class GdnChunkScanKernel>(
                sycl::nd_range<1>(scan_global, scan_local),
                sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>},
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                    functor(item);
                });
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
        });
    }
}

bool gdn_chunk_step(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream,
                    int64_t ld16) {
    if (T < 128) return false;
    const char* env = std::getenv("STRATA_GDN_CHUNKED");
    if (!env || std::atoi(env) == 0) return false;
    if (!gdn_chunk_available(stream)) return false;

    if (ld16 <= 0) ld16 = (int64_t)GDN_HV * GDN_S;
    auto* q = static_cast<sycl::queue*>(stream);
    const int64_t n_chunks = (T + GDN_CK - 1) / GDN_CK;

    auto& ws = get_workspace();
    ws.ensure((size_t)n_chunks, *q);

    // 1. Launch preparation kernel
    {
        const auto prep_global = sycl::range<2>((size_t)n_chunks, (size_t)GDN_HK * 48);
        const auto prep_local = sycl::range<2>(1, 48);
        q->submit([&](sycl::handler& cgh) {
            GdnChunkPrepFunctor functor{
                h, gate, beta, ws.ws_T, ws.ws_P, ws.ws_gamma, ws.ws_beta, T, n_chunks,
                sycl::local_accessor<float, 2>(sycl::range<2>(GDN_CK, GDN_S), cgh),
                sycl::local_accessor<float, 2>(sycl::range<2>(GDN_CK, GDN_S), cgh),
                sycl::local_accessor<float, 2>(sycl::range<2>(GDN_CK, GDN_CK), cgh),
                sycl::local_accessor<float, 2>(sycl::range<2>(GDN_CK, GDN_CK), cgh),
                sycl::local_accessor<float, 2>(sycl::range<2>(GDN_VPK, GDN_CK * GDN_CK), cgh),
                sycl::local_accessor<float, 2>(sycl::range<2>(GDN_VPK, GDN_CK), cgh),
                sycl::local_accessor<float, 2>(sycl::range<2>(GDN_VPK, GDN_CK), cgh)
            };
            cgh.parallel_for<class GdnChunkPrepKernelStep>(
                sycl::nd_range<2>(prep_global, prep_local),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                    functor(item);
                });
        });
    }

    // 2. Launch fused scan + norm kernel
    {
        const auto scan_global = sycl::range<1>((size_t)GDN_HV * 128);
        const auto scan_local = sycl::range<1>(128);
        q->submit([&](sycl::handler& cgh) {
            GdnChunkScanFunctor functor{
                state, h, ws.ws_T, ws.ws_P, ws.ws_gamma, ws.ws_beta,
                z, gamma, eps, y, y16, ld16, T, n_chunks,
                sycl::local_accessor<float, 2>(sycl::range<2>(GDN_CK, GDN_S), cgh),
                sycl::local_accessor<float, 2>(sycl::range<2>(GDN_CK, GDN_S), cgh),
                sycl::local_accessor<float, 2>(sycl::range<2>(GDN_CK, GDN_CK), cgh),
                sycl::local_accessor<float, 2>(sycl::range<2>(GDN_CK, GDN_CK), cgh),
                sycl::local_accessor<float, 1>(sycl::range<1>(GDN_CK), cgh),
                sycl::local_accessor<float, 1>(sycl::range<1>(GDN_CK), cgh),
                sycl::local_accessor<float, 2>(sycl::range<2>(GDN_CK, 8), cgh)
            };
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wignored-attributes"
#endif
            cgh.parallel_for<class GdnChunkScanFusedKernel>(
                sycl::nd_range<1>(scan_global, scan_local),
                sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>},
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                    functor(item);
                });
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
        });
    }
    return true;
}

}  // namespace strata::prefill
