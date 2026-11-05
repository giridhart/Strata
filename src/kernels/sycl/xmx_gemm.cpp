// src/kernels/sycl/xmx_gemm.cpp - see include/strata/kernels/xmx_gemm.hpp.
//
// 2D systolic XMX prefill GEMM with workgroup cache striping (GM=2) and GRF256.
// Computes Y[T, N] = X[T, K] . W[N, K]^T with FP32 accumulation and FP16/BF16 inputs.
// Each sub-group computes a 32x64 tile of Y using a 4x4 array of 8x16 hardware DPAS accumulators.
// W is read directly column-major straight from rows in memory without conversion or pack overhead.
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/core/sycl_arch.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/matrix/matrix-unified.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::kernels {
namespace {

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace ix = sycl::ext::intel::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
using bf16_t = sycl::ext::oneapi::bfloat16;

inline sycl::queue& default_queue() {
    static sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    return q;
}

constexpr int TM = 8, TN = 16, TK = 16;     // one XMX tile (FP16 and BF16)
constexpr int SGM = 4, SGN = 4;             // accumulators a sub-group
constexpr int MS = TM * SGM, NS = TN * SGN; // a sub-group's tile of Y: 32 x 64
constexpr int KC = 32;                     // K advance per loop iteration

template<typename E>
inline auto global(const E* p) {
    return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
        const_cast<E*>(p));
}

// One sub-group's tile at (m0, n0) of a WM x WN work-group tile at (wm0, wn0).
// CM / CN: the tile may cross the last row / column, so loads/stores use bounds checking.
// ACC: Y += product (compile-time specialization to avoid runtime overhead).
template<typename E, int WM, int WN, bool CM, bool CN, bool ACC>
inline void tile(sycl::sub_group sg, int sgid, const E* X, const E* W, float* Y, int64_t T, int64_t N, int64_t K,
                 int64_t ldy, int64_t wm0, int64_t wn0) {
    constexpr int NSG_N = WN / NS;
    const int64_t m0 = wm0 + (int64_t)(sgid / NSG_N) * MS;
    const int64_t n0 = wn0 + (int64_t)(sgid % NSG_N) * NS;

    int64_t bn[SGN];
    for (int j = 0; j < SGN; ++j) {
        bn[j] = CN ? sycl::min(n0 + (int64_t)j * TN, N - TN) : n0 + (int64_t)j * TN;
    }

    const auto gx = global(X), gw = global(W);
    const auto gy = global(Y);
    mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> c[SGM][SGN];

    for (int i = 0; i < SGM; ++i) {
        for (int j = 0; j < SGN; ++j) {
            if constexpr (!ACC) {
                mx::joint_matrix_fill(sg, c[i][j], 0.0f);
            } else if constexpr (CM || CN) {
                ix::joint_matrix_load_checked(sg, c[i][j], gy, ldy, mx::layout::row_major, T, N,
                                              m0 + (int64_t)i * TM, n0 + (int64_t)j * TN);
            } else {
                mx::joint_matrix_load(sg, c[i][j], gy + (m0 + (int64_t)i * TM) * ldy + n0 + (int64_t)j * TN, ldy,
                                      mx::layout::row_major);
            }
        }
    }

    for (int64_t k0 = 0; k0 < K; k0 += KC) {
        for (int kk = 0; kk < KC; kk += TK) {
            const int64_t k = k0 + kk;
            mx::joint_matrix<sycl::sub_group, E, mx::use::a, TM, TK, mx::layout::row_major> a[SGM];
            mx::joint_matrix<sycl::sub_group, E, mx::use::b, TK, TN, mx::layout::col_major> b[SGN];

            for (int i = 0; i < SGM; ++i) {
                if constexpr (CM) {
                    ix::joint_matrix_load_checked(sg, a[i], gx, K, T, K, m0 + (int64_t)i * TM, k);
                } else {
                    mx::joint_matrix_load(sg, a[i], gx + (m0 + (int64_t)i * TM) * K + k, K);
                }
            }
            for (int j = 0; j < SGN; ++j) {
                mx::joint_matrix_load(sg, b[j], gw + bn[j] * K + k, K);
            }
            for (int i = 0; i < SGM; ++i) {
                for (int j = 0; j < SGN; ++j) {
                    mx::joint_matrix_mad(sg, c[i][j], a[i], b[j], c[i][j]);
                }
            }
        }
    }

    for (int i = 0; i < SGM; ++i) {
        for (int j = 0; j < SGN; ++j) {
            if constexpr (CM || CN) {
                ix::joint_matrix_store_checked(sg, c[i][j], gy, ldy, mx::layout::row_major, T, N,
                                               m0 + (int64_t)i * TM, n0 + (int64_t)j * TN);
            } else {
                mx::joint_matrix_store(sg, c[i][j], gy + (m0 + (int64_t)i * TM) * ldy + n0 + (int64_t)j * TN, ldy,
                                       mx::layout::row_major);
            }
        }
    }
}

template<typename E, int WM, int WN, bool ACC = false>
inline void tile_any(sycl::sub_group sg, int sgid, const E* X, const E* W, float* Y, int64_t T, int64_t N, int64_t K,
                     int64_t ldy, int64_t wm0, int64_t wn0) {
    if (wm0 + WM <= T && wn0 + WN <= N) {
        tile<E, WM, WN, false, false, ACC>(sg, sgid, X, W, Y, T, N, K, ldy, wm0, wn0);
    } else {
        tile<E, WM, WN, true, true, ACC>(sg, sgid, X, W, Y, T, N, K, ldy, wm0, wn0);
    }
}

constexpr int GM = 2; // row tiles per cache-striping group

#define STRATA_XMX_PROPERTIES                                                                                          \
    auto get(syclex::properties_tag) const {                                                                           \
        return syclex::properties{sycl::ext::intel::experimental::grf_size<256>, syclex::sub_group_size<16>};         \
    }

template<typename E, int WM, int WN, bool ACC = false>
struct GemmKernel {
    const E* X;
    const E* W;
    float* Y;
    int64_t T, N, K, ldy, nM, nN;
    void operator()(sycl::nd_item<1> it) const {
        const auto sg = it.get_sub_group();
        const int64_t g = (int64_t)it.get_group(0);
        const int64_t strip = g / (GM * nN), r = g % (GM * nN);
        const int64_t gm = std::min<int64_t>(GM, nM - strip * GM);
        const int64_t tm = strip * GM + r % gm, tn = r / gm;
        tile_any<E, WM, WN, ACC>(sg, (int)sg.get_group_linear_id(), X, W, Y, T, N, K, ldy, tm * WM, tn * WN);
    }
    STRATA_XMX_PROPERTIES
};

template<int WM, int WN>
struct GroupedKernel {
    const sycl::half* X;
    const sycl::half* W;
    float* Y;
    const int32_t* bounds;
    int64_t w_stride, tiles_m, nN, N, K;
    void operator()(sycl::nd_item<1> it) const {
        const auto sg = it.get_sub_group();
        const int64_t g = (int64_t)it.get_group(0);
        const int64_t ex = g / (tiles_m * nN), r = g % (tiles_m * nN), tm = r / nN, tn = r % nN;
        const int64_t row0 = bounds[ex], rows = bounds[ex + 1] - row0;
        if (tm * WM >= rows) return;
        tile_any<sycl::half, WM, WN>(sg, (int)sg.get_group_linear_id(), X + row0 * K, W + ex * w_stride,
                                     Y + row0 * N, rows, N, K, N, tm * WM, tn * WN);
    }
    STRATA_XMX_PROPERTIES
};
#undef STRATA_XMX_PROPERTIES

template<typename E, int WM, int WN, bool ACC = false>
sycl::event launch(sycl::queue& q, const E* X, const E* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy) {
    constexpr int NSG = (WM / MS) * (WN / NS);
    const int64_t nM = (T + WM - 1) / WM, nN = (N + WN - 1) / WN;
    return q.parallel_for(sycl::nd_range<1>((size_t)(nM * nN * NSG * 16), (size_t)NSG * 16),
                          GemmKernel<E, WM, WN, ACC>{X, W, Y, T, N, K, ldy, nM, nN});
}

inline int64_t fill_rows(const sycl::queue& q, int64_t b70_rows) {
    const int64_t cu = (int64_t)q.get_device().get_info<sycl::info::device::max_compute_units>();
    return std::max<int64_t>(1, b70_rows * cu / 256);
}

template<typename E>
sycl::event dispatch(sycl::queue& q, const E* X, const E* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy) {
    const int64_t many = fill_rows(q, 4096), few = fill_rows(q, 1024);
    if (N <= 64) return launch<E, 256, 64>(q, X, W, Y, T, N, K, ldy);
    if (T >= many) {
        if (N <= 128) return launch<E, 256, 128>(q, X, W, Y, T, N, K, ldy);
        if (N <= 384 && K >= 4096) return launch<E, 128, 512>(q, X, W, Y, T, N, K, ldy);
        return launch<E, 256, 256>(q, X, W, Y, T, N, K, ldy);
    }
    if (N <= 512 || (T < few && N < 2048)) return launch<E, 128, 256>(q, X, W, Y, T, N, K, ldy);
    return launch<E, 256, 256>(q, X, W, Y, T, N, K, ldy);
}

bool device_has_xmx(const sycl::device& d, mx::matrix_type in) {
    if (const char* v = std::getenv("STRATA_NO_XMX"); v != nullptr && std::strtol(v, nullptr, 10) != 0) return false;
    const auto sg = d.get_info<sycl::info::device::sub_group_sizes>();
    if (std::find(sg.begin(), sg.end(), size_t(16)) == sg.end()) return false;
    try {
        const auto combos = d.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
        for (const auto& c : combos) {
            const bool m = c.msize == TM || (c.msize == 0 && c.max_msize >= TM);
            const bool n = c.nsize == TN || (c.nsize == 0 && c.max_nsize >= TN);
            const bool k = c.ksize == TK || (c.ksize == 0 && c.max_ksize >= TK);
            if (c.atype == in && c.btype == in && c.ctype == mx::matrix_type::fp32 && c.dtype == mx::matrix_type::fp32 &&
                m && n && k)
                return true;
        }
    } catch (...) {
        return false;
    }
    return false;
}

bool large_grf_ok(const sycl::device& d, const sycl::context& ctx) {
    try {
        (void)sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            ctx, {d}, {sycl::get_kernel_id<GemmKernel<sycl::half, 128, 256>>()});
        return true;
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "strata: the GPU does not build kernels with large register file (%s): fallback used\n",
                     e.what());
        return false;
    }
}

struct XmxUse {
    bool f16 = false;
    bool bf16 = false;
};

std::mutex g_xmx_mutex;
std::unordered_map<sycl::device, XmxUse> g_xmx_use;

bool use_xmx(const sycl::device& dev, const sycl::context& ctx, XmxType t) {
    std::lock_guard<std::mutex> lock(g_xmx_mutex);
    auto it = g_xmx_use.find(dev);
    if (it != g_xmx_use.end()) {
        return t == XmxType::f16 ? it->second.f16 : it->second.bf16;
    }
    const bool f16_xmx = device_has_xmx(dev, mx::matrix_type::fp16);
    const bool bf16_xmx = device_has_xmx(dev, mx::matrix_type::bf16);
    const bool grf = (f16_xmx || bf16_xmx) && large_grf_ok(dev, ctx);
    XmxUse u{grf && f16_xmx, grf && bf16_xmx};
    g_xmx_use[dev] = u;
    return t == XmxType::f16 ? u.f16 : u.bf16;
}

template<int SG>
sycl::event rows_kernel(sycl::queue& q, bool half, const uint16_t* X, const uint16_t* W, float* Y, int64_t T,
                        int64_t N, int64_t K, int64_t ldy, bool accumulate) {
    constexpr int NB = 16;
    const bool vec = K % 8 == 0 && ((uintptr_t)X | (uintptr_t)W) % 16 == 0;
    const int64_t nb = (N + NB - 1) / NB;
    return q.parallel_for(
        sycl::nd_range<2>({(size_t)nb, (size_t)T * SG}, {1, (size_t)SG}),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const auto sg = it.get_sub_group();
            const int64_t r = (int64_t)it.get_group(1), n0 = (int64_t)it.get_group(0) * NB;
            const int lane = (int)sg.get_local_linear_id();
            const auto val = [half](uint16_t b) {
                return half ? (float)sycl::bit_cast<sycl::half>(b) : strata::kernels::f32_from_bf16(b);
            };
            float acc[NB] = {};
            const uint16_t* x = X + r * K;
            if (vec) {
                using V = sycl::vec<uint16_t, 8>;
                for (int64_t k = (int64_t)lane * 8; k < K; k += SG * 8ll) {
                    const V xv = *reinterpret_cast<const V*>(x + k);
                    for (int n = 0; n < NB; ++n) {
                        if (n0 + n >= N) break;
                        const V wv = *reinterpret_cast<const V*>(W + (n0 + n) * K + k);
                        for (int j = 0; j < 8; ++j) acc[n] = sycl::fma(val(xv[j]), val(wv[j]), acc[n]);
                    }
                }
            } else {
                for (int64_t k = lane; k < K; k += SG) {
                    for (int n = 0; n < NB; ++n) {
                        if (n0 + n >= N) break;
                        acc[n] = sycl::fma(val(x[k]), val(W[(n0 + n) * K + k]), acc[n]);
                    }
                }
            }
            for (int n = 0; n < NB; ++n) {
                if (n0 + n >= N) break;
                const float s = sycl::reduce_over_group(sg, acc[n], sycl::plus<float>());
                if (lane == 0) Y[r * ldy + n0 + n] = accumulate ? Y[r * ldy + n0 + n] + s : s;
            }
        });
}

}  // namespace

bool xmx_available(XmxType t) {
    auto& q = default_queue();
    return use_xmx(q.get_device(), q.get_context(), t);
}

bool xmx_available(void* stream, XmxType t) {
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    return use_xmx(q->get_device(), q->get_context(), t);
}

bool f16_matrix_engines() {
    return xmx_available(XmxType::f16);
}

bool f16_only_matrix_engines() {
    return xmx_available(XmxType::f16) && !xmx_available(XmxType::bf16);
}

const char* gemm_path(XmxType t) {
    return xmx_available(t) ? "XMX" : "Scalar";
}

bool xmx_gemm_ok(const void* X, const void* W, const float* Y, int64_t N, int64_t K, int64_t ldy) noexcept {
    const auto al = [](const void* p) { return ((uintptr_t)p & 63) == 0; };
    return N >= 16 && N % 16 == 0 && K >= 32 && K % 32 == 0 && ldy >= N && ldy % 16 == 0 && al(X) && al(W) && al(Y);
}

void gemm_rows(XmxType t, const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               void* stream, bool accumulate) {
    if (T <= 0 || N <= 0) return;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    const bool half = t == XmxType::f16;
    const auto sg_sizes = q->get_device().get_info<sycl::info::device::sub_group_sizes>();
    const int sg = std::find(sg_sizes.begin(), sg_sizes.end(), size_t(16)) != sg_sizes.end() ? 16 : 32;
    if (sg == 16) {
        rows_kernel<16>(*q, half, X, W, Y, T, N, K, ldy, accumulate);
    } else {
        rows_kernel<32>(*q, half, X, W, Y, T, N, K, ldy, accumulate);
    }
}

void xmx_gemm(XmxType t, const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
              void* stream, bool accumulate) {
    if (T <= 0 || N <= 0) return;
    if (!xmx_gemm_ok(X, W, Y, N, K, ldy)) {
        gemm_rows(t, X, W, Y, T, N, K, ldy, stream, accumulate);
        return;
    }
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    if (!use_xmx(q->get_device(), q->get_context(), t)) {
        gemm_rows(t, X, W, Y, T, N, K, ldy, stream, accumulate);
        return;
    }
    if (accumulate) {
        if (t == XmxType::bf16) {
            launch<bf16_t, 128, 256, true>(*q, reinterpret_cast<const bf16_t*>(X),
                                           reinterpret_cast<const bf16_t*>(W), Y, T, N, K, ldy);
        } else {
            launch<sycl::half, 128, 256, true>(*q, reinterpret_cast<const sycl::half*>(X),
                                               reinterpret_cast<const sycl::half*>(W), Y, T, N, K, ldy);
        }
        return;
    }
    if (t == XmxType::f16) {
        dispatch(*q, reinterpret_cast<const sycl::half*>(X), reinterpret_cast<const sycl::half*>(W), Y, T, N, K, ldy);
    } else {
        dispatch(*q, reinterpret_cast<const bf16_t*>(X), reinterpret_cast<const bf16_t*>(W), Y, T, N, K, ldy);
    }
}

void xmx_gemm_grouped(const uint16_t* X, const uint16_t* W, int64_t w_stride, float* Y, const int32_t* bounds, int G,
                      int64_t max_rows, int64_t N, int64_t K, void* stream) {
    if (G <= 0 || max_rows <= 0) return;
    if (!xmx_gemm_ok(X, W, Y, N, K, N) || (w_stride * 2) % 64 != 0) return;
    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue();
    if (!xmx_available(stream, XmxType::f16)) return;
    constexpr int WM = 128, WN = 256, NSG = (WM / MS) * (WN / NS);
    const int64_t tiles_m = (max_rows + WM - 1) / WM, nN = (N + WN - 1) / WN;
    q->parallel_for(
        sycl::nd_range<1>((size_t)(G * tiles_m * nN * NSG * 16), (size_t)NSG * 16),
        GroupedKernel<WM, WN>{reinterpret_cast<const sycl::half*>(X), reinterpret_cast<const sycl::half*>(W), Y,
                              bounds, w_stride, tiles_m, nN, N, K});
}

}  // namespace strata::kernels
