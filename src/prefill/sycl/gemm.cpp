// src/prefill/gemm.cu - see include/strata/prefill/gemm.hpp.
#include <sycl/sycl.hpp>
#include "strata/core/sycl_arch.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
// The HIP compatibility shim maps CUDA shuffle spellings to Strata helpers.
// hipBLASLt's public headers declare native HIP shuffle functions, so keep
// those declarations from being macro-expanded in this translation unit.
#undef __shfl_xor_sync
#undef __shfl_down_sync
#undef __shfl_up_sync
#undef __shfl_sync
#undef __ballot_sync
#endif

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
#include "hipblaslt_tuning.hpp"
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <map>
#include <set>
#include <tuple>
#endif

namespace strata::prefill {
namespace {

// #247/#325: on Windows (seen on gfx1201), hipBLAS can return success with the correct BF16/FP16 product for some

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
struct HipLtCallKey {
    strata::prefill::hipblaslt::InputType type;
    int t;
    int n;
    int k;
    int ldy;
    uint32_t beta_bits;

    bool operator<(const HipLtCallKey& other) const {
        return std::tie(type, n, k, ldy, t, beta_bits) <
               std::tie(other.type, other.n, other.k, other.ldy, other.t, other.beta_bits);
    }
};

struct HipLtCachedAlgo {
    bool supported = false;
    hipblasLtMatmulAlgo_t algo{};
    size_t workspace_bytes = 0;
};

struct HipLtState {
    hipblasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    strata::prefill::hipblaslt::TuningTable table;
    std::map<HipLtCallKey, HipLtCachedAlgo> cache;
    uint64_t lt_launches = 0;
    uint64_t fallbacks = 0;
    std::set<std::tuple<strata::prefill::hipblaslt::InputType, int, int, int, int>> fallback_shapes;

    ~HipLtState() {
        if (std::getenv("STRATA_HIPBLASLT_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: hipBLASLt summary launches=%llu fallbacks=%llu unique_fallback_shapes=%zu\n",
                         (unsigned long long) lt_launches, (unsigned long long) fallbacks, fallback_shapes.size());
            for (const auto& shape : fallback_shapes) {
                const auto type = std::get<0>(shape);
                std::fprintf(stderr, "prefill gemm: fallback shape dtype=%s T=%d N=%d K=%d ldy=%d\n",
                             type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16",
                             std::get<1>(shape), std::get<2>(shape), std::get<3>(shape), std::get<4>(shape));
            }
        }
        if (handle) hipblasLtDestroy(handle);
    }
};

struct HipLtDescriptors {
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a = nullptr;
    hipblasLtMatrixLayout_t b = nullptr;
    hipblasLtMatrixLayout_t c = nullptr;

    ~HipLtDescriptors() {
        if (op) hipblasLtMatmulDescDestroy(op);
        if (a) hipblasLtMatrixLayoutDestroy(a);
        if (b) hipblasLtMatrixLayoutDestroy(b);
        if (c) hipblasLtMatrixLayoutDestroy(c);
    }

    bool init(hipDataType type, int t, int n, int k, int ldy) {
        const hipblasOperation_t trans_a = HIPBLAS_OP_T;
        const hipblasOperation_t trans_b = HIPBLAS_OP_N;
        if (hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)) !=
                HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)) !=
                HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&a, type, k, n, k) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&b, type, k, t, k) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&c, HIP_R_32F, n, t, ldy) != HIPBLAS_STATUS_SUCCESS) {
            return false;
        }
        return true;
    }
};

std::unique_ptr<HipLtState> create_hipblaslt_state(void* workspace, size_t workspace_bytes) {
    const char* path = std::getenv("STRATA_HIPBLASLT_TUNING");
    if (!path || !*path) return nullptr;

    auto state = std::make_unique<HipLtState>();
    state->workspace = workspace;
    state->workspace_bytes = workspace_bytes;
    if (hipblasLtCreate(&state->handle) != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: hipBLASLt handle creation failed; using hipBLASEx\n");
        return nullptr;
    }

    int version = 0;
    if (hipblasLtGetVersion(state->handle, &version) != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: hipBLASLt version query failed; using hipBLASEx\n");
        return nullptr;
    }
    int device = 0;
    hipDeviceProp_t properties{};
    if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&properties, device) != hipSuccess) {
        std::fprintf(stderr, "prefill gemm: HIP device query failed; using hipBLASEx\n");
        return nullptr;
    }
    std::string arch(properties.gcnArchName);
    const auto suffix = arch.find(':');
    if (suffix != std::string::npos) arch.resize(suffix);

    std::string error;
    if (!state->table.load(path, arch, version, error)) {
        std::fprintf(stderr, "prefill gemm: %s; using hipBLASEx\n", error.c_str());
        return nullptr;
    }
    std::fprintf(stderr, "prefill gemm: hipBLASLt tuning enabled (%zu rows, %s, version %d)\n",
                 state->table.rows().size(), arch.c_str(), version);
    return state;
}

HipLtCachedAlgo resolve_hipblaslt_algo(HipLtState& state, strata::prefill::hipblaslt::InputType type, int t,
                                       int n, int k, int ldy, float beta) {
    uint32_t beta_bits = 0;
    static_assert(sizeof(beta_bits) == sizeof(beta));
    std::memcpy(&beta_bits, &beta, sizeof(beta));
    const HipLtCallKey key{type, t, n, k, ldy, beta_bits};
    const auto cached = state.cache.find(key);
    if (cached != state.cache.end()) return cached->second;

    HipLtCachedAlgo resolved;
    const bool verbose = std::getenv("STRATA_HIPBLASLT_VERBOSE") != nullptr;
    const auto* row = state.table.closest(type, n, k, ldy, t);
    if (!row) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; no calibration for dtype=%s T=%d N=%d K=%d ldy=%d\n",
                         type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16", t, n, k, ldy);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    HipLtDescriptors desc;
    const hipDataType input_type = type == strata::prefill::hipblaslt::InputType::bf16 ? HIP_R_16BF : HIP_R_16F;
    if (!desc.init(input_type, t, n, k, ldy)) {
        return state.cache.emplace(key, resolved).first->second;
    }

    std::vector<int> solution_ids{row->solution_id};
    std::vector<hipblasLtMatmulHeuristicResult_t> candidates;
    if (hipblaslt_ext::getAlgosFromIndex(state.handle, solution_ids, candidates) != HIPBLAS_STATUS_SUCCESS ||
        candidates.empty() || candidates.front().state != HIPBLAS_STATUS_SUCCESS ||
        hipblaslt_ext::getIndexFromAlgo(candidates.front().algo) != row->solution_id) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution %d unavailable for T=%d N=%d K=%d ldy=%d\n",
                         row->solution_id, t, n, k, ldy);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    const float alpha = 1.0f;
    size_t required_workspace = 0;
    auto algo = candidates.front().algo;
    if (hipblaslt_ext::matmulIsAlgoSupported(state.handle, desc.op, &alpha, desc.a, desc.b, &beta, desc.c, desc.c,
                                             algo, required_workspace) != HIPBLAS_STATUS_SUCCESS) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution %d rejects actual T=%d N=%d K=%d ldy=%d beta=%.9g\n",
                         row->solution_id, t, n, k, ldy, beta);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    resolved.supported = true;
    resolved.algo = algo;
    resolved.workspace_bytes = required_workspace;
    if (verbose) {
        std::fprintf(stderr,
                     "prefill gemm: Lt solution=%d dtype=%s T=%d N=%d K=%d ldy=%d beta=%.9g workspace=%zu\n",
                     row->solution_id, type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16", t, n,
                     k, ldy, beta, required_workspace);
    }
    return state.cache.emplace(key, resolved).first->second;
}

bool try_hipblaslt(void* opaque_state, strata::prefill::hipblaslt::InputType type, const uint16_t* x,
                   const uint16_t* w, float* y, int64_t t, int64_t n, int64_t k, int64_t ldy, float beta,
                   void* stream) {
    auto* state = static_cast<HipLtState*>(opaque_state);
    if (!state || t <= 0 || n <= 0 || k <= 0 || t > INT_MAX || n > INT_MAX || k > INT_MAX || ldy > INT_MAX ||
        ldy < n) {
        return false;
    }
    const auto resolved = resolve_hipblaslt_algo(*state, type, (int) t, (int) n, (int) k, (int) ldy, beta);
    if (!resolved.supported) {
        ++state->fallbacks;
        state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
        return false;
    }
    if (resolved.workspace_bytes > state->workspace_bytes) {
        ++state->fallbacks;
        state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
        if (std::getenv("STRATA_HIPBLASLT_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution needs %zu workspace bytes, have %zu\n",
                         resolved.workspace_bytes, state->workspace_bytes);
        }
        return false;
    }

    HipLtDescriptors desc;
    const hipDataType input_type = type == strata::prefill::hipblaslt::InputType::bf16 ? HIP_R_16BF : HIP_R_16F;
    if (!desc.init(input_type, (int) t, (int) n, (int) k, (int) ldy)) return false;
    const float alpha = 1.0f;
    const hipblasStatus_t status = hipblasLtMatmul(state->handle, desc.op, &alpha, w, desc.a, x, desc.b, &beta, y,
                                                   desc.c, y, desc.c, &resolved.algo, state->workspace,
                                                   state->workspace_bytes, (hipStream_t) stream);
    if (status == HIPBLAS_STATUS_SUCCESS) {
        ++state->lt_launches;
        return true;
    }

    std::fprintf(stderr, "prefill gemm: hipBLASLt launch failed with status %d\n", (int) status);
    if (beta != 0.0f) {
        std::fprintf(stderr, "prefill gemm: refusing a fallback after hipBLASLt failed with nonzero beta\n");
        std::exit(1);
    }
    auto* mutable_state = static_cast<HipLtState*>(opaque_state);
    uint32_t beta_bits = 0;
    std::memcpy(&beta_bits, &beta, sizeof(beta_bits));
    auto cached = mutable_state->cache.find(HipLtCallKey{type, (int) t, (int) n, (int) k, (int) ldy, beta_bits});
    if (cached != mutable_state->cache.end()) cached->second.supported = false;
    ++mutable_state->fallbacks;
    mutable_state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
    return false;
}
#endif

}  // namespace

Gemm::~Gemm() {
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    delete static_cast<HipLtState*>(hipblaslt_state_);
#endif
    if (!external_) {
        auto* q = static_cast<sycl::queue*>(stream_);
        if (q) {
            if (scratch_) sycl::free(scratch_, *q);
            if (workspace_) sycl::free(workspace_, *q);
        }
    }
}

bool Gemm::init_external(void *stream, uint16_t *scratch, int64_t scratch_elems,
                         void *workspace, size_t ws_bytes,
                         std::string &err) {
    (void)ws_bytes;
    (void)err;
    stream_ = stream;
    handle_ = stream;
    external_ = true;
    workspace_ = workspace;
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    hipblaslt_state_ = create_hipblaslt_state(workspace_, ws_bytes).release();
#endif
    return true;
}

void Gemm::rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes) {
    (void)ws_bytes;
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    workspace_ = workspace;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (hipblaslt_state_) {
        auto* state = static_cast<HipLtState*>(hipblaslt_state_);
        state->workspace = workspace_;
        state->workspace_bytes = ws_bytes;
    }
#endif
}

bool Gemm::init(void *stream, int64_t scratch_elems, std::string &err) {
    auto* q = static_cast<sycl::queue*>(stream);
    if (!q) {
        err = "prefill gemm: invalid sycl queue pointer";
        return false;
    }
    stream_ = stream;
    handle_ = stream;
    const size_t ws = 32u << 20;
    try {
        workspace_ = sycl::malloc_device(ws, *q);
        if (!workspace_) {
            err = "prefill gemm: workspace allocation of 32 MiB failed";
            return false;
        }
        if (scratch_elems > 0) {
            scratch_ = (uint16_t*)sycl::malloc_device((size_t)scratch_elems * 2, *q);
            if (!scratch_) {
                err = "prefill gemm: dequant scratch of " +
                      std::to_string(scratch_elems * 2 >> 20) + " MiB failed";
                return false;
            }
        }
    } catch (const sycl::exception& exc) {
        err = std::string("prefill gemm: ") + exc.what();
        return false;
    }
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    hipblaslt_state_ = create_hipblaslt_state(workspace_, ws).release();
#endif
    scratch_elems_ = scratch_elems;
    return true;
}

template <bool IS_BF16>
void sycl_prefill_gemm_kernel(const uint16_t* __restrict__ X,
                              const uint16_t* __restrict__ W,
                              float* __restrict__ Y,
                              int64_t T, int64_t N, int64_t K, int64_t ldy,
                              float alpha, float beta,
                              sycl::nd_item<2> item) {
    const int64_t t = (int64_t)item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    const int64_t n = (int64_t)item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
    if (t >= T || n >= N) return;

    const uint16_t* x_row = X + t * K;
    const uint16_t* w_row = W + n * K;

    float sum = 0.0f;
    int64_t k = 0;
    #pragma unroll 4
    for (; k + 4 <= K; k += 4) {
        if constexpr (IS_BF16) {
            sum += strata::kernels::f32_from_bf16(x_row[k+0]) * strata::kernels::f32_from_bf16(w_row[k+0])
                 + strata::kernels::f32_from_bf16(x_row[k+1]) * strata::kernels::f32_from_bf16(w_row[k+1])
                 + strata::kernels::f32_from_bf16(x_row[k+2]) * strata::kernels::f32_from_bf16(w_row[k+2])
                 + strata::kernels::f32_from_bf16(x_row[k+3]) * strata::kernels::f32_from_bf16(w_row[k+3]);
        } else {
            sum += static_cast<float>(sycl::bit_cast<sycl::half>(x_row[k+0])) * static_cast<float>(sycl::bit_cast<sycl::half>(w_row[k+0]))
                 + static_cast<float>(sycl::bit_cast<sycl::half>(x_row[k+1])) * static_cast<float>(sycl::bit_cast<sycl::half>(w_row[k+1]))
                 + static_cast<float>(sycl::bit_cast<sycl::half>(x_row[k+2])) * static_cast<float>(sycl::bit_cast<sycl::half>(w_row[k+2]))
                 + static_cast<float>(sycl::bit_cast<sycl::half>(x_row[k+3])) * static_cast<float>(sycl::bit_cast<sycl::half>(w_row[k+3]));
        }
    }
    for (; k < K; ++k) {
        if constexpr (IS_BF16) {
            sum += strata::kernels::f32_from_bf16(x_row[k]) * strata::kernels::f32_from_bf16(w_row[k]);
        } else {
            sum += static_cast<float>(sycl::bit_cast<sycl::half>(x_row[k])) * static_cast<float>(sycl::bit_cast<sycl::half>(w_row[k]));
        }
    }

    const int64_t y_idx = t * ldy + n;
    if (beta == 0.0f) {
        Y[y_idx] = alpha * sum;
    } else {
        Y[y_idx] = alpha * sum + beta * Y[y_idx];
    }
}

void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta) {
    if (T <= 0 || N <= 0 || K <= 0) return;
    if (ldy <= 0) ldy = N;
    auto* q = static_cast<sycl::queue*>(stream_);
    if (!q) return;

    if (T == 1 && beta == 0.0f && ldy == N) {
        strata::kernels::bf16_gemv(X, W, Y, K, N, stream_);
        return;
    }

    const size_t local_t = 1;
    const size_t local_n = 256;
    const size_t global_t = (size_t)T;
    const size_t global_n = ((size_t)N + local_n - 1) / local_n * local_n;
    q->parallel_for(
        sycl::nd_range<2>(sycl::range<2>(global_t, global_n), sycl::range<2>(local_t, local_n)),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(strata::sycl_arch::SUB_GROUP_SIZE)]] {
            sycl_prefill_gemm_kernel<true>(X, W, Y, T, N, K, ldy, 1.0f, beta, item);
        });
}

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta) {
    if (T <= 0 || N <= 0 || K <= 0) return;
    if (ldy <= 0) ldy = N;
    auto* q = static_cast<sycl::queue*>(stream_);
    if (!q) return;

    const size_t local_t = 1;
    const size_t local_n = 256;
    const size_t global_t = (size_t)T;
    const size_t global_n = ((size_t)N + local_n - 1) / local_n * local_n;
    q->parallel_for(
        sycl::nd_range<2>(sycl::range<2>(global_t, global_n), sycl::range<2>(local_t, local_n)),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(strata::sycl_arch::SUB_GROUP_SIZE)]] {
            sycl_prefill_gemm_kernel<false>(X, W, Y, T, N, K, ldy, 1.0f, beta, item);
        });
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta) {
    if (N * K > scratch_elems_) {
        // Too large for the scratch at once: in row slices.
        const int64_t rows = scratch_elems_ / K;
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            f16(X, scratch_, Y + r0, T, n, K, ldy, beta);
        }
        return;
    }
    strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
    f16(X, scratch_, Y, T, N, K, ldy, beta);
}

}  // namespace strata::prefill
