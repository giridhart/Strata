// include/strata/kernels/xmx_gemm.hpp - matrix products on Intel XMX systolic hardware (SYCL joint_matrix).
//
// Y[T, N] (fp32, row stride ldy) = X[T, K] . W[N, K]^T with X and W row-major in FP16 or BF16 and FP32 accumulation.
#pragma once

#include <cstdint>

namespace strata::kernels {

enum class XmxType { f16, bf16 };

/// Whether the compute device runs the XMX kernels for inputs of type `t` (joint_matrix 8 x 16 x 16 with FP32
/// accumulators and 16-wide sub-groups; STRATA_NO_XMX=1 says no).
bool xmx_available(XmxType t);
bool xmx_available(void* stream, XmxType t);

/// Whether the compute device runs FP16 products on its matrix engines.
bool f16_matrix_engines();

/// Whether the compute device's matrix engines take FP16 inputs and not BF16 ones.
bool f16_only_matrix_engines();

/// The path xmx_gemm takes for inputs of type `t` on the compute device.
const char* gemm_path(XmxType t);

/// Whether xmx_gemm takes the product: N a multiple of 16, K of 32, ldy of 16, and X, W, Y 64-byte aligned (the 2D
/// block loads and stores). The others go to gemm_rows or scalar fallback.
bool xmx_gemm_ok(const void* X, const void* W, const float* Y, int64_t N, int64_t K, int64_t ldy) noexcept;

/// `accumulate`: Y += X . W^T (Y read before the product is added, in FP32).
void xmx_gemm(XmxType t, const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
              void* stream, bool accumulate = false);

/// The products of G experts in one launch (FP16): expert e multiplies rows bounds[e] .. bounds[e + 1] of X (and
/// writes them in Y, ldy = N) by its weights at W + e * w_stride. `bounds` is on the device; `max_rows` bounds every
/// expert's row count (it sizes the launch).
void xmx_gemm_grouped(const uint16_t* X, const uint16_t* W, int64_t w_stride, float* Y, const int32_t* bounds, int G,
                      int64_t max_rows, int64_t N, int64_t K, void* stream);

/// Y = X . W^T, one sub-group an output: the small products xmx_gemm does not take.
void gemm_rows(XmxType t, const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               void* stream, bool accumulate = false);

}  // namespace strata::kernels
