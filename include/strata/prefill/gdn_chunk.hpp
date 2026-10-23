// include/strata/prefill/gdn_chunk.hpp - the chunked GDN prompt recurrence (FlashQLA gdn-chunk).
//
// The prompt recurrence (`gdn_recurrence`, src/prefill/kernels.cu) walks the chunk's tokens one by one; each
// token depends on the state left by the previous one, so the whole launch is a serial chain of T steps per
// (head, column-block).
//
// The chunked form (the FlashQLA "gdn-chunk" structure of 1Cat-vLLM: cumsum -> kkt -> fwd, 64-token chunks)
// trades the token chain for a CHUNK chain: inside one 64-token chunk everything runs as dense matmuls and one
// triangular solve, and only the chunks stay in order (each needs the previous chunk's state). Same recurrence,
// different summation order - so FP32-level, NOT bit-exact against `gdn_recurrence`.
//
// THE MATH, derived from the recurrence in `strata/kernels/gdn.hpp` (per head; the state columns j are
// independent, so everything below is per column and can be written in matrix form over the i axis):
//
//     a_t = exp(gate_t)                              (decay)
//     W_t = a_t W_{t-1} + k_t delta_t^T              (state, i x j)
//     delta_t = beta_t (v_t - a_t k_t^T W_{t-1})
//     o_t = q_t^T W_t                                (then * rsqrtf(S) by the caller-side convention)
//
// Unrolled over a chunk (c_t = inclusive cumsum of gate inside the chunk, sigma = c_63, W_in = state at the
// chunk's first token - one step per token, no approximations):
//
//     W_t   = exp(c_t) W_in + sum_{s<=t} exp(c_t-c_s) k_s delta_s^T
//     x_t   := a_t k_t^T W_{t-1}
//           = exp(c_t) k_t^T W_in + sum_{s<t} exp(c_t-c_s) (k_t.k_s) delta_s
//     delta_t = beta_t (v_t - x_t)
//
// which is one lower-triangular solve per chunk and column:
//
//     (I + strict_lower(A)) delta = b        A[t,s] = beta_t exp(c_t-c_s) (k_t.k_s)   (s < t)
//     b_t = beta_t ( v_t - exp(c_t) k_t^T W_in )
//     o_t = exp(c_t) q_t^T W_in + sum_{s<=t} P[t,s] delta_s     P[t,s] = exp(c_t-c_s) (q_t.k_s)
//     W_out = exp(sigma) W_in + sum_s k_s exp(sigma-c_s) delta_s^T

#pragma once

#include <cstdint>

namespace strata::prefill {

/// Check if chunked GDN prompt recurrence is available (Volta cc 7.0 for CUDA, or Intel Xe2 for SYCL).
bool gdn_chunk_available(void* stream = nullptr);

/// The chunked recurrence over `T` tokens (T a multiple of 64):
void gdn_chunk_recurrence(float* state, const float* h, const float* gate, const float* beta, float* oc,
                          int64_t T, void* stream);

/// Chunked GDN step including recurrence and output norm + gating (RMSNorm * gamma * sigmoid(z)):
bool gdn_chunk_step(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream,
                    int64_t ld16 = 0);

}  // namespace strata::prefill
}  // namespace strata::prefill
