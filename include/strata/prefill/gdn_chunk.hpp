// include/strata/prefill/gdn_chunk.hpp - chunked GDN prompt recurrence (FlashQLA gdn-chunk) for Intel Xe2 / SYCL 2020.
#pragma once

#include <cstdint>

namespace strata::prefill {

/// Check if chunked GDN prompt recurrence is available on the specified stream/queue.
/// Evaluates hardware support (Intel GPU with 16-lane sub-groups) and respects STRATA_GDN_CHUNKED.
bool gdn_chunk_available(void* stream = nullptr);

/// Pure recurrence over T tokens writing un-normed output oc (o * rsqrtf(S)):
///   state: [S, HV, S] in row-major layout (rs = HV * S), updated in-place.
///   h:     [T, C] post-conv activations with Q, K, V heads (C = 10240).
///   gate:  [T, HV] log-decay gate pre-exp.
///   beta:  [T, HV] update gate post-sigmoid.
///   oc:    [T, HV, S] un-normed output.
void gdn_chunk_recurrence(float* state, const float* h, const float* gate, const float* beta, float* oc,
                          int64_t T, void* stream);

/// Chunked GDN step including recurrence and output norm + gating (RMSNorm * gamma * sigmoid(z)):
///   Produces FP32 y and FP16 y16 with row stride ld16 (default HV * S).
///   Returns true if chunked recurrence was dispatched (T >= 128 and STRATA_GDN_CHUNKED=1).
///   Returns false if criteria not met, signaling fallback to token-by-token recurrence.
bool gdn_chunk_step(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream,
                    int64_t ld16 = 0);

}  // namespace strata::prefill
