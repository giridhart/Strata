// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/core/draft_head.hpp - the MTP drafts' head at 2 bits, for Qwen3.8-Flash-Next's drafter (mtp.cpp) and
// the other models': the coarse products of every row of the draft head (its token subset, or the whole
// vocabulary), the k best of them taken again with the head's own rows and written back.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

class DraftHeadInt2 {
public:
    /// The head's own product of `n` rows (`rows`, in the head's form, gathered in candidate order) into y[0..n).
    using Exact = std::function<void(const uint8_t* rows, int n, float* y, void* stream)>;

    DraftHeadInt2() = default;
    ~DraftHeadInt2();
    DraftHeadInt2(const DraftHeadInt2&) = delete;
    DraftHeadInt2& operator=(const DraftHeadInt2&) = delete;

    /// The int2 copy of `n_rows` rows of `n_in` values of GGUF type `type`, row i's bytes at `row(i)` (host memory),
    /// and the buffers of the k candidates (`exact_row_bytes` a row of the head's own form).  False (and `err`) when
    /// the type has no host dequantizer or the memory is not there; the caller then keeps its head.
    bool build(int type, int64_t n_rows, int n_in, const std::function<const uint8_t*(int64_t)>& row,
               size_t exact_row_bytes, std::string& err);
    bool built() const { return codes_ != nullptr; }
    /// The device bytes it holds.
    uint64_t bytes() const { return bytes_; }

    /// The logits of the draft head's rows for the float row `x` (device, n_in values): coarse, but the k best, which
    /// `exact` computes from `head_rows` (the head's rows in the same order as the int2 ones).  `pos` (device, the
    /// draft row's position; null: never) at kPadFrom or more leaves the k alone, the rest at a probability of 0.
    void logits(const float* x, const uint8_t* head_rows, const Exact& exact, float* out, void* stream,
                const int32_t* pos = nullptr) const;
    static constexpr int32_t kPadFrom = 49152;

    /// Whether this device drafts faster with it than with `full` (the head's own product of every row into `out`):
    /// both timed here, after a warm-up.  STRATA_DRAFT_INT2=0|1 says it instead.  Logs the times.
    bool choose(const float* x, const uint8_t* head_rows, const Exact& exact, const std::function<void(float*, void*)>& full,
                float* out, void* stream, const char* who);

private:
    uint8_t* codes_ = nullptr;
    int32_t* cand_ = nullptr;
    uint8_t* cand_rows_ = nullptr;
    float* cand_y_ = nullptr;
    void* scratch_ = nullptr;
    uint64_t bytes_ = 0;
    int64_t n_rows_ = 0;
    int n_in_ = 0, k_ = 0;
    size_t exact_row_bytes_ = 0;
};

}  // namespace strata::core
