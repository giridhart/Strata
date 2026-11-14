// include/strata/core/mtp_kv_ring.hpp - MTP draft layer K/V ring sizing helper.
#pragma once

#include <cstdint>

namespace strata::core {

/// Compute the `ring_cells` argument passed to `qsa_state_bytes` / `qsa_state_init` for the MTP draft layer.
///
/// When `--mtp-window` is active (`0 < window < max_cells`), the drafter's K/V is sized as a circular ring of
/// `window + 4 * max_t + 64` cells (`kv_mode = 2`) backed by a pinned host copy for checkpoint resume.
/// When no window is set (`window <= 0`) or the window already covers `max_cells`, returns `-1` so
/// `--kv-resident` keeps the dense drafter K/V fully resident (`kv_mode = 0`) instead of assigning
/// un-resolved Clock-LRU paging (`kv_mode = 1`).
inline constexpr int64_t mtp_kv_ring_cells(int64_t window, int64_t max_cells, int max_t) {
    return (window > 0 && window < max_cells) ? window + 4 * static_cast<int64_t>(max_t) + 64 : -1;
}

}  // namespace strata::core
