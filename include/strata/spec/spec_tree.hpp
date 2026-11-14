// include/strata/spec/spec_tree.hpp - opt-in multi-branch speculative tree construction and acceptance walk
// (STRATA_SPEC_TREE / --spec-tree).
//
// Trunk-first topological layout:
//   * Slots 0 .. T_base - 1 hold the primary trunk proposal (parent[0] = -1, parent[t] = t - 1 for 1 <= t < T_base).
//   * Slots T_base .. T - 1 hold secondary branches from Suffix/MTP divergence and device-side MTP runner-up picks
//     (0 <= parent[t] < t).
//
// Because the trunk occupies 0 .. T_base - 1, any window where acceptance stays on the trunk satisfies path[i] == i
// for all 0 <= i < k (needs_compact == false), requiring zero post-verify compaction before Verifier::commit(k).
#pragma once

#include "strata/kernels/verify_kernels.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace strata::spec {

inline int spec_tree_env() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_SPEC_TREE");
        return e != nullptr ? std::max(0, std::atoi(e)) : 0;
    }();
    return v;
}

inline float spec_tree_min_p_env() {
    static const float v = [] {
        const char* e = std::getenv("STRATA_SPEC_TREE_MIN_P");
        return (e != nullptr && *e != '\0') ? (float) std::atof(e) : -1.0f;
    }();
    return v;
}

/// True when a tree node at logical depth `cand_depth` from `pos0` needs no pooled QSA indexer block beyond what
/// root token 0 at `pos0` completes (either dense regime `lpos + 1 <= 2048` or same partial tail block `(lpos + 1) / 4`).
inline bool indexer_tree_safe(int64_t pos0, int cand_depth) {
    const int64_t lpos = pos0 + cand_depth;
    return (lpos + 1 <= 2048) || (((lpos + 1) / 4) == ((pos0 + 1) / 4));
}

struct SpecTreeInput {
    int mode = 0;                     // 1: compact (<=6), 2: wide (<=8), 3: tile-4 disciplined (<=max(T_base,4))
    int64_t pos0 = 0;
    int T_base = 1;                   // current linear trunk length in `window[0 .. T_base - 1]`
    int max_room = 1;                 // min(spec_cap, max_context - pos0)
    bool from_sfx = false;            // trunk came from suffix lookup
    bool draft_on = false;            // `drafts` / `dprob` hold valid MTP chain outputs
    int mtp_n = 0;                    // number of valid MTP drafts in `drafts`
    const int32_t* drafts = nullptr;  // MTP greedy chain [0 .. mtp_n - 1]
    const float* dprob = nullptr;     // MTP greedy probabilities [0 .. mtp_n - 1]
    int32_t mtp_top2_0 = -1;          // MTP runner-up at step 0
    float mtp_top2_p0 = 0.0f;
    int32_t mtp_top2_1 = -1;          // MTP runner-up at step 1
    float mtp_top2_p1 = 0.0f;
    int sfx_k = 0;                    // number of suffix lookup tokens in `sbuf`
    int sfx_match = 0;                // suffix context match length
    const int32_t* sbuf = nullptr;    // suffix proposal [0 .. sfx_k - 1]
};

/// Populates `window[T_base .. T_out - 1]` and `map` when at least one valid secondary branch is added.
/// Returns false (leaving `window[0 .. T_base - 1]` untouched) when no branch applies or `pos0` cannot host the tree's
/// depth under the single-append QSA indexer invariant.
inline bool build_spec_tree(const SpecTreeInput& in, int32_t* window, int& T_out,
                            strata::kernels::VerifyTreeMap& map) {
    using strata::kernels::kVerifyNarrowT;
    using strata::kernels::kVerifyMaxT;
    T_out = in.T_base;
    if (in.mode <= 0 || in.T_base < 1 || in.T_base >= (int) kVerifyNarrowT || in.max_room < 2) return false;

    int T_trunk = std::min(in.T_base, std::min(in.max_room, (int) kVerifyNarrowT));
    // Check that every trunk node satisfies the QSA indexer block invariant.
    for (int d = 0; d < T_trunk; ++d) {
        if (!indexer_tree_safe(in.pos0, d)) return false;
    }

    int cap = T_trunk;
    if (in.mode == 1) {
        cap = std::min(std::max(T_trunk + 2, 4), 6);
    } else if (in.mode == 2) {
        cap = std::min(std::max(T_trunk + 4, 6), (int) kVerifyNarrowT);
    } else {
        // Mode 3: strictly stay within the 4-column MMVQ/MoE tile when T_trunk <= 4.
        // If T_trunk == 4 and a strong root branch exists while the 3rd draft tail is weak, trade slot 3 for the branch.
        cap = std::max(T_trunk, 4);
    }
    cap = std::min(cap, std::min(in.max_room, (int) kVerifyNarrowT));

    const float env_min_p = spec_tree_min_p_env();
    const float min_p2_tile4 = env_min_p >= 0.0f ? env_min_p : 0.06f;
    const float min_p2_wide  = env_min_p >= 0.0f ? env_min_p : (in.mode == 1 ? 0.12f : 0.08f);

    // Mode 3 tail replacement: when T_trunk == 4 (not suffix), if tail value p0*p1*p2 is weaker than a strong root branch,
    // shorten trunk to 3 so slot 3 holds the root branch inside the same 4-token tile.
    if (in.mode == 3 && T_trunk == 4 && cap == 4 && !in.from_sfx && in.draft_on && in.dprob != nullptr) {
        const float tail_ev = in.dprob[0] * in.dprob[1] * in.dprob[2];
        const bool strong_sfx = in.sfx_k > 0 && in.sfx_match >= 3 && in.sbuf != nullptr && in.sbuf[0] != window[1];
        const bool strong_t2  = in.mtp_top2_0 >= 0 && in.mtp_top2_0 != window[1] &&
                                in.mtp_top2_p0 >= std::max(min_p2_tile4, tail_ev * 0.85f);
        if ((strong_sfx && tail_ev < 0.45f) || strong_t2) {
            T_trunk = 3;
        }
    }

    if (cap <= T_trunk) return false;

    int32_t par[kVerifyMaxT];
    int32_t dep[kVerifyMaxT];
    par[0] = -1;
    dep[0] = 0;
    for (int t = 1; t < T_trunk; ++t) {
        par[t] = t - 1;
        dep[t] = t;
    }
    int cur_T = T_trunk;

    auto has_child_tok = [&](int p, int32_t tok) {
        if (tok < 0) return true;
        for (int i = 1; i < cur_T; ++i) {
            if (par[i] == p && window[i] == tok) return true;
        }
        return false;
    };

    auto add_node = [&](int p, int32_t tok, bool allow_past_4) -> int {
        if (p < 0 || p >= cur_T || tok < 0 || cur_T >= cap) return -1;
        if (!allow_past_4 && cur_T >= 4) return -1;
        const int cand_d = dep[p] + 1;
        if (cand_d >= kVerifyMaxT || !indexer_tree_safe(in.pos0, cand_d)) return -1;
        if (has_child_tok(p, tok)) return -1;
        const int idx = cur_T++;
        window[idx] = tok;
        par[idx] = p;
        dep[idx] = cand_d;
        return idx;
    };

    // 1. Cross-source divergence branch (Suffix vs MTP): highest multi-token payoff on code / agent edits / JSON.
    if (!in.from_sfx && in.sfx_k > 0 && in.sbuf != nullptr && in.sfx_match >= 3) {
        int div = -1;
        const int lim = std::min(T_trunk - 1, in.sfx_k);
        for (int d = 0; d < lim; ++d) {
            if (in.sbuf[d] != window[d + 1]) { div = d; break; }
        }
        if (div >= 0) {
            const bool expand_ok = (in.mode == 2) || (in.mode == 1 && in.sfx_match >= 3);
            int b = add_node(div, in.sbuf[div], expand_ok);
            const int max_ext = in.mode == 2 ? 2 : 1;
            for (int e = 1; b >= 0 && e <= max_ext && div + e < in.sfx_k; ++e) {
                b = add_node(b, in.sbuf[div + e], expand_ok && in.sfx_match >= 4);
            }
        }
    } else if (in.from_sfx && in.draft_on && in.drafts != nullptr && in.mtp_n > 0) {
        int div = -1;
        const int lim = std::min(T_trunk - 1, in.mtp_n);
        for (int d = 0; d < lim; ++d) {
            if (in.drafts[d] != window[d + 1]) { div = d; break; }
        }
        if (div >= 0) {
            const float p_div = in.dprob != nullptr ? in.dprob[div] : 1.0f;
            if (p_div >= min_p2_tile4) {
                int b = add_node(div, in.drafts[div], p_div >= min_p2_wide);
                if (b >= 0 && in.mode == 2 && div + 1 < in.mtp_n &&
                    (in.dprob == nullptr || in.dprob[div + 1] >= 0.20f)) {
                    (void) add_node(b, in.drafts[div + 1], true);
                }
            }
        }
    }

    // 2. Device-side MTP Top-2 runner-up at root (parent 0).
    if (in.draft_on && in.mtp_top2_0 >= 0 && in.mtp_top2_p0 >= min_p2_tile4) {
        (void) add_node(0, in.mtp_top2_0, in.mtp_top2_p0 >= min_p2_wide);
    }

    // 3. Device-side MTP Top-2 runner-up at step 1 (parent 1 on MTP trunk).
    if (!in.from_sfx && in.draft_on && T_trunk >= 2 && in.mtp_top2_1 >= 0 && in.mtp_top2_p1 >= min_p2_tile4) {
        (void) add_node(1, in.mtp_top2_1, in.mode == 2 && in.mtp_top2_p1 >= min_p2_wide);
    }

    // Only engage the tree graph if at least one non-linear branch was formed.
    bool has_branch = false;
    for (int t = 1; t < cur_T; ++t) {
        if (par[t] != t - 1) { has_branch = true; break; }
    }
    if (!has_branch) return false;

    map = {};
    map.pos0 = in.pos0;
    map.n_tok = cur_T;
    map.branch_mask = 0;
    for (int t = 0; t < cur_T; ++t) {
        map.parent[t] = par[t];
        map.depth[t] = dep[t];
        if (t > 0 && par[t] != t - 1) {
            map.branch_mask |= (1u << (unsigned) par[t]);
        }
        int cur = t;
        for (int k = dep[t]; k >= 0; --k) {
            map.anc_slot[t][k] = cur;
            cur = par[cur];
        }
        for (int j = 0; j < 4; ++j) {
            const int rel = dep[t] - (3 - j);
            map.anc_conv[t][j] = rel < 0 ? rel : map.anc_slot[t][rel];
        }
    }
    T_out = cur_T;
    return true;
}

/// Walks the verified tree greedily from root 0: at node `cur`, if some child `c` (`tree.parent[c] == cur`) proposed
/// `window[c] == outv[cur]`, advances to `c`. Writes the accepted physical slot sequence to `path[0 .. k - 1]` and
/// returns `k` (`1 <= k <= tree.n_tok`). Sets `needs_compact` if any accepted slot `path[i] != i`, and `branch_hit`
/// if any accepted step took a non-trunk edge (`path[i] >= T_trunk` or `tree.parent[path[i]] != path[i] - 1`).
inline int walk_accepted_tree(const strata::kernels::VerifyTreeMap& tree, const int32_t* window, const int32_t* outv,
                              int* path, bool& needs_compact, bool& branch_hit) {
    needs_compact = false;
    branch_hit = false;
    int k = 0;
    int cur = 0;
    path[k++] = 0;
    while (true) {
        const int32_t want = outv[cur];
        int next_c = -1;
        for (int c = cur + 1; c < tree.n_tok; ++c) {
            if (tree.parent[c] == cur && window[c] == want) {
                next_c = c;
                break;
            }
        }
        if (next_c < 0) break;
        if (next_c != k) needs_compact = true;
        if (tree.parent[next_c] != next_c - 1) branch_hit = true;
        path[k++] = next_c;
        cur = next_c;
    }
    return k;
}

/// Ancestor-exact penalty history rows for a tree verify window (`tree.n_tok` rows of `h` slots).
/// For each node `t`, its history is the last `h` tokens of `tail[0..n_tail)` followed by its ancestor path
/// `window[tree.anc_slot[t][0 .. tree.depth[t]]]` — bitwise identical to what linear decode sees at that step.
inline void penalty_rows_tree(const int32_t* tail, int64_t n_tail, const int32_t* window,
                              const strata::kernels::VerifyTreeMap& tree, int h, int32_t* out) {
    for (int t = 0; t < tree.n_tok; ++t) {
        int32_t* row = out + (size_t) t * (size_t) h;
        const int d = tree.depth[t];
        const int64_t avail = n_tail + d + 1;
        const int take = (int) std::min<int64_t>(h, avail);
        std::fill(row, row + (h - take), -1);
        for (int j = 0; j < take; ++j) {
            const int64_t i = avail - take + j;
            row[h - take + j] = i < n_tail ? tail[i] : window[tree.anc_slot[t][i - n_tail]];
        }
    }
}

}  // namespace strata::spec
