# SYCL Decode Optimizations — A/B Testing & Config Tuning Plan (Arc B580)

This document defines the target-platform verification, A/B benchmark plan, and parameter sweep on **Intel Arc B580 (12 GB)** for the native SYCL decode fixes and existing opt-in decode knobs.

## What Was Fixed

1. **`record_commit()` GDN fast-path bug & batched commit (`src/core/sycl/verify.cpp`)**
   - Fixed missing `t_out_begin = (int)MT` argument to `gdn_step_norm_multi` (previously defaulted to `0`, forcing the full heavy RMSNorm + output write across all `MT` verify/tree tokens on every GDN layer instead of the lightweight commit-only kernel).
   - Wired opt-in **`STRATA_COMMIT_BATCH=1`** to call the already-implemented `gdn_commit_layers()` (2 launches for all GDN layers) and `native_qsa_indexer_append_steps_layers()` (1 launch for all QSA layers), collapsing ~200+ direct ungraphed SYCL commit submissions per decode round down to ~4 launches. Bitwise identical.
2. **Shared expert & MoE input deduplication (`src/core/sycl/verify.cpp`, `src/kernels/sycl/shared_expert.cpp`)**
   - Skipped dead `f32_to_bf16_bulk` calls on native packs when `shared_expert_native_bf16_enabled()` is true (`sh_bf16_` is unread).
   - Wired opt-in **`STRATA_VERIFY_QDEDUP=1`** (`x_q8_1_ready`) so `xm` is quantized to Q8_1 once per layer instead of twice (eliminates ~60 redundant quantize launches/window).
   - Wired `native_mmvq_pair` in `shared_expert_multi` for paired Q8_0 gate+up projections (activated when **`STRATA_XE2_KQ=1`**).
3. **Epilogue, PCIe grid, and local barrier (`src/core/sycl/verify.cpp`, `src/kernels/sycl/native_mmvq.cpp`)**
   - Wired `gr_write_multi()` for the final layer residual write (1 launch instead of `T` per-token launches).
   - Passed `kPcieGroupRows` (`4`) as `grid_groups` to the PCIe `native_expert_grouped` launch instead of full `T * K` group rows.
   - Changed `native_mmvq_multi_kernel`'s `item_ct1.barrier()` (full global+local fence) to `sycl::access::fence_space::local_space`.

---

## 1. Build & Baseline Setup on Target B580

```bash
bash ./tools/build_sycl.sh -r strata
```

Export your current best baseline config (with `UR_L0_DEVICE_SCOPE_EVENTS=2` and `SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2` spelled cleanly):

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export SYCL_CACHE_PERSISTENT=1
export NEO_CACHE_PERSISTENT=1
export UR_L0_DEVICE_SCOPE_EVENTS=2
export SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2

export STRATA_COPY_ENGINE=1
export STRATA_PARTIAL_PIN_GIB=48
export STRATA_PREFILL_TIMING=1
export STRATA_DECODE_TIMING=1
export STRATA_ATTN_MERGE_ACTIVE=1
export STRATA_MMVQ_DECODE_256GRF=0
export STRATA_B6_MMVQ_ROWS=1
export STRATA_VERIFY_GRAPH=split
export STRATA_VERIFY_GRAPH_CHUNK=4
export STRATA_Q2_BITPLANE=1
export STRATA_EXP_MODE=0
export STRATA_IQ_DT_W=11
export STRATA_MTP_SYCL_PRIO=1
export STRATA_QSA_TOPK_WIDE_MIN=7168
export STRATA_PF_DMA_PAGEABLE=1
export STRATA_MMVF_ROWS=0
export STRATA_PCIE_BALANCE=1
export STRATA_PREFILL_LAYER_MAJOR=1
export STRATA_PREFILL_LM_TOKENS=16384
export STRATA_ROUTE_RESIDENT_RANKS=3-9
export STRATA_ROUTE_BUDGET=0.10
export STRATA_ROUTE_BUDGET_TAIL=0.25
export STRATA_LAYER_LIVE=1
export STRATA_LAYER_SHARE=0.50
export STRATA_TAIL_WEIGHT_FROM=3
export STRATA_PREFILL_CPU_SHARE=auto
export STRATA_PREFILL_CPU_SHARE_MAX=4096

export STRATA_BIN=./build_sycl/strata
export MODEL=/path/to/Qwen3.8-Flash-Next-shard-00001.gguf
export PROMPT=/path/to/prompt_4k.tokens
export BASE_ARGS="--native $MODEL --tokens-file $PROMPT --expert-cache 1536 --resident-experts --max-context 262144 --kv int8 --kv-resident 20480 --prefill 4096 --spec 4 --spec-min-p 0.5 --spec-tree 3 --pool-workers 26 --ple-io ram --pcie-frac 0.2 --mtp-window 8192 --vision --max-tokens 256 --temperature 0"
```

---

## 2. Phase 1 — Correctness & Bit-Identity Check

Before benchmarking, verify that greedy output tokens (`--temperature 0 --max-tokens 128`) match between baseline and the new opt-in flags:

| Arm | Extra Env Vars | Expected |
| :--- | :--- | :--- |
| **B0 (Baseline)** | *(none)* | Reference tokens |
| **B1 (Batched commit)** | `STRATA_COMMIT_BATCH=1` | Identical tokens to B0 |
| **B2 (Q8_1 dedup)** | `STRATA_VERIFY_QDEDUP=1` | Identical tokens to B0 |
| **B3 (Xe2 KQ + pair)** | `STRATA_XE2_KQ=1` | Identical (or within 1-ULP) tokens to B0 |
| **B4 (All new fixes)** | `STRATA_COMMIT_BATCH=1 STRATA_VERIFY_QDEDUP=1 STRATA_XE2_KQ=1` | Matches B0/B3 |

---

## 3. Phase 2 — A/B Benchmark of the New Decode Fixes

Run **3 warm iterations** per arm (discard first JIT/cache-warm run) and record median **decode tok/s** and **accepted tokens/round**:

```bash
for ARM in "B0_base" "B1_commit" "B2_qdedup" "B3_xe2kq" "B4_all"; do
  echo "=== $ARM ==="
  for RUN in 1 2 3; do
    case "$ARM" in
      B0_base)   $STRATA_BIN $BASE_ARGS ;;
      B1_commit) STRATA_COMMIT_BATCH=1 $STRATA_BIN $BASE_ARGS ;;
      B2_qdedup) STRATA_VERIFY_QDEDUP=1 $STRATA_BIN $BASE_ARGS ;;
      B3_xe2kq)  STRATA_XE2_KQ=1 $STRATA_BIN $BASE_ARGS ;;
      B4_all)    STRATA_COMMIT_BATCH=1 STRATA_VERIFY_QDEDUP=1 STRATA_XE2_KQ=1 $STRATA_BIN $BASE_ARGS ;;
    esac 2>&1 | grep -E "decode|tok/s|accept|verify"
  done
done
```

Keep whichever subset of `{STRATA_COMMIT_BATCH=1, STRATA_VERIFY_QDEDUP=1, STRATA_XE2_KQ=1}` wins as your new `WIN_ENV` for Phase 3.

---

## 4. Phase 3 — Tuning Sweep to Find Best Config Values

Test one group at a time on top of `WIN_ENV` (3 warm runs each, keep winners before moving to the next group):

### Group A: SYCL Graph & Host-Wait Overhead
| Knob | Values to Test | What It Tests |
| :--- | :--- | :--- |
| `STRATA_SYCL_GRAPH_OPT` | `0` vs `1` | `no_cycle_check` + `enable_native_recording` + multi-queue `end_recording()` |
| `STRATA_VERIFY_GRAPH_CHUNK` | `4` (current), `8`, `12`, `16` | Fewer split graph submissions per window vs Level Zero graph size |
| `STRATA_FUSE_WAIT` | `0` vs `1` | Fuses `wait_flag_ge(m_flagA_)` + `copy_i32_from_mapped` (active since `DEVICE_PLAN=0`) |

### Group B: Expert Decode Kernels
| Knob | Values to Test | What It Tests |
| :--- | :--- | :--- |
| `STRATA_EXP_MODE` | `0` (current), `7` (SLM), `8` (fused) | Expert gate/up kernel mode on Xe2 |
| `STRATA_XE2_IQ` | `0` (current) vs `1` | Xe2 IQ2_XXS / IQ2_S + Q2_0 transposed fast path (`native_down_kernel_dtx`) |
| `STRATA_EXPDEC` | `0` (current) vs `1` | Alternative decode-once expert kernel with 256 GRF on Battlemage |

### Group C: Speculative Tree & Acceptance Rate
| Knob | Values to Test | What It Tests |
| :--- | :--- | :--- |
| `--spec` / `--spec-tree` | `4 / 3` (current), `4 / 2`, `5 / 2`, `4 / 0` (linear) | Tree width vs verify latency per round |
| `--spec-min-p` | `0.4`, `0.5` (current), `0.6` | Draft pruning threshold vs wasted tree tokens |

### Group D: PCIe / CPU Balance & Threads
| Knob | Values to Test | What It Tests |
| :--- | :--- | :--- |
| `--pcie-frac` | `0.15`, `0.20` (current), `0.25`, `0.30` | PCIe vs CPU split seed (with `STRATA_PCIE_BALANCE=1`) |
| `STRATA_PCIE_EARLY` | `0` (current), `1`, `2` | Overlaps `pcie_mode=2` staging on side queue `pf_cs_` (`2` keeps it inside graph capture) |
| `--pool-workers` | `20`, `24`, `26` (current) | Avoids oversubscribing CPU cores against GPU submission + stager threads |

### Group E: Dense GEMV Kernels
| Knob | Values to Test | What It Tests |
| :--- | :--- | :--- |
| `STRATA_B6_MMVQ_ROWS` | `1` (current) vs `0` | Multi-row Q4_K/Q5_K/Q6_K MMVQ |
| `STRATA_MMVF_ROWS` | `0` (current) vs `1` | Multi-row BF16 GEMV (4 rows/block) |

---

## 5. Recording Results

Fill in median **decode tok/s** (256 tokens, `--temperature 0`) and **mean accepted tokens/round** from your runs:

| Step / Config Change | Decode tok/s | Delta vs Baseline | Accepted / Round | Keep? |
| :--- | :---: | :---: | :---: | :---: |
| Baseline (`B0`) | | — | | ref |
| `+ STRATA_COMMIT_BATCH=1` (`B1`) | | | | |
| `+ STRATA_VERIFY_QDEDUP=1` (`B2`) | | | | |
| `+ STRATA_XE2_KQ=1` (`B3`) | | | | |
| Combined (`B4`) | | | | |
| Best Group A (Graph / Wait) | | | | |
| Best Group B (Expert Kernels) | | | | |
| Best Group C (Spec / Tree) | | | | |
| Best Group D (PCIe / Pool) | | | | |
| Best Group E (Dense GEMV) | | | | |
