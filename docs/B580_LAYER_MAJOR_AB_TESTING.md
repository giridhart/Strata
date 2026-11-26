# Layer-Major Prefill Race Fix — A/B Testing Plan (Arc B580, Native SYCL)

This document defines the target-platform verification and A/B test plan on **Intel Arc B580 (12 GB)** for the native SYCL layer-major prefill race fixes in `src/prefill/sycl/prefill.cpp`.

## What Was Fixed

1. **`Stager::work()` cross-layer & skipped-job DMA race (`src/prefill/sycl/prefill.cpp`)**
   - Moved `dma_done[b]->wait_and_throw()` outside `if (j >= kRing)` so layer `l`'s first `kRing` prefetched experts wait for layer `l - 1`'s in-flight `m.copy` DMAs before overwriting `buf[b]`.
   - Added `while (!ready[j - kRing].load(acquire))` inside `if (j >= kRing)` so `pass()` cannot let job `j` overwrite `buf[b]` while job `j - kRing`'s worker thread is still copying into it.
   - Allocated `Stager::buf[i]` on `*m.cs` (same SYCL context as `m.copy`) instead of `default_queue()`.
2. **Pinned host upload ring (`SyclPinnedRing`) for `m.steps_host` and `m.tok_host`**
   - Replaced async `m.cs->memcpy` from reused unpinned `std::vector` buffers with a 4-slot pinned host ring (`m.up_ring.h2d`) guarded by per-slot `sycl::event` waits.
   - Prevents the host loop over sub-chunks `c = 0..nc-1` from overwriting `m.steps_host` (QSA layers, starting at layer 3) or `m.tok_host` (layer 0) while `m.cs` is still executing the previous layer's `lm_experts`.

---

## 1. Build on Target

```bash
bash ./tools/build_sycl.sh -r strata
```

Set shared environment variables for all runs below:

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export STRATA_BIN=./build_sycl/strata
export MODEL=/path/to/Qwen3.8-Flash-Next-shard-00001.gguf
export PROMPT_32K=/path/to/prompt_32k.tokens
export COMMON_ARGS="--native $MODEL --tokens-file $PROMPT_32K --prefill 4096 --max-tokens 32 --temperature 0"
```

---

## 2. Correctness & Determinism A/B Matrix

Compare `GDN_HASH` (via `STRATA_STATE_HASH_GDN=1`), absence of NaNs, and generated output across all configurations.

| Case | Env Vars | Purpose | Pass Criteria |
| :--- | :--- | :--- | :--- |
| **A (Chunk-major ref)** | `STRATA_PREFILL_LAYER_MAJOR=0 STRATA_STATE_HASH_GDN=1` | Non-layer-major reference | Clean output, finite GDN state |
| **B (`nc=2` ref)** | `STRATA_PREFILL_LAYER_MAJOR=1 STRATA_PREFILL_LM_TOKENS=8192 STRATA_STATE_HASH_GDN=1` | Previously clean layer-major baseline | Clean output, matches Case A tokens |
| **C (`nc=3` repro, 10x)** | `STRATA_PREFILL_LAYER_MAJOR=1 STRATA_PREFILL_LM_TOKENS=12288 STRATA_STATE_HASH_GDN=1` | Primary failing case (`nc=3`) | 10/10 runs clean, identical `GDN_HASH` across runs, matches Case A/B |
| **D1 (`nc=4` wrap)** | `STRATA_PREFILL_LAYER_MAJOR=1 STRATA_PREFILL_LM_TOKENS=16384 STRATA_STATE_HASH_GDN=1` | Fills all 4 `SyclPinnedRing` slots per layer | Clean output, deterministic `GDN_HASH` |
| **D2 (`nc=6` wrap)** | `STRATA_PREFILL_LAYER_MAJOR=1 STRATA_PREFILL_LM_TOKENS=12288 STRATA_STATE_HASH_GDN=1` with `--prefill 2048` | Forces `SyclPinnedRing` wrap-around (`nc > 4`) | Clean output, deterministic `GDN_HASH` |

### Repeatability loop for Case C (10 runs)

```bash
for i in $(seq 1 10); do
  echo "=== Run $i ==="
  STRATA_PREFILL_LAYER_MAJOR=1 \
  STRATA_PREFILL_LM_TOKENS=12288 \
  STRATA_STATE_HASH_GDN=1 \
  $STRATA_BIN $COMMON_ARGS 2>&1 | tee "lm_nc3_run_${i}.log" \
    | grep -E "layer-major|GDN_HASH|NaN|nan|tok/s"
done
grep "GDN_HASH" lm_nc3_run_*.log | sort -u
```

Expected: `sort -u` prints a **single unique `GDN_HASH` line** across all 10 runs, with zero `NaN` tokens.

---

## 3. Subsystem Isolation & Stress Matrix

Check the layer-major summary line in the logs (`R in VRAM` vs `R in pinned RAM`), then test both residual placements and stager stress:

| Case | Env Vars | Purpose |
| :--- | :--- | :--- |
| **E1 (`R` in VRAM, `nc=3`)** | `STRATA_PREFILL_LAYER_MAJOR=1 STRATA_PREFILL_LM_TOKENS=12288 STRATA_PREFILL_LM_R=1 STRATA_STATE_HASH_GDN=1` | Exercises `lb.r_dev = true` at `nc=3` (if VRAM allows) |
| **E2 (`R` in pinned RAM, `nc=3`)** | `STRATA_PREFILL_LAYER_MAJOR=1 STRATA_PREFILL_LM_TOKENS=12288 STRATA_PREFILL_LM_R=2 STRATA_STATE_HASH_GDN=1` | Forces `lb.r_dev = false` (`m.copy` <-> `m.cs` ping-pong) at `nc=3` |
| **E3 (`R` in pinned RAM, `nc=2`)** | `STRATA_PREFILL_LAYER_MAJOR=1 STRATA_PREFILL_LM_TOKENS=8192 STRATA_PREFILL_LM_R=2 STRATA_STATE_HASH_GDN=1` | Forces `lb.r_dev = false` at `nc=2` |
| **F1 (Prefetch off)** | `STRATA_PREFILL_LAYER_MAJOR=1 STRATA_PREFILL_LM_TOKENS=12288 STRATA_LM_PREFETCH=0 STRATA_STATE_HASH_GDN=1` | Confirms `lm_prefetch=1` matches `lm_prefetch=0` bit-for-bit |
| **F2 (Tight stager ring)** | `STRATA_PREFILL_LAYER_MAJOR=1 STRATA_PREFILL_LM_TOKENS=12288 STRATA_STAGER_RING=4 STRATA_STATE_HASH_GDN=1` | Maximizes `Stager` wrap-around and cross-layer `dma_done` waits |

---

## 4. Performance A/B (No-Regression Check)

Run 3 warm iterations of each (without `STRATA_STATE_HASH_GDN`) and record median prompt tok/s:

1. **Before vs After patch on `nc=2` (`STRATA_PREFILL_LM_TOKENS=8192`):**
   - Check out `HEAD~1`, build, measure 3x prompt tok/s.
   - Check out patched commit, build, measure 3x prompt tok/s.
   - Expected: within run-to-run noise (~0% delta).
2. **`nc=3` (`12288`) vs `nc=2` (`8192`) on patched build:**
   - Measure 3x prompt tok/s at `12288` vs `8192` to verify the expected layer-major expert-amortization speedup at `nc=3`.

---

## 5. Triage If Any NaN Still Reproduces

- **If Case E1 (`LM_R=1`) is clean and Case E2/E3 (`LM_R=2`) fails:** the remaining race is in Level Zero's cross-queue `ext_oneapi_submit_barrier` between `m.copy` and `m.cs` during `!lb.r_dev` residual ping-pong (`src/prefill/sycl/prefill.cpp:4300-4330, 4645-4655`).
- **If Case F1 (`STRATA_LM_PREFETCH=0`) is clean while default (`STRATA_LM_PREFETCH=1`) fails:** re-inspect `Stager::skip` / `Stager::pass` / `Stager::issued_one` ordering on `m.copy`.
- **If both E1 and F1 still fail at `nc=3`:** capture `STRATA_PREFILL_TIMING=1` and check whether `m.ab` vs `m.mixed_bf` vs `m.R` is the first non-finite tensor at layer 4 `c=1`.
