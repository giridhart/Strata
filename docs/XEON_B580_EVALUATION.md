# Xeon w7-3465X + Arc B580 Port Evaluation & Config Tuning Guide

This guide explains how to evaluate the ported optimizations on **Intel Xeon w7-3465X + Intel Arc B580** so an LLM (or user) can generate or update a Strata server JSON config (such as `strata-b580-*.json` used by `sycl/serve/server_intel.py`).

For initial B580 setup and JSON schema context, see [B580_SETUP_GUIDE.md](B580_SETUP_GUIDE.md).

---

## 1. Build Verification & Benchmark Targets

From the repository root after sourcing Intel oneAPI (`setvars.sh`):

```bash
bash ./tools/build_sycl.sh -r \
  strata \
  mtp_kv_resident_test \
  draft_kv_plan_test \
  cpu_dual_row_parity \
  sycl_spec_tree \
  sycl_iq_fused_bench \
  sycl_p2_kernels_bench
```

Run correctness parity first (especially to exercise AVX-512 paths on Xeon w7-3465X and Xe2 kernels on B580):

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 bash ./tools/test_sycl.sh \
  -R "^(mtp_kv_resident_test|draft_kv_plan_test|cpu_dual_row_parity|sycl_spec_tree|sycl_iq_fused_bench|sycl_p2_kernels_bench)$" \
  --output-on-failure
```

Confirm `[3/3] AVX-512 IQ checks` in `cpu_dual_row_parity` runs and reports `OK` (not `SKIP`).

---

## 2. Microbenchmarks (Before vs. After per Subsystem)

Run these four commands on the target machine and capture stdout:

```bash
# A. CPU expert kernels (AVX2, AVX-512 dual-row, Q2_0 bitplane legacy vs rows_bp4_nt1)
STRATA_Q2_BITPLANE=1 ./build_sycl/cpu_dual_row_parity --bench --mb 256 --reps 5

# B. GPU IQ Gate/Up (Mode 0 vs Mode 8) and Q2_0 Down (kDtW=11 vs old / kDtW=10)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build_sycl/sycl_iq_fused_bench --bench-only
ONEAPI_DEVICE_SELECTOR=level_zero:0 STRATA_IQ_DT_W=10 ./build_sycl/sycl_iq_fused_bench --bench-only

# C. GPU MTP priority queue contention, QSA Top-K prefill (reg vs wide), BF16 rows vs multi
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build_sycl/sycl_p2_kernels_bench --bench-only

# D. Causal speculative tree kernel costs & multi-branch acceptance simulation
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build_sycl/sycl_spec_tree --bench
```

---

## 3. End-to-End Model A/B Matrix (`strata`)

Compare before/after end-to-end `tok/s` on your exact GGUF pack (`IQ3_S`, `IQ2_XS`, `IQ1_M`, etc.) with identical prompt, `--max-tokens`, and `--kv` / `--kv-resident` flags:

```bash
# Baseline (disable new defaults + keep opt-in features off)
ONEAPI_DEVICE_SELECTOR=level_zero:0 \
STRATA_MTP_SYCL_PRIO=0 \
STRATA_QSA_TOPK_PREFILL_WIDE=0 \
STRATA_IQ_DT_W=10 \
STRATA_Q2_BITPLANE=0 \
STRATA_EXP_MODE=0 \
./build_sycl/strata <model_args> --spec-tree 0

# Tuned A: Mode 0 GPU + CPU Q2 bitplane + SpecTree 3 (recommended start for IQ3_S / IQ2_XXS)
ONEAPI_DEVICE_SELECTOR=level_zero:0 \
STRATA_Q2_BITPLANE=1 \
STRATA_EXP_MODE=0 \
./build_sycl/strata <model_args> --spec-tree 3

# Tuned B: Mode 8 fused GPU + CPU Q2 bitplane + SpecTree 3 (recommended A/B for IQ1_M / IQ2_XS)
ONEAPI_DEVICE_SELECTOR=level_zero:0 \
STRATA_Q2_BITPLANE=1 \
STRATA_EXP_MODE=8 \
./build_sycl/strata <model_args> --spec-tree 3
```

Also compare `--spec-tree 2` vs `--spec-tree 3` on a coding or multi-turn prompt if GPU verify time barely changes between `T=4` and `T=6..8`.

---

## 4. Decision Rules for Updating the Server Config JSON

Use the benchmark outputs above to set `"args"` and `"env"` in your `strata-b580-*.json` config:

| Config Location | Key | Default | Evaluation Rule from Benchmarks |
| :--- | :--- | :--- | :--- |
| `"args"` | `--spec-tree` | `0` (off) | Set `"3"` (tile-4 disciplined) when `--mtp` and `--spec >= 2` are enabled and `--spec-tree 3` beats `0`. Upgrade to `"2"` only if wider `T` verify improves net `tok/s`. |
| `"args"` | `--kv-resident` | model-dep | Safe to use without `--mtp-window` (ring bug fixed: `-132` to `-264 MiB` saved vs old bug). Keep `--mtp-window` omitted unless intentionally capping drafter KV. |
| `"env"` | `STRATA_Q2_BITPLANE` | `"0"` | Set **`"1"`** whenever any MoE experts spill to CPU/RAM (`Q2_0` down-proj) and `cpu_dual_row_parity --bench` shows `q2_0 bitplane` faster than `legacy`. |
| `"env"` | `STRATA_EXP_MODE` | `"0"` | Check the target model's Gate/Up format row (`IQ3_S`, `IQ2_XS`, `IQ1_M`, etc.) in `sycl_iq_fused_bench --bench-only` across `T=1..4` and end-to-end `strata`. Set **`"8"`** if `Mode8(us) < Base(us)` (common for `IQ1_M` / `IQ2_XS`); otherwise keep `"0"` (often faster for `IQ3_S`). |
| `"env"` | `STRATA_GU_LNT` | auto | Leave unset (`1/2/4` dynamic) unless fixed `"1"`, `"2"`, or `"4"` wins when `STRATA_EXP_MODE="8"`. |
| `"env"` | `STRATA_IQ_DT_W` | `"11"` | Omit (defaults to `11`-dword conflict-free SLM). Set `"10"` only if `STRATA_IQ_DT_W=10` measures faster on `Q2_0` down-projection in `sycl_iq_fused_bench`. |
| `"env"` | `STRATA_MTP_SYCL_PRIO` | `"1"` | Omit (defaults on with safe fallback). Set `"0"` only if `sycl_p2_kernels_bench` shows `got_high=1` regressing latency or driver instability. |
| `"env"` | `STRATA_QSA_TOPK_WIDE_MIN` | `"7168"` | Omit (defaults on). Check `sycl_p2_kernels_bench --bench-only`: if `wide` beats `reg` at smaller `(nq * nb)`, lower threshold or set `STRATA_QSA_TOPK_PREFILL_WIDE="1"`; if `reg` wins across all prefill sizes on B580, set `"0"` on `STRATA_QSA_TOPK_PREFILL_WIDE`. |
| `"env"` | `STRATA_MMVF_ROWS` | auto | Leave unset unless `sycl_p2_kernels_bench` shows `rows` beating `multi` on your dominant BF16 token counts. |

---

## 5. Template Patch Block for `strata-b580-*.json`

An LLM updating an existing `strata-b580-*.json` should only need to update `"args"` (`--spec-tree`) and `"env"` with the keys below (preserving existing paths, `--pack`, `--native`, `--vram-reserve-mib`, etc.):

```json
{
  "args": [
    "...existing model/cache/kv flags...",
    "--spec", "4",
    "--spec-tree", "3"
  ],
  "env": {
    "ONEAPI_DEVICE_SELECTOR": "level_zero:0",
    "ZES_ENABLE_SYSMAN": "1",
    "STRATA_Q2_BITPLANE": "1",
    "STRATA_EXP_MODE": "0",
    "STRATA_MTP_SYCL_PRIO": "1",
    "STRATA_QSA_TOPK_WIDE_MIN": "7168"
  }
}
```

- For **`IQ3_S`** models, start with `"STRATA_EXP_MODE": "0"` and verify `"0"` vs `"8"` with `sycl_iq_fused_bench --bench-only`.
- For **`IQ1_M` / `IQ2_XS`** models, test `"STRATA_EXP_MODE": "8"` first.
- Keep `"--spec-tree": "3"` when paired with `"--mtp"` unless end-to-end testing shows `"2"` is faster.
