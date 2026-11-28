# B580 HotRod Runtime Tuning & Systematic A/B Testing Guide

This guide documents the opt-in performance levers ported from `Strata-HotRod` into the native SYCL backend (`src/core/sycl/`, `src/kernels/sycl/`, `src/program/sycl/`) for **Intel Arc B580 (12 GB)** + **Xeon** co-execution, and provides a step-by-step protocol to benchmark and find the fastest reliable configuration.

> **Default safety:** Every option below is **OFF by default**. With none of these environment variables set, the engine executes the exact same default path as before.

---

## 1. Reference: All Runtime Environment Variables

### Group A — Exact / Bit-Identical (Zero Output Change)
These options never alter which experts run or their arithmetic. Greedy (`--temperature 0`) outputs must remain identical to baseline.

| Env Var | Default | What it does | Notes for B580 |
| :--- | :---: | :--- | :--- |
| `STRATA_FUSE_WAIT=1` | `0` (off) | Fuses `wait_flag_ge(m_flagA_)` + `copy_i32_from_mapped` into a single `wait_copy_i32_from_mapped` launch in `record_layer_post`. | Saves 1 kernel launch per half-window per layer (~96 launches/window). Works with both `STRATA_VERIFY_GRAPH=split` and `direct`. |
| `STRATA_PCIE_MIN1=1` | `0` (off) | When `nmiss >= 2` and `(nmiss * pcie_num) >> 8` rounds to `0`, routes at least 1 miss to PCIe. | Important when `--pcie-frac 0.2` (`pcie_num = 51`), where `(nmiss * 51) >> 8 == 0` for all `nmiss <= 5`. |
| `STRATA_PCIE_BALANCE=1` | `0` (off) | Fits a running linear model of per-layer CPU pool time and GPU step time (`PcieModel`) and dynamically picks each layer's PCIe miss count `m` to minimize `max(GPU, CPU)`. | Adapts automatically across layers with different quant types (`IQ2_XXS` vs `IQ3_S`) and hit rates. |
| `STRATA_PCIE_EARLY=1` | `0` (off) | Runs `pcie_mode == 2` staging (`fetch_blobs` + `rebase_ptrs`) on a dedicated side SYCL queue (`pf_cs_`) concurrently with the layer's VRAM experts. | Activates automatically under `STRATA_VERIFY_GRAPH=direct`. Use `STRATA_PCIE_EARLY=force` to test under `STRATA_VERIFY_GRAPH=split` (watch for driver/graph capture issues). |

### Group B — Approximate Routing & Tail-Skipping (Changes Output)
These options reduce CPU/PCIe expert misses by substituting nearby VRAM-resident experts or skipping very low-weight missed tail experts. **They change model outputs**—always pair speed measurements with quality checks (code, arithmetic, structured JSON).

| Env Var | Default | What it does | Notes for B580 |
| :--- | :---: | :--- | :--- |
| `STRATA_ROUTE_RESIDENT=<margin>` | `0` (off) | Replaces missed experts at ranks `lo..hi` with the best VRAM-resident expert within `<margin>` logits (e.g. `1e30` = any resident stand-in). | Existing SYCL knob, extended below. |
| `STRATA_ROUTE_RESIDENT_RANKS=lo-hi` | `6-9` | Sets the rank range `[lo, hi]` for `STRATA_ROUTE_RESIDENT` (e.g. `3-9`). | Ranks `0` and `1` should stay exact unless using a tight budget. |
| `STRATA_ROUTE_RESIDENT_EDGE=<logits>` | `-1` (off) | Fractional floor: rank `lo - 1` (e.g. rank `2` when `RANKS=3-9`) is replaced only if a VRAM stand-in is within `<logits>` (HotRod's `"2E1"` rule: `RANKS=3-9`, `EDGE=1.0`, `RESIDENT=1e30`). | Much safer than opening rank 2 unconditionally. |
| `STRATA_ROUTE_BUDGET=<b>` | `0` (off) | Confidence-based stand-ins for ranks `< lo`: replaces a missed pick only when `router_weight * max(logit_gap, 0) <= b` (try `0.05`–`0.20`). | Replaces heavy misses only when a very close stand-in exists; lighter misses accept wider gaps. |
| `STRATA_ROUTE_BUDGET_TAIL=<b>` | `0` (off) | Confidence-based stand-ins for ranks `>= lo` (`router_weight * max(logit_gap, 0) <= b`). | Safer than unlimited `STRATA_ROUTE_RESIDENT=1e30` on tail ranks. |
| `STRATA_TAIL_WEIGHT=<tau>` | `0` (off) | Host-side weight rule: skips any missed expert whose every token in the slice routed at rank `>= floor` with router weight `< tau` (try `0.02`, `0.05`, `0.08`). | Zero CPU compute and zero PCIe fetch for skipped experts. Floor defaults to `lo` from `STRATA_ROUTE_RESIDENT_RANKS` (or `2`; override with `STRATA_TAIL_WEIGHT_FROM=R`). |
| `STRATA_LAYER_LIVE=1` + `STRATA_LAYER_SHARE=<0..1>` | `0` / `0.5` | System-driven per-layer weight rule: builds a running log-binned histogram of missed tail weights per layer and solves per-layer thresholds to skip `SHARE` of missed tail picks where weights are smallest. | Needs ~2,000 missed tail picks (~a few seconds) before skipping starts; settles automatically once thresholds stabilize within 3%. |
| `STRATA_TAIL_CPU=0` / `STRATA_ROUTE_TAIL_SKIP=R` | `1` / `0` (off) | Rank-based tail skip: skips all missed experts whose every token routed at rank `>= R` (`STRATA_TAIL_CPU=0` sets `R = lo` or `3`). | Coarser than `STRATA_TAIL_WEIGHT`; useful as an upper-bound speed reference. |
| `STRATA_RANK_CUTOFF=R` | `0` (off) | Skips ALL experts (even VRAM-resident) at rank `>= R`. | Aggressive diagnostic/testing knob only. |

### Group C — Diagnostics (No Output Change)
| Env Var | Default | What it does |
| :--- | :---: | :--- |
| `STRATA_LAYER_STATS=1` | `0` (off) | Every 1024 verify windows, prints per-depth-band (`B0`..`B3`) percentages of routed picks served by VRAM, PCIe, CPU, or Skipped. Turn off for final speed runs. |

---

## 2. Build & Verify

```bash
bash ./tools/build_sycl.sh -r strata route_resident_parity hotrod_tuning_test
ONEAPI_DEVICE_SELECTOR=level_zero:0 ctest --test-dir build_sycl \
  -R "route_resident_parity|hotrod_" --output-on-failure
```

All 9 CTest cases (`hotrod_gpu_exact`, `hotrod_dispatch_default`, `hotrod_pcie_min1`, `hotrod_tail_rank`, `hotrod_tail_cpu_off`, `hotrod_rank_cutoff`, `hotrod_tail_weight`, `hotrod_layer_live`, and `route_resident_parity` covering both legacy and extended `edge`/`budget`/`budget_tail` modes) should pass before starting A/B sweeps.

---

## 3. Systematic A/B Testing Protocol

### Common Baseline Setup
Keep all base flags identical across runs (matching your `b580-config.txt`), use greedy decoding (`--temperature 0`), and keep a fixed prompt set (e.g., 1 coding prompt, 1 multi-step math prompt, 1 structured JSON prompt, ~300–512 generated tokens each):

```bash
export SYCL_CACHE_PERSISTENT=1
export NEO_CACHE_PERSISTENT=1
export UR_L0_DEVICE_SCOPE_EVENTS=2
export SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2
export STRATA_COPY_ENGINE=1
export STRATA_PARTIAL_PIN_GIB=48
export STRATA_VERIFY_DEVICE_PLAN=1
export STRATA_ATTN_MERGE_ACTIVE=1
export STRATA_MMVQ_DECODE_256GRF=0
export STRATA_B6_MMVQ_ROWS=1
export STRATA_VERIFY_GRAPH=split
export STRATA_VERIFY_GRAPH_CHUNK=4
export STRATA_FUSED_GDN=1

BASE_ARGS="--expert-cache 1000 --resident-experts --max-context 131072 \
  --kv int8 --kv-resident 32768 --prefill 4096 --spec 4 --spec-min-p 0.5 \
  --pool-workers 26 --ple-io ram --pcie-frac 0.2 --mtp-window 8192 \
  --temperature 0 --max-new 384"
```

*(Note: verify the spelling of `UR_L0_DEVICE_SCOPE_EVENTS` and `SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS` in your shell—typos like `SCCOPE` / `LEVVEL` are silently ignored by the runtime.)*

---

### Phase 0: Baseline & Bottleneck Visibility
1. Run the baseline **twice** (discard the first cold run or warm the cache first) and save the greedy output to `baseline.txt`.
2. Run once with `STRATA_LAYER_STATS=1` to see your actual VRAM / PCIe / CPU breakdown across depth bands `B0`–`B3`:
   ```bash
   STRATA_LAYER_STATS=1 ./build_sycl/strata $BASE_ARGS ...
   ```
   * If `pcie=0%` or very low despite `--pcie-frac 0.2`, `STRATA_PCIE_MIN1=1` / `STRATA_PCIE_BALANCE=1` will immediately show up here.
   * If `cpu` is high (e.g., > 40%), Group B routing/tail-skipping will yield the largest gains.

---

### Phase 1: Exact (Bit-Identical) Sweep
Test each exact lever **individually** against Phase 0, and `diff` each run's output against `baseline.txt` (must be 0 differences):

| Run ID | Extra Env Vars | What to check |
| :--- | :--- | :--- |
| **E0** | *(none)* | Baseline `tok/s`, save `baseline.txt`. |
| **E1** | `STRATA_FUSE_WAIT=1` | Does fusing `wait_flag_ge` + `copy_i32_from_mapped` cut Level Zero launch overhead? |
| **E2** | `STRATA_PCIE_MIN1=1` | Does keeping PCIe active on 2–5 miss layers beat E0? |
| **E3** | `STRATA_PCIE_BALANCE=1` | Does dynamic per-layer balancing beat fixed `--pcie-frac 0.2`? (Run at least 2–3 prompts so `PcieModel` converges.) |
| **E4** | `STRATA_VERIFY_GRAPH=direct STRATA_PCIE_EARLY=1` | Compare against `STRATA_VERIFY_GRAPH=direct` alone (E4a) and against E0 (`split`). Note: check if overlapping PCIe fetch outweighs direct-mode launch overhead. |
| **E5** | `STRATA_FUSE_WAIT=1 STRATA_PCIE_BALANCE=1 STRATA_PCIE_MIN1=1` | Combined exact winners. Verify `diff baseline.txt` is empty. |

**Lock in your Phase 1 winner** (keep every exact flag that improved or tied `tok/s` with zero regressions).

---

### Phase 2: Approximate Stand-In Routing Sweep (on top of Phase 1 Winner)
Test these progressive configurations, checking **both** `tok/s` and **answer correctness** (code compiles/runs, math answer exact, JSON valid):

| Run ID | Extra Env Vars (on top of Phase 1 winner) | Profile |
| :--- | :--- | :--- |
| **R1** | `STRATA_ROUTE_RESIDENT=2.0 STRATA_ROUTE_RESIDENT_RANKS=6-9` | Conservative baseline (tail ranks 6–9 within 2 logits). |
| **R2** | `STRATA_ROUTE_RESIDENT=1e30 STRATA_ROUTE_RESIDENT_RANKS=3-9 STRATA_ROUTE_RESIDENT_EDGE=1.0` | HotRod `"2E1"` rule: ranks 3–9 open, rank 2 within 1 logit, ranks 0–1 exact. |
| **R3** | `STRATA_ROUTE_RESIDENT_RANKS=3-9 STRATA_ROUTE_BUDGET=0.05 STRATA_ROUTE_BUDGET_TAIL=0.15` | Conservative confidence budget (`w * gap <= b`). |
| **R4** | `STRATA_ROUTE_RESIDENT_RANKS=3-9 STRATA_ROUTE_BUDGET=0.10 STRATA_ROUTE_BUDGET_TAIL=0.25` | Medium confidence budget (HotRod's typical sweet spot). |
| **R5** | `STRATA_ROUTE_RESIDENT_RANKS=3-9 STRATA_ROUTE_BUDGET=0.20 STRATA_ROUTE_BUDGET_TAIL=0.35` | Aggressive confidence budget—watch closely for typos/JSON glitches. |

---

### Phase 3: Approximate Tail-Skipping Sweep (on top of Phase 1 Winner, and optionally Phase 2 Winner)
Test weight-based vs live per-layer vs rank-based skipping:

| Run ID | Extra Env Vars | Profile |
| :--- | :--- | :--- |
| **T1** | `STRATA_TAIL_WEIGHT=0.05 STRATA_TAIL_WEIGHT_FROM=3` | Flat weight rule: skips missed experts at rank `>= 3` with weight `< 0.05` (~11% of tail picks). |
| **T2** | `STRATA_TAIL_WEIGHT=0.08 STRATA_TAIL_WEIGHT_FROM=3` | Stronger flat weight rule (~64% of tail picks). |
| **T3** | `STRATA_LAYER_LIVE=1 STRATA_LAYER_SHARE=0.30 STRATA_TAIL_WEIGHT_FROM=3` | Live per-layer adaptive thresholds skipping 30% of missed tail picks (allocates skips to low-weight layers). |
| **T4** | `STRATA_LAYER_LIVE=1 STRATA_LAYER_SHARE=0.50 STRATA_TAIL_WEIGHT_FROM=3` | HotRod default share (50% of missed tail picks, adaptive per layer). |
| **T5** | `STRATA_TAIL_CPU=0` *(or `STRATA_ROUTE_TAIL_SKIP=3`)* | Pure rank rule (all missed experts at rank `>= 3` skipped). Upper-bound speed test. |

Finally, test your best Phase 2 + best Phase 3 together (e.g. **R2 + T3** or **R4 + T4**) — note that when stand-ins already cover most tail misses, tail-skipping mainly catches the remaining un-substituted misses.

---

## 4. Results Worksheet

Record at least 2 alternating runs (`A -> B -> B -> A`) per candidate:

| Run ID | Flags | Decode `tok/s` (mean) | Speedup vs E0 | Greedy Parity / Quality Notes |
| :--- | :--- | ---: | ---: | :--- |
| **E0** | Baseline | | 1.00x | Exact reference |
| **E1** | `FUSE_WAIT=1` | | | Must match E0 bit-for-bit |
| **E2** | `PCIE_MIN1=1` | | | Must match E0 bit-for-bit |
| **E3** | `PCIE_BALANCE=1` | | | Must match E0 bit-for-bit |
| **E4** | `GRAPH=direct PCIE_EARLY=1` | | | Must match E0 bit-for-bit |
| **E5** | Best Exact Combo | | | Must match E0 bit-for-bit |
| **R2** | `2E1` stand-ins | | | Check code / math / JSON |
| **R4** | `BUDGET=0.10 / TAIL=0.25` | | | Check code / math / JSON |
| **T3** | `LAYER_LIVE=1 SHARE=0.30` | | | Check code / math / JSON |
| **T4** | `LAYER_LIVE=1 SHARE=0.50` | | | Check code / math / JSON |
| **Combo** | Best Exact + Best Approx | | | Check code / math / JSON |

### Selection Rule
1. **Always-on production config (exact):** Use the fastest **Phase 1 (Exact)** combination (`E5`).
2. **High-throughput interactive config (approximate):** Add the fastest **Phase 2 / Phase 3** combination that passes 100% of your coding, math, and JSON checks with no formatting/loop regressions.
