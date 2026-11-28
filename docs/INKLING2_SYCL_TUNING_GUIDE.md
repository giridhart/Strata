# Inkling-2-Mini SYCL Platform Tuning Guide

This guide explains how to systematically test build flags and runtime environment variables on any Intel Arc / Xe2 GPU (Arc B580 `bmg_g21`, Arc Pro B70 `bmg_g31`, Lunar Lake `lnl_m`, Alchemist `acm_g10`) to find the fastest **parity-clean** configuration for `strata-inkling-sycl`.

Every optimization flag in `engine/src/inkling2/sycl/` is **opt-in**. The default path stays unchanged until a flag is proven both parity-clean and faster on the target GPU.

---

## 1. Ground Rules

1. **Test one variable at a time.** Hold all other flags fixed while sweeping one knob so you know what helped and what regressed.
2. **Parity before speed.** Never keep a flag that fails `sycl_inkling2_kernels_parity`, `sycl_inkling2_graph_limit_test`, or greedy (`temp=0`) token-ID match on a real model.
3. **Always set `IGC_EnableGEPSimplification=0` on Xe2 / Battlemage.** On `bmg_g21`, `bmg_g31`, and `lnl_m`, IGC GEP simplification can cause miscompilation or `UR_RESULT_ERROR_DEVICE_LOST`. The runtime logs a startup warning if this variable is not `0`.
4. **Record your environment with every result.** Always save `sycl-ls`, `icpx --version`, the exact env vars, and the benchmark JSON output.

---

## 2. Available Tuning Knobs

### Build-Time Options (`engine/src/inkling2/sycl/build_inkling_sycl.sh`)

| Variable | Values | Purpose |
|---|---|---|
| `TARGET_MODE` (arg 2) | `bmg` \| `fat` \| `acm` \| `jit` | Selects AOT/JIT targets (`bmg` targets `intel_gpu_bmg_g21,intel_gpu_bmg_g31,spir64` when `ocloc` is installed, else `spir64` JIT). |
| `AOT_DEVICE` | `bmg_g21` \| `bmg_g31` \| `lnl_m` \| `acm_g10` | Overrides the AOT target device (adds `spir64` fallback). |
| `GRF_MODE` | unset \| `128` \| `256` | Forces AOT compiler register file mode (`-ze-opt-regular-grf-kernel` vs `-ze-opt-large-register-file`). |
| `STRATA_SYCL_USE_ONEMKL` | `0` (default) \| `1` | Links legacy oneMKL BLAS instead of zero-oneMKL native SG16 `joint_matrix` XMX. |

### Runtime Environment Variables

| Env Var | Values | Default | Scope |
|---|---|---|---|
| `IGC_EnableGEPSimplification` | `0` | unset | **Required on Xe2/Battlemage** to avoid IGC miscompilation / `DEVICE_LOST`. |
| `STRATA_INK_SYCL_MOE_VEC` | `0` \| `1` | `0` | Vectorized MoE `w13` + `w2` (`SF4` `uint2`+`float4`, `MX8` coalesced `uchar16`+`float4`, templated `S2..S8`). |
| `STRATA_INK_SYCL_KREG` | `1` \| `2` \| `4` | `1` | Multi-row-per-subgroup dense decode GEMV (`Q8_ROW` dp4a, `MX8`, `S2..S8`). |
| `STRATA_INK_SYCL_256GRF` | unset \| `0` \| `1` | unset | Per-kernel GRF override (`0` = 128 GRF, `1` = 256 GRF) on opt-in vectorized/KREG kernels. |
| `STRATA_INK_SYCL_VEC_LSC` | `0` \| `1` | `0` | Independent override for `k_linear_q8row_vec_dp4a_sg16` in `InkSyRunner`. |
| `STRATA_INK_SYCL_XMX` | `0` \| `1` | `0` | Independent override for SG16 `joint_matrix` XMX prefill GEMM (`M >= 16`). |
| `INK_SY_GRAPH` | `0` \| `1` | `0` | Enables SYCL `command_graph` capture/replay for decode & verify steps. |
| `STRATA_INK_SYCL_GRAPH_CAP` | `-1` \| `0` \| `N` | `16` | LRU cap on cached executable `command_graph` keys (`-1` disables graphs, `0` unbounded, `N` caps at `N`). |
| `STRATA_INK_SYCL_L0_V2` | `0` \| `1` | `0` | Independent override for Level Zero V2 immediate command lists. |
| `STRATA_INK_INTELARC` | `0` \| `1` | `0` | Enables the full Intel Arc profile (auto-clamped to `EXACT` on Battlemage unless individual flags above override it). |
| `STRATA_INK_INTELARC_EXACT` | `0` \| `1` | `0` | Restricts `STRATA_INK_INTELARC=1` to 100% lossless exact-math optimizations. |
| `SYCL_CACHE_PERSISTENT` | `1` | unset | Persists compiled SPIR-V/IGC kernel binaries across runs to cut startup time. |
| `NEO_CACHE_PERSISTENT` | `1` | unset | Persists Intel NEO driver shader cache across runs. |
| `STRATA_INK_PROF` | `0` \| `1` | `0` | Prints per-stage host/device timing breakdown during `strata-inkling-sycl` runs. |

---

## 3. Step-by-Step Tuning Protocol

### Stage 0: Build the Test & Benchmark Harness and Capture Baseline

Source oneAPI and build the standalone binary plus the parity and Google Benchmark targets:

```bash
source /opt/intel/oneapi/setvars.sh

# 1. Record hardware & toolchain
sycl-ls
icpx --version | head -n 2

# 2. Build strata-inkling-sycl for your target (example: Arc B580 bmg_g21)
IGC_EnableGEPSimplification=0 AOT_DEVICE=bmg_g21 \
  engine/src/inkling2/sycl/build_inkling_sycl.sh build-inkling-bmg bmg

# 3. Configure and build parity tests + microbenchmark harness
cmake -S engine -B build-inkling-sycl \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DSTRATA_ENABLE_SYCL=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_PERF=ON
cmake --build build-inkling-sycl \
  --target sycl_inkling2_kernels_parity sycl_inkling2_graph_limit_test perf_inkling2_kernels -j$(nproc)

# 4. Verify baseline parity and graph cache hygiene
IGC_EnableGEPSimplification=0 ctest --test-dir build-inkling-sycl -R "sycl_inkling2_" --output-on-failure

# 5. Record baseline microbenchmarks
mkdir -p engine/perf/results
IGC_EnableGEPSimplification=0 ./build-inkling-sycl/perf_inkling2_kernels \
  --benchmark_min_time=0.25s \
  --benchmark_out=engine/perf/results/baseline.json \
  --benchmark_out_format=json
```

### Stage 1: MoE Vectorization (`STRATA_INK_SYCL_MOE_VEC`)

Test `STRATA_INK_SYCL_MOE_VEC=1` against baseline on parity and all MoE formats (`SF4`, `MX8`, `S4`):

```bash
IGC_EnableGEPSimplification=0 STRATA_INK_SYCL_MOE_VEC=1 \
  ./build-inkling-sycl/sycl_inkling2_kernels_parity

IGC_EnableGEPSimplification=0 STRATA_INK_SYCL_MOE_VEC=1 \
  ./build-inkling-sycl/perf_inkling2_kernels \
  --benchmark_min_time=0.25s --benchmark_filter="BM_MoE" \
  --benchmark_out=engine/perf/results/moe_vec.json --benchmark_out_format=json
```

**Decision rule:** Enable `STRATA_INK_SYCL_MOE_VEC=1` if all parity checks pass (`ALL PASS`) and your model's expert format (`SF4`, `MX8`, or `S_K`) shows higher GB/s and lower µs/call.

### Stage 2: Dense Decode Row-Reuse (`STRATA_INK_SYCL_KREG`)

Sweep `KREG=1`, `2`, `4` on dense decode GEMV (`Q8_ROW` dp4a, `MX8`, `S4_K`):

```bash
for kreg in 1 2 4; do
  echo "=== KREG=${kreg} ==="
  IGC_EnableGEPSimplification=0 STRATA_INK_SYCL_KREG=${kreg} \
    ./build-inkling-sycl/sycl_inkling2_kernels_parity | grep -E "Q8_ROW|MX8 linear|S4_K linear|ALL PASS|FAIL"
  IGC_EnableGEPSimplification=0 STRATA_INK_SYCL_KREG=${kreg} \
    ./build-inkling-sycl/perf_inkling2_kernels \
    --benchmark_min_time=0.25s --benchmark_filter="BM_Dense" \
    --benchmark_out="engine/perf/results/dense_kreg${kreg}.json" --benchmark_out_format=json
done
```

**Decision rule:** Pick the `KREG` value with the lowest µs/call on your model's dense formats. Typically `KREG=2` is the best balanced choice (faster on both `Q8_ROW` and `S_K` without spilling); `KREG=4` can be faster on `Q8_ROW` at larger batch sizes (`M=8`) but may regress `S_K` due to register pressure.

### Stage 3: Register File Size (`STRATA_INK_SYCL_256GRF` / `GRF_MODE`)

Compare compiler default (unset), forced 128 GRF (`0`), and forced 256 GRF (`1`):

```bash
for grf in "" 0 1; do
  tag="${grf:-default}"
  echo "=== 256GRF=${tag} ==="
  IGC_EnableGEPSimplification=0 STRATA_INK_SYCL_MOE_VEC=1 STRATA_INK_SYCL_KREG=2 \
    STRATA_INK_SYCL_256GRF="${grf}" \
    ./build-inkling-sycl/perf_inkling2_kernels \
    --benchmark_min_time=0.25s --benchmark_filter="BM_MoE.*(/1/|/16/)|BM_Dense" \
    --benchmark_out="engine/perf/results/grf_${tag}.json" --benchmark_out_format=json
done
```

**Decision rule:** Keep `STRATA_INK_SYCL_256GRF` unset (or `0`) unless `1` measurably improves throughput. Memory-bound decode kernels usually prefer 128 GRF for higher thread occupancy.

### Stage 4: Prefill XMX (`STRATA_INK_SYCL_XMX`)

Check `k_jm_gemm_q8row_s8_sg16` parity and speed against `k_gemm_q8row_sg16`:

```bash
IGC_EnableGEPSimplification=0 ./build-inkling-sycl/sycl_inkling2_kernels_parity \
  | grep -E "q8row|ALL PASS|FAIL"

IGC_EnableGEPSimplification=0 ./build-inkling-sycl/perf_inkling2_kernels \
  --benchmark_min_time=0.25s --benchmark_filter="BM_Prefill_Q8Row" \
  --benchmark_out=engine/perf/results/prefill_xmx.json --benchmark_out_format=json
```

**Decision rule:** Enable `STRATA_INK_SYCL_XMX=1` if `k_jm_gemm_q8row_s8 (XMX)` passes parity without `DEVICE_LOST` and beats `k_gemm_q8row_sg16` at `M=64, 128`.

### Stage 5: SYCL `command_graph` Cache Cap & Level Zero V2

1. Stress the bounded LRU cache with your target cap (`8`, `16`, `32`):
   ```bash
   for cap in 8 16 32; do
     IGC_EnableGEPSimplification=0 STRATA_INK_SYCL_GRAPH_CAP=${cap} \
       ./build-inkling-sycl/sycl_inkling2_graph_limit_test
   done
   ```
2. On multi-slot serving (up to 32 slots), keep `STRATA_INK_SYCL_GRAPH_CAP=16` (default) or `32` so Level Zero command lists stay bounded and never hit `UR_RESULT_ERROR_DEVICE_LOST`.

### Stage 6: End-to-End Real Model A/B Verification

Before locking in a config for a platform, run `strata-inkling-sycl` on a real Inkling-2-Mini model (`.strata` or `.gguf`) with greedy decoding (`--temp 0`):

```bash
MODEL=/path/to/inkling2-mini.strata
PROMPT="Explain why memory bandwidth limits autoregressive token decode in one paragraph."

# 1. Baseline run
IGC_EnableGEPSimplification=0 SYCL_CACHE_PERSISTENT=1 NEO_CACHE_PERSISTENT=1 \
  ./build-inkling-bmg/strata-inkling-sycl --model "$MODEL" --prompt "$PROMPT" \
  --max-tokens 128 --temp 0 --seed 42 > /tmp/ink_base.txt

# 2. Candidate optimized config run (add STRATA_INK_PROF=1 to inspect per-stage timings)
IGC_EnableGEPSimplification=0 SYCL_CACHE_PERSISTENT=1 NEO_CACHE_PERSISTENT=1 \
  STRATA_INK_SYCL_MOE_VEC=1 STRATA_INK_SYCL_KREG=2 STRATA_INK_SYCL_XMX=1 \
  INK_SY_GRAPH=1 STRATA_INK_SYCL_GRAPH_CAP=16 \
  ./build-inkling-bmg/strata-inkling-sycl --model "$MODEL" --prompt "$PROMPT" \
  --max-tokens 128 --temp 0 --seed 42 > /tmp/ink_opt.txt

# 3. Verify greedy output match
diff -u /tmp/ink_base.txt /tmp/ink_opt.txt
```

Also test with `STRATA_INK_SYCL_VEC_LSC=1` and `STRATA_INK_SYCL_L0_V2=1` one at a time on the real model to confirm they improve tok/s and stay crash-free on long prompts and multi-slot runs.

---

## 4. Automated Microbenchmark Sweep Script

Save and run this script on a new GPU to sweep Stages 0–4 automatically and store all JSON results in `engine/perf/results/<tag>/`:

```bash
#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-target_$(date +%Y%m%d_%H%M%S)}"
OUT="engine/perf/results/${TAG}"
BIN="./build-inkling-sycl"
mkdir -p "${OUT}"

export IGC_EnableGEPSimplification=0
export SYCL_CACHE_PERSISTENT=1
export NEO_CACHE_PERSISTENT=1

{ sycl-ls; icpx --version | head -n 2; } > "${OUT}/env.txt"

echo "[1/5] Parity + graph limit gate..."
"${BIN}/sycl_inkling2_kernels_parity" > "${OUT}/parity_base.txt"
STRATA_INK_SYCL_MOE_VEC=1 STRATA_INK_SYCL_KREG=2 "${BIN}/sycl_inkling2_kernels_parity" > "${OUT}/parity_opt.txt"
"${BIN}/sycl_inkling2_graph_limit_test" > "${OUT}/graph_limit.txt"

echo "[2/5] Baseline perf..."
"${BIN}/perf_inkling2_kernels" --benchmark_min_time=0.25s \
  --benchmark_out="${OUT}/00_baseline.json" --benchmark_out_format=json

echo "[3/5] MoE vec..."
STRATA_INK_SYCL_MOE_VEC=1 "${BIN}/perf_inkling2_kernels" --benchmark_min_time=0.25s \
  --benchmark_filter="BM_MoE" \
  --benchmark_out="${OUT}/01_moe_vec.json" --benchmark_out_format=json

echo "[4/5] Dense KREG=2 and KREG=4..."
for kreg in 2 4; do
  STRATA_INK_SYCL_KREG=${kreg} "${BIN}/perf_inkling2_kernels" --benchmark_min_time=0.25s \
    --benchmark_filter="BM_Dense" \
    --benchmark_out="${OUT}/02_dense_kreg${kreg}.json" --benchmark_out_format=json
done

echo "[5/5] 256GRF A/B..."
STRATA_INK_SYCL_MOE_VEC=1 STRATA_INK_SYCL_KREG=2 STRATA_INK_SYCL_256GRF=1 \
  "${BIN}/perf_inkling2_kernels" --benchmark_min_time=0.25s \
  --benchmark_filter="BM_MoE.*(/1/|/16/)|BM_Dense" \
  --benchmark_out="${OUT}/03_256grf.json" --benchmark_out_format=json

echo "Done. Results saved in ${OUT}/"
```

---

## 5. Reference Measured Results & Example Config

Measured on `Intel(R) Graphics [0x64a0]` (Lunar Lake Xe2, Intel Core Ultra 5 226V, 7 Xe2 cores), Intel oneAPI DPC++ 2026.1.1, Level Zero V2 NEO 26.05.37020.3, `IGC_EnableGEPSimplification=0` (full JSONs in [b580_inkling2_baseline.json](../engine/perf/results/b580_inkling2_baseline.json) and [b580_inkling2_optimized.json](../engine/perf/results/b580_inkling2_optimized.json)):

| Knob | Measured Effect on Lunar Lake Xe2 (`0x64a0`) | Verdict on `0x64a0` |
|---|---|---|
| `STRATA_INK_SYCL_MOE_VEC=1` | `MX8` **1.81x–2.02x**, `S4` **1.34x–1.50x**, `SF4` **1.06x** faster; parity `rel_l2 ~ 2e-7` | **Keep ON** |
| `STRATA_INK_SYCL_KREG=2` | `Q8_ROW` **1.17x–1.74x** (bit-exact), `S4_K` **1.48x–1.61x** faster | **Keep `2`** |
| `STRATA_INK_SYCL_KREG=4` | `Q8_ROW` slightly faster at `M=8` (201 vs 204 µs), but regressed `S4_K` (294 vs 153 µs) | Use only if model has no `S_K` dense layers |
| `STRATA_INK_SYCL_256GRF=1` | ~15–20% slower on MoE `MX8` and `S4` vs default 128 GRF | **Leave unset** |
| `STRATA_INK_SYCL_XMX=1` | `Q8_ROW` prefill GEMM **1.47x** (`M=64`) to **1.61x** (`M=128`), bit-exact to dp4a | **Keep ON** |
| `STRATA_INK_SYCL_GRAPH_CAP=16` | 96 distinct graph keys ran cleanly (80 LRU evictions, 0 `DEVICE_LOST`) | **Keep `16`** |

### Resulting Platform Config (Example `.env` for Lunar Lake Xe2 `0x64a0`)

```bash
export IGC_EnableGEPSimplification=0
export SYCL_CACHE_PERSISTENT=1
export NEO_CACHE_PERSISTENT=1
export STRATA_INK_SYCL_MOE_VEC=1
export STRATA_INK_SYCL_KREG=2
export STRATA_INK_SYCL_XMX=1
export INK_SY_GRAPH=1
export STRATA_INK_SYCL_GRAPH_CAP=16
```

On a different GPU (such as Arc B580 `bmg_g21` with 20 Xe2 cores and 456 GB/s GDDR6, or Arc Pro B70 `bmg_g31`), re-run Stages 0–6 above rather than assuming identical winners—especially `KREG=2` vs `4`, `GRF_MODE=128` vs `256`, and `STRATA_INK_SYCL_GRAPH_CAP=16` vs `32`.
