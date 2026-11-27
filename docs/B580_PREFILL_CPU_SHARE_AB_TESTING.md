# Prefill CPU Share (`STRATA_PREFILL_CPU_SHARE`) — A/B Testing Plan (Arc B580, Native SYCL)

This document defines the target-platform verification and A/B benchmark plan on **Intel Arc B580 (12 GB)** for the native SYCL port of `STRATA_PREFILL_CPU_SHARE` in `src/prefill/sycl/prefill.cpp` and `src/program/sycl/generate.cpp`.

## What Was Ported

- **Concurrent CPU + GPU expert execution for routed-only prefill chunks** (`src/prefill/sycl/prefill.cpp`):
  - Opt-in via `STRATA_PREFILL_CPU_SHARE=auto` (or a fixed fraction in `(0, 1]`, e.g., `0.3`). When unset or `0`, the default SYCL prefill path is unchanged and byte-identical.
  - On non-`stream_all` chunks, non-resident routed experts with `cnt <= MAXT` tokens (fewest tokens first) are dispatched to the existing `--pool-workers` CPU `ExpertPool` in parallel with GPU PCIe streaming and compute.
  - In `auto` mode, per-layer SYCL profiling events and host CPU timings dynamically balance the split (`g / (c + g)`) and gate sharing off automatically if CPU contention makes a layer slower (`STRATA_DBG_CPU_GATE=1` logs each gate decision).
  - Wired `Prefill::arm_cpu_share()` and `Prefill::set_cpu_pool(&pool)` in both CLI and `--serve` paths of `src/program/sycl/generate.cpp`, and added CPU expert count/share reporting to the prefill summary line.

---

## Important Config Interactions

1. **Unset `STRATA_PREFILL_STREAM_MIN` when testing `<= 4k` prompts with CPU share.**
   - `STRATA_PREFILL_CPU_SHARE` only applies to routed-only chunks (`T < stream_all_min()`).
   - If `STRATA_PREFILL_STREAM_MIN=512` is set, it overrides `STRATA_PREFILL_CPU_SHARE_MAX` and forces `stream_all = true` for all `T >= 512`, so CPU share will **only** fire for `< 512`-token chunks (e.g. +200-token chat follow-ups).
   - Leaving `STRATA_PREFILL_STREAM_MIN` **unset** lets `STRATA_PREFILL_CPU_SHARE_MAX` (default `3072`, or set `4096`) control the threshold automatically.
2. **Fix the two Level Zero event env-var typos if using your existing B580 config:**
   - `UR_L0_DEVICE_SCOPE_EVENTS=2` (not `SCCOPE`)
   - `SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2` (not `LEVVEL`)

---

## 1. Build on Target

```bash
bash ./tools/build_sycl.sh -r strata
```

Set shared environment variables for all runs below:

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export SYCL_CACHE_PERSISTENT=1
export NEO_CACHE_PERSISTENT=1
export UR_L0_DEVICE_SCOPE_EVENTS=2
export SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2
export STRATA_COPY_ENGINE=1
export STRATA_PARTIAL_PIN_GIB=48
export STRATA_PREFILL_TIMING=1
export STRATA_PREFILL_LAYER_MAJOR=1
export STRATA_PREFILL_LM_TOKENS=16384

export STRATA_BIN=./build_sycl/strata
export MODEL=/path/to/Qwen3.8-Flash-Next-shard-00001.gguf
export BASE_ARGS="--native $MODEL --expert-cache 1000 --resident-experts --max-context 131072 --kv int8 --kv-resident 32768 --prefill 4096 --spec 4 --spec-min-p 0.5 --pool-workers 26 --ple-io ram --pcie-frac 0.2 --mitp-window 8192 --max-tokens 16 --temperature 0"
```

---

## 2. Correctness & Default-Path Bit-Identity Check

Run on a 1k–2k prompt before timing:

| Case | Env Vars | Pass Criteria |
| :--- | :--- | :--- |
| **Ref-0 (Default unset)** | *(unset `STRATA_PREFILL_CPU_SHARE`)* `STRATA_STATE_HASH_GDN=1` | Baseline `GDN_HASH` and tokens |
| **Ref-1 (Explicit off)** | `STRATA_PREFILL_CPU_SHARE=0 STRATA_STATE_HASH_GDN=1` | Identical `GDN_HASH` and tokens to Ref-0 |
| **Auto** | `STRATA_PREFILL_CPU_SHARE=auto STRATA_DBG_CPU_GATE=1` *(unset `STRATA_PREFILL_STREAM_MIN`)* | Finite outputs, coherent tokens matching Ref-0, logs show `cpu N (share X%)` |
| **Fixed 0.35** | `STRATA_PREFILL_CPU_SHARE=0.35` *(unset `STRATA_PREFILL_STREAM_MIN`)* | Finite outputs, coherent tokens matching Ref-0, logs show `cpu N (share 35%)` |

*(Note: CPU share uses the CPU's native activation quantization format, so `GDN_HASH` will differ slightly from GPU-only while greedy tokens and perplexity remain equivalent.)*

---

## 3. A/B Benchmark Matrix — Short Prompts (`<= 4k`)

Prepare prompts of length **200, 512, 1024, 2048, and 4096** tokens (`prompt_200.tokens`, `prompt_512.tokens`, `prompt_1k.tokens`, `prompt_2k.tokens`, `prompt_4k.tokens`). Run **3 warm iterations** per cell (discarding the first cold run) and record median **prompt tok/s** and **wall ms**:

| Arm | Env Vars | Target Prompt Lengths |
| :--- | :--- | :--- |
| **A0 (Baseline, current config)** | `STRATA_PREFILL_STREAM_MIN=512 STRATA_PREFILL_CPU_SHARE=0` | 200, 512, 1k, 2k, 4k |
| **A1 (Auto, default max 3072)** | `unset STRATA_PREFILL_STREAM_MIN; STRATA_PREFILL_CPU_SHARE=auto` | 200, 512, 1k, 2k, 4k |
| **A2 (Auto, max 4096)** | `unset STRATA_PREFILL_STREAM_MIN; STRATA_PREFILL_CPU_SHARE=auto STRATA_PREFILL_CPU_SHARE_MAX=4096` | 200, 512, 1k, 2k, 4k |
| **A3 (Fixed 0.25, max 4096)** | `unset STRATA_PREFILL_STREAM_MIN; STRATA_PREFILL_CPU_SHARE=0.25 STRATA_PREFILL_CPU_SHARE_MAX=4096` | 200, 512, 1k, 2k, 4k |
| **A4 (Fixed 0.35, max 4096)** | `unset STRATA_PREFILL_STREAM_MIN; STRATA_PREFILL_CPU_SHARE=0.35 STRATA_PREFILL_CPU_SHARE_MAX=4096` | 200, 512, 1k, 2k, 4k |
| **A5 (Fixed 0.50, max 4096)** | `unset STRATA_PREFILL_STREAM_MIN; STRATA_PREFILL_CPU_SHARE=0.50 STRATA_PREFILL_CPU_SHARE_MAX=4096` | 200, 512, 1k, 2k, 4k |

### Sweep Script (CLI)

```bash
for LEN in 200 512 1k 2k 4k; do
  PROMPT="/path/to/prompt_${LEN}.tokens"
  for ARM in "off" "auto_3072" "auto_4096" "fix_0.25" "fix_0.35" "fix_0.50"; do
    echo "=== len=${LEN} arm=${ARM} ==="
    for RUN in 1 2 3; do
      case "$ARM" in
        off)       env STRATA_PREFILL_STREAM_MIN=512 STRATA_PREFILL_CPU_SHARE=0 $STRATA_BIN $BASE_ARGS --tokens-file "$PROMPT" ;;
        auto_3072) env -u STRATA_PREFILL_STREAM_MIN STRATA_PREFILL_CPU_SHARE=auto $STRATA_BIN $BASE_ARGS --tokens-file "$PROMPT" ;;
        auto_4096) env -u STRATA_PREFILL_STREAM_MIN STRATA_PREFILL_CPU_SHARE=auto STRATA_PREFILL_CPU_SHARE_MAX=4096 $STRATA_BIN $BASE_ARGS --tokens-file "$PROMPT" ;;
        fix_0.25)  env -u STRATA_PREFILL_STREAM_MIN STRATA_PREFILL_CPU_SHARE=0.25 STRATA_PREFILL_CPU_SHARE_MAX=4096 $STRATA_BIN $BASE_ARGS --tokens-file "$PROMPT" ;;
        fix_0.35)  env -u STRATA_PREFILL_STREAM_MIN STRATA_PREFILL_CPU_SHARE=0.35 STRATA_PREFILL_CPU_SHARE_MAX=4096 $STRATA_BIN $BASE_ARGS --tokens-file "$PROMPT" ;;
        fix_0.50)  env -u STRATA_PREFILL_STREAM_MIN STRATA_PREFILL_CPU_SHARE=0.50 STRATA_PREFILL_CPU_SHARE_MAX=4096 $STRATA_BIN $BASE_ARGS --tokens-file "$PROMPT" ;;
      esac 2>&1 | grep -E "strata generate: prefill|cpu gate:"
    done
  done
done
```

---

## 4. Chat / Agent Multi-Turn Prompt Reuse A/B (`--serve`)

To measure the **+200-token follow-up** (cached prefix reuse) in server mode:

1. Start the server with Arm A0 (`STRATA_PREFILL_CPU_SHARE=0`), then repeat with the winning arm from Section 3 (e.g., `STRATA_PREFILL_CPU_SHARE=auto STRATA_PREFILL_CPU_SHARE_MAX=4096`, `STRATA_PREFILL_STREAM_MIN` unset):
   ```bash
   env -u STRATA_PREFILL_STREAM_MIN \
     STRATA_PREFILL_CPU_SHARE=auto \
     STRATA_PREFILL_CPU_SHARE_MAX=4096 \
     $STRATA_BIN $BASE_ARGS --serve
   ```
2. Send Turn 1 (e.g., 4k system + context prompt) so the KV/GDN prefix is cached.
3. Send Turn 2 appending **+200 tokens** (and separately **+512** and **+1024** tokens) of new user/tool output to the exact same prefix.
4. Record **TTFT (time to first token)** and **delta prefill tok/s** for Turn 2 across 3 runs of each arm.

---

## 5. Long-Prompt No-Regression Check (`16k` & `32k`)

Verify that `STRATA_PREFILL_CPU_SHARE=auto` does not regress layer-major prefill on 16k and 32k prompts:

| Prompt | Baseline (`CPU_SHARE=0`, `STREAM_MIN=512`) | Patched (`CPU_SHARE=auto`, `CPU_SHARE_MAX=4096`, `STREAM_MIN` unset) | Expected |
| :--- | :--- | :--- | :--- |
| **16k** | ~950 tok/s | Measure 3x median | Within noise (~0% delta; layer-major handles `n > 4096`) |
| **32k** | ~1025 tok/s | Measure 3x median | Within noise (~0% delta) |

---

## 6. Interpreting Results & Tuning Tips

- **Check `STRATA_DBG_CPU_GATE=1` on `auto`:**
  - If the gate says `-> do not share` on most layers, your 26 CPU workers may be competing with the `Stager` host-copy threads or PCIe submission thread. Try lowering `--pool-workers` slightly (e.g., `20` or `22`) or testing a fixed share (`0.25`–`0.40`).
- **Compare `A1` (`max=3072`) vs `A2` (`max=4096`) at 4k:**
  - At `T = 4096`, `A1` uses `stream_all` (GPU-only sequential streaming), while `A2` uses routed-only + CPU co-execution. Whichever is faster at 4k tells you the optimal `STRATA_PREFILL_CPU_SHARE_MAX` crossover point for your CPU + B580 pair.
- **If `200`–`1k` wins big but `2k`–`4k` is slower with CPU share:**
  - Set `STRATA_PREFILL_CPU_SHARE=auto` and `STRATA_PREFILL_CPU_SHARE_MAX=1024` (or `2048`) so chat follow-ups and short tool outputs get CPU co-execution while `>= 2k` stays on `stream_all`.
