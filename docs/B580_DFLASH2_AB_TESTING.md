# DFlash2 vs. MTP A/B Testing Guide (Xeon w7-3465X + Arc B580, IQ3_S)

This guide explains how to build, validate, and run a controlled A/B benchmark comparing **MTP** against **DFlash2** on **Qwen3.8-Flash-Next IQ3_S** with an **Intel Arc B580 (12 GB)** and **Xeon w7-3465X**.

---

## 1. Which SYCL Binary to Build

This repository has two SYCL build trees:

* **`sycl/` (`cmake -S sycl -B build-sycl` -> `build-sycl/strata`)**: Upstream's DPCT-migrated SYCL engine (compiles `.dp.cpp` files). **Includes DFlash2** (`STRATA_HAVE_DFLASH2=1`) and the planar Q6_K draft-head fix (`#5286`, `Dflash2Drafter::gather_draft_head`). **Use this build for DFlash2 A/B testing.**
* **Root `CMakeLists.txt` (`tools/build_sycl.sh` -> `build_sycl/strata`)**: The fork's native `.cpp` SYCL port (`src/*/sycl/`). Currently compiles **without** DFlash2 (`STRATA_HAVE_DFLASH2` is disabled for `STRATA_ENABLE_SYCL`).

### Build and verify the planar Q6_K fix

Source Intel oneAPI and build `strata` plus `dflash2_head_test` (optionally add `-DSTRATA_SYCL_AOT=bmg-g21` for Arc B580 ahead-of-time compilation, or omit it for SPIR-V JIT):

```bash
source /opt/intel/oneapi/setvars.sh --force

cmake -S sycl -B build-sycl -G Ninja \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DSTRATA_SYCL_PARITY=ON \
  -DSTRATA_SYCL_IQ_DECODE_FAST=ON

cmake --build build-sycl -j "$(nproc)" --target strata dflash2_head_test dflash2_bench
```

*(If your `ocloc` version crashes on ESIMD int8 kernels with AOT enabled, add `-DSTRATA_SYCL_I8_ESIMD=OFF`.)*

Run the unit tests first to confirm planar Q6_K draft-head gathering works on the Arc GPU:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build-sycl/dflash2_head_test
# Expected: "all ok" (exit 0)

ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build-sycl/dflash2_head_test --legacy
# Expected: FAIL (proves the pre-#5286 GGUF-row gather fails on planar Q6_K)
```

*(Optional microbenchmark of the DFlash2 Q4_0 drafter kernels alone:)*

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build-sycl/dflash2_bench
```

---

## 2. Download & Pack the DFlash2 Drafter (~1.2 GB download -> 522 MB pack)

DFlash2 is **only a small 5-layer drafter** (not a full 80+ GB model). It runs on top of your existing IQ3_S Qwen3.8-Flash-Next GGUF and **replaces** the MTP layer (only one drafter is loaded at a time).

| Drafter | Approx. VRAM (SYCL) | Delta vs. MTP (12 GB B580) |
| :--- | :---: | :--- |
| MTP layer | ~1,082 MiB | baseline |
| DFlash2 (`--q4_0`, BF16 selector) | ~890 MiB | **-192 MiB** (~+95–100 IQ3_S expert cache slots) |

> **Important SYCL note:** `sycl/src/core/dflash2.cpp` reads **BF16 selector codebooks only**. Do **not** pass `--q4_0-selector` (that layout is CUDA/HIP/Metal only and SYCL will reject it).

### Download the two pinned files

```bash
mkdir -p dflash2-src && cd dflash2-src

curl -fL --retry 3 -o config.json \
  "https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/resolve/a22a1744e85c17364dcc54a6a420af7135d4c7f6/config.json"

curl -fL --retry 3 -o model.safetensors \
  "https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/resolve/a22a1744e85c17364dcc54a6a420af7135d4c7f6/model.safetensors"
```

### Pack into the SYCL runtime file (numpy only)

```bash
python3 /path/to/Strata/tools/dflash2_pack.py \
  model.safetensors config.json \
  qwen3.8-flash-next-dflash2.q4_0.pack \
  --q4_0
```

Verify the packed file (522,458,112 bytes):

```bash
echo "cda40d62d0eb8f37ed6e272142641a1d2f0de4a32f9e14d0416e7a71534849f1  qwen3.8-flash-next-dflash2.q4_0.pack" | sha256sum -c -
```

> **Note:** `--draft dflash2` still needs `--mtp /path/to/mtp_dir` (or `--mtp-draft-vocab /path/to/draft_vocab.bin`) so it can load `draft_vocab.bin`. The MTP layer weights are **not** loaded when `--draft dflash2` is set.

---

## 3. Controlled A/B Protocol

To keep comparisons fair on a hybrid Xeon + B580 setup:

1. **Pin identical model, KV, prefill, CPU pool, and PCIe settings** across both arms.
2. **Use greedy decoding** (`--temp 0`) and identical prompts + `--max-new` so you can verify output parity and compare exact acceptance rates and `tok/s`.
3. **Alternate arm order (`A -> B -> B -> A`)** across at least 3 rounds to cancel thermal/boost drift and page-cache effects.
4. **Test at least two prompt regimes:**
   * **Short/medium coding or structured output** (e.g., 512–2048 input tokens, 256–512 output tokens) — DFlash2 typically shines here.
   * **Long-context or reasoning/chat** (e.g., 4096 input tokens, 512+ output tokens) — checks whether wider draft windows trigger extra CPU expert misses.
5. **Run two B580 cache comparisons:**
   * **Fixed-cache A/B:** identical `--expert-cache 1000` on both arms (isolates pure drafter speed/acceptance).
   * **VRAM-equalized A/B:** bump `--expert-cache` on the DFlash2 arm by **+95–100 slots** (or use `--expert-cache auto` on both arms to reclaim the ~192 MiB saved by DFlash2).

### Shared environment (from `b580-config.txt`)

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export SYCL_CACHE_PERSISTENT=1
export NEO_CACHE_PERSISTENT=1
export UR_L0_DEVICE_SCOPE_EVENTS=2
export ZES_ENABLE_SYSMAN=1
export STRATA_COPY_ENGINE=1
export STRATA_PARTIAL_PIN_GIB=48
export STRATA_ATTN_MERGE_ACTIVE=1
export STRATA_MMVQ_DECODE_256GRF=0
export STRATA_B6_MMVQ_ROWS=1
export STRATA_VERIFY_GRAPH=split
export STRATA_VERIFY_GRAPH_CHUNK=4
export STRATA_FUSED_GDN=1
```

### Common model flags

```bash
COMMON_ARGS=(
  --model /path/to/qwen3.8-flash-next-iq3_s.gguf
  --native
  --resident-experts
  --max-context 131072
  --kv int8
  --kv-resident 32768
  --prefill 4096
  --pool-workers 26
  --ple-io ram
  --pcie-frac 0.2
  --temp 0
  --max-new 512
)
```

### Arm A — MTP Baseline

```bash
./build-sycl/strata "${COMMON_ARGS[@]}" \
  --expert-cache 1000 \
  --mtp /path/to/mtp_dir \
  --spec 4 --spec-min-p 0.5 \
  --prompt-file /path/to/prompt.txt \
  2>&1 | tee ab_mtp.log
```

### Arm B — DFlash2 (Confidence Sweep)

On Intel Arc, `--draft-conf` defaults to `0.6`. Sweep `0.35`, `0.5`, `0.6`, and `0.7` because on a 12 GB hybrid card, a lower confidence threshold drafts wider windows (more potential accepted tokens per round, but more unique CPU-side expert misses per verify step):

```bash
for CONF in 0.35 0.5 0.6 0.7; do
  ./build-sycl/strata "${COMMON_ARGS[@]}" \
    --expert-cache 1000 \
    --mtp /path/to/mtp_dir \
    --draft dflash2 \
    --draft-model /path/to/qwen3.8-flash-next-dflash2.q4_0.pack \
    --spec 6 \
    --draft-conf "$CONF" \
    --prompt-file /path/to/prompt.txt \
    2>&1 | tee "ab_dflash2_c${CONF}.log"
done
```

### Arm B+ — DFlash2 with Reclaimed VRAM Slots

Once you pick the best `--draft-conf`, re-run with the ~192 MiB saved by DFlash2 converted into extra VRAM expert slots (e.g. `1000 -> 1095`, or use `--expert-cache auto` on both arms):

```bash
./build-sycl/strata "${COMMON_ARGS[@]}" \
  --expert-cache 1095 \
  --mtp /path/to/mtp_dir \
  --draft dflash2 \
  --draft-model /path/to/qwen3.8-flash-next-dflash2.q4_0.pack \
  --spec 6 \
  --draft-conf 0.6 \
  --prompt-file /path/to/prompt.txt \
  2>&1 | tee ab_dflash2_1095slots.log
```

---

## 4. What to Record & Decision Criteria

Extract these metrics from each log:

* **Decode `tok/s`** and **TPOT (ms/tok)**
* **Prefill / TTFT (ms)** (should be within noise; verify no regression)
* **Drafting overhead (`dflash2 ... ms/round drafting`)** vs. MTP drafting time
* **Mean accepted tokens / round** (confirm acceptance is healthy—if it is near `0`, check that `dflash2_head_test` passed and that `STRATA_Q6_PLANAR` did not fail to gather)
* **VRAM usage / free headroom** at startup (ensure `--vram-reserve-mib` is respected)
* **Exact output match** (greedy `--temp 0` should produce identical tokens between MTP and DFlash2 on the same prompt)

### Decision Rules

* **Keep DFlash2** if the best `--draft-conf` (with reclaimed `--expert-cache` slots) beats MTP on decode `tok/s` across your target prompts **and** you only need `--parallel 1` (single concurrent request).
* **Stay on MTP** if DFlash2 is slower or within noise on your workload, or if you need `--parallel > 1` batched serving.
