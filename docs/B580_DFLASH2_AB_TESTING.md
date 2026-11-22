# DFlash2 vs. MTP A/B Testing Guide (Xeon w7-3465X + Arc B580, IQ3_S)

This guide explains how to build, validate, and run a controlled A/B benchmark comparing **MTP** against **DFlash2** on **Qwen3.8-Flash-Next IQ3_S** using the **native SYCL port (`build_sycl/strata`)** on **Intel Arc B580 (12 GB)** + **Xeon w7-3465X**.

---

## 1. Build & Verify (Native SYCL `build_sycl/`)

DFlash2 is ported directly into the native SYCL tree (`src/kernels/sycl/dflash2.cpp`, `src/core/sycl/dflash2.cpp`, `src/program/sycl/generate.cpp`) with:

* Pure SYCL 2020 (no DPCT `.dp.cpp` files)
* Planar Q6_K draft-head fix (`#5286`, `Dflash2Drafter::gather_draft_head`)
* Full **`--q4_0-selector`** support (saves **~367 MiB VRAM** vs. MTP on 12 GB B580)

Build `strata` and the DFlash2 unit tests with `tools/build_sycl.sh`:

```bash
bash ./tools/build_sycl.sh -r strata dflash2_head_test dflash2_ring_test
```

Run the planar Q6_K draft-head tests first:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ctest --test-dir build_sycl -R "dflash2_head" --output-on-failure
# Expected:
#   dflash2_head_test        ... Passed (0 of 4500 logits differ)
#   dflash2_head_test_legacy ... Passed (WILL_FAIL: 4500 of 4500 differ)
```

---

## 2. Download & Pack the DFlash2 Drafter (~1.2 GB download -> 340 MB pack)

DFlash2 is **only a small 5-layer drafter** (not an 80+ GB model). It runs on top of your existing IQ3_S Qwen3.8-Flash-Next GGUF and **replaces** the MTP layer (only one drafter is loaded at a time).

| Drafter | Approx. VRAM | Delta vs. MTP (12 GB B580) |
| :--- | :---: | :--- |
| MTP layer | ~1,082 MiB | baseline |
| DFlash2 (`--q4_0`) | ~890 MiB | **-192 MiB** (~+98 IQ3_S expert cache slots) |
| DFlash2 (`--q4_0 --q4_0-selector`) | ~715 MiB | **-367 MiB** (~+188 IQ3_S expert cache slots) |

### Download the two pinned files

```bash
mkdir -p dflash2-src && cd dflash2-src

curl -fL --retry 3 -o config.json \
  "https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/resolve/a22a1744e85c17364dcc54a6a420af7135d4c7f6/config.json"

curl -fL --retry 3 -o model.safetensors \
  "https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/resolve/a22a1744e85c17364dcc54a6a420af7135d4c7f6/model.safetensors"
```

Verify downloads (optional):

```bash
sha256sum -c - <<'EOF'
2409575986f9ea24d455343eb469c0e6940def5edab1d1be503bb8eb4cc3b1b9  config.json
d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f  model.safetensors
EOF
```

### Pack with Q4_0 weights + Q4_0 selector codebooks (numpy only)

```bash
python3 /path/to/Strata/tools/dflash2_pack.py \
  model.safetensors config.json \
  qwen3.8-flash-next-dflash2.q4_0-selq4.pack \
  --q4_0 --q4_0-selector
```

Verify the packed file (339,694,592 bytes):

```bash
echo "4a42b98220ea7880664002c8edda1563aecbae754b762a2e2f31b2b91ee93ecd  qwen3.8-flash-next-dflash2.q4_0-selq4.pack" | sha256sum -c -
```

> **Note:** `--draft dflash2` still needs `--mtp /path/to/mtp_dir` (or `--mtp-draft-vocab /path/to/draft_vocab.bin`) so it can read `draft_vocab.bin`. The MTP layer weights are **not** loaded when `--draft dflash2` is set.

---

## 3. Controlled A/B Protocol

To keep comparisons fair on a hybrid Xeon + B580 setup:

1. **Pin identical model, KV, prefill, CPU pool, and PCIe settings** across both arms.
2. **Use greedy decoding** (`--temperature 0`) and identical pretokenized prompts (`--tokens-file`) + `--max-new` so you can verify exact token parity and compare acceptance rates and `tok/s`.
3. **Alternate arm order (`A -> B -> B -> A`)** across at least 3 rounds to cancel thermal/boost drift and page-cache effects.
4. **Test at least two prompt regimes:**
   * **Short/medium coding or structured output** (e.g., 512–2048 input tokens, 256–512 output tokens) — DFlash2 typically shines here.
   * **Long-context or reasoning/chat** (e.g., 4096 input tokens, 512+ output tokens) — checks whether wider draft windows trigger extra CPU expert misses.
5. **Run two B580 cache comparisons:**
   * **Fixed-cache A/B:** identical `--expert-cache 1000` on both arms (isolates pure drafter speed/acceptance).
   * **VRAM-equalized A/B:** bump `--expert-cache` on the DFlash2 arm by **+150 to +180 slots** (or use `--expert-cache auto` on both arms to automatically reclaim the ~367 MiB saved by DFlash2).

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
  --native /path/to/qwen3.8-flash-next-iq3_s-00001-of-0000N.gguf
  --resident-experts
  --max-context 131072
  --kv int8
  --kv-resident 32768
  --prefill 4096
  --pool-workers 26
  --ple-io ram
  --pcie-frac 0.2
  --temperature 0
  --max-new 512
)
```

*(Note: `build_sycl/strata` takes pretokenized IDs via `--tokens-file` or `--tokens`, not raw text. For example, `sycl/tools/data/prompt_deadlock_2744.tok` is a ready-to-use token file in this repo, or you can run through `serve/server.py` / `strata-b580-*.json`.)*

### Arm A — MTP Baseline

```bash
./build_sycl/strata "${COMMON_ARGS[@]}" \
  --expert-cache 1000 \
  --mtp /path/to/mtp_dir \
  --spec 4 --spec-min-p 0.5 \
  --tokens-file /path/to/prompt.tok \
  2>&1 | tee ab_mtp.log
```

### Arm B — DFlash2 (Confidence Sweep)

On Intel Arc, `--draft-conf` defaults to `0.6`. Sweep `0.35`, `0.5`, `0.6`, and `0.7` because on a 12 GB hybrid card, a lower confidence threshold drafts wider windows (more potential accepted tokens per round, but more unique CPU-side expert misses per verify step):

```bash
for CONF in 0.35 0.5 0.6 0.7; do
  ./build_sycl/strata "${COMMON_ARGS[@]}" \
    --expert-cache 1000 \
    --mtp /path/to/mtp_dir \
    --draft dflash2 \
    --draft-model /path/to/qwen3.8-flash-next-dflash2.q4_0-selq4.pack \
    --spec 6 \
    --draft-conf "$CONF" \
    --tokens-file /path/to/prompt.tok \
    2>&1 | tee "ab_dflash2_c${CONF}.log"
done
```

### Arm B+ — DFlash2 with Reclaimed VRAM Slots

Once you pick the best `--draft-conf`, re-run with the ~367 MiB saved by the Q4_0-selector DFlash2 pack converted into extra VRAM expert slots (e.g., `1000 -> 1180`, or use `--expert-cache auto` on both arms):

```bash
./build_sycl/strata "${COMMON_ARGS[@]}" \
  --expert-cache 1180 \
  --mtp /path/to/mtp_dir \
  --draft dflash2 \
  --draft-model /path/to/qwen3.8-flash-next-dflash2.q4_0-selq4.pack \
  --spec 6 \
  --draft-conf 0.6 \
  --tokens-file /path/to/prompt.tok \
  2>&1 | tee ab_dflash2_1180slots.log
```

---

## 4. What to Record & Decision Criteria

Extract these metrics from each log:

* **Decode `tok/s`** and **TPOT (ms/tok)**
* **Prefill / TTFT (ms)** (should be within noise; verify no regression)
* **Drafting overhead (`dflash2 ... ms/round drafting`)** vs. MTP drafting time
* **Mean accepted tokens / round** (confirm acceptance is healthy—if it is near `0`, check that `dflash2_head_test` passed and `strata dflash2: ... selector codebooks: Q4_0 rows` printed at startup)
* **VRAM usage / free headroom** at startup (ensure `--vram-reserve-mib` is respected)
* **Exact output match** (greedy `--temperature 0` should produce identical tokens between MTP and DFlash2 on the same prompt)

### Decision Rules

* **Keep DFlash2** if the best `--draft-conf` (with reclaimed `--expert-cache` slots) beats MTP on decode `tok/s` across your target prompts **and** you only need `--parallel 1` (single concurrent request).
* **Stay on MTP** if DFlash2 is slower or within noise on your workload, or if you need `--parallel > 1` batched serving.
