# Intel Arc B580 (12 GB) + 64 GB+ RAM: Model Recommendation & Setup Guide

This guide provides model recommendations, download instructions (including downloading on another machine), and execution steps for the **Intel Arc B580 (12 GB VRAM)** on a system with **64 GB or more of system RAM**.

---

## 1. Model Recommendation

### Verdict: **IQ2_XS** (Recommended) or **Q2_0** (Fastest Decode)

With **more than 64 GB of system RAM**, you should **not** use the Coder (IQ1_M) unless you only care about code.

| Criterion | **IQ2_XS** (Recommended) | **Q2_0** | **Coder (IQ1_M)** |
|---|---|---|---|
| **Experts Retained** | **All 512 experts** | **All 512 experts** | 256 experts (pruned for code only) |
| **Model Quality** | Higher precision, strong across reasoning, math, multilingual, and code | Good baseline quality, full domain knowledge | Strong in code/agentic tasks, weaker outside English and code |
| **RAM Requirement** | ~35.5 GB for experts (**fits 64 GB+ RAM**) | ~34.0 GB for experts (**fits 64 GB+ RAM**) | ~23.0 GB for experts (designed for 32 GB RAM PCs) |
| **Decode Speed (12 GB GPU)** | **79 tok/s** short chat, **63 tok/s** 128K context | **94 tok/s** short chat, **76 tok/s** 128K context | 55 tok/s short chat, 43 tok/s 128K context |
| **Prompt Reading** | 2,090 tok/s | 2,650 tok/s | 2,180 tok/s |

*(Performance numbers measured on a 12 GB GPU with 64 GB RAM from [docs/MODELS.md](MODELS.md)).*

**Why IQ2_XS is best for your setup:**
1. **You have the RAM**: The Coder was pruned to fit budget 32 GB PCs. Your machine has >64 GB RAM, which easily accommodates IQ2_XS's 35.5 GB expert footprint with >25 GB of headroom remaining for Linux, OS caches, and KV context.
2. **Superior Decode Speed**: Counter-intuitively, on 12 GB GPUs, IQ2_XS decodes faster (79 tok/s) than the Coder (55 tok/s) because the Coder stores its 256 remaining experts in heavier IQ3_S quantization format.
3. **Hardware Balance on B580**: The B580's 12 GB VRAM holds the dense layers and an active VRAM cache tier of ~2,500–3,800 hot experts (`--expert-cache auto`), while your 64 GB+ RAM hosts the rest in pinned memory (`--resident-experts`), yielding optimal throughput.

---

## 2. Model Download Steps

The model consists of two GGUF shards:
* **Shard 1 (`...-00001-of-00002.gguf`, ~39.2 GB)**: Contains attention, dense layers, and MoE experts.
* **Shard 2 (`...-00002-of-00002.gguf`, ~28.8 GB)**: Contains the Layer 1 n-gram PLE lookup table (`per_layer_token_embd.weight`).

HuggingFace Repository: [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)

### Option A: Download on Another Machine (Machine A) and Transfer

#### 1. Download the files on Machine A
Using `huggingface-cli`:
```bash
huggingface-cli download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF \
  --include "IQ2_XS/*" \
  --local-dir ./strata_iq2xs_shards
```

Or using `wget`:
```bash
mkdir -p strata_iq2xs_shards && cd strata_iq2xs_shards
HF_BASE="https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/IQ2_XS"

wget -c "${HF_BASE}/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf"
wget -c "${HF_BASE}/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf"
cd ..
```

#### 2. Copy the files to the target B580 machine
Transfer the directory over your local network:
```bash
rsync -avP ./strata_iq2xs_shards/ user@target-machine:/data/models/IQ2_XS/
```

---

### Option B: Download Directly on Target Machine via `setup.py`

Run Strata's installer in non-interactive download mode:
```bash
python3 setup.py --model IQ2_XS --no-start --yes
```
This automatically downloads both shards into `../Strata-data/models/IQ2_XS/`, builds the native pack, and prepares the speculative decoding MTP layer.

---

## 3. Building the Engine on the Target B580 Machine

Ensure oneAPI 2025.3 / 2026.1 and compute runtimes are installed, then compile the native SYCL engine:

```bash
# 1. Source Intel oneAPI environment
source /opt/intel/oneapi/setvars.sh --force 2>/dev/null
export SYCL_CACHE_PERSISTENT=0
export ONEAPI_DEVICE_SELECTOR=level_zero:0

# 2. Configure with Xe2 optimizations enabled
cmake -B build_sycl -G Ninja \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_SYCL=ON \
  -DSTRATA_TARGET_XE2=ON

# 3. Build the strata engine and verification tools
cmake --build build_sycl --target strata strata-device -j$(nproc)

# 4. Verify GPU detection (reports Intel Arc B580, Xe2 Architecture, ~12 GB VRAM)
./build_sycl/strata-device
```

---

## 4. Preparing the Pack (If Transferring Shards Manually)

If you copied raw GGUF files without running `setup.py`, build the pack index locally (takes ~10 seconds on CPU):

```bash
python3 tools/iq_pack.py \
  --gguf /data/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
  --out /data/models/IQ2_XS/pack
```

---

## 5. Starting Strata on Arc B580

### Key Settings for Arc B580 (12 GB VRAM) + 64 GB+ RAM:
* `--resident-experts`: Pins the cold experts in your 64 GB+ RAM arena.
* `--expert-cache auto`: Automatically fills free VRAM on the B580 with hot experts.
* `--vram-reserve-mib 1536`: Retains a 1.5 GB safety buffer to prevent Intel `xe` driver out-of-memory livelocks.
* `--kv q8_0`: Quantizes the attention KV cache to 8-bit to save VRAM for longer contexts.
* `SYCL_CACHE_PERSISTENT=0`: Prevents Xe2 JIT persistent cache crashes.
* **Do NOT use `STRATA_VERIFY_NO_HOST=1`**: The B580 cannot fit all experts in VRAM, so host-device synchronization is required.

---

### Step 5A: Single-Prompt CLI Inference Test

```bash
source /opt/intel/oneapi/setvars.sh --force 2>/dev/null
export SYCL_CACHE_PERSISTENT=0

./build_sycl/strata \
  --pack /data/models/IQ2_XS/pack \
  --native /data/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
  --ple-gguf /data/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf \
  --expert-profile data/expert-profile.bin \
  --expert-cache auto \
  --resident-experts \
  --vram-reserve-mib 1536 \
  --max-context 32768 \
  --kv q8_0 \
  --prefill auto \
  --tokens 128 \
  --max-new 128
```

---

### Step 5B: Running the Resident OpenAI / Anthropic Server & Web App

1. Create a server configuration file `strata-b580-iq2xs.json`:

```json
{
  "backend": "sycl",
  "exe": "build_sycl/strata",
  "args": [
    "--pack", "/data/models/IQ2_XS/pack",
    "--native", "/data/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf",
    "--ple-gguf", "/data/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf",
    "--expert-profile", "data/expert-profile.bin",
    "--expert-cache", "auto",
    "--resident-experts",
    "--vram-reserve-mib", "1536",
    "--max-context", "32768",
    "--kv", "q8_0",
    "--prefill", "auto"
  ]
}
```

2. Start the server (includes Xe GPU telemetry for Arc GPUs):

```bash
source /opt/intel/oneapi/setvars.sh --force 2>/dev/null
export SYCL_CACHE_PERSISTENT=0

python3 sycl/serve/server_intel.py --engine strata --config strata-b580-iq2xs.json --port 8080
```

The Web UI will be accessible at `http://localhost:8080`, and OpenAI/Anthropic API endpoints (`/v1/chat/completions`, `/v1/messages`) are active immediately.

---

## 6. Coding-Optimized Setup: Qwen3.8-Flash-Next Coder (for OpenCode)

If your primary workflow is software engineering, codebase indexing, and multi-file agentic coding via **OpenCode** (or other coding agents such as Aider, Cline, or Continue), you can run the dedicated **Coder (IQ1_M)** variant.

### 6.1 Why Coder for OpenCode?
* **Tuned for Software Engineering**: Fine-tuned by ISTA-DASLab specifically on code datasets, tool-calling benchmarks, and software engineering agent tasks (91% SWE-bench Verified score, 99% LiveCodeBench).
* **Pruned Expert Footprint**: Retains 256 code-specialized experts (dropping the 256 non-code experts). Shard 1 is only **29.6 GB** (23.0 GB of experts in RAM), allowing fast memory transfers and leaving maximum headroom in your 64 GB+ RAM system.
* **Fast Prompt Ingestion**: Reads prompts at **2,180 tok/s** (scaling to ~1,420 tok/s on large prompts), ensuring low time-to-first-token when OpenCode sends large repository maps or multi-file context.

---

### 6.2 Context Length & Memory Scaling on B580 (12 GB VRAM)

Coding agents require long context windows (multi-file context, lint outputs, test failures, git diffs, conversation history). Strata's hybrid architecture makes long contexts exceptionally efficient on 12 GB GPUs:

1. **Hybrid Architecture Sizing**:
   Only 12 of the model's 48 layers are full attention (Query-Shared Attention, QSA); the remaining 36 layers are Gated-Delta-Net (GDN) linear recurrent layers. The GDN state has a fixed footprint that does **not** grow with context length.
2. **KV Footprint**:
   With 8-bit quantized KV (`--kv q8_0`), the KV cache consumes only **~20 KiB per token** (~0.6 GB per 32,768 tokens in VRAM):
   * **32,768 (32K) Context**: ~0.6 GB VRAM KV cache. (Standard single-file / short coding sessions)
   * **65,536 (64K) Context**: ~1.2 GB VRAM KV cache. (Recommended baseline for OpenCode)
   * **131,072 (128K) Context**: ~2.4 GB VRAM KV cache. (Deep repository multi-file context)
   * **262,144 (256K) Context**: ~4.8 GB VRAM KV cache, or using **KV streaming** (`--kv-resident 32768`) where the full 256K KV cache lives in your 64 GB+ system RAM (~3.5 GB RAM) and only the active 32K window resides in VRAM.

3. **Recommended Settings for Long-Context Coding**:
   * For contexts > 32K, set `--prefill 4096` to keep prompt batch buffers compact.
   * Increase VRAM reserve to `--vram-reserve-mib 2048` for 64K+ contexts.
   * Always retain `--resident-experts` and `--expert-cache auto`.

---

### 6.3 Download Steps for Coder (IQ1_M)

The Coder variant consists of two shards:
* **Shard 1 (`...-Coder-IQ1_M-00001-of-00002.gguf`, ~29.6 GB)**: Dense layers and 256 coding experts.
* **Shard 2 (`...-Coder-IQ1_M-00002-of-00002.gguf`, ~28.8 GB)**: Layer 1 n-gram embedding table (`per_layer_token_embd.weight`).

HuggingFace Repository: [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)

#### Option A: Download on Another Machine (Machine A) & Transfer
```bash
# On Machine A (download machine) using huggingface-cli:
huggingface-cli download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF \
  --include "IQ1_M/*" \
  --local-dir ./strata_coder_shards

# Or using wget:
mkdir -p strata_coder_shards && cd strata_coder_shards
HF_BASE="https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/resolve/main/IQ1_M"
wget -c "${HF_BASE}/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-00001-of-00002.gguf"
wget -c "${HF_BASE}/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-00002-of-00002.gguf"
cd ..

# Transfer to target B580 machine:
rsync -avP ./strata_coder_shards/ user@target-machine:/data/models/Coder/
```

#### Option B: Download on Target Machine via `setup.py`
```bash
python3 setup.py --family coder --model Coder --no-start --yes
```

---

### 6.4 Preparing the Coder Pack Index

If transferred manually, build the Coder native pack index (takes ~5–10 seconds on CPU):
```bash
python3 tools/iq_pack.py \
  --gguf /data/models/Coder/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-00001-of-00002.gguf \
  --out /data/models/Coder/pack
```

---

### 6.5 Server Configurations for OpenCode (64K, 128K, and 256K Context)

Depending on your codebase size and context requirements, choose the matching configuration below:

#### Configuration 1: 64K Context (`strata-b580-coder-64k.json`)
Recommended baseline for general multi-file editing and agentic workflows. Uses ~1.2 GB VRAM for the KV cache.

```json
{
  "backend": "sycl",
  "exe": "build_sycl/strata",
  "args": [
    "--pack", "/data/models/Coder/pack",
    "--native", "/data/models/Coder/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-00001-of-00002.gguf",
    "--ple-gguf", "/data/models/Coder/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-00002-of-00002.gguf",
    "--expert-profile", "data/expert-profile-coder.bin",
    "--expert-cache", "auto",
    "--resident-experts",
    "--vram-reserve-mib", "2048",
    "--max-context", "65536",
    "--kv", "q8_0",
    "--prefill", "4096"
  ]
}
```

#### Configuration 2: 128K Context (`strata-b580-coder-128k.json`)
Ideal for large codebases, full repository AST maps, and long debugging/test histories. The 128K KV cache consumes ~2.4 GB VRAM directly on the B580:

```json
{
  "backend": "sycl",
  "exe": "build_sycl/strata",
  "args": [
    "--pack", "/data/models/Coder/pack",
    "--native", "/data/models/Coder/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-00001-of-00002.gguf",
    "--ple-gguf", "/data/models/Coder/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-00002-of-00002.gguf",
    "--expert-profile", "data/expert-profile-coder.bin",
    "--expert-cache", "auto",
    "--resident-experts",
    "--vram-reserve-mib", "2048",
    "--max-context", "131072",
    "--kv", "q8_0",
    "--prefill", "4096"
  ]
}
```

#### Configuration 3: 256K Context with KV Streaming (`strata-b580-coder-256k.json`)
For full repository coverage up to the model's native 262,144-token maximum.
Because keeping a full 256K KV cache in VRAM would take ~4.8 GB and reduce the hot-expert cache room, this configuration uses **KV streaming** (`--kv-resident 32768`):
* The active 32K attention window resides in the B580's VRAM (~0.6 GB).
* The full 256K context KV cache lives in pinned system RAM (~3.5 GB of your 64 GB+ RAM).
* Maximum VRAM remains dedicated to the MoE expert cache tier.

```json
{
  "backend": "sycl",
  "exe": "build_sycl/strata",
  "args": [
    "--pack", "/data/models/Coder/pack",
    "--native", "/data/models/Coder/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-00001-of-00002.gguf",
    "--ple-gguf", "/data/models/Coder/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-00002-of-00002.gguf",
    "--expert-profile", "data/expert-profile-coder.bin",
    "--expert-cache", "auto",
    "--resident-experts",
    "--vram-reserve-mib", "2048",
    "--max-context", "262144",
    "--kv", "q8_0",
    "--kv-resident", "32768",
    "--prefill", "4096"
  ]
}
```

---

#### Starting the Server

Launch the resident server with your chosen configuration (replace config file name as desired):

```bash
source /opt/intel/oneapi/setvars.sh --force 2>/dev/null
export SYCL_CACHE_PERSISTENT=0

# Example launching 128K context:
python3 sycl/serve/server_intel.py --engine strata --config strata-b580-coder-128k.json --port 8080

# Or launching 256K context with KV streaming:
# python3 sycl/serve/server_intel.py --engine strata --config strata-b580-coder-256k.json --port 8080
```

---

### 6.6 Connecting OpenCode to Strata

OpenCode connects using Strata's standard OpenAI-compatible API endpoint:

* **API Base URL**: `http://127.0.0.1:8080/v1`
* **API Key**: Any string (e.g. `dummy` or `strata`)
* **Model Name**: `qwen3.8-flash-next-coder` (or `default`)
* **Context Window**: Match your server configuration (`65536`, `131072`, or `262144`)
* **Max Tokens**: Set to at least `8192` (or `16384`)

#### Handling Thinking / Reasoning in OpenCode:
By default, the Coder model reasons inside `<think>...</think>` tags before generating code or tool invocations:
* **Option 1 (Deep reasoning with high token limit)**: Keep thinking enabled in OpenCode, but ensure `max_tokens` is set high (e.g., 8192+) so the model does not run out of generation tokens before writing the actual code.
* **Option 2 (Disable thinking for immediate code generation)**: Create a file named `strata-b580-coder.shared-settings.json` next to your config file with:
  ```json
  {
    "reasoning_effort": "none"
  }
  ```
  Or configure OpenCode to pass `chat_template_kwargs: {"enable_thinking": false}` in completion requests.
