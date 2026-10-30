# Intel Arc B580 (Xe2) Prefill Optimization Tasks

Tracking document for engineering tasks to optimize Strata's prefill throughput on the **Intel Arc B580 (12 GB VRAM, Xe2-HPG)** and close the performance gap with the CUDA backend.

---

## Task Matrix & Status Overview

| ID | Task | Impact Area | Priority | Status |
|:---|:---|:---|:---|:---|
| **TASK-01** | [Chunked USM Host Pinned Memory Allocation](#task-01-chunked-usm-host-pinned-memory-allocation) | PCIe DMA / Pinned RAM | **P0 (Highest)** | `TODO` |
| **TASK-02** | [Enable GPU MoE Grouping (Remove CUDA Compute Capability Check)](#task-02-enable-gpu-moe-grouping-remove-cuda-check) | CPU Stalls / Latency | **P0 (Highest)** | `TODO` |
| **TASK-03** | [Layer-Major Prefill Status & Multi-Super-Chunk Limitations](#task-03-layer-major-prefill-status--multi-super-chunk-limitations) | PCIe Weight Bandwidth | Deprioritized | `KNOWN LIMITATION` |
| **TASK-04** | [Enforce Ahead-of-Time (AOT) BMG Compilation in Build Script](#task-04-enforce-ahead-of-time-aot-bmg-compilation) | JIT Overhead (5–15s pause) | P2 (Medium) | `TODO` |
| **TASK-05** | [Port Prompt Attention to Native Xe2 SIMD16 XMX (DPAS)](#task-05-port-prompt-attention-to-native-xe2-xmx-dpas) | Compute / Tensor Acceleration | P2 (Medium) | `TODO` |
| **TASK-06** | [Bypass Prompt Tail Window Non-Finite Logits (Issue #3465)](#task-06-bypass-prompt-tail-window-non-finite-logits-issue-3465) | Numerical Stability / NaNs | P1 (High) | `TODO` |
| **TASK-07+**| [User Backlog & Custom Tasks](#user-backlog--custom-tasks) | Custom | — | `OPEN` |

---

## Key Architectural Findings (Summary from Profiling & Debugging)

1. **Layer-Major Prefill on Single GPU is Unstable for $N > \text{super-chunk}$**:
   - Layer-Major prefill (`STRATA_PREFILL_LAYER_MAJOR=1`) was authored for multi-GPU layer splits where each GPU handles a fixed subset of layers in a pipeline (`stages >= 2`).
   - On a single GPU with prompts longer than the super-chunk size (e.g., $N = 4,652$ or $9,000$ with super-chunk $4,096$), the outer loop loops through all 48 layers for Chunk 0, then restarts at Layer 0 for Chunk 1.
   - In SYCL, this super-chunk boundary breaks state synchronization: the GDN DeltaNet recurrent state, PLE n-gram table (`ss.ple.hist`), and QSA causal indexer experience numerical discontinuity, resulting in **`non-finite logits error (#879)` (NaN/Inf)**.
   - When using `STRATA_PREFILL_LM_R="host"`, Level Zero deadlocks on cross-queue barriers between `m.copy` and `m.cs` whenever prompt length is not an exact multiple of the chunk size.
   - **Conclusion**: The production path forward on a single GPU is **Token-Major prefill (`STRATA_PREFILL_LAYER_MAJOR=0`)**, which processes tokens sequentially left-to-right with 100% numerical stability. Throughput is recovered by fixing **TASK-01** and **TASK-02**.

2. **Why Baseline Token-Major Prefill is Currently ~30 tok/s on Arc B580**:
   - **PCIe Fallback to Pageable Memory**: Level Zero driver rejects a single 51.5 GB pinned allocation (`sycl::malloc_host`), forcing pageable bounce buffers.
   - **Host CPU Synchronization Stalls**: `src/prefill/sycl/moe_fused.cpp` falsely rejects Intel Arc GPUs (`cc >= 80` check), causing 48 `cs->wait()` CPU roundtrips per chunk to sort tokens on the host CPU.

---

## Detailed Task Breakdown

### TASK-01: Chunked USM Host Pinned Memory Allocation
* **Problem**: `FileExpertSource::pin_cache_complement` attempts a single contiguous `sycl::malloc_host(51.5 GB)`. On Linux systems with 64–128 GB RAM, the Level Zero driver and Linux kernel reject this monolithic locked allocation, forcing a silent fallback to pageable memory. This disables hardware PCIe DMA and forces synchronous CPU bounce buffering, severely choking throughput on the PCIe 4.0 x8 bus.
* **Objective**: Segment the host-pinned allocation into 1–2 GB chunks (or per-layer arrays) so `sycl::malloc_host` succeeds reliably.
* **Files Affected**:
  - [`src/core/sycl/expert_source.cpp`](src/core/sycl/expert_source.cpp#L1280-L1305)
  - [`include/strata/core/expert_source.hpp`](include/strata/core/expert_source.hpp)
* **Action Items**:
  - [ ] Replace single `void* arena` with a vector of chunked USM host allocations (`std::vector<void*> chunks_`).
  - [ ] Implement offset mapping helper to resolve expert offsets to `(chunk_idx, chunk_offset)`.
  - [ ] Update cleanup lambda `release()` to iterate and free each chunk with `sycl::free(ptr, default_queue())`.
  - [ ] Verify startup logs confirm `resident RAM mode: ... GiB of experts in RAM (page-locked)` instead of `page-locking refused` or `pageable`.
* **Verification**:
  - Launch engine and inspect startup logs for `page-locked and mapped (USM host)`.
  - Confirm PCIe DMA transfer throughput reaches 12–14 GB/s line rate.

---

### TASK-02: Enable GPU MoE Grouping (Remove CUDA Check)
* **Problem**: In [`src/prefill/sycl/moe_fused.cpp`](src/prefill/sycl/moe_fused.cpp#L506-L553), `available()` checks `dev_info().cc >= 80`. Because Level Zero reports driver version (e.g., `1.3`), `cc` evaluates to `13` instead of an NVIDIA compute capability. This wrongly classifies the Arc B580 as "unsupported", forcing `src/prefill/sycl/prefill.cpp` into a CPU fallback that calls `m.cs->wait()` and sorts tokens on the host CPU 48 times per prompt chunk.
* **Objective**: Decouple the device capability check on SYCL so Intel Xe2 GPUs execute the on-GPU grouping kernel (`count_kernel`, `scan_kernel`, `place_kernel`).
* **Files Affected**:
  - [`src/prefill/sycl/moe_fused.cpp`](src/prefill/sycl/moe_fused.cpp#L502-L570)
  - [`src/prefill/sycl/prefill.cpp`](src/prefill/sycl/prefill.cpp#L2440-L2450)
* **Action Items**:
  - [ ] Replace `d.cc >= 80` in `available()` with SYCL device check (`dev.is_gpu()` and Intel Xe generation check `intel_gpu_gen(dev) >= IntelGpuGen::Alchemist`).
  - [ ] Ensure `count_kernel`, `scan_kernel`, and `place_kernel` compile cleanly under native SIMD16 / SIMD32 workgroups without undefined symbols.
  - [ ] Wire the on-GPU grouping output directly into the MMQ prefill launch pipeline to remove the 48 `m.cs->memcpy` and `m.cs->wait()` CPU roundtrips.
* **Verification**:
  - Run `ctest -R sycl_prefill_test` or `sycl_lm_kernels_parity`.
  - Benchmark prefill TTFT with `STRATA_TRACE=1` to ensure `kPfHostGroup` and host synchronization barriers are removed.

---

### TASK-03: Layer-Major Prefill Status & Multi-Super-Chunk Limitations
* **Status**: `KNOWN LIMITATION / DEPRIORITIZED FOR 1 GPU`
* **Findings**:
  - Layer-Major prefill works on prompts smaller than the super-chunk size with `STRATA_PREFILL_LM_R=vram`.
  - When $N > \text{super-chunk}$, multiple super-chunks corrupt GDN and QSA recurrent/indexer state, generating NaNs (`non-finite logits error (#879)`).
  - When using `STRATA_PREFILL_LM_R=host`, cross-queue barrier dependencies deadlock Level Zero on non-multiple prompt lengths.
* **Guidance**: Use Token-Major prefill on single Intel GPUs:
  ```bash
  unset STRATA_PREFILL_LAYER_MAJOR
  unset STRATA_PREFILL_LM_TOKENS
  unset STRATA_PREFILL_LM_R
  ```

---

### TASK-04: Enforce Ahead-of-Time (AOT) BMG Compilation
* **Problem**: With `SYCL_CACHE_PERSISTENT=0`, the Level Zero compiler must JIT-compile kernels in RAM at runtime, causing a 5–15 second pause during the first request.
* **Objective**: Ensure the build configuration compiles directly to Battlemage machine code (`bmg` ISA) Ahead-of-Time.
* **Files Affected**:
  - [`tools/build_sycl.sh`](tools/build_sycl.sh#L45-L55)
  - [`CMakeLists.txt`](CMakeLists.txt#L96-L105)
* **Action Items**:
  - [ ] Verify `tools/build_sycl.sh` passes `-DSTRATA_AOT_BMG=ON` when targeting B580.
  - [ ] Verify `-Xs "-device bmg"` generates native binary sections in `build_sycl/strata`.
  - [ ] Verify first-prompt execution latency starts immediately without runtime driver compilation delay.
* **Verification**:
  - Check binary format: `ocloc disasm -file ...` or inspect startup logs for zero JIT compilation pause.

---

### TASK-05: Port Prompt Attention to Native Xe2 XMX (DPAS)
* **Problem**: Prompt attention in SYCL falls back to pre-Ampere generic vector loops because `qsa_prompt_attn` PTX instructions (`cp.async`, `ldmatrix`, `mma.sync`) were not ported to Xe2 DPAS instructions.
* **Objective**: Implement a high-throughput Xe2 prompt attention kernel using `sycl::ext::oneapi::experimental::matrix` (`joint_matrix_mad`) with SIMD16 $16 \times 16 \times 16$ tile dimensions.
* **Files Affected**:
  - [`sycl/src/kernels/cuda/qsa_prompt_attn_xmx.dp.cpp`](sycl/src/kernels/cuda/qsa_prompt_attn_xmx.dp.cpp)
  - [`src/kernels/sycl/qsa_prompt_attn.cpp`](src/kernels/sycl/qsa_prompt_attn.cpp)
* **Action Items**:
  - [ ] Refactor tile configurations in `qsa_prompt_attn_xmx.dp.cpp` from DG2 (subgroup 8, $8 \times 8 \times 16$) to Xe2 (subgroup 16, $16 \times 16 \times 16$).
  - [ ] Replace global memory loads with 2D block loads / prefetch into SLM.
  - [ ] Benchmark against baseline fallback to ensure throughput improvement.
* **Verification**:
  - Run standalone matrix test: `./build_sycl/sycl_xmx_joint_matrix`.

---

### TASK-06: Bypass Prompt Tail Window Non-Finite Logits (Issue #3465)
* **Problem**: By default, `--short-read 64` directs the prompt tail (tokens after the last turn marker) through the speculative verifier windows (`read_windows`). On offloaded models, evaluating this unaligned tail window can trigger non-finite logits (#879).
* **Objective**: Port upstream commit `894f133b` to automatically read prompt tails through the batched prefill path on failure, or default to `--short-read 0`.
* **Files Affected**:
  - [`src/program/sycl/generate.cpp`](src/program/sycl/generate.cpp#L6290-L6340)
* **Action Items**:
  - [ ] In `src/program/sycl/generate.cpp`, add the `nf_prompt_windows` and `short_read_off_once` auto-recovery mechanism from `894f133b`.
  - [ ] Advise running with `--short-read 0` as a command-line safeguard.
* **Verification**:
  - Verify that prompt reading completes and generation produces tokens without encountering `non-finite logits (#879)`.

---

## User Backlog & Custom Tasks
*(Add your custom tasks, experiments, and notes below)*

- [ ] **TASK-07**:
- [ ] **TASK-08**:
- [ ] **TASK-09**:
