# VTune Kernel Profiling, Register Spill Inspection & Optimization Roadmap

**Date**: 2026-10-31  
**Target Architecture**: Intel Arc Battlemage B580 / Lunar Lake Xe2 (`[0x64a0]`)  
**Status**: Active Engineering & Tracking Document (Session Handover Ready)

---

## Executive Summary & Task Dashboard

This document tracks all identified kernel bottlenecks, register spills, compiler properties, and fusion opportunities on Intel Arc B580 / Xe2. Use this dashboard to resume engineering work in subsequent sessions.

### Status Tracker

| Status | Component / Opportunity | Root Cause / Description | Verification / Action |
|---|---|---|---|
| **`[DONE]`** | **GrMulti Down Projection** | 2,432 B spill, 45.8% stall in hyper-connection down projection | Specialized $T=1..4$ unrolled paths; single-token latency dropped from 240 µs to 188 µs (`11b19222`). |
| **`[DONE]`** | **Architecture Detection Fix** | `mmvq_large_grf` ignored B580, forcing 128 GRF default | Added `IntelGpuGen::Battlemage` check; unblocked 256-GRF paths across all MMVQ kernels (`8d42ca55`). |
| **`[DONE]`** | **NativeExpertLayout (MoE Decode)** | Up to 16,896 B spills in `expdec_gu_r` and `expdec_down_r` | Implemented `ExpdecGuGrf256Functor` and `ExpdecDownGrf256Functor` (256 GRF mode) (`8d42ca55`). |
| **`[DONE]`** | **Q6KTraits (MMVQ Q6_K)** | 17.67s time (6.08%), 57.5% pipeline stalls, 12.3 KB spills | Added `B6RowsGrf256Functor` (256 GRF); 8-col dropped from 3.08 ms to 2.25 ms with bit-exact parity (`8d42ca55`). |
| **`[DONE]`** | **Q5KTraits (MMVQ Q5_K)** | 5,504 B spill under wide multi-token columns | Added `NativeMmvqMultiGrf256Functor` (256 GRF); 8-col head dropped from 37.01 ms to 32.57 ms (`9af85016`). |
| **`[DONE]`** | **bf16_gemv_fp32_mmvf** | 32 accumulators $\times$ SIMD32 = 64 GRFs alone; heavy spills | Added `Bf16MmvfRowsGrf256Functor` & `Bf16MmvfMultiGrf256Functor` (256 GRF) (`9af85016`). |
| **`[DONE]`** | **SmallTraits (Q4_0, Q5_0, Q8_0, IQ4_NL)** | Multi-column register pressure in `native_mmvq_multi_kernel` | Routed through `NativeMmvqMultiGrf256Functor` (256 GRF); bit-exact parity confirmed (`9af85016`). |
| **`[DONE]`** | **Offline IGC Dump Infrastructure** | Need to check spills without running full Strata server | Created automated IGC dump scanner script parsing `.zeinfo` for spills, GRF, and EU thread counts. |
| **`[DONE]`** | **Failure Recovery & Troubleshooting Guide** | DPC++ functor syntax, resource limits, fallback flags | Documented all failure modes, DPC++ functor rules, and environment switches in Section 3. |
| **`[DONE]`** | **Command Graph Replay vs. Fusion Analysis** | Understand small kernel counts (~300–500/token) and fusion | Documented that SYCL Command Graphs eliminate launch overhead (1 dispatch/token); analyzed fusion trade-offs. |
| **`[TODO]`** | **`s2_expert_grouped.cpp` 1 KB Spill Elimination** | Confirmed 1,024 B spill in `grouped_s2_impl` under 128 GRFs | **Priority 1 for next session**: Implement `GroupedS2Grf256Functor` (256 GRF) in `s2_expert_grouped.cpp`. |
| **`[TODO]`** | **Extend L3 Streaming Cache Hints** | `cache_mode::streaming` exists only in `bf16_gemv.cpp` | **Priority 2 for next session**: Plumb streaming hints to MMVQ and MoE weight loads to protect 16 MB L3 cache. |
| **`[TODO]`** | **GDN Epilogue Quantization Fusion** | `fused_gdn_step_norm` writes FP32 $b.y$, then calls `quantize_act` | Add direct Q8_1/Q8_0 epilogue output inside `fused_gdn_step_norm` to save ~20 KB DRAM roundtrip per GDN layer. |
| **`[TODO]`** | **Fused MoE Hit Accumulate + Residual Add** | `moe_hit_add` and `gr_write` run as separate back-to-back adds | Combine into a single vector FMA pass: $\text{res} += \text{gpu\_hits} + \text{cpu\_misses}$. |
| **`[TODO]`** | **End-to-End VTune Re-Profile on B580** | Validate global wall-clock speedup and stall reduction | Run full `vtune -collect gpu-hotspots` on decode pass; verify EU active % increase and zero spills. |
| **`[REJECTED]`** | **Force 256 GRF on `ncols == 1` Decode MMVQ** | Proposal to force 256 GRF on single-token decode | **Rejected**: IGC dumps prove 0 B spills in 128 GRF; forcing 256 GRF halves EU thread occupancy (8 $\rightarrow$ 4) for 0 gain. |
| **`[REJECTED]`** | **Regenerate `sycl/src/kernels/cuda/*.dp.cpp`** | Claimed stale mirror files exist | **Rejected**: Path does not exist; pure SYCL kernels are compiled directly from `src/kernels/sycl/*.cpp`. |
| **`[REJECTED]`** | **SIMD16 + 256 GRF for Mainline Code** | Proposal to switch kernels to SIMD16 | **Rejected for Mainline**: 16-lane tree reductions break bit-exact numerical parity against CUDA/single-column baselines. |

---

## 1. How to Check Kernels, Spills, and Occupancy Offline (Without Running Full Strata)

You can inspect all kernel performance metrics—including register spill sizes, GRF allocation, SIMD execution width, and thread occupancy—**offline** without launching the full Strata server or running full inference sessions.

### Method: Intel Graphics Compiler (IGC) Dump

Intel's oneAPI Level Zero driver / OpenCL runtime invokes the Intel Graphics Compiler (IGC). By setting the runtime environment variables:

```bash
source /opt/intel/oneapi/setvars.sh
export IGC_ShaderDumpEnable=1
export IGC_DumpToCustomDir=/tmp/sycl_igc_dump
mkdir -p /tmp/sycl_igc_dump
```

When running any test binary or targeted kernel microbenchmark in `build_sycl/` (such as `./build_sycl/b6_mmvq_bench`, `./build_sycl/fused_gr_bench`, or `./build_sycl/s2_expert_grouped_parity`), IGC automatically emits compiler artifacts for every compiled device kernel into the target directory:

- `*.zeinfo`: YAML file detailing:
  - `grf_count`: Assigned register file size (128 or 256 GRFs).
  - `simd_size`: Executed sub-group width (SIMD16 or SIMD32).
  - `spill_size`: **Register spill size in bytes** to private scratch memory.
  - `eu_thread_count`: Number of hardware threads per Execution Unit (EU occupancy).
  - `per_thread_memory_buffers`: Allocated scratch memory space per thread.
- `*.asm` / `*.visaasm`: Intel Gen ISA assembly instructions, showing `send` messages, latency cycles, and scratch memory read/write traffic.

### Quick Offline Scanner Script

Run this Python one-liner to parse and list all compiled kernels, spill sizes, and GRF allocations:

```bash
python3 -c "
import glob, re, subprocess
for fn in glob.glob('/tmp/sycl_igc_dump/*.zeinfo'):
    with open(fn) as f: content = f.read()
    for entry in content.split('  - name: ')[1:]:
        name = entry.split('\n')[0].strip()
        spill_m = re.search(r'spill_size:\s*(\d+)', entry)
        if spill_m:
            spill = int(spill_m.group(1))
            grf = re.search(r'grf_count:\s*(\d+)', entry).group(1)
            simd = re.search(r'simd_size:\s*(\d+)', entry).group(1)
            demangled = subprocess.check_output(['c++filt', name]).decode().strip() if name else 'unknown'
            status = 'SPILL' if spill > 0 else 'CLEAN'
            print(f'[{status}] {spill:6d} B | SIMD{simd} | GRF {grf} | {demangled}')
"
```

---

## 2. Completed Kernel Optimizations Catalog (`[DONE]`)

### A. `NativeExpertLayout` (MoE Decode) `[DONE]`
- **Files**: [`src/kernels/sycl/iq_kernels.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/iq_kernels.cpp)
- **VTune Issue**: Register spills up to **16,896 bytes** per work-item in `expdec_gu_r` and `expdec_down_r`.
- **Root Cause**:
  1. Passing 88-byte `NativeExpertLayout L` struct by value.
  2. Monolithic 4-way `switch (n)` inside the kernel loop instantiating 4 simultaneous unrolled reduction trees.
  3. Default 128-GRF mode in SIMD32.
- **Optimization Applied**:
  - Implemented specialized `ExpdecGuGrf256Functor` and `ExpdecDownGrf256Functor` requesting `grf_size<256>`.
  - Splitting entry dispatches eliminates compiler register spilling.

### B. `strata::kernels::Q6KTraits` (MMVQ Q6_K) `[DONE]`
- **File**: [`src/kernels/sycl/native_mmvq.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/native_mmvq.cpp#L1388)
- **VTune Issue**: 17.67s execution time (6.08% share), **57.5% pipeline stalls**, spills up to 12.3 KB in wide column modes.
- **Root Cause**: `mmvq_large_grf` previously only checked for `IntelGpuGen::Alchemist`. On Battlemage B580 (`IntelGpuGen::Battlemage`), it returned `false`, forcing 128 GRFs. The 57.5% stall rate was caused by EUs waiting on high-latency roundtrips to private scratch memory.
- **Optimization Applied**:
  - Extended architecture check to include `IntelGpuGen::Battlemage`.
  - Added `B6RowsGrf256Functor` and `NativeMmvqMultiGrf256Functor` with `grf_size<256>`.
  - Verification: 8-column Q6_K in `b6_mmvq_bench` dropped from **3.08 ms** to **2.25 ms** with bit-exact parity (`ok ok ok`).

### C. `strata::kernels::...::GrMulti` (Down Projection) `[DONE]`
- **File**: [`src/kernels/sycl/native_gr.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/native_gr.cpp)
- **VTune Issue**: 23.70s execution time (8.15% share), 2,432 B register spill, 45.8% stalls.
- **Root Cause**: Multi-token down projection used generalized loops and a large 192-byte `GrDownParams` struct, causing live-range overlap across registers.
- **Optimization Applied**:
  - Specialized token count paths for $T=1, 2, 3, 4$ using compile-time constants.
  - Compact 144-byte descriptor `GrDownArgs`.
  - Direct weight streaming and register unrolling.
  - Verification: Single-token latency in `fused_gr_bench` dropped from **240 µs** to **188 µs** (all 16 benchmark configs bitwise exact).

### D. `Q5KTraits` (MMVQ Q5_K) `[DONE]`
- **File**: [`src/kernels/sycl/native_mmvq.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/native_mmvq.cpp#L927)
- **VTune Issue**: Register spills up to 5,504 bytes when decoding multi-column tokens.
- **Root Cause**: Unpacking 5-bit scales, 2-bit high bits, and 4-bit low nibbles per lane while holding 8 token accumulators in SIMD32 exceeded 128 GRFs.
- **Optimization Applied**:
  - Enabled 256-GRF functors `B6RowsGrf256Functor` and `NativeMmvqMultiGrf256Functor`.
  - Verification: `head Q5_K` (cols=8) in `b6_mmvq_bench` dropped from **37.01 ms** to **32.57 ms** (4.44 ms faster per call).

### E. `bf16_gemv_fp32_mmvf` `[DONE]`
- **File**: [`src/kernels/sycl/native_bf16.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/native_bf16.cpp#L400)
- **VTune Issue**: Register spills and high stall rates in multi-token MoE router and PLE projection.
- **Root Cause**: In multi-row mode (`STRATA_MMVF_ROWS=1`) and multi-token mode ($T=2..8$), 4 rows $\times$ 8 tokens = 32 float accumulators. In SIMD32, accumulators alone consume 64 GRFs, pushing total register requirements beyond the 128-GRF ceiling.
- **Optimization Applied**:
  - Functorized both multi-row (`Bf16MmvfRowsGrf256Functor`) and multi-column (`Bf16MmvfMultiGrf256Functor`) with `grf_size<256>`.
  - Verification: Zero register spills across all token batch sizes with bit-exact outputs.

### F. `SmallTraits` `[DONE]`
- **File**: [`src/kernels/sycl/native_mmvq.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/native_mmvq.cpp#L1151)
- **VTune Issue**: High register pressure in 32-element quantized blocks (`Q4_0`, `Q5_0`, `Q8_0`, `IQ4_NL`).
- **Optimization Applied**:
  - Routed through `NativeMmvqMultiGrf256Functor` (256 GRFs).
  - Verification: `shexp down IQ4_NL` passes all columns with bitwise equality in `b6_mmvq_bench`.

---

## 3. Small Kernels: Fusion vs. SYCL Command Graph Replay (`[DONE]`)

During token decode, models execute **~280 to 550 small kernels per token** (e.g. 48 layers $\times$ ~6–8 kernels/layer):
1. **Host-GPU Doorbell & Synchronization**: `doorbell_publish` (writes IDs to host pinned USM), `doorbell_wait` (in-graph GPU barrier), `copy_from_mapped` (copies CPU expert results).
2. **Normalization & RoPE**: `finish_norm_rope` (fused RMSNorm + RoPE) or standalone `rmsnorm` / `native_rope_apply`.
3. **Activation Quantization & MoE Dispatch**: `quantize_q8_0_scaled`, `moe_hit_select`, `moe_hit_add`.
4. **GDN Recurrence**: `fused_gdn_conv_l2` (conv + RMSNorm), `fused_gdn_ab` (gating), `fused_gdn_step_norm` (state update + gate + norm).
5. **Hyper-Connection**: `fused_gr_read` (residual read/norm), `gr_write` (residual write).

### Architectural Reality: Why Graph Replay Handles 90% of the Problem
* In standard eager submission, 400 launches $\times$ 10 µs host latency = **4.0 ms dispatch overhead per token**.
* Strata captures the entire forward pass via **SYCL Command Graphs** (`sycl::ext::oneapi::experimental::command_graph` in `CapturedGraph`).
* In graph replay, the entire multi-hundred kernel sequence is submitted in **a single hardware dispatch** (`q->ext_oneapi_graph(*exec)`), cutting host dispatch latency to near zero.
* Intermediate activations (8–20 KB) fit entirely inside the Intel Xe2 **16 MB L2 cache**, meaning they do not hit external DRAM.
* Handshake kernels (`doorbell_publish`, `doorbell_wait`) **must not be fused** into compute kernels because they serve as asynchronous CPU-GPU pipeline synchronization barriers.

---

## 4. Pending Action Items for Next Session (`[TODO]`)

The following items are verified and ready for engineering implementation:

### Item 1: Eliminate 1,024 B Register Spill in `s2_expert_grouped.cpp` `[TODO]`
* **Location**: [`src/kernels/sycl/s2_expert_grouped.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/s2_expert_grouped.cpp)
* **Problem**: In IGC compiler dumps, `grouped_s2_impl` exhibits a **1,024 byte register spill** per thread when running in 128-GRF mode during MoE expert decode.
* **Implementation Plan**:
  1. Define a `GroupedS2Grf256Functor` wrapper implementing `get(sycl::ext::oneapi::experimental::properties_tag) const` returning `properties{grf_size<256>, sub_group_size<32>}`.
  2. Check `intel_gpu_gen(q->get_device())` for `Alchemist` and `Battlemage`.
  3. Dispatch through the 256-GRF functor when supported.
  4. Verify with `./build_sycl/s2_expert_grouped_parity` and scan IGC dump to confirm 0 B spill.

### Item 2: Extend L3 Streaming Cache Hints to MMVQ and MoE Weight Reads `[TODO]`
* **Location**: [`src/kernels/sycl/native_mmvq.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/native_mmvq.cpp) and [`src/kernels/sycl/s2_expert_grouped.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/s2_expert_grouped.cpp)
* **Problem**: Multi-gigabyte MMVQ and MoE weight matrices are read once per token with zero intra-token temporal reuse. Reading them without cache control pollutes Intel Xe2's 16 MB L3 cache, evicting the KV cache and hot activations.
* **Implementation Plan**:
  1. Follow the reference implementation in [`src/kernels/sycl/bf16_gemv.cpp#L464`](file:///home/girtam/projects/Strata/src/kernels/sycl/bf16_gemv.cpp#L464):
     ```cpp
     #include <sycl/ext/intel/experimental/cache_control_properties.hpp>
     // Apply:
     sycl::ext::intel::experimental::cache_control<
         sycl::ext::intel::experimental::cache_mode::streaming,
         sycl::ext::oneapi::experimental::cache_level::L3>
     ```
  2. Apply streaming hints to the global weight buffer accessors or pointers.
  3. Validate that KV cache hit rates and throughput increase during multi-layer inference.

### Item 3: GDN Epilogue Quantization Fusion `[TODO]`
* **Location**: [`src/kernels/sycl/fused_gdn.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/fused_gdn.cpp)
* **Problem**: `fused_gdn_step_norm` writes output vector $y$ as FP32 to `b.y`, followed immediately by `quantize_act` / `quantize_row_q8_1` to convert it to Q8 blocks for down-projection.
* **Implementation Plan**:
  1. Add a templated variant `fused_gdn_step_norm<QuantizeQ8 = true>`.
  2. Compute group max/min scales directly across sub-group registers after RMSNorm + gate.
  3. Write 34-byte Q8_1 blocks directly to device memory, eliminating the standalone `quantize_act` kernel launch across all GDN layers.

### Item 4: Fused MoE Hit Accumulate + Residual Add `[TODO]`
* **Location**: [`src/core/sycl/layer.cpp`](file:///home/girtam/projects/Strata/src/core/sycl/layer.cpp) and [`src/kernels/sycl/elementwise.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/elementwise.cpp)
* **Problem**: After CPU experts complete, the pipeline executes `moe_hit_add` (combining GPU hits and CPU expert outputs) followed shortly by `gr_write` (adding to the hidden state residual).
* **Implementation Plan**:
  1. Fuse both elementwise additions into a single kernel pass:
     $$\mathbf{res}[i] \leftarrow \mathbf{res}[i] + \mathbf{gpu\_hits}[i] + \mathbf{cpu\_misses}[i]$$
  2. Eliminates 1 intermediate global memory read and write per MoE layer.

### Item 5: End-to-End VTune Re-Profile on Battlemage B580 `[TODO]`
* **Plan**:
  1. Launch full inference session with VTune GPU hotspot collection:
     ```bash
     source /opt/intel/oneapi/setvars.sh
     vtune -collect gpu-hotspots -r /tmp/vtune_b580_post_opt -- strata launch --family qwen --model q2_0 --no-tui
     ```
  2. Verify:
     - Pipeline stall reduction from initial **57.5% baseline**.
     - Register spills across all active kernels = **0 Bytes**.
     - EU active percentage increase.

---

## 5. Troubleshooting & Failure Recovery Guide

If you encounter issues during compilation, offline profiling, benchmark execution, or runtime inference, follow these diagnosis and remediation steps:

### Failure Mode 1: DPC++ Compilation Error with `launch_config`
* **Symptom**:
  ```text
  error: launch_config does not allow properties with compile-time kernel effects
  ```
* **Cause**: Intel oneAPI DPC++ requires properties that affect register allocation (such as `grf_size<256>` or `sub_group_size<32>`) to be known at kernel compilation time. Passing them via runtime `sycl::ext::oneapi::experimental::launch_config` to `parallel_for` is rejected.
* **Resolution**:
  Use a kernel **Functor class** implementing `get(properties_tag)`:
  ```cpp
  #include <sycl/ext/intel/experimental/grf_size_properties.hpp>

  struct MyKernelFunctor {
      // Kernel arguments as struct members ...
      auto get(sycl::ext::oneapi::experimental::properties_tag) const {
          return sycl::ext::oneapi::experimental::properties{
              sycl::ext::intel::experimental::grf_size<256>,
              sycl::ext::oneapi::experimental::sub_group_size<32>
          };
      }
      void operator()(sycl::nd_item<1> item) const {
          // Kernel body ...
      }
  };
  ```

### Failure Mode 2: Out of Resources or Scratch Allocation Failure
* **Symptom**:
  ```text
  CL_OUT_OF_RESOURCES (-5) or ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
  ```
* **Cause**:
  1. Enabling 256 GRF mode cuts the number of hardware EU threads per subslice in half (e.g., from 8 threads down to 4 threads on Xe-HPG/Xe2).
  2. If a kernel requests a workgroup size larger than the hardware thread limit or attempts to dispatch excessive simultaneous blocks with large local memory, the driver may fail to schedule.
* **Immediate Workaround / Fallback**:
  You can dynamically disable 256-GRF mode without recompilation by setting environment variables:
  ```bash
  # Force MMVQ to stay in 128-GRF mode
  export STRATA_MMVQ_LARGE_GRF=0

  # Disable fused BF16 MMVF rows mode
  export STRATA_MMVF_ROWS=0

  # Disable B6 experimental MMVQ rows mode
  export STRATA_B6_MMVQ_ROWS=0
  ```

### Failure Mode 3: IGC Dump Failure, Test Slowdown, or Disk Exhaustion
* **Symptom**:
  Commands hang or run out of disk space in `/tmp`.
* **Cause**:
  `IGC_ShaderDumpEnable=1` dumps `.zeinfo`, `.asm`, and `.visaasm` files for every single kernel template instantiation. A full test run can produce gigabytes of dump files.
* **Resolution**:
  1. **Do not enable IGC dump for full test suites** (`./tools/test_sycl.sh all`).
  2. Enable dumps only for isolated microbenchmarks:
     ```bash
     mkdir -p /tmp/kernel_dump
     export IGC_ShaderDumpEnable=1
     export IGC_DumpToCustomDir=/tmp/kernel_dump
     ./build_sycl/b6_mmvq_bench
     unset IGC_ShaderDumpEnable
     unset IGC_DumpToCustomDir
     ```
  3. Clean up the dump directory afterwards:
     ```bash
     rm -rf /tmp/kernel_dump
     ```

### Failure Mode 4: Numerical Mismatch or Bitwise Parity Regression
* **Symptom**:
  `b6_mmvq_bench` prints `FAIL` or `bitwise B=A: mismatch`.
* **Cause**:
  Cross-warp reductions in multi-column layouts group partial sums differently than single-column calls. Floating-point non-associativity can cause small numerical divergences if exact reduction trees are bypassed.
* **Resolution**:
  Ensure exact reduction ordering is enforced:
  ```bash
  # Enable exact bitwise parity reduction path
  export STRATA_MMVQ_EXACT=1
  ```
  Run the test suite to verify:
  ```bash
  ./tools/test_sycl.sh fast
  ```

### Failure Mode 5: VTune Profiling Missing Symbols or In-Kernel Counters
* **Symptom**:
  VTune shows mangled kernel names or reports:
  ```text
  Cannot collect GPU in-kernel profiling data: permissions denied
  ```
* **Resolution**:
  1. Ensure the build was configured with debug symbols (`-gline-tables-only` or `RelWithDebInfo`):
     ```bash
     cmake -B build_sycl -DCMAKE_BUILD_TYPE=RelWithDebInfo
     ```
  2. Adjust Linux kernel performance counter permissions:
     ```bash
     sudo sysctl -w dev.i915.perf_stream_paranoid=0
     ```
  3. Run VTune with the Intel oneAPI environment sourced:
     ```bash
     source /opt/intel/oneapi/setvars.sh
     vtune -collect gpu-hotspots -r /tmp/vtune_report -- ./build_sycl/b6_mmvq_bench
     ```

---

## 6. Environment Variables Reference Table

Use this quick-reference table to toggle or isolate specific kernel optimizations during debugging or incident response:

| Environment Variable | Default | Purpose / Fallback Behavior |
|---|---|---|
| `STRATA_MMVQ_LARGE_GRF` | Auto (`1` on B580) | Set to `0` to force 128-GRF mode for all MMVQ kernels. |
| `STRATA_MMVF_ROWS` | `0` | Set to `0` to disable BF16 multi-row fused kernel and use standard multi-token kernel. |
| `STRATA_B6_MMVQ_ROWS` | `0` | Set to `0` to disable B6 multi-row MMVQ optimization. |
| `STRATA_MMVQ_EXACT` | `1` | Enforces exact bitwise reduction matching single-column execution. |
| `IGC_ShaderDumpEnable` | `0` | Set to `1` to generate offline `.zeinfo` and `.asm` compiler dumps. |
| `IGC_DumpToCustomDir` | None | Destination path for IGC shader dumps. |
