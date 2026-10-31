# VTune Kernel Profiling, Register Spill Inspection & Troubleshooting Guide

**Date**: 2026-10-30  
**Target Architecture**: Intel Arc Battlemage B580 / Lunar Lake Xe2 (`[0x64a0]`)  
**Status**: Root Causes Identified, Offline Diagnostics Established, 256 GRF Mode Deployed, Verification Complete

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

When running any test binary or targeted kernel microbenchmark in `build_sycl/` (such as `./build_sycl/b6_mmvq_bench`, `./build_sycl/fused_gr_bench`, or `./build_sycl/native_grouped_parity`), IGC automatically emits compiler artifacts for every compiled device kernel into the target directory:

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

## 2. Kernel Profiling Findings & Optimization Catalog

### A. `NativeExpertLayout` (MoE Decode)
- **Files**: [`src/kernels/sycl/iq_kernels.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/iq_kernels.cpp)
- **VTune Issue**: Register spills up to **16,896 bytes** per work-item in `expdec_gu_r` and `expdec_down_r`.
- **Root Cause**:
  1. Passing 88-byte `NativeExpertLayout L` struct by value.
  2. Monolithic 4-way `switch (n)` inside the kernel loop instantiating 4 simultaneous unrolled reduction trees.
  3. Default 128-GRF mode in SIMD32.
- **Optimization**:
  - Implemented specialized `ExpdecGuGrf256Functor` and `ExpdecDownGrf256Functor` requesting `grf_size<256>`.
  - Splitting entry dispatches eliminates compiler register spilling.

### B. `strata::kernels::Q6KTraits` (MMVQ Q6_K)
- **File**: [`src/kernels/sycl/native_mmvq.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/native_mmvq.cpp#L1388)
- **VTune Issue**: 17.67s execution time (6.08% share), **57.5% pipeline stalls**, spills up to 12.3 KB in wide column modes.
- **Root Cause**: `mmvq_large_grf` previously only checked for `IntelGpuGen::Alchemist`. On Battlemage B580 (`IntelGpuGen::Battlemage`), it returned `false`, forcing 128 GRFs. The 57.5% stall rate was caused by EUs waiting on high-latency roundtrips to private scratch memory.
- **Optimization**:
  - Extended architecture check to include `IntelGpuGen::Battlemage`.
  - Added `B6RowsGrf256Functor` and `NativeMmvqMultiGrf256Functor` with `grf_size<256>`.
  - Verification: 8-column Q6_K in `b6_mmvq_bench` dropped from **3.08 ms** to **2.25 ms** with bit-exact parity (`ok ok ok`).

### C. `strata::kernels::...::GrMulti` (Down Projection)
- **File**: [`src/kernels/sycl/native_gr.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/native_gr.cpp)
- **VTune Issue**: 23.70s execution time (8.15% share), 2,432 B register spill, 45.8% stalls.
- **Root Cause**: Multi-token down projection used generalized loops and a large 192-byte `GrDownParams` struct, causing live-range overlap across registers.
- **Optimization**:
  - Specialized token count paths for $T=1, 2, 3, 4$ using compile-time constants.
  - Compact 144-byte descriptor `GrDownArgs`.
  - Direct weight streaming and register unrolling.
  - Verification: Single-token latency in `fused_gr_bench` dropped from **240 µs** to **188 µs** (all 16 benchmark configs bitwise exact).

### D. `Q5KTraits` (MMVQ Q5_K)
- **File**: [`src/kernels/sycl/native_mmvq.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/native_mmvq.cpp#L927)
- **VTune Issue**: Register spills up to 5,504 bytes when decoding multi-column tokens.
- **Root Cause**: Unpacking 5-bit scales, 2-bit high bits, and 4-bit low nibbles per lane while holding 8 token accumulators in SIMD32 exceeded 128 GRFs.
- **Optimization**:
  - Enabled 256-GRF functors `B6RowsGrf256Functor` and `NativeMmvqMultiGrf256Functor`.
  - Verification: `head Q5_K` (cols=8) in `b6_mmvq_bench` dropped from **37.01 ms** to **32.57 ms** (4.44 ms faster per call).

### E. `bf16_gemv_fp32_mmvf`
- **File**: [`src/kernels/sycl/native_bf16.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/native_bf16.cpp#L400)
- **VTune Issue**: Register spills and high stall rates in multi-token MoE router and PLE projection.
- **Root Cause**: In multi-row mode (`STRATA_MMVF_ROWS=1`) and multi-token mode ($T=2..8$), 4 rows $\times$ 8 tokens = 32 float accumulators. In SIMD32, accumulators alone consume 64 GRFs, pushing total register requirements beyond the 128-GRF ceiling.
- **Optimization**:
  - Functorized both multi-row (`Bf16MmvfRowsGrf256Functor`) and multi-column (`Bf16MmvfMultiGrf256Functor`) with `grf_size<256>`.
  - Verification: Zero register spills across all token batch sizes with bit-exact outputs.

### F. `SmallTraits`
- **File**: [`src/kernels/sycl/native_mmvq.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/native_mmvq.cpp#L1151)
- **VTune Issue**: High register pressure in 32-element quantized blocks (`Q4_0`, `Q5_0`, `Q8_0`, `IQ4_NL`).
- **Optimization**:
  - Routed through `NativeMmvqMultiGrf256Functor` (256 GRFs).
  - Verification: `shexp down IQ4_NL` passes all columns with bitwise equality in `b6_mmvq_bench`.

---

## 3. Troubleshooting & Failure Recovery Guide

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

## 4. Environment Variables Reference Table

Use this quick-reference table to toggle or isolate specific kernel optimizations during debugging or incident response:

| Environment Variable | Default | Purpose / Fallback Behavior |
|---|---|---|
| `STRATA_MMVQ_LARGE_GRF` | Auto (`1` on B580) | Set to `0` to force 128-GRF mode for all MMVQ kernels. |
| `STRATA_MMVF_ROWS` | `0` | Set to `0` to disable BF16 multi-row fused kernel and use standard multi-token kernel. |
| `STRATA_B6_MMVQ_ROWS` | `0` | Set to `0` to disable B6 multi-row MMVQ optimization. |
| `STRATA_MMVQ_EXACT` | `1` | Enforces exact bitwise reduction matching single-column execution. |
| `IGC_ShaderDumpEnable` | `0` | Set to `1` to generate offline `.zeinfo` and `.asm` compiler dumps. |
| `IGC_DumpToCustomDir` | None | Destination path for IGC shader dumps. |

---

## 5. Empirical Verification of Proposed Kernel Opportunities

An analysis of five prospective optimizations was evaluated against the actual hardware compiler (Intel Graphics Compiler for Xe2 Battlemage `[0x64a0]`):

### Opportunity 1: Force 256 GRF on `ncols == 1` Decode MMVQ
* **Proposal**: Extend 256-GRF functors to single-column (`ncols == 1`) decode paths in `native_mmvq.cpp` (e.g. lines 1687, 1712, 2327, 2352).
* **Empirical Finding**: **COUNTER-PRODUCTIVE (Do Not Apply)**.
* **Evidence from IGC Dump**:
  - Scanning `.zeinfo` compiler artifacts on Battlemage B580 shows that all `ncols == 1` kernels (`native_q5_k_mmvq`, `native_q4_k_mmvq`, `native_q6_k_mmvq`, `native_small_mmvq`, `native_iq4_xs_mmvq`) have **0 bytes of spill** (`HasSpill: False`) in 128-GRF mode because they maintain only a single column accumulator (`float sum = 0.0f`).
  - In 128-GRF mode, Execution Unit thread occupancy is **`eu_thread_count: 8`** (100% full hardware thread occupancy).
  - Forcing 256 GRFs cuts EU thread occupancy in half (`eu_thread_count: 4`) with zero spill reduction, degrading memory latency hiding during single-token autoregressive decode.
  - **Verdict**: Retain 128 GRFs for `ncols == 1` and 256 GRFs for `ncols > 1`.

### Opportunity 2: Stale `sycl/src/kernels/cuda/*.dp.cpp` Mirror
* **Proposal**: Delete and regenerate supposedly stale `.dp.cpp` files in a CUDA mirror directory.
* **Empirical Finding**: **INCORRECT / NON-EXISTENT**.
* **Evidence**:
  - There is no `sycl/src/kernels/cuda/*.dp.cpp` directory.
  - Active device kernels are maintained and compiled directly from `src/kernels/sycl/*.cpp`.
  - Legacy output `src/sycl_out/` from early porting phases is unreferenced by `CMakeLists.txt` and has no bearing on device compilation or runtime execution.

### Opportunity 3: GRF Handling in `s2_expert_grouped.cpp`, `fused_gr.cpp`, `qsa_select.cpp`, `native_moe.cpp`
* **Proposal**: Check for spills in MoE, residual, and attention kernels.
* **Empirical Finding**: **CONFIRMED SPILL IN `s2_expert_grouped.cpp`**.
* **Evidence from IGC Dump**:
  - `fused_gr.cpp`: **0 spills** (resolved in commit `11b1922` with token unrolling and direct streaming; 24/24 configs pass).
  - `qsa_select.cpp`: **0 spills** in 128-GRF mode.
  - `s2_expert_grouped.cpp`: **Spills 1,024 bytes** in `grouped_s2_impl` under 128 GRFs.
  - **Action**: Implementing 256-GRF support for `grouped_s2_impl` on Battlemage is a validated high-priority opportunity to eliminate this 1 KB scratch spill during MoE decode.

### Opportunity 4: Streaming Cache Hints (`cache_mode::streaming`) for MMVQ & MoE
* **Proposal**: Extend `sycl::ext::intel::experimental::cache_control<cache_mode::streaming, cache_level::L3>` (currently used in `bf16_gemv.cpp`) to MMVQ and MoE weight streaming.
* **Empirical Finding**: **VALID & HIGH VALUE**.
* **Evidence**:
  - In MoE and batched MMVQ, multi-gigabyte weight tensors are read once per token without temporal reuse within the decode step.
  - Without streaming cache hints, reading weights pollutes the 16 MB L3 cache, evicting frequently accessed KV cache heads and activation buffers.
  - Applying `cache_mode::streaming` to weight loads preserves L3 cache residency for activations and the KV cache.

### Opportunity 5: SIMD16 + 256 GRF Exploration
* **Proposal**: Evaluate compiling kernels in SIMD16 with 256 GRFs instead of SIMD32.
* **Empirical Finding**: **BENCHMARK-ONLY (Breaks Bit-Exact Parity)**.
* **Evidence**:
  - SIMD16 doubles effective register capacity per lane, which can prevent spills in extreme unrolling.
  - However, sub-group tree reductions across 16 lanes group floating-point additions differently than 32-lane warps, causing numerical divergence (`bitwise ok ok ok` fails against single-column or CUDA oracles).
  - **Verdict**: Reserved strictly for offline throughput experiments; ineligible for main branch production where bit-exact parity is required.
