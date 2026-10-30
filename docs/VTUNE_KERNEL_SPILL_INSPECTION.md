# VTune Kernel Profiling & Register Spill Inspection Report

**Date**: 2026-10-30  
**Target Architecture**: Intel Arc Battlemage B580 / Lunar Lake Xe2 (`[0x64a0]`)  
**Status**: Root Causes Identified, Offline Diagnostic Established, Optimization Strategy Defined

---

## 1. How to Check Kernels, Spills, and Occupancy Without Running Strata

It is completely possible to inspect all kernel metrics—including register spill sizes, GRF counts, SIMD widths, and thread occupancy—**offline** without starting the Strata server or running full LLM inference sessions.

### Method: Intel Graphics Compiler (IGC) Dump

Intel's oneAPI Level Zero driver / OpenCL runtime invokes the Intel Graphics Compiler (IGC). By setting the runtime environment variables:

```bash
source /opt/intel/oneapi/setvars.sh
export IGC_ShaderDumpEnable=1
export IGC_DumpToCustomDir=/path/to/dump_dir
```

When running any test binary or targeted kernel microbenchmark in `build_sycl/` (such as `./build_sycl/b6_mmvq_bench`, `./build_sycl/native_grouped_parity`, or `./build_sycl/gr_multi_parity`), IGC automatically emits compiler artifacts for every compiled device kernel into the target directory:

- `*.zeinfo`: YAML file detailing:
  - `grf_count`: Assigned register file size (128 or 256 GRFs).
  - `simd_size`: Executed sub-group width (SIMD16 or SIMD32).
  - `spill_size`: **Register spill size in bytes** to private scratch memory.
  - `eu_thread_count`: Number of hardware threads per Execution Unit (EU occupancy).
  - `per_thread_memory_buffers`: Allocated scratch memory space per thread.
- `*.asm` / `*.visaasm`: The actual Intel Gen ISA assembly instructions, showing `send` messages, latency cycles, and scratch read/write traffic.

### Quick Offline Scanner Script

You can inspect all kernel spills at once with this Python one-liner:

```bash
python3 -c "
import glob, re, subprocess
for fn in glob.glob('/tmp/dump/*.zeinfo'):
    with open(fn) as f: content = f.read()
    for entry in content.split('  - name: ')[1:]:
        name = entry.split('\n')[0].strip()
        spill_m = re.search(r'spill_size:\s*(\d+)', entry)
        if spill_m:
            spill = int(spill_m.group(1))
            grf = re.search(r'grf_count:\s*(\d+)', entry).group(1)
            simd = re.search(r'simd_size:\s*(\d+)', entry).group(1)
            demangled = subprocess.check_output(['c++filt', name]).decode().strip() if name else 'unknown'
            print(f'SPILL {spill:5d} B | SIMD{simd} | GRF {grf} | {demangled}')
"
```

---

## 2. Finding: `NativeExpertLayout` Register Spills

### Problem Statement
In the VTune profile, a kernel associated with `NativeExpertLayout` exhibited register spills.

### Root Cause Analysis
Running the offline IGC dump against `native_grouped_parity` revealed the exact culprits in [`src/kernels/sycl/iq_kernels.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/iq_kernels.cpp):

| Kernel Function | Template Parameters | Spill Size (Bytes) |
|---|---|---|
| `expdec_gu_r` | `<21, 2, 4>` (IQ2_XXS, 2 rows, minb 4) | **16,896 Bytes** |
| `expdec_gu_r` | `<22, 2, 4>` (IQ2_XS, 2 rows, minb 4) | **15,168 Bytes** |
| `expdec_gu_r` | `<17, 2, 4>` (IQ3_S, 2 rows, minb 4) | **14,272 Bytes** |
| `expdec_gu_r` | `<21, 1, 2>` (IQ2_XXS, 1 row, minb 2) | **12,096 Bytes** |
| `expdec_gu_r` | `<18, 1, 2>` (IQ3_M, 1 row, minb 2) | **11,584 Bytes** |
| `expdec_gu_r` | `<16, 1, 2>` (IQ3_XXS, 1 row, minb 2) | **11,392 Bytes** |
| `launch_gu` | `<42>` (IQ4_NL) | **9,984 Bytes** |
| `expdec_down_r` | `<42, 4>` (IQ4_NL, 4 rows) | **5,760 Bytes** |
| `launch_gu` | `<21>` (IQ2_XXS) | **5,056 Bytes** |
| `launch_gu` | `<16>` (IQ3_XXS) | **3,840 Bytes** |

### Why Does It Spill Up to 16.8 KB?
1. **Pass-by-Value Struct Bloat**: `NativeExpertLayout L` is 88 bytes, passed by value into the kernel lambdas.
2. **Monolithic In-Kernel Switch Instantiation**: In `expdec_gu_kernel`:
   ```cpp
   switch (n) {
       case 1: run(std::integral_constant<int, 1>{}); break;
       case 2: run(std::integral_constant<int, 2>{}); break;
       case 3: run(std::integral_constant<int, 3>{}); break;
       default: run(std::integral_constant<int, 4>{}); break;
   }
   ```
   Each `run` instantiates a fully unrolled copy of `ed_gu_rows<TG, R, NC>`, creating 4 simultaneous unrolled computational blocks in a single function body.
3. **SIMD32 Vector Width Pressure**: In SIMD32, every variable spans 32 lanes (128 bytes per 32-bit register). Having 4 unrolled reduction paths causes extreme register live-range overlap, exceeding the 128-GRF hardware limit.

---

## 3. Finding: `strata::kernels::Q6KTraits` High Stalls & Spills

### VTune Profile Metrics
- **Time**: 17.67s (6.08% total kernel time share)
- **Call Count**: 91,799 calls (mean duration 192.54 µs)
- **Stall Rate**: **57.5% High Stall** (low active execution)
- **Work Size**: `global=40960x4, local=32x4`
- **Memory Bandwidth**: 80.66 GB/s read, 24.92 GB/s write

### Exact Kernel Identified
- **File**: [`src/kernels/sycl/native_mmvq.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/native_mmvq.cpp#L1388)
- **Function**: `b6_mmvq_rows_kernel<Q6KTraits, NCOLS, ROWS>` and `launch_q6k_wide<NCOLS>`.
- The workgroup size `local=32x4` matches `threads(1, WARPS, WARP)` (`WARPS=4`, `WARP=32`).
- The grid size `global=40960x4` represents 40,960 blocks executing multi-token/batched MMVQ for Q6_K matrices.

### Root Cause Analysis: The B580 Architecture Misdetection
In `src/kernels/sycl/native_mmvq.cpp` line 1766:

```cpp
inline bool mmvq_large_grf(sycl::queue* s) {
    static const bool v = [s] {
        const char* e = std::getenv("STRATA_MMVQ_LARGE_GRF");
        if (e != nullptr) return std::atoi(e) != 0;
        return strata::intel_gpu_gen(s->get_device()) == strata::IntelGpuGen::Alchemist;
    }();
```

1. **Battlemage B580 Ignored**: The runtime checks if the GPU is `Alchemist`. On Intel Arc Battlemage (B580) and Lunar Lake Xe2, `strata::intel_gpu_gen(...)` returns `IntelGpuGen::Battlemage`. Therefore, `mmvq_large_grf` returned **`false`**!
2. **Missing Large GRF (256 Registers)**:
   Because `mmvq_large_grf` was false, the kernel fell back to the 128-GRF default.
   In 128-GRF mode, the IGC dump proves `Q6KTraits` spills massively:
   - `launch_q6k_wide<5>`: **12,352 bytes spill**
   - `launch_q6k_wide<6>`: **10,816 bytes spill**
   - `launch_q6k_wide<8>`: **10,752 bytes spill**
   - `b6_launch<Q6KTraits, 8>`: **3,904 bytes spill**
   - `launch_q6k_wide<4>`: **3,456 bytes spill**
   - `b6_launch<Q6KTraits, 6>`: **1,728 bytes spill**
3. **Why 57.5% Stalls?**:
   - The GPU read bandwidth was measured at only **80.66 GB/s** (compared to B580's peak 456 GB/s bandwidth, a utilization of under 18%).
   - The Execution Units spend 57.5% of their time stalled waiting for high-latency roundtrips to private scratch memory to service the 10 KB+ spills per thread.
   - Additionally, `load_int_b2` performs two 16-bit unaligned loads and bitwise recombination for weights, causing sub-optimal memory transactions.

---

## 4. Actionable Optimization Plan

### Step 1: Enable Large GRF for Battlemage in `native_mmvq.cpp`
Update line 1770 in `src/kernels/sycl/native_mmvq.cpp`:
```cpp
inline bool mmvq_large_grf(sycl::queue* s) {
    static const bool v = [s] {
        const char* e = std::getenv("STRATA_MMVQ_LARGE_GRF");
        if (e != nullptr) return std::atoi(e) != 0;
        const auto gen = strata::intel_gpu_gen(s->get_device());
        return gen == strata::IntelGpuGen::Alchemist || gen == strata::IntelGpuGen::Battlemage;
    }();
    return v;
}
```
Applying `sycl::ext::intel::experimental::grf_size<256>` instantly doubles register capacity from 128 to 256 GRFs, eliminating the spill in wide MMVQ.

### Step 2: Specialize `expdec_gu_kernel` Entry Loops
In `src/kernels/sycl/iq_kernels.cpp`, remove the 4-way `switch (n)` inside the kernel loop. Instead, dispatch separate kernel entry points for `NC = 1, 2, 3, 4` so each kernel only compiles one active unrolled reduction path, slashing register pressure and eliminating the 16.8 KB spill.

### Step 3: Stream Unaligned Q6_K Weights
Convert `load_int_b2` scalar pairs into vectorized 32-bit aligned reads where block parity allows, or prefer planar layouts (`STRATA_Q6_PLANAR=1`).
