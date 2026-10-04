# Handover Document: Intel Arc B580 / Xe2 SYCL Optimization & Upstream Rebase

**Date & Time**: 2026-10-04 23:45 (Local)  
**Project**: Strata Engine (`/workspace/Strata`)  
**Current Branch**: `main` (cleanly rebased on upstream `origin/main` @ commit `41d50fd`, current HEAD @ commit `0e37fc5`)  
**Backup Branch**: `backup-main-before-rebase` (points to pre-rebase HEAD `6e28497`, preserves full 84-commit history)

---

## 1. Executive Summary

This project optimizes Strata's native SYCL backend for the **Intel Arc B580 (Xe2-HPG Battlemage)** discrete GPU, validated on the available **Intel Core Ultra 5 226V (Lunar Lake Xe2-LPG, Device ID `0x64a0`)** which shares the exact same Intel Xe2 execution engine (native SIMD16 execution vector engines, XMX 2.0 systolic matrix engines, 64 KB SLM).

The target model for these optimizations is **Qwen3.8-Flash-Next** (as well as `UD-Q4_K_XL`).

---

## 2. Completed Milestones

1. **Upstream Rebase (`v0.1.39`)**:
   - Rebased cleanly on top of `origin/main` (foundation commit `41d50fd` contains all 257 migrated SYCL files, +109,860 lines).
   - Preserved full 84-commit history on `backup-main-before-rebase`.
   - Build verified on live GPU: `check-sycl` passes 16/16 tests in ~3.8s with 0 warnings under `-Wall -Wextra`.
2. **SIMD16 Quantized GEMV Optimization** (Commit `cbc2121`):
   - [`src/kernels/sycl/s2_gemv.cpp`](../src/kernels/sycl/s2_gemv.cpp): Added `s2_gemv_subgroup_kernel` using SIMD16 subgroup reduction with `sycl::reduce_over_group`.
   - [`src/kernels/sycl/s_gemv.cpp`](../src/kernels/sycl/s_gemv.cpp): Replaced 32-lane hardcoded CUDA warp shuffles with native `SG_SIZE = 16`, stride `SG_SIZE * QE`, and single-cycle hardware `sycl::reduce_over_group`.
   - [`src/kernels/sycl/s2_gemv_fast.cpp`](../src/kernels/sycl/s2_gemv_fast.cpp) & [`src/kernels/sycl/s2_gemv_quads.cpp`](../src/kernels/sycl/s2_gemv_quads.cpp): Replaced shared memory tree reduction loops with hierarchical subgroup reductions.
   - Verified on GPU: `s2_gemv_parity`, `s_gemv_parity`, `s_gemv_q8k_parity`, and `s2_gemv_q8_parity` all pass with 0 failures.
3. **Documentation Consolidation** (Commits `a8fe5f9` & `0e37fc5`):
   - Merged scattered, redundant markdown files into canonical documents in `docs/`:
     - [`docs/SYCL_STATUS.md`](../docs/SYCL_STATUS.md): Master status, 2026.1 toolchain, and complete 44-kernel registry.
     - [`docs/SYCL_PORTING_GUIDE.md`](../docs/SYCL_PORTING_GUIDE.md): 6 translation directives, command graph specs, and AI/developer refactoring prompt rules.
     - [`docs/INTEL_B580_OPTIMIZATION_PLAN.md`](../docs/INTEL_B580_OPTIMIZATION_PLAN.md): B580 microarchitecture plan, SIMD16 status, XMX capabilities, and the full Section 6 microbenchmarking suite.
     - [`STATUS.md`](../STATUS.md): Clean root index pointing directly to `docs/`.
4. **Google Benchmark Xe2 Performance Suite (`perf/sycl/`)**:
   - Added `STRATA_BUILD_PERF` CMake option with automated `FetchContent` Google Benchmark v1.8.3 integration and `perf-sycl` target.
   - Bypasses host CPU dispatch/queue jitter (150–280 µs overhead) using `benchmark::UseManualTime()` with SYCL event timestamps (`command_start` to `command_end`).
   - Implemented and benchmarked all 5 planned performance microbenchmarks on live Lunar Lake Xe2 GPU (`Intel Graphics [0x64a0]`):
     1. [`perf_xe2_gemv.cpp`](../perf/sycl/perf_xe2_gemv.cpp): All 8 Qwen projections, SIMD16 vs SIMD32, and SLM split reduction (+25.3% speedup on `ssm_alpha`).
     2. [`perf_xe2_xmx_tiles.cpp`](../perf/sycl/perf_xe2_xmx_tiles.cpp): Hardware XMX systolic tiles ($16 \times 16 \times 16$ achieves **1.23 TFLOPS**, 1.73x over Tile8).
     3. [`perf_qsa_attention.cpp`](../perf/sycl/perf_qsa_attention.cpp): FlashAttention throughput up to 611k tok/s and Top-K sparse gather bandwidth (22.2 GB/s).
     4. [`perf_moe_accumulate.cpp`](../perf/sycl/perf_moe_accumulate.cpp): MoE 10-expert reduction (Subgroup Reduce: **81.3 GB/s**, 3.40x faster than Atomic at $B=32$).
     5. [`perf_dequant_dpas.cpp`](../perf/sycl/perf_dequant_dpas.cpp): Direct-to-register 4-bit dequant + DPAS (**423.7 GFLOPS**, **+28.5% faster** than SLM buffering).

---

## 3. Performance Test Suite (`perf/sycl/`) - All 5 Completed & Verified

1. **`perf_xe2_gemv.cpp` (Done)**: BF16 decode/prefill sweeps, SIMD16 vs SIMD32, and SLM split reduction.
2. **`perf_xe2_xmx_tiles.cpp` (Done)**: Hardware XMX systolic tile sweeps ($8 \times 16 \times 16$ vs $16 \times 16 \times 16$: **1.23 TFLOPS** verified).
3. **`perf_qsa_attention.cpp` (Done)**: Prompt FlashAttention throughput and Top-K sparse gather memory bandwidth.
4. **`perf_moe_accumulate.cpp` (Done)**: MoE 10-expert weighted reduction comparing Atomic vs Direct Combine vs Subgroup Reduce (**81.3 GB/s**).
5. **`perf_dequant_dpas.cpp` (Done)**: Weight-only streaming dequantization feeding XMX DPAS (**+28.5% faster** direct in-register).

---

## 4. Next Priorities: Applying Perf Findings to Production Kernels

1. **Hardware XMX Systolic Attention (`src/kernels/sycl/qsa_prompt_attn.cpp`)**:
   - The old kernel produced NaNs due to CUDA warp-shuffle emulation (`mma16816`) exceeding Xe2 SIMD16 subgroup bounds.
   - Replace with native SYCL 2020 `joint_matrix` systolic tile multiplication ($16 \times 16 \times 16$) verified in `perf_xe2_xmx_tiles.cpp` and `perf_qsa_attention.cpp`.
   - Verify parity with `src/kernels/sycl/qsa_prompt_attn_parity.cpp`.
2. **Production SLM Split Reduction for Narrow Projections**:
   - Port the verified $TPR=128$ SLM split reduction from `perf_xe2_gemv.cpp` into `src/kernels/sycl/native_gdn.cpp` for `ssm_alpha` and `ssm_beta` (+25.3% speedup).
3. **MoE Subgroup Reduce Integration**:
   - Replace any global atomic expert accumulation in `src/kernels/sycl/native_moe.cpp` with hierarchical `sycl::reduce_over_group` across SIMD16 subgroups for 3.4x faster prefill reduction.

---

## 5. Development & Profiling Commands

```bash
# Source Intel oneAPI 2026.1 environment
source /opt/intel/oneapi/setvars.sh --force 2>/dev/null

# Compile and run unit tests (fast CI)
make -C build_sycl -j$(nproc) check-sycl

# Configure and build all performance benchmark suites
cmake -B build_sycl -DSTRATA_BUILD_PERF=ON
make -C build_sycl perf-sycl -j4

# Run individual microbenchmark suites
./build_sycl/perf_xe2_gemv
./build_sycl/perf_xe2_xmx_tiles
./build_sycl/perf_qsa_attention
./build_sycl/perf_moe_accumulate
./build_sycl/perf_dequant_dpas
```

---

## 6. Recommended Prompts for the Next Session

### Option 1 (Production Kernel Fix): Port `qsa_prompt_attn.cpp` to Native XMX Joint Matrix
> *"Replace the emulated mma16816 in `src/kernels/sycl/qsa_prompt_attn.cpp` with the native SYCL 2020 joint_matrix systolic multiplication verified in our perf suite, and verify `qsa_prompt_attn_parity` on the GPU."*

### Option 2: Integrate SLM Split Reduction into GDN Kernel
> *"Port the TPR=128 SLM split reduction from `perf_xe2_gemv.cpp` into `src/kernels/sycl/native_gdn.cpp` for narrow ssm_alpha and ssm_beta projections to lock in the +25.3% speedup."*

### Option 3: End-to-End Inference Verification with Qwen3.8 GGUF Model
> *"Run Strata's end-to-end model inference pipeline on the Intel Xe2 GPU with a Qwen3.8 GGUF model and measure tokens/sec against our baseline numbers."*

