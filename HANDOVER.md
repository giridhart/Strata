# Handover Document: Intel Arc B580 / Xe2 SYCL Optimization & Upstream Rebase

**Date & Time**: 2026-10-04 23:45 (Local)  
**Project**: Strata Engine (`/home/girtam/projects/Strata`)  
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
   - [`src/kernels/sycl/s2_gemv.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/s2_gemv.cpp): Added `s2_gemv_subgroup_kernel` using SIMD16 subgroup reduction with `sycl::reduce_over_group`.
   - [`src/kernels/sycl/s_gemv.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/s_gemv.cpp): Replaced 32-lane hardcoded CUDA warp shuffles with native `SG_SIZE = 16`, stride `SG_SIZE * QE`, and single-cycle hardware `sycl::reduce_over_group`.
   - [`src/kernels/sycl/s2_gemv_fast.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/s2_gemv_fast.cpp) & [`src/kernels/sycl/s2_gemv_quads.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/s2_gemv_quads.cpp): Replaced shared memory tree reduction loops with hierarchical subgroup reductions.
   - Verified on GPU: `s2_gemv_parity`, `s_gemv_parity`, `s_gemv_q8k_parity`, and `s2_gemv_q8_parity` all pass with 0 failures.
3. **Documentation Consolidation** (Commits `a8fe5f9` & `0e37fc5`):
   - Merged scattered, redundant markdown files into canonical documents in `docs/`:
     - [`docs/SYCL_STATUS.md`](file:///home/girtam/projects/Strata/docs/SYCL_STATUS.md): Master status, 2026.1 toolchain, and complete 44-kernel registry.
     - [`docs/SYCL_PORTING_GUIDE.md`](file:///home/girtam/projects/Strata/docs/SYCL_PORTING_GUIDE.md): 6 translation directives, command graph specs, and AI/developer refactoring prompt rules.
     - [`docs/INTEL_B580_OPTIMIZATION_PLAN.md`](file:///home/girtam/projects/Strata/docs/INTEL_B580_OPTIMIZATION_PLAN.md): B580 microarchitecture plan, SIMD16 status, XMX capabilities, and the full Section 6 microbenchmarking suite (event profiling, Google Benchmark sweeps across all 8 Qwen3.8-Flash-Next shapes, and the 5 critical decode bottlenecks).
     - [`STATUS.md`](file:///home/girtam/projects/Strata/STATUS.md): Clean root index pointing directly to `docs/`.

---

## 3. In Progress: Hardware XMX Systolic Attention (`qsa_prompt_attn.cpp`)

- **Diagnosis**: The existing `qsa_prompt_attn.cpp` kernel was producing 100% NaNs because `mma16816` emulated a 32-lane CUDA warp using `sycl::select_from_group` with lane indices up to 31, which is illegal/out-of-bounds on Intel's SIMD16 execution units.
- **Parity Test Fix**: Updated `src/kernels/sycl/qsa_prompt_attn_parity.cpp` to explicitly check for `NaN` and `Inf` in outputs, ensuring parity failures are properly detected.
- **Hardware Validation**: Tested native XMX systolic matrix execution on the live GPU (`0x64a0`):
  - $16 \times 16 \times 16$ tile multiplication using `joint_matrix<sub_group, half, use::a, 16, 16, layout::row_major>` and `use::b, 16, 16, layout::col_major` loaded directly from SLM (`local_space`).
  - Verified FP16 and INT8 FlashAttention pipelines on synthetic context: 0 NaNs, matches double-precision reference within $3 \times 10^{-4}$.
- **Next Step**: Port `src/kernels/sycl/qsa_prompt_attn.cpp` to this native XMX systolic kernel implementation and verify with `qsa_prompt_attn_parity`.

---

## 4. Development Environment

```bash
# Source Intel oneAPI 2026.1 environment (do NOT prepend 2025.3 library paths)
source /opt/intel/oneapi/setvars.sh --force 2>/dev/null

# Compile and run test suite
make -C build_sycl -j$(nproc) check-sycl
```

---

## 5. Recommended Prompts for the Next Session

Pick any of the following high-impact tasks to start the next session:

### Option 1 (Top Priority / In Progress): Complete XMX Systolic Attention
> *"Implement and verify the Intel Xe2 hardware XMX systolic FlashAttention kernel in `src/kernels/sycl/qsa_prompt_attn.cpp` to replace `mma16816`, and make sure `qsa_prompt_attn_parity` and `check-sycl` pass on the GPU."*

### Option 2 (Performance Tuning): Implement Critical Decode Bottleneck Microbenchmarks
> *"Implement the dedicated microbenchmark and optimized kernel for Weight-Only Dequantization + XMX DPAS on the Gate projection `[2560, 6144]` as outlined in `docs/INTEL_B580_OPTIMIZATION_PLAN.md` Section 6.5."*

### Option 3 (End-to-End Validation): Run Full Model Inference
> *"Run end-to-end token generation with Strata using Qwen3.8-Flash-Next weights on the Intel GPU, and benchmark throughput with and without SYCL Command Graph capture."*
