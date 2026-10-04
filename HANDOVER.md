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
     - [`docs/INTEL_B580_OPTIMIZATION_PLAN.md`](file:///home/girtam/projects/Strata/docs/INTEL_B580_OPTIMIZATION_PLAN.md): B580 microarchitecture plan, SIMD16 status, XMX capabilities, and the full Section 6 microbenchmarking suite.
     - [`STATUS.md`](file:///home/girtam/projects/Strata/STATUS.md): Clean root index pointing directly to `docs/`.
4. **Google Benchmark Xe2 Performance Suite (`perf/sycl/perf_xe2_gemv.cpp`)**:
   - Added `STRATA_BUILD_PERF` CMake option with automated `FetchContent` Google Benchmark v1.8.3 integration and `perf-sycl` target.
   - Bypasses host CPU dispatch/queue jitter (150–280 µs overhead) using `benchmark::UseManualTime()` with SYCL event timestamps (`command_start` to `command_end`).
   - Implemented automated sweeps across all 8 Qwen3.8-Flash-Next decode projections, expert layers, and chunked prefill shapes ($B=24, 32$).
   - Implemented microarchitectural parameter sweeps: SIMD16 vs SIMD32 across work-group sizes, and SLM split reductions across `threads_per_row` (64, 128, 256).
   - Validated on live GPU (`Intel Graphics [0x64a0]`): confirmed SIMD16 efficiency, measured up to 102.4 GB/s memory bandwidth on expert shapes, and demonstrated **+25.3% speedup** on `ssm_alpha` with SLM split reduction ($TPR=128$).
5. **Hardware XMX Systolic Tile Benchmark Suite (`perf/sycl/perf_xe2_xmx_tiles.cpp`)**:
   - Built with SYCL 2020 `joint_matrix` systolic matrix multiplication across tile dimensions ($8 \times 16 \times 16$ vs $16 \times 16 \times 16$).
   - Measured on live Lunar Lake Xe2 GPU (`Intel Graphics [0x64a0]`): on `attn_qkv` ($M=32, K=2560, N=10240$), Tile16 ($16 \times 16 \times 16$) reached **1.23 TFLOPS** (1.73x faster than Tile8's 710 GFLOPS) by amortizing weight bandwidth across 16 rows.

---

## 3. Performance Test Suite Extension Roadmap (`perf/sycl/`)

To systematically drive kernel optimization on Xe2, performance tests are organized in `perf/sycl/`:

1. **`perf_xe2_gemv.cpp` (Done)**: Parameterized BF16 decode/prefill sweeps, SIMD16 vs SIMD32, and SLM split reduction.
2. **`perf_xe2_xmx_tiles.cpp` (Done)**: Hardware XMX systolic tile sweeps ($8 \times 16 \times 16$ vs $16 \times 16 \times 16$: **1.23 TFLOPS** verified).
3. **`perf_qsa_attention.cpp` (Next)**: Prompt FlashAttention throughput and Top-K sparse gather memory bandwidth.
4. **`perf_moe_accumulate.cpp` (Planned)**: MoE 10-expert weighted reduction bandwidth comparing atomic adds vs hierarchical subgroup reductions.
5. **`perf_dequant_dpas.cpp` (Planned)**: Weight-only streaming dequantization (Q2_K / IQ4_XS) feeding XMX DPAS instructions.

---

## 4. In Progress: Hardware XMX Systolic Attention (`qsa_prompt_attn.cpp`)

- **Diagnosis**: The existing `qsa_prompt_attn.cpp` kernel was producing 100% NaNs because `mma16816` emulated a 32-lane CUDA warp using `sycl::select_from_group` with lane indices up to 31, which is illegal/out-of-bounds on Intel's SIMD16 execution units.
- **Parity Test Fix**: Updated `src/kernels/sycl/qsa_prompt_attn_parity.cpp` to explicitly check for `NaN` and `Inf` in outputs, ensuring parity failures are properly detected.
- **Hardware Validation**: Tested native XMX systolic matrix execution on the live GPU (`0x64a0`):
  - $16 \times 16 \times 16$ tile multiplication using `joint_matrix<sub_group, half, use::a, 16, 16, layout::row_major>` and `use::b, 16, 16, layout::col_major` loaded directly from SLM (`local_space`).
  - Verified FP16 and INT8 FlashAttention pipelines on synthetic context: 0 NaNs, matches double-precision reference within $3 \times 10^{-4}$.
- **Next Step**: Port `src/kernels/sycl/qsa_prompt_attn.cpp` to this native XMX systolic kernel implementation and verify with `qsa_prompt_attn_parity`.

---

## 5. Development & Profiling Commands

```bash
# Source Intel oneAPI 2026.1 environment
source /opt/intel/oneapi/setvars.sh --force 2>/dev/null

# Compile and run unit tests (fast CI)
make -C build_sycl -j$(nproc) check-sycl

# Configure and run the performance benchmark suite
cmake -B build_sycl -DSTRATA_BUILD_PERF=ON
make -C build_sycl perf-sycl

# Filter specific microbenchmarks
./build_sycl/perf_xe2_gemv --benchmark_filter='^BM_Xe2_GEMV/'
```

---

## 6. Recommended Prompts for the Next Session

### Option 1 (Extend Perf Tests): Implement `perf_qsa_attention.cpp`
> *"Create `perf/sycl/perf_qsa_attention.cpp` using Google Benchmark to sweep QSA FlashAttention throughput and Top-K sparse gather bandwidth across prompt lengths (512 to 32768), add it to CMakeLists.txt, and benchmark on the Intel GPU."*

### Option 2: Complete XMX Systolic Attention Kernel
> *"Implement and verify the Intel Xe2 hardware XMX systolic FlashAttention kernel in `src/kernels/sycl/qsa_prompt_attn.cpp` to replace `mma16816`, and make sure `qsa_prompt_attn_parity` and `check-sycl` pass on the GPU."*

### Option 3: Weight-Only Dequantization + XMX DPAS Microbenchmark
> *"Implement `perf/sycl/perf_dequant_dpas.cpp` to benchmark streaming 2-bit/4-bit weight dequantization directly feeding XMX DPAS instructions on the gate projection `[2560, 6144]`."*

