# Handover Document: Intel Arc B580 / Xe2 SYCL Optimization & Upstream Rebase

**Date & Time**: 2026-10-04 23:10 (Local)  
**Project**: Strata Engine (`/home/girtam/projects/Strata`)  
**Current Branch**: `main` (cleanly rebased on `origin/main` @ commit `41d50fd`)  
**Backup Branch**: `backup-main-before-rebase` (points to pre-rebase HEAD `6e28497`, preserves full 84-commit history)

---

## 1. Executive Summary

This project optimizes Strata's native SYCL backend for the **Intel Arc B580 (Xe2-HPG Battlemage)** discrete GPU, validated on the available **Intel Core Ultra 5 226V (Lunar Lake Xe2-LPG, Device ID `0x64a0`)** which shares the exact same Intel Xe2 execution engine (native SIMD16 execution vector engines, XMX 2.0 systolic matrix engines, 64 KB SLM).

The target model for these optimizations is **Qwen3.8-Flash-Next** (as well as `UD-Q4_K_XL`).

---

## 2. Current Status & Git Topology

### Rebase on Upstream `origin/main` Completed
- Upstream tag: `v0.1.39` (commit `6f32ec0`).
- Local `main` is now rebased cleanly on top of `origin/main` at commit `41d50fd` (*"Port Strata to pure SYCL 2020 with Intel Xe2 (Lunar Lake / Battlemage B580) optimizations"*).
- Resolved merge conflicts cleanly:
  1. `.gitignore`: Preserved upstream additions (`/.ccache/`, `/.rocm-win/`) alongside migration entries (`*_out/`, `scratch/`, `MainSourceFiles.yaml`).
  2. `CMakeLists.txt`:
     - Retained all upstream additions (including AMD `STRATA_HIP_GFX906`, new CUDA parity tests `qsa_topk_active_parity`, `gdn_rec_parity`, `gemm_bf16_parity`, `native_grouped_parity`, `fused_gr_bench`, `route_window_parity`, `qsa_topk_parity`, `native_expert_bench`).
     - Renamed upstream's experimental `sycl/` subdirectory option from `STRATA_ENABLE_SYCL` to `STRATA_ENABLE_SYCL_SUBDIR` to prevent it from intercepting our first-class unified SYCL backend.
     - Preserved our native SYCL flags (`STRATA_TARGET_XE2`, `STRATA_AOT_BMG`), targets (`strata_core`, `strata_kernels`), test executables, and the unified `check-sycl` target.
- Pre-rebase backup: `backup-main-before-rebase` safely tracks the entire 84-commit granular migration history.

---

## 3. Optimizations Implemented & Verified on Live Xe2 GPU

All kernels and tests compile with `icpx` and pass on the live Lunar Lake Xe2 GPU (`0x64a0`):

1. **Architecture Abstraction Layer** ([`include/strata/core/sycl_arch.hpp`](file:///home/girtam/projects/Strata/include/strata/core/sycl_arch.hpp)):
   - Configured `SUB_GROUP_SIZE = 16` on Intel Xe2 / Battlemage / Lunar Lake (`STRATA_TARGET_XE2`), falling back to 32 on legacy architectures.
   - Implemented `subgroup_reduce_add()` and `subgroup_reduce_max()` using native hardware `sycl::reduce_over_group`.
2. **BF16 GEMV Optimization** ([`src/kernels/sycl/bf16_gemv.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/bf16_gemv.cpp)):
   - Replaced emulated 32-lane warp shuffle loops with native `[[sycl::reqd_sub_group_size(16)]]` and `sycl::reduce_over_group`.
3. **Native BF16 MMVF Optimization** ([`src/kernels/sycl/native_bf16.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/native_bf16.cpp)):
   - Migrated single-token and multi-token matrix-vector multiplications to SIMD16 subgroup reductions.
4. **Weighted RMSNorm Optimization** ([`src/kernels/sycl/native_gr_norm.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/native_gr_norm.cpp)):
   - Replaced multi-step warp shuffle reduction loops with SIMD16 + `sycl::reduce_over_group` block reduction.
5. **Standalone Unit Tests** ([`tests/sycl/`](file:///home/girtam/projects/Strata/tests/sycl/)):
   - [`tests/sycl/native_gr_norm_test.cpp`](file:///home/girtam/projects/Strata/tests/sycl/native_gr_norm_test.cpp): Validates RMSNorm across dimensions `320`, `2560`, `10240` and batch sizes `1..16` with zero failures.
   - [`tests/sycl/xe2_subgroup_gemv_test.cpp`](file:///home/girtam/projects/Strata/tests/sycl/xe2_subgroup_gemv_test.cpp): Validates SIMD16 BF16 GEMV against CPU float reference across all 8 model projection shapes (zero failures).
   - [`tests/sycl/xmx_joint_matrix_test.cpp`](file:///home/girtam/projects/Strata/tests/sycl/xmx_joint_matrix_test.cpp): Validates Intel XMX systolic execution via SYCL `joint_matrix` ($8 \times 16 \times 16$ BF16/FP32 single-tile and $32 \times 64 \times 64$ tiled). Zero failures.

---

## 4. Qwen3.8-Flash-Next: Model Operations and Dimensions

For kernel dimension tuning and verification:

| Model Parameter | Dimension / Value | Architectural Role |
| :--- | :--- | :--- |
| **Hidden Size ($d_{model}$)** | **2560** | Base hidden activation width |
| **Intermediate Size ($d_{ffn}$)** | **10240** (or 6400 in dense) | MoE Expert feed-forward width |
| **Number of Experts** | **256** routed + **1** shared | Sparse MoE routing |
| **Active Experts per Token ($k$)** | **8** | Top-K routing per token |
| **Attention Heads ($N_q$)** | **20** heads ($d_{head} = 128$) | Query projection width 2560 |
| **KV Heads ($N_{kv}$)** | **4** heads ($d_{head} = 128$) | GQA Grouped Query Attention (5:1) |
| **QSA Routing Dimension** | **320** ($10240 \times 320$) | Gated attention selection |
| **Router Input / Output** | **$[2560 \times 256]$** | Router Top-K logit calculation |

### Key GEMM / GEMV Shapes in the Engine
- Attention Q Projection: `[2560, 2560]`
- Attention K Projection: `[2560, 512]` ($4 \times 128$)
- Attention V Projection: `[2560, 512]` ($4 \times 128$)
- Attention Output: `[2560, 2560]`
- MoE Router: `[2560, 256]`
- MoE Gate & Up Projections: `[2560, 10240]` per active expert
- MoE Down Projection: `[10240, 2560]` per active expert
- PLE Projections: `[2560, 48]`, `[2560, 128]`, `[2560, 640]`, `[640, 2560]`

---

## 5. Development Environment & Build Instructions

### oneAPI Compiler Sourcing
The system has Intel oneAPI `2026.1` installed at `/opt/intel/oneapi/compiler/latest`.
```bash
# Source oneAPI without mixing older library paths
source /opt/intel/oneapi/setvars.sh --force 2>/dev/null
```
> [!WARNING]
> Do NOT set `LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH`. Prepending `2025.3` causes symbol mismatch (`urGraphInstantiateGraphExp@LIBUR_LOADER_0.12`) with the 2026.1 runtime. Plain `setvars.sh` correctly sets `2026.1`.

### CMake Configuration
The build folder `build_sycl` is already configured for the rebased tree:
```bash
cmake -B build_sycl \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_C_COMPILER=icx \
  -DSTRATA_ENABLE_SYCL=ON \
  -DSTRATA_TARGET_XE2=ON \
  -DSTRATA_BUILD_TESTS=ON
```

### Running the Test Suite
```bash
make -C build_sycl -j$(nproc) check-sycl
```
Or run individual standalone tests directly:
```bash
./build_sycl/sycl_native_gr_norm
./build_sycl/sycl_xe2_subgroup_gemv
./build_sycl/sycl_xmx_joint_matrix
```

---

## 6. Action Items for the Next Session

1. **Verify the Rebased Build**:
   - Run `make -C build_sycl -j$(nproc) check-sycl` to compile all targets and run the test suite on the GPU.
2. **Phase 3: Hardware XMX Systolic Attention / GEMV**:
   - Port [`src/kernels/sycl/qsa_prompt_attn.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/qsa_prompt_attn.cpp) from software shuffle loops (`mma16816`) to hardware systolic `joint_matrix` operations demonstrated in [`tests/sycl/xmx_joint_matrix_test.cpp`](file:///home/girtam/projects/Strata/tests/sycl/xmx_joint_matrix_test.cpp).
3. **SIMD16 Optimization for Remaining Quantized Kernels**:
   - Apply SIMD16 subgroup reductions to [`src/kernels/sycl/s2_gemv.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/s2_gemv.cpp), [`src/kernels/sycl/s2_gemv_fast.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/s2_gemv_fast.cpp), and [`src/kernels/sycl/s_gemv.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/s_gemv.cpp).
