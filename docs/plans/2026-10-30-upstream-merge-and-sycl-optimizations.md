# Upstream v0.2.8 Merge & SYCL Optimization Port Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the merge of upstream v0.2.8 changes, adopt all upstream SYCL optimizations into the pure SYCL Xe2 engine (`src/kernels/sycl/`, `src/core/sycl/`, `src/prefill/sycl/`), port upstream CUDA wide-window (16-row) optimizations to SYCL, add required tests, verify with build and test scripts, and push to branch `sycl-b580`.

**Architecture:** Pure SYCL 2020 backend targeting Intel Arc Xe2 (Battlemage B580 / Lunar Lake) with subgroup reductions, XMX DPAS, adaptive queue dispatch, and wide-window capability.

**Tech Stack:** C++20, Intel oneAPI DPC++ (icpx), SYCL 2020, Level Zero, Google Benchmark, CTest.

---

### Task 1: Adopt Upstream SYCL Optimizations in `qsa_decode_attn.cpp` and `prefill.cpp`

**Files:**
- Modify: `src/kernels/sycl/qsa_decode_attn.cpp`
- Modify: `src/prefill/sycl/prefill.cpp`

**Step 1: Inspect and update `lane_cell_attn` in `src/kernels/sycl/qsa_decode_attn.cpp`**
- In `src/kernels/sycl/qsa_decode_attn.cpp`:
  Update `lane_cell_attn` to accept `int64_t n_q` and `sycl::queue* st`.
  When `STRATA_ATTN_LANECELL` is not explicitly set, auto-enable if `n_q >= 16` and device architecture is `IntelGpuGen::Battlemage`.

**Step 2: Update `src/prefill/sycl/prefill.cpp` with dynamic query batching & grouping table copy**
- Implement `attn_batch_setting()` and `attn_batch_for(size_t T)` respecting `STRATA_ATTN_BATCH` (default 32).
- Update `bytes_needed_impl` and `qsa_set_bytes` to use `attn_batch_for(T)` ensuring allocation and memory budgeting match.
- Set `STRATA_GROUP_COPY` default to ON (`gc == nullptr || gc[0] != '0'`) for optimal PCIe throughput.

---

### Task 2: Port Upstream CUDA Optimizations to SYCL (Wide Window & Slicing)

**Files:**
- Modify: `src/kernels/sycl/fused_gr.cpp`
- Modify: `src/kernels/sycl/verify_kernels.cpp`
- Modify: `src/kernels/sycl/native_mmvq.cpp`
- Modify: `src/kernels/sycl/s2_expert_grouped.cpp`
- Modify: `src/kernels/sycl/native_bf16.cpp`
- Modify: `src/core/sycl/verify.cpp`

**Step 1: Port `fused_gr_read_multi` slicing in `src/kernels/sycl/fused_gr.cpp`**
- Support `n_tok > kFusedGrMaxT` (up to `kFusedGrWideT = 16`) by processing in slices of `kFusedGrMaxT = 8`.

**Step 2: Port `resident_plan` wide-window handling in `src/kernels/sycl/verify_kernels.cpp`**
- When `n_entries > 128` (up to 256 entries), launch a 256-work-item workgroup with sized local memory accessors.

**Step 3: Port `native_mmvq` column slicing in `src/kernels/sycl/native_mmvq.cpp`**
- If `ncols > MAX_NCOLS` (8), execute consecutive `MAX_NCOLS`-column passes over the weights with correct activation/output offsets.

**Step 4: Implement `moe_grouped_s2_wide` in `src/kernels/sycl/s2_expert_grouped.cpp`**
- Implement `moe_grouped_s2_wide` with `SKIP = GMAX` (8) pass when `max_group > 8`.

**Step 5: Fix profiling stream in `src/core/sycl/verify.cpp`**
- Ensure stamp readbacks use `cs_->memcpy(...)` on the verifier's own stream rather than the default queue.

---

### Task 3: Integrate `pipeline_auto` and Unit Tests

**Files:**
- Create: `include/strata/pipeline_auto.hpp`
- Modify: `CMakeLists.txt`
- Modify: `src/program/sycl/generate.cpp`

**Step 1: Add `include/strata/pipeline_auto.hpp`**
- Header defining `PipelineAutoInputs` and `pipeline_auto_on(...)`.

**Step 2: Register `pipeline_auto_test` and `window_limits_test` in root `CMakeLists.txt`**
- Add test executables to root `CMakeLists.txt` so they run in `./tools/test_sycl.sh`.

**Step 3: Wire automatic pipeline selection in `src/program/sycl/generate.cpp`**
- Automatically select `pipeline_windows = 2` when running 2-stage split on two Battlemage cards.

---

### Task 4: Compilation, Full Test Suite & Validation

**Step 1: Compile Strata targets and tests**
- Run `./tools/build_sycl.sh strata`
- Run `./tools/build_sycl.sh -r`

**Step 2: Run CTest test suite**
- Run `./tools/test_sycl.sh` and ensure 100% test pass rate.
- Run standalone parity tests.

---

### Task 5: Commit and Push to Branch

**Step 1: Stage and commit changes with conventional commit format**
- `feat(sycl): adopt upstream v0.2.8 SYCL optimizations and port CUDA wide-window enhancements`

**Step 2: Push to remote branch**
- Push to `origin sycl-b580`.
