# Intel Arc B580 (Battlemage Xe2) Optimization Plan for Strata

> **Target Hardware**: Intel Arc B580 Desktop GPU  
> **Architecture**: Intel Xe2-HPG ("Battlemage")  
> **Die**: BMG-G21 | 20 Xe2-cores (160 Vector Engines, 160 XMX Matrix Engines)  
> **VRAM**: 12 GB GDDR6 @ 19 Gbps on 192-bit bus (456 GB/s bandwidth)  
> **Caches**: 16 MB L2/L3 Adamantine cache  
> **Peak Compute**: ~117 TFLOPS (FP16/BF16 XMX), ~233 TOPS (INT8 XMX)  
> **Compiler / Toolchain**: Intel oneAPI DPC++ 2025.3 (`icpx -fsycl`)  
> **Target Status**: Codebase fully ported to pure SYCL 2020 (0 DPCT dependencies)  

---

## 1. Executive Summary & Motivation

The Strata inference engine has achieved a complete, warning-free migration from CUDA/DPCT to **pure standard SYCL 2020** across all 44 computational kernels, the command graph execution engine, the core memory runtime, and the prefill pipeline.

With all legacy DPCT wrappers eliminated, Strata is now in the ideal state to exploit the **architectural breakthroughs of Intel's Xe2 Battlemage microarchitecture**, specifically the **Intel Arc B580**:
1. **Xe2 SIMD16 Native Execution**: Unlike Xe1 (which used dual-issue SIMD8) and NVIDIA (which uses 32-lane warps), Xe2 execution units execute natively in **16-wide SIMD**. Tuning kernels to SIMD16 doubles the effective per-thread register budget, eliminating register spilling in large kernels.
2. **Xe Matrix Extensions (XMX)**: 160 hardware systolic matrix engines provide massive throughput for GEMV and prompt attention compared to vector ALUs.
3. **64 KB Unified SLM & 16-Bank Structure**: Shared Local Memory (SLM) throughput can be maximized by avoiding 16-bank stride collisions.
4. **Ahead-of-Time (AOT) Binary Compilation**: Direct machine-code generation for `bmg` eliminates runtime JIT compilation overhead and driver stutter.

---

## 2. Architectural Comparison: Xe1 vs. Xe2 (B580) vs. NVIDIA Ampere/Ada

| Feature | NVIDIA Ada (e.g. RTX 4070) | Intel Xe1 (Arc A770) | Intel Xe2 (Arc B580) | Strata Opportunity |
|---|---|---|---|---|
| **Warp / Subgroup Native Width** | 32 (Warp) | 8 (Dual-issue) or 16 | **16 (Native SIMD16)** | Change `reqd_sub_group_size(32)` to `16` for lower register pressure |
| **Matrix Hardware** | Tensor Cores (MMA) | XMX (Xe1) | **XMX 2.0 (Xe2)** | Accelerate BF16 GEMV & QSA prompt attention with `joint_matrix` |
| **Native Quantized Math** | INT8 / FP8 / INT4 | INT8 / BF16 / FP16 | **INT2 / INT4 / INT8 / FP8 / BF16 / FP16** | Native hardware execution for Q4_0, Q4_K, IQ4_XS, and S2 |
| **Shared Local Memory (SLM)** | 100 KB configurable | 64 KB per Xe-core | **64 KB per Xe-core** | Tile sizes aligned with 64 KB boundaries |
| **Memory Bandwidth** | 504 GB/s | 560 GB/s | **456 GB/s** | Memory-bound decode requires strict cache-line coalescing (64B/128B) |

---

## 3. Six-Phase Optimization Strategy

```mermaid
flowchart TD
    P1["Phase 1: Ahead-Of-Time (AOT) BMG Compilation"] --> P2["Phase 2: Sub-group Size Tuning (SIMD16 vs SIMD32)"]
    P2 --> P3["Phase 3: Intel XMX Matrix Systolic Acceleration (joint_matrix)"]
    P3 --> P4["Phase 4: SLM Alignment & Bank Conflict Avoidance"]
    P4 --> P5["Phase 5: OneMKL Xe2 Dispatch for Prefill GEMM"]
    P5 --> P6["Phase 6: End-to-End Inference Benchmarking & TTFT Profiling"]
    style P1 fill:#d4edda,stroke:#28a745,color:#155724
    style P2 fill:#d4edda,stroke:#28a745,color:#155724
    style P3 fill:#cce5ff,stroke:#004085,color:#004085
    style P4 fill:#cce5ff,stroke:#004085,color:#004085
    style P5 fill:#fff3cd,stroke:#856404,color:#856404
    style P6 fill:#d4edda,stroke:#28a745,color:#155724
```

---

### Phase 1: Ahead-of-Time (AOT) Compilation for Battlemage (`bmg`)

**Objective**: Eliminate JIT compilation pauses on GPU startup and enable machine-code optimizations specifically for the Xe2 instruction set architecture.

- **Current State**: `CMakeLists.txt` builds generic SPIR-V bytecode via `icpx -fsycl`. The Intel OpenCL/Level Zero driver compiles SPIR-V to GPU ISA at runtime upon the first kernel invocation or graph capture.
- **Battlemage AOT Flag**:
  ```cmake
  # In CMakeLists.txt:
  option(STRATA_AOT_BMG "Compile Ahead-of-Time directly for Intel Arc Battlemage (B580)" OFF)
  if(STRATA_AOT_BMG)
    add_compile_options(-fsycl-targets=spir64_gen -Xs "-device bmg")
  endif()
  ```
- **Actions**:
  1. Add `STRATA_AOT_BMG` CMake option to `CMakeLists.txt`.
  2. Verify binary generation with `strata-device` and check that Level Zero reports pre-compiled device modules.

---

### Phase 2: Native SIMD16 Sub-group Execution Tuning

**Objective**: Align subgroup reductions and register allocations with the native 16-wide execution units of the Xe2 architecture.

- **Current State**: Several kernels specify `[[sycl::reqd_sub_group_size(32)]]` to emulate CUDA 32-lane warps. On Xe2, a 32-thread subgroup is split across two hardware execution cycles or runs with doubled register demand.
- **Optimization**:
  1. Define architecture-aware subgroup sizing:
     ```cpp
     // include/strata/core/sycl_arch.hpp
     #pragma once
     #include <sycl/sycl.hpp>

     namespace strata::sycl_arch {
     #if defined(STRATA_TARGET_BMG) || defined(__INTEL_XE2__)
     constexpr int SUB_GROUP_SIZE = 16;
     #else
     constexpr int SUB_GROUP_SIZE = 32;
     #endif
     }
     ```
  2. Update reduction loops from `for (int off = 16; off > 0; off >>= 1)` to dynamic subgroup tree reductions (`sycl::reduce_over_group(item.get_sub_group(), val, sycl::plus<float>())`). Standard SYCL group algorithms automatically map to optimal single-cycle hardware reduction primitives on Xe2.
  3. **High-Impact Kernels**:
     - `src/kernels/sycl/bf16_gemv.cpp`: Warp kernel -> Subgroup SIMD16 reduction.
     - `src/kernels/sycl/router_top10.cpp`: Top-10 selection with 16-wide registers per wave.
     - `src/kernels/sycl/fused_gr.cpp`: Hyper-connection reduction across streams.

---

### Phase 3: Hardware Matrix Acceleration via Intel XMX (`joint_matrix`)

**Objective**: Leverage the 160 Xe Matrix Extensions (XMX) systolic engines to accelerate matrix-vector multiplications and attention calculations.

- **Architecture Details**:
  - Each Xe2 core contains 8 XMX engines.
  - XMX systolic array multiplies a $M \times K$ matrix by a $K \times N$ matrix every clock cycle.
  - Native support for BF16, FP16, and INT8.
- **SYCL API**: Standard `sycl::ext::oneapi::experimental::matrix::joint_matrix`.
- **Target Kernels for XMX**:
  1. **`bf16_gemv.cpp` (BF16 GEMV)**:
     - Map GEMV with batch $> 1$ or tiled GEMV ($16 \times 16$) to `joint_matrix<sub_group, sycl::ext::oneapi::bfloat16, ...>`.
     - 117 TFLOPS theoretical peak vs ~15 TFLOPS vector ALU peak.
  2. **`qsa_prompt_attn.cpp` (QSA Prompt Attention)**:
     - Prompt attention accumulates queries against keys ($Q \cdot K^T$) and attention against values ($P \cdot V$).
     - The CUDA implementation uses tensor-core MMAs (`m16n8k16`).
     - Map directly to `joint_matrix<sub_group, sycl::half, ...>` for FP16 and INT8 KV pools.
  3. **`s2_expert_grouped.cpp` (MoE Expert Grouped GEMV)**:
     - In MoE token generation, 10 experts are dispatched in parallel. Grouping tokens into XMX matrix tiles gives near-linear scaling.

---

### Phase 4: Shared Local Memory (SLM) & Cache Line Alignment

**Objective**: Maximize memory bandwidth on B580's 456 GB/s bus and eliminate SLM bank conflicts.

- **Xe2 Memory Specifics**:
  - 64 KB SLM per Xe-core divided into 16 banks (4 bytes per bank).
  - Memory requests are served in 64-byte or 128-byte cache lines.
- **Optimization Rules**:
  1. **Bank Conflict Avoidance**: Ensure consecutive work-items in a SIMD16 subgroup access consecutive 4-byte words:
     $$\text{Address Offset} = \text{lane} \times 4 \quad (\text{mod } 64)$$
     Avoid strides of 16 or 32 floats, which map all lanes to the same SLM bank.
  2. **Cache-Line Coalescing**:
     - Ensure USM allocations for weights and activations are aligned to 64 bytes (`alignas(64)` or aligned USM allocation).
     - Standardize strided loads in `iq_kernels.cpp` and `kv_stream.cpp` to 64-byte aligned blocks.

---

### Phase 5: OneMKL Xe2 Dispatch for Prefill GEMM

**Objective**: Ensure prompt prefilling automatically routes large matrices to B580 XMX engines via optimized Intel OneMKL primitives.

- **Current Implementation**: [`src/prefill/sycl/gemm.cpp`](file:///home/girtam/projects/Strata/src/prefill/sycl/gemm.cpp) uses `oneapi::mkl::blas::column_major::gemm`.
- **Optimization**:
  - When compiled with `-Xs "-device bmg"`, OneMKL selects Xe2-specialized matrix kernels optimized for Adamantine L3 cache tiling.
  - Verify that workspace allocations use `sycl::malloc_device` and queue is configured with `sycl::property::queue::in_order()`.

---

### Phase 6: End-to-End Validation & Execution Protocol

**Validation Steps for Next Session**:
1. **Tool Verification**:
   ```bash
   ./build_sycl/strata-device
   ```
   Verify device name `Intel(R) Arc(TM) B580 Graphics` or `Intel(R) Graphics [0x...], Xe2 Architecture`.
2. **Baseline Micro-Benchmarks**:
   ```bash
   ./build_sycl/qsa_select_bench 32768 256 3 262144
   ./build_sycl/iq_parity build_sycl/iq_fixture
   ./build_sycl/iq_multi_parity build_sycl/iq_fixture
   ```
3. **Dedicated SYCL CI Suite**:
   ```bash
   make -C build_sycl check-sycl
   ```
4. **Token Generation Throughput Benchmark**:
   ```bash
   ./build_sycl/strata --tokens 128 --no-capture
   ./build_sycl/strata --tokens 128
   ```
   Measure tokens per second (`tok/s`) and GPU execution overhead.

---

| Priority | Task | Target File | Status | Impact |
|---|---|---|---|---|
| **P1** | Add Battlemage & Xe2 CMake flags (`STRATA_TARGET_XE2`, `STRATA_AOT_BMG`) | [`CMakeLists.txt`](file:///home/girtam/projects/Strata/CMakeLists.txt) | **COMPLETED** | Enables Xe2 ISA flags and optional direct `bmg` AOT compilation |
| **P2** | Define architecture-aware subgroup sizing (`sycl_arch.hpp`) | [`include/strata/core/sycl_arch.hpp`](file:///home/girtam/projects/Strata/include/strata/core/sycl_arch.hpp) | **COMPLETED** | Sets native SIMD16 execution for Xe2 (B580/Lunar Lake); cuts register pressure |
| **P3** | Port GEMV & Norm kernels to SIMD16 + `sycl::reduce_over_group` | [`src/kernels/sycl/bf16_gemv.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/bf16_gemv.cpp), [`src/kernels/sycl/native_bf16.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/native_bf16.cpp), [`src/kernels/sycl/native_gr_norm.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/native_gr_norm.cpp) | **COMPLETED** | 100% test pass on live Xe2 GPU; direct hardware reduction primitives |
| **P4** | Implement Joint Matrix (XMX) prompt attention | [`src/kernels/sycl/qsa_prompt_attn.cpp`](file:///home/girtam/projects/Strata/src/kernels/sycl/qsa_prompt_attn.cpp) | **IN PROGRESS** | Probed 53 XMX combos; verified `joint_matrix` multiply-accumulate on live Xe2 |
| **P5** | Profile and benchmark with Intel VTune / Advisor | Metrics collection | Upcoming | Identify memory vs compute bottlenecks on Xe2 |

---

## 5. Live Xe2 Hardware Test & Validation (Core Ultra 5 226V / Lunar Lake)

The test machine's mobile GPU (`Intel(R) Graphics [0x64a0]`, Lunar Lake Core Ultra 5 226V) shares the exact **Xe2 microarchitecture generation** (Xe2-LPG) as the target Intel Arc B580 (Xe2-HPG):
1. **Native SIMD16 Execution**:
   - `sub_group_sizes`: `16 32` reported by the Level Zero driver.
   - Kernels executing with `[[sycl::reqd_sub_group_size(16)]]` and `sycl::reduce_over_group` run natively on hardware vector engines.
   - Tested and verified with 0 failures on `bf16_gemv_parity`, `sycl_native_bf16`, `gr_parity`, and the 13-test `check-sycl` CI suite.
2. **Xe Matrix Extensions (XMX)**:
   - Probed via Level Zero runtime: **53 matrix combinations** supported directly on hardware (including `m=16, n=16, k=16` FP16/BF16 and `m=8, n=16, k=32` INT8).
   - Validated stand-alone `joint_matrix` execution via SYCL unified matrix APIs (`joint_matrix_fill`, `joint_matrix_load`, `joint_matrix_mad`, `joint_matrix_store`).

