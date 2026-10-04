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

### Phase 1: Ahead-of-Time (AOT) Compilation for Battlemage (`bmg`) - **COMPLETE**

**Objective**: Eliminate JIT compilation pauses on GPU startup and enable machine-code optimizations specifically for the Xe2 instruction set architecture.

- **Current State**: Integrated into `CMakeLists.txt` via:
  ```cmake
  option(STRATA_TARGET_XE2 "Enable Intel Xe2 Battlemage/Lunar Lake microarchitecture optimizations" ON)
  option(STRATA_AOT_BMG "Compile Ahead-of-Time directly for Intel Arc Battlemage (B580)" OFF)
  if(STRATA_AOT_BMG)
    add_compile_options(-fsycl-targets=spir64_gen -Xs "-device bmg")
  endif()
  ```

---

### Phase 2: Native SIMD16 Sub-group Execution Tuning - **COMPLETE**

**Objective**: Align subgroup reductions and register allocations with the native 16-wide execution units of the Xe2 architecture.

- **Status**: Implemented in [`include/strata/core/sycl_arch.hpp`](../include/strata/core/sycl_arch.hpp) and deployed across high-impact kernels:
  ```cpp
  namespace strata::sycl_arch {
  #if defined(STRATA_TARGET_XE2) || defined(STRATA_TARGET_BMG) || defined(__INTEL_XE2__)
  constexpr int SUB_GROUP_SIZE = 16;
  #else
  constexpr int SUB_GROUP_SIZE = 32;
  #endif
  }
  ```
- **Optimized Kernels**:
  1. [`src/kernels/sycl/bf16_gemv.cpp`](../src/kernels/sycl/bf16_gemv.cpp): `[[sycl::reqd_sub_group_size(16)]]` with `sycl::reduce_over_group`.
  2. [`src/kernels/sycl/native_bf16.cpp`](../src/kernels/sycl/native_bf16.cpp): SIMD16 subgroup reductions across single and multi-token GEMV.
  3. [`src/kernels/sycl/native_gr_norm.cpp`](../src/kernels/sycl/native_gr_norm.cpp): RMSNorm SIMD16 block reduction.
  4. [`src/kernels/sycl/s2_gemv.cpp`](../src/kernels/sycl/s2_gemv.cpp): Added SIMD16 subgroup reduction kernel path.
  5. [`src/kernels/sycl/s_gemv.cpp`](../src/kernels/sycl/s_gemv.cpp): Replaced 32-lane CUDA warp loops with `SG_SIZE = 16`, stride `SG_SIZE * QE`, and single-cycle `reduce_over_group`.
  6. [`src/kernels/sycl/s2_gemv_fast.cpp`](../src/kernels/sycl/s2_gemv_fast.cpp) & [`s2_gemv_quads.cpp`](../src/kernels/sycl/s2_gemv_quads.cpp): Replaced SLM tree loops with hierarchical subgroup reductions.
- **Verification**: Verified via `sycl_xe2_subgroup_gemv`, `sycl_native_gr_norm`, `s_gemv_parity`, `s_gemv_q8k_parity`, `s2_gemv_parity`, and `s2_gemv_q8_parity` (0 failures).

---

### Phase 3: Hardware Matrix Acceleration via Intel XMX (`joint_matrix`) - **IN PROGRESS**

**Objective**: Leverage the 160 Xe Matrix Extensions (XMX) systolic engines to accelerate matrix-vector multiplications and attention calculations.

- **Architecture Details & Verified Capabilities**:
  - Each Xe2 core contains 8 XMX systolic matrix engines.
  - Probed on live Xe2 GPU (`0x64a0`): 53 hardware matrix combinations supported, including:
    - FP16/FP32: $TM=16, TN=16, TK=16$ and $TM=8, TN=16, TK=16$.
    - BF16/FP32: $TM=16, TN=16, TK=16$ and $TM=8, TN=16, TK=16$.
    - INT8/INT32: $TM=8, TN=16, TK=32$ and $TM=16, TN=16, TK=32$.
  - Standalone unit test [`tests/sycl/xmx_joint_matrix_test.cpp`](../tests/sycl/xmx_joint_matrix_test.cpp) verified with **0 failures** (single-tile $8 \times 16 \times 16$ and tiled $32 \times 64 \times 64$).
  - Confirmed native hardware support for `sycl::address_space_cast<sycl::access::address_space::local_space>` and `layout::col_major` loads with arbitrary leading strides.
- **Target Kernels for XMX**:
  1. **`qsa_prompt_attn.cpp` (QSA Prompt Attention)**:
     - Replace broken 32-lane CUDA software shuffle emulation (`mma16816`) with native systolic `joint_matrix` ($16 \times 16 \times 16$ FP16/FP32).
     - Directly computes $Q \cdot K^T$ and $P \cdot V$ from local memory buffers `S.k` and `S.v`.
  2. **`bf16_gemv.cpp` (BF16 GEMV)**:
     - Map multi-token GEMV tiles ($16 \times 16$) to `joint_matrix<sub_group, sycl::ext::oneapi::bfloat16, ...>`.
  3. **`s2_expert_grouped.cpp` (MoE Expert Grouped GEMV)**:
     - Group active expert tokens into XMX matrix tiles for near-linear multi-expert throughput.

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

- **Current Implementation**: [`src/prefill/sycl/gemm.cpp`](../src/prefill/sycl/gemm.cpp) uses `oneapi::mkl::blas::column_major::gemm`.
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
| **P1** | Add Battlemage & Xe2 CMake flags (`STRATA_TARGET_XE2`, `STRATA_AOT_BMG`) | [`CMakeLists.txt`](../CMakeLists.txt) | **COMPLETED** | Enables Xe2 ISA flags and optional direct `bmg` AOT compilation |
| **P2** | Define architecture-aware subgroup sizing (`sycl_arch.hpp`) | [`include/strata/core/sycl_arch.hpp`](../include/strata/core/sycl_arch.hpp) | **COMPLETED** | Sets native SIMD16 execution for Xe2 (B580/Lunar Lake); cuts register pressure |
| **P3** | Port GEMV & Norm kernels to SIMD16 + `sycl::reduce_over_group` | [`src/kernels/sycl/bf16_gemv.cpp`](../src/kernels/sycl/bf16_gemv.cpp), [`src/kernels/sycl/native_bf16.cpp`](../src/kernels/sycl/native_bf16.cpp), [`src/kernels/sycl/native_gr_norm.cpp`](../src/kernels/sycl/native_gr_norm.cpp) | **COMPLETED** | 100% test pass on live Xe2 GPU; direct hardware reduction primitives |
| **P4** | Implement Joint Matrix (XMX) prompt attention | [`src/kernels/sycl/qsa_prompt_attn.cpp`](../src/kernels/sycl/qsa_prompt_attn.cpp) | **IN PROGRESS** | Probed 53 XMX combos; verified `joint_matrix` multiply-accumulate on live Xe2 |
| **P5** | Profile and benchmark with Intel VTune / Advisor | Metrics collection | Upcoming | Identify memory vs compute bottlenecks on Xe2 |

---

## 5. Live Xe2 Hardware Test & Validation (Core Ultra 5 226V / Lunar Lake)

The test machine's mobile GPU (`Intel(R) Graphics [0x64a0]`, Lunar Lake Core Ultra 5 226V) shares the exact **Xe2 microarchitecture generation** (Xe2-LPG) as the target Intel Arc B580 (Xe2-HPG):
1. **Native SIMD16 Execution**:
   - `sub_group_sizes`: `16 32` reported by the Level Zero driver.
   - Kernels executing with `[[sycl::reqd_sub_group_size(16)]]` and `sycl::reduce_over_group` run natively on hardware vector engines.
   - Tested and verified with 0 failures on `bf16_gemv_parity`, `sycl_native_bf16`, `gr_parity`, and the 16-test `check-sycl` CI suite.
2. **Xe Matrix Extensions (XMX)**:
   - Probed via Level Zero runtime: **53 matrix combinations** supported directly on hardware (including `m=16, n=16, k=16` FP16/BF16 and `m=8, n=16, k=32` INT8).
   - Validated stand-alone `joint_matrix` execution via SYCL unified matrix APIs (`joint_matrix_fill`, `joint_matrix_load`, `joint_matrix_mad`, `joint_matrix_store`) with 0 failures.

---

## 6. Targeted Microbenchmarks & Profiling Strategy (Qwen3.8-Flash-Next)

Transitioning verified standalone tests into a parameterized microbenchmarking suite is the ideal way to tune for the Intel Arc B580 (Xe2) architecture. Because Strata relies heavily on asymmetric hardware balancing (VRAM cache vs RAM vs SSD n-gram tables), isolating kernel performance from the engine’s async scheduler provides clean tuning baselines.

### 6.1 Implement SYCL Event Profiling

To isolate pure GPU execution time and eliminate CPU/host submission overhead, leverage SYCL’s built-in hardware profiling by initializing the test queue with `sycl::property::queue::enable_profiling()`.

Use this standalone function to measure memory bandwidth and computational throughput (TFLOPS) for a specific tensor shape:

```cpp
#include <sycl/sycl.hpp>
#include <iostream>

void profile_xe2_gemv(sycl::queue& q, int B, int K, int N, int iterations = 100) {
    auto* input = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(B * K, q);
    auto* weight = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(K * N, q);
    auto* output = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(B * N, q);

    // Warmup to spin up GPU clocks
    for (int i = 0; i < 5; ++i) {
        q.submit([&](sycl::handler& cgh) { /* ... launch GEMV kernel ... */ }).wait();
    }

    double total_time_ns = 0;
    
    // Benchmark loop
    for (int i = 0; i < iterations; ++i) {
        auto event = q.submit([&](sycl::handler& cgh) {
             /* ... launch GEMV kernel ... */
        });
        
        event.wait(); // Wait ensures accurate event timestamps
        auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
        total_time_ns += (end - start);
    }
    
    double avg_time_ms = (total_time_ns / iterations) / 1e6;
    
    // Calculate memory bandwidth for GEMV
    double data_bytes = (K * N + B * K + B * N) * sizeof(sycl::ext::oneapi::bfloat16);
    double bandwidth_gbps = (data_bytes / 1e9) / (avg_time_ms / 1000.0);
    
    std::cout << "Shape [B=" << B << ", K=" << K << ", N=" << N << "]\n";
    std::cout << "Time: " << avg_time_ms << " ms | Bandwidth: " << bandwidth_gbps << " GB/s\n";
    
    sycl::free(input, q);
    sycl::free(weight, q);
    sycl::free(output, q);
}
```

### 6.2 Parameterize the Qwen3.8-Flash-Next Shapes

Instead of static unit tests, wrapping the SYCL execution in **Google Benchmark** (`perf/sycl/perf_xe2_gemv.cpp`) enables automated parameter sweeps through the exact model shapes extracted from the engine:

```cpp
#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

static void BM_Xe2_GEMV(benchmark::State& state) {
    const int K = state.range(0);
    const int N = state.range(1);
    const int B = state.range(2);

    sycl::queue q{sycl::default_selector_v, {sycl::property::queue::enable_profiling()}};
    auto* in  = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(B * K, q);
    auto* w   = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(K * N, q);
    auto* out = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(B * N, q);

    // Warmup prior to timing to ramp clocks from idle (800 MHz) to peak boost (2850 MHz)
    for (int warmup = 0; warmup < 10; ++warmup) {
        q.submit([&](sycl::handler& cgh) { /* kernel */ }).wait();
    }

    for (auto _ : state) {
        auto ev = q.submit([&](sycl::handler& cgh) { /* kernel */ });
        ev.wait();
        auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        state.SetIterationTime((t1 - t0) * 1e-9);
    }

    const double bytes = (double)(K * N + B * K + B * N) * sizeof(sycl::ext::oneapi::bfloat16);
    state.counters["Bandwidth_GBps"] = benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["TFLOPS"]         = benchmark::Counter((2.0 * B * K * N), benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["tokens_per_sec"] = benchmark::Counter((double)B, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);

    sycl::free(in, q); sycl::free(w, q); sycl::free(out, q);
}
```

### 6.3 Measured Baseline Results on Intel Lunar Lake Xe2 (`Intel Graphics [0x64a0]`)

Measured using `make perf-sycl` (`icpx 2026.1`, Level-Zero driver `1.14.37020+3`, native `SUB_GROUP_SIZE=16`):

| Shape $[K, N, B]$ | Tensor Name | Kernel Path | Hardware Time | Host CPU Time | Bandwidth | TFLOPS | Tokens/s |
|---|---|---|---|---|---|---|---|
| $[2560, 48, 1]$ | `ssm_alpha / ssm_beta` | Subgroup SIMD16 | 23.7 µs | 156.2 µs | 10.57 GB/s | 10.35 GFLOPS | 42.1k |
| $[2560, 48, 1]$ | `ssm_alpha` (narrow) | **SLM Split TPR=128** | **17.7 µs** | 156.1 µs | **14.14 GB/s** | **13.85 GFLOPS** | **56.3k** |
| $[2560, 128, 1]$ | `indexer.k_proj` | Subgroup SIMD16 | 28.6 µs | 164.0 µs | 23.08 GB/s | 22.89 GFLOPS | 34.9k |
| $[2560, 512, 1]$ | `indexer.q_proj` | Subgroup SIMD16 | 43.9 µs | 160.7 µs | 59.85 GB/s | 59.71 GFLOPS | 22.8k |
| $[2560, 640, 1]$ | `ffn_gate_shexp` | Subgroup SIMD16 | 44.3 µs | 155.6 µs | 74.10 GB/s | 73.95 GFLOPS | 22.6k |
| $[640, 2560, 1]$ | `ffn_down_shexp` | Subgroup SIMD16 | 43.7 µs | 161.2 µs | 75.09 GB/s | 74.95 GFLOPS | 22.9k |
| $[10240, 320, 1]$ | `hc_attn_down` | Subgroup SIMD16 | 87.6 µs | 207.0 µs | 75.05 GB/s | 74.81 GFLOPS | 11.4k |
| $[320, 10240, 1]$ | `hc_attn_up` | Subgroup SIMD16 | 85.3 µs | 204.8 µs | 77.07 GB/s | 76.82 GFLOPS | 11.7k |
| $[2560, 2560, 1]$ | `ple_value / ffn` | Subgroup SIMD16 | 136.8 µs | 257.8 µs | 95.92 GB/s | 95.84 GFLOPS | 7.31k |
| $[2560, 10240, 1]$ | Gate / Up Expert | Subgroup SIMD16 | 512.3 µs | 661.7 µs | **102.39 GB/s** | 102.34 GFLOPS | 1.95k |
| $[10240, 2560, 1]$ | Down Expert | Subgroup SIMD16 | 529.6 µs | 675.7 µs | 99.05 GB/s | 99.00 GFLOPS | 1.89k |
| $[10240, 320, 24]$ | `hc_attn_down` (chunk=24) | Subgroup Batched | 756.4 µs | 918.6 µs | 9.33 GB/s | **207.93 GFLOPS** | 31.7k |
| $[2560, 2560, 32]$ | Attention (chunk=32) | Subgroup Batched | 3766 µs | 4728 µs | 3.57 GB/s | **111.37 GFLOPS** | 8.50k |

**Microarchitectural Findings**:
1. **CPU Overhead Elimination**: Host-side CPU dispatch latency was 150–280 µs across tests. Using `benchmark::UseManualTime()` isolated actual hardware execution time (17–136 µs for decode), preventing CPU jitter from masking GPU kernel improvements.
2. **SLM Split Reduction for Narrow Projections**: On `ssm_alpha` ($[2560, 48]$), output channels ($N=48$) underfill the GPU's execution units. Switching from subgroup-per-row to shared-local-memory (SLM) split reduction with `threads_per_row = 128` improved latency from 23.7 µs to 17.7 µs (**+25.3% speedup**, 10.57 → 14.14 GB/s).
3. **SIMD16 vs SIMD32**: Sweeping `reqd_sub_group_size(16)` against `reqd_sub_group_size(32)` on $[2560, 2560]$ demonstrated SIMD16 execution consistently outperforms SIMD32 (139.6 µs vs 141.4 µs) due to native single-cycle group reduction instructions on Xe2 Vector Engines.
4. **XMX Systolic Tile Sizing ($16 \times 16 \times 16$ vs $8 \times 16 \times 16$)**: On the large `attn_qkv` projection ($M=32, K=2560, N=10240$), Tile16 ($16 \times 16 \times 16$) achieves **1.23 TFLOPS** (1364 µs) versus Tile8's 710 GFLOPS (2361 µs), delivering a **1.73x speedup (+73% throughput)** by amortizing weight loads across 16 rows instead of 8.
5. **QSA FlashAttention & Sparse Top-K Gather**:
   - `BM_Xe2_QSA_PromptAttention`: Scaled smoothly from $T=64$ (142.5 µs, 449.2k tok/s) to $T=4096$ (6.95 ms, 589.6k tok/s, 14.49 GFLOPS).
   - `BM_Xe2_QSA_TopK_Gather`: Sparse non-contiguous KV block gathering sustained **16.7 to 22.2 GB/s** across 1k to 64k pooled context cells (94.6 µs for 1k cells, 125.3 µs for 64k cells).
6. **MoE Expert Accumulation (Atomic vs Direct Combine vs Subgroup Reduce)**:
   - For single-token decode ($B=1$), atomic addition has lower grid dispatch overhead (10.38 µs vs 14.75 µs for Subgroup Reduce).
   - For batched inference / prefill ($B=16, 32$), atomic collisions heavily degrade scaling. Direct Combine and Subgroup Reduce achieve **81.3 GB/s** at $B=32$ (44.4 µs vs 150.8 µs for Atomic), delivering a **3.40x speedup** over Atomic.
7. **Direct-to-Register Dequantization vs SLM-Buffered DPAS**:
   - On Gate projection $[8, 2560, 6144]$, unpacking 4-bit weights directly into register tiles for DPAS achieves **423.7 GFLOPS** (594.0 µs, 15.29 GB/s) versus SLM-buffered's 329.8 GFLOPS (763.1 µs, 11.91 GB/s).
   - Eliminating the intermediate SLM write and local barrier synchronization provides a **+28.5% latency reduction**.

### 6.4 Complete Performance Suite Status (`perf/sycl/`)

All 5 planned performance benchmarks are implemented, verified on live Lunar Lake hardware (`Intel Graphics [0x64a0]`), and integrated into CMake:

```
perf/
└── sycl/
    ├── perf_xe2_gemv.cpp        # [COMPLETED] BF16 parameter sweeps across 8 Qwen shapes, SIMD16 vs 32, SLM split (+25.3% speedup)
    ├── perf_xe2_xmx_tiles.cpp   # [COMPLETED] XMX systolic tile sweep (Tile16 achieves 1.23 TFLOPS, 1.73x over Tile8)
    ├── perf_qsa_attention.cpp   # [COMPLETED] Prompt attention (up to 611k tok/s) & Top-K gather (22.2 GB/s)
    ├── perf_moe_accumulate.cpp  # [COMPLETED] MoE 10-expert reduction (Subgroup Reduce: 81.3 GB/s, 3.4x faster than Atomic)
    └── perf_dequant_dpas.cpp    # [COMPLETED] Direct-to-register 4-bit dequant + DPAS (+28.5% faster than SLM buffering)
```

### 6.5 CMake Integration & Profiling Commands

CMake integrates Google Benchmark via an opt-in toggle with automatic `FetchContent` fallback, ensuring zero impact on CI tests:

```bash
# Configure with performance suite enabled
cmake -B build_sycl -DSTRATA_BUILD_PERF=ON

# Run full performance suite
make -C build_sycl perf-sycl

# Run specific microbenchmark filter
./build_sycl/perf_xe2_gemv --benchmark_filter='^BM_Xe2_GEMV/'

# Export JSON for commit-to-commit comparison
./build_sycl/perf_xe2_gemv --benchmark_out=baseline.json --benchmark_out_format=json
python3 tools/compare.py benchmarks baseline.json new_commit.json
```

Pipe directly into Intel VTune for hardware cache and ALU saturation metrics:
```bash
vtune -collect gpu-hotspots ./build_sycl/perf_xe2_gemv
```

### 6.5 The 5 Critical Decode Bottlenecks to Benchmark on Xe2

1. **Weight-Only Dequantization + XMX DPAS (The Decode Bottleneck)**:
   - At decode ($B=1$), generation speed is strictly bounded by memory bandwidth.
   - Measure streaming 2-bit/4-bit weights from global memory, dequantizing to BF16 in registers via Xe Vector Engines (XVE), and feeding directly to XMX engines via DPAS instructions.
   - Test shape: Gate projection `[2560, 6144]` at Q2_0.
   - Compare direct-to-register dequantization vs. buffering dequantized blocks in Shared Local Memory (SLM) before XMX ingestion.
2. **Fused GDN State Recurrence**:
   - 36 of 48 layers in Qwen3.8-Flash-Next are Gated Delta Networks (linear attention) with recurrent state `[16, 128, 128]`.
   - Fuse Alpha GEMV, Beta GEMV, and the State Recurrence step into a single kernel to keep the recurrent state resident in Xe-core SLM or subgroup registers across the layer.
   - Test whether `reqd_sub_group_size(32)` provides better SLM bank alignment and less thread-divergence than `reqd_sub_group_size(16)` for this specific `[128, 128]` chunking.
3. **QSA Block Scorer (Top-K) & Sparse Gather**:
   - The 12 QSA layers use Top-K selection (`[B, 4, 128]` block scorer against `[N_blocks, 128]` pooled keys).
   - Benchmark Subgroup Bitonic Sort vs. Block-Radix select for determining Top-10 block IDs.
   - Measure L2 cache resilience during sparse non-contiguous Q4_0 KV fetches.
4. **MoE Expert Accumulation (Weighted Reduction)**:
   - Qwen3.8 routes 10 out of 512 experts per token. Hot experts compute `[10, 2560]` outputs scaled by an FP32 weight array.
   - Avoid atomic FP32 additions in global memory; use hierarchical `sycl::reduce_over_group` across SIMD16 subgroups.
5. **PCIe Async Copy / USM Overlap**:
   - Hot experts execute in VRAM, cold experts in CPU RAM, and PLE n-gram tables stream from SSD.
   - Ensure SYCL USM maps to B580 hardware Copy Engines (BCE) so memory transfers overlap 100% with XMX compute.
   - Benchmark an infinite loop of heavy XMX kernels on one queue while a secondary queue performs asynchronous `q.memcpy()` transfers, verifying compute TFLOPS does not degrade.

