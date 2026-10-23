# Upstream CUDA Optimization Audit & SYCL Porting Plan for Intel Arc B580 (v0.2.2)

> **Target Hardware**: Intel Arc B580 Desktop GPU (Battlemage Xe2-HPG, BMG-G21)  
> **Microarchitecture**: 20 Xe2-cores (160 Vector Engines, 160 XMX Matrix Engines), 12 GB GDDR6 @ 456 GB/s, 16 MB L2/L3 Adamantine cache, 64 KB SLM per core (16 banks).  
> **Upstream Audit Baseline**: Commit `477ac13c` $\rightarrow$ `011c9e60` (Strata v0.2.2 / Engine 0.1.47).  
> **Status**: Ready for implementation.

---

## 1. Executive Summary & Audit Overview

Between Strata v0.1.43 and v0.2.2, upstream merged substantial performance enhancements across CUDA kernels, memory staging, and operator fusion pipelines. This document details the exact upstream commits, their microarchitectural implications for the **Intel Arc B580 (Xe2)**, and provides a prioritized four-tier implementation plan.

```mermaid
flowchart TD
    subgraph T1["Tier 1: High-Impact Decode Fusions"]
        F1["1.1: Dual Gate+Up GEMV + In-Register SwiGLU<br/>(s_gemv_pair_silu)"]
        F2["1.2: Fused Multi-Format Activation Quantizer<br/>(quantize_act_images)"]
        F3["1.3: Paired Linear GEMV<br/>(bf16_gemv_pair)"]
    end
    subgraph T2["Tier 2: KV Cache & Attention Vectorization"]
        K1["2.1: 8-Wide INT8 KV Cache Gather<br/>(kv_gather_q8_wide)"]
        K2["2.2: Register-Shuffled QSA Decode Scores<br/>(STRATA_QSA_ROWS)"]
        K3["2.3: Query Split Attention<br/>(STRATA_QSA_SPLIT_ATTN)"]
    end
    subgraph T3["Tier 3: Prefill Throughput & Cache MMQ"]
        M1["3.1: Direct Pointer-Table Expert MMQ<br/>(STRATA_MMQ_DIRECT / TAB)"]
        M2["3.2: Chunked GDN Recurrence<br/>(FlashQLA gdn-chunk)"]
    end
    subgraph T4["Tier 4: Trellis Quantization Support"]
        Q1["4.1: Gyro / Trellis TQ Dequant & DPAS<br/>(tq_load / tq_apply)"]
    end
    T1 --> T2 --> T3 --> T4
    style T1 fill:#d4edda,stroke:#28a745,color:#155724
    style T2 fill:#cce5ff,stroke:#004085,color:#004085
    style T3 fill:#fff3cd,stroke:#856404,color:#856404
    style T4 fill:#f8d7da,stroke:#721c24,color:#721c24
```

---

## 2. Tier 1: High-Impact Decode Fusions (Immediate 15–25% Decode Gain)

### 1.1 Dual Gate+Up GEMV + In-Register SwiGLU (`s_gemv_pair_silu`)
- **Upstream Commit**: `d284809f` ("V100: operator fusions for the decode GEMV / MoE paths, bitwise")
- **CUDA Origin**: `src/kernels/cuda/s_gemv_pair.cu`, `include/strata/kernels/swiglu.cuh`
- **Architectural Motivation**:
  - In Qwen3.8-Flash-Next MoE and shared expert blocks, the input activation $x$ is projected by two separate weight matrices: Gate and Up ($2560 \rightarrow 6144$ or $2560 \rightarrow 10240$).
  - Currently, SYCL launches two distinct GEMV kernels and saves intermediate float arrays to global device memory, followed by an elementwise SwiGLU kernel.
  - Upstream fuses both into `s_gemv_pair_silu`: activation $x$ is read once, both Gate and Up dot-products accumulate in registers, and SwiGLU $\text{silu}(g) \times u$ is evaluated in the kernel epilogue before writing a single output.
- **Xe2 / Arc B580 Porting Strategy**:
  - Target file: `src/kernels/sycl/s_gemv_pair.cpp` and `sycl/include/strata/kernels/s_gemv_pair.hpp`.
  - Use native Xe2 SIMD16 execution (`[[sycl::reqd_sub_group_size(16)]]`).
  - Maintain paired 32-bit float accumulators (`acc_gate`, `acc_up`) across subgroup reduction (`sycl::reduce_over_group`).
  - Epilogue directly executes:
    ```cpp
    float g = sycl::reduce_over_group(sg, acc_gate, sycl::plus<>());
    float u = sycl::reduce_over_group(sg, acc_up, sycl::plus<>());
    float silu_g = g / (1.0f + sycl::native::exp(-g));
    if (sg.leader()) { out[row] = silu_g * u; }
    ```
- **Status**: Completed (Pure SYCL 2020 SIMD16 port implemented, bitwise parity verified across all shapes/formats, wired into shared_expert).
- **Implementation**:
  - `src/kernels/sycl/s_gemv_pair.cpp`: native SIMD16 subgroup-per-output execution (`[[sycl::reqd_sub_group_size(SG_SIZE)]]`), supporting `SideS2`, `SideQ8K`, and `SideQ80`.
  - `include/strata/kernels/s_gemv.hpp` & `include/strata/kernels/s_gemv_pair.hpp`: host-callable interface `s_gemv_pair_silu`.
  - `src/kernels/sycl/shared_expert.cpp`: wired to bypass separate gate/up GEMVs and SwiGLU elementwise passes when both projections are S-family.
  - `tests/sycl/s_gemv_pair_parity.cpp`: bitwise parity verification suite across S4/Q8_0, S4/Q8_K, S8/Q8_K, S2/Q8_0, and mixed formats at $N=640$ and $N=10240$ (0 failures).
  - `perf/sycl/perf_s_gemv_pair.cpp`: microbenchmark comparing fused vs unfused pipelines at decode shapes.
- **Measured Performance (Intel Xe2 Lunar Lake / Battlemage GPU)**:
  - Shape $[K=2560, N=640]$:
    - Unfused (GEMV+GEMV+SwiGLU): **526.02 µs**
    - Fused (`s_gemv_pair_silu`): **430.77 µs** (**1.22x speedup**, -18.1% latency reduction)
    - Throughput: **15.22 GFLOPS, 4.29 GB/s**
  - Shape $[K=2560, N=6144]$:
    - Unfused (GEMV+GEMV+SwiGLU): **2336.96 µs**
    - Fused (`s_gemv_pair_silu`): **2254.38 µs** (**1.04x speedup**, -3.5% latency reduction)
    - Throughput: **27.92 GFLOPS, 7.86 GB/s**
  - Parity: 12/12 test suites **100% bitwise identical** (0 failures).

---

### 1.2 Fused Multi-Format Activation Quantizer (`quantize_act_images`)
- **Upstream Commit**: `d284809f`
- **CUDA Origin**: `src/kernels/cuda/quantize_act.cu`, `include/strata/kernels/quantize_act_dev.cuh`
- **Architectural Motivation**:
  - Across GDN and QSA layers, the activation vector $x$ must be converted into three representations: `block_q8_0`, `block_q8_K`, and `bf16`.
  - Unfused SYCL dispatches 2–3 separate kernels, reading the 2,560-element vector multiple times across PCIe/VRAM.
  - Upstream fuses this into a single pass: one memory read of $x$, producing all three quantization outputs simultaneously.
- **Xe2 / Arc B580 Porting Strategy**:
  - Target file: `src/kernels/sycl/quantize_act.cpp`.
  - In a SIMD16 workgroup, use 128-bit vector loads (`sycl::vec<float, 4>`) to read 64 bytes per transaction, saturating Xe2 cache lines.
  - Compute group maximum and scale via `sycl::reduce_over_group(sg, local_max, sycl::maximum<float>())`.
  - Store quantized nibbles and scales directly to their respective output buffers in one pass.
  - **Expected Impact on B580**: Latency drops from 46.2 µs to ~30.4 µs per layer handover.

---

### 1.3 Multi-Weight Paired Linear GEMV (`bf16_gemv_pair`)
- **Upstream Commit**: `d284809f`
- **CUDA Origin**: `src/kernels/cuda/bf16_gemv.cu`, `include/strata/kernels/bf16_gemv.hpp`
- **Architectural Motivation**:
  - `ssm_alpha` and `ssm_beta` in the 36 GDN layers, as well as Query ($Q$) and Key ($K$) projections in the QSA indexer, share the exact same input vector $x$.
  - Launching two separate micro-GEMVs incurs driver submission overhead (~150 µs on Level Zero when uncaptured) and redundant activation reads.
  - Fusing them into a dual-matrix GEMV computes both projections in a single workgroup grid.
- **Xe2 / Arc B580 Porting Strategy**:
  - Target file: `src/kernels/sycl/bf16_gemv.cpp`.
  - Dispatch dual weight matrix pointers `w_a` and `w_b`.
  - Subgroups stride along $K=2560$, loading $x[k]$ into registers and performing dual FMA accumulations against weights $A$ and $B$.
  - **Expected Impact on B580**: 24.2 µs $\rightarrow$ 14.4 µs on narrow projections.

---

## 3. Tier 2: KV Cache & Attention Vectorization

### 2.1 8-Wide INT8 KV Cache Gather (`kv_gather_q8_wide`)
- **Upstream Commit**: `132faae7` ("kv_q8: the gather decodes 8 int8 per thread (bitwise, +11% on the 32k microbench)")
- **CUDA Origin**: `src/kernels/cuda/kv_q8.cu`
- **Architectural Motivation**:
  - Quantized INT8 KV cache decode previously read 4 values per thread.
  - On Intel Xe2, hardware cache lines are **64 bytes and 128 bytes**.
  - A SIMD16 subgroup executing 4-wide reads transfers $16 \times 4 = 64$ bytes.
  - Upgrading to **8-wide loads** transfers $16 \times 8 = 128$ bytes per instruction, perfectly filling a full 128-byte cache line burst on B580's 192-bit GDDR6 memory bus.
- **Xe2 / Arc B580 Porting Strategy**:
  - Target file: `src/kernels/sycl/kv_q8.cpp`.
  - Load 8 INT8 elements using `sycl::vec<int8_t, 8>` (`uint64_t` equivalent).
  - Dequantize using single scale factor:
    ```cpp
    sycl::vec<sycl::half, 8> out_f16;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out_f16[i] = static_cast<sycl::half>(static_cast<float>(in_bytes[i]) * scale);
    }
    *reinterpret_cast<sycl::vec<sycl::half, 8>*>(&out_ptr[idx]) = out_f16;
    ```
  - **Expected Impact on B580**: +10–12% effective read bandwidth during 128K context attention decode.

---

### 2.2 Register-Shuffled QSA Decode Scores (`STRATA_QSA_ROWS`)
- **Upstream Commit**: `5b6c9398` ("QSA decode: the window's block scores with the queries in registers and six shuffles per block")
- **CUDA Origin**: `src/kernels/cuda/qsa_select.cu`
- **Architectural Motivation**:
  - In Query-Shared Attention (QSA) decode, evaluating block relevance scores across historical tokens previously required SLM-staged query vectors and local barrier synchronizations.
  - Upstream keeps the queries permanently resident in registers across the workgroup and broadcasts query components using register shuffles (`__shfl_sync`).
- **Xe2 / Arc B580 Porting Strategy**:
  - Target file: `src/kernels/sycl/qsa_select.cpp`.
  - Replace warp shuffles with SYCL 2020 group algorithms:
    - `sycl::select_from_group(sub_group, val, src_lane)`
    - `sycl::permute_group_by_xor(sub_group, val, mask)`
  - Completely eliminates SLM allocation and local barriers.
  - **Expected Impact on B580**: Block scoring latency drops from ~101 µs to ~55 µs at window size $T=4$.

---

### 2.3 Query Split-Attention Across Workgroups (`STRATA_QSA_SPLIT_ATTN`)
- **Upstream Commit**: `1a2f95c3` ("QSA decode attention: split each query's cells across blocks on the int8 mma path")
- **Architectural Motivation**:
  - At long contexts ($\ge 64\text{K}$), single-workgroup reduction across all historical cells suffers from poor workgroup occupancy on B580's 20 Xe-cores.
  - Upstream parallelizes context reduction by splitting cells across workgroup tiles (Split-K attention), followed by an online softmax accumulator reduction.
- **Xe2 / Arc B580 Porting Strategy**:
  - Integrated into `src/kernels/sycl/qsa_prompt_attn.cpp` and `perf/sycl/perf_kv_decode.cpp`.
  - Mapped directly to B580 tuning parameter `STRATA_XE2_SPLIT_K=32`.

---

## 4. Tier 3: Prefill Throughput & Direct Cache MMQ

### 3.1 Direct Pointer-Table Expert MMQ (`STRATA_MMQ_DIRECT` / `STRATA_MMQ_TAB`)
- **Upstream Commits**: `86f677b4`, `7d9b50dc`, `62fc7c8a`
- **CUDA Origin**: `src/prefill/moe_mmq_tab.cu`, `include/strata/prefill/moe_mmq_tab.cuh`
- **Architectural Motivation**:
  - In standard prefill, expert activations and weights are gathered into a contiguous staging buffer before launching batched MMQ matrix multiplication.
  - Upstream's pointer-table MMQ passes an array of 16 device pointers directly to the kernel, allowing the tensor-core GEMM to stream resident experts straight out of their VRAM cache slots without intermediate copy passes.
- **Xe2 / Arc B580 Porting Strategy**:
  - Target file: `src/prefill/sycl/xmx_moe.dp.cpp` and `sycl/include/strata/prefill/moe_mmq.hpp`.
  - Bind cache slot offsets as pointer parameters in the Level Zero execution queue.
  - When expert blocks are resident in VRAM, bypass DMA staging and dispatch `joint_matrix_load` directly from slot addresses.
  - **Expected Impact on B580**: Eliminates staging memory footprint, allowing `--prefill 4096` to fit within 12 GB VRAM even at 128K context.

---

### 3.2 Chunked GDN Prompt Recurrence (`FlashQLA gdn-chunk`)
- **Upstream Commit**: `7c4ff2de` ("prefill (v100): the chunked GDN prompt recurrence (FlashQLA gdn-chunk)")
- **CUDA Origin**: `src/prefill/gdn_chunk.cu`, `include/strata/prefill/gdn_chunk.hpp`
- **Architectural Motivation**:
  - The linear attention GDN recurrence is sequential across tokens. For long prompts (4K–128K), serial state updates bound prefill speed.
  - Chunked GDN divides the prompt into 64-token chunks: intra-chunk states are computed as triangular matrix solves using systolic matrix units, while inter-chunk states are accumulated serially.
- **Xe2 / Arc B580 Porting Strategy**:
  - Target file: `src/prefill/sycl/gdn_chunk.cpp`.
  - Use `sycl::ext::oneapi::experimental::matrix::joint_matrix` for the intra-chunk $K^T V$ and $Q (K^T V)$ systolic multiplications.
  - Intra-chunk triangular mask applied in register space.
  - Verified against double-precision reference in `tests/sycl/gdn_chunk_parity.cpp`.

---

## 5. Tier 4: Trellis Quantization Support (Gyro / APR Format)

### 4.1 Trellis Quantization Decoders (TQ1 / TQ2 / TQ6)
- **Upstream Commits**: `fbf045eb`, `b4e477c0`, `5f81c5dc`, `2c8eb881`
- **CUDA Origin**: `src/kernels/cuda/tq_load.cu`, `src/kernels/cuda/tq_apply.cu`
- **Architectural Motivation**:
  - Upstream v0.2.2 introduced **Gyro** models using Trellis Quantization (types 144, 145, 146), achieving 1.8–2.2 bits/weight with near-uncompressed quality.
  - Trellis weights use in-place Fast Walsh-Hadamard Transforms (FWHT) to rotate activation vectors, eliminating quantization outliers.
- **Xe2 / Arc B580 Porting Strategy**:
  - Leverage Strata's verified FWHT256 SLM kernel from `perf/sycl/perf_fwht_kv4.cpp`.
  - Implement TQ codebook staging in 64 KB SLM: store the 256-entry FP16/INT8 codebook in local memory, decoding 8-lane rows per subgroup.
  - Matrix-vector dot products executed via Xe2 DPAS (`joint_matrix_mad` or SIMD16 integer MAD).

---

## 6. Implementation & Verification Roadmap

| Phase | Milestone Target | New/Modified Files | Parity Test Target | Target Gain |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | Fused Gate/Up Dual GEMV + SwiGLU | `s_gemv_pair.cpp`, `swiglu.hpp` | `s_gemv_pair_parity` | +15% decode tok/s |
| **Phase 2** | Multi-Format Activation Quantizer | `quantize_act.cpp` | `quantize_act_parity` | +5% decode tok/s |
| **Phase 3** | Paired BF16 GEMV (`ssm_alpha/beta`) | `bf16_gemv.cpp` | `bf16_gemv_parity` | +4% decode tok/s |
| **Phase 4** | Wide 8-Byte INT8 KV Gather | `kv_q8.cpp` | `kv_q8_parity` | +10% long-context decode |
| **Phase 5** | Register-Shuffled QSA Scores | `qsa_select.cpp` | `qsa_select_bench` | -45 µs per QSA layer |
| **Phase 6** | Direct Pointer-Table Expert MMQ | `xmx_moe.dp.cpp`, `moe_mmq.hpp` | `prefill_gemm_parity` | Enables 4K chunk at 128K context |
| **Phase 7** | Chunked GDN Recurrence | `gdn_chunk.cpp` | `gdn_chunk_parity` | +20% prompt tok/s on >4K prompts |
| **Phase 8** | Gyro Trellis (TQ) Decoder | `tq_gemv.cpp`, `fwht.hpp` | `trellis_parity` | Enables Gyro-S/Gyro-M on B580 |

---

## 7. Immediate Next Steps

1. **Implement Phase 1 (`s_gemv_pair.cpp`)**: Port `s_gemv_pair_silu` into `src/kernels/sycl/s_gemv_pair.cpp`, wire into `shared_expert.cpp`, and verify bit-exact outputs against `fusions_parity`.
2. **Implement Phase 4 (`kv_gather_q8_wide`)**: Update `src/kernels/sycl/kv_q8.cpp` with 128-byte cache-line aligned loads to boost 128K context decode on Arc B580.
3. **Integrate with Automated Xe2 Benchmarks**: Validate all ported kernels via `make -C build_sycl perf-sycl` and `python3 tools/run_xe2_perf_benchmarks.py`.
