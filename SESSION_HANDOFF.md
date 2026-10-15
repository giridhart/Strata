# Session Handoff: Upstream v0.1.44.1 SYCL Porting & Performance Benchmarks

**Date**: 2026-10-15  
**Branch**: `sycl-b580`  
**Current HEAD**: `e1239d81`  
**Target Hardware**: Intel Arc B580 (Xe2-HPG Battlemage) & Core Ultra (Xe2-LPG Lunar Lake, `Intel(R) Graphics [0x64a0]`)  
**Toolchain**: Intel oneAPI 2026.1 (`icpx -fsycl -qmkl`)  

---

## 1. User Requests in this Session (Chronological Order)

1. **"@[SESSION_HANDOFF.md]"**
   - Loaded previous session context, architecture specs, and porting status.
2. **"Check upstream and perform same thing what was done previously with new chagnes."**
   - Merged 212 upstream commits from `upstream/main` (`v0.1.44.1`, base commit `296b59d2`) into `sycl-b580`.
   - Resolved merge conflicts in 8 files (`CMakeLists.txt`, `gguf_reader.hpp`, `dequant_bf16.hpp`, `fused_gdn.hpp`, `native_router.hpp`, `verify.cpp`, `vmm.cpp`, `native_expert.cpp`).
   - Ported 5 key upstream CUDA optimizations to SYCL:
     1. `GR_PICK` compile-time selects in fused GR read (`src/kernels/sycl/fused_gr.cpp`).
     2. Vectorized neighbour-lane block dequantizers with `store_run` vector stores (`src/kernels/sycl/iq_kernels.cpp`).
     3. Subgroup-shuffle GDN prompt recurrence `STRATA_GDN_XE` (`include/strata/prefill/gdn_xe.hpp`, `src/prefill/sycl/kernels.cpp`).
     4. `KqSplit` grouped MoE weight reuse across multi-token decode windows (`src/kernels/sycl/iq_kernels.cpp`, `include/strata/kernels/iq_kernels.hpp`).
     5. Dense Q8_0 GEMV 1-row block dispatch for $n_{out} \ge 2048$ (`src/kernels/sycl/native_mmvq.cpp`).
3. **"Are thse 212 commits made since yesterday?"**
   - Verified upstream commit timestamps: commits span from Oct 11 to Oct 15, reflecting cumulative upstream v0.1.44.1 releases (Metal port, multi-GPU pipeline, STRATA_HC_Q8, grouped GEMM, and MoE weight reuse).
4. **"can you update script to run all tests ? tools/test_sycl.sh all, all as default?"**
   - Updated `tools/test_sycl.sh` so that running `./tools/test_sycl.sh` with `all` or without arguments defaults to executing all 121 hardware and CPU test suites with automated single-retry on transient timeouts.
5. **"pressed esc by mistake continue with the work"**
   - Continued and completed full build (`warnings: 0`), diagnosed and resolved FMA contraction disparity in `s2_expert_grouped.cpp`, achieved 121/121 passing tests, and executed full Xe2 microbenchmarks.

---

## 2. Commit History from this Session

| Commit | Scope | Description |
|---|---|---|
| `b7cdf920` | `merge` | Merge branch 'upstream/main' into sycl-b580 (v0.1.44.1, 212 commits) |
| `23d4990e` | `feat(test)` | Make all tests default in test_sycl.sh and support explicit all/cpu modes |
| `7c9e4614` | `perf(sycl)` | Port GR_PICK compile-time selects to eliminate private memory spill |
| `381ac803` | `perf(sycl)` | Port neighbour-lane coalesced dequantization and store_run vector stores |
| `5f71020e` | `perf(sycl)` | Port STRATA_GDN_XE subgroup shuffle prompt recurrence |
| `8f938b2b` | `perf(sycl)` | Port KqSplit grouped MoE weight reuse (with `-fp-model=precise` in CMake) |
| `a862c704` | `perf(sycl)` | Dispatch 1-row blocks in dense Q8_0 GEMV for n_out >= 2048 |
| `e1239d81` | `fix(sycl)` | Unify float accumulation in gu_grouped_kernel to match FMA contraction |

---

## 3. Key Findings & Engineering Insights

1. **Clang `-fp-model=precise` vs Inline Function FMA Contraction**:
   - In Clang / IntelLLVM `icpx`, `-fp-model=precise` sets `-ffp-contract=on` (expression-level contraction only).
   - In `gu_grouped_kernel`, intermediate products were returned from an inline helper `chunk_dot()`, which Clang evaluated as a separate expression, disabling FMA contraction. In `gu_grouped_t_kernel`, `acc0 += dw * dx * ...` was a single expression and was contracted into an FMA.
   - This caused a 1 ULP difference in float results (`5.96e-8`). Replacing `chunk_dot()` with `chunk_int_diff()` and keeping `acc += dw * dx * (float)(diff)` in a single statement ensured identical compiler FMA contraction and achieved 100% bitwise parity.
2. **Xe2 EU Concurrency via 1-Row Work-Group Blocks**:
   - Wide projection layers ($n_{out} \ge 2048$) benefit significantly from 1-row block tiling (`ROWS = 1`), doubling the number of concurrent work-groups dispatched across Xe2 Dual Sub-Slices without increasing memory traffic.
3. **MoE Weight Reuse (`KqSplit`)**:
   - Reusing decoded weight words across multi-token verify windows (`Split<12/13/14/7/8>`) eliminates redundant weight decompression, yielding 2.5x to 4.0x speedups in multi-column GEMV and MoE grouped accumulation.

---

## 4. Measured Benchmark Results on Live Intel Xe2 Hardware

Hardware tested: `Intel(R) Graphics [0x64a0]` (Lunar Lake Xe2-LPG, matching Battlemage Xe2-HPG ISA, Sub-group sizes: 16, 32).

### Multi-Column GEMV & Grouped MoE (`iq_multi_parity --bench`)
| Benchmark | Configuration | Old (per-col decode) | New (decode once) | Speedup |
|---|---|---|---|---|
| **IQ2_XXS mmvq** | 8192 x 2560, ncols=8 | 2338.9 µs | **613.6 µs** | **3.81x faster** |
| **IQ2_XS mmvq** | 8192 x 2560, ncols=8 | 2176.3 µs | **638.5 µs** | **3.41x faster** |
| **IQ3_XXS mmvq** | 8192 x 2560, ncols=8 | 2775.2 µs | **693.6 µs** | **4.00x faster** |
| **IQ3_S mmvq** | 8192 x 2560, ncols=8 | 2661.0 µs | **686.5 µs** | **3.88x faster** |
| **IQ2_S mmvq** | 8192 x 2560, ncols=8 | 2213.0 µs | **599.6 µs** | **3.69x faster** |
| **IQ2_XXS / Q2_0 MoE** | 16 groups, m=8 entries | 8334.0 µs | **3235.1 µs** | **2.58x faster** |
| **IQ2_XS / Q2_0 MoE** | 16 groups, m=8 entries | 8313.8 µs | **3248.5 µs** | **2.56x faster** |
| **IQ3_XXS / Q2_0 MoE** | 16 groups, m=8 entries | 8129.0 µs | **3242.1 µs** | **2.51x faster** |
| **IQ3_S / Q2_0 MoE** | 16 groups, m=8 entries | 8054.4 µs | **3135.0 µs** | **2.57x faster** |

### Grouped S2 Expert Routing (`s2_expert_grouped_parity --bench`)
| Kernel / Workload | Previous | New (Optimized) | Speedup / Bandwidth |
|---|---|---|---|
| **`moe_hit_grouped_s2` (9 hits)** | 5018.8 µs (2 GB/s) | **430.0 µs (29 GB/s)** | **11.67x faster** |
| **`moe_hit_grouped_s2` (5 hits)** | 2858.0 µs (2 GB/s) | **323.4 µs (21 GB/s)** | **8.84x faster** |
| **`moe_grouped_s2` (4 tokens, 30% share)** | 1644.7 µs (26 GB/s) | **840.4 µs (50 GB/s)** | **1.96x faster** |
| **`moe_grouped_s2` (8 tokens, 30% share)** | 3178.5 µs (26 GB/s) | **1499.5 µs (56 GB/s)** | **2.12x faster** |
| **`moe_grouped_s2` (8 tokens, all shared)** | 2667.4 µs (5 GB/s) | **921.3 µs (15 GB/s)** | **2.90x faster** |

### Xe2 GEMV Microbenchmarks (`perf_xe2_gemv`)
- **Peak Bandwidth**: Up to **101.8 GB/s** (2560 x 10240, 1 token).
- **Peak Compute**: Up to **210.5 GFLOPS** (10240 x 320, 24 tokens).

### Prefill GDN Recurrence (`gdn_parity`)
- **Bitwise & Relative Parity**: 0 mismatches across $T=1, 7, 256, 1000$ (relative error $\le 2.06 \times 10^{-7}$).

---

## 5. How to Build & Test in the Next Session

### Environment Setup
```bash
source /opt/intel/oneapi/setvars.sh --force
```

### Clean Build Using Context-Protective Scripts
```bash
# Build strata executable (warnings: 0):
./tools/build_sycl.sh

# Build all targets, tests, and benchmarks (warnings: 0):
./tools/build_sycl.sh all

# Run all 121 test suites (verifies 100% pass):
./tools/test_sycl.sh all
```

### Running Standalone Microbenchmarks (`--bench` flag)
```bash
./build_sycl/iq_multi_parity --bench
./build_sycl/s2_expert_grouped_parity --bench
./build_sycl/fused_gr_bench
./build_sycl/gdn_parity
./build_sycl/perf_xe2_gemv
```
