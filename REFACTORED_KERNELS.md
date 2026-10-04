# Refactored SYCL Kernels Registry

This document tracks all `.dp.cpp` files migrated from legacy Intel DPCT compatibility code to **Pure SYCL 2020** in the Strata codebase.

For comprehensive architectural directives, translation patterns, and SYCL Command Graph extension specifications, see **[docs/SYCL_PORTING_GUIDE.md](docs/SYCL_PORTING_GUIDE.md)**.

---

## 1. Pure SYCL 2020 Kernels (Fully Refactored & Verified)

All kernels listed below have had:
- All `<dpct/dpct.hpp>`, `dpct::` calls, and DPCT comments eradicated.
- DPCT 3D reversed ranges (`sycl::range<3>(1, 1, N)`) flattened to 1D ranges (`sycl::range<1>(N)`).
- Local accessors and atomic operations strictly scoped (`sycl::memory_scope::work_group` or `device`, `sycl::memory_order::relaxed`).
- Queue and stream scoping properly handled via local fallback default queues.
- Mathematical parity verified with **0 failures** against reference oracles.

| # | Kernel File | Chunk / Commit | Parity Test Target | Parity Status | Key Refactoring Notes |
|---|---|---|---|---|---|
| 1 | [`src/kernels/sycl/rope.dp.cpp`](../src/kernels/sycl/rope.dp.cpp) | Chunk 1 (`6771146`) | `rope_parity` | **PASS (0 failures)** | 1D range flattening, queue scoping |
| 2 | [`src/kernels/sycl/gdn.dp.cpp`](../src/kernels/sycl/gdn.dp.cpp) | Chunk 1 (`88796b3`) | `gdn_parity` | **PASS (0 failures)** | Multi-dim `local_accessor<float, 2>`, `shift_group_left` |
| 3 | [`src/kernels/sycl/kv_stream.dp.cpp`](../src/kernels/sycl/kv_stream.dp.cpp) | Chunk 1 (`de05375`) | `kv_stream_parity` | **PASS (0 failures)** | `sycl::atomic_ref` with device/workgroup memory scoping |
| 4 | [`src/kernels/sycl/s2_gemv.dp.cpp`](../src/kernels/sycl/s2_gemv.dp.cpp) | Chunk 3 (`f6b0ffd`) | `s2_gemv_parity` | **PASS (0 failures)** | Native 1D mapping, eradicated `dpct::` headers and macros |
| 5 | [`src/kernels/sycl/s2_gemv_q8.dp.cpp`](../src/kernels/sycl/s2_gemv_q8.dp.cpp) | Chunk 3 (`f6b0ffd`) | `s2_gemv_q8_parity` | **PASS (0 failures)** | Native 1D indexing, cleaned error checks |
| 6 | [`src/kernels/sycl/bf16_gemv.dp.cpp`](../src/kernels/sycl/bf16_gemv.dp.cpp) | Chunk 3 (`f6b0ffd`) | `bf16_gemv_parity` | **PASS (0 failures)** | `sycl::ext::oneapi::bfloat16`, subgroup sizing (32) |
| 7 | [`src/kernels/sycl/router_top10.dp.cpp`](../src/kernels/sycl/router_top10.dp.cpp) | Chunk 3 (`f6b0ffd`) | `router_top10_parity` | **PASS (0 failures)** | Tree-sitter AST stream scoping, `shift_group_left` |
| 8 | [`src/kernels/sycl/dequant_s2.dp.cpp`](../src/kernels/sycl/dequant_s2.dp.cpp) | Chunk 2 (`f6b0ffd`) | `dequant_s2_parity` | **PASS (0 failures)** | Native ranges, pure SYCL math built-ins |
| 9 | [`src/kernels/sycl/elementwise.dp.cpp`](../src/kernels/sycl/elementwise.dp.cpp) | Chunk 4 (`ad56c9b`) | `elementwise_parity` | **PASS (0 failures)** | Volatile qualifiers preserved, DPCT comments stripped |
| 10 | [`src/kernels/sycl/ple.dp.cpp`](../src/kernels/sycl/ple.dp.cpp) | Chunk 4 (`ad56c9b`) | `ple_parity` | **PASS (0 failures)** | Inline replacements, unwrapped try/catch blocks |
| 11 | [`src/kernels/sycl/quantize_act.dp.cpp`](../src/kernels/sycl/quantize_act.dp.cpp) | Chunk 4 (`ad56c9b`) | `quantize_act_parity` | **PASS (0 failures)** | 1D range flattening, queue scoping, exact tie parity |
| 12 | [`src/kernels/sycl/cvec.dp.cpp`](../src/kernels/sycl/cvec.dp.cpp) | Chunk 5 (`7bae2f0`) | `cvec_parity` | **PASS (0 failures)** | Pure SYCL queue scoping, removed dpct device selection, eliminated try/catch |
| 13 | [`src/kernels/sycl/kv_q4.dp.cpp`](../src/kernels/sycl/kv_q4.dp.cpp) | Chunk 5 (`064d010`) | `kv_q4_parity` | **PASS (0 failures)** | Pure SYCL subgroups, item passing, removed all DPCT headers/types |
| 14 | [`src/kernels/sycl/kv_q8.dp.cpp`](../src/kernels/sycl/kv_q8.dp.cpp) | Chunk 5 (`22bb657`) | `kv_q8_parity` | **PASS (0 failures)** | Flattened 1D gather range, pure SYCL subgroups, local accessor |
| 15 | [`src/kernels/sycl/shared_expert.dp.cpp`](../src/kernels/sycl/shared_expert.dp.cpp) | Chunk 5 (`47722e4`) | `shared_expert_parity` | **PASS (0 failures)** | Pure SYCL 2020: 2D row-scaling range, pure subgroup warp reductions, local accessors, eradicated DPCT crutches |
| 16 | [`src/kernels/sycl/s_gemv.dp.cpp`](../src/kernels/sycl/s_gemv.dp.cpp) | Chunk 5 (`97c22f7`) | `s_gemv_parity`, `s_gemv_q8k_parity` | **PASS (0 failures)** | Pure SYCL 2020: constexpr constant codebook, 1D ranges, subgroup shuffles, f16 bit exactness, eradicated `dpct::constant_memory` and crutches |
| 17 | [`src/kernels/sycl/s2_expert_grouped.dp.cpp`](../src/kernels/sycl/s2_expert_grouped.dp.cpp) | Chunk 5 (`ba1b7f1`) | `s2_expert_grouped_parity` | **PASS (0 failures)** | Pure SYCL 2020: vector-based `dp4a`, bitwise permute, subgroup shift, in-order queue scoping, eradicated `dpct::dim3`, `dpct::err0`, and all DPCT headers/crutches |
| 18 | [`src/kernels/sycl/qsa.dp.cpp`](../src/kernels/sycl/qsa.dp.cpp) | Chunk 5 (`c877939`) | `qsa_parity` | **PASS (0 failures)** | Pure SYCL 2020: `sycl::permute_group_by_xor`, in-order queue management, pure local accessor, clean device local memory queries, eradicated all DPCT headers/crutches |
| 19 | [`src/kernels/sycl/iq_kernels.dp.cpp`](../src/kernels/sycl/iq_kernels.dp.cpp) | Chunk 5 (`b215856`) | `iq_parity`, `iq_multi_parity` | **PASS (0 failures)** | Pure SYCL 2020: vector-based `dp4a`, bit-exact `vectorized_binary` with SYCL vector ops, bitwise `byte_permute`, `sycl::permute_group_by_xor`, in-order default queue, clean `range<3>`, eradicated all DPCT crutches |
| 20 | [`src/kernels/sycl/gr.dp.cpp`](../src/kernels/sycl/gr.dp.cpp) | Chunk 5 (`dd9c3bd`) | `gr_parity` | **PASS (0 failures)** | Pure SYCL 2020: 1D ranges, warp reductions via subgroups, eradicated all `dpct::` headers and crutches |
| 21 | [`src/kernels/sycl/native_gr_norm.dp.cpp`](../src/kernels/sycl/native_gr_norm.dp.cpp) | Chunk 5 (`dd9c3bd`) | `gr_parity` | **PASS (0 failures)** | Pure SYCL 2020: 1D range, `permute_group_by_xor`, local accessor `sums`, queue fallback, eradicated all `dpct::` |
| 22 | [`src/kernels/sycl/native_gr_postops.dp.cpp`](../src/kernels/sycl/native_gr_postops.dp.cpp) | Chunk 5 (`dd9c3bd`) | `gr_parity` | **PASS (0 failures)** | Pure SYCL 2020: 1D ranges (`down_silu`, `pre_gated`, `post`), in-order queue scoping, eradicated all `dpct::` |
| 23 | [`src/kernels/sycl/sampler.dp.cpp`](../src/kernels/sycl/sampler.dp.cpp) | Chunk 5 (`ca1bf62`) | `sampler_parity` | **PASS (0 failures)** | Pure SYCL 2020: 1D & 2D ranges, subgroup shuffles (`permute_group_by_xor`, `shift_group_left`, `select_from_group`), local atomic_ref, pure SYCL queue management, eradicated all DPCT headers/crutches |
| 24 | [`src/kernels/sycl/native_moe.dp.cpp`](../src/kernels/sycl/native_moe.dp.cpp) | Chunk 5 (`8b599a9`) | `native_expert_parity` | **PASS (0 failures)** | Pure SYCL 2020: unified 2D token-column ranges, pure queue management, eradicated all DPCT headers/crutches |
| 25 | [`src/kernels/sycl/qsa_decode_attn.dp.cpp`](../src/kernels/sycl/qsa_decode_attn.dp.cpp) | Chunk 5 (`6fcf95e`) | `kv_hybrid_parity`, `qsa_parity` | **PASS (0 failures)** | Pure SYCL 2020: unified 3D chunk range, 2D merge range, templated KV_MODE launcher, subgroup warp reductions, eradicated all DPCT headers/crutches |
| 26 | [`src/kernels/sycl/qsa_prompt_attn.dp.cpp`](../src/kernels/sycl/qsa_prompt_attn.dp.cpp) | Chunk 5 (`5ffaffc`) | `qsa_prompt_attn_parity`, `kv_hybrid_parity` | **PASS (0 failures)** | Pure SYCL 2020: pure SYCL `select_from_group` MMA emu, 2D nd_range, native subgroups, queue fallback, 4x prompt speedup, eradicated all DPCT headers/crutches |
| 27 | [`src/kernels/sycl/qsa_select.dp.cpp`](../src/kernels/sycl/qsa_select.dp.cpp) | Chunk 5 (`f1e0aff`) | `qsa_parity`, `kv_hybrid_parity` | **PASS (0 failures)** | Pure SYCL 2020: flattened 1D/2D ranges, workgroup-scoped atomic_ref radix histogram, pure subgroup shuffles, eliminated 1204 lines of DPCT/cluster boilerplate |
| 28 | [`src/kernels/sycl/native_qsa.dp.cpp`](../src/kernels/sycl/native_qsa.dp.cpp) | Chunk 5 (`2073f83`) | `qsa_parity`, `kv_hybrid_parity` | **PASS (0 failures)** | Pure SYCL 2020: 2D row ranges, subgroup reduction shuffles, 1D auto-tuned gate range, eradicated all DPCT headers/crutches |
| 29 | [`src/kernels/sycl/native_qsa_indexer.dp.cpp`](../src/kernels/sycl/native_qsa_indexer.dp.cpp) | Chunk 5 (`294325d`) | `qsa_parity`, `kv_hybrid_parity` | **PASS (0 failures)** | Pure SYCL 2020: flattened 1D workgroup ranges, pure subgroup shuffles, in-order queue management, bit-identical state parity, eradicated all DPCT headers/crutches |
| 30 | [`src/kernels/sycl/native_qsa_score.dp.cpp`](../src/kernels/sycl/native_qsa_score.dp.cpp) | Chunk 5 (`68cd0a2`) | `sycl_native_qsa_score`, `qsa_parity`, `kv_hybrid_parity` | **PASS (0 failures)** | Pure SYCL 2020: 1D workgroup per row, subgroup collective select broadcast, in-order queue fallback, zero shared memory, eradicated all DPCT headers/crutches |
| 31 | [`src/kernels/sycl/native_rope.dp.cpp`](../src/kernels/sycl/native_rope.dp.cpp) | Chunk 5 (`f73a57d`) | `rope_parity` | **PASS (0 failures)** | Pure SYCL 2020: 2D row/pair grid, in-order queue management, pure SYCL pow/math, eradicated all DPCT headers/crutches |
| 32 | [`src/kernels/sycl/native_router.dp.cpp`](../src/kernels/sycl/native_router.dp.cpp) | Chunk 5 (`b9946ce`) | `sycl_native_router`, `router_top10_parity` | **PASS (0 failures)** | Pure SYCL 2020: 1D workgroup per token, pure subgroup reductions (`permute_group_by_xor`), in-order queue management, eradicated all DPCT headers/crutches |
| 33 | [`src/kernels/sycl/native_ple_postops.dp.cpp`](../src/kernels/sycl/native_ple_postops.dp.cpp) | Chunk 5 (`59b61e0`) | `sycl_native_ple_postops`, `ple_parity` | **PASS (0 failures)** | Pure SYCL 2020: 1D ranges, subgroup shuffles (`permute_group_by_xor`), queue fallback, eradicated all DPCT headers/crutches |
| 34 | [`src/kernels/sycl/native_bf16.dp.cpp`](../src/kernels/sycl/native_bf16.dp.cpp) | Chunk 5 (`5da60d7`) | `sycl_native_bf16`, `bf16_gemv_parity` | **PASS (0 failures)** | Pure SYCL 2020: 1D ranges, subgroup shuffles (`permute_group_by_xor`), multi-token bit-exact parity, eradicated all DPCT headers/crutches |
| 35 | [`src/kernels/sycl/native_gdn.dp.cpp`](../src/kernels/sycl/native_gdn.dp.cpp) | Chunk 5 (`57eef4a`) | `sycl_native_gdn`, `gdn_parity` | **PASS (0 failures)** | Pure SYCL 2020: 2D head/col-warp range, subgroup reductions (`permute_group_by_xor`), eradicated all DPCT headers/crutches |
| 36 | [`src/kernels/sycl/native_gdn_preprocess.dp.cpp`](../src/kernels/sycl/native_gdn_preprocess.dp.cpp) | Chunk 5 (`c8a2dd1`) | `sycl_native_gdn_preprocess`, `gdn_parity` | **PASS (0 failures)** | Pure SYCL 2020: 1D ranges, subgroup shuffles (`permute_group_by_xor`), queue fallback, eradicated all DPCT headers/crutches |
| 37 | [`src/kernels/sycl/native_flash_attn.dp.cpp`](../src/kernels/sycl/native_flash_attn.dp.cpp) | Chunk 5 (`28f464d`) | `sycl_native_flash_attn` | **PASS (0 failures)** | Pure SYCL 2020: 2D head/warp-thread grid, subgroup shuffles & warp reductions, local accessors, eradicated all DPCT headers/crutches |
| 38 | [`src/kernels/sycl/native_mmvq.dp.cpp`](../src/kernels/sycl/native_mmvq.dp.cpp) | Chunk 5 (`9b0c1c4`) | `sycl_native_mmvq` | **PASS (0 failures)** | Pure SYCL 2020: custom pure SYCL `dp4a`, `byte_permute`, and `vsubss4` (`sub_sat` on `vec<int8_t, 4>`), pure subgroup reductions, 3D thread ranges, zero DPCT headers/types |
| 39 | [`src/kernels/sycl/s2_gemv_fast.dp.cpp`](../src/kernels/sycl/s2_gemv_fast.dp.cpp) | Chunk 5 (`1cca50d`) | `s_gemv_parity --bench` | **PASS (0 failures)** | Pure SYCL 2020: constexpr CodeLut code table, native 1D nd_range, pure local_accessor, in-order queue management, zero DPCT headers/types |
| 40 | [`src/kernels/sycl/s2_gemv_quads.dp.cpp`](../src/kernels/sycl/s2_gemv_quads.dp.cpp) | Chunk 5 (`22c148d`) | `s_gemv_parity --bench` | **PASS (0 failures)** | Pure SYCL 2020: native 1D nd_range, pure local_accessor, in-order queue management, zero DPCT headers/types |
| 41 | [`src/kernels/sycl/fused_gdn.dp.cpp`](../src/kernels/sycl/fused_gdn.dp.cpp) | Chunk 5 (`8891983`) | `sycl_fused_gdn` | **PASS (0 failures)** | Pure SYCL 2020: 2D head/row-group range, subgroup shuffles (`permute_group_by_xor`), local accessors, in-order queues, zero DPCT headers/types |
| 42 | [`src/kernels/sycl/fused_gr.dp.cpp`](../src/kernels/sycl/fused_gr.dp.cpp) | Chunk 5 (`845d44b`) | `gr_parity`, `cvec_parity` | **PASS (0 failures)** | Pure SYCL 2020: native subgroup shuffles (`permute_group_by_xor`), pure in-order default queues, clean device queries, zero DPCT headers/types |
| 43 | [`src/kernels/sycl/dequant_bf16.dp.cpp`](../src/kernels/sycl/dequant_bf16.dp.cpp) | Chunk 5 (`a26b76e`) | `sycl_dequant_bf16`, `elementwise_parity` | **PASS (0 failures)** | Pure SYCL 2020: 1D nd_range, compile-time constexpr lookup tables, native in-order queue management, zero DPCT headers/types |
| 44 | [`src/kernels/sycl/verify_kernels.dp.cpp`](../src/kernels/sycl/verify_kernels.dp.cpp) | Chunk 5 (`1771434`) | `sycl_verify_kernels` | **PASS (0 failures)** | Pure SYCL 2020: 1D/2D ranges, native subgroup shuffles (`permute_group_by_xor`), pure local accessors, in-order queue management, zero DPCT headers/types |

---

## 2. Pure SYCL 2020 Core & Graph Runtime (Step 2 Complete)

All core graph components listed below have been migrated from mock stubs and legacy DPCT wrappers to the native **SYCL Command Graph Extension** (`sycl::ext::oneapi::experimental::command_graph`):
- Pure queue recording (`graph_->begin_recording(*q)`, `graph_->end_recording(*q)`).
- Graph compilation/instantiation via `.finalize()` generating an `executable_graph`.
- Zero-latency submissions via `q->ext_oneapi_graph(*exec_)`.
- Non-blocking execution status queries via `sycl::info::event::command_execution_status == event_command_status::complete`.
- Bridge adapter for CUDA-syntax callers (`cudaStreamBeginCapture`, `cudaStreamEndCapture`, `cudaGraphInstantiate`, `cudaGraphLaunch`).
- Comprehensive verification with 0 failures in unit testing (`sycl_graph`) and full CTest suite on Intel GPU (`Intel(R) Graphics [0x64a0]`).

| # | File | Role | Test Target | Status | Key Refactoring Notes |
|---|---|---|---|---|---|
| 1 | [`include/strata/core/graph.hpp`](../include/strata/core/graph.hpp) | Core Graph Interface | `sycl_graph` | **PASS (0 failures)** | Pure SYCL `command_graph<graph_state::modifiable>` & `<graph_state::executable>`, `std::unique_ptr` lifecycle, eradicated `<dpct/dpct.hpp>` |
| 2 | [`src/core/sycl/graph.cpp.dp.cpp`](../src/core/sycl/graph.cpp.dp.cpp) | Core Graph Implementation | `sycl_graph` | **PASS (0 failures)** | Queue recording, node size tracking, executable finalization, `ext_oneapi_graph` launch, spinwait status query, `GraphRegistry` caching |
| 3 | [`include/strata/core/mock_cuda_graph.hpp`](../include/strata/core/mock_cuda_graph.hpp) | CUDA Graph SYCL Bridge | `sycl_graph` | **PASS (0 failures)** | Real `command_graph` backed implementation of `cudaStreamBeginCapture`, `cudaStreamEndCapture`, `cudaGraphInstantiate`, `cudaGraphLaunch` |
| 4 | [`src/kernels/sycl/mock_cuda_graph.hpp`](../src/kernels/sycl/mock_cuda_graph.hpp) | Kernel Mock Graph Bridge | `sycl_graph` | **PASS (0 failures)** | Native SYCL stream/event bridge types forwarding to core graph |
| 5 | [`tests/sycl/graph_test.cpp`](../tests/sycl/graph_test.cpp) | Unit & Integration Test | `sycl_graph` | **PASS (0 failures)** | Multi-kernel queue capture, node counting verification, replay evaluation, registry caching and replay |

---

## 3. Partially Refactored Files

- [`src/program/sycl/generate.cpp.dp.cpp`](../src/program/sycl/generate.cpp.dp.cpp): Main executable host code migrated from CUDA runtime to SYCL queue submission, linked against OneMKL 2025.3.

---

## 4. Pending Refactoring

### Remaining Computational Kernels (`src/kernels/sycl/`)
*(None — 100% of production computational kernels in `src/kernels/sycl/` (44/44) are fully migrated to Pure SYCL 2020!)*

### Remaining Core Runtime & Prefill (`src/core/sycl/`, `src/prefill/sycl/`)
- `src/core/sycl/session.cpp.dp.cpp`, `layer.cpp.dp.cpp`, etc. (Migrate from CUDA capture calls to pure `CapturedGraph` / `GraphRegistry` APIs)
- `src/prefill/sycl/*.dp.cpp` (Prefill gemm and pipeline modernization)


---

## 4. Verification Workflow

Each kernel must be validated against its parity binary using `verify_kernel.sh`:
```bash
# Verify kernel compilation and test parity
./verify_kernel.sh <kernel_name>
```

Environment prerequisite (automatically loaded in `verify_kernel.sh` and root `Makefile`):
```bash
source /opt/intel/oneapi/setvars.sh --force
source /opt/intel/oneapi/compiler/2025.3/env/vars.sh
source /opt/intel/oneapi/mkl/2025.3/env/vars.sh
```
