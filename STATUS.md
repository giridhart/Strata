# Strata SYCL Porting & Intel Xe2 Optimization Status

The Strata native SYCL 2020 backend documentation has been consolidated into the [`docs/`](docs/) directory:

## Canonical Documentation Index

1. **[`docs/SYCL_STATUS.md`](docs/SYCL_STATUS.md)**: **Master Status & Kernel Registry**
   - Current build, toolchain (oneAPI 2026.1), and verification status (16/16 SYCL unit tests passing).
   - Complete registry of all 44 refactored pure SYCL computational kernels with parity targets.
   - Upstream rebase status on `v0.1.39` (`origin/main`).

2. **[`docs/SYCL_PORTING_GUIDE.md`](docs/SYCL_PORTING_GUIDE.md)**: **Architecture & Engineering Guide**
   - Core 6 translation directives (memory hierarchy, subgroups, atomics, queue synchronizations).
   - SYCL Command Graph extension architecture and transparent CUDA graph capture adapter.
   - Developer and AI refactoring guidelines.

3. **[`docs/INTEL_B580_OPTIMIZATION_PLAN.md`](docs/INTEL_B580_OPTIMIZATION_PLAN.md)**: **Intel Xe2 / Arc B580 Optimization Plan**
   - Microarchitecture details for Intel Arc B580 (Xe2-HPG) and Lunar Lake (Xe2-LPG `0x64a0`).
   - Phase-by-phase implementation: Ahead-of-Time compilation (`bmg`), native SIMD16 subgroup reductions (`SUB_GROUP_SIZE = 16`), hardware XMX systolic matrix acceleration (`joint_matrix`), and SLM bank alignment.
   - Complete 19-benchmark performance tuning suite in `perf/sycl/` and `tools/` with real measured hardware throughputs and latency improvements across all critical Xe2/B580 execution stages (including Tier 3 composite pipelines and Tier 4 system endpoints).

4. **[`HANDOVER.md`](HANDOVER.md)**: **Active Session Handover**
   - Work items, active task state, and reproduction steps for live GPU validation.

---

## Intel Arc B580 & Xe2 Target Build Resolution

Recent fixes applied for Intel Arc B580 discrete GPU (`STRATA_AOT_BMG=ON`, `STRATA_BUILD_PERF=ON`, `STRATA_ENABLE_SYCL=ON`):

1. **Host Compiler Flags & AOT Target**:
   - Fixed `-Xs "-device bmg"` unused argument warnings (`[-Wunused-command-line-argument]`) by appending `-Qunused-arguments` and scoping AOT options to [`CMAKE_CXX_FLAGS`](CMakeLists.txt#L197) so they never leak into C compiler (`icx`) during `ggml` builds.
   - Declared [`STRATA_NATIVE_EXPERTS`](CMakeLists.txt#L98) early in `CMakeLists.txt` so tests guarded by it are registered on the first configure pass.
2. **Eliminated `double` (FP64) Usage in SYCL Kernels**:
   - Replaced accidental `double` accumulators in [`src/kernels/sycl/s2_gemv.cpp`](src/kernels/sycl/s2_gemv.cpp) with native `float` matching the CUDA reference.
   - Intel Xe2 hardware has **0 native FP64 ALUs**; software-emulated FP64 doubled register pressure and caused `[bmg] SIMD16 allocated 128 regs and spilled 18`. Native `float` eliminated all spills (**0 spilled registers**) with optimal SIMD16 execution.
3. **Expert Cache SYCL Integration & Segmented Test Port**:
   - Implemented [`ExpertCache::mapped_bytes()`, `::shrink()`, `::grow()`, `::slots_within()`](src/core/sycl/expert_cache.cpp#L260-L300) in SYCL core, and initialized `live_slots_`.
   - Created native SYCL test [`tests/core/sycl/expert_cache_segmented_test.cpp`](tests/core/sycl/expert_cache_segmented_test.cpp), replacing CUDA runtime calls (`cudaMemcpy`, `cudaMemGetInfo`, `cudaGetDeviceCount`) with SYCL 2020 equivalents. The test cleanly verifies unsegmented cache behavior and exits with code 77 (SKIP) when segmented virtual memory management (`cuMem*`) is unsupported on the platform.
4. **Vendored `llama.cpp` as Submodule**:
   - Added [`.gitmodules`](.gitmodules) and pinned [`third_party/llama.cpp`](third_party/llama.cpp) to commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
   - [`CMakeLists.txt`](CMakeLists.txt#L1378-L1390) now prioritizes local `third_party/llama.cpp` over network `FetchContent`.
   - `iq_parity_fixtures` and `iq_parity` now automatically locate `gguf-py` and pass 10/10 quant formats with 0 failures.
5. **CTest Parity Suite Hardening**:
   - [`s2_gemv_parity`](src/kernels/sycl/s2_gemv_parity.cpp): Added synchronous queue flush to eliminate readback race condition on discrete GPUs; passes with 0 tolerance violations.
   - [`kv_stream_parity`](src/kernels/sycl/kv_stream.cpp): Queue configured with `in_order()` and explicit kernel dependency graph; passes 306 batches across INT8, FP16, and Q4_0.
   - [`kv_hybrid_parity`](src/kernels/sycl/kv_hybrid_parity.cpp): Subtest [5/5] handles graceful refusal when hardware has $\le 64\text{ KB}$ SLM (B580 has 64 KB SLM per Xe-core; XMX prompt attention requires $\ge 85\text{ KB}$), matching HIP fallback behavior and passing with `ALL PASS`.
   - [`ple_parity`](src/kernels/sycl/ple_parity.cpp): Returns exit code 77 when un-shipped multi-GB GGUF model files are absent; CTest configured with `SKIP_RETURN_CODE "77;2"` so missing models are cleanly reported as `Skipped`.

---

## Quick Start

```bash
# Source Intel oneAPI 2026.1 environment
source /opt/intel/oneapi/setvars.sh --force 2>/dev/null

# Compile and run SYCL unit test suite on GPU (CI fast check: 16/16 pass)
make -C build_sycl -j$(nproc) check-sycl

# Run full parity suite
./build_sycl/s2_gemv_parity --selftest
./build_sycl/kv_stream_parity
./build_sycl/kv_hybrid_parity
ctest --test-dir build_sycl -R iq_parity -V
ctest --test-dir build_sycl -R ple_parity -V

# Build and run Intel Xe2 performance tuning suite (Google Benchmark)
cmake -B build_sycl -DSTRATA_BUILD_PERF=ON
make -C build_sycl perf-sycl

# Run automated Xe2 / Arc B580 benchmark runner and generate Markdown report
make -C build_sycl perf-report
# Or directly via Python:
python3 tools/run_xe2_perf_benchmarks.py --quick --out-dir bench_results/smoke_test
```
