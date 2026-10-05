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
   - Complete 12-benchmark performance tuning suite in `perf/sycl/` and `tools/` with real measured hardware throughputs and latency improvements across all critical Xe2/B580 execution stages.

4. **[`HANDOVER.md`](HANDOVER.md)**: **Active Session Handover**
   - Work items, active task state, and reproduction steps for live GPU validation.

---

## Quick Start

```bash
# Source Intel oneAPI 2026.1 environment
source /opt/intel/oneapi/setvars.sh --force 2>/dev/null

# Compile and run SYCL unit test suite on GPU (CI fast check)
make -C build_sycl -j$(nproc) check-sycl

# Build and run Intel Xe2 performance tuning suite (Google Benchmark)
cmake -B build_sycl -DSTRATA_BUILD_PERF=ON
make -C build_sycl perf-sycl

# Run automated Xe2 / Arc B580 benchmark runner and generate Markdown report
make -C build_sycl perf-report
# Or directly via Python:
python3 tools/run_xe2_perf_benchmarks.py --quick --out-dir bench_results/smoke_test
```
