# Archived Plans & Specifications

This directory contains completed, superseded, or historical design documents and porting plans from earlier phases of the Strata SYCL / Intel Xe2 port.

| Archived Document | Original Purpose | Completion / Superseding Document |
|---|---|---|
| [`B580_OPTIMIZATION_AND_BENCHMARKING_PLAN.md`](B580_OPTIMIZATION_AND_BENCHMARKING_PLAN.md) | Initial 5-workstream microbenchmarking plan and Lunar Lake baseline measurements. | Consolidated into [`docs/INTEL_B580_OPTIMIZATION_PLAN.md`](../INTEL_B580_OPTIMIZATION_PLAN.md#9-multi-version-architecture--target-switching-strategy). |
| [`SYCL_CUDA_PORTING_PLAN_V022.md`](SYCL_CUDA_PORTING_PLAN_V022.md) | Upstream CUDA optimization audit and 8-phase SYCL porting roadmap for Strata v0.2.2. | All 8 phases completed & merged in Strata v0.2.6. Active optimization plan: [`docs/INTEL_B580_OPTIMIZATION_PLAN.md`](../INTEL_B580_OPTIMIZATION_PLAN.md). |
| [`2026-10-25-sycl-kernel-tuning.md`](2026-10-25-sycl-kernel-tuning.md) | Task-by-task implementation plan for on-device kernel autotuning, shape capture, and memory hints. | Completed in commit `3741d6c8`. Active guide: [`docs/SYCL_TUNING.md`](../SYCL_TUNING.md). |
