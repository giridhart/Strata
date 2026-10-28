# SYCL Command Graph Hang Investigation (Intel Arc B580 / Xe2)

## 1. Problem Statement

When running `strata serve` on Intel Arc B580 (Battlemage / Xe2 architecture, Level Zero runtime), speculative decoding triggers a 60-second watchdog stall report during verify window graph capture:

```text
verify dbg: capture start: T=4
verify dbg: capture recorded: T=4 in 34.35 ms, instantiating graph
strata serve: no progress for 60 s during a request (verify window: instantiating graph 4) - stopping the engine so the server starts it again (issue #29)
 expert pool: epoch 0, batch epoch 0: 0 of 0 jobs claimed, 0 done; 26 of 26 workers parked, 26 sleeping; mode 0
 verify window: 0 tokens at position 0, host at layer step 1; the GPU rang 0; flags: served 0, plan (A) 0, copies (B) 0
 memory: 96485 MiB resident, 0 MiB in swap, 72590 MiB RAM available; 10 major page faults so far
strata: released the verify window's GPU waits (#267): the GPU finished in 0 ms
*** stack smashing detected ***: terminated
```

---

## 2. Key Findings So Far

1. **Recording is NOT the Hang**:
   `record_window(T, cs_, rerr)` finishes in **34.35 ms** and `cudaStreamEndCapture` returns immediately. There are no blocking `.wait()` calls during capture.
2. **Pure Node Count is NOT the Limit**:
   A standalone benchmark (`test_graph_limit.cpp`) was compiled and run on the Arc B580. It verified that:
   - 1,000 nodes finalize in 0.96 ms.
   - 1,300 nodes finalize in 1.67 ms.
   - 3,000 nodes (with heterogeneous USM host + device memory) finalize in **5.75 ms**.
   The Level Zero compiler handles 3,000 simple sequential nodes without hanging.
3. **Exact Failure Point**:
   The deadlock occurs exclusively inside `cudaGraphInstantiate(&exec_[T], graph, 0)`:
   ```cpp
   auto* exec = new strata_graph_detail::exec_graph_t(g->finalize());
   ```
   `g->finalize()` calls `urGraphInstantiateGraphExp` in Level Zero (`libur_adapter_level_zero.so` / `libze_intel_gpu.so`), which deadlocks when compiling the specific combination of kernel features recorded in `record_window`.
4. **Current Baseline**:
   Branch `sycl-b580` at commit `bb9429d1` has direct execution fallback active by default for `Verifier`, with graph capture opt-in via `STRATA_VERIFY_GRAPH=1`.

---

## 3. Investigation Plan for New Session

Create a standalone test tool (`test_graph_hang_repro.cpp`) in the repository root to isolate which specific kernel construct or dependency causes `g->finalize()` to deadlock.

### Test Matrix

| Config | Feature Under Test | Implementation in `record_window` |
|:---|:---|:---|
| **Config 1** | **Shared Local Memory (SLM)** | Multi-dimensional `sycl::local_accessor<float[4][32], 0>` used by `native_mmvq` and `native_expert_grouped`. |
| **Config 2** | **System-Scope Atomic Spin Loops** | `strata::sys_load(flag)` (`atomic_fence` with `memory_scope::system`) and `strata_spin_pause()` in `wait_flag_ge`. |
| **Config 3** | **Explicit Subgroup Directives** | `[[sycl::reqd_sub_group_size(32)]]` on kernels inside a command graph. |
| **Config 4** | **Intel Cache-Control Annotated Pointers** | `sycl::ext::intel::experimental::read_hint<...cache_mode::uncached...>` used in `doorbell_uncached_read`. |
| **Config 5** | **Layer Stacking (Scale-Up)** | Realistic layers combining GEMVs + SLM + atomics for $L \in \{1, 5, 10, 20, 61\}$ layers to test topology interactions. |

---

## 4. Standalone Tool Specification (`test_graph_hang_repro.cpp`)

```cpp
// Standalone harness to run individual configs with unbuffered logging
// Usage: ./test_graph_hang_repro --config <1-5>
//        ./test_graph_hang_repro --all
```

For each configuration:
1. Initialize an in-order queue: `sycl::queue q{sycl::property::queue::in_order{}};`
2. Begin recording: `graph.begin_recording(q);`
3. Dispatch test kernels for the selected config.
4. End recording: `graph.end_recording(q);`
5. Print `"[CONFIG X] Calling graph.finalize()..."` with unbuffered flush.
6. Call `auto exec = graph.finalize();`
7. Print `"[CONFIG X] SUCCESS in Y ms"`.
8. Replay: `q.ext_oneapi_graph(exec); q.wait();`

---

## 5. Build and Execution Commands

```bash
# 1. Source oneAPI environment
source /opt/intel/oneapi/setvars.sh

# 2. Compile standalone test
icpx -fsycl -O2 -Iinclude -Isrc test_graph_hang_repro.cpp -o test_graph_hang_repro

# 3. Run individual config with 15s timeout
timeout 15s ./test_graph_hang_repro --config 1
timeout 15s ./test_graph_hang_repro --config 2
timeout 15s ./test_graph_hang_repro --config 3
timeout 15s ./test_graph_hang_repro --config 4
timeout 15s ./test_graph_hang_repro --config 5

# 4. Or run full suite with 60s timeout
timeout 60s ./test_graph_hang_repro --all
```

---

## 6. Target Resolution Once Root Cause is Identified

1. **If Config 1 (SLM Accessor) hangs**:
   Replace local accessor capture in graph with static descriptor buffers or bypass graph for kernels with dynamic SLM.
2. **If Config 2 (Spin Loop / System Scope) hangs**:
   Move `wait_flag_ge` outside the recorded command graph (host-side synchronization before launching the next subgraph chunk).
3. **If Config 5 (Stacking threshold at layer $K$) hangs**:
   Chunk the verify window graphs by layer (e.g. 1 layer per graph or chunks of 5 layers, mirroring `session_capture`).
