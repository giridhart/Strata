# SYCL Command Graph Improvements & Tracking

Tracking document for inspecting Strata's native SYCL command-graph port (`src/core/sycl/` + `include/strata/core/mock_cuda_graph.hpp`), comparing it against [syclgraphusageguide.md](../syclgraphusageguide.md), implementing opt-in improvements behind runtime environment variables, and recording measured A/B results.

---

## 1. Current State of SYCL Graph Record & Replay (`src/core/sycl/`)

- **CUDA-to-SYCL Graph Bridge**: `include/strata/core/mock_cuda_graph.hpp` maps `cudaStreamBeginCapture`, `cudaStreamEndCapture`, `cudaGraphInstantiate`, and `cudaGraphLaunch` onto `sycl::ext::oneapi::experimental::command_graph<modifiable|executable>`.
- **Direct SYCL Graph Users**:
  - `src/core/sycl/graph.cpp` (`CapturedGraph` / `GraphRegistry`)
  - `src/core/sycl/dflash2.cpp` (captures one graph per draft count `n`)
  - `src/core/sycl/mtp.cpp` and `src/core/sycl/session.cpp` (via `mock_cuda_graph.hpp`)
- **Verifier (`src/core/sycl/verify.cpp`)**:
  - Defaults to `VerifyGraphMode::Direct` (`STRATA_VERIFY_GRAPH` unset/`0`/`direct`) — graphs are OFF by default.
  - Opt-in `Split` mode (`STRATA_VERIFY_GRAPH=split` or `1`): `ChunkRecorder` captures prologue + chunks of `STRATA_VERIFY_GRAPH_CHUNK` layers (default `1`) + epilogue into `exec_chunks_[T]`.
  - Opt-in `Whole` mode (`STRATA_VERIFY_GRAPH=whole` or `2`): captures the full window into one monolithic `exec_[T]` (previously observed to hang in `g->finalize()` / `urGraphInstantiateGraphExp` on Arc Xe2; see [SYCL_GRAPH_HANG_INVESTIGATION.md](SYCL_GRAPH_HANG_INVESTIGATION.md)).
  - `capture_commit()` is disabled (`commit_exec_ = nullptr`) due to Level Zero JIT hangs on Arc Xe2.
  - `capture_res()` and `capture_tree()` still use monolithic capture.

---

## 2. Issues Identified (Static Analysis)

1. **Forked side queues left in recording state**: `mock_cuda_graph.hpp` called `g->end_recording(*q)` on only the primary queue `cs_`. When `STRATA_PCIE_EARLY=force` forks work onto `pf_cs_` during capture, `pf_cs_` is not ended unless parameterless `g->end_recording()` is used.
2. **No graph properties passed**: `command_graph` was constructed with default properties everywhere — neither `property::graph::enable_native_recording{}` nor `property::graph::no_cycle_check{}` was used.
3. **Cold JIT during graph capture**: Capturing before any warmup pass can trigger cold SPIR-V JIT compilation across many unique kernels simultaneously inside `finalize()`.
4. **Opaque error handling in `mock_cuda_graph.hpp`**: `cudaStreamBeginCapture`, `cudaStreamEndCapture`, and `cudaGraphLaunch` swallowed exceptions with `catch (...)` without printing `e.what()` even when `STRATA_VERIFY_DEBUG` / `STRATA_TRACE` was enabled.
5. **Doc/code default mismatch**: `docs/SYCL_B580_EXPERT_CACHE_GUIDE.md` stated `STRATA_VERIFY_GRAPH=split` with chunk size `2` is the default, while `src/core/sycl/verify.cpp` defaults to `Direct` and `chunk_layers = 1`.

---

## 3. Items Taken from `syclgraphusageguide.md`

Checked against Intel oneAPI 2026.1 (`/opt/intel/oneapi/compiler/2026.1/include/sycl/ext/oneapi/experimental/`):

| Item from Guide | oneAPI 2026.1 Support | Status | Runtime Env Var (Opt-in) |
|:---|:---|:---|:---|
| `property::graph::enable_native_recording{}` | Yes (`graph_properties.def`) | Implemented (opt-in) | `STRATA_SYCL_GRAPH_NATIVE=1` |
| `property::graph::no_cycle_check{}` | Yes (`graph_properties.def`) | Implemented (opt-in) | `STRATA_SYCL_GRAPH_NO_CYCLE_CHECK=1` |
| Parameterless `g->end_recording()` (multi-queue) | Yes | Implemented (opt-in) | `STRATA_SYCL_GRAPH_END_ALL=1` |
| Umbrella switch for all three above | Yes | Implemented (opt-in) | `STRATA_SYCL_GRAPH_OPT=1` |
| Warmup before `begin_recording()` | Yes | Tested in `sycl_graph_opt_test` | N/A (usage pattern) |
| Whole-graph update (`updatable` + `exec.update`) | Yes | Tested in `sycl_graph_opt_test` | Future candidate for dynamic pointers |
| `graph_external` + `reusable_events` (in-graph timing / external sync) | **No** (upstream DPC++ only) | Tracked for future oneAPI | — |
| `async_malloc` / `async_free` | Yes | Not needed (Strata uses static arenas) | — |
| `ext_codeplay_enqueue_native_command` | Yes | Not needed (pure SYCL kernels) | — |

---

## 4. Runtime Environment Variables

All controls default to **OFF** so the default path remains unchanged:

- `STRATA_SYCL_GRAPH_OPT=1`: Enables `NATIVE`, `NO_CYCLE_CHECK`, and `END_ALL` together. Individual flags set to `0` override it.
- `STRATA_SYCL_GRAPH_NATIVE=1|0`: Construct `command_graph` with `property::graph::enable_native_recording{}`.
- `STRATA_SYCL_GRAPH_NO_CYCLE_CHECK=1|0`: Construct `command_graph` with `property::graph::no_cycle_check{}`.
- `STRATA_SYCL_GRAPH_END_ALL=1|0`: Call `g->end_recording()` (all queues) instead of `g->end_recording(*q)`.

---

## 5. Measurements on Local Hardware

Measured on **Intel Core Ultra 5 226V / Intel(R) Graphics [0x64a0]** (Lunar Lake Xe2-LPG, Level Zero V2 driver `1.14.37020+3`, Intel oneAPI 2026.1):

### 5.1 Unit & Correctness Tests

- `ctest --test-dir build_sycl -R "^(sycl_graph|sycl_graph_opt_test)$"` — **2/2 PASS** (0.43 s)
- `tests/sycl/graph_opt_test.cpp` verified:
  - Env var parsing & overrides (`STRATA_SYCL_GRAPH_OPT=1` + per-flag overrides) — **PASS**
  - `CapturedGraph` and `mock_cuda_graph.hpp` numeric exactness across `default`, `no_cycle_check`, `native_recording`, and `opt_all` — **PASS**
  - Multi-queue fork/join (`q_main` → `q_side` → `q_main`, mirroring `STRATA_PCIE_EARLY`) with `STRATA_SYCL_GRAPH_END_ALL=1`, confirming `q_side` exits recording state cleanly — **PASS**
  - Whole-graph update (`property::graph::updatable` + `exec.update(modGraph)`) — **PASS**
  - **Runtime quirk discovered & fixed**: `graph->get_nodes()` throws under `enable_native_recording` on Level Zero (`get_nodes() is not supported for graphs created with the enable_native_recording property`). Fixed in `src/core/sycl/graph.cpp` (`CapturedGraph::end` skips `get_nodes()` when `native_recording` is active).

### 5.2 Microbenchmark (`./build_sycl/sycl_graph_opt_test --bench`)

Warm kernel binary, 4,096 FP32 elements per node, 50 replay iterations at 64 nodes and 15 replay iterations at 1,200 nodes:

| Mode | Nodes | Record (ms) | Finalize (ms) | Replay Median (µs) | Replay Mean (µs) |
|:---|---:|---:|---:|---:|---:|
| `eager (direct)` | 64 | — | — | 765.1 | 879.8 |
| `graph_default` | 64 | 0.09 | 0.20 | 141.7 | 146.5 |
| `graph_no_cycle` | 64 | 0.07 | 0.12 | 140.8 | 145.1 |
| `graph_native` | 64 | 0.21 | 1.73 | 186.5 | 193.2 |
| `graph_opt_all` | 64 | 0.11 | 0.07 | 187.4 | 188.8 |
| `eager (direct)` | 1200 | — | — | 14026.6 | 13825.0 |
| `graph_default` | 1200 | 1.69 | 2.45 | 940.5 | 952.3 |
| `graph_no_cycle` | 1200 | 1.51 | 1.21 | 940.7 | 949.6 |
| `graph_native` | 1200 | 1.36 | 0.43 | 1658.8 | 1676.6 |
| `graph_opt_all` | 1200 | 1.16 | 0.34 | 1641.3 | 1689.1 |

**Key takeaways from the microbenchmark**:
- **`no_cycle_check` is a clean win**: cuts 1,200-node `finalize()` time in half (**2.45 ms → 1.21 ms**) with identical replay latency (**940.7 µs vs 940.5 µs**).
- **`enable_native_recording` trades replay overhead for much faster finalize**: cuts 1,200-node `finalize()` by **5.7×–7.2×** (**2.45 ms → 0.34–0.43 ms**), which may help avoid Level Zero finalize stalls on large chunks, **but** adds per-node replay overhead on this Level Zero V2 driver (**1,658.8 µs vs 940.5 µs** for 1,200 tiny kernels). On real heavy transformer kernels (milliseconds per layer), that ~0.7 ms host/driver delta may be negligible compared to fewer chunks or faster capture — so it must be A/B tested on the target Arc B580 workload.

### 5.3 End-to-End Synthetic Verifier (`./build_sycl/mock_graph_test`)

All 4 modes (`default`, `no_cycle`, `native`, `opt_all`) passed all checks on 48-layer Qwen 3.8 Flash-Next fake weights (`Intel(R) Graphics [0x64a0]`):

- First-capture + first-window wall time dropped from **858.8 ms** (`default`) → **803.6 ms** (`no_cycle`) → **665.3 ms** (`native`) → **627.5 ms** (`opt_all`).
- *(Note: `mock_graph_test` runs back-to-back on an iGPU with dynamic clock/thermal scaling, so treat its single-run replay ms as smoke-only and use interleaved A/B runs on Arc B580 for decode tok/s decisions.)*

---

## 6. How to A/B Test on Arc B580

Your current best baseline on Arc B580 is:

```bash
export STRATA_VERIFY_GRAPH=split
export STRATA_VERIFY_GRAPH_CHUNK=4
```

### Step 1: Build & Run the Local Sanity Benchmark

```bash
source /opt/intel/oneapi/setvars.sh
cmake --build build_sycl --target sycl_graph sycl_graph_opt_test strata -j$(nproc)
./build_sycl/sycl_graph_opt_test --bench
```

Confirm all 4 unit tests pass and record the B580 microbenchmark table (discrete Battlemage may differ from Lunar Lake on `graph_native` replay latency).

### Step 2: Interleaved A/B Runs (A-B-A-B) on Your Real Model

Use a fixed prompt, fixed seed / greedy (`--temp 0`), and fixed output length so token IDs stay bit-identical across runs. Keep all other flags identical to your best B580 config.

**Baseline (A)**:
```bash
STRATA_VERIFY_GRAPH=split STRATA_VERIFY_GRAPH_CHUNK=4 \
  <your strata command>
```

**Candidate B1 — `no_cycle_check` only (lowest risk)**:
```bash
STRATA_VERIFY_GRAPH=split STRATA_VERIFY_GRAPH_CHUNK=4 \
STRATA_SYCL_GRAPH_NO_CYCLE_CHECK=1 \
  <your strata command>
```

**Candidate B2 — `enable_native_recording` + `no_cycle_check` + `end_all` (`STRATA_SYCL_GRAPH_OPT=1`)**:
```bash
STRATA_VERIFY_GRAPH=split STRATA_VERIFY_GRAPH_CHUNK=4 \
STRATA_SYCL_GRAPH_OPT=1 \
  <your strata command>
```

**Candidate B3 — Larger chunks or `whole` unlocked by `STRATA_SYCL_GRAPH_OPT=1`**:
Since `enable_native_recording` cut 1,200-node finalize time by ~7× in the microbenchmark, test whether it allows bigger chunks (fewer graph submissions per window) or monolithic `whole` without hitting the Level Zero finalize hang:
```bash
# Try CHUNK=8, CHUNK=16, or whole:
STRATA_VERIFY_GRAPH=split STRATA_VERIFY_GRAPH_CHUNK=8 STRATA_SYCL_GRAPH_OPT=1 \
  <your strata command>

STRATA_VERIFY_GRAPH=whole STRATA_SYCL_GRAPH_OPT=1 \
  <your strata command>
```

**Candidate B4 (Optional) — `STRATA_PCIE_EARLY=force` inside graph capture**:
If your model uses `pcie_mode==2` expert offloading, `STRATA_SYCL_GRAPH_END_ALL=1` fixes the forked `pf_cs_` queue cleanup at `end_recording()`:
```bash
STRATA_VERIFY_GRAPH=split STRATA_VERIFY_GRAPH_CHUNK=4 \
STRATA_PCIE_EARLY=force STRATA_SYCL_GRAPH_END_ALL=1 STRATA_SYCL_GRAPH_NO_CYCLE_CHECK=1 \
  <your strata command>
```

### Step 3: Measure Capture/Finalize Time Explicitly

Add `STRATA_VERIFY_DEBUG=1` for one run per candidate to see exact per-chunk `strata graph: finalized N nodes in X.XX ms` lines and total capture time (remove `STRATA_VERIFY_DEBUG=1` when timing steady-state tok/s so stderr logging doesn't skew latency).

### Step 4: What to Compare

1. **Exactness**: Greedy output tokens must match Baseline (A) identically.
2. **Capture / Finalize time**: Total time to capture the window graphs on first request (and whether `CHUNK=8`/`16`/`whole` completes cleanly without a watchdog stall).
3. **Steady-state decode throughput (tok/s)** and **ms/window**: Compare median of at least 3 interleaved runs (`A, B, A, B, A, B`) after GPU clocks stabilize.
4. **Decision rule**: Only keep a flag on if it is measurably faster (or unlocks larger chunks that are measurably faster) with identical output.

