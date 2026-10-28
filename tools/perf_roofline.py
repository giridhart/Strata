#!/usr/bin/env python3
"""tools/perf_roofline.py - Intel Xe/Xe2 Hardware-Counter & Roofline Optimization Engine.

A mechanical, script-driven tool to profile SYCL kernels, compute theoretical
hardware ceilings (Peak Bandwidth vs Peak Compute), quantify distance-to-theoretical,
and generate actionable SYCL optimization prescriptions.

Features:
1. Device Auto-Discovery: Identifies active Intel GPU (e.g. Lunar Lake Xe2-LPG, Arc B580)
   and queries theoretical silicon ceilings (GB/s, TFLOPS, L3, SLM).
2. Live Benchmark Execution: Executes compiled Google Benchmark binaries (`perf_*`)
   under automated oneAPI environment sourcing.
3. Roofline Analysis: Computes operational intensity (FLOPs/byte), theoretical ceilings,
   measured performance, and the exact "Gap to Theoretical Limit" (%).
4. Pareto Hotspot Ranking: Sorts kernels by runtime share or efficiency gap.
5. Mechanical Diagnostics: Rule-based translation of bottlenecks into SYCL code fixes
   (Split-K TPR, Xe2 streaming L3 hints, SLM padding, SIMD16 subgroup sizing).
6. Comparison Mode: Mechanically evaluates baseline vs tuned variants (e.g. GEMV vs Split-K).
7. VTune CLI Integration & Fallback: Invokes `vtune` when installed or prints exact recipes.
8. Mock Mode: Runs offline/dry-run analysis with realistic benchmark traces.

Usage:
    # Run Roofline analysis on live GPU for GEMV microbenchmark
    python3 tools/perf_roofline.py --benchmark perf_xe2_gemv --filter "BM_Xe2_GEMV/2560/48/1"

    # Compare untuned GEMV baseline vs tuned Split-K on live GPU
    python3 tools/perf_roofline.py --benchmark perf_xe2_gemv \
        --compare "BM_Xe2_GEMV/2560/48/1" "BM_Xe2_GEMV_SplitReduction/2560/48/1/128"

    # Compare unfused GDN recurrence vs fused SLM
    python3 tools/perf_roofline.py --benchmark perf_fused_gdn \
        --compare "BM_Xe2_GDN_Unfused/16/16" "BM_Xe2_GDN_Fused_SLM/16/16"

    # Generate Markdown report
    python3 tools/perf_roofline.py --benchmark perf_xe2_gemv --report roofline_report.md

    # Offline / dry-run test mode
    python3 tools/perf_roofline.py --mock --report mock_report.md
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent


# ==============================================================================
# 1. Hardware Specifications Database
# ==============================================================================

@dataclass(frozen=True)
class HardwareSpecs:
    name: str
    arch: str
    xe_cores: int
    vector_engines: int
    peak_dram_bw_gbps: float
    peak_l3_bw_gbps: float
    peak_bf16_tflops: float
    peak_fp32_tflops: float
    slm_size_per_core_kb: int
    slm_banks: int
    l3_cache_mb: int
    subgroup_native: int


HW_SPECS_DATABASE = {
    # Desktop Intel Arc B580 (Xe2-HPG BMG-G21)
    "b580": HardwareSpecs(
        name="Intel Arc B580 (Desktop)",
        arch="Intel Xe2-HPG (BMG-G21)",
        xe_cores=20,
        vector_engines=160,
        peak_dram_bw_gbps=456.0,
        peak_l3_bw_gbps=1800.0,
        peak_bf16_tflops=117.0,
        peak_fp32_tflops=14.6,
        slm_size_per_core_kb=64,
        slm_banks=16,
        l3_cache_mb=16,
        subgroup_native=16,
    ),
    # Lunar Lake Xe2-LPG (Mobile Intel Graphics 0x64a0)
    "0x64a0": HardwareSpecs(
        name="Intel Graphics [0x64a0] (Lunar Lake)",
        arch="Intel Xe2-LPG",
        xe_cores=8,
        vector_engines=64,
        peak_dram_bw_gbps=67.2,
        peak_l3_bw_gbps=750.0,
        peak_bf16_tflops=46.8,
        peak_fp32_tflops=5.8,
        slm_size_per_core_kb=64,
        slm_banks=16,
        l3_cache_mb=8,
        subgroup_native=16,
    ),
    # Intel Arc A770 (Xe-HPG Alchemist)
    "a770": HardwareSpecs(
        name="Intel Arc A770",
        arch="Intel Xe-HPG",
        xe_cores=32,
        vector_engines=512,
        peak_dram_bw_gbps=560.0,
        peak_l3_bw_gbps=1200.0,
        peak_bf16_tflops=131.0,
        peak_fp32_tflops=19.7,
        slm_size_per_core_kb=64,
        slm_banks=16,
        l3_cache_mb=16,
        subgroup_native=16,
    ),
}


# ==============================================================================
# 2. Environment & Device Discovery
# ==============================================================================

def ensure_oneapi_env() -> dict[str, str]:
    """Ensure Intel oneAPI environment variables and dynamic libraries are loaded."""
    env = os.environ.copy()
    if "CMPLR_ROOT" in env or "ONEAPI_ROOT" in env:
        return env

    setvars = Path("/opt/intel/oneapi/setvars.sh")
    if not setvars.exists():
        return env

    cmd = f"source {setvars} --force 2>/dev/null && env"
    proc = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, text=True)
    if proc.returncode == 0:
        for line in proc.stdout.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                env[k] = v
    return env


def detect_target_hardware(build_dir: Path, env: dict[str, str], override_target: Optional[str] = None) -> HardwareSpecs:
    """Discover the active Intel GPU or return requested target specs."""
    if override_target:
        key = override_target.lower()
        for k, v in HW_SPECS_DATABASE.items():
            if key in k or key in v.name.lower():
                return v

    device_bin = build_dir / "strata-device"
    if device_bin.exists():
        try:
            proc = subprocess.run([str(device_bin)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            output = proc.stdout
            if "0x64a0" in output:
                return HW_SPECS_DATABASE["0x64a0"]
            if "B580" in output or "bmg" in output.lower():
                return HW_SPECS_DATABASE["b580"]
            if "A770" in output:
                return HW_SPECS_DATABASE["a770"]
        except Exception:
            pass

    # Default to Lunar Lake 0x64a0 if detected or fallback
    return HW_SPECS_DATABASE["0x64a0"]


# ==============================================================================
# 3. Kernel Workload & Roofline Mathematics
# ==============================================================================

@dataclass
class HardwareCounters:
    source: str = "sycl_event_timer"  # "intel_vtune_pmu" or "sycl_event_timer"
    eu_active_pct: Optional[float] = None
    eu_stalled_pct: Optional[float] = None
    eu_idle_pct: Optional[float] = None
    xmx_active_pct: Optional[float] = None
    slm_bank_conflicts_pct: Optional[float] = None
    l3_hit_rate_pct: Optional[float] = None
    scratch_bytes: Optional[int] = None


@dataclass
class KernelWorkload:
    name: str
    run_name: str
    time_ns: float
    cpu_time_ns: float
    iterations: int
    user_bandwidth_gbps: Optional[float] = None
    user_tflops: Optional[float] = None
    
    # Derived parameters
    k: int = 0
    n: int = 0
    batch: int = 1
    heads: int = 0
    precision_bytes: float = 2.0
    family: str = "generic"

    # Roofline metrics
    algorithmic_flops: float = 0.0
    algorithmic_bytes: float = 0.0
    arithmetic_intensity: float = 0.0
    theoretical_bw_ceiling_gbps: float = 0.0
    theoretical_tflops_ceiling: float = 0.0
    measured_bw_gbps: float = 0.0
    measured_tflops: float = 0.0
    efficiency_pct: float = 0.0
    gap_pct: float = 0.0
    regime: str = "Memory Bound"
    
    # Hardware performance counters
    counters: HardwareCounters = field(default_factory=HardwareCounters)

    # Mechanical prescriptions
    prescriptions: list[dict[str, str]] = field(default_factory=list)


def parse_benchmark_name_params(name: str) -> dict[str, Any]:
    """Extract matrix dimensions, heads, and quantization types from benchmark name."""
    params: dict[str, Any] = {
        "k": 2560,
        "n": 2560,
        "b": 1,
        "precision_bytes": 2.0,
        "family": "gemv",
    }

    # GDN Recurrence benchmarks
    if "GDN" in name:
        params["family"] = "gdn"
        m_gdn = re.search(r"/(\d+)/(\d+)", name)
        if m_gdn:
            params["heads"] = int(m_gdn.group(1))
            params["k"] = 128
            params["n"] = 128
        return params

    # Quantization weight formats
    if "Q4_K" in name or "Q4_0" in name:
        params["precision_bytes"] = 0.5625  # ~4.5 bits/weight with scale
    elif "Q6_K" in name:
        params["precision_bytes"] = 0.8125  # ~6.5 bits/weight
    elif "Q8_0" in name or "q8" in name.lower():
        params["precision_bytes"] = 1.0625  # 8 bits + scale
    elif "BF16" in name or "bf16" in name.lower():
        params["precision_bytes"] = 2.0

    # Dimension pattern /K/N/B/
    dims = [int(p) for p in re.findall(r"/(\d+)", name)]
    if len(dims) >= 2:
        params["k"] = dims[0]
        params["n"] = dims[1]
    if len(dims) >= 3:
        params["b"] = dims[2]

    if "SplitReduction" in name:
        params["family"] = "gemv_split"
    elif "Prefill" in name or "GEMM" in name:
        params["family"] = "gemm"
    elif "GEMV" in name:
        params["family"] = "gemv"

    return params


def compute_roofline_metrics(workload: KernelWorkload, hw: HardwareSpecs) -> None:
    """Calculate arithmetic intensity, ceilings, and efficiency against hardware specs."""
    k = max(workload.k, 1)
    n = max(workload.n, 1)
    b = max(workload.batch, 1)
    w_bytes = workload.precision_bytes

    if workload.family == "gdn":
        # Recurrent state: 16 heads of 128x128 FP32 (64 KB each) = 1 MB total
        heads = workload.heads if workload.heads > 0 else 16
        state_elements = heads * 128 * 128
        workload.algorithmic_flops = 4.0 * state_elements  # Multiplications + additions per step
        workload.algorithmic_bytes = state_elements * 4.0  # Read + write state
        workload.arithmetic_intensity = workload.algorithmic_flops / max(workload.algorithmic_bytes, 1.0)
    elif workload.family in ("gemv", "gemv_split"):
        workload.algorithmic_flops = 2.0 * k * n * b
        # Bytes = Weight matrix + Input vector + Output vector
        workload.algorithmic_bytes = (k * n * w_bytes) + (k * b * 2.0) + (n * b * 2.0)
        workload.arithmetic_intensity = workload.algorithmic_flops / max(workload.algorithmic_bytes, 1.0)
    elif workload.family == "gemm":
        workload.algorithmic_flops = 2.0 * k * n * b
        workload.algorithmic_bytes = (k * n * w_bytes) + (k * b * 2.0) + (n * b * 2.0)
        workload.arithmetic_intensity = workload.algorithmic_flops / max(workload.algorithmic_bytes, 1.0)
    else:
        workload.algorithmic_flops = 2.0 * k * n * b
        workload.algorithmic_bytes = (k * n * w_bytes) + (k * b * 2.0) + (n * b * 2.0)
        workload.arithmetic_intensity = workload.algorithmic_flops / max(workload.algorithmic_bytes, 1.0)

    # Measured Performance
    time_s = workload.time_ns * 1e-9
    if time_s > 0:
        if workload.user_bandwidth_gbps is not None and workload.user_bandwidth_gbps > 0:
            workload.measured_bw_gbps = workload.user_bandwidth_gbps
        else:
            workload.measured_bw_gbps = (workload.algorithmic_bytes / 1e9) / time_s

        if workload.user_tflops is not None and workload.user_tflops > 0:
            workload.measured_tflops = workload.user_tflops
        else:
            workload.measured_tflops = (workload.algorithmic_flops / 1e12) / time_s

    # Hardware Roofline Ceilings
    knee_point = (hw.peak_bf16_tflops * 1000.0) / hw.peak_dram_bw_gbps  # FLOPs/byte
    if workload.arithmetic_intensity < knee_point:
        workload.regime = "Memory Bound"
        workload.theoretical_bw_ceiling_gbps = hw.peak_dram_bw_gbps
        workload.theoretical_tflops_ceiling = (workload.arithmetic_intensity * hw.peak_dram_bw_gbps) / 1000.0
        workload.efficiency_pct = (workload.measured_bw_gbps / hw.peak_dram_bw_gbps) * 100.0
    else:
        workload.regime = "Compute Bound"
        workload.theoretical_bw_ceiling_gbps = (hw.peak_bf16_tflops * 1000.0) / workload.arithmetic_intensity
        workload.theoretical_tflops_ceiling = hw.peak_bf16_tflops
        workload.efficiency_pct = (workload.measured_tflops / hw.peak_bf16_tflops) * 100.0

    workload.efficiency_pct = min(max(workload.efficiency_pct, 0.0), 100.0)
    workload.gap_pct = 100.0 - workload.efficiency_pct


# ==============================================================================
# 4. Diagnostic & Prescription Engine
# ==============================================================================

def prescribe_optimizations(w: KernelWorkload, hw: HardwareSpecs) -> None:
    """Analyze gap to theoretical and assign concrete SYCL code modifications."""
    prescriptions: list[dict[str, str]] = []

    # 1. Narrow column under-occupancy
    if w.family == "gemv" and w.n <= 64 and w.gap_pct > 60.0:
        prescriptions.append({
            "lever": "Split-K Threads-Per-Row (TPR)",
            "observation": f"Narrow output dimension (N={w.n}) under-subscribes {hw.xe_cores} Xe-cores. Measured efficiency is only {w.efficiency_pct:.1f}%.",
            "action": "Switch dispatch from 1 subgroup per row to Split-K reduction (TPR=64 or 128) using SLM partial sums.",
            "code_ref": "cgh.parallel_for(sycl::nd_range<1>(N * TPR, LWS), ...); SLM reduction across TPR lanes.",
        })

    # 2. Memory bound GEMV with large weight footprint
    if w.regime == "Memory Bound" and w.k * w.n >= 2560 * 2560 and w.gap_pct > 20.0:
        prescriptions.append({
            "lever": "Intel Xe2 Streaming L3 Cache Hints",
            "observation": f"Single-use weights (size: {(w.k * w.n * w.precision_bytes)/1e6:.1f} MB) evict hot KV and activations from {hw.l3_cache_mb} MB L3 cache.",
            "action": "Wrap weight pointer in sycl::ext::intel::experimental::read_hint with cache_mode::streaming.",
            "code_ref": "annotated_ptr<const T, read_hint<cache_control<cache_mode::streaming, cache_level::L3>>> w_stream(w_ptr);",
        })

    # 3. Unfused recurrence round-trips
    if "Unfused" in w.name:
        prescriptions.append({
            "lever": "Shared Local Memory (SLM) Fusion",
            "observation": "Global memory round-trips for state tensor S (1.05 MB) saturate memory bus with intermediate writes.",
            "action": "Pin head state matrix directly in Xe-core 64 KB SLM, keeping state in SLM across step and RMSNorm.",
            "code_ref": "sycl::local_accessor<float, 2> s_state(sycl::range<2>(16, 16 + SLM_PAD), cgh);",
        })

    # 4. SLM padding for bank conflicts
    if "Fused_SLM" in w.name and "Padded" not in w.name:
        prescriptions.append({
            "lever": "SLM Bank Padding (SLM_PAD = 4)",
            "observation": f"Power-of-2 SLM tile rows collide with {hw.slm_banks} hardware memory banks, serializing SIMD16 access.",
            "action": "Add compile-time padding of 4 elements to leading SLM dimension: range<2>(16, 16 + 4).",
            "code_ref": "constexpr int SLM_PAD = 4; sycl::local_accessor<float, 2> s_tile(sycl::range<2>(16, 16 + SLM_PAD), cgh);",
        })

    w.prescriptions = prescriptions


# ==============================================================================
# 5. VTune CLI Hardware PMU Collection
# ==============================================================================

def find_vtune_binary() -> Optional[Path]:
    """Check PATH and standard oneAPI install locations for vtune binary."""
    which_path = shutil.which("vtune")
    if which_path:
        return Path(which_path)
    candidates = [
        Path("/opt/intel/oneapi/vtune/latest/bin64/vtune"),
        Path("/opt/intel/oneapi/vtune/latest/bin32/vtune"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def run_vtune_collection(
    vtune_bin: Path,
    target_bin: Path,
    filter_pattern: str,
    result_dir: Path,
    env: dict[str, str],
) -> Optional[HardwareCounters]:
    """Execute Intel VTune CLI to collect hardware PMU counters for GPU execution."""
    if result_dir.exists():
        shutil.rmtree(result_dir)

    collect_cmd = [
        str(vtune_bin),
        "-collect", "gpu-hotspots",
        "-knob", "profile-gpu-instructions=true",
        "-knob", "memory-bandwidth=true",
        "-result-dir", str(result_dir),
        "--",
        str(target_bin),
        f"--benchmark_filter={filter_pattern}",
        "--benchmark_min_time=0.05s",
    ]

    print(f"[VTune CLI] Executing: {' '.join(collect_cmd[:7])} ...")
    proc = subprocess.run(collect_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if proc.returncode != 0:
        return None

    # Generate summary CSV
    csv_report = result_dir / "summary.csv"
    report_cmd = [
        str(vtune_bin),
        "-report", "summary",
        "-result-dir", str(result_dir),
        "-format", "csv",
        "-report-output", str(csv_report),
    ]
    subprocess.run(report_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

    if not csv_report.exists():
        return None

    counters = HardwareCounters(source="intel_vtune_pmu")
    try:
        content = csv_report.read_text()
        for line in content.splitlines():
            line_l = line.lower()
            parts = [p.strip().replace("%", "") for p in line.split(",")]
            if "eu array active" in line_l:
                for p in parts[1:]:
                    try:
                        counters.eu_active_pct = float(p)
                        break
                    except ValueError:
                        continue
            elif "eu array stalled" in line_l:
                for p in parts[1:]:
                    try:
                        counters.eu_stalled_pct = float(p)
                        break
                    except ValueError:
                        continue
            elif "eu array idle" in line_l:
                for p in parts[1:]:
                    try:
                        counters.eu_idle_pct = float(p)
                        break
                    except ValueError:
                        continue
            elif "xmx active" in line_l:
                for p in parts[1:]:
                    try:
                        counters.xmx_active_pct = float(p)
                        break
                    except ValueError:
                        continue
            elif "slm bank conflict" in line_l:
                for p in parts[1:]:
                    try:
                        counters.slm_bank_conflicts_pct = float(p)
                        break
                    except ValueError:
                        continue
            elif "l3 cache hit" in line_l:
                for p in parts[1:]:
                    try:
                        counters.l3_hit_rate_pct = float(p)
                        break
                    except ValueError:
                        continue
    except Exception:
        pass

    return counters


# ==============================================================================
# 6. Benchmark Execution & Extraction
# ==============================================================================

def run_google_benchmark(
    bin_path: Path,
    filter_pattern: str,
    min_time: float,
    env: dict[str, str],
) -> list[KernelWorkload]:
    """Execute compiled Google Benchmark and extract workloads."""
    out_json = Path(f"/tmp/perf_bm_{int(time.time() * 1000)}.json")
    cmd = [
        str(bin_path),
        f"--benchmark_out={out_json}",
        "--benchmark_out_format=json",
        f"--benchmark_min_time={min_time}s",
    ]
    if filter_pattern:
        cmd.append(f"--benchmark_filter={filter_pattern}")

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    workloads: list[KernelWorkload] = []

    if not out_json.exists():
        return workloads

    try:
        with open(out_json, "r") as f:
            data = json.load(f)
        for bm in data.get("benchmarks", []):
            name = bm.get("name", "")
            time_ns = float(bm.get("real_time", 0.0))
            cpu_ns = float(bm.get("cpu_time", 0.0))
            iters = int(bm.get("iterations", 1))
            bw_gbps = bm.get("Bandwidth_GBps")
            tflops = bm.get("TFLOPS")

            params = parse_benchmark_name_params(name)
            w = KernelWorkload(
                name=name.split("/manual_time")[0],
                run_name=name,
                time_ns=time_ns,
                cpu_time_ns=cpu_ns,
                iterations=iters,
                user_bandwidth_gbps=float(bw_gbps) / 1e9 if bw_gbps else None,
                user_tflops=float(tflops) / 1e12 if tflops else None,
                k=params["k"],
                n=params["n"],
                batch=params["b"],
                heads=params.get("heads", 0),
                precision_bytes=params["precision_bytes"],
                family=params["family"],
            )
            workloads.append(w)
    finally:
        if out_json.exists():
            out_json.unlink()

    return workloads


def get_mock_workloads() -> list[KernelWorkload]:
    """Generate representative mock workloads for offline/dry-run analysis with hardware counters."""
    return [
        KernelWorkload(
            name="BM_Xe2_GEMV/2560/48/1",
            run_name="BM_Xe2_GEMV/2560/48/1/manual_time",
            time_ns=23598.0,
            cpu_time_ns=156680.0,
            iterations=3019,
            user_bandwidth_gbps=10.64,
            user_tflops=0.0104,
            k=2560,
            n=48,
            batch=1,
            precision_bytes=2.0,
            family="gemv",
            counters=HardwareCounters(
                source="intel_vtune_pmu",
                eu_active_pct=14.2,
                eu_stalled_pct=21.0,
                eu_idle_pct=64.8,
                xmx_active_pct=0.0,
                slm_bank_conflicts_pct=0.1,
                l3_hit_rate_pct=18.5,
                scratch_bytes=0,
            ),
        ),
        KernelWorkload(
            name="BM_Xe2_GEMV_SplitReduction/2560/48/1/128",
            run_name="BM_Xe2_GEMV_SplitReduction/2560/48/1/128/manual_time",
            time_ns=18230.0,
            cpu_time_ns=151466.0,
            iterations=3868,
            user_bandwidth_gbps=13.77,
            user_tflops=0.0135,
            k=2560,
            n=48,
            batch=1,
            precision_bytes=2.0,
            family="gemv_split",
            counters=HardwareCounters(
                source="intel_vtune_pmu",
                eu_active_pct=52.8,
                eu_stalled_pct=29.0,
                eu_idle_pct=18.2,
                xmx_active_pct=0.0,
                slm_bank_conflicts_pct=0.2,
                l3_hit_rate_pct=24.5,
                scratch_bytes=0,
            ),
        ),
        KernelWorkload(
            name="BM_Xe2_GEMV/2560/2560/1",
            run_name="BM_Xe2_GEMV/2560/2560/1/manual_time",
            time_ns=136800.0,
            cpu_time_ns=257800.0,
            iterations=800,
            user_bandwidth_gbps=95.92,
            user_tflops=0.0958,
            k=2560,
            n=2560,
            batch=1,
            precision_bytes=2.0,
            family="gemv",
            counters=HardwareCounters(
                source="intel_vtune_pmu",
                eu_active_pct=81.4,
                eu_stalled_pct=14.2,
                eu_idle_pct=4.4,
                xmx_active_pct=0.0,
                slm_bank_conflicts_pct=0.0,
                l3_hit_rate_pct=64.1,
                scratch_bytes=0,
            ),
        ),
        KernelWorkload(
            name="BM_Xe2_GDN_Unfused/16/16",
            run_name="BM_Xe2_GDN_Unfused/16/16/manual_time",
            time_ns=1066292.0,
            cpu_time_ns=1262720.0,
            iterations=55,
            user_bandwidth_gbps=2.00,
            user_tflops=0.002,
            k=128,
            n=128,
            batch=1,
            heads=16,
            precision_bytes=4.0,
            family="gdn",
            counters=HardwareCounters(
                source="intel_vtune_pmu",
                eu_active_pct=8.1,
                eu_stalled_pct=78.4,
                eu_idle_pct=13.5,
                xmx_active_pct=0.0,
                slm_bank_conflicts_pct=0.0,
                l3_hit_rate_pct=12.2,
                scratch_bytes=0,
            ),
        ),
        KernelWorkload(
            name="BM_Xe2_GDN_Fused_SLM/16/16",
            run_name="BM_Xe2_GDN_Fused_SLM/16/16/manual_time",
            time_ns=60555.0,
            cpu_time_ns=202328.0,
            iterations=1178,
            user_bandwidth_gbps=35.31,
            user_tflops=0.035,
            k=128,
            n=128,
            batch=1,
            heads=16,
            precision_bytes=4.0,
            family="gdn",
            counters=HardwareCounters(
                source="intel_vtune_pmu",
                eu_active_pct=48.2,
                eu_stalled_pct=34.1,
                eu_idle_pct=17.7,
                xmx_active_pct=0.0,
                slm_bank_conflicts_pct=6.8,
                l3_hit_rate_pct=72.0,
                scratch_bytes=0,
            ),
        ),
    ]


# ==============================================================================
# 6. Reporting & Terminal Dashboard
# ==============================================================================

def print_terminal_dashboard(workloads: list[KernelWorkload], hw: HardwareSpecs) -> None:
    """Print high-visibility terminal summary table."""
    has_pmu = any(w.counters.source == "intel_vtune_pmu" for w in workloads)
    source_str = "Hardware PMU Counters (via Intel VTune gpu-hotspots)" if has_pmu else "SYCL Event Timers (Software Throughput)"

    print("\n" + "=" * 105)
    print(f" INTEL XE/XE2 ROOFLINE ANALYSIS DASHBOARD")
    print(f" Target Device:      {hw.name} | Arch: {hw.arch}")
    print(f" Peak Silicon Limits: {hw.peak_dram_bw_gbps:.1f} GB/s Memory BW | {hw.peak_bf16_tflops:.1f} TFLOPS BF16 Compute")
    print(f" Measurement Source:  {source_str}")
    print("=" * 105)

    header = f"{'Kernel / Workload':<38} {'Time':>10} {'Intensity':>12} {'Ceiling':>14} {'Measured':>14} {'Efficiency':>12}"
    print(header)
    print("-" * 105)

    for w in workloads:
        time_str = f"{w.time_ns / 1000.0:.2f} µs" if w.time_ns < 1e6 else f"{w.time_ns / 1e6:.2f} ms"
        int_str = f"{w.arithmetic_intensity:.2f} FLOP/B"

        if w.regime == "Memory Bound":
            ceil_str = f"{w.theoretical_bw_ceiling_gbps:.1f} GB/s"
            meas_str = f"{w.measured_bw_gbps:.2f} GB/s"
        else:
            ceil_str = f"{w.theoretical_tflops_ceiling:.1f} TFLOPS"
            meas_str = f"{w.measured_tflops:.2f} TFLOPS"

        eff_str = f"{w.efficiency_pct:.1f}%"
        status = "🔴" if w.gap_pct > 50 else ("🟡" if w.gap_pct > 25 else "🟢")

        print(f"{w.name[:37]:<38} {time_str:>10} {int_str:>12} {ceil_str:>14} {meas_str:>14} {status} {eff_str:>10}")

    if has_pmu:
        print("\n" + "-" * 105)
        print(" HARDWARE PMU COUNTER DETAILS (Intel VTune gpu-hotspots)")
        print("-" * 105)
        pmu_hdr = f"{'Kernel / Workload':<38} {'EU Active':>12} {'EU Stalled':>12} {'EU Idle':>12} {'SLM Conf':>12} {'L3 Hit':>12}"
        print(pmu_hdr)
        print("-" * 105)
        for w in workloads:
            c = w.counters
            act = f"{c.eu_active_pct:.1f}%" if c.eu_active_pct is not None else "-"
            stl = f"{c.eu_stalled_pct:.1f}%" if c.eu_stalled_pct is not None else "-"
            idl = f"{c.eu_idle_pct:.1f}%" if c.eu_idle_pct is not None else "-"
            slm = f"{c.slm_bank_conflicts_pct:.1f}%" if c.slm_bank_conflicts_pct is not None else "-"
            l3 = f"{c.l3_hit_rate_pct:.1f}%" if c.l3_hit_rate_pct is not None else "-"
            print(f"{w.name[:37]:<38} {act:>12} {stl:>12} {idl:>12} {slm:>12} {l3:>12}")

    print("=" * 105)


def print_comparison_dashboard(b_workload: KernelWorkload, t_workload: KernelWorkload, hw: HardwareSpecs) -> None:
    """Display side-by-side delta between baseline and tuned kernel."""
    speedup = b_workload.time_ns / max(t_workload.time_ns, 1.0)
    bw_delta = t_workload.measured_bw_gbps - b_workload.measured_bw_gbps
    bw_pct = (bw_delta / max(b_workload.measured_bw_gbps, 1e-6)) * 100.0
    eff_delta = t_workload.efficiency_pct - b_workload.efficiency_pct
    gap_reduction = b_workload.gap_pct - t_workload.gap_pct

    print("\n" + "=" * 90)
    print(f" MECHANICAL OPTIMIZATION DELTA REPORT")
    print(f" Baseline: {b_workload.name}")
    print(f" Tuned:    {t_workload.name}")
    print("=" * 90)
    print(f"{'Metric':<30} {'Baseline':>16} {'Tuned':>16} {'Delta / Impact':>24}")
    print("-" * 90)

    b_time = f"{b_workload.time_ns / 1000.0:.2f} µs"
    t_time = f"{t_workload.time_ns / 1000.0:.2f} µs"
    time_delta = f"-{(1.0 - 1.0 / speedup) * 100.0:.1f}% ({speedup:.2f}x speedup)"
    print(f"{'Execution Latency':<30} {b_time:>16} {t_time:>16} {time_delta:>24}")

    b_bw = f"{b_workload.measured_bw_gbps:.2f} GB/s"
    t_bw = f"{t_workload.measured_bw_gbps:.2f} GB/s"
    bw_impact = f"+{bw_pct:.1f}% ({bw_delta:+.2f} GB/s)"
    print(f"{'Effective Bandwidth':<30} {b_bw:>16} {t_bw:>16} {bw_impact:>24}")

    b_eff = f"{b_workload.efficiency_pct:.1f}%"
    t_eff = f"{t_workload.efficiency_pct:.1f}%"
    eff_impact = f"{eff_delta:+.1f}% of hardware peak"
    print(f"{'Roofline Efficiency':<30} {b_eff:>16} {t_eff:>16} {eff_impact:>24}")

    b_gap = f"{b_workload.gap_pct:.1f}%"
    t_gap = f"{t_workload.gap_pct:.1f}%"
    gap_impact = f"{gap_reduction:+.1f}% gap closed"
    print(f"{'Gap to Theoretical Peak':<30} {b_gap:>16} {t_gap:>16} {gap_impact:>24}")
    print("=" * 90)


def generate_markdown_report(workloads: list[KernelWorkload], hw: HardwareSpecs, report_path: Path) -> None:
    """Generate detailed markdown report with tables and recommendations."""
    lines: list[str] = [
        "# Intel Xe/Xe2 Hardware-Counter & Roofline Optimization Report",
        "",
        f"- **Generated At**: `{datetime.datetime.now(datetime.timezone.utc).isoformat()}`",
        f"- **Target Hardware**: `{hw.name}` (`{hw.arch}`)",
        f"- **Compute Capacity**: `{hw.xe_cores} Xe-cores` (`{hw.vector_engines} Vector Engines`), Subgroup Size `{hw.subgroup_native}`",
        f"- **Physical Ceilings**: `{hw.peak_dram_bw_gbps:.1f} GB/s` Memory Bandwidth, `{hw.peak_bf16_tflops:.1f} TFLOPS` BF16 XMX Compute",
        "",
        "---",
        "",
        "## 1. Kernel Roofline Efficiency Matrix",
        "",
        "| Kernel / Workload | Time (µs) | Arithmetic Intensity | Regime | Theoretical Ceiling | Measured Perf | **Efficiency (%)** | **Gap to Peak** |",
        "|---|---:|---:|:---:|---:|---:|---:|---:|",
    ]

    for w in workloads:
        time_us = w.time_ns / 1000.0
        ceil_str = f"{w.theoretical_bw_ceiling_gbps:.1f} GB/s" if w.regime == "Memory Bound" else f"{w.theoretical_tflops_ceiling:.1f} TFLOPS"
        meas_str = f"{w.measured_bw_gbps:.2f} GB/s" if w.regime == "Memory Bound" else f"{w.measured_tflops:.2f} TFLOPS"
        status = "🔴" if w.gap_pct > 50 else ("🟡" if w.gap_pct > 25 else "🟢")
        lines.append(
            f"| `{w.name}` | {time_us:.2f} µs | {w.arithmetic_intensity:.2f} FLOP/B | {w.regime} | {ceil_str} | {meas_str} | {status} **{w.efficiency_pct:.1f}%** | **{w.gap_pct:.1f}%** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Mechanical SYCL Optimization Prescriptions",
        "",
    ])

    prescribed_any = False
    for w in workloads:
        if w.prescriptions:
            prescribed_any = True
            lines.append(f"### Kernel: `{w.name}` (Gap to Theoretical: **{w.gap_pct:.1f}%**)")
            lines.append("")
            for p in w.prescriptions:
                lines.append(f"- **Lever**: `{p['lever']}`")
                lines.append(f"  - *Observation*: {p['observation']}")
                lines.append(f"  - *Prescribed Fix*: {p['action']}")
                lines.append(f"  - *SYCL Implementation*: `{p['code_ref']}`")
                lines.append("")

    if not prescribed_any:
        lines.append("All analyzed kernels are operating near peak physical limits (Gap < 25%). No urgent architectural modifications required.")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Intel VTune CLI Deep-Dive Profiling Recipes",
        "",
        "To inspect hardware instruction and memory stalls on target hardware, run:",
        "",
        "```bash",
        "# 1. Profile instruction hotspots and execution stalls:",
        f"vtune -collect gpu-hotspots -knob profile-gpu-instructions=true -- ./build_sycl/perf_xe2_gemv",
        "",
        "# 2. Profile memory subsystem and SLM bank conflicts:",
        f"vtune -collect gpu-hotspots -knob memory-bandwidth=true -- ./build_sycl/perf_fused_gdn",
        "```",
    ])

    report_path.write_text("\n".join(lines))
    print(f"\n[Report] Markdown report successfully written to: {report_path.resolve()}")


# ==============================================================================
# 7. Main CLI Entry Point
# ==============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Intel Xe/Xe2 Hardware-Counter & Roofline Optimization Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="perf_xe2_gemv",
        help="Google Benchmark binary in build_sycl/ (e.g. perf_xe2_gemv, perf_fused_gdn)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="Regex filter pattern passed to Google Benchmark --benchmark_filter",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Force hardware target (b580, 0x64a0, a770)",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE_REGEX", "TUNED_REGEX"),
        help="Compare two benchmark filter patterns and output optimization delta",
    )
    parser.add_argument(
        "--min-time",
        type=float,
        default=0.05,
        help="Minimum execution duration per benchmark iteration",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Path to export Markdown optimization report",
    )
    parser.add_argument(
        "--vtune",
        action="store_true",
        help="Collect real hardware PMU counters via Intel VTune CLI (requires 'vtune')",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run offline mock analysis using representative captured traces",
    )

    args = parser.parse_args()
    build_dir = ROOT / "build_sycl"

    # Environment & Device Resolution
    env = ensure_oneapi_env()
    hw = detect_target_hardware(build_dir, env, override_target=args.target)

    # Check VTune CLI availability if requested
    vtune_bin = find_vtune_binary()
    if args.vtune and not vtune_bin:
        print("\n" + "!" * 85, file=sys.stderr)
        print("[-] VTune CLI Error: 'vtune' executable not found on system PATH.", file=sys.stderr)
        print("    Hardware PMU counters (EU Active, Stalled, SLM Bank Conflicts) require Intel VTune.", file=sys.stderr)
        print("", file=sys.stderr)
        print("    To install Intel VTune Profiler on Linux:", file=sys.stderr)
        print("      Ubuntu/Debian: sudo apt install intel-oneapi-vtune", file=sys.stderr)
        print("      RHEL/Fedora:   sudo dnf install intel-oneapi-vtune", file=sys.stderr)
        print("", file=sys.stderr)
        print("    If already installed, source your oneAPI VTune environment:", file=sys.stderr)
        print("      source /opt/intel/oneapi/vtune/latest/env/vars.sh", file=sys.stderr)
        print("!" * 85 + "\n", file=sys.stderr)
        return 1

    workloads: list[KernelWorkload] = []

    if args.mock:
        print(f"[Mock Mode] Loading representative workloads for {hw.name}...")
        workloads = get_mock_workloads()
    elif args.compare:
        b_filter, t_filter = args.compare
        bin_path = build_dir / args.benchmark
        if not bin_path.exists():
            print(f"Error: Benchmark executable not found at {bin_path}", file=sys.stderr)
            return 1

        print(f"[Running Baseline] {bin_path.name} with filter: '{b_filter}'...")
        b_workloads = run_google_benchmark(bin_path, b_filter, args.min_time, env)
        print(f"[Running Tuned]    {bin_path.name} with filter: '{t_filter}'...")
        t_workloads = run_google_benchmark(bin_path, t_filter, args.min_time, env)

        if not b_workloads or not t_workloads:
            print("Error: Failed to collect benchmarks for comparison.", file=sys.stderr)
            return 1

        b_w = b_workloads[0]
        t_w = t_workloads[0]
        compute_roofline_metrics(b_w, hw)
        compute_roofline_metrics(t_w, hw)
        prescribe_optimizations(b_w, hw)
        prescribe_optimizations(t_w, hw)

        print_comparison_dashboard(b_w, t_w, hw)
        if args.report:
            generate_markdown_report([b_w, t_w], hw, args.report)
        return 0
    else:
        bin_path = build_dir / args.benchmark
        if not bin_path.exists():
            print(f"Error: Benchmark executable not found at {bin_path}", file=sys.stderr)
            return 1

        print(f"[Running Benchmark] {bin_path.name} (Filter: '{args.filter}')...")
        workloads = run_google_benchmark(bin_path, args.filter, args.min_time, env)
        if not workloads:
            print("No benchmarks matched the specified filter.", file=sys.stderr)
            return 1

        # Collect VTune hardware PMU counters if requested
        if args.vtune and vtune_bin:
            for idx, w in enumerate(workloads):
                vtune_dir = Path(f"/tmp/vtune_res_{int(time.time())}_{idx}")
                print(f"[VTune Profiler] Collecting hardware PMU counters for {w.name}...")
                counters = run_vtune_collection(vtune_bin, bin_path, w.name, vtune_dir, env)
                if counters:
                    w.counters = counters

    # Compute Roofline metrics & Prescriptions
    for w in workloads:
        compute_roofline_metrics(w, hw)
        prescribe_optimizations(w, hw)

    # Sort by gap percentage (Hotspot priority)
    workloads.sort(key=lambda x: x.gap_pct, reverse=True)

    print_terminal_dashboard(workloads, hw)

    # Print prescriptions
    has_prescriptions = any(w.prescriptions for w in workloads)
    if has_prescriptions:
        print("\n" + "=" * 90)
        print(" ACTIONABLE MECHANICAL SYCL OPTIMIZATIONS")
        print("=" * 90)
        for w in workloads:
            if w.prescriptions:
                print(f"\n▶ Workload: {w.name} (Roofline Gap: {w.gap_pct:.1f}%)")
                for p in w.prescriptions:
                    print(f"  • Lever:        {p['lever']}")
                    print(f"    Observation:  {p['observation']}")
                    print(f"    Prescription: {p['action']}")
                    print(f"    SYCL Syntax:  {p['code_ref']}")
        print("=" * 90)

    if args.report:
        generate_markdown_report(workloads, hw, args.report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
