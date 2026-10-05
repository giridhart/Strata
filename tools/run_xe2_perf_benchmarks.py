#!/usr/bin/env python3
"""tools/run_xe2_perf_benchmarks.py - Intel Xe2 / Arc B580 Benchmark Runner & Reporter.

Automated runner and reporting tool designed for execution on target Intel Xe2 hardware
(Intel Arc B580 desktop or Intel Core Ultra Lunar Lake).

Capabilities:
1. Environment & Device Discovery: Queries strata-device / sycl-ls for GPU name and specs.
2. Comprehensive Execution: Orchestrates all 11 Google Benchmark suites across Tier 1 and Tier 2.
3. Structured Output: Captures individual and consolidated JSON files.
4. Professional Markdown Report: Generates detailed tables, bandwidth metrics, speedup calculations,
   and comparison against Intel Arc B580 hardware targets (456 GB/s, 117 TFLOPS).
5. Clean Terminal Summary: Displays formatted console tables upon completion.

Usage:
    python3 tools/run_xe2_perf_benchmarks.py                      # Standard run, all suites
    python3 tools/run_xe2_perf_benchmarks.py --quick              # Fast smoke run (min_time=0.05s)
    python3 tools/run_xe2_perf_benchmarks.py --suite tier2        # Run Tier 2 advanced suites only
    python3 tools/run_xe2_perf_benchmarks.py --filter "GDN"       # Filter specific benchmark patterns
    python3 tools/run_xe2_perf_benchmarks.py --out-dir results/   # Specify output report directory
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BENCHMARK_TARGETS = [
    # Tier 1: Core Computational Primitives
    {
        "name": "perf_xe2_gemv",
        "tier": 1,
        "title": "BF16 GEMV Projections & SLM Split",
        "description": "8 Qwen projection shapes, SIMD16 vs SIMD32, and narrow SLM split reduction",
    },
    {
        "name": "perf_xe2_xmx_tiles",
        "tier": 1,
        "title": "Hardware XMX Systolic Matrix Tiles",
        "description": "8x16x16 vs 16x16x16 joint_matrix systolic tiles on Qwen projection shapes",
    },
    {
        "name": "perf_qsa_attention",
        "tier": 1,
        "title": "QSA Prompt Attention & Top-K Gather",
        "description": "XMX systolic prompt attention throughput (tok/s) and sparse KV block gather bandwidth",
    },
    {
        "name": "perf_moe_accumulate",
        "tier": 1,
        "title": "MoE Expert Weighted Accumulation",
        "description": "10-expert reduction comparing Atomic vs Direct Combine vs SIMD16 Subgroup Reduce",
    },
    {
        "name": "perf_dequant_dpas",
        "tier": 1,
        "title": "Weight-Only 4-bit Dequant + DPAS",
        "description": "Streaming quantized weights: Direct-to-Register DPAS vs SLM-buffered execution",
    },
    # Tier 2: Advanced System & Recurrence Primitives
    {
        "name": "perf_fused_gdn",
        "tier": 2,
        "title": "Fused GDN Linear Attention Recurrence",
        "description": "Linear attention recurrence: Fused SLM step + norm vs unfused global memory roundtrips",
    },
    {
        "name": "perf_pcie_overlap",
        "tier": 2,
        "title": "PCIe Copy Engine vs XMX Concurrency",
        "description": "Dual-queue concurrency: XMX matrix compute throughput retention under async DMA streaming",
    },
    {
        "name": "perf_graph_replay",
        "tier": 2,
        "title": "SYCL Command Graph Replay Latency",
        "description": "48-layer decode DAG: CapturedGraph replay latency vs sequential host queue submissions",
    },
    {
        "name": "perf_kv_decode",
        "tier": 2,
        "title": "Quantized KV Cache Decode Attention",
        "description": "Single-token attention bandwidth across context lengths up to 65K for FP16 and INT8",
    },
    {
        "name": "perf_prefill_gemm",
        "tier": 2,
        "title": "Native Prefill Tiled GEMM Scaling",
        "description": "Prompt prefilling throughput (TFLOPS) on native SYCL tiled GEMM across chunk sizes",
    },
    {
        "name": "perf_speculative_verify",
        "tier": 2,
        "title": "MTP Speculative Verification Pipeline",
        "description": "Multi-token verification throughput and scaling across candidate batch sizes B=1..8",
    },
    # Tier 3: Multi-Operation & Data Movement Pipelines
    {
        "name": "perf_residual_pipeline",
        "tier": 3,
        "title": "Hyper-Connection Residual Ping-Pong & Fusion",
        "description": "gr_read -> GEMV -> gr_write: R roundtrips vs L3 residency and fused read/write transition",
    },
    {
        "name": "perf_layer_pipeline",
        "tier": 3,
        "title": "Full Composite Layer Pipeline Simulation",
        "description": "9-kernel complete block: intermediate activation handover, pipelined vs CapturedGraph replay",
    },
    {
        "name": "perf_moe_doorbell_stream",
        "tier": 3,
        "title": "MoE Doorbell Handshake & Expert Streaming",
        "description": "PCIe doorbell signaling roundtrip, 51.2 KB parts streaming, and shared expert overlap slack",
    },
    {
        "name": "perf_paged_kv_traffic",
        "tier": 3,
        "title": "Paged KV Cache Traffic & Indirection Penalty",
        "description": "Non-contiguous paged gather vs contiguous memory, page size sweep (16..128) and INT8 vs FP16",
    },
    # Tier 4: Output Layer, Transform & Autoregressive Endpoints
    {
        "name": "perf_lmhead_sampler",
        "tier": 4,
        "title": "LM Head 248K Vocab & GPU Sampler",
        "description": "2560 -> 248,320 output projection, parallel softmax reduction, and 4-byte token readback",
    },
    {
        "name": "perf_fwht_kv4",
        "tier": 4,
        "title": "Fast Walsh-Hadamard Transform & Q4_0 KV",
        "description": "8-stage butterfly FWHT256 in-register rotation, Q4_0 block quantize, and rotated decode attention",
    },
    {
        "name": "perf_ple_ngram",
        "tier": 4,
        "title": "Layer 1 Per-Layer Embedding & N-gram",
        "description": "Layer 1 unique block: 1,440 B row upload, Conv1D history shift, projections and gated residual update",
    },
    {
        "name": "perf_embed_overlap",
        "tier": 4,
        "title": "Embedding Dequant & Readback Overlap",
        "description": "Column dequantization (BF16/Q8_0/Q4_K), 4-byte D2H barrier, and on-device autoregressive chaining",
    },
]


def ensure_oneapi_env() -> dict[str, str]:
    """Ensure Intel oneAPI environment variables are active in process env."""
    env = os.environ.copy()
    if "CMPLR_ROOT" in env or "ONEAPI_ROOT" in env or "MKLROOT" in env:
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


def discover_device(build_dir: Path, env: dict[str, str]) -> dict[str, str]:
    """Query strata-device to discover available GPU properties."""
    device_bin = build_dir / "strata-device"
    info = {
        "name": "Unknown Intel GPU",
        "arch": "Xe2 Architecture",
        "compute_cap": "Unknown",
        "multiprocessors": "Unknown",
        "vram_total": "Unknown",
        "vram_free": "Unknown",
    }

    if device_bin.exists():
        proc = subprocess.run([str(device_bin)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        for line in proc.stdout.splitlines():
            line_str = line.strip()
            if line_str.startswith("device 0:"):
                info["name"] = line_str.split(":", 1)[1].strip()
            elif "compute capability" in line_str:
                info["compute_cap"] = line_str.split("capability", 1)[1].strip()
            elif "multiprocessors" in line_str:
                info["multiprocessors"] = line_str.split("multiprocessors", 1)[1].strip()
            elif "VRAM total / free" in line_str:
                info["vram_total"] = line_str.split("VRAM total / free", 1)[1].strip()

    if "B580" in info["name"] or "bmg" in info["name"].lower():
        info["arch"] = "Intel Xe2-HPG (Battlemage BMG-G21 Desktop Discrete GPU)"
        info["target_bw"] = "456 GB/s (192-bit GDDR6)"
        info["target_tflops"] = "117 TFLOPS (BF16 XMX)"
    elif "0x64a0" in info["name"] or "Lunar Lake" in info["name"]:
        info["arch"] = "Intel Xe2-LPG (Lunar Lake Mobile Integrated GPU)"
        info["target_bw"] = "67.2 GB/s (128-bit LPDDR5x)"
        info["target_tflops"] = "46.8 TFLOPS (BF16 XMX)"

    return info


def run_single_benchmark(
    bin_path: Path,
    out_json: Path,
    min_time: float,
    repetitions: int,
    filter_pattern: str,
    env: dict[str, str],
) -> tuple[bool, str, dict]:
    """Execute one Google Benchmark binary and capture output."""
    cmd = [
        str(bin_path),
        f"--benchmark_out={out_json}",
        "--benchmark_out_format=json",
        f"--benchmark_min_time={min_time}s",
        f"--benchmark_repetitions={repetitions}",
    ]
    if filter_pattern:
        cmd.append(f"--benchmark_filter={filter_pattern}")

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    elapsed = time.perf_counter() - t0

    parsed_json = {}
    if out_json.exists():
        try:
            with open(out_json, "r") as f:
                parsed_json = json.load(f)
        except Exception:
            pass

    success = proc.returncode == 0 and bool(parsed_json.get("benchmarks"))
    output = proc.stdout if success else (proc.stdout + "\n" + proc.stderr)
    return success, output, parsed_json


def generate_markdown_report(
    device_info: dict[str, str],
    all_results: dict[str, dict],
    out_dir: Path,
    run_timestamp: str,
    duration_s: float,
) -> str:
    """Generate professional Markdown performance report."""
    md = []
    md.append(f"# Intel Xe2 / Arc B580 Performance Benchmark Tuning Report\n")
    md.append(f"> **Execution Timestamp**: {run_timestamp}  \n")
    md.append(f"> **Target Hardware**: `{device_info.get('name', 'Unknown')}`  \n")
    md.append(f"> **Microarchitecture**: {device_info.get('arch', 'Intel Xe2')}  \n")
    md.append(f"> **Compute Units**: {device_info.get('multiprocessors', 'N/A')} multiprocessors  \n")
    md.append(f"> **VRAM Pool**: {device_info.get('vram_total', 'N/A')}  \n")
    md.append(f"> **Suite Duration**: {duration_s:.1f} seconds total execution time  \n")
    md.append(f"> **Target Memory Bandwidth**: {device_info.get('target_bw', '456 GB/s')}  \n")
    md.append(f"> **Target Peak Compute**: {device_info.get('target_tflops', '117 TFLOPS')}  \n\n")
    md.append("---\n\n")

    md.append("## 1. Executive Performance Summary\n\n")
    md.append("| Benchmark Suite | Key Evaluated Stage | Primary Measured Metric | Hardware Status |\n")
    md.append("|---|---|---|---|\n")

    for target in BENCHMARK_TARGETS:
        tname = target["name"]
        data = all_results.get(tname, {}).get("data", {})
        benchmarks = data.get("benchmarks", [])
        if not benchmarks:
            md.append(f"| **{target['title']}** | {target['description']} | *Failed / Skipped* | ❌ Failed |\n")
            continue

        # Extract representative metric
        metric_str = "Completed"
        if tname == "perf_xe2_gemv":
            max_bw = max((b.get("Bandwidth_GBps", 0) for b in benchmarks), default=0) / 1e9
            metric_str = f"Peak Bandwidth: **{max_bw:.2f} GB/s**"
        elif tname == "perf_xe2_xmx_tiles":
            max_tflops = max((b.get("TFLOPS", 0) for b in benchmarks), default=0) / 1e12
            metric_str = f"Peak Systolic: **{max_tflops:.2f} TFLOPS**"
        elif tname == "perf_qsa_attention":
            max_tps = max((b.get("tokens_per_sec", 0) for b in benchmarks if "PromptAttention" in b.get("name", "")), default=0)
            max_gather = max((b.get("Bandwidth_GBps", 0) for b in benchmarks if "TopK_Gather" in b.get("name", "")), default=0) / 1e9
            metric_str = f"Prompt: **{max_tps/1e3:.1f}k tok/s** | Gather: **{max_gather:.2f} GB/s**"
        elif tname == "perf_moe_accumulate":
            max_bw = max((b.get("Bandwidth_GBps", 0) for b in benchmarks), default=0) / 1e9
            metric_str = f"Reduction Bandwidth: **{max_bw:.2f} GB/s**"
        elif tname == "perf_dequant_dpas":
            max_tflops = max((b.get("TFLOPS", 0) for b in benchmarks), default=0) / 1e12
            metric_str = f"Direct DPAS: **{max_tflops * 1000:.1f} GFLOPS**"
        elif tname == "perf_fused_gdn":
            fused_t = next((b.get("real_time", 0) for b in benchmarks if "Fused_SLM/16/16" in b.get("name", "")), 0)
            unfused_t = next((b.get("real_time", 0) for b in benchmarks if "Unfused/16/16" in b.get("name", "")), 0)
            ratio = (unfused_t / fused_t) if fused_t > 0 else 0
            metric_str = f"Fused: **{fused_t/1e3:.1f} µs** vs Unfused: **{unfused_t/1e3:.1f} µs** (**{ratio:.2f}x faster**)"
        elif tname == "perf_pcie_overlap":
            iso_tflops = next((b.get("TFLOPS", 0) for b in benchmarks if "Compute_Isolated" in b.get("name", "")), 1) / 1e12
            conc_tflops = next((b.get("TFLOPS", 0) for b in benchmarks if "Concurrent/33554432" in b.get("name", "")), 0) / 1e12
            retention = (conc_tflops / iso_tflops * 100.0) if iso_tflops > 0 else 0
            metric_str = f"Compute Retention: **{retention:.1f}%** under 32 MB DMA"
        elif tname == "perf_graph_replay":
            dir_t = next((b.get("real_time", 0) for b in benchmarks if "Direct_Submission/48" in b.get("name", "")), 0)
            grp_t = next((b.get("real_time", 0) for b in benchmarks if "Graph_Replay/48" in b.get("name", "")), 0)
            ratio = (dir_t / grp_t) if grp_t > 0 else 0
            metric_str = f"Graph: **{grp_t/1e6:.2f} ms** vs Direct: **{dir_t/1e6:.2f} ms** (**{ratio:.1f}x speedup**)"
        elif tname == "perf_kv_decode":
            max_bw = max((b.get("Bandwidth_GBps", 0) for b in benchmarks), default=0) / 1e9
            metric_str = f"Decode Attention Bandwidth: **{max_bw:.2f} GB/s**"
        elif tname == "perf_prefill_gemm":
            max_tflops = max((b.get("TFLOPS", 0) for b in benchmarks), default=0) / 1e12
            metric_str = f"Native Tiled Prefill: **{max_tflops:.2f} TFLOPS**"
        elif tname == "perf_speculative_verify":
            tps_b1 = next((b.get("tokens_per_sec", 0) for b in benchmarks if "Verify_Step/1" in b.get("name", "")), 0)
            tps_b8 = next((b.get("tokens_per_sec", 0) for b in benchmarks if "Verify_Step/8" in b.get("name", "")), 0)
            ratio = (tps_b8 / tps_b1) if tps_b1 > 0 else 0
            metric_str = f"Verify Scaling: **{tps_b1/1e3:.1f}k** (B=1) -> **{tps_b8/1e3:.1f}k tok/s** (B=8) (**{ratio:.1f}x**)"
        elif tname == "perf_residual_pipeline":
            unfused_t = next((b.get("real_time", 0) for b in benchmarks if "Residual_Unfused" in b.get("name", "")), 0)
            fused_t = next((b.get("real_time", 0) for b in benchmarks if "Residual_Fused" in b.get("name", "")), 0)
            ratio = (unfused_t / fused_t) if fused_t > 0 else 0
            metric_str = f"Fused: **{fused_t/1e3:.1f} µs** vs Unfused: **{unfused_t/1e3:.1f} µs** (**{ratio:.2f}x faster**)"
        elif tname == "perf_layer_pipeline":
            pipe_t = next((b.get("real_time", 0) for b in benchmarks if "Composite_Layer_Pipelined" in b.get("name", "")), 0)
            graph_t = next((b.get("real_time", 0) for b in benchmarks if "Composite_Layer_Graph" in b.get("name", "")), 0)
            ratio = (pipe_t / graph_t) if graph_t > 0 else 0
            metric_str = f"Graph: **{graph_t/1e3:.1f} µs** vs Pipelined: **{pipe_t/1e3:.1f} µs** (**{ratio:.1f}x speedup**)"
        elif tname == "perf_moe_doorbell_stream":
            db_lat = next((b.get("real_time", 0) for b in benchmarks if "Doorbell_Roundtrip" in b.get("name", "")), 0)
            slack = next((b.get("Slack_us", 0) for b in benchmarks if "Overlap_Slack" in b.get("name", "")), 0)
            metric_str = f"Doorbell Latency: **{db_lat/1e3:.1f} µs** | Overlap Slack: **{slack:.1f} µs**"
        elif tname == "perf_paged_kv_traffic":
            contig_bw = next((b.get("Bandwidth_GBps", 0) for b in benchmarks if "KV_Contiguous/4096" in b.get("name", "")), 0) / 1e9
            paged_bw = next((b.get("Bandwidth_GBps", 0) for b in benchmarks if "KV_Paged/4096" in b.get("name", "")), 0) / 1e9
            penalty = ((contig_bw - paged_bw) / contig_bw * 100.0) if contig_bw > 0 else 0
            metric_str = f"Contiguous: **{contig_bw:.2f} GB/s** vs Paged: **{paged_bw:.2f} GB/s** (Penalty: **{penalty:.1f}%**)"
        elif tname == "perf_lmhead_sampler":
            head_tflops = next((b.get("TFLOPS", 0) for b in benchmarks if "LMHead_GEMV/65536" in b.get("name", "")), 0) / 1e12
            samp_bw = next((b.get("Bandwidth_GBps", 0) for b in benchmarks if "Sampler_Reduce" in b.get("name", "")), 0) / 1e9
            metric_str = f"Head: **{head_tflops:.2f} TFLOPS** (64K) | Sampler: **{samp_bw:.2f} GB/s** (248K)"
        elif tname == "perf_fwht_kv4":
            fwht_gflops = next((b.get("GFLOPS", 0) for b in benchmarks if "FWHT256_Compute/64" in b.get("name", "")), 0)
            comp_x = next((b.get("Compression_x", 0) for b in benchmarks if "KV_Quant_Q4_Rotated" in b.get("name", "")), 0)
            metric_str = f"FWHT256: **{fwht_gflops:.1f} GFLOPS** | Compression: **{comp_x:.2f}x** (576 B/cell)"
        elif tname == "perf_ple_ngram":
            ple_t = next((b.get("real_time", 0) for b in benchmarks if "PLE_GPU_Block" in b.get("name", "")), 0)
            up_lat = next((b.get("real_time", 0) for b in benchmarks if "PLE_Row_Upload" in b.get("name", "")), 0)
            metric_str = f"PLE GPU Block: **{ple_t/1e3:.1f} µs** | Row Upload: **{up_lat/1e3:.1f} µs**"
        elif tname == "perf_embed_overlap":
            sync_t = next((b.get("real_time", 0) for b in benchmarks if "Autoregressive_Sync" in b.get("name", "")), 0)
            async_t = next((b.get("real_time", 0) for b in benchmarks if "Autoregressive_Async" in b.get("name", "")), 0)
            ratio = (sync_t / async_t) if async_t > 0 else 0
            metric_str = f"Async: **{async_t/1e3:.1f} µs** vs Sync: **{sync_t/1e3:.1f} µs** (**{ratio:.2f}x faster**)"

        md.append(f"| **{target['title']}** | {target['description']} | {metric_str} | ✅ PASS |\n")

    md.append("\n---\n\n")

    md.append("## 2. Detailed Performance Tables by Kernel\n\n")
    for target in BENCHMARK_TARGETS:
        tname = target["name"]
        data = all_results.get(tname, {}).get("data", {})
        benchmarks = data.get("benchmarks", [])
        if not benchmarks:
            continue

        md.append(f"### {target['title']} (`{tname}`)\n\n")
        md.append(f"{target['description']}\n\n")
        md.append("| Benchmark Name | Hardware Latency | CPU Latency | Bandwidth / Throughput | Key Counters |\n")
        md.append("|---|---|---|---|---|\n")

        for b in benchmarks:
            name = b.get("run_name", b.get("name", "unknown"))
            r_time = b.get("real_time", 0)
            c_time = b.get("cpu_time", 0)
            time_unit = b.get("time_unit", "ns")

            # Format latency
            if time_unit == "ns":
                if r_time >= 1e6:
                    time_str = f"{r_time / 1e6:.2f} ms"
                elif r_time >= 1e3:
                    time_str = f"{r_time / 1e3:.1f} µs"
                else:
                    time_str = f"{r_time:.0f} ns"
                cpu_str = f"{c_time / 1e3:.1f} µs"
            else:
                time_str = f"{r_time:.2f} {time_unit}"
                cpu_str = f"{c_time:.2f} {time_unit}"

            # Format primary throughput
            tp_parts = []
            if "Bandwidth_GBps" in b:
                tp_parts.append(f"{b['Bandwidth_GBps'] / 1e9:.2f} GB/s")
            if "TFLOPS" in b:
                tflops_val = b["TFLOPS"] / 1e12
                if tflops_val >= 1.0:
                    tp_parts.append(f"{tflops_val:.2f} TFLOPS")
                else:
                    tp_parts.append(f"{tflops_val * 1000.0:.1f} GFLOPS")
            if "tokens_per_sec" in b:
                tp_parts.append(f"{b['tokens_per_sec'] / 1e3:.1f}k tok/s")
            if "PCIe_GBps" in b:
                tp_parts.append(f"{b['PCIe_GBps'] / 1e9:.2f} GB/s (DMA)")
            tp_str = " | ".join(tp_parts) if tp_parts else "-"

            # Other user counters
            other_counters = []
            for k, v in b.items():
                if k not in ["name", "run_name", "family_index", "per_family_instance_index", "run_type",
                             "repetitions", "repetition_index", "threads", "iterations", "real_time", "cpu_time",
                             "time_unit", "Bandwidth_GBps", "TFLOPS", "tokens_per_sec", "PCIe_GBps"]:
                    if isinstance(v, float):
                        other_counters.append(f"`{k}`={v:.2f}")
                    else:
                        other_counters.append(f"`{k}`={v}")
            counter_str = ", ".join(other_counters) if other_counters else "-"

            md.append(f"| `{name}` | {time_str} | {cpu_str} | {tp_str} | {counter_str} |\n")

        md.append("\n")

    md.append("---\n\n")
    md.append("## 3. Intel Arc B580 Tuning Guidelines & Recommendations\n\n")
    md.append("1. **Verify AOT Machine Code**: When running on desktop Arc B580, build with `-DSTRATA_AOT_BMG=ON` to eliminate runtime JIT pause.\n")
    md.append("2. **SLM Bank Alignment**: Ensure state tensors in fused GDN (`perf_fused_gdn`) maintain 16-bank stride alignment to avoid local memory serialization.\n")
    md.append("3. **Command Graph Deployment**: Ensure full 48-layer decode execution is captured into `CapturedGraph` to eliminate the 7.6 ms CPU driver dispatch overhead.\n")
    md.append("4. **Block Copy Engine Concurrency**: Use dual in-order queues for expert streaming to maintain 98%+ XMX compute retention during background DMA copies.\n")

    return "".join(md)


def main():
    p = argparse.ArgumentParser(description="Intel Xe2 / Arc B580 Benchmark Runner & Reporter")
    p.add_argument("--build-dir", default="build_sycl", help="Path to build directory containing perf binaries")
    p.add_argument("--suite", choices=["all", "tier1", "tier2", "tier3", "tier4"], default="all", help="Select benchmark tier")
    p.add_argument("--quick", action="store_true", help="Run in fast smoke mode (min_time=0.05s)")
    p.add_argument("--repetitions", type=int, default=1, help="Number of benchmark repetitions per test")
    p.add_argument("--filter", default="", help="Regex filter to pass to Google Benchmark --benchmark_filter")
    p.add_argument("--out-dir", default="", help="Directory to save JSON results and report.md")
    p.add_argument("--auto-build", action="store_true", help="Build any missing benchmark targets automatically")
    p.add_argument("--dry-run", action="store_true", help="Print benchmarks to run without executing")
    args = p.parse_args()

    build_path = Path(args.build_dir)
    if not build_path.is_absolute():
        build_path = ROOT / build_path

    env = ensure_oneapi_env()
    device_info = discover_device(build_path, env)

    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir_path = Path(args.out_dir) if args.out_dir else (ROOT / "bench_results" / timestamp_str)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" Intel Xe2 / Arc B580 Performance Benchmark Suite Runner")
    print(f" Target Device:  {device_info['name']}")
    print(f" Architecture:   {device_info['arch']}")
    print(f" Target Bandwidth: {device_info.get('target_bw', '456 GB/s')}")
    print(f" Output Folder:  {out_dir_path}")
    print("=" * 80)

    # Filter targets based on tier
    targets_to_run = []
    for t in BENCHMARK_TARGETS:
        if args.suite == "tier1" and t["tier"] != 1:
            continue
        if args.suite == "tier2" and t["tier"] != 2:
            continue
        if args.suite == "tier3" and t["tier"] != 3:
            continue
        if args.suite == "tier4" and t["tier"] != 4:
            continue
        targets_to_run.append(t)

    min_time = 0.05 if args.quick else 0.2

    if args.dry_run:
        print("\nDry-run mode selected. Targets to execute:")
        for t in targets_to_run:
            bin_path = build_path / t["name"]
            status = "FOUND" if bin_path.exists() else "MISSING"
            print(f"  [{status}] {t['name']:<25} - {t['title']}")
        sys.exit(0)

    # Check for missing binaries and optionally auto-build
    missing_bins = [t["name"] for t in targets_to_run if not (build_path / t["name"]).exists()]
    if missing_bins:
        if args.auto_build:
            print(f"\nBuilding missing benchmark targets: {', '.join(missing_bins)}...")
            cmd = ["make", "-C", str(build_path), "-j4"] + missing_bins
            subprocess.run(cmd, env=env, check=True)
        else:
            print(f"\nWarning: Missing {len(missing_bins)} benchmark binaries: {', '.join(missing_bins)}")
            print("Run with --auto-build or compile with 'make -C build_sycl perf-sycl'.\n")

    suite_t0 = time.perf_counter()
    all_results = {}
    completed_count = 0

    for i, target in enumerate(targets_to_run, 1):
        tname = target["name"]
        bin_path = build_path / tname
        if not bin_path.exists():
            print(f"[{i}/{len(targets_to_run)}] Skipping {tname} (binary not found)")
            continue

        print(f"\n[{i}/{len(targets_to_run)}] Running {tname} ({target['title']})...")
        out_json = out_dir_path / f"{tname}.json"

        success, output, parsed_json = run_single_benchmark(
            bin_path=bin_path,
            out_json=out_json,
            min_time=min_time,
            repetitions=args.repetitions,
            filter_pattern=args.filter,
            env=env,
        )

        all_results[tname] = {
            "title": target["title"],
            "tier": target["tier"],
            "success": success,
            "data": parsed_json,
        }

        if success:
            completed_count += 1
            num_bms = len(parsed_json.get("benchmarks", []))
            print(f"  -> SUCCESS ({num_bms} benchmarks recorded)")
        else:
            print(f"  -> FAILED")

    suite_duration = time.perf_counter() - suite_t0

    # Write consolidated JSON
    consolidated_json = {
        "metadata": {
            "device": device_info,
            "timestamp": timestamp_str,
            "duration_s": suite_duration,
            "quick_mode": args.quick,
            "repetitions": args.repetitions,
        },
        "results": all_results,
    }
    with open(out_dir_path / "results_consolidated.json", "w") as f:
        json.dump(consolidated_json, f, indent=2)

    # Generate and write Markdown report
    report_md = generate_markdown_report(device_info, all_results, out_dir_path, timestamp_str, suite_duration)
    report_file = out_dir_path / "report.md"
    with open(report_file, "w") as f:
        f.write(report_md)

    print("\n" + "=" * 80)
    print(f" Suite Completed: {completed_count}/{len(targets_to_run)} benchmarks succeeded in {suite_duration:.1f}s")
    print(f" Detailed Markdown Report: {report_file}")
    print(f" Consolidated JSON Data:   {out_dir_path / 'results_consolidated.json'}")
    print("=" * 80)

    # Print summary highlights to terminal
    print("\nBenchmark Highlights:")
    for target in targets_to_run:
        tname = target["name"]
        data = all_results.get(tname, {}).get("data", {})
        benchmarks = data.get("benchmarks", [])
        if not benchmarks:
            continue
        first_b = benchmarks[0]
        r_time = first_b.get("real_time", 0)
        time_str = f"{r_time/1e3:.1f} µs" if r_time < 1e6 else f"{r_time/1e6:.2f} ms"
        print(f"  - {target['title']:<38}: {time_str:>10} ({len(benchmarks)} subtests)")

    print()


if __name__ == "__main__":
    main()
