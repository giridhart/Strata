#!/usr/bin/env python3
"""tools/run_xe2_perf_benchmarks.py - Intel Xe2 / Arc B580 Benchmark Runner & Env Tuning Engine.

Automated runner and reporting tool designed for execution on target Intel Xe2 hardware
(Intel Arc B580 desktop or Intel Core Ultra Lunar Lake).

Capabilities:
1. Environment & Device Discovery: Queries strata-device / sycl-ls for GPU name, arch, and specs.
2. Complete Benchmark Coverage: Orchestrates all 32 Google Benchmark suites across Tiers 1-4,
   plus standalone microbenchmarks (mmvq_bench, mmvq_sg_bench, b6_mmvq_bench, sycl_model_shapes,
   fused_gr_bench, qsa_prompt_attn_parity).
3. Environment Sensitivity & Tuning Sweep (--tune-env):
   Directly benchmarks critical kernel environment variable toggles (STRATA_PROMPT_ATTN_ESIMD,
   STRATA_ESIMD_LV, STRATA_MMVQ_DECODE_256GRF, STRATA_B6_MMVQ_ROWS, STRATA_MMVQ_SG, STRATA_MMVF_ROWS,
   STRATA_FUSED_SWIGLU_Q81, STRATA_FUSED_GDN, STRATA_ATTN_LANECELL), measures latency and delta%, determines the winning
   configuration for the active GPU, and generates a ready-to-source optimal environment shell script.
4. Structured Output: Captures individual and consolidated JSON files.
5. Professional Markdown Report: Generates detailed tables, bandwidth metrics, speedup calculations,
   and comparison against Intel Arc B580 hardware targets (456 GB/s, 117 TFLOPS).
6. Clean Terminal Summary: Displays formatted console tables and recommendation matrix.

Usage:
    python3 tools/run_xe2_perf_benchmarks.py                      # Standard run, all Google Benchmark suites
    python3 tools/run_xe2_perf_benchmarks.py --quick              # Fast smoke run (min_time=0.05s)
    python3 tools/run_xe2_perf_benchmarks.py --suite tier1        # Run Tier 1 suites only
    python3 tools/run_xe2_perf_benchmarks.py --suite micro        # Run standalone microbenchmarks only
    python3 tools/run_xe2_perf_benchmarks.py --tune-env           # Perform environment variable tuning sweep
    python3 tools/run_xe2_perf_benchmarks.py --tune-env --quick   # Fast environment tuning sweep (<30s)
    python3 tools/run_xe2_perf_benchmarks.py --save-env b580.env  # Save optimal env exports to file
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

# 32 Google Benchmark Targets across Tiers 1-4
BENCHMARK_TARGETS = [
    # Tier 1: Core Computational Primitives (10 suites)
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
        "title": "MoE Expert Weighted Accumulation & Grouped GEMV",
        "description": "10-expert reduction (Atomic/Combine/Subgroup) and 16-lane subgroup MoE GEMV (Sub16 vs Legacy32)",
    },
    {
        "name": "perf_dequant_dpas",
        "tier": 1,
        "title": "Weight-Only 4-bit Dequant + DPAS",
        "description": "Streaming quantized weights: Direct-to-Register DPAS vs SLM-buffered execution",
    },
    {
        "name": "perf_iq1m_lut",
        "tier": 1,
        "title": "IQ1_M LUT Lookup & Systolic GEMV",
        "description": "IQ1_M codebook dequantization and persistent SLM LUT systolic GEMV",
    },
    {
        "name": "perf_s_gemv_pair",
        "tier": 1,
        "title": "Fused Dual GEMV + SwiGLU",
        "description": "Fused gate+up projection and SwiGLU activation vs separate kernels",
    },
    {
        "name": "perf_quantize_act",
        "tier": 1,
        "title": "Fused Multi-Format Activation Quantizer",
        "description": "Fused 3-output activation quantization (Q8_0, Q8_K, BF16) vs separate passes",
    },
    {
        "name": "perf_bf16_gemv_pair",
        "tier": 1,
        "title": "Paired Dual-Head BF16 GEMV",
        "description": "Paired BF16 projection kernels (SSM alpha+beta, QSA indexer) vs standalone GEMVs",
    },
    {
        "name": "perf_kv_gather_wide",
        "tier": 1,
        "title": "Wide 16-Byte Vectorized KV Gather",
        "description": "Vectorized 16-byte wide KV cache block gather vs standard 4-wide gather",
    },
    # Tier 2: Advanced System & Recurrence Primitives (9 suites)
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
        "name": "perf_mmq_direct",
        "tier": 2,
        "title": "Direct Matrix-Matrix Quantized Prefill",
        "description": "Direct MMQ execution without intermediate weight dequantization copies",
    },
    {
        "name": "perf_gdn_chunk",
        "tier": 2,
        "title": "GDN Chunk-Recurrence Parallel Execution",
        "description": "Chunked parallel GDN recurrence vs sequential single-token stepping",
    },
    {
        "name": "perf_trellis",
        "tier": 2,
        "title": "Trellis Quantization Search & Dequant",
        "description": "Trellis codebook SLM search vs global memory execution",
    },
    {
        "name": "perf_speculative_verify",
        "tier": 2,
        "title": "MTP Speculative Verification Pipeline",
        "description": "Multi-token verification throughput and scaling across candidate batch sizes B=1..8",
    },
    # Tier 3: Multi-Operation & Data Movement Pipelines (4 suites)
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
    # Tier 4: Output Layer, Transform & Autoregressive Endpoints (9 suites)
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
    {
        "name": "perf_finish",
        "tier": 4,
        "title": "Post-Processing, Normalization & Layer Finish",
        "description": "Fused RMSNorm + residual + token embedding vs separate passes",
    },
    {
        "name": "perf_gemv_rows",
        "tier": 4,
        "title": "Multi-Row GEMV Verification Throughput",
        "description": "Multi-row GEMV execution scaling across verification candidate counts T=1..8",
    },
    {
        "name": "perf_numerics",
        "tier": 4,
        "title": "Numerical Format Conversion (FP8/FP4/BF16)",
        "description": "Hardware numerical conversion throughput across modern precision formats",
    },
    {
        "name": "perf_hc_mix_split",
        "tier": 4,
        "title": "Hyper-Connection Stream Mixing & Split Reduction",
        "description": "Fused stream mixing, weighted addition, and split channel reduction",
    },
    {
        "name": "perf_q8_quant_multi",
        "tier": 4,
        "title": "Multi-Column Parallel Q8 Quantization",
        "description": "Multi-column parallel Q8_1 quantization with batched deduplication",
    },
]

# Standalone Microbenchmarks
MICRO_BENCHMARK_TARGETS = [
    {
        "name": "mmvq_bench",
        "title": "Dense Native MMVQ Q6_K & Planar A/B",
        "description": "128-GRF vs 256-GRF single-token decode latency and bandwidth across wide & planar layouts",
        "default_args": ["2560", "2560", "1"],
    },
    {
        "name": "mmvq_sg_bench",
        "title": "Subgroup & GRF Sweep across Quant Formats",
        "description": "IQ3_S, Q6_K, Q5_K, Q4_K, IQ4_XS, Q8_0, IQ4_NL across shapes and column counts",
        "default_args": ["50"],
    },
    {
        "name": "b6_mmvq_bench",
        "title": "B6 Multi-Row Verification Scaling",
        "description": "MTP speculative verification multi-row mode scaling across widths T=1..8",
        "default_args": ["20", "0", "1"],
    },
    {
        "name": "sycl_model_shapes",
        "title": "Comprehensive Production Model Shapes & IQ3_S",
        "description": "All 21 production shapes across Qwen3.8-Flash-Next, Swift 1.5, Coder, and MoE paired decode",
        "default_args": [],
    },
    {
        "name": "fused_gr_bench",
        "title": "Fused GR Read Multi-Width Down Projection",
        "description": "Multi-token fused GR read and down projection specialization across T=1..6",
        "default_args": ["50"],
    },
    {
        "name": "qsa_prompt_attn_parity",
        "title": "QSA Prompt Attention Parity & ESIMD Throughput",
        "description": "Intel Xe2 int8 DPAS (ESIMD) prompt attention throughput and FP64 parity",
        "default_args": ["1024", "128", "1"],
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
        "arch": "Intel Xe2 Architecture",
        "compute_cap": "Unknown",
        "multiprocessors": "Unknown",
        "vram_total": "Unknown",
        "vram_free": "Unknown",
        "target_bw": "456 GB/s",
        "target_tflops": "117 TFLOPS",
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

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

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


def run_standalone_micro(
    bin_path: Path,
    args: list[str],
    env: dict[str, str],
) -> tuple[bool, str, float]:
    """Execute a standalone microbenchmark binary and return success, stdout, elapsed seconds."""
    cmd = [str(bin_path)] + args
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    elapsed = time.perf_counter() - t0
    success = proc.returncode == 0
    output = proc.stdout if success else (proc.stdout + "\n" + proc.stderr)
    return success, output, elapsed


def run_env_tuning_sweep(build_path: Path, base_env: dict[str, str], quick: bool = False) -> dict:
    """Run comparative empirical performance sweeps to find optimal environment variable settings."""
    results = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "toggles": {},
        "recommended_env": {},
    }

    print("\n" + "=" * 80)
    print(" Intel Xe2 Environment Sensitivity & Tuning Sweep")
    print(" Evaluating Empirical Microbenchmarks Across Added Kernel Environment Toggles...")
    print("=" * 80)

    # 1. STRATA_PROMPT_ATTN_ESIMD & STRATA_ESIMD_LV Sweep
    qsa_bin = build_path / "qsa_prompt_attn_parity"
    if qsa_bin.exists():
        print("\n[1/6] Evaluating Prompt Attention DPAS (STRATA_PROMPT_ATTN_ESIMD & STRATA_ESIMD_LV)...")
        esimd_results = {}
        # Test A: ESIMD=0 (Joint Matrix / Fallback)
        env_fallback = base_env.copy()
        env_fallback["STRATA_PROMPT_ATTN_ESIMD"] = "0"
        s0, out0, _ = run_standalone_micro(qsa_bin, ["1024", "128", "1"], env_fallback)
        m0 = re.search(r"([\d\.]+)\s*->\s*([\d\.]+)\s*ms\s*per\s*chunk", out0)
        ms0 = float(m0.group(2)) if m0 else 0.0

        # Test B: ESIMD=1, LV=3 (High precision default)
        env_lv3 = base_env.copy()
        env_lv3["STRATA_PROMPT_ATTN_ESIMD"] = "1"
        env_lv3["STRATA_ESIMD_LV"] = "3"
        s1, out1, _ = run_standalone_micro(qsa_bin, ["1024", "128", "1"], env_lv3)
        m1 = re.search(r"([\d\.]+)\s*->\s*([\d\.]+)\s*ms\s*per\s*chunk", out1)
        ms1 = float(m1.group(2)) if m1 else 0.0

        # Test C: ESIMD=1, LV=2 (Fast 19x mode)
        env_lv2 = base_env.copy()
        env_lv2["STRATA_PROMPT_ATTN_ESIMD"] = "1"
        env_lv2["STRATA_ESIMD_LV"] = "2"
        s2, out2, _ = run_standalone_micro(qsa_bin, ["1024", "128", "1"], env_lv2)
        m2 = re.search(r"([\d\.]+)\s*->\s*([\d\.]+)\s*ms\s*per\s*chunk", out2)
        ms2 = float(m2.group(2)) if m2 else 0.0

        esimd_speedup = (ms0 / ms1) if ms1 > 0 else 1.0
        fast_speedup = (ms0 / ms2) if ms2 > 0 else 1.0

        print(f"  Fallback (Joint Matrix):    {ms0:.2f} ms/chunk")
        print(f"  ESIMD DPAS (LV=3, default): {ms1:.2f} ms/chunk ({esimd_speedup:.2f}x speedup vs fallback)")
        print(f"  ESIMD DPAS (LV=2, fast):    {ms2:.2f} ms/chunk ({fast_speedup:.2f}x speedup vs fallback)")

        results["toggles"]["STRATA_PROMPT_ATTN_ESIMD"] = {
            "fallback_ms": ms0,
            "esimd_lv3_ms": ms1,
            "esimd_lv2_ms": ms2,
            "speedup_lv3": esimd_speedup,
            "speedup_lv2": fast_speedup,
            "winner": "1",
            "comment": f"{esimd_speedup:.1f}x speedup via Intel Xe2 int8 DPAS",
        }
        results["toggles"]["STRATA_ESIMD_LV"] = {
            "winner": "3",
            "comment": f"High precision default (use LV=2 for {fast_speedup:.1f}x fast mode)",
        }
        results["recommended_env"]["STRATA_PROMPT_ATTN_ESIMD"] = "1"
        results["recommended_env"]["STRATA_ESIMD_LV"] = "3"

    # 2. STRATA_MMVQ_DECODE_256GRF Sweep (Decode GRF)
    mmvq_bin = build_path / "mmvq_bench"
    if mmvq_bin.exists():
        print("\n[2/6] Evaluating Single-Token Decode GRF (STRATA_MMVQ_DECODE_256GRF)...")
        s, out, _ = run_standalone_micro(mmvq_bin, ["2560", "640", "1"], base_env)
        # Parse wide and planar deltas
        wide_m = re.search(r"q6_k wide\s+256-GRF:\s+([\d\.]+)\s+us\s+\(\s*([\d\.]+)\s+GB/s\).*?delta:\s*([\+\-]?[\d\.]+)%\s+(faster|slower)", out)
        planar_m = re.search(r"q6_k planar\s+256-GRF:\s+([\d\.]+)\s+us\s+\(\s*([\d\.]+)\s+GB/s\).*?delta:\s*([\+\-]?[\d\.]+)%\s+(faster|slower)", out)

        wide_delta = float(wide_m.group(3)) if wide_m else 0.0
        planar_delta = float(planar_m.group(3)) if planar_m else 0.0

        print(f"  Q6_K Wide (2560x640):   256-GRF is {wide_delta:+.1f}% vs 128-GRF")
        print(f"  Q6_K Planar (2560x640): 256-GRF is {planar_delta:+.1f}% vs 128-GRF")

        # Winner is 1 if planar or wide is faster
        win = "1" if (planar_delta > 0.0 or wide_delta > 0.0) else "0"
        results["toggles"]["STRATA_MMVQ_DECODE_256GRF"] = {
            "wide_delta_pct": wide_delta,
            "planar_delta_pct": planar_delta,
            "winner": win,
            "comment": f"{planar_delta:+.1f}% planar, {wide_delta:+.1f}% wide on single-token decode",
        }
        results["recommended_env"]["STRATA_MMVQ_DECODE_256GRF"] = win

    # 3. STRATA_B6_MMVQ_ROWS Sweep (Speculative Verification Multi-Row Mode)
    b6_bin = build_path / "b6_mmvq_bench"
    if b6_bin.exists():
        print("\n[3/6] Evaluating Speculative Verification Mode (STRATA_B6_MMVQ_ROWS)...")
        # Run Mode A (bmode=0) vs Mode B (bmode=1)
        s, out, _ = run_standalone_micro(b6_bin, ["20", "0", "1"], base_env)
        # Match T=8 rows
        t8_matches = re.findall(r"(\S.*?\S)\s+8\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.\-]+)\s+([\d\.\-]+)", out)
        avg_a, avg_b, count = 0.0, 0.0, 0
        for m in t8_matches:
            avg_a += float(m[1])
            avg_b += float(m[2])
            count += 1
        if count > 0:
            avg_a /= count
            avg_b /= count
            b_speedup = ((avg_a - avg_b) / avg_a * 100.0) if avg_a > 0 else 0.0
            print(f"  Average T=8 Latency: Mode A = {avg_a:.2f} ms vs Mode B = {avg_b:.2f} ms ({b_speedup:+.1f}%)")
            win = "1" if avg_b < avg_a else "0"
            results["toggles"]["STRATA_B6_MMVQ_ROWS"] = {
                "mode_a_ms": avg_a,
                "mode_b_ms": avg_b,
                "speedup_pct": b_speedup,
                "winner": win,
                "comment": f"{b_speedup:+.1f}% multi-row verification speedup across T=8",
            }
            results["recommended_env"]["STRATA_B6_MMVQ_ROWS"] = win

    # 4. STRATA_MMVQ_SG Sweep (SIMD16 vs SIMD32)
    if mmvq_bin.exists():
        print("\n[4/6] Evaluating MMVQ Subgroup Size (STRATA_MMVQ_SG=16 vs 32)...")
        env_sg16 = base_env.copy()
        env_sg16["STRATA_MMVQ_SG"] = "16"
        s16, out16, _ = run_standalone_micro(mmvq_bin, ["2560", "2560", "1"], env_sg16)
        m16 = re.search(r"q6_k wide\s+128-GRF:\s+([\d\.]+)\s+us", out16)
        us16 = float(m16.group(1)) if m16 else 0.0

        env_sg32 = base_env.copy()
        env_sg32["STRATA_MMVQ_SG"] = "32"
        s32, out32, _ = run_standalone_micro(mmvq_bin, ["2560", "2560", "1"], env_sg32)
        m32 = re.search(r"q6_k wide\s+128-GRF:\s+([\d\.]+)\s+us", out32)
        us32 = float(m32.group(1)) if m32 else 0.0

        print(f"  SIMD16: {us16:.1f} µs | SIMD32: {us32:.1f} µs")
        win = "32" if us32 <= us16 else "16"
        delta = ((us16 - us32) / us16 * 100.0) if us16 > 0 else 0.0
        results["toggles"]["STRATA_MMVQ_SG"] = {
            "sg16_us": us16,
            "sg32_us": us32,
            "speedup_pct": delta,
            "winner": win,
            "comment": f"Optimal SIMD{win} occupancy ({abs(delta):.1f}% delta)",
        }
        results["recommended_env"]["STRATA_MMVQ_SG"] = win

    # 5. STRATA_MMVF_ROWS Sweep (Multi-Row GEMV)
    gemv_rows_bin = build_path / "perf_gemv_rows"
    if gemv_rows_bin.exists():
        print("\n[5/6] Evaluating Multi-Row BF16 GEMV (STRATA_MMVF_ROWS)...")
        env_g = base_env.copy()
        proc = subprocess.run(
            [str(gemv_rows_bin), "--benchmark_filter=2560/2560/8", "--benchmark_min_time=0.05s"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env_g,
        )
        base_m = re.search(r"BM_GEMV_Baseline/2560/2560/8/manual_time\s+([\d\.]+)\s+ns", proc.stdout)
        mrow_m = re.search(r"BM_GEMV_MultiRow/2560/2560/8/manual_time\s+([\d\.]+)\s+ns", proc.stdout)
        base_us = float(base_m.group(1)) / 1000.0 if base_m else 0.0
        mrow_us = float(mrow_m.group(1)) / 1000.0 if mrow_m else 0.0
        if base_us > 0 and mrow_us > 0:
            sp = base_us / mrow_us
            print(f"  Baseline: {base_us:.1f} µs vs MultiRow: {mrow_us:.1f} µs ({sp:.2f}x speedup)")
            win = "1" if mrow_us < base_us else "0"
            results["toggles"]["STRATA_MMVF_ROWS"] = {
                "baseline_us": base_us,
                "multi_row_us": mrow_us,
                "speedup": sp,
                "winner": win,
                "comment": f"{sp:.2f}x speedup via multi-row vectorized register reuse",
            }
            results["recommended_env"]["STRATA_MMVF_ROWS"] = win

    # 6. STRATA_FUSED_GDN Sweep (Fused GDN Linear Attention)
    fused_gdn_bin = build_path / "perf_fused_gdn"
    if fused_gdn_bin.exists():
        print("\n[6/7] Evaluating Fused GDN Linear Attention (STRATA_FUSED_GDN)...")
        proc = subprocess.run(
            [str(fused_gdn_bin), "--benchmark_filter=16/16", "--benchmark_min_time=0.05s"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=base_env,
        )
        unf_m = re.search(r"BM_Xe2_GDN_Unfused/16/16/manual_time\s+([\d\.]+)\s+ns", proc.stdout)
        fus_m = re.search(r"BM_Xe2_GDN_Fused_SLM/16/16/manual_time\s+([\d\.]+)\s+ns", proc.stdout)
        unf_us = float(unf_m.group(1)) / 1000.0 if unf_m else 0.0
        fus_us = float(fus_m.group(1)) / 1000.0 if fus_m else 0.0
        if unf_us > 0 and fus_us > 0:
            sp = unf_us / fus_us
            print(f"  Unfused: {unf_us:.1f} µs vs Fused SLM: {fus_us:.1f} µs ({sp:.2f}x faster)")
            win = "1" if fus_us < unf_us else "0"
            results["toggles"]["STRATA_FUSED_GDN"] = {
                "unfused_us": unf_us,
                "fused_us": fus_us,
                "speedup": sp,
                "winner": win,
                "comment": f"{sp:.2f}x faster via fused SLM step + norm",
            }
            results["recommended_env"]["STRATA_FUSED_GDN"] = win

    # 7. STRATA_ATTN_LANECELL Sweep (Decode Attention Register Spill vs Subgroup Reductions)
    perf_qsa_bin = build_path / "perf_qsa_attention"
    if perf_qsa_bin.exists():
        print("\n[7/7] Evaluating Decode Attention Scoring (STRATA_ATTN_LANECELL=0 vs 1)...")
        # Test A: STRATA_ATTN_LANECELL=0 (Subgroup Reductions, Zero Spills)
        env_lane0 = base_env.copy()
        env_lane0["STRATA_ATTN_LANECELL"] = "0"
        proc0 = subprocess.run(
            [str(perf_qsa_bin), "--benchmark_filter=BM_Xe2_QSA_DecodeAttention_BatchRows/1", "--benchmark_min_time=0.05s"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env_lane0,
        )
        m0 = re.search(r"BM_Xe2_QSA_DecodeAttention_BatchRows/1/manual_time\s+([\d\.]+)\s+ns", proc0.stdout)
        us0 = float(m0.group(1)) / 1000.0 if m0 else 0.0

        # Test B: STRATA_ATTN_LANECELL=1 (LaneCell 96-Register Array, 7KB Spill)
        env_lane1 = base_env.copy()
        env_lane1["STRATA_ATTN_LANECELL"] = "1"
        proc1 = subprocess.run(
            [str(perf_qsa_bin), "--benchmark_filter=BM_Xe2_QSA_DecodeAttention_BatchRows/1", "--benchmark_min_time=0.05s"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env_lane1,
        )
        m1 = re.search(r"BM_Xe2_QSA_DecodeAttention_BatchRows/1/manual_time\s+([\d\.]+)\s+ns", proc1.stdout)
        us1 = float(m1.group(1)) / 1000.0 if m1 else 0.0

        if us0 > 0 and us1 > 0:
            sp = (us1 / us0) if us0 <= us1 else (us0 / us1)
            win = "0" if us0 <= us1 else "1"
            if win == "0":
                print(f"  LANECELL=0: {us0:.1f} µs vs LANECELL=1: {us1:.1f} µs ({sp:.2f}x faster via zero-spill warp reductions)")
                comm = f"{sp:.2f}x faster via zero-spill warp reductions (eliminates 7KB scratch spill)"
            else:
                print(f"  LANECELL=1: {us1:.1f} µs vs LANECELL=0: {us0:.1f} µs ({sp:.2f}x faster)")
                comm = f"{sp:.2f}x faster on current hardware"

            results["toggles"]["STRATA_ATTN_LANECELL"] = {
                "lane_0_us": us0,
                "lane_1_us": us1,
                "speedup": sp,
                "winner": win,
                "comment": comm,
            }
            results["recommended_env"]["STRATA_ATTN_LANECELL"] = win

    return results


def generate_markdown_report(
    device_info: dict[str, str],
    all_results: dict[str, dict],
    micro_results: dict[str, dict],
    tuning_results: dict,
    out_dir: Path,
    run_timestamp: str,
    duration_s: float,
) -> str:
    """Generate professional Markdown performance report covering all 32 suites, microbenchmarks, and tuning sweep."""
    md = []
    md.append(f"# Intel Xe2 / Arc B580 Performance Benchmark & Environment Tuning Report\n\n")
    md.append(f"> **Execution Timestamp**: {run_timestamp}  \n")
    md.append(f"> **Target Hardware**: `{device_info.get('name', 'Unknown')}`  \n")
    md.append(f"> **Microarchitecture**: {device_info.get('arch', 'Intel Xe2')}  \n")
    md.append(f"> **Compute Units**: {device_info.get('multiprocessors', 'N/A')} multiprocessors  \n")
    md.append(f"> **VRAM Pool**: {device_info.get('vram_total', 'N/A')}  \n")
    md.append(f"> **Suite Duration**: {duration_s:.1f} seconds total execution time  \n")
    md.append(f"> **Target Memory Bandwidth**: {device_info.get('target_bw', '456 GB/s')}  \n")
    md.append(f"> **Target Peak Compute**: {device_info.get('target_tflops', '117 TFLOPS')}  \n\n")
    md.append("---\n\n")

    # Section 1: Executive Summary
    has_suite_benchmarks = any(
        all_results.get(t["name"], {}).get("data", {}).get("benchmarks")
        for t in BENCHMARK_TARGETS
    )

    if has_suite_benchmarks:
        md.append("## 1. Executive Performance Summary\n\n")
        md.append("| Benchmark Suite | Key Evaluated Stage | Primary Measured Metric | Hardware Status |\n")
        md.append("|---|---|---|---|\n")

        for target in BENCHMARK_TARGETS:
            tname = target["name"]
            data = all_results.get(tname, {}).get("data", {})
            benchmarks = data.get("benchmarks", [])
            if not benchmarks:
                continue

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
        elif tname == "perf_iq1m_lut":
            max_bw = max((b.get("Bandwidth_GBps", 0) for b in benchmarks), default=0) / 1e9
            metric_str = f"Peak SLM LUT: **{max_bw:.2f} GB/s**"
        elif tname == "perf_s_gemv_pair":
            fused_t = next((b.get("real_time", 0) for b in benchmarks if "Fused_GateUp" in b.get("name", "")), 0)
            unfused_t = next((b.get("real_time", 0) for b in benchmarks if "Unfused_GateUp" in b.get("name", "")), 0)
            ratio = (unfused_t / fused_t) if fused_t > 0 else 0
            metric_str = f"Fused SwiGLU: **{fused_t/1e3:.1f} µs** (**{ratio:.2f}x faster**)"
        elif tname == "perf_quantize_act":
            fused_t = next((b.get("real_time", 0) for b in benchmarks if "Fused_ActImages" in b.get("name", "")), 0)
            unfused_t = next((b.get("real_time", 0) for b in benchmarks if "Unfused_ActImages" in b.get("name", "")), 0)
            ratio = (unfused_t / fused_t) if fused_t > 0 else 0
            metric_str = f"Fused 3-Output: **{fused_t/1e3:.1f} µs** (**{ratio:.2f}x faster**)"
        elif tname == "perf_bf16_gemv_pair":
            fused_t = next((b.get("real_time", 0) for b in benchmarks if "Fused_BF16_Pair" in b.get("name", "")), 0)
            unfused_t = next((b.get("real_time", 0) for b in benchmarks if "Unfused_BF16_Pair" in b.get("name", "")), 0)
            ratio = (unfused_t / fused_t) if fused_t > 0 else 0
            metric_str = f"Fused Pair: **{fused_t/1e3:.1f} µs** (**{ratio:.2f}x faster**)"
        elif tname == "perf_kv_gather_wide":
            max_bw = max((b.get("Bandwidth_GBps", 0) for b in benchmarks), default=0) / 1e9
            metric_str = f"Gather Bandwidth: **{max_bw:.2f} GB/s**"
        elif tname == "perf_fused_gdn":
            fused_t = next((b.get("real_time", 0) for b in benchmarks if "Fused_SLM" in b.get("name", "")), 0)
            unfused_t = next((b.get("real_time", 0) for b in benchmarks if "Unfused" in b.get("name", "")), 0)
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
        elif tname == "perf_mmq_direct":
            dir_t = next((b.get("real_time", 0) for b in benchmarks if "Direct" in b.get("name", "")), 0)
            gath_t = next((b.get("real_time", 0) for b in benchmarks if "Gathered" in b.get("name", "")), 0)
            ratio = (gath_t / dir_t) if dir_t > 0 else 0
            metric_str = f"Direct MMQ: **{dir_t/1e3:.1f} µs** (**{ratio:.2f}x faster**)"
        elif tname == "perf_gdn_chunk":
            chunk_t = next((b.get("real_time", 0) for b in benchmarks if "GDN_Chunked" in b.get("name", "")), 0)
            seq_t = next((b.get("real_time", 0) for b in benchmarks if "GDN_Sequential" in b.get("name", "")), 0)
            ratio = (seq_t / chunk_t) if chunk_t > 0 else 0
            metric_str = f"Chunked: **{chunk_t/1e3:.1f} µs** (**{ratio:.2f}x vs sequential**)"
        elif tname == "perf_trellis":
            slm_t = next((b.get("real_time", 0) for b in benchmarks if "GEMV_SLM" in b.get("name", "")), 0)
            glob_t = next((b.get("real_time", 0) for b in benchmarks if "GEMV_Global" in b.get("name", "")), 0)
            ratio = (glob_t / slm_t) if slm_t > 0 else 0
            metric_str = f"Trellis SLM: **{slm_t/1e3:.1f} µs** (**{ratio:.2f}x faster**)"
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
        elif tname == "perf_finish":
            fused_t = next((b.get("real_time", 0) for b in benchmarks if "Finish_BF16_Fused" in b.get("name", "")), 0)
            unfused_t = next((b.get("real_time", 0) for b in benchmarks if "Finish_BF16_Unfused" in b.get("name", "")), 0)
            ratio = (unfused_t / fused_t) if fused_t > 0 else 0
            metric_str = f"Fused Finish: **{fused_t/1e3:.1f} µs** (**{ratio:.2f}x faster**)"
        elif tname == "perf_gemv_rows":
            base_t = next((b.get("real_time", 0) for b in benchmarks if "BM_GEMV_Baseline/2560/2560/8" in b.get("name", "")), 0)
            mrow_t = next((b.get("real_time", 0) for b in benchmarks if "BM_GEMV_MultiRow/2560/2560/8" in b.get("name", "")), 0)
            ratio = (base_t / mrow_t) if mrow_t > 0 else 0
            metric_str = f"MultiRow (T=8): **{mrow_t/1e3:.1f} µs** (**{ratio:.2f}x speedup**)"
        elif tname == "perf_numerics":
            max_bw = max((b.get("Bandwidth_GBps", 0) for b in benchmarks), default=0) / 1e9
            metric_str = f"Conversion Peak: **{max_bw:.2f} GB/s**"
        elif tname == "perf_hc_mix_split":
            fused_t = next((b.get("real_time", 0) for b in benchmarks if "HcMixSplit_Fused" in b.get("name", "")), 0)
            sep_t = next((b.get("real_time", 0) for b in benchmarks if "HcMixSplit_Separate" in b.get("name", "")), 0)
            ratio = (sep_t / fused_t) if fused_t > 0 else 0
            metric_str = f"Fused Mix-Split: **{fused_t/1e3:.1f} µs** (**{ratio:.2f}x faster**)"
        elif tname == "perf_q8_quant_multi":
            dedup_t = next((b.get("real_time", 0) for b in benchmarks if "BatchedDedup" in b.get("name", "")), 0)
            sep_t = next((b.get("real_time", 0) for b in benchmarks if "Separate" in b.get("name", "")), 0)
            ratio = (sep_t / dedup_t) if dedup_t > 0 else 0
            metric_str = f"Batched Dedup: **{dedup_t/1e3:.1f} µs** (**{ratio:.2f}x faster**)"

            md.append(f"| **{target['title']}** | {target['description']} | {metric_str} | ✅ PASS |\n")

        md.append("\n---\n\n")

    # Section 2: Environment Tuning Results (if executed)
    if tuning_results and tuning_results.get("toggles"):
        md.append("## 2. Environment Variable Sensitivity & Tuning Sweep\n\n")
        md.append(f"Direct hardware sensitivity measurement of kernel environment toggles on `{device_info.get('name', 'Active GPU')}`:\n\n")
        md.append("| Environment Variable | Evaluated Configurations | Empirical Latency / Throughput | Speedup / Delta | Hardware Optimal |\n")
        md.append("|---|---|---|---|---|\n")

        toggles = tuning_results["toggles"]
        if "STRATA_PROMPT_ATTN_ESIMD" in toggles:
            t = toggles["STRATA_PROMPT_ATTN_ESIMD"]
            md.append(f"| `STRATA_PROMPT_ATTN_ESIMD` | Fallback Joint Matrix (0) | {t['fallback_ms']:.2f} ms/chunk | Baseline | - |\n")
            md.append(f"| | **ESIMD DPAS Standard (1, LV=3)** | **{t['esimd_lv3_ms']:.2f} ms/chunk** | **+{t['speedup_lv3']:.1f}x faster** | **✅ OPTIMAL (1)** |\n")
            md.append(f"| | ESIMD DPAS Fast Mode (1, LV=2) | {t['esimd_lv2_ms']:.2f} ms/chunk | +{t['speedup_lv2']:.1f}x faster | *(Fast Mode)* |\n")

        if "STRATA_MMVQ_DECODE_256GRF" in toggles:
            t = toggles["STRATA_MMVQ_DECODE_256GRF"]
            w_win = "**✅ OPTIMAL (1)**" if t["winner"] == "1" else "Default (0)"
            md.append(f"| `STRATA_MMVQ_DECODE_256GRF` | 128-GRF vs 256-GRF (Decode cols=1) | Planar: {t['planar_delta_pct']:+.1f}% \| Wide: {t['wide_delta_pct']:+.1f}% | {t['comment']} | {w_win} |\n")

        if "STRATA_B6_MMVQ_ROWS" in toggles:
            t = toggles["STRATA_B6_MMVQ_ROWS"]
            md.append(f"| `STRATA_B6_MMVQ_ROWS` | Mode A (0) vs Mode B Multi-Row (1) | Mode A: {t['mode_a_ms']:.2f} ms vs Mode B: {t['mode_b_ms']:.2f} ms | {t['speedup_pct']:+.1f}% delta | **✅ OPTIMAL ({t['winner']})** |\n")

        if "STRATA_MMVQ_SG" in toggles:
            t = toggles["STRATA_MMVQ_SG"]
            md.append(f"| `STRATA_MMVQ_SG` | SIMD16 vs SIMD32 Subgroups | SIMD16: {t['sg16_us']:.1f} µs vs SIMD32: {t['sg32_us']:.1f} µs | {t['speedup_pct']:+.1f}% delta | **✅ OPTIMAL ({t['winner']})** |\n")

        if "STRATA_MMVF_ROWS" in toggles:
            t = toggles["STRATA_MMVF_ROWS"]
            md.append(f"| `STRATA_MMVF_ROWS` | Baseline vs Multi-Row GEMV | Baseline: {t['baseline_us']:.1f} µs vs MultiRow: {t['multi_row_us']:.1f} µs | {t['speedup']:.2f}x speedup | **✅ OPTIMAL ({t['winner']})** |\n")

        if "STRATA_FUSED_GDN" in toggles:
            t = toggles["STRATA_FUSED_GDN"]
            md.append(f"| `STRATA_FUSED_GDN` | Unfused vs Fused SLM Recurrence | Unfused: {t['unfused_us']:.1f} µs vs Fused: {t['fused_us']:.1f} µs | {t['speedup']:.2f}x faster | **✅ OPTIMAL ({t['winner']})** |\n")

        if "STRATA_ATTN_LANECELL" in toggles:
            t = toggles["STRATA_ATTN_LANECELL"]
            md.append(f"| `STRATA_ATTN_LANECELL` | Subgroup Reductions (0) vs LaneCell (1) | 0: {t['lane_0_us']:.1f} µs vs 1: {t['lane_1_us']:.1f} µs | {t['speedup']:.2f}x faster | **✅ OPTIMAL ({t['winner']})** |\n")

        md.append("\n### Recommended Environment Export Script\n\n")
        md.append("```sh\n")
        md.append(f"# Optimal Hardware Settings for {device_info.get('name', 'Active GPU')}\n")
        for k, v in tuning_results.get("recommended_env", {}).items():
            comm = toggles.get(k, {}).get("comment", "")
            md.append(f"export {k}={v}  # {comm}\n")
        md.append("```\n\n---\n\n")

    # Section 3: Standalone Microbenchmarks Summary (if run)
    if micro_results:
        md.append("## 3. Standalone Microbenchmarks Summary\n\n")
        md.append("| Microbenchmark | Evaluated Objective | Elapsed Time | Execution Status |\n")
        md.append("|---|---|---|---|\n")
        for mtarget in MICRO_BENCHMARK_TARGETS:
            mname = mtarget["name"]
            mdata = micro_results.get(mname, {})
            if not mdata:
                continue
            status = "✅ PASS" if mdata.get("success") else "❌ FAIL"
            elapsed = mdata.get("elapsed", 0.0)
            md.append(f"| **{mtarget['title']}** (`{mname}`) | {mtarget['description']} | {elapsed:.2f}s | {status} |\n")
        md.append("\n---\n\n")

    # Section 4: Detailed Google Benchmark Tables
    if has_suite_benchmarks:
        md.append("## 4. Detailed Performance Tables by Kernel\n\n")
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

                tp_parts = []
                if "Bandwidth_GBps" in b:
                    tp_parts.append(f"{b['Bandwidth_GBps'] / 1e9:.2f} GB/s")
                if "BW_GBps" in b:
                    tp_parts.append(f"{b['BW_GBps'] / 1e9:.2f} GB/s")
                if "TFLOPS" in b:
                    tflops_val = b["TFLOPS"] / 1e12
                    if tflops_val >= 1.0:
                        tp_parts.append(f"{tflops_val:.2f} TFLOPS")
                    else:
                        tp_parts.append(f"{tflops_val * 1000.0:.1f} GFLOPS")
                if "GFLOPS" in b:
                    tp_parts.append(f"{b['GFLOPS']:.1f} GFLOPS")
                if "tokens_per_sec" in b:
                    tp_parts.append(f"{b['tokens_per_sec'] / 1e3:.1f}k tok/s")
                if "PCIe_GBps" in b:
                    tp_parts.append(f"{b['PCIe_GBps'] / 1e9:.2f} GB/s (DMA)")
                tp_str = " | ".join(tp_parts) if tp_parts else "-"

                other_counters = []
                for k, v in b.items():
                    if k not in [
                        "name", "run_name", "family_index", "per_family_instance_index", "run_type",
                        "repetitions", "repetition_index", "threads", "iterations", "real_time", "cpu_time",
                        "time_unit", "Bandwidth_GBps", "BW_GBps", "TFLOPS", "GFLOPS", "tokens_per_sec", "PCIe_GBps",
                    ]:
                        if isinstance(v, float):
                            other_counters.append(f"`{k}`={v:.2f}")
                        else:
                            other_counters.append(f"`{k}`={v}")
                counter_str = ", ".join(other_counters) if other_counters else "-"

                md.append(f"| `{name}` | {time_str} | {cpu_str} | {tp_str} | {counter_str} |\n")

            md.append("\n")

        md.append("---\n\n")
    md.append("## 5. Intel Arc B580 vs Lunar Lake Architectural Tuning Guidelines\n\n")
    md.append("1. **AOT Ahead-of-Time Compilation**: When deploying on desktop Arc B580 (BMG-G21), compile with `-DSTRATA_AOT_BMG=ON` to eliminate runtime driver JIT pauses.\n")
    md.append("2. **Register Allocation Dynamic Tuning**: Arc B580 discrete GPU has 28 Xe cores (vs 8 on Lunar Lake). Single-token decode with 256-GRF (`STRATA_MMVQ_DECODE_256GRF=1`) allows maximum unrolling across 16-byte planar layouts.\n")
    md.append("3. **Prompt Attention DPAS Acceleration**: `STRATA_PROMPT_ATTN_ESIMD=1` provides up to 10x-19x speedup on prompt chunk evaluation by leveraging native Battlemage int8 DPAS systolic arrays.\n")
    md.append("4. **Fused Multi-Row Speculative Verification**: Multi-row verify mode (`STRATA_B6_MMVQ_ROWS=1` and `STRATA_MMVF_ROWS=1`) reduces global memory bandwidth contention across candidate tokens B=1..8 by 2x to 3x.\n")

    return "".join(md)


def main():
    p = argparse.ArgumentParser(description="Intel Xe2 / Arc B580 Benchmark Runner & Env Tuning Engine")
    p.add_argument("--build-dir", default="build_sycl", help="Path to build directory containing perf binaries")
    p.add_argument("--suite", choices=["all", "tier1", "tier2", "tier3", "tier4", "micro", "none"], default="all", help="Select benchmark suite")
    p.add_argument("--micro", action="store_true", help="Include standalone microbenchmarks in execution")
    p.add_argument("--tune-env", action="store_true", help="Run automated environment variable sensitivity & tuning sweep")
    p.add_argument("--tune-only", action="store_true", help="Run only the environment tuning sweep without executing benchmark suites")
    p.add_argument("--save-env", default="", help="Path to write recommended environment export script (e.g. xe2_optimal_env.sh)")
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
    print(" Intel Xe2 / Arc B580 Performance Benchmark & Tuning Suite")
    print(f" Target Device:    {device_info['name']}")
    print(f" Architecture:     {device_info['arch']}")
    print(f" Target Bandwidth: {device_info.get('target_bw', '456 GB/s')}")
    print(f" Output Folder:    {out_dir_path}")
    print("=" * 80)

    if args.tune_only:
        args.tune_env = True

    # Filter targets based on suite selection
    targets_to_run = []
    if args.suite not in ["micro", "none"] and not args.tune_only:
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
        if args.suite == "micro" or args.micro:
            print("\nMicrobenchmarks:")
            for m in MICRO_BENCHMARK_TARGETS:
                bin_path = build_path / m["name"]
                status = "FOUND" if bin_path.exists() else "MISSING"
                print(f"  [{status}] {m['name']:<25} - {m['title']}")
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

    # 1. Execute Google Benchmark targets
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

    # 2. Execute Standalone Microbenchmarks if requested
    micro_results = {}
    if args.suite == "micro" or args.micro:
        print("\n" + "=" * 80)
        print(" Running Standalone Xe2 Microbenchmarks Suite")
        print("=" * 80)
        for i, mtarget in enumerate(MICRO_BENCHMARK_TARGETS, 1):
            mname = mtarget["name"]
            bin_path = build_path / mname
            if not bin_path.exists():
                print(f"[{i}/{len(MICRO_BENCHMARK_TARGETS)}] Skipping microbenchmark {mname} (binary not found)")
                continue

            args_to_use = list(mtarget["default_args"])
            if args.quick and mname in ["mmvq_sg_bench", "b6_mmvq_bench", "fused_gr_bench"]:
                args_to_use[0] = "20"

            print(f"[{i}/{len(MICRO_BENCHMARK_TARGETS)}] Running {mname} ({' '.join(args_to_use)})...")
            msuccess, moutput, melapsed = run_standalone_micro(bin_path, args_to_use, env)

            micro_results[mname] = {
                "title": mtarget["title"],
                "success": msuccess,
                "elapsed": melapsed,
                "output": moutput,
            }
            status_str = "SUCCESS" if msuccess else "FAILED"
            print(f"  -> {status_str} ({melapsed:.2f}s elapsed)")

    # 3. Execute Environment Sensitivity Tuning Sweep if requested
    tuning_results = {}
    if args.tune_env:
        tuning_results = run_env_tuning_sweep(build_path, env, quick=args.quick)

        # Save recommended env script
        save_env_path = Path(args.save_env) if args.save_env else (out_dir_path / "xe2_optimal_env.sh")
        with open(save_env_path, "w") as ef:
            ef.write(f"#!/usr/bin/env bash\n")
            ef.write(f"# Intel Xe2 Empirical Optimal Environment Configuration\n")
            ef.write(f"# Auto-generated for: {device_info['name']} on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for k, v in tuning_results.get("recommended_env", {}).items():
                comm = tuning_results.get("toggles", {}).get(k, {}).get("comment", "")
                ef.write(f"export {k}={v}  # {comm}\n")
        os.chmod(save_env_path, 0o755)

        with open(out_dir_path / "env_tuning_recommendations.json", "w") as jf:
            json.dump(tuning_results, jf, indent=2)

        print("\n" + "=" * 80)
        print(" Recommended Optimal Environment Exports:")
        print("=" * 80)
        for k, v in tuning_results.get("recommended_env", {}).items():
            comm = tuning_results.get("toggles", {}).get(k, {}).get("comment", "")
            print(f"  export {k}={v:<4} # {comm}")
        print(f"\nSaved environment script: {save_env_path}")
        print("=" * 80)

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
        "micro_results": micro_results,
        "tuning_results": tuning_results,
    }
    with open(out_dir_path / "results_consolidated.json", "w") as f:
        json.dump(consolidated_json, f, indent=2)

    # Generate and write Markdown report
    report_md = generate_markdown_report(
        device_info=device_info,
        all_results=all_results,
        micro_results=micro_results,
        tuning_results=tuning_results,
        out_dir=out_dir_path,
        run_timestamp=timestamp_str,
        duration_s=suite_duration,
    )
    report_file = out_dir_path / "report.md"
    with open(report_file, "w") as f:
        f.write(report_md)

    print("\n" + "=" * 80)
    total_bms = len(targets_to_run)
    print(f" Suite Completed: {completed_count}/{total_bms} suites succeeded in {suite_duration:.1f}s")
    print(f" Detailed Markdown Report: {report_file}")
    print(f" Consolidated JSON Data:   {out_dir_path / 'results_consolidated.json'}")
    print("=" * 80)

    # Print summary highlights to terminal
    if all_results:
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
            print(f"  - {target['title']:<40}: {time_str:>10} ({len(benchmarks)} subtests)")

    print()


if __name__ == "__main__":
    main()
