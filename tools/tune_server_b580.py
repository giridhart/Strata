#!/usr/bin/env python3
"""tools/tune_server_b580.py - Empirical Hardware Tuning & Decision Engine for Strata on Intel Arc B580.

Does NOT treat predefined settings as gospel. Instead, systematically evaluates candidate
options side-by-side on the active system, measures empirical throughput across 3 prompt/output
token profiles, determines the winning setting for every toggle based strictly on measured runs,
and auto-generates an optimal environment configuration file ('b580_empirical_best.env').

Capabilities:
  1. Empirical A/B Parameter Tournaments:
     - Tests candidate options side-by-side (e.g. 128-GRF vs 256-GRF, ESIMD LV=2 vs LV=3 vs Fallback,
       Graph chunks 2 vs 4 vs 0, LaneCell 0 vs 1, Fused GDN 0 vs 1, Route Resident 0.0 vs 0.10).
  2. 3 Standard Prompt/Output Evaluation Sets:
     - Set 1: Short / Interactive Chat         (~128 prompt,  128 output)
     - Set 2: Medium / Agentic Code & Summary   (~1024 prompt, 512 output)
     - Set 3: Long / Repository Context & RAG   (~4096 prompt, 256 output)
  3. Interactive Tweak Control:
     - Allows reviewing, tweaking, adding custom toggles, or skipping between any test run.
  4. Decision Engine & Winner Generation:
     - Identifies winning parameters by prefill tok/s, decode tok/s, and TTFT.
     - Automatically writes 'b580_empirical_best.env' containing only empirically verified winners.
  5. Multi-Mode Support:
     - Live Mode (--url): Run tournaments or sets against an already-running server.
     - Managed Mode (--config): Automatically launches, benchmarks, and cycles server_intel.py.

Usage:
  # 1. Full empirical decision tournament (cycles server_intel.py through candidate settings):
  python3 tools/tune_server_b580.py --config strata-b580-iq3s.json --port 8080 --mode tournament

  # 2. Test tiered cumulative stacks:
  python3 tools/tune_server_b580.py --config strata-b580-iq3s.json --port 8080 --mode stack

  # 3. Benchmark currently running server across the 3 sets:
  python3 tools/tune_server_b580.py --url http://127.0.0.1:8080

  # 4. Interactive mode (prompts to tweak/override before each test):
  python3 tools/tune_server_b580.py --config strata-b580-iq3s.json --interactive

  # 5. Fast quick mode (short runs for rapid validation):
  python3 tools/tune_server_b580.py --config strata-b580-iq3s.json --quick
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# -----------------------------------------------------------------------------
# 3 Standard Evaluation Sets (Prompt Length / Output Length)
# -----------------------------------------------------------------------------
STANDARD_SETS = [
    {
        "set_id": "set_1_short",
        "name": "Set 1: Short / Interactive Chat",
        "prompt_tokens": 128,
        "output_tokens": 128,
        "description": "Evaluates interactive TTFT responsiveness and single-token decode latency",
    },
    {
        "set_id": "set_2_medium",
        "name": "Set 2: Medium / Code & Summary",
        "prompt_tokens": 1024,
        "output_tokens": 512,
        "description": "Evaluates sustained decode throughput, KV gather, and GDN recurrence",
    },
    {
        "set_id": "set_3_long",
        "name": "Set 3: Long / Repo Context & RAG",
        "prompt_tokens": 4096,
        "output_tokens": 256,
        "description": "Evaluates ESIMD INT8 DPAS prefill scaling and PCIe streaming stability",
    },
]

QUICK_SETS = [
    {
        "set_id": "set_1_short",
        "name": "Set 1: Short (Quick)",
        "prompt_tokens": 64,
        "output_tokens": 64,
        "description": "Quick test for interactive latency",
    },
    {
        "set_id": "set_2_medium",
        "name": "Set 2: Medium (Quick)",
        "prompt_tokens": 512,
        "output_tokens": 128,
        "description": "Quick test for medium prefill & decode",
    },
    {
        "set_id": "set_3_long",
        "name": "Set 3: Long (Quick)",
        "prompt_tokens": 1536,
        "output_tokens": 128,
        "description": "Quick test for long context prefill",
    },
]

# -----------------------------------------------------------------------------
# Empirical A/B Parameter Tournaments
# -----------------------------------------------------------------------------
TOURNAMENT_CATEGORIES = [
    {
        "category_id": "grf_allocation",
        "parameter_name": "STRATA_MMVQ_DECODE_256GRF",
        "title": "Register File Mode (128-GRF Occupancy vs 256-GRF Spill Prevention)",
        "metric_focus": "decode_tok_per_sec",
        "candidates": [
            {
                "label": "128-GRF (High thread occupancy)",
                "env": {"STRATA_MMVQ_DECODE_256GRF": "0"},
            },
            {
                "label": "256-GRF (Zero-spill wide register file)",
                "env": {"STRATA_MMVQ_DECODE_256GRF": "1"},
            },
        ],
    },
    {
        "category_id": "prefill_attention",
        "parameter_name": "STRATA_PROMPT_ATTN_ESIMD",
        "title": "Hardware Systolic Prefill (Generic Fallback vs ESIMD INT8 DPAS)",
        "metric_focus": "prefill_tok_per_sec",
        "candidates": [
            {
                "label": "Fallback (Joint Matrix Generic)",
                "env": {"STRATA_PROMPT_ATTN_ESIMD": "0"},
            },
            {
                "label": "ESIMD INT8 DPAS (Standard LV=3)",
                "env": {"STRATA_PROMPT_ATTN_ESIMD": "1", "STRATA_ESIMD_LV": "3", "STRATA_PF_XMX": "2"},
            },
            {
                "label": "ESIMD INT8 DPAS (Fast LV=2)",
                "env": {"STRATA_PROMPT_ATTN_ESIMD": "1", "STRATA_ESIMD_LV": "2", "STRATA_PF_XMX": "2"},
            },
        ],
    },
    {
        "category_id": "attn_reduction",
        "parameter_name": "STRATA_ATTN_LANECELL",
        "title": "Attention Scoring Reduction (LaneCell vs Subgroup Warp Reductions)",
        "metric_focus": "decode_tok_per_sec",
        "candidates": [
            {
                "label": "Subgroup Warp Reductions (0, Zero-Spill)",
                "env": {"STRATA_ATTN_LANECELL": "0"},
            },
            {
                "label": "LaneCell SLM Accumulation (1)",
                "env": {"STRATA_ATTN_LANECELL": "1"},
            },
        ],
    },
    {
        "category_id": "gdn_fusion",
        "parameter_name": "STRATA_FUSED_GDN",
        "title": "Linear Recurrence (Unfused Dispatches vs Fused SLM Recurrence)",
        "metric_focus": "decode_tok_per_sec",
        "candidates": [
            {
                "label": "Unfused Recurrence Passes (0)",
                "env": {"STRATA_FUSED_GDN": "0"},
            },
            {
                "label": "Fused SLM Recurrence + RMSNorm (1)",
                "env": {"STRATA_FUSED_GDN": "1"},
            },
        ],
    },
    {
        "category_id": "graph_replay",
        "parameter_name": "STRATA_SYCL_GRAPH",
        "title": "Execution Graph (Direct CPU Queue Submissions vs Command Graph Replay)",
        "metric_focus": "decode_tok_per_sec",
        "candidates": [
            {
                "label": "Direct CPU Submissions (0, No graph)",
                "env": {"STRATA_SYCL_GRAPH": "0", "STRATA_VERIFY_GRAPH": "0"},
            },
            {
                "label": "Chunked SYCL Graph (CHUNK=4 layers)",
                "env": {
                    "STRATA_SYCL_GRAPH": "1",
                    "STRATA_VERIFY_GRAPH": "split",
                    "STRATA_VERIFY_GRAPH_CHUNK": "4",
                    "STRATA_VERIFY_DEVICE_PLAN": "1",
                    "STRATA_VERIFY_LEAN": "1",
                },
            },
            {
                "label": "Chunked SYCL Graph (CHUNK=2 layers)",
                "env": {
                    "STRATA_SYCL_GRAPH": "1",
                    "STRATA_VERIFY_GRAPH": "split",
                    "STRATA_VERIFY_GRAPH_CHUNK": "2",
                    "STRATA_VERIFY_DEVICE_PLAN": "1",
                    "STRATA_VERIFY_LEAN": "1",
                },
            },
        ],
    },
    {
        "category_id": "pcie_resident_bias",
        "parameter_name": "STRATA_ROUTE_RESIDENT",
        "title": "PCIe 4.0 x8 MoE Streaming (Resident Margin Bias)",
        "metric_focus": "decode_tok_per_sec",
        "candidates": [
            {
                "label": "Pure Router Selection (0.00 margin)",
                "env": {"STRATA_ROUTE_RESIDENT": "0.00", "STRATA_PIN_SLICES": "1"},
            },
            {
                "label": "Resident Bias 0.10 (Ranks 6-9)",
                "env": {
                    "STRATA_ROUTE_RESIDENT": "0.10",
                    "STRATA_ROUTE_RESIDENT_RANKS": "6-9",
                    "STRATA_PIN_SLICES": "1",
                    "STRATA_ARENA_PIN_GIB": "16",
                },
            },
        ],
    },
]

# -----------------------------------------------------------------------------
# Cumulative Progressive Stacks (Alternative Mode)
# -----------------------------------------------------------------------------
PROGRESSIVE_STACKS = [
    {
        "id": "stack_0_baseline",
        "name": "0. Baseline (All special toggles OFF)",
        "description": "Standard un-graphed direct dispatches, generic joint_matrix fallback, 128-GRF, unfused",
        "env": {
            "STRATA_MMVQ_DECODE_256GRF": "0",
            "STRATA_PROMPT_ATTN_ESIMD": "0",
            "STRATA_ATTN_LANECELL": "1",
            "STRATA_FUSED_GDN": "0",
            "STRATA_FUSED_SWIGLU_Q81": "0",
            "STRATA_SYCL_GRAPH": "0",
            "STRATA_B6_MMVQ_ROWS": "0",
            "STRATA_MMVF_ROWS": "0",
            "STRATA_PIN_SLICES": "0",
            "STRATA_ROUTE_RESIDENT": "0.00",
        },
    },
    {
        "id": "stack_1_registers",
        "name": "1. Compute & Register Spills (256-GRF + LaneCell=0)",
        "description": "256-GRF to eliminate 12-16KB spills on quantized weights + zero-spill subgroup reductions",
        "env": {
            "STRATA_MMVQ_DECODE_256GRF": "1",
            "STRATA_ATTN_LANECELL": "0",
            "STRATA_MMVF_SG": "32",
            "STRATA_PROMPT_ATTN_ESIMD": "0",
            "STRATA_FUSED_GDN": "0",
            "STRATA_SYCL_GRAPH": "0",
        },
    },
    {
        "id": "stack_2_esimd_dpas",
        "name": "2. Hardware Systolic Prefill (ESIMD INT8 DPAS)",
        "description": "Stack 1 + Intel Xe2 INT8 DPAS systolic matrix prompt attention (LV=2)",
        "env": {
            "STRATA_MMVQ_DECODE_256GRF": "1",
            "STRATA_ATTN_LANECELL": "0",
            "STRATA_MMVF_SG": "32",
            "STRATA_PROMPT_ATTN_ESIMD": "1",
            "STRATA_ESIMD_LV": "2",
            "STRATA_PF_XMX": "2",
            "STRATA_PREFILL_MMQ": "0",
            "STRATA_FUSED_GDN": "0",
            "STRATA_SYCL_GRAPH": "0",
        },
    },
    {
        "id": "stack_3_fusions",
        "name": "3. Kernel Fusions (Fused GDN + SwiGLU Q8_1 + Multi-Row)",
        "description": "Stack 2 + Fused SLM recurrence + SwiGLU Q8_1 epilogue + Multi-Row GEMV",
        "env": {
            "STRATA_MMVQ_DECODE_256GRF": "1",
            "STRATA_ATTN_LANECELL": "0",
            "STRATA_MMVF_SG": "32",
            "STRATA_PROMPT_ATTN_ESIMD": "1",
            "STRATA_ESIMD_LV": "2",
            "STRATA_PF_XMX": "2",
            "STRATA_PREFILL_MMQ": "0",
            "STRATA_FUSED_GDN": "1",
            "STRATA_FUSED_SWIGLU_Q81": "1",
            "STRATA_B6_MMVQ_ROWS": "1",
            "STRATA_MMVF_ROWS": "1",
            "STRATA_SYCL_GRAPH": "0",
        },
    },
    {
        "id": "stack_4_sycl_graph",
        "name": "4. SYCL Command Graph Replay (Zero CPU Dispatch Overhead)",
        "description": "Stack 3 + Level Zero Command Graph replay (chunked 4 layers)",
        "env": {
            "STRATA_MMVQ_DECODE_256GRF": "1",
            "STRATA_ATTN_LANECELL": "0",
            "STRATA_MMVF_SG": "32",
            "STRATA_PROMPT_ATTN_ESIMD": "1",
            "STRATA_ESIMD_LV": "2",
            "STRATA_PF_XMX": "2",
            "STRATA_PREFILL_MMQ": "0",
            "STRATA_FUSED_GDN": "1",
            "STRATA_FUSED_SWIGLU_Q81": "1",
            "STRATA_B6_MMVQ_ROWS": "1",
            "STRATA_MMVF_ROWS": "1",
            "STRATA_SYCL_GRAPH": "1",
            "STRATA_VERIFY_GRAPH": "split",
            "STRATA_VERIFY_GRAPH_CHUNK": "4",
            "STRATA_VERIFY_DEVICE_PLAN": "1",
            "STRATA_VERIFY_LEAN": "1",
        },
    },
    {
        "id": "stack_5_full_optimal",
        "name": "5. Fully Optimized + PCIe 4.0 x8 Guard + Upstream Merges",
        "description": "Stack 4 + Active chunk merge + Gate/Up reuse + L3 cache hint + PCIe resident bias",
        "env": {
            "STRATA_MMVQ_DECODE_256GRF": "1",
            "STRATA_ATTN_LANECELL": "0",
            "STRATA_MMVF_SG": "32",
            "STRATA_PROMPT_ATTN_ESIMD": "1",
            "STRATA_ESIMD_LV": "2",
            "STRATA_PF_XMX": "2",
            "STRATA_PREFILL_MMQ": "0",
            "STRATA_FUSED_GDN": "1",
            "STRATA_FUSED_SWIGLU_Q81": "1",
            "STRATA_B6_MMVQ_ROWS": "1",
            "STRATA_MMVF_ROWS": "1",
            "STRATA_SYCL_GRAPH": "1",
            "STRATA_VERIFY_GRAPH": "split",
            "STRATA_VERIFY_GRAPH_CHUNK": "4",
            "STRATA_VERIFY_DEVICE_PLAN": "1",
            "STRATA_VERIFY_LEAN": "1",
            "STRATA_ATTN_MERGE_ACTIVE": "1",
            "STRATA_IQ3S_GU_ROW_REUSE": "1",
            "STRATA_XE2_CACHE_HINT": "1",
            "STRATA_PIN_SLICES": "1",
            "STRATA_ARENA_PIN_GIB": "16",
            "STRATA_ROUTE_RESIDENT": "0.10",
            "STRATA_ROUTE_RESIDENT_RANKS": "6-9",
        },
    },
]


def generate_prompt_payload(tokens_count: int) -> str:
    """Generate deterministic English text payload matching targeted token count (~4 chars/token)."""
    base_sentences = [
        "The quick brown fox jumps over the lazy dog and explores the Intel Arc GPU architecture. ",
        "Modern deep learning inference systems leverage systolic tensor arrays for matrix multiplication. ",
        "Dynamic register file allocation reduces scratchpad memory spill overhead on execution units. ",
        "Level Zero command graph replay eliminates CPU driver submission latency during autoregressive token generation. ",
    ]
    block = "".join(base_sentences)
    repeat = max(1, int((tokens_count * 4) / len(block)) + 1)
    full_text = block * repeat
    return full_text[: tokens_count * 4]


def benchmark_streaming_request(url: str, prompt_tokens: int, max_tokens: int, api_key: str = "") -> dict:
    """Send an OpenAI-compatible streaming completion request and measure TTFT and generation rate."""
    prompt_text = generate_prompt_payload(prompt_tokens)
    payload = {
        "model": "strata",
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
    )

    t0 = time.perf_counter()
    t_first_token = None
    output_tokens = 0
    reported_usage = None

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: ") :].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                    if "usage" in chunk and chunk["usage"]:
                        reported_usage = chunk["usage"]

                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if delta.get("content"):
                            if t_first_token is None:
                                t_first_token = time.perf_counter()
                            output_tokens += 1
                except json.JSONDecodeError:
                    continue

    except urllib.error.URLError as e:
        return {
            "success": False,
            "error": str(e),
            "prompt_target": prompt_tokens,
            "output_target": max_tokens,
        }

    t_end = time.perf_counter()
    total_time_s = t_end - t0
    ttft_s = (t_first_token - t0) if t_first_token is not None else total_time_s
    ttft_ms = ttft_s * 1000.0

    actual_prompt_tokens = reported_usage.get("prompt_tokens", prompt_tokens) if reported_usage else prompt_tokens
    actual_output_tokens = (
        reported_usage.get("completion_tokens", output_tokens) if reported_usage else output_tokens
    )

    prefill_tok_per_sec = (actual_prompt_tokens / ttft_s) if ttft_s > 0 else 0.0
    gen_time_s = max(1e-4, total_time_s - ttft_s)
    decode_tok_per_sec = (actual_output_tokens / gen_time_s) if actual_output_tokens > 0 else 0.0

    return {
        "success": True,
        "prompt_target": prompt_tokens,
        "output_target": max_tokens,
        "actual_prompt_tokens": actual_prompt_tokens,
        "actual_output_tokens": actual_output_tokens,
        "ttft_ms": ttft_ms,
        "prefill_tok_per_sec": prefill_tok_per_sec,
        "decode_tok_per_sec": decode_tok_per_sec,
        "total_time_s": total_time_s,
    }


def wait_for_server_ready(url: str, timeout_s: float = 180.0) -> bool:
    """Poll server /health endpoint until 'loaded' is true or timeout expires."""
    t0 = time.time()
    health_url = f"{url.rstrip('/')}/health"
    while time.time() - t0 < timeout_s:
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data.get("loaded", False):
                        return True
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            pass
        time.sleep(1.0)
    return False


def launch_server_process(config_path: str, port: int, env_vars: dict[str, str], engine_bin: str = "strata") -> subprocess.Popen:
    """Launch sycl/serve/server_intel.py subprocess with specified environment overrides."""
    full_env = os.environ.copy()
    full_env.update(env_vars)
    full_env["ONEAPI_DEVICE_SELECTOR"] = full_env.get("ONEAPI_DEVICE_SELECTOR", "level_zero:0")
    full_env["ZES_ENABLE_SYSMAN"] = "1"

    cmd = [
        sys.executable,
        str(ROOT / "sycl" / "serve" / "server_intel.py"),
        "--engine",
        engine_bin,
        "--config",
        config_path,
        "--port",
        str(port),
    ]

    print(f"\n[Manager] Spawning server_intel.py on port {port}...")
    proc = subprocess.Popen(cmd, env=full_env, cwd=str(ROOT))
    return proc


def stop_server_process(proc: subprocess.Popen, timeout_s: float = 15.0):
    """Gracefully terminate server subprocess."""
    if proc.poll() is not None:
        return
    print("[Manager] Terminating server subprocess...")
    proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print("[Manager] Force killing server subprocess (SIGKILL)...")
        proc.kill()
        proc.wait()


def interactive_tweak_prompt(title: str, env_dict: dict[str, str]) -> tuple[bool, dict[str, str]]:
    """Display environment variables and let the user tweak them before running."""
    print("\n" + "=" * 80)
    print(f" Upcoming Test Run: {title}")
    print("=" * 80)
    for k, v in env_dict.items():
        print(f"  export {k}={v}")
    print("-" * 80)
    print(" Options:")
    print("   [Enter] Run this test candidate")
    print("   [t]     Tweak or set an environment variable (e.g. STRATA_ESIMD_LV=3)")
    print("   [s]     Skip this candidate")
    print("   [q]     Quit tuning session and finalize report")

    while True:
        try:
            choice = input("\nChoice [Enter/t/s/q]: ").strip().lower()
        except EOFError:
            return True, env_dict

        if choice in ("", "run", "y"):
            return True, env_dict
        elif choice == "s":
            print(f"Skipping {title}...")
            return False, env_dict
        elif choice == "q":
            print("Aborting remaining tests...")
            return None, env_dict
        elif choice == "t":
            var_input = input("Enter KEY=VALUE to set/override: ").strip()
            if "=" in var_input:
                k, v = var_input.split("=", 1)
                env_dict[k.strip()] = v.strip()
                print(f"  -> Updated: {k.strip()} = {v.strip()}")
            else:
                print("Invalid format. Expected KEY=VALUE.")
            print("\nCurrent active variables for this test:")
            for k, v in env_dict.items():
                print(f"  export {k}={v}")


def evaluate_active_server_across_sets(target_url: str, bench_sets: list[dict], api_key: str = "", warmup: bool = True) -> dict:
    """Run all evaluation sets against the active server and return results dict."""
    if warmup:
        print("  Performing discardable warmup pass (64 in / 32 out)...")
        benchmark_streaming_request(target_url, 64, 32, api_key)

    set_results = {}
    for bset in bench_sets:
        print(f"  Testing {bset['name']} (Prompt: ~{bset['prompt_tokens']}, Output: {bset['output_tokens']})...")
        r = benchmark_streaming_request(target_url, bset["prompt_tokens"], bset["output_tokens"], api_key)
        if r.get("success"):
            print(f"    -> TTFT: {r['ttft_ms']:.1f} ms ({r['prefill_tok_per_sec']:.1f} p-tok/s) | Generation: {r['decode_tok_per_sec']:.1f} tok/s")
        else:
            print(f"    -> FAILED: {r.get('error')}")
        set_results[bset["set_id"]] = r

    # Compute aggregate scores
    valid_decodes = [v["decode_tok_per_sec"] for v in set_results.values() if v.get("success")]
    valid_prefills = [v["prefill_tok_per_sec"] for v in set_results.values() if v.get("success")]
    valid_ttfts = [v["ttft_ms"] for v in set_results.values() if v.get("success")]

    avg_decode = sum(valid_decodes) / len(valid_decodes) if valid_decodes else 0.0
    avg_prefill = sum(valid_prefills) / len(valid_prefills) if valid_prefills else 0.0
    avg_ttft = sum(valid_ttfts) / len(valid_ttfts) if valid_ttfts else 0.0

    return {
        "sets": set_results,
        "summary": {
            "avg_decode_tok_s": avg_decode,
            "avg_prefill_tok_s": avg_prefill,
            "avg_ttft_ms": avg_ttft,
        },
    }


def format_tournament_markdown(tournaments: list[dict], winning_env: dict[str, str], bench_sets: list[dict]) -> str:
    """Generate Markdown report detailing tournament results, decisions, and winning environment."""
    lines = []
    lines.append("# Intel Arc B580 Empirical Decision Report\n\n")
    lines.append(f"> **Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
    lines.append("> **Hardware Target**: Intel Arc B580 (Xe2 Battlemage, 456 GB/s GDDR6, PCIe 4.0 x8)  \n")
    lines.append("> **Rule Enforced**: No predefined settings taken as gospel. Every decision is backed strictly by measured runs.  \n\n")

    lines.append("## 1. Empirical Decision Summary & Winning Profile\n\n")
    lines.append("| Parameter | Evaluated Candidates | Empirical Winner | Measured Advantage | Target Subsystem |\n")
    lines.append("|---|---|---|---|---|\n")

    for t in tournaments:
        pname = t["parameter_name"]
        winner = t.get("winner", {})
        delta_str = t.get("decision_note", "Evaluated")
        c_count = len(t.get("candidates", []))
        lines.append(f"| `{pname}` | {c_count} candidates tested | **`{winner.get('label', 'Default')}`** | {delta_str} | {t['title']} |\n")

    lines.append("\n### Auto-Generated Optimal Environment Profile (`b580_empirical_best.env`)\n\n")
    lines.append("```bash\n")
    lines.append("#!/usr/bin/env bash\n")
    lines.append("# Auto-generated based on empirical live runs on Intel Arc B580\n")
    for k, v in winning_env.items():
        lines.append(f"export {k}={v}\n")
    lines.append("```\n\n---\n\n")

    lines.append("## 2. Detailed Parameter Tournaments Across 3 Evaluation Sets\n\n")

    for t in tournaments:
        lines.append(f"### {t['title']} (`{t['parameter_name']}`)\n\n")
        lines.append(f"**Primary Decision Metric**: `{t['metric_focus']}`  \n\n")

        # Table header
        header = ["Candidate Option"]
        for s in bench_sets:
            header.append(f"{s['name']}<br>(P:{s['prompt_tokens']} / Out:{s['output_tokens']})")
        header.append("Average Throughput")
        header.append("Status")

        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        winner_label = t.get("winner", {}).get("label")

        for c in t.get("candidates", []):
            label = c["label"]
            row = [f"**{label}**"]
            c_sets = c.get("results", {}).get("sets", {})

            for s in bench_sets:
                sid = s["set_id"]
                sres = c_sets.get(sid, {})
                if sres.get("success"):
                    ttft = sres.get("ttft_ms", 0)
                    p_tps = sres.get("prefill_tok_per_sec", 0)
                    d_tps = sres.get("decode_tok_per_sec", 0)
                    row.append(f"TTFT: {ttft:.0f} ms<br>Prefill: {p_tps:.0f} t/s<br>Decode: **{d_tps:.1f} t/s**")
                else:
                    row.append("❌ Failed")

            avg_m = c.get("results", {}).get("summary", {})
            focus_val = avg_m.get("avg_decode_tok_s" if t["metric_focus"] == "decode_tok_per_sec" else "avg_prefill_tok_s", 0.0)
            row.append(f"**{focus_val:.1f} tok/s**")

            status_str = "🏆 **WINNER**" if label == winner_label else "Runner-Up"
            row.append(status_str)

            lines.append("| " + " | ".join(row) + " |")

        lines.append("\n\n")

    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Empirical Hardware Tuning & Decision Engine for Strata on Intel Arc B580")
    parser.add_argument("--url", default="", help="Connect to an already running server URL (e.g. http://127.0.0.1:8080)")
    parser.add_argument("--config", default="", help="Path to model config JSON to launch server_intel.py automatically")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--api-key", default="", help="Optional API key for server authorization")
    parser.add_argument("--mode", choices=["tournament", "stack"], default="tournament",
                        help="Tuning mode: 'tournament' (decides each variable by A/B test) or 'stack' (progressive cumulative tiers)")
    parser.add_argument("--interactive", action="store_true", help="Prompt before each test run to allow tweaking environment variables")
    parser.add_argument("--auto", action="store_true", help="Run unattended without pausing for user input")
    parser.add_argument("--quick", action="store_true", help="Use faster, shorter token lengths for rapid validation")
    parser.add_argument("--sets", default="", help="Custom prompt/output lengths format: 'P1/O1,P2/O2,P3/O3'")
    parser.add_argument("--out-dir", default="bench_results/tuning", help="Directory to save JSON & Markdown reports")
    parser.add_argument("--no-warmup", action="store_true", help="Skip discardable warmup request")
    args = parser.parse_args()

    # Determine evaluation sets
    if args.sets:
        bench_sets = []
        for i, pair in enumerate(args.sets.split(","), 1):
            if "/" in pair:
                p, o = pair.split("/", 1)
                bench_sets.append({
                    "set_id": f"custom_set_{i}",
                    "name": f"Custom Set {i}",
                    "prompt_tokens": int(p.strip()),
                    "output_tokens": int(o.strip()),
                    "description": f"Custom evaluation {p.strip()} in / {o.strip()} out",
                })
    elif args.quick:
        bench_sets = list(QUICK_SETS)
    else:
        bench_sets = list(STANDARD_SETS)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print("=" * 80)
    print(" Intel Arc B580 Empirical Decision & Tuning Engine")
    print(f" Execution Mode:     {args.mode.upper()} (No assumptions as gospel; decisions backed by runs)")
    print(f" Benchmark Sets:     {len(bench_sets)} distinct prompt/output profiles:")
    for s in bench_sets:
        print(f"   * {s['name']}: Prompt ~{s['prompt_tokens']} tokens -> Output {s['output_tokens']} tokens")
    print(f" Output Folder:      {out_dir}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Mode 1: Live Testing against an already running server instance
    # -------------------------------------------------------------------------
    if args.url:
        target_url = args.url.rstrip("/")
        print(f"\n[Runner] Operating in Live Connection Mode against {target_url}...")
        if not wait_for_server_ready(target_url, timeout_s=10.0):
            print(f"Error: Server at {target_url} is not responding or model is not loaded.")
            sys.exit(1)

        print(f"\nEvaluating active server across {len(bench_sets)} sets...")
        active_res = evaluate_active_server_across_sets(target_url, bench_sets, args.api_key, warmup=not args.no_warmup)

        print("\n" + "=" * 80)
        print(" Active Server Results Summary:")
        print(f"   Average TTFT:        {active_res['summary']['avg_ttft_ms']:.1f} ms")
        print(f"   Average Prefill:     {active_res['summary']['avg_prefill_tok_s']:.1f} tok/s")
        print(f"   Average Decode:      {active_res['summary']['avg_decode_tok_s']:.1f} tok/s")
        print("=" * 80)

        json_path = out_dir / f"live_server_results_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump({"timestamp": timestamp, "results": active_res}, f, indent=2)
        print(f"Saved results to {json_path}")
        return

    # -------------------------------------------------------------------------
    # Mode 2: Server Lifecycle Manager - Empirical Tournaments
    # -------------------------------------------------------------------------
    if not args.config:
        print("Error: In server management mode, --config <strata-model.json> is required.")
        print("Alternatively, pass --url http://127.0.0.1:8080 to test an already running server.")
        sys.exit(1)

    target_url = f"http://127.0.0.1:{args.port}"
    interactive = args.interactive or not args.auto
    base_env = {}

    winning_env = {}
    tournament_records = []

    if args.mode == "tournament":
        print("\nStarting Empirical Parameter Tournaments...")

        for t_idx, tournament in enumerate(TOURNAMENT_CATEGORIES, 1):
            pname = tournament["parameter_name"]
            p_title = tournament["title"]
            focus_metric = tournament["metric_focus"]

            print("\n" + "=" * 80)
            print(f" Tournament [{t_idx}/{len(TOURNAMENT_CATEGORIES)}]: {p_title}")
            print(f" Optimizing: `{pname}` (Focus Metric: {focus_metric})")
            print("=" * 80)

            candidate_results = []

            for c_idx, candidate in enumerate(tournament["candidates"], 1):
                label = candidate["label"]
                cand_env = dict(winning_env)  # Carry forward previous winning choices
                cand_env.update(candidate["env"])

                if interactive:
                    decision, cand_env = interactive_tweak_prompt(f"{p_title} -> {label}", cand_env)
                    if decision is None:  # User abort
                        print("Terminating tournament sweep early...")
                        break
                    if not decision:  # User skipped
                        continue

                print(f"\n  Evaluating Candidate [{c_idx}/{len(tournament['candidates'])}]: {label}...")
                server_proc = launch_server_process(args.config, args.port, cand_env)

                try:
                    print(f"  [Manager] Waiting for server at {target_url} to load and initialize...")
                    ready = wait_for_server_ready(target_url, timeout_s=180.0)
                    if not ready:
                        print(f"  ❌ Server failed to load for candidate: {label}")
                        continue

                    print("  ✅ Server loaded. Running 3 benchmark sets...")
                    run_eval = evaluate_active_server_across_sets(target_url, bench_sets, args.api_key, warmup=not args.no_warmup)
                    candidate_results.append({
                        "label": label,
                        "env": candidate["env"],
                        "active_env": cand_env,
                        "results": run_eval,
                    })

                finally:
                    stop_server_process(server_proc)
                    time.sleep(2.0)

            if not candidate_results:
                print(f"Warning: No valid candidate runs for {pname}. Skipping decision.")
                continue

            # Determine empirical winner for this category
            def score_candidate(cand):
                s = cand["results"]["summary"]
                return s["avg_decode_tok_s"] if focus_metric == "decode_tok_per_sec" else s["avg_prefill_tok_s"]

            winner = max(candidate_results, key=score_candidate)
            winner_score = score_candidate(winner)

            baseline_score = score_candidate(candidate_results[0])
            delta_pct = ((winner_score - baseline_score) / baseline_score * 100.0) if baseline_score > 0 else 0.0

            decision_note = f"+{delta_pct:.1f}% faster" if delta_pct > 0 else "Optimal baseline"
            print("\n" + "-" * 80)
            print(f" 🏆 Empirical Winner for {pname}: **{winner['label']}** ({decision_note})")
            print("-" * 80)

            # Lock in winning variables into accumulated best env
            winning_env.update(winner["env"])

            tournament_records.append({
                "category_id": tournament["category_id"],
                "parameter_name": pname,
                "title": p_title,
                "metric_focus": focus_metric,
                "winner": winner,
                "decision_note": decision_note,
                "candidates": candidate_results,
            })

        # Save optimal script and report
        best_env_path = out_dir / "b580_empirical_best.env"
        with open(best_env_path, "w") as ef:
            ef.write("#!/usr/bin/env bash\n")
            ef.write(f"# Auto-generated optimal environment based on live runs on {timestamp}\n")
            ef.write(f"# Evaluated model config: {args.config}\n\n")
            for k, v in winning_env.items():
                ef.write(f"export {k}={v}\n")
        os.chmod(best_env_path, 0o755)

        md_report = format_tournament_markdown(tournament_records, winning_env, bench_sets)
        md_path = out_dir / f"tuning_decision_report_{timestamp}.md"
        with open(md_path, "w") as mf:
            mf.write(md_report)

        json_path = out_dir / f"tournament_results_{timestamp}.json"
        with open(json_path, "w") as jf:
            json.dump({
                "timestamp": timestamp,
                "winning_env": winning_env,
                "tournaments": tournament_records,
            }, jf, indent=2)

        print("\n" + "=" * 80)
        print(" Empirical Tuning Complete! All decisions decided by live runs.")
        print(f" Optimal Environment Script: {best_env_path}")
        print(f" Detailed Decision Report:   {md_path}")
        print("=" * 80)
        print("\n" + md_report)

    elif args.mode == "stack":
        print("\nRunning Progressive Cumulative Stacks...")
        # Runs the PROGRESSIVE_STACKS matrix and generates report
        stack_results = []
        for idx, cfg in enumerate(PROGRESSIVE_STACKS, 1):
            env_to_use = dict(cfg["env"])
            if interactive:
                decision, env_to_use = interactive_tweak_prompt(cfg["name"], env_to_use)
                if decision is None:
                    break
                if not decision:
                    continue

            print(f"\n[{idx}/{len(PROGRESSIVE_STACKS)}] Evaluating Stack: {cfg['name']}...")
            server_proc = launch_server_process(args.config, args.port, env_to_use)

            try:
                print(f"[Manager] Waiting for server at {target_url} to initialize...")
                ready = wait_for_server_ready(target_url, timeout_s=180.0)
                if not ready:
                    print(f"❌ Server failed to load for {cfg['name']}.")
                    continue

                run_eval = evaluate_active_server_across_sets(target_url, bench_sets, args.api_key, warmup=not args.no_warmup)
                stack_results.append({"config": {**cfg, "env": env_to_use}, "sets": run_eval["sets"]})
            finally:
                stop_server_process(server_proc)
                time.sleep(2.0)

        json_path = out_dir / f"stack_results_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump({"timestamp": timestamp, "results": stack_results}, f, indent=2)
        print(f"\nSaved stack results to {json_path}")


if __name__ == "__main__":
    main()
