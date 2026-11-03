#!/usr/bin/env python3
"""tools/tune_server_b580.py - Systematic Environment Tuning & Benchmarking Suite for Strata on Intel Arc B580.

Orchestrates systematic environment variable evaluation across the 5 optimization tiers using
sycl/serve/server_intel.py, measuring:
  - Time-To-First-Token (TTFT, ms) & Prompt Processing Throughput (tok/s)
  - Autoregressive Generation Throughput (output tok/s)
  - Latency scaling across 3 distinct Prompt/Output sets:
      * Set 1: Short / Interactive Chat         (~128 prompt,  128 output)
      * Set 2: Medium / Agentic Code & Summary   (~1024 prompt, 512 output)
      * Set 3: Long / Repository Context & RAG   (~4096 prompt, 256 output)

Features:
  - Interactive Tweak Mode: Allows reviewing, modifying, or adding env vars before each test.
  - Non-Interactive / Automated Mode (--auto): Runs full matrix unattended.
  - Dual Execution Modes:
      * Live Testing: Benchmark against an already running server (--url http://127.0.0.1:8080).
      * Server Lifecycle Manager: Automatically boots, tests, and cycles server_intel.py with configs.
  - Markdown & JSON reporting: Generates formatted performance comparison tables and speedups.

Usage:
  # 1. Test currently running server on port 8080 across all 3 sets:
  python3 tools/tune_server_b580.py --url http://127.0.0.1:8080

  # 2. Automatically launch and sweep configurations with a model config:
  python3 tools/tune_server_b580.py --config strata-b580-iq3s.json --port 8080

  # 3. Interactive step-by-step tweaking mode:
  python3 tools/tune_server_b580.py --config strata-b580-iq3s.json --interactive

  # 4. Custom prompt/output lengths:
  python3 tools/tune_server_b580.py --url http://127.0.0.1:8080 --sets "64/64,512/256,2048/128"
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
DEFAULT_SETS = [
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

# -----------------------------------------------------------------------------
# Systematic Environment Tuning Configurations
# -----------------------------------------------------------------------------
TUNING_CONFIGS = [
    {
        "id": "config_0_baseline",
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
        "id": "config_1_registers",
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
        "id": "config_2_esimd_dpas",
        "name": "2. Hardware Systolic Prefill (ESIMD INT8 DPAS)",
        "description": "Config 1 + Intel Xe2 INT8 DPAS systolic matrix prompt attention (LV=2)",
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
        "id": "config_3_fusions",
        "name": "3. Kernel Fusions (Fused GDN + SwiGLU Q8_1 + Multi-Row)",
        "description": "Config 2 + Fused SLM recurrence + SwiGLU Q8_1 epilogue + Multi-Row GEMV",
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
        "id": "config_4_sycl_graph",
        "name": "4. SYCL Command Graph Replay (Zero CPU Dispatch Overhead)",
        "description": "Config 3 + Level Zero Command Graph replay (chunked 4 layers)",
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
        "id": "config_5_full_optimal",
        "name": "5. Fully Optimized + PCIe 4.0 x8 Guard + Upstream Merges",
        "description": "Config 4 + Active chunk merge + Gate/Up reuse + L3 cache hint + PCIe resident bias",
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


def wait_for_server_ready(url: str, timeout_s: float = 120.0) -> bool:
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


def interactive_tweak_prompt(cfg_name: str, env_dict: dict[str, str]) -> tuple[bool, dict[str, str]]:
    """Display environment variables and let the user tweak them before running."""
    print("\n" + "=" * 80)
    print(f" Upcoming Configuration: {cfg_name}")
    print("=" * 80)
    for k, v in env_dict.items():
        print(f"  export {k}={v}")
    print("-" * 80)
    print(" Options:")
    print("   [Enter] Proceed with this test")
    print("   [t]     Tweak or set an environment variable (e.g. STRATA_ESIMD_LV=3)")
    print("   [s]     Skip this test configuration")
    print("   [q]     Quit tuning session and generate report")

    while True:
        try:
            choice = input("\nChoice [Enter/t/s/q]: ").strip().lower()
        except EOFError:
            return True, env_dict

        if choice in ("", "run", "y"):
            return True, env_dict
        elif choice == "s":
            print(f"Skipping {cfg_name}...")
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


def format_markdown_report(results: list[dict], bench_sets: list[dict]) -> str:
    """Generate Markdown comparison table showing TTFT, prefill tok/s, and decode tok/s."""
    lines = []
    lines.append("# Intel Arc B580 Systematic Environment Tuning Report\n\n")
    lines.append(f"> **Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
    lines.append("> **Hardware**: Intel Arc B580 (Xe2 Battlemage, 456 GB/s GDDR6, PCIe 4.0 x8)  \n\n")

    lines.append("## 1. Multi-Stage Performance Comparison Matrix\n\n")

    # Table Header
    header = ["Configuration"]
    for s in bench_sets:
        header.append(f"{s['name']}<br>(P:{s['prompt_tokens']} / Out:{s['output_tokens']})")
    header.append("Relative Speedup")

    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    baseline_decode = None

    for r in results:
        cfg = r["config"]
        row = [f"**{cfg['name']}**"]

        set_decodes = []
        for s in bench_sets:
            sid = s["set_id"]
            sres = r.get("sets", {}).get(sid, {})
            if sres.get("success"):
                ttft = sres.get("ttft_ms", 0)
                p_tps = sres.get("prefill_tok_per_sec", 0)
                d_tps = sres.get("decode_tok_per_sec", 0)
                set_decodes.append(d_tps)
                cell = f"TTFT: **{ttft:.0f} ms** ({p_tps:.0f} p-tok/s)<br>Decode: **{d_tps:.1f} tok/s**"
            else:
                cell = "❌ Failed"
            row.append(cell)

        avg_decode = sum(set_decodes) / len(set_decodes) if set_decodes else 0.0
        if baseline_decode is None and avg_decode > 0:
            baseline_decode = avg_decode
            row.append("1.00x *(Baseline)*")
        elif baseline_decode and baseline_decode > 0:
            ratio = avg_decode / baseline_decode
            row.append(f"**{ratio:.2f}x faster**")
        else:
            row.append("-")

        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n---\n\n")
    lines.append("## 2. Tested Environment Configurations Detail\n\n")
    for r in results:
        cfg = r["config"]
        lines.append(f"### {cfg['name']}\n\n")
        lines.append(f"*{cfg.get('description', '')}*\n\n")
        lines.append("```bash\n")
        for k, v in cfg.get("env", {}).items():
            lines.append(f"export {k}={v}\n")
        lines.append("```\n\n")

    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Systematic Environment Tuning Suite for Strata on Intel Arc B580")
    parser.add_argument("--url", default="", help="Connect to an already running server URL (e.g. http://127.0.0.1:8080)")
    parser.add_argument("--config", default="", help="Path to model config JSON to launch server_intel.py automatically")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--api-key", default="", help="Optional API key for server authorization")
    parser.add_argument("--interactive", action="store_true", help="Prompt before each test to allow tweaking environment variables")
    parser.add_argument("--auto", action="store_true", help="Run unattended without user prompts")
    parser.add_argument("--sets", default="", help="Custom prompt/output lengths format: 'P1/O1,P2/O2,P3/O3'")
    parser.add_argument("--out-dir", default="bench_results/tuning", help="Directory to save JSON & Markdown reports")
    parser.add_argument("--warmup", action="store_true", default=True, help="Perform 1 discardable warmup request before recording")
    args = parser.parse_args()

    # Parse evaluation sets
    bench_sets = list(DEFAULT_SETS)
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print("=" * 80)
    print(" Intel Arc B580 Systematic Environment Tuning Suite")
    print(f" Target Port:        {args.port}")
    print(f" Benchmark Sets:     {len(bench_sets)} distinct prompt/output profiles")
    for s in bench_sets:
        print(f"   * {s['name']}: Prompt ~{s['prompt_tokens']} tokens -> Output {s['output_tokens']} tokens")
    print(f" Tuning Matrix:      {len(TUNING_CONFIGS)} optimization configurations")
    print("=" * 80)

    results = []

    # -------------------------------------------------------------------------
    # Mode A: Benchmark against an already running server instance
    # -------------------------------------------------------------------------
    if args.url:
        target_url = args.url.rstrip("/")
        print(f"\n[Runner] Operating in Live Connection Mode against {target_url}...")
        if not wait_for_server_ready(target_url, timeout_s=10.0):
            print(f"Error: Server at {target_url} is not responding or model is not loaded.")
            sys.exit(1)

        cfg_meta = {
            "id": "live_running_server",
            "name": "Live Running Server Configuration",
            "description": f"Active environment of server running on {target_url}",
            "env": {k: v for k, v in os.environ.items() if k.startswith("STRATA_")},
        }

        print(f"\nEvaluating active server across {len(bench_sets)} sets...")
        set_results = {}
        for bset in bench_sets:
            print(f"\n  Running {bset['name']} (Prompt: ~{bset['prompt_tokens']} tok, Output: {bset['output_tokens']} tok)...")
            r = benchmark_streaming_request(target_url, bset["prompt_tokens"], bset["output_tokens"], args.api_key)
            if r.get("success"):
                print(f"    -> TTFT: {r['ttft_ms']:.1f} ms ({r['prefill_tok_per_sec']:.1f} p-tok/s) | Generation: {r['decode_tok_per_sec']:.1f} tok/s")
            else:
                print(f"    -> FAILED: {r.get('error')}")
            set_results[bset["set_id"]] = r

        results.append({"config": cfg_meta, "sets": set_results})

    # -------------------------------------------------------------------------
    # Mode B: Lifecycle manager: cycles through configurations with server_intel.py
    # -------------------------------------------------------------------------
    else:
        if not args.config:
            print("Error: In server management mode, --config <strata-model.json> is required.")
            print("Alternatively, pass --url http://127.0.0.1:8080 to test an already running server.")
            sys.exit(1)

        target_url = f"http://127.0.0.1:{args.port}"
        interactive = args.interactive or not args.auto

        for idx, cfg in enumerate(TUNING_CONFIGS, 1):
            env_to_use = dict(cfg["env"])

            if interactive:
                decision, env_to_use = interactive_tweak_prompt(cfg["name"], env_to_use)
                if decision is None:  # User quit
                    break
                if not decision:  # User skipped
                    continue

            print(f"\n[{idx}/{len(TUNING_CONFIGS)}] Preparing: {cfg['name']}...")
            server_proc = launch_server_process(args.config, args.port, env_to_use)

            try:
                print(f"[Manager] Waiting for server at {target_url} to load weights and initialize...")
                ready = wait_for_server_ready(target_url, timeout_s=180.0)
                if not ready:
                    print(f"❌ Server did not become ready within timeout for {cfg['name']}.")
                    results.append({"config": cfg, "sets": {"error": "Server failed to reach ready state within 180s"}})
                    continue

                print("✅ Server is ready. Running evaluation sets...")

                # Warmup
                if args.warmup:
                    print("  Performing discardable warmup pass...")
                    benchmark_streaming_request(target_url, 64, 32, args.api_key)

                set_results = {}
                for bset in bench_sets:
                    print(f"  Testing {bset['name']} (Prompt: ~{bset['prompt_tokens']}, Output: {bset['output_tokens']})...")
                    r = benchmark_streaming_request(target_url, bset["prompt_tokens"], bset["output_tokens"], args.api_key)
                    if r.get("success"):
                        print(f"    -> TTFT: {r['ttft_ms']:.1f} ms ({r['prefill_tok_per_sec']:.1f} p-tok/s) | Generation: {r['decode_tok_per_sec']:.1f} tok/s")
                    else:
                        print(f"    -> FAILED: {r.get('error')}")
                    set_results[bset["set_id"]] = r

                results.append({"config": {**cfg, "env": env_to_use}, "sets": set_results})

            finally:
                stop_server_process(server_proc)
                time.sleep(2.0)  # Brief pause for socket release

    # -------------------------------------------------------------------------
    # Generate Output Reports
    # -------------------------------------------------------------------------
    json_path = out_dir / f"tuning_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({"timestamp": timestamp, "results": results}, f, indent=2)

    md_report = format_markdown_report(results, bench_sets)
    md_path = out_dir / f"tuning_report_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write(md_report)

    print("\n" + "=" * 80)
    print(" Tuning Session Complete!")
    print(f" Markdown Report: {md_path}")
    print(f" JSON Raw Data:   {json_path}")
    print("=" * 80)
    print("\n" + md_report)


if __name__ == "__main__":
    main()
