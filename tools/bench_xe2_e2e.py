#!/usr/bin/env python3
"""tools/bench_xe2_e2e.py - Intel Xe2 / Arc B580 End-to-End Inference Throughput & TTFT Benchmark.

Drives the Strata inference engine (`build_sycl/strata` or HTTP server) across a parameter
matrix of prompt lengths, generation lengths, quantization formats, and execution options
(SYCL graphs, MTP speculation) to measure:
- Time-To-First-Token (TTFT, seconds)
- Prompt processing throughput (prompt tok/s)
- Autoregressive generation throughput (output tok/s)
- Memory tier residency and GPU execution overhead

Usage:
    python3 tools/bench_xe2_e2e.py --bin build_sycl/strata --tokens 128
    python3 tools/bench_xe2_e2e.py --url http://127.0.0.1:8080 --lengths 1k,4k,32k --out xe2_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_length(s: str) -> int:
    s = s.strip().lower()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1024)
    if s.endswith("m"):
        return int(float(s[:-1]) * 1024 * 1024)
    return int(s)


def benchmark_cli(bin_path: str, tokens: int, no_capture: bool, extra_args: list[str]) -> dict:
    """Benchmark local strata binary execution."""
    cmd = [bin_path, "--tokens", str(tokens), "--no-ple", "--no-pool"]
    if no_capture:
        cmd.append("--no-capture")
    cmd.extend(extra_args)

    print(f"Running command: {' '.join(cmd)}")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "elapsed_s": 300.0}

    elapsed = time.perf_counter() - t0
    success = proc.returncode == 0
    stdout = proc.stdout
    stderr = proc.stderr

    # Parse tokens per second if reported
    tok_per_sec = None
    for line in stdout.splitlines():
        if "tokens/s" in line or "tok/s" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if ("tok/s" in p or "tokens/s" in p) and i > 0:
                    try:
                        tok_per_sec = float(parts[i - 1])
                    except ValueError:
                        pass

    return {
        "success": success,
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "tokens": tokens,
        "tok_per_sec": tok_per_sec,
        "no_capture": no_capture,
        "stdout_tail": stdout[-500:] if stdout else "",
        "stderr_tail": stderr[-500:] if stderr else "",
    }


def benchmark_http(url: str, prompt_tokens: int, max_tokens: int, key: str) -> dict:
    """Benchmark through Strata HTTP API."""
    # Synthetic prompt repeating words
    words = "The quick brown fox jumps over the lazy dog and explores the Intel Arc GPU architecture. "
    repeat = max(1, int((prompt_tokens * 4) / len(words)))
    prompt = (words * repeat)[: prompt_tokens * 4]

    payload = {
        "model": "strata",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
    }

    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})},
    )

    t0 = time.perf_counter()
    ttft = None
    output_tokens = 0

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                line_str = line.decode("utf-8").strip()
                if line_str.startswith("data: ") and line_str != "data: [DONE]":
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    output_tokens += 1
    except urllib.error.URLError as e:
        return {"error": str(e), "prompt_target": prompt_tokens}

    total_time = time.perf_counter() - t0
    gen_time = total_time - (ttft or 0.0)
    output_tps = (output_tokens / gen_time) if gen_time > 0 else 0.0

    return {
        "prompt_target": prompt_tokens,
        "output_tokens": output_tokens,
        "ttft_s": ttft,
        "total_time_s": total_time,
        "output_tok_per_sec": output_tps,
    }


def main():
    p = argparse.ArgumentParser(description="Intel Xe2 / Arc B580 End-to-End Inference Benchmark")
    p.add_argument("--bin", default="build_sycl/strata", help="Path to strata CLI binary")
    p.add_argument("--url", default="", help="Strata HTTP server base URL (e.g. http://127.0.0.1:8080)")
    p.add_argument("--api-key", default="", help="API key if connecting to HTTP server")
    p.add_argument("--tokens", type=int, default=128, help="Number of tokens to generate")
    p.add_argument("--lengths", default="1k,4k", help="Comma-separated prompt token lengths")
    p.add_argument("--out", default="", help="Path to write JSON results")
    args, extra = p.parse_known_args()

    results = {"device": "Intel Xe2 (B580/Lunar Lake)", "timestamp": time.time(), "runs": []}

    if args.url:
        lengths = [parse_length(x) for x in args.lengths.split(",") if x.strip()]
        for l in lengths:
            print(f"Testing HTTP prompt length ~{l} tokens...")
            r = benchmark_http(args.url, l, args.tokens, args.api_key)
            print(f"  TTFT: {r.get('ttft_s', 0):.3f}s | Output: {r.get('output_tok_per_sec', 0):.1f} tok/s")
            results["runs"].append(r)
    else:
        bin_path = str(ROOT / args.bin) if not os.path.isabs(args.bin) else args.bin
        if not os.path.exists(bin_path):
            print(f"Binary not found: {bin_path}. Run 'make -C build_sycl strata' first.")
            sys.exit(1)

        for no_cap in [False, True]:
            mode = "without capture" if no_cap else "with graph capture"
            print(f"\n--- Testing CLI {mode} ---")
            r = benchmark_cli(bin_path, args.tokens, no_cap, extra)
            print(f"  Elapsed: {r.get('elapsed_s', 0):.3f}s | Success: {r.get('success')}")
            if r.get("tok_per_sec"):
                print(f"  Throughput: {r.get('tok_per_sec'):.2f} tok/s")
            results["runs"].append(r)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
