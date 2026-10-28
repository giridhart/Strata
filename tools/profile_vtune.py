#!/usr/bin/env python3
"""tools/profile_vtune.py - Profile Strata Inference using Intel VTune Profiler.

Profiles Strata's execution on Intel Arc Xe2 GPU and CPU when generating responses
to a specific prompt (default: "write a cpp program to compute Nth fibonacci number").

Usage:
    # 1. Profile via Strata server (attaches to active inference or starts local instance)
    python3 tools/profile_vtune.py --analysis gpu-hotspots

    # 2. Profile a server requiring an API key
    python3 tools/profile_vtune.py --api-key "my-strata-key"
    # Or via environment variable:
    STRATA_API_KEY="my-strata-key" python3 tools/profile_vtune.py

    # 3. Profile with custom prompt or generation parameters
    python3 tools/profile_vtune.py --prompt "write a cpp program to compute Nth fibonacci number" --max-tokens 512

    # 4. Profile CPU hotspots or XPU offload analysis
    python3 tools/profile_vtune.py --analysis hotspots
    python3 tools/profile_vtune.py --analysis xpu-offload
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def find_vtune() -> str:
    """Locate Intel VTune binary."""
    v = shutil.which("vtune")
    if v:
        return v
    for candidate in [
        "/opt/intel/oneapi/vtune/latest/bin64/vtune",
        "/opt/intel/oneapi/vtune/2026.6/bin64/vtune",
    ]:
        if Path(candidate).is_file():
            return candidate
    return ""


def get_environment() -> dict[str, str]:
    """Inherit environment and ensure Intel oneAPI / Level Zero paths are populated."""
    env = os.environ.copy()
    setvars = Path("/opt/intel/oneapi/setvars.sh")
    if setvars.is_file() and "ONEAPI_ROOT" not in env:
        cmd = f"source {setvars} --force 2>/dev/null && env"
        proc = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, text=True)
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    env[k] = v
    env.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero:gpu")
    env.setdefault("ZE_ENABLE_TRACING", "1")
    env.setdefault("STRATA_VERIFY_GRAPH", "0")
    return env


def is_server_ready(url: str, api_key: str = "") -> tuple[bool, str]:
    """Check if the Strata HTTP server is running and ready. Returns (is_ready, error_msg)."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        req = urllib.request.Request(f"{url.rstrip('/')}/v1/models", headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200, ""
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return False, "401 Unauthorized: API key missing or invalid."
        return False, f"HTTP Error {e.code}: {e.reason}"
    except Exception as e:
        return False, str(e)


def find_target_pid() -> int | None:
    """Find running Strata engine or server process PID."""
    patterns = ["build_sycl/strata", "strata.py serve", "serve/server.py"]
    for p in patterns:
        try:
            out = subprocess.check_output(["pgrep", "-f", p], text=True).strip()
            if out:
                pids = [int(x) for x in out.splitlines() if x.strip().isdigit()]
                if pids:
                    return pids[-1]
        except Exception:
            continue
    return None


def send_prompt(url: str, prompt: str, max_tokens: int, temperature: float, api_key: str = "") -> str:
    """Send prompt request to Strata /v1/chat/completions with streaming."""
    endpoint = f"{url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": "strata",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    req = urllib.request.Request(endpoint, data=data, headers=headers)
    
    print(f"\n[Prompt Request] Sending prompt: \"{prompt}\"")
    print(f"[Streaming Output from {url}]:\n" + "-" * 40)
    full_output = []
    t0 = time.perf_counter()
    
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line.startswith("data: "):
                    chunk_str = line[len("data: "):].strip()
                    if chunk_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(chunk_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            sys.stdout.write(content)
                            sys.stdout.flush()
                            full_output.append(content)
                    except Exception:
                        pass
    except urllib.error.HTTPError as e:
        if e.code == 401:
            sys.exit(f"\n[-] Authentication Error: Server returned 401 Unauthorized. Pass correct key via --api-key or STRATA_API_KEY.")
        raise

    elapsed = time.perf_counter() - t0
    print("\n" + "-" * 40)
    print(f"[Timing] Completed in {elapsed:.2f} s ({len(''.join(full_output))} chars generated).\n")
    return "".join(full_output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        default="write a cpp program to compute Nth fibonacci number",
        help="Inference prompt text (default: 'write a cpp program to compute Nth fibonacci number')",
    )
    parser.add_argument(
        "-k", "--api-key",
        default=os.environ.get("STRATA_API_KEY", ""),
        help="API key for Strata server authentication (default: reads $STRATA_API_KEY)",
    )
    parser.add_argument(
        "--analysis",
        choices=["gpu-hotspots", "hotspots", "xpu-offload", "performance-snapshot"],
        default="gpu-hotspots",
        help="Intel VTune analysis type (default: gpu-hotspots)",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="Strata server URL")
    parser.add_argument("--port", type=int, default=8080, help="Port to start server on if not running")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--out-dir", help="VTune output directory path")
    args = parser.parse_args()

    vtune = find_vtune()
    if not vtune:
        sys.exit("Error: 'vtune' binary not found. Please install Intel oneAPI Base Toolkit.")

    env = get_environment()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or (ROOT / "vtune_profiles" / f"strata_fibonacci_{args.analysis}_{timestamp}"))
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    print("=======================================================")
    print("      Strata Intel VTune Profiling Runner              ")
    print("=======================================================")
    print(f"VTune Binary     : {vtune}")
    print(f"Analysis Type    : {args.analysis}")
    print(f"Target Prompt    : \"{args.prompt}\"")
    print(f"Server URL       : {args.url}")
    if args.api_key:
        masked = args.api_key[:4] + "..." + args.api_key[-4:] if len(args.api_key) > 8 else "***"
        print(f"API Key Auth     : Enabled ({masked})")
    else:
        print("API Key Auth     : None (open access)")
    print(f"Result Directory : {out_dir}")
    print("=======================================================")

    # Check if server is already running
    server_proc = None
    ready, err = is_server_ready(args.url, args.api_key)
    if not ready:
        if "401" in err:
            sys.exit(f"Error: Server at {args.url} rejected credentials ({err}). Check your --api-key.")
        print(f"[Strata] No active server found at {args.url}. Starting local instance...")
        server_cmd = [sys.executable, str(ROOT / "strata.py"), "serve", "--port", str(args.port)]
        if args.api_key:
            server_cmd.extend(["--api-key", args.api_key])
        server_proc = subprocess.Popen(server_cmd, cwd=ROOT, env=env)
        
        # Wait for server ready
        for _ in range(60):
            ready, _ = is_server_ready(args.url, args.api_key)
            if ready:
                break
            time.sleep(1)
        else:
            if server_proc:
                server_proc.kill()
            sys.exit(f"Error: Strata server failed to become ready at {args.url} within 60s.")

    try:
        target_pid = find_target_pid()
        if target_pid:
            print(f"[VTune] Attaching VTune to Strata Engine PID: {target_pid}...")
            vtune_cmd = [
                vtune,
                "-collect", args.analysis,
                "-result-dir", str(out_dir),
                "-target-pid", str(target_pid),
            ]
            vtune_proc = subprocess.Popen(vtune_cmd, env=env)
            time.sleep(2)  # Allow VTune collector to initialize hooks

            # Send prompt
            send_prompt(args.url, args.prompt, args.max_tokens, args.temperature, args.api_key)

            # Stop VTune
            print("[VTune] Stopping VTune data collection...")
            stop_cmd = [vtune, "-r", str(out_dir), "-command", "stop"]
            subprocess.run(stop_cmd, env=env, check=False)
            vtune_proc.wait(timeout=30)
        else:
            print("[VTune] Running prompt client under VTune wrapper...")
            runner = [
                sys.executable, "-c",
                f"import sys; sys.path.insert(0, '{ROOT}/tools'); "
                f"import profile_vtune; profile_vtune.send_prompt('{args.url}', '{args.prompt}', {args.max_tokens}, {args.temperature}, '{args.api_key}')"
            ]
            vtune_cmd = [vtune, "-collect", args.analysis, "-result-dir", str(out_dir), "--", *runner]
            subprocess.run(vtune_cmd, env=env, check=False)

        print("\n=======================================================")
        print(f"[Success] VTune profile generated at:\n  {out_dir}")
        print("To view interactive report:")
        print(f"  vtune-gui {out_dir}")
        print("=======================================================")

    finally:
        if server_proc:
            print("[Cleanup] Stopping local Strata server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
