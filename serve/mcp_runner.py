"""serve/mcp_runner.py - Python execution environment for MCP Code Mode.

Runs in a child process (sys.executable) with non-elevated user rights. Communicates with Strata's McpHub
over stdin/stdout JSON-RPC pipes. Captures user prints and return values; intermediate data stays in memory.
"""
from __future__ import annotations

import ast
import io
import json
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serve.mcp import McpHub

ROOT = Path(__file__).resolve().parent.parent

COMMON_PACKAGE_MAP = {
    "yaml": "pyyaml",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "bs4": "beautifulsoup4",
    "fitz": "pymupdf",
    "Bio": "biopython",
    "serial": "pyserial",
    "usb": "pyusb",
    "dateutil": "python-dateutil",
}


def strip_markdown_fences(code: str) -> str:
    """Strips markdown code block wrappers (e.g. ```python ... ```) if present."""
    s = code.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return s


def extract_dependencies(code: str) -> list[str]:
    """Extract PEP 723 script dependencies or top-level imported third-party packages."""
    deps = set()
    clean = strip_markdown_fences(code)
    # 1. PEP 723 script metadata:
    m = re.search(r"#\s*///\s*script\s*\n(.*?)\n#\s*///", clean, re.DOTALL)
    if m:
        block = m.group(1)
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("#"):
                line = line.lstrip("#").strip()
            dm = re.search(r"dependencies\s*=\s*\[(.*?)\]", line)
            if dm:
                for item in dm.group(1).split(","):
                    item = item.strip().strip("\"'").strip()
                    if item:
                        deps.add(item)
    # 2. AST parsing for imported modules:
    try:
        tree = ast.parse(clean)
        stdlib = getattr(sys, "stdlib_module_names", set())
        local_builtins = {"mcp", "serve", "strata", "__main__", "builtins", "json", "math", "os", "sys"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root and root not in stdlib and root not in local_builtins:
                        pkg = COMMON_PACKAGE_MAP.get(root, root)
                        deps.add(pkg)
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                if root and root not in stdlib and root not in local_builtins:
                    pkg = COMMON_PACKAGE_MAP.get(root, root)
                    deps.add(pkg)
    except Exception:
        pass
    return sorted(deps)


def _kill_tree(p):
    """Terminate a process and its child processes."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=10)
        else:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        p.kill()
    except OSError:
        pass


# ------------------------------------------------------------------------------------------------ the runner process (child)
def _run_child():
    """Child process entry point (`python -m serve.mcp_runner`)."""
    from serve.sandbox import LLMSandbox

    ipc_in = sys.stdin.buffer
    ipc_out = sys.stdout.buffer

    schemas = {}
    caller = None
    mcp_root = None
    sandbox = LLMSandbox()

    class McpCaller:
        def __init__(self):
            self._id = 0

        def call(self, tool_name: str, **kwargs):
            self._id += 1
            req = {"type": "call", "id": self._id, "tool": tool_name, "arguments": kwargs}
            ipc_out.write(json.dumps(req, ensure_ascii=False).encode("utf-8") + b"\n")
            ipc_out.flush()
            resp_line = ipc_in.readline()
            if not resp_line:
                raise RuntimeError("MCP bridge disconnected")
            resp = json.loads(resp_line.decode("utf-8"))
            if not resp.get("ok"):
                raise RuntimeError(resp.get("error") or resp.get("text") or "tool call failed")
            val = resp.get("text", "")
            if isinstance(val, str):
                s = val.strip()
                if (s.startswith(("{", "[")) and s.endswith(("}", "]"))) or s in ("true", "false", "null") or s.lstrip("-").isdigit():
                    try:
                        return json.loads(s)
                    except Exception:
                        pass
            return val

    class ServerProxy:
        def __init__(self, caller: McpCaller, server_name: str):
            self._caller = caller
            self._server = server_name

        def __getattr__(self, name: str):
            def _invoke(*args, **kwargs):
                target = f"{self._server}__{name}"
                schema = schemas.get(target) or {}
                props = list((schema.get("parameters", {}).get("properties") or {}).keys())
                for i, arg in enumerate(args):
                    if i < len(props) and props[i] not in kwargs:
                        kwargs[props[i]] = arg
                return self._caller.call(target, **kwargs)
            return _invoke

    class McpRoot:
        def __init__(self, caller: McpCaller, tools_list: list[dict]):
            self._caller = caller
            self._tools = tools_list
            self.call = caller.call

        def __getattr__(self, server_name: str):
            return ServerProxy(self._caller, server_name)

        def search(self, query: str) -> list[dict]:
            q = query.lower()
            return [t for t in self._tools if q in t.get("name", "").lower() or q in (t.get("description") or "").lower()]

        def list_tools(self) -> list[dict]:
            return self._tools

    while True:
        line = ipc_in.readline()
        if not line:
            break
        try:
            req_data = json.loads(line.decode("utf-8"))
        except Exception as e:
            ipc_out.write(json.dumps({"type": "done", "ok": False, "error": f"invalid init payload: {e}"}).encode("utf-8") + b"\n")
            ipc_out.flush()
            continue

        if req_data.get("type") == "exit":
            break

        code = req_data.get("code") or ""
        tools = req_data.get("tools") or []
        session_id = req_data.get("session_id") or "default"

        if caller is None or tools:
            for t in tools:
                if isinstance(t, dict) and t.get("name"):
                    schemas[t["name"]] = t
            caller = caller or McpCaller()
            mcp_root = McpRoot(caller, tools)

        res_dict = sandbox.execute(code, session_id=session_id, custom_tools={"mcp": mcp_root})
        raw_res = res_dict.get("result")
        serializable_res = raw_res if isinstance(raw_res, (int, float, str, bool, list, dict)) else None
        ipc_out.write(json.dumps({
            "type": "done",
            "ok": res_dict["ok"],
            "output": res_dict["output"],
            "result": serializable_res,
            "error": res_dict.get("error")
        }, ensure_ascii=False).encode("utf-8") + b"\n")
        ipc_out.flush()


# ------------------------------------------------------------------------------------------------ the host caller (parent)
class RunnerHandle:
    def __init__(self, proc, q, reader_thread, stop_reader, clean_deps):
        self.proc = proc
        self.q = q
        self.reader_thread = reader_thread
        self.stop_reader = stop_reader
        self.deps = set(clean_deps)
        self.last_used = time.monotonic()
        self.lock = threading.Lock()

    def alive(self) -> bool:
        return self.proc.poll() is None

    def close(self):
        self.stop_reader.set()
        try:
            self.proc.stdin.write(json.dumps({"type": "exit"}).encode("utf-8") + b"\n")
            self.proc.stdin.flush()
        except Exception:
            pass
        for f in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                if f:
                    f.close()
            except Exception:
                pass
        _kill_tree(self.proc)


_RUNNER_POOL: dict[str, RunnerHandle] = {}
_RUNNER_LOCK = threading.Lock()


def cleanup_idle_runners(ttl_s: float = 900.0):
    now = time.monotonic()
    with _RUNNER_LOCK:
        expired = [sid for sid, r in _RUNNER_POOL.items() if (now - r.last_used) > ttl_s or not r.alive()]
        for sid in expired:
            r = _RUNNER_POOL.pop(sid, None)
            if r:
                r.close()


def close_all_runners():
    with _RUNNER_LOCK:
        for r in _RUNNER_POOL.values():
            r.close()
        _RUNNER_POOL.clear()


def execute_in_runner(code: str, hub: McpHub, timeout_s: float, cancel: threading.Event | None = None,
                      dependencies: list[str] | None = None, session_id: str = "default") -> dict:
    """Run `code` in a child Python runner process, routing MCP calls back through `hub`.

    Supports stateful persistence across turns for the same `session_id`.
    If `uv` is available, uses `uv run -q` for fast, isolated execution and supports on-the-fly PyPI dependencies.
    Falls back to `sys.executable` if `uv` is not present.
    -> {"ok": bool, "text": str, "ms": int}
    """
    from serve import lifeline
    from serve.mcp import McpCancelled

    t0 = time.monotonic()
    tools_info = hub.template_tools()

    clean_code = strip_markdown_fences(code)
    all_deps = list(dependencies or [])
    for d in extract_dependencies(clean_code):
        if d not in all_deps:
            all_deps.append(d)
    clean_deps = [str(d).strip() for d in all_deps if str(d).strip()]

    cleanup_idle_runners()

    with _RUNNER_LOCK:
        existing = _RUNNER_POOL.get(session_id)
        if existing and existing.alive() and set(clean_deps).issubset(existing.deps):
            runner = existing
        else:
            if existing:
                existing.close()
                _RUNNER_POOL.pop(session_id, None)

            extra = {}
            if os.name == "nt":
                extra["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            else:
                extra["start_new_session"] = True

            uv_bin = shutil.which("uv")
            if uv_bin:
                cmd = [uv_bin, "run", "-q"]
                if clean_deps:
                    cmd += ["--with", ",".join(clean_deps)]
                cmd += ["python", "-m", "serve.mcp_runner"]
            else:
                cmd = [sys.executable, "-m", "serve.mcp_runner"]

            try:
                proc = lifeline.popen(
                    cmd,
                    "mcp",
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(ROOT),
                    env=dict(os.environ),
                    **extra
                )
            except OSError as e:
                return {"ok": False, "text": f"error: could not start Python runner: {e}", "ms": int((time.monotonic() - t0) * 1000)}

            q = queue.Queue()
            stop_reader = threading.Event()

            def reader():
                for line in proc.stdout:
                    q.put(line)
                    if stop_reader.is_set():
                        break
                q.put(None)

            reader_thread = threading.Thread(target=reader, daemon=True)
            reader_thread.start()

            runner = RunnerHandle(proc, q, reader_thread, stop_reader, clean_deps)
            _RUNNER_POOL[session_id] = runner

    with runner.lock:
        runner.last_used = time.monotonic()
        proc = runner.proc
        q = runner.q

        req_payload = json.dumps({
            "type": "exec",
            "code": clean_code,
            "tools": tools_info,
            "session_id": session_id,
        }, ensure_ascii=False).encode("utf-8") + b"\n"

        try:
            proc.stdin.write(req_payload)
            proc.stdin.flush()
        except (OSError, ValueError):
            with _RUNNER_LOCK:
                _RUNNER_POOL.pop(session_id, None)
            runner.close()
            return {"ok": False, "text": "error: runner process closed unexpectedly", "ms": int((time.monotonic() - t0) * 1000)}

        deadline = time.monotonic() + timeout_s
        done = False
        output = None

        try:
            while time.monotonic() < deadline and not done:
                if cancel is not None and cancel.is_set():
                    with _RUNNER_LOCK:
                        _RUNNER_POOL.pop(session_id, None)
                    runner.close()
                    raise McpCancelled("stopped")

                try:
                    line = q.get(timeout=0.05)
                except queue.Empty:
                    continue

                if line is None:
                    break

                try:
                    msg = json.loads(line.decode("utf-8"))
                except Exception:
                    continue

                if msg.get("type") == "call":
                    tool_name = msg.get("tool")
                    tool_args = msg.get("arguments") or {}
                    try:
                        r = hub.call(tool_name, tool_args, cancel)
                        ans = {"ok": r.get("ok", False), "text": r.get("text", "")}
                    except McpCancelled:
                        with _RUNNER_LOCK:
                            _RUNNER_POOL.pop(session_id, None)
                        runner.close()
                        raise
                    except Exception as e:
                        ans = {"ok": False, "error": str(e)}

                    try:
                        proc.stdin.write(json.dumps(ans, ensure_ascii=False).encode("utf-8") + b"\n")
                        proc.stdin.flush()
                    except (OSError, ValueError):
                        break
                elif msg.get("type") == "done":
                    done = True
                    output = msg
                    break
        except Exception:
            with _RUNNER_LOCK:
                _RUNNER_POOL.pop(session_id, None)
            runner.close()
            raise

    ms = int((time.monotonic() - t0) * 1000)
    if not done:
        with _RUNNER_LOCK:
            _RUNNER_POOL.pop(session_id, None)
        runner.close()
        if cancel is not None and cancel.is_set():
            raise McpCancelled("stopped")
        return {"ok": False, "text": f"error: Code Mode Python execution timed out after {timeout_s:g} s", "ms": ms}

    if output.get("ok"):
        return {"ok": True, "text": output.get("output", ""), "ms": ms, "result": output.get("result")}
    else:
        err = output.get("error", "unknown error")
        return {"ok": False, "text": f"error: {err}", "ms": ms}


if __name__ == "__main__":
    _run_child()
