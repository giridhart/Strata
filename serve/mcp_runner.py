"""serve/mcp_runner.py - Python execution environment for MCP Code Mode.

Runs in a child process (sys.executable) with non-elevated user rights. Communicates with Strata's McpHub
over stdin/stdout JSON-RPC pipes. Captures user prints and return values; intermediate data stays in memory.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import queue
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
    ipc_in = sys.stdin.buffer
    ipc_out = sys.stdout.buffer

    line = ipc_in.readline()
    if not line:
        return
    try:
        init_data = json.loads(line.decode("utf-8"))
    except Exception as e:
        ipc_out.write(json.dumps({"type": "done", "ok": False, "error": f"invalid init payload: {e}"}).encode("utf-8") + b"\n")
        ipc_out.flush()
        return

    code = init_data.get("code") or ""
    tools = init_data.get("tools") or []

    # Map of namespaced tool name -> schema for positional argument mapping
    schemas = {t.get("name"): t for t in tools if isinstance(t, dict) and t.get("name")}

    # Redirect user stdout / stderr so print() writes to in-memory buffers
    user_stdout = io.StringIO()
    user_stderr = io.StringIO()
    sys.stdout = user_stdout
    sys.stderr = user_stderr

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

    caller = McpCaller()
    user_env = {
        "__name__": "__main__",
        "mcp": McpRoot(caller, tools),
        "json": json,
    }

    try:
        compiled = compile(code, "<codemode>", "exec")
        exec(compiled, user_env)
        ret = user_env.get("result")
        if ret is None and callable(user_env.get("main")):
            ret = user_env["main"]()
        stdout_val = user_stdout.getvalue().strip()
        stderr_val = user_stderr.getvalue().strip()

        if stdout_val and ret is not None:
            output_text = f"{stdout_val}\n\nResult: {ret}"
        elif stdout_val:
            output_text = stdout_val
        elif ret is not None:
            output_text = json.dumps(ret, indent=2, ensure_ascii=False) if isinstance(ret, (dict, list)) else str(ret)
        elif stderr_val:
            output_text = stderr_val
        else:
            output_text = "(completed with no output)"

        res = {"type": "done", "ok": True, "output": output_text}
    except SystemExit as e:
        stdout_val = user_stdout.getvalue().strip()
        if e.code == 0 or e.code is None:
            res = {"type": "done", "ok": True, "output": stdout_val or "(exited 0)"}
        else:
            res = {"type": "done", "ok": False, "error": f"sys.exit({e.code})\n{user_stderr.getvalue()}"}
    except Exception:
        res = {"type": "done", "ok": False, "error": traceback.format_exc()}

    ipc_out.write(json.dumps(res, ensure_ascii=False).encode("utf-8") + b"\n")
    ipc_out.flush()


# ------------------------------------------------------------------------------------------------ the host caller (parent)
def execute_in_runner(code: str, hub: McpHub, timeout_s: float, cancel: threading.Event | None = None,
                      dependencies: list[str] | None = None) -> dict:
    """Run `code` in a child Python runner process, routing MCP calls back through `hub`.

    If `uv` is available, uses `uv run -q` for fast, isolated execution and supports on-the-fly PyPI dependencies.
    Falls back to `sys.executable` if `uv` is not present.
    -> {"ok": bool, "text": str, "ms": int}
    """
    from serve import lifeline
    from serve.mcp import McpCancelled

    t0 = time.monotonic()
    tools_info = hub.template_tools()

    extra = {}
    if os.name == "nt":
        extra["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        extra["start_new_session"] = True

    uv_bin = shutil.which("uv")
    clean_deps = [str(d).strip() for d in (dependencies or []) if str(d).strip()]
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

    init_payload = json.dumps({"code": code, "tools": tools_info}, ensure_ascii=False).encode("utf-8") + b"\n"
    try:
        proc.stdin.write(init_payload)
        proc.stdin.flush()
    except (OSError, ValueError):
        _kill_tree(proc)
        return {"ok": False, "text": "error: runner process closed unexpectedly", "ms": int((time.monotonic() - t0) * 1000)}

    deadline = time.monotonic() + timeout_s
    done = False
    output = None

    try:
        while time.monotonic() < deadline and not done:
            if cancel is not None and cancel.is_set():
                _kill_tree(proc)
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
                    _kill_tree(proc)
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
    finally:
        stop_reader.set()
        for f in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if f:
                    f.close()
            except Exception:
                pass
        if not done or proc.poll() is None:
            _kill_tree(proc)

    ms = int((time.monotonic() - t0) * 1000)
    if not done:
        if cancel is not None and cancel.is_set():
            raise McpCancelled("stopped")
        return {"ok": False, "text": f"error: Code Mode Python execution timed out after {timeout_s:g} s", "ms": ms}

    if output.get("ok"):
        return {"ok": True, "text": output.get("output", ""), "ms": ms}
    else:
        err = output.get("error", "unknown error")
        return {"ok": False, "text": f"error: {err}", "ms": ms}


if __name__ == "__main__":
    _run_child()
