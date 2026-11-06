"""serve/js_sandbox.py - Embedded QuickJS Execution Sandbox for MCP Code Mode.

Provides a fast, memory-safe, hermetically sandboxed JavaScript (ES6) execution environment:
- Powered by QuickJS (zero host filesystem, network, or OS access by default).
- Pre-injected `mcp.<server>.<tool>(args)` transparent proxy for invoking MCP tools.
- In-memory console logging capture (`console.log`, `console.error`, `console.warn`).
- Automatic expression evaluation and return value serialization (Jupyter/REPL style).
- State and variable persistence across consecutive turns in the same conversation session.
- Sub-millisecond execution overhead and microsecond memory cleanup.
"""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

try:
    import quickjs
    HAS_QUICKJS = True
except ImportError:
    quickjs = None
    HAS_QUICKJS = False

DEFAULT_SESSION_TTL_S = 900.0  # 15 minutes
MAX_SESSIONS = 100


def strip_markdown_fences(code: str) -> str:
    """Strips markdown code block wrappers (e.g. ```javascript ... ```) or XML prompt tags."""
    s = code.strip()
    s = re.sub(r"^\s*<tool_call>\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*</tool_call>\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*<function=[^>]+>\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*</function>\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*<parameter=[^>]+>\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*</parameter>\s*$", "", s, flags=re.IGNORECASE)
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return s


def inject_js_loop_guards(code: str, max_iters: int = 500_000) -> str:
    """Injects iteration limit guards into JavaScript loops to prevent runaway 30s timeouts."""
    guard = f" if (++globalThis._st_loop > {max_iters}) throw new Error('Loop iteration limit exceeded in Code Mode ({max_iters:,} iterations)'); "
    # Handle while (...) { and for (...) {
    guarded = re.sub(r"\b(while\s*\([^)]*\)\s*\{)", r"\1" + guard, code)
    guarded = re.sub(r"\b(for\s*\([^)]*\)\s*\{)", r"\1" + guard, guarded)
    return "globalThis._st_loop = 0;\n" + guarded


class QuickJSSandbox:
    """Stateful JavaScript sandbox managing isolated QuickJS contexts per session."""

    def __init__(self, ttl_s: float = DEFAULT_SESSION_TTL_S, max_sessions: int = MAX_SESSIONS):
        self.ttl_s = ttl_s
        self.max_sessions = max_sessions
        self.lock = threading.Lock()
        # session_id -> (last_access_timestamp, context)
        self._sessions: dict[str, tuple[float, Any]] = {}

    def _cleanup_expired(self, now: float) -> None:
        """Removes sessions that have exceeded the inactivity TTL."""
        expired = [sid for sid, (ts, _) in self._sessions.items() if (now - ts) > self.ttl_s]
        for sid in expired:
            self._sessions.pop(sid, None)

        if len(self._sessions) > self.max_sessions:
            sorted_sessions = sorted(self._sessions.items(), key=lambda kv: kv[1][0])
            to_remove = len(self._sessions) - self.max_sessions
            for sid, _ in sorted_sessions[:to_remove]:
                self._sessions.pop(sid, None)

    def reset_session(self, session_id: str | None = None) -> None:
        with self.lock:
            if session_id is None:
                self._sessions.clear()
            else:
                self._sessions.pop(session_id, None)

    def _init_context(self, hub: Any) -> Any:
        if not HAS_QUICKJS:
            raise RuntimeError(
                "QuickJS is not installed. To use JavaScript Code Mode, install quickjs with: "
                "python -m pip install quickjs (or uv pip install quickjs)"
            )

        cx = quickjs.Context()
        # 32 MB heap limit
        cx.set_memory_limit(32 * 1024 * 1024)
        # 1 MB max call stack
        cx.set_max_stack_size(1024 * 1024)

        def host_mcp_call(tool_name: str, args_json: str) -> str:
            if hub is None:
                return json.dumps({"ok": False, "error": "MCP hub unavailable"})
            try:
                args = json.loads(args_json) if args_json else {}
            except Exception as e:
                return json.dumps({"ok": False, "error": f"Invalid JSON arguments: {e}"})

            try:
                res = hub.call(str(tool_name), args)
                return json.dumps(res, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

        cx.add_callable("__mcp_call", host_mcp_call)

        prelude = """
        globalThis.__st_logs = [];
        const console = {
            log: (...args) => globalThis.__st_logs.push(args.map(String).join(' ')),
            error: (...args) => globalThis.__st_logs.push('[error] ' + args.map(String).join(' ')),
            warn: (...args) => globalThis.__st_logs.push('[warn] ' + args.map(String).join(' ')),
            info: (...args) => globalThis.__st_logs.push(args.map(String).join(' '))
        };

        function render_interactive_plot(fig) {
            if (fig && typeof fig === 'object') {
                const payload = {
                    type: 'application/vnd.plotly.v1+json',
                    content: fig.data || fig.layout ? fig : { data: fig.data || [], layout: fig.layout || {} }
                };
                globalThis._last_plot = payload;
                return payload;
            }
            throw new TypeError("render_interactive_plot() requires a figure object (with data/layout)");
        }

        const mcp = new Proxy({}, {
            get(target, server) {
                if (server === 'call') {
                    return function(toolName, args = {}) {
                        const resStr = __mcp_call(String(toolName), JSON.stringify(args || {}));
                        try { return JSON.parse(resStr); } catch (e) { return resStr; }
                    };
                }
                return new Proxy({}, {
                    get(target, tool) {
                        return function(args = {}) {
                            const targetName = String(server) + '__' + String(tool);
                            const resStr = __mcp_call(targetName, JSON.stringify(args || {}));
                            try { return JSON.parse(resStr); } catch (e) { return resStr; }
                        };
                    }
                });
            }
        });
        """
        cx.eval(prelude)
        return cx

    def execute(
        self,
        code: str,
        hub: Any,
        timeout_s: float = 60.0,
        cancel: threading.Event | None = None,
        session_id: str = "default",
    ) -> dict[str, Any]:
        """Executes JavaScript in the session's QuickJS context and auto-captures results."""
        t0 = time.monotonic()
        clean_code = strip_markdown_fences(code)
        if not clean_code.strip():
            return {
                "ok": True,
                "text": "(completed with no output)",
                "result": None,
                "plot": None,
                "ms": 0,
            }

        guarded_code = inject_js_loop_guards(clean_code)

        now = time.monotonic()
        with self.lock:
            self._cleanup_expired(now)
            if session_id in self._sessions:
                _, cx = self._sessions[session_id]
                self._sessions[session_id] = (now, cx)
            else:
                try:
                    cx = self._init_context(hub)
                except Exception as e:
                    return {
                        "ok": False,
                        "text": f"error: could not initialize JavaScript engine: {e}",
                        "ms": int((time.monotonic() - t0) * 1000),
                    }
                self._sessions[session_id] = (now, cx)

        # Clear turn logs & previous plot
        try:
            cx.eval("globalThis.__st_logs = []; globalThis._last_plot = null;")
        except Exception:
            pass

        if cancel is not None and cancel.is_set():
            return {
                "ok": False,
                "text": "error: execution stopped",
                "ms": int((time.monotonic() - t0) * 1000),
            }

        raw_res = None
        plot = None
        ok = True
        err_msg = None

        try:
            raw_res = cx.eval(guarded_code)
        except Exception as exc:
            ok = False
            exc_str = str(exc)
            if "Loop iteration limit exceeded" in exc_str:
                err_msg = f"Code Mode JavaScript execution timed out: {exc_str}"
            else:
                err_msg = exc_str

        ms = int((time.monotonic() - t0) * 1000)

        if not ok:
            return {"ok": False, "text": f"error: {err_msg}", "ms": ms}

        # Convert result if it's a QuickJS object or array
        serializable_res = raw_res
        if raw_res is not None and not isinstance(raw_res, (int, float, str, bool)):
            try:
                cx.set("_st_temp_res", raw_res)
                json_str = cx.eval("JSON.stringify(_st_temp_res)")
                cx.eval("delete globalThis._st_temp_res;")
                if json_str:
                    serializable_res = json.loads(json_str)
                else:
                    serializable_res = str(raw_res)
            except Exception:
                serializable_res = str(raw_res)

        # Check if render_interactive_plot was called
        try:
            plot_json = cx.eval("globalThis._last_plot ? JSON.stringify(globalThis._last_plot) : null")
            if plot_json:
                plot = json.loads(plot_json)
        except Exception:
            pass

        # Retrieve console logs
        logs = []
        try:
            logs_json = cx.eval("JSON.stringify(globalThis.__st_logs)")
            if logs_json:
                logs = json.loads(logs_json)
        except Exception:
            pass

        stdout_val = "\n".join(logs).strip()

        def _fmt(v: Any) -> str:
            if isinstance(v, dict) and v.get("type") == "application/vnd.plotly.v1+json":
                return json.dumps(v, ensure_ascii=False)
            if isinstance(v, (dict, list)):
                try:
                    return json.dumps(v, indent=2, ensure_ascii=False)
                except Exception:
                    return str(v)
            return str(v)

        if stdout_val and serializable_res is not None:
            text = f"{stdout_val}\n\nResult: {_fmt(serializable_res)}"
        elif stdout_val:
            text = stdout_val
        elif serializable_res is not None:
            text = _fmt(serializable_res)
        elif plot is not None:
            text = json.dumps(plot, ensure_ascii=False)
        else:
            text = "(completed with no output)"

        out: dict[str, Any] = {
            "ok": True,
            "text": text,
            "ms": ms,
            "result": serializable_res,
        }
        if plot:
            out["plot"] = plot
        return out


_GLOBAL_JS_SANDBOX: QuickJSSandbox | None = None
_GLOBAL_JS_LOCK = threading.Lock()


def get_js_sandbox() -> QuickJSSandbox:
    global _GLOBAL_JS_SANDBOX
    with _GLOBAL_JS_LOCK:
        if _GLOBAL_JS_SANDBOX is None:
            _GLOBAL_JS_SANDBOX = QuickJSSandbox()
        return _GLOBAL_JS_SANDBOX


def execute_js_in_sandbox(
    code: str,
    hub: Any,
    timeout_s: float = 60.0,
    cancel: threading.Event | None = None,
    session_id: str = "default",
) -> dict[str, Any]:
    """Top-level entry point to execute JavaScript Code Mode."""
    sandbox = get_js_sandbox()
    return sandbox.execute(code, hub, timeout_s=timeout_s, cancel=cancel, session_id=session_id)
