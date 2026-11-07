"""serve/sandbox.py - Zero-Verbosity AST Execution Sandbox for MCP Code Mode.

Provides a stateful Python execution environment with:
- Pre-imported mathematics, physics, logic, and developer libraries (SymPy, NumPy, Pint, Z3, Math).
- Pre-injected symbolic, dimensional, and constraint tools for direct invocation without token overhead.
- AST splitting: Auto-evaluates and returns the final statement if it is an expression (Jupyter/REPL style).
- State persistence across consecutive turns in the same conversation session (with LRU/TTL cleanup).
- Comprehensive error formatting, timeout awareness, and output encapsulation.
"""
from __future__ import annotations

import ast
import builtins
import cmath
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = Path(os.environ.get("STRATA_WORKSPACE") or (ROOT / "workspace")).resolve()

# Import domain tools for pre-injection into the sandbox namespace
try:
    from tools.mcp_symbolic import (
        compile_equation_to_code,
        dimensional_analysis,
        solve_and_derive,
        verify_equivalence,
    )
except ImportError:
    compile_equation_to_code = None
    dimensional_analysis = None
    solve_and_derive = None
    verify_equivalence = None

try:
    from tools.mcp_logic import solve_constraints
except ImportError:
    solve_constraints = None

try:
    from tools.mcp_devtools import calculate_time_delta, lint_python_code
except ImportError:
    calculate_time_delta = None
    lint_python_code = None


DEFAULT_SESSION_TTL_S = 900.0  # 15 minutes
MAX_SESSIONS = 100


def strip_markdown_fences(code: str) -> str:
    """Strips markdown code block wrappers (e.g. ```python ... ```) or stray XML tags if present."""
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


def attempt_syntax_repair(code: str) -> str:
    """Attempts lightweight auto-repair for common LLM syntax slips like unclosed brackets/parens
    and C/JS-style line comments (//) in Python Code Mode."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    inv = {v: k for k, v in pairs.items()}

    repaired = code
    if "//" in repaired:
        lines = []
        for line in repaired.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("//"):
                indent = line[:len(line) - len(stripped)]
                lines.append(indent + "#" + stripped[2:])
            else:
                lines.append(line)
        repaired = "\n".join(lines)

    for _ in range(5):
        try:
            ast.parse(repaired)
            return repaired
        except SyntaxError as e:
            lines = repaired.splitlines()
            if not lines:
                return code

            target_idx = (e.lineno - 1) if (e.lineno and 1 <= e.lineno <= len(lines)) else (len(lines) - 1)
            target_line = lines[target_idx]

            stack = []
            in_q = None
            esc = False
            for ch in target_line:
                if in_q:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == in_q:
                        in_q = None
                else:
                    if ch in ('"', "'"):
                        in_q = ch
                    elif ch in pairs:
                        stack.append(pairs[ch])
                    elif ch in inv:
                        if stack and stack[-1] == ch:
                            stack.pop()

            patch = (in_q or "") + "".join(reversed(stack))
            if patch:
                lines[target_idx] = target_line + patch
                repaired = "\n".join(lines)
                continue

            # Fallback: check unclosed delimiters across the entire code buffer
            all_stack = []
            in_q_all = None
            esc_all = False
            for ch in repaired:
                if in_q_all:
                    if esc_all:
                        esc_all = False
                    elif ch == "\\":
                        esc_all = True
                    elif ch == in_q_all:
                        in_q_all = None
                else:
                    if ch in ('"', "'"):
                        in_q_all = ch
                    elif ch in pairs:
                        all_stack.append(pairs[ch])
                    elif ch in inv:
                        if all_stack and all_stack[-1] == ch:
                            all_stack.pop()

            patch_all = (in_q_all or "") + "".join(reversed(all_stack))
            if patch_all:
                lines[-1] = lines[-1] + patch_all
                repaired = "\n".join(lines)
                continue
            break

    try:
        ast.parse(repaired)
        return repaired
    except Exception:
        return code


BANNED_MODULES = {
    "os", "sys", "subprocess", "socket", "urllib", "requests", "http",
    "shutil", "ctypes", "pathlib", "pty", "platform", "psutil", "winreg",
    "posix", "importlib", "builtins", "__builtin__"
}
BANNED_ATTRIBUTES = {
    "__subclasses__", "__globals__", "__code__", "__bases__", "__builtins__",
    "__import__", "__loader__", "__spec__"
}


class SecurityVisitor(ast.NodeVisitor):
    """Enforces AST-level sandboxing (Layer 2): blocks host inspection, subprocesses, and sandbox escapes."""

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in BANNED_MODULES:
                raise PermissionError(
                    f"Direct access to module '{root}' is restricted in Code Mode sandbox for host protection. "
                    f"Use mcp tools for host or external operations."
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            root = node.module.split(".")[0]
            if root in BANNED_MODULES:
                raise PermissionError(
                    f"Direct import from module '{root}' is restricted in Code Mode sandbox for host protection."
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr in BANNED_ATTRIBUTES:
            raise PermissionError(
                f"Access to attribute '{node.attr}' is restricted in Code Mode sandbox."
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id in ("__import__", "exec", "eval", "compile"):
                raise PermissionError(
                    f"Direct invocation of '{node.func.id}()' is restricted in Code Mode sandbox."
                )
        self.generic_visit(node)


class LoopBoundingTransformer(ast.NodeTransformer):
    """Injects iteration counters into while and for loops to prevent runaway execution / 30s timeouts."""

    def __init__(self, max_iters: int = 500_000):
        self.max_iters = max_iters

    def _guard_stmts(self) -> list[ast.stmt]:
        return [ast.Expr(value=ast.Call(
            func=ast.Name(id="_st_loop_tick", ctx=ast.Load()),
            args=[],
            keywords=[]
        ))]

    def visit_While(self, node: ast.While) -> ast.While:
        self.generic_visit(node)
        node.body = self._guard_stmts() + node.body
        return node

    def visit_For(self, node: ast.For) -> ast.For:
        self.generic_visit(node)
        node.body = self._guard_stmts() + node.body
        return node


def render_interactive_plot(fig: Any) -> dict[str, Any]:
    """Accepts a Plotly Figure object and serializes it to an interactive JSON schema.

    Args:
        fig: A Plotly Figure object (from plotly.express or plotly.graph_objects).

    Returns:
        dict: {"type": "application/vnd.plotly.v1+json", "content": json.loads(fig.to_json())}

    Raises:
        ValueError: If fig is None or cannot be parsed.
        TypeError: If fig is not a Plotly Figure.
    """
    if fig is None:
        raise ValueError("render_interactive_plot() requires a Plotly Figure, but received None.")

    is_plotly = False
    try:
        import plotly.graph_objects as go
        if isinstance(fig, go.Figure):
            is_plotly = True
    except ImportError:
        pass

    if not is_plotly:
        # Duck-typing: must have to_json, layout, and data (to reject DataFrames or arbitrary objects)
        if hasattr(fig, "data") and hasattr(fig, "layout") and callable(getattr(fig, "to_json", None)):
            is_plotly = True

    if is_plotly:
        try:
            raw_json = fig.to_json()
            content = json.loads(raw_json)
        except Exception as exc:
            raise ValueError(f"Failed to serialize Plotly figure to JSON: {exc}") from exc
    elif isinstance(fig, dict) and ("data" in fig or "layout" in fig):
        content = fig
    else:
        received_type = type(fig).__name__
        hint = ""
        if received_type == "DataFrame":
            hint = " You passed a pandas DataFrame instead of a Plotly figure."
        raise TypeError(
            f"Expected a Plotly Figure object (from plotly.express or plotly.graph_objects), "
            f"got {received_type!r}.{hint} Example usage:\n"
            f"  df = pd.DataFrame({{'Category': ['A', 'B', 'C'], 'Values': [10, 20, 15]}})\n"
            f"  fig = px.bar(df, x='Category', y='Values', title='Sample Chart')\n"
            f"  render_interactive_plot(fig)"
        )

    return {
        "type": "application/vnd.plotly.v1+json",
        "content": content,
    }


class ImportAliasTransformer(ast.NodeTransformer):
    """Rewrites convenience aliases (e.g. `import px` -> `import plotly.express as px`).

    LLMs frequently write pseudo-imports like `import px`, `import pd`, `import np`, `import sp`,
    or `from px import ...` based on prompt descriptions. This transformer normalizes them
    to standard, valid Python imports before AST execution.
    """
    REWRITES = {
        "px": ("plotly.express", "px"),
        "go": ("plotly.graph_objects", "go"),
        "sp": ("sympy", "sp"),
        "np": ("numpy", "np"),
        "pd": ("pandas", "pd"),
    }
    MODULE_REWRITES = {
        "px": "plotly.express",
        "go": "plotly.graph_objects",
        "sp": "sympy",
        "np": "numpy",
        "pd": "pandas",
    }

    def visit_Import(self, node: ast.Import) -> ast.Import:
        new_names = []
        for alias in node.names:
            parts = alias.name.split(".", 1)
            root = parts[0]
            if root in self.REWRITES:
                mod, default_as = self.REWRITES[root]
                rest = f".{parts[1]}" if len(parts) > 1 else ""
                full_mod = mod + rest
                asname = alias.asname or (default_as if not rest else None)
                new_names.append(ast.alias(name=full_mod, asname=asname))
            else:
                new_names.append(alias)
        node.names = new_names
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        if node.module:
            parts = node.module.split(".", 1)
            root = parts[0]
            if root in self.MODULE_REWRITES:
                mod = self.MODULE_REWRITES[root]
                rest = f".{parts[1]}" if len(parts) > 1 else ""
                node.module = mod + rest
        return node


def extract_assigned_names(tree: ast.AST) -> set[str]:
    """Collects variable names assigned or bound in an AST."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                for sub in ast.walk(t):
                    if isinstance(sub, ast.Name):
                        names.add(sub.id)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            for sub in ast.walk(node.target):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
        elif isinstance(node, ast.NamedExpr):
            for sub in ast.walk(node.target):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            for sub in ast.walk(node.target):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
    return names


def create_jailed_open(workspace_dir: Path | str):
    """Creates a jailed open() function that restricts file I/O to the workspace directory."""
    real_ws = os.path.realpath(str(workspace_dir))
    os.makedirs(real_ws, exist_ok=True)

    def _jailed_open(file, mode="r", *args, **kwargs):
        if isinstance(file, int):
            raise PermissionError("Access to raw file descriptors is restricted in Code Mode sandbox.")
        raw_path = str(file)
        if not os.path.isabs(raw_path):
            target = os.path.realpath(os.path.join(real_ws, raw_path))
        else:
            target = os.path.realpath(raw_path)

        try:
            common = os.path.commonpath([real_ws, target])
        except ValueError:
            raise PermissionError(
                f"Access denied: '{file}' is outside the allowed workspace directory ({real_ws})."
            )

        if common != real_ws:
            raise PermissionError(
                f"Access denied: '{file}' is outside the allowed workspace directory ({real_ws})."
            )

        # Automatically create parent directories on write / append
        if any(m in mode for m in ("w", "a", "x", "+")):
            os.makedirs(os.path.dirname(target), exist_ok=True)

        return builtins.open(target, mode, *args, **kwargs)

    return _jailed_open


def create_base_globals(custom_tools: dict[str, Any] | None = None) -> dict[str, Any]:
    """Builds a rich, pre-populated execution namespace so models never waste tokens on imports."""
    env: dict[str, Any] = {
        "__name__": "__main__",
        "math": math,
        "cmath": cmath,
        "json": json,
    }

    # Pre-import numpy as np if present
    try:
        import numpy as np
        env["np"] = np
        env["numpy"] = np
        sys.modules.setdefault("np", np)
    except ImportError:
        pass

    # Pre-import sympy as sp if present
    try:
        import sympy as sp
        env["sp"] = sp
        env["sympy"] = sp
        sys.modules.setdefault("sp", sp)
    except ImportError:
        pass

    # Pre-import pint if present
    try:
        import pint
        env["pint"] = pint
    except ImportError:
        pass

    # Pre-import z3 if present
    try:
        import z3
        env["z3"] = z3
    except ImportError:
        pass

    # Pre-import pandas as pd if present
    try:
        import pandas as pd
        env["pd"] = pd
        env["pandas"] = pd
        sys.modules.setdefault("pd", pd)
    except ImportError:
        pass

    # Pre-import plotly.express as px and plotly.graph_objects as go if present
    _plot_wrapper = None
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        env["px"] = px
        env["go"] = go
        sys.modules.setdefault("px", px)
        sys.modules.setdefault("go", go)
        if hasattr(px, "colors"):
            sys.modules.setdefault("px.colors", px.colors)
    except ImportError:
        pass

    # Pre-import astropy if present
    try:
        import astropy
        env["astropy"] = astropy
    except ImportError:
        pass

    # Injected interactive plotting helper that captures plot in state so it is never lost
    def _plot_wrapper(fig: Any) -> dict[str, Any]:
        plot_dict = render_interactive_plot(fig)
        env["_last_plot"] = plot_dict
        env["result"] = plot_dict
        return plot_dict

    env["render_interactive_plot"] = _plot_wrapper

    # Monkey patch Plotly Figure.show so standard Plotly fig.show() routes to render_interactive_plot
    try:
        import plotly.graph_objects as go
        go.Figure.show = lambda self, *a, **kw: _plot_wrapper(self)
    except ImportError:
        pass

    def _st_loop_tick():
        env["_st_loop_count"] = env.get("_st_loop_count", 0) + 1
        if env["_st_loop_count"] > 500_000:
            raise TimeoutError("Loop iteration limit exceeded in Code Mode (500,000 iterations)")

    env["_st_loop_tick"] = _st_loop_tick
    env["_st_loop_count"] = 0

    def _blocked_input(*a, **kw):
        raise RuntimeError("Interactive input() is not permitted in Code Mode.")

    env["input"] = _blocked_input
    env["getpass"] = _blocked_input
    env["open"] = create_jailed_open(WORKSPACE_DIR)
    env["WORKSPACE_DIR"] = str(WORKSPACE_DIR)

    def _list_workspace_files() -> list[str]:
        real_ws = os.path.realpath(str(WORKSPACE_DIR))
        files = []
        for root, _, filenames in os.walk(real_ws):
            for fn in filenames:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, real_ws)
                files.append(rel)
        return sorted(files)

    env["list_workspace_files"] = _list_workspace_files

    # Injected symbolic & logic tools
    if verify_equivalence is not None:
        env["verify_equivalence"] = verify_equivalence
    if dimensional_analysis is not None:
        env["dimensional_analysis"] = dimensional_analysis
    if solve_and_derive is not None:
        env["solve_and_derive"] = solve_and_derive
    if compile_equation_to_code is not None:
        env["compile_equation_to_code"] = compile_equation_to_code
    if solve_constraints is not None:
        env["solve_constraints"] = solve_constraints
    if lint_python_code is not None:
        env["lint_python_code"] = lint_python_code
    if calculate_time_delta is not None:
        env["calculate_time_delta"] = calculate_time_delta

    if custom_tools:
        env.update(custom_tools)

    return env


class LLMSandbox:
    """Stateful AST execution sandbox with session management."""

    def __init__(self, ttl_s: float = DEFAULT_SESSION_TTL_S, max_sessions: int = MAX_SESSIONS):
        self.ttl_s = ttl_s
        self.max_sessions = max_sessions
        # session_id -> (last_access_timestamp, globals_dict)
        self._sessions: dict[str, tuple[float, dict[str, Any]]] = {}

    def _cleanup_expired(self, now: float) -> None:
        """Removes sessions that have exceeded the inactivity TTL."""
        expired = [sid for sid, (ts, _) in self._sessions.items() if (now - ts) > self.ttl_s]
        for sid in expired:
            self._sessions.pop(sid, None)

        # LRU eviction if maximum session count is exceeded
        if len(self._sessions) > self.max_sessions:
            sorted_sessions = sorted(self._sessions.items(), key=lambda kv: kv[1][0])
            to_remove = len(self._sessions) - self.max_sessions
            for sid, _ in sorted_sessions[:to_remove]:
                self._sessions.pop(sid, None)

    def get_session_state(self, session_id: str, custom_tools: dict[str, Any] | None = None) -> dict[str, Any]:
        """Gets or initializes the state dictionary for a given session."""
        now = time.monotonic()
        self._cleanup_expired(now)

        if session_id in self._sessions:
            _, state = self._sessions[session_id]
            if custom_tools:
                state.update(custom_tools)
            self._sessions[session_id] = (now, state)
            return state

        state = create_base_globals(custom_tools)
        self._sessions[session_id] = (now, state)
        return state

    def reset_session(self, session_id: str | None = None) -> None:
        """Resets the state of a specific session or all sessions."""
        if session_id is None:
            self._sessions.clear()
        else:
            self._sessions.pop(session_id, None)

    def execute(
        self,
        code: str,
        session_id: str = "default",
        custom_tools: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Executes LLM-generated code with AST expression auto-return and session persistence.

        Returns:
            dict containing:
            - 'ok': bool
            - 'output': str (human/model readable combined output)
            - 'result': Any (the raw evaluated return value, if any)
            - 'error': str | None (error message if failed)
        """
        clean_code = strip_markdown_fences(code)
        if not clean_code.strip():
            return {
                "ok": True,
                "output": "(completed with no output)",
                "result": None,
                "error": None,
            }

        clean_code = attempt_syntax_repair(clean_code)

        state = self.get_session_state(session_id, custom_tools)
        # Clear previous run's result and plot variables so each turn evaluates fresh
        state["result"] = None
        state["_last_plot"] = None
        state["_st_loop_count"] = 0

        def _st_loop_tick():
            state["_st_loop_count"] += 1
            if state["_st_loop_count"] > 500_000:
                raise TimeoutError("Loop iteration limit exceeded in Code Mode (500,000 iterations)")

        state["_st_loop_tick"] = _st_loop_tick

        user_stdout = io.StringIO()
        user_stderr = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr

        try:
            tree = ast.parse(clean_code, "<codemode>")
            # Layer 2 Security sandbox static checks
            SecurityVisitor().visit(tree)
            ImportAliasTransformer().visit(tree)
            # Layer 2 Loop bounding / timeout guard injection
            LoopBoundingTransformer().visit(tree)
            ast.fix_missing_locations(tree)
            assigned_names = extract_assigned_names(tree)
            last_expr = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = tree.body.pop()

            sys.stdout = user_stdout
            sys.stderr = user_stderr

            # 1. Execute all statements prior to the final expression
            if tree.body:
                exec_code = compile(tree, "<codemode>", "exec")
                exec(exec_code, state)

            # 2. Evaluate the final expression (if present)
            evaluated_val = None
            if last_expr is not None:
                eval_tree = ast.Expression(body=last_expr.value)
                eval_code = compile(eval_tree, "<codemode>", "eval")
                evaluated_val = eval(eval_code, state)

                # Auto-render Plotly Figure if left as standalone expression
                if evaluated_val is not None:
                    is_fig = False
                    try:
                        import plotly.graph_objects as go
                        if isinstance(evaluated_val, go.Figure):
                            is_fig = True
                    except ImportError:
                        pass
                    if not is_fig and hasattr(evaluated_val, "data") and hasattr(evaluated_val, "layout") and callable(getattr(evaluated_val, "to_json", None)):
                        is_fig = True
                    if is_fig:
                        evaluated_val = render_interactive_plot(evaluated_val)

                # Set result in state if not explicitly overwritten
                if state.get("result") is None and evaluated_val is not None:
                    state["result"] = evaluated_val

            ret = state.get("result")
            # If result is not set, fallback to the last captured plot
            if ret is None and state.get("_last_plot") is not None:
                ret = state["_last_plot"]

            if ret is None and callable(state.get("main")):
                ret = state["main"]()

            # If still None, check if any variable assigned during THIS execution holds an unrendered Plotly figure
            if ret is None and assigned_names:
                preferred = [k for k in ("fig", "figure", "plot", "chart") if k in assigned_names]
                other = [k for k in assigned_names if k not in preferred and not k.startswith("_")]
                for k in preferred + other:
                    val = state.get(k)
                    if val is None:
                        continue
                    is_candidate_fig = False
                    try:
                        import plotly.graph_objects as go
                        if isinstance(val, go.Figure):
                            is_candidate_fig = True
                    except ImportError:
                        pass
                    if not is_candidate_fig and hasattr(val, "data") and hasattr(val, "layout") and callable(getattr(val, "to_json", None)):
                        is_candidate_fig = True
                    if is_candidate_fig:
                        try:
                            ret = render_interactive_plot(val)
                            state["_last_plot"] = ret
                            break
                        except Exception:
                            pass

            stdout_val = user_stdout.getvalue().strip()
            stderr_val = user_stderr.getvalue().strip()

            def _format_val(v: Any) -> str:
                if isinstance(v, dict) and v.get("type") == "application/vnd.plotly.v1+json":
                    return json.dumps(v, ensure_ascii=False)
                if isinstance(v, (dict, list)):
                    try:
                        return json.dumps(v, indent=2, ensure_ascii=False)
                    except Exception:
                        return str(v)
                return str(v)

            if stdout_val and ret is not None:
                formatted_ret = _format_val(ret)
                # Avoid duplicating output if stdout already contains the JSON plot payload
                if isinstance(ret, dict) and ret.get("type") == "application/vnd.plotly.v1+json" and (
                    formatted_ret in stdout_val or stdout_val.startswith('{"type": "application/vnd.plotly.v1+json"')
                ):
                    output_text = stdout_val
                else:
                    output_text = f"{stdout_val}\n\nResult: {formatted_ret}"
            elif stdout_val:
                output_text = stdout_val
            elif ret is not None:
                output_text = _format_val(ret)
            elif stderr_val:
                output_text = stderr_val
            else:
                output_text = "(completed with no output)"

            return {
                "ok": True,
                "output": output_text,
                "result": ret,
                "error": None,
            }
        except SystemExit as exc:
            stdout_val = user_stdout.getvalue().strip()
            if exc.code == 0 or exc.code is None:
                return {
                    "ok": True,
                    "output": stdout_val or "(exited 0)",
                    "result": None,
                    "error": None,
                }
            return {
                "ok": False,
                "output": f"sys.exit({exc.code})",
                "result": None,
                "error": f"sys.exit({exc.code})\n{user_stderr.getvalue()}",
            }
        except Exception as exc:
            return {
                "ok": False,
                "output": f"Error: {type(exc).__name__}: {exc}",
                "result": None,
                "error": traceback.format_exc(),
            }
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
