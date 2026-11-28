"""tools/mcp_math.py - Model Context Protocol (MCP) server for precise math calculations.

Provides a safe, high-precision calculation tool over standard library MCP (stdio transport):
- Safe AST evaluation (no eval/exec, no arbitrary attribute access or code execution).
- Operators: +, -, *, /, //, %, ** (or ^), bitwise, and comparisons.
- Math functions: sqrt, cbrt, exp, log, log2, log10, sin, cos, tan, asin, acos, atan, atan2,
  sinh, cosh, tanh, ceil, floor, round, abs, factorial, gcd, lcm, comb, perm.
- Constants: pi, e, tau, inf.
- Standard library only (no external dependencies, works with Python 3.10+).

Usage:
    python tools/mcp_math.py              # Stdio MCP server (JSON-RPC 2.0 lines)
    python tools/mcp_math.py --test       # Run self-tests
"""
from __future__ import annotations

import ast
import json
import math
import operator
import sys
from typing import Any

PROTOCOL = "2025-06-18"

# Safety limits to prevent Denial of Service (DoS) via resource exhaustion
MAX_EXPONENT = 10000
MAX_FACTORIAL = 10000


def _safe_pow(base: Any, exp: Any) -> Any:
    if isinstance(exp, (int, float)):
        if abs(exp) > MAX_EXPONENT:
            raise ValueError(f"exponent {exp} exceeds safe limit (+/-{MAX_EXPONENT})")
    return operator.pow(base, exp)


def _safe_factorial(n: Any) -> int:
    if not isinstance(n, int):
        raise TypeError(f"factorial requires an integer, got {type(n).__name__}")
    if n < 0:
        raise ValueError("factorial requires a non-negative integer")
    if n > MAX_FACTORIAL:
        raise ValueError(f"factorial argument {n} exceeds safe limit ({MAX_FACTORIAL})")
    return math.factorial(n)


SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _safe_pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Not: operator.not_,
    ast.Invert: operator.invert,
}

SAFE_COMPARISONS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

SAFE_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "cbrt": getattr(math, "cbrt", lambda x: math.pow(x, 1 / 3)),
    "exp": math.exp,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "pow": _safe_pow,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "asinh": getattr(math, "asinh", None),
    "acosh": getattr(math, "acosh", None),
    "atanh": getattr(math, "atanh", None),
    "radians": math.radians,
    "degrees": math.degrees,
    "ceil": math.ceil,
    "floor": math.floor,
    "trunc": math.trunc,
    "factorial": _safe_factorial,
    "gcd": math.gcd,
    "lcm": getattr(math, "lcm", lambda a, b: abs(a * b) // math.gcd(a, b)),
    "comb": getattr(math, "comb", None),
    "perm": getattr(math, "perm", None),
}
SAFE_FUNCTIONS = {k: v for k, v in SAFE_FUNCTIONS.items() if v is not None}

SAFE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex, bool)):
            return node.value
        raise TypeError(f"unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.Name):
        if node.id in SAFE_CONSTANTS:
            return SAFE_CONSTANTS[node.id]
        raise NameError(f"unknown constant or identifier: '{node.id}'")

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            val = _eval_node(node.operand)
            return SAFE_OPERATORS[op_type](val)
        raise TypeError(f"unsupported unary operator: {op_type.__name__}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type is ast.BitXor:
            # Convenience: models and users often write 2^8 instead of 2**8
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            return _safe_pow(left, right)
        if op_type in SAFE_OPERATORS:
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            return SAFE_OPERATORS[op_type](left, right)
        raise TypeError(f"unsupported binary operator: {op_type.__name__}")

    if isinstance(node, ast.Compare):
        left = _eval_node(node.left)
        for cmp_op, comparator in zip(node.ops, node.comparators):
            op_type = type(cmp_op)
            if op_type not in SAFE_COMPARISONS:
                raise TypeError(f"unsupported comparison operator: {op_type.__name__}")
            right = _eval_node(comparator)
            if not SAFE_COMPARISONS[op_type](left, right):
                return False
            left = right
        return True

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise TypeError("only direct function calls are allowed (e.g. sqrt(4))")
        fname = node.func.id
        if fname not in SAFE_FUNCTIONS:
            raise NameError(f"unknown or disallowed function: '{fname}'")
        func = SAFE_FUNCTIONS[fname]
        args = [_eval_node(arg) for arg in node.args]
        kwargs = {kw.arg: _eval_node(kw.value) for kw in node.keywords if kw.arg}
        return func(*args, **kwargs)

    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_node(elt) for elt in node.elts]

    raise TypeError(f"disallowed expression structure: {type(node).__name__}")


def evaluate_expression(expr: str) -> str:
    """Safely evaluates a math expression and returns the formatted string result."""
    expr = expr.strip()
    if not expr:
        raise ValueError("empty expression")

    # Convenience: replace caret (^) with Python power operator (**) before AST parsing.
    # In Python grammar, BitXor (^) has lower precedence than addition (+), which would
    # incorrectly parse '5 + 2^3' as '(5 + 2)^3' instead of '5 + (2**3)'.
    expr = expr.replace("^", "**")

    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"syntax error: {e}") from None

    res = _eval_node(parsed.body)
    if isinstance(res, bool):
        return "true" if res else "false"
    if isinstance(res, int):
        return str(res)
    if isinstance(res, float):
        if math.isnan(res):
            return "NaN"
        if math.isinf(res):
            return "Infinity" if res > 0 else "-Infinity"
        return f"{res:.14g}"
    return str(res)


TOOLS = [
    {
        "name": "calculate",
        "description": (
            "Evaluate one or more mathematical expressions with precision. Supports standard operators "
            "(+, -, *, /, //, %, **, ^), comparisons, math functions (sqrt, cbrt, exp, log, "
            "log2, log10, sin, cos, tan, asin, acos, atan, ceil, floor, round, abs, factorial, "
            "gcd, lcm, comb, perm), and constants (pi, e, tau). "
            "To evaluate multiple independent expressions at once in a single round, pass a list under "
            "'expressions' rather than making multiple separate tool calls."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A single math expression to evaluate, e.g. 'sqrt(144) + 2.5 * 10' or 'factorial(10) / (2^5)'",
                },
                "expressions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of multiple math expressions to evaluate all at once, e.g. ['sqrt(1025)', '2^10', '5 * sqrt(41)']",
                },
            },
        },
    }
]


def text_response(t: str, error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": t}], "isError": error}


def handle_request(msg: dict) -> dict | None:
    """Processes one JSON-RPC 2.0 message."""
    method = msg.get("method")
    rid = msg.get("id")
    params = msg.get("params") or {}

    # Notifications do not return an answer
    if rid is None:
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {
                "protocolVersion": params.get("protocolVersion", PROTOCOL),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "math", "version": "0.1.0"},
            },
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "calculate":
            exprs = args.get("expressions")
            expr = args.get("expression")
            if exprs is not None and isinstance(exprs, list):
                out_lines = []
                for e in exprs:
                    e_str = str(e).strip()
                    if not e_str:
                        continue
                    try:
                        ans = evaluate_expression(e_str)
                        out_lines.append(f"{e_str} = {ans}")
                        print(f"eval: {e_str} = {ans}", file=sys.stderr, flush=True)
                    except Exception as err:
                        out_lines.append(f"{e_str} = error: {err}")
                        print(f"eval error: {e_str}: {err}", file=sys.stderr, flush=True)
                ans_text = "\n".join(out_lines) if out_lines else "error: empty expressions list"
                return {"jsonrpc": "2.0", "id": rid, "result": text_response(ans_text, error=False)}

            if expr is not None:
                expr_str = str(expr).strip()
                try:
                    ans = evaluate_expression(expr_str)
                    print(f"eval: {expr_str} = {ans}", file=sys.stderr, flush=True)
                    return {"jsonrpc": "2.0", "id": rid, "result": text_response(ans, error=False)}
                except Exception as e:
                    print(f"eval error: {expr_str}: {e}", file=sys.stderr, flush=True)
                    return {"jsonrpc": "2.0", "id": rid, "result": text_response(f"error: {e}", error=True)}

            return {"jsonrpc": "2.0", "id": rid, "result": text_response("error: provide 'expression' or 'expressions'", error=True)}

        return {
            "jsonrpc": "2.0",
            "id": rid,
            "error": {"code": -32602, "message": f"unknown tool: {name!r}"},
        }

    return {
        "jsonrpc": "2.0",
        "id": rid,
        "error": {"code": -32601, "message": f"unsupported method: {method!r}"},
    }


def run_stdio():
    """Runs standard newline-delimited JSON-RPC 2.0 on stdin/stdout."""
    for f in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(f, "reconfigure"):
            f.reconfigure(encoding="utf-8")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        ans = handle_request(msg)
        if ans is not None:
            sys.stdout.write(json.dumps(ans) + "\n")
            sys.stdout.flush()


def run_selftest():
    """Validates math evaluation operations."""
    test_cases = [
        ("2 + 2", "4"),
        ("10 - 3 * 2", "4"),
        ("(10 - 3) * 2", "14"),
        ("2^10", "1024"),
        ("2 ** 10", "1024"),
        ("sqrt(144)", "12"),
        ("factorial(5)", "120"),
        ("gcd(48, 18)", "6"),
        ("abs(-42)", "42"),
        ("round(3.14159, 2)", "3.14"),
        ("pi > 3", "true"),
        ("14389.25 * 38.2", "549669.35"),
    ]
    for expr, expected in test_cases:
        res = evaluate_expression(expr)
        assert res == expected, f"Expected {expr} == {expected}, got {res}"

    # Error cases
    for bad in ("1 / 0", "sqrt(-1)", "__import__('os')", "2 ** 100000"):
        try:
            evaluate_expression(bad)
            assert False, f"Expression {bad} should have failed"
        except (ValueError, ZeroDivisionError, NameError, TypeError):
            pass

    print(f"Self-test passed: {len(test_cases)} assertions verified.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_selftest()
    else:
        run_stdio()
