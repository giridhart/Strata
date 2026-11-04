"""tools/mcp_logic.py - Constraint logic solver using the Z3 Theorem Prover.

Provides:
- solve_constraints: Dynamically parses string-based constraints and variables into Z3,
  solves or optimizes them, and returns satisfying assignments or proofs of unsatisfiability.
"""
from __future__ import annotations

from typing import Any
import z3


def solve_constraints(
    variables: list[str],
    constraints: list[str],
    types: dict[str, str] | None = None,
    maximize: str | None = None,
    minimize: str | None = None,
) -> dict[str, Any]:
    """Takes a list of variable names and constraint strings, returning a satisfying model.

    Args:
        variables: List of variable names (e.g. ['Alice', 'Bob', 'Charlie', 'x', 'y'])
        constraints: List of constraint strings (e.g. ['Alice < Bob', 'x + y <= 10', 'x > 0'])
        types: Optional dict mapping variable names to 'Int', 'Real', or 'Bool' (default is 'Int')
        maximize: Optional string expression to maximize (e.g. 'x + 2*y')
        minimize: Optional string expression to minimize (e.g. 'Alice')
    """
    type_map = types or {}
    z3_vars: dict[str, Any] = {}

    for var in variables:
        v_type = type_map.get(var, "Int").strip().capitalize()
        if v_type == "Real":
            z3_vars[var] = z3.Real(var)
        elif v_type == "Bool":
            z3_vars[var] = z3.Bool(var)
        else:
            z3_vars[var] = z3.Int(var)

    # Safe evaluation environment containing Z3 functions and variables
    eval_env: dict[str, Any] = {
        "__builtins__": {},
        "And": z3.And,
        "Or": z3.Or,
        "Not": z3.Not,
        "Implies": z3.Implies,
        "If": z3.If,
        "Distinct": z3.Distinct,
        "Sum": z3.Sum,
        "abs": abs,
        "min": min,
        "max": max,
        **z3_vars,
    }

    use_optimizer = bool(maximize or minimize)
    solver = z3.Optimize() if use_optimizer else z3.Solver()

    try:
        for c in constraints:
            c_str = str(c).strip()
            if not c_str:
                continue
            z3_expr = eval(c_str, eval_env, {})
            solver.add(z3_expr)

        if maximize:
            opt_expr = eval(str(maximize).strip(), eval_env, {})
            solver.maximize(opt_expr)
        if minimize:
            opt_expr = eval(str(minimize).strip(), eval_env, {})
            solver.minimize(opt_expr)

        result = solver.check()
        if result == z3.sat:
            model = solver.model()
            solution = {}
            for var in variables:
                z_val = model[z3_vars[var]]
                if z_val is None:
                    continue
                if z3.is_int(z_val):
                    solution[var] = z_val.as_long()
                elif z3.is_real(z_val):
                    # Convert algebraic or fraction numbers to float/str
                    try:
                        solution[var] = float(z_val.as_decimal(10).rstrip("?"))
                    except Exception:
                        solution[var] = str(z_val)
                elif z3.is_bool(z_val):
                    solution[var] = z3.is_true(z_val)
                else:
                    solution[var] = str(z_val)

            return {
                "satisfiable": True,
                "model": solution,
                "status": "sat",
            }
        elif result == z3.unsat:
            return {
                "satisfiable": False,
                "error": "UNSATISFIABLE: The provided constraints contradict each other.",
                "status": "unsat",
            }
        else:
            return {
                "satisfiable": False,
                "error": "UNKNOWN: The solver timed out or could not prove satisfiability.",
                "status": "unknown",
            }
    except Exception as exc:
        return {
            "satisfiable": False,
            "error": f"Constraint syntax or evaluation error: {type(exc).__name__}: {exc}",
            "status": "error",
        }
