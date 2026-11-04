"""tools/mcp_symbolic.py - Symbolic mathematics and dimensional analysis tools.

Provides:
- verify_equivalence: Algebraic and transcendental equivalence verification via SymPy.
- dimensional_analysis: Physical unit and dimensionality verification via Pint.
- solve_and_derive: Symbolic equation solving and differentiation.
- compile_equation_to_code: Compilation of SymPy expressions into C, Rust, Python, JS, etc.
"""
from __future__ import annotations

import ast
import re
from typing import Any
import pint
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
_UREF = None


def get_unit_registry() -> pint.UnitRegistry:
    """Singleton unit registry for Pint."""
    global _UREF
    if _UREF is None:
        _UREF = pint.UnitRegistry()
    return _UREF


def _clean_math_str(s: str) -> str:
    """Normalizes math string representations (e.g. caret exponents, unicode)."""
    s = s.strip()
    s = s.replace("×", "*").replace("÷", "/")
    return s


def verify_equivalence(expr1: str, expr2: str) -> dict[str, Any]:
    """Parses two math strings and verifies whether they are mathematically equivalent.

    Uses sympy.simplify(expr1 - expr2) == 0 first, falling back to .equals()
    for transcendental functions, roots, and complex branch evaluations.
    """
    try:
        e1 = parse_expr(_clean_math_str(expr1), transformations=TRANSFORMS)
        e2 = parse_expr(_clean_math_str(expr2), transformations=TRANSFORMS)

        # 1. Direct structural equality
        if e1 == e2:
            return {
                "equivalent": True,
                "difference": "0",
                "method": "structural_identity",
                "explanation": "Expressions are structurally identical.",
            }

        # 2. Algebraic simplification of the difference
        diff = sp.simplify(e1 - e2)
        if diff == 0:
            return {
                "equivalent": True,
                "difference": "0",
                "method": "algebraic_simplification",
                "explanation": "Simplified difference (expr1 - expr2) is exactly 0.",
            }

        # 3. Transcendental equivalence fallback via .equals()
        is_eq = e1.equals(e2)
        if is_eq is True:
            return {
                "equivalent": True,
                "difference": str(diff),
                "method": "transcendental_evaluation",
                "explanation": "Expressions evaluate to equivalent values across the domain.",
            }
        elif is_eq is False:
            return {
                "equivalent": False,
                "difference": str(diff),
                "method": "domain_counterexample",
                "explanation": f"Expressions differ; simplified difference is {diff}.",
            }
        else:
            return {
                "equivalent": False,
                "difference": str(diff),
                "method": "undetermined",
                "explanation": f"Could not prove equivalence; simplified difference is {diff}.",
            }
    except Exception as exc:
        return {
            "equivalent": False,
            "error": f"Parse or evaluation error: {type(exc).__name__}: {exc}",
        }


class _PintDimEvaluator:
    def __init__(self, ureg: pint.UnitRegistry, var_map: dict[str, Any]):
        self.ureg = ureg
        self.var_map = var_map

    def eval_node(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return self.eval_node(node.body)
        elif isinstance(node, ast.Constant):
            return self.ureg.Quantity(float(node.value))
        elif isinstance(node, ast.Name):
            if node.id in self.var_map:
                return self.var_map[node.id]
            if node.id in ("pi", "e"):
                return self.ureg.Quantity(1.0)
            raise ValueError(f"Unknown variable in dimensional analysis: {node.id!r}")
        elif isinstance(node, ast.BinOp):
            left = self.eval_node(node.left)
            right = self.eval_node(node.right)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                if left.dimensionality != right.dimensionality:
                    raise ValueError(
                        f"Cannot add/subtract incompatible dimensions: [{left.dimensionality}] and [{right.dimensionality}]"
                    )
                return left
            elif isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.Div):
                return left / right
            elif isinstance(node.op, (ast.Pow, ast.BitXor)):
                exp = float(right.magnitude) if hasattr(right, "magnitude") else float(right)
                return left ** exp
            raise ValueError(f"Unsupported operator in dimension evaluation: {type(node.op).__name__}")
        elif isinstance(node, ast.UnaryOp):
            return self.eval_node(node.operand)
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else ""
            args = [self.eval_node(a) for a in node.args]
            if func_name == "sqrt" and len(args) == 1:
                return args[0] ** 0.5
            elif func_name == "cbrt" and len(args) == 1:
                return args[0] ** (1 / 3)
            elif func_name in ("sin", "cos", "tan", "exp", "log") and len(args) == 1:
                if not args[0].dimensionless:
                    raise ValueError(f"Argument to {func_name}() must be dimensionless, got [{args[0].dimensionality}]")
                return self.ureg.Quantity(1.0)
            raise ValueError(f"Unsupported function in dimension evaluation: {func_name}")
        raise ValueError(f"Unsupported syntax node: {type(node).__name__}")


def dimensional_analysis(equation_lhs: str, equation_rhs: str, variables: dict[str, str]) -> dict[str, Any]:
    """Verifies that the left and right sides of an equation have identical physical dimensions.

    Args:
        equation_lhs: String of the left side of the equation (e.g. 'F')
        equation_rhs: String of the right side of the equation (e.g. 'm * a')
        variables: Dict mapping variable names to unit strings (e.g. {'F': 'newton', 'm': 'kg', 'a': 'm/s^2'})
    """
    ureg = get_unit_registry()
    try:
        pint_vars: dict[str, Any] = {}
        for var_name, unit_str in variables.items():
            u = str(unit_str).strip()
            if u.lower() in ("dimensionless", "none", "scalar", "1"):
                pint_vars[var_name] = ureg.Quantity(1.0)
            else:
                pint_vars[var_name] = ureg.Quantity(1.0, u)

        def _eval_side(side_str: str) -> Any:
            norm = _clean_math_str(side_str).replace("^", "**")
            tree = ast.parse(norm, mode="eval")
            evaluator = _PintDimEvaluator(ureg, pint_vars)
            return evaluator.eval_node(tree)

        lhs_dim = _eval_side(equation_lhs)
        rhs_dim = _eval_side(equation_rhs)

        lhs_d = getattr(lhs_dim, "dimensionality", ureg.Quantity(1.0).dimensionality)
        rhs_d = getattr(rhs_dim, "dimensionality", ureg.Quantity(1.0).dimensionality)

        is_valid = (lhs_d == rhs_d)
        return {
            "is_valid": is_valid,
            "lhs_dimensionality": str(lhs_d),
            "rhs_dimensionality": str(rhs_d),
            "feedback": (
                "Equation is dimensionally sound."
                if is_valid
                else f"Dimensional mismatch: LHS has [{lhs_d}] while RHS has [{rhs_d}]."
            ),
        }
    except pint.UndefinedUnitError as exc:
        return {"is_valid": False, "error": f"Unknown unit provided: {exc}"}
    except Exception as exc:
        return {"is_valid": False, "error": f"Dimensional evaluation error: {type(exc).__name__}: {exc}"}


def solve_and_derive(
    equation: str,
    solve_for: str | None = None,
    derive_wrt: str | None = None,
) -> dict[str, Any]:
    """Solves an equation for target variables or computes its derivative symbolically.

    Args:
        equation: Math equation (e.g. 'x^2 - 4 = 0' or 'x^2 - 4' or 'sin(x)*exp(x)')
        solve_for: Variable name to solve for (optional)
        derive_wrt: Variable name to compute derivative with respect to (optional)
    """
    try:
        raw = _clean_math_str(equation)
        if "=" in raw:
            parts = raw.split("=", 1)
            lhs = parse_expr(parts[0], transformations=TRANSFORMS)
            rhs = parse_expr(parts[1], transformations=TRANSFORMS)
            expr = lhs - rhs
        else:
            expr = parse_expr(raw, transformations=TRANSFORMS)

        result: dict[str, Any] = {"expression": str(expr)}

        if solve_for:
            target_sym = sp.Symbol(solve_for.strip())
            solutions = sp.solve(expr, target_sym)
            result["solutions"] = [str(s) for s in solutions]

        if derive_wrt:
            wrt_sym = sp.Symbol(derive_wrt.strip())
            derivative = sp.diff(expr, wrt_sym)
            result["derivative"] = str(derivative)

        return result
    except Exception as exc:
        return {"error": f"Symbolic math error: {type(exc).__name__}: {exc}"}


def compile_equation_to_code(expr: str, target_lang: str) -> dict[str, Any]:
    """Compiles a SymPy expression into target language code (C, Rust, Python, JavaScript, Julia).

    Args:
        expr: SymPy expression string (e.g. 'sin(x) * exp(-y**2)')
        target_lang: 'c', 'rust', 'python' ('py'), 'javascript' ('js'), 'julia', 'fortran'
    """
    try:
        parsed = parse_expr(_clean_math_str(expr), transformations=TRANSFORMS)
        lang = target_lang.strip().lower()

        if lang in ("c", "cpp", "c++"):
            code = sp.ccode(parsed)
        elif lang in ("rust", "rs"):
            code = sp.rust_code(parsed)
        elif lang in ("python", "py"):
            code = sp.pycode(parsed)
        elif lang in ("javascript", "js"):
            code = sp.jscode(parsed)
        elif lang in ("julia", "jl"):
            code = sp.julia_code(parsed)
        elif lang in ("fortran", "f90"):
            code = sp.fcode(parsed)
        else:
            return {"error": f"Unsupported target language: {target_lang}. Supported: c, rust, python, javascript, julia, fortran"}

        return {"target_lang": lang, "code": code}
    except Exception as exc:
        return {"error": f"Code compilation error: {type(exc).__name__}: {exc}"}
