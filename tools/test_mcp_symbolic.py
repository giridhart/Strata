"""tools/test_mcp_symbolic.py - Unit tests for symbolic math and dimensional analysis tools.

    python -m unittest tools/test_mcp_symbolic.py -v
"""
from __future__ import annotations

import unittest
from tools.mcp_symbolic import (
    compile_equation_to_code,
    dimensional_analysis,
    solve_and_derive,
    verify_equivalence,
)


class TestMcpSymbolic(unittest.TestCase):
    """Test algebraic equivalence, dimensional analysis, solving, and code compilation."""

    def test_verify_equivalence_algebraic(self):
        # Structurally identical
        res1 = verify_equivalence("x**2 - 1", "x**2 - 1")
        self.assertTrue(res1["equivalent"])
        self.assertEqual(res1["method"], "structural_identity")

        # Factored vs expanded
        res2 = verify_equivalence("(x - 1)*(x + 1)", "x**2 - 1")
        self.assertTrue(res2["equivalent"])
        self.assertEqual(res2["difference"], "0")

        # Trig identity
        res3 = verify_equivalence("sin(x)**2 + cos(x)**2", "1")
        self.assertTrue(res3["equivalent"])

        # Non-equivalent expressions
        res4 = verify_equivalence("x**2 + 1", "x**2 - 1")
        self.assertFalse(res4["equivalent"])

    def test_dimensional_analysis_valid(self):
        # Force = mass * acceleration
        res1 = dimensional_analysis(
            "F",
            "m * a",
            {"F": "newton", "m": "kg", "a": "meter / second**2"},
        )
        self.assertTrue(res1["is_valid"])
        self.assertIn("[mass]", res1["lhs_dimensionality"])

        # Period of a pendulum: T = 2 * pi * sqrt(L / g)
        res2 = dimensional_analysis(
            "T",
            "2 * pi * sqrt(L / g)",
            {"T": "second", "L": "meter", "g": "meter / second^2", "pi": "dimensionless"},
        )
        self.assertTrue(res2["is_valid"])
        self.assertEqual(res2["lhs_dimensionality"], "[time]")
        self.assertEqual(res2["rhs_dimensionality"], "[time]")

    def test_dimensional_analysis_invalid(self):
        # Intentionally wrong pendulum formula (mass instead of gravity)
        res = dimensional_analysis(
            "T",
            "2 * pi * sqrt(L / m)",
            {"T": "second", "L": "meter", "m": "kg", "pi": "dimensionless"},
        )
        self.assertFalse(res["is_valid"])
        self.assertIn("Dimensional mismatch", res["feedback"])

    def test_solve_and_derive(self):
        # Solving polynomial
        res1 = solve_and_derive("x^2 - 25 = 0", solve_for="x")
        self.assertEqual(sorted(res1["solutions"]), ["-5", "5"])

        # Differentiation
        res2 = solve_and_derive("sin(x) * exp(x)", derive_wrt="x")
        self.assertIn("exp(x)*sin(x) + exp(x)*cos(x)", res2["derivative"])

    def test_compile_equation_to_code(self):
        expr = "sin(x) * exp(-y**2)"

        c_res = compile_equation_to_code(expr, "c")
        self.assertEqual(c_res["target_lang"], "c")
        self.assertIn("sin(x)", c_res["code"])

        rust_res = compile_equation_to_code(expr, "rust")
        self.assertEqual(rust_res["target_lang"], "rust")
        self.assertIn("sin()", rust_res["code"])

        py_res = compile_equation_to_code(expr, "py")
        self.assertEqual(py_res["target_lang"], "py")
        self.assertIn("math.sin(x)", py_res["code"])

        js_res = compile_equation_to_code(expr, "js")
        self.assertEqual(js_res["target_lang"], "js")
        self.assertIn("Math.sin(x)", js_res["code"])


if __name__ == "__main__":
    unittest.main()
