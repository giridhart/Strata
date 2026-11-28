"""tools/test_mcp_logic.py - Unit tests for Z3 constraint logic solver.

    python -m unittest tools/test_mcp_logic.py -v
"""
from __future__ import annotations

import unittest
from tools.mcp_logic import solve_constraints


class TestMcpLogic(unittest.TestCase):
    """Test Z3 constraint satisfaction, optimization, and conflict handling."""

    def test_scheduling_satisfiable(self):
        variables = ["Alice", "Bob", "Charlie"]
        constraints = [
            "Alice >= 9", "Alice <= 17",
            "Bob >= 9", "Bob <= 17",
            "Charlie == 14",
            "Alice < Bob",
            "Alice != Bob",
            "Alice != Charlie",
            "Bob != Charlie",
        ]
        res = solve_constraints(variables, constraints)
        self.assertTrue(res["satisfiable"])
        self.assertEqual(res["status"], "sat")
        model = res["model"]
        self.assertEqual(model["Charlie"], 14)
        self.assertLess(model["Alice"], model["Bob"])
        self.assertNotEqual(model["Alice"], model["Charlie"])
        self.assertNotEqual(model["Bob"], model["Charlie"])

    def test_unsatisfiable_conflict(self):
        variables = ["x"]
        constraints = ["x > 10", "x < 5"]
        res = solve_constraints(variables, constraints)
        self.assertFalse(res["satisfiable"])
        self.assertEqual(res["status"], "unsat")
        self.assertIn("UNSATISFIABLE", res["error"])

    def test_optimization_maximize(self):
        variables = ["x", "y"]
        constraints = ["x + y <= 10", "x >= 0", "y >= 0"]
        res = solve_constraints(variables, constraints, maximize="x + 2*y")
        self.assertTrue(res["satisfiable"])
        self.assertEqual(res["model"]["x"], 0)
        self.assertEqual(res["model"]["y"], 10)

    def test_types_real_variables(self):
        variables = ["x", "y"]
        types = {"x": "Real", "y": "Real"}
        constraints = ["x + y == 5.5", "x == 2.0"]
        res = solve_constraints(variables, constraints, types=types)
        self.assertTrue(res["satisfiable"])
        self.assertAlmostEqual(res["model"]["x"], 2.0)
        self.assertAlmostEqual(res["model"]["y"], 3.5)


if __name__ == "__main__":
    unittest.main()
