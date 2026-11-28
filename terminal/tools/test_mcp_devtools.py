"""tools/test_mcp_devtools.py - Unit tests for developer tools (Ruff linting & dateutil scheduling).

    python -m unittest tools/test_mcp_devtools.py -v
"""
from __future__ import annotations

import unittest
from tools.mcp_devtools import calculate_time_delta, lint_python_code


class TestMcpDevTools(unittest.TestCase):
    """Test Ruff linting and cross-timezone scheduling."""

    def test_lint_python_clean(self):
        code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        res = lint_python_code(code)
        self.assertTrue(res["valid"])
        self.assertEqual(len(res["errors"]), 0)

    def test_lint_python_syntax_error(self):
        bad_code = "def broken(:\n    pass\n"
        res = lint_python_code(bad_code)
        self.assertFalse(res["valid"])
        self.assertGreater(len(res["errors"]), 0)
        self.assertEqual(res["errors"][0]["source"], "syntax")

    def test_calculate_time_delta(self):
        # 4 business days between Monday 9am and Friday 5pm
        res = calculate_time_delta(
            "2026-11-02 09:00",
            "2026-11-06 17:00",
            timezone1="UTC",
            timezone2="UTC",
            business_days_only=True,
        )
        self.assertEqual(res["duration"]["days"], 4)
        self.assertEqual(res["duration"]["hours"], 8)
        self.assertEqual(res["business_days"], 5)

    def test_calculate_time_delta_cross_timezone(self):
        # 12:00 in London (UTC) vs 12:00 in New York (EST, UTC-5)
        # New York is 5 hours behind, so 12:00 NY is 17:00 UTC (5 hours later)
        res = calculate_time_delta(
            "2026-11-04 12:00",
            "2026-11-04 12:00",
            timezone1="UTC",
            timezone2="America/New_York",
        )
        self.assertEqual(res["duration"]["hours"], 5)


if __name__ == "__main__":
    unittest.main()
