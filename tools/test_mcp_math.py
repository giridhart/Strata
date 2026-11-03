"""tools/test_mcp_math.py - unit tests for tools/mcp_math.py and Strata MCP integration.

    python -m unittest tools/test_mcp_math.py -v
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve.mcp import McpHub, hub_from_config
from tools import mcp_math


class TestMathEvaluator(unittest.TestCase):
    """Direct tests for the safe AST math evaluator."""

    def test_arithmetic(self):
        cases = [
            ("1 + 1", "2"),
            ("10 - 20", "-10"),
            ("3 * 7", "21"),
            ("10 / 4", "2.5"),
            ("10 // 3", "3"),
            ("10 % 3", "1"),
            ("2 ** 8", "256"),
            ("2 ^ 8", "256"),  # convenient caret exponent
            ("5 + 2^3", "13"),  # caret operator precedence over addition
            ("sqrt(1025) + 2^10 + pow(20, 34)", "1.7179869184e+44"),
            ("-(5 + 3)", "-8"),
            ("+(5 + 3)", "8"),
        ]
        for expr, expected in cases:
            with self.subTest(expr=expr):
                self.assertEqual(mcp_math.evaluate_expression(expr), expected)

    def test_comparisons(self):
        cases = [
            ("5 > 3", "true"),
            ("5 < 3", "false"),
            ("10 == 10", "true"),
            ("10 != 5", "true"),
            ("1 < 2 < 3", "true"),
            ("1 < 2 > 3", "false"),
        ]
        for expr, expected in cases:
            with self.subTest(expr=expr):
                self.assertEqual(mcp_math.evaluate_expression(expr), expected)

    def test_functions_and_constants(self):
        cases = [
            ("sqrt(144)", "12"),
            ("cbrt(27)", "3"),
            ("abs(-99)", "99"),
            ("min(5, 2, 9)", "2"),
            ("max(5, 2, 9)", "9"),
            ("round(3.14159, 3)", "3.142"),
            ("floor(4.9)", "4"),
            ("ceil(4.1)", "5"),
            ("factorial(6)", "720"),
            ("gcd(54, 24)", "6"),
            ("pi > 3.14", "true"),
            ("e > 2.71", "true"),
        ]
        for expr, expected in cases:
            with self.subTest(expr=expr):
                self.assertEqual(mcp_math.evaluate_expression(expr), expected)

    def test_floating_point_rounding(self):
        # 14389.25 * 38.2 in raw IEEE 754 is 549669.3500000001
        self.assertEqual(mcp_math.evaluate_expression("14389.25 * 38.2"), "549669.35")

    def test_safety_and_errors(self):
        errors = [
            ("1 / 0", ZeroDivisionError),
            ("sqrt(-1)", ValueError),
            ("__import__('os')", NameError),
            ("open('/etc/passwd')", NameError),
            ("eval('2+2')", NameError),
            ("2 ** 100000", ValueError),
            ("factorial(100000)", ValueError),
            ("", ValueError),
        ]
        for expr, err in errors:
            with self.subTest(expr=expr):
                with self.assertRaises(err):
                    mcp_math.evaluate_expression(expr)


class TestJsonRpcProtocol(unittest.TestCase):
    """Test JSON-RPC request and notification handling in mcp_math."""

    def test_initialize(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}}
        ans = mcp_math.handle_request(req)
        self.assertEqual(ans["id"], 1)
        self.assertEqual(ans["result"]["serverInfo"]["name"], "math")
        self.assertIn("tools", ans["result"]["capabilities"])

    def test_notification(self):
        note = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        self.assertIsNone(mcp_math.handle_request(note))

    def test_tools_list(self):
        req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        ans = mcp_math.handle_request(req)
        tools = ans["result"]["tools"]
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "calculate")

    def test_tools_call_success(self):
        req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "calculate", "arguments": {"expression": "25 * 4 + 7"}},
        }
        ans = mcp_math.handle_request(req)
        self.assertEqual(ans["result"]["content"][0]["text"], "107")
        self.assertFalse(ans["result"]["isError"])

    def test_tools_call_error(self):
        req = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "calculate", "arguments": {"expression": "1 / 0"}},
        }
        ans = mcp_math.handle_request(req)
        self.assertTrue(ans["result"]["isError"])
        self.assertIn("division by zero", ans["result"]["content"][0]["text"])


    def test_tools_call_batch(self):
        req = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "calculate", "arguments": {"expressions": ["sqrt(1025)", "2^10", "14389.25 * 38.2"]}},
        }
        ans = mcp_math.handle_request(req)
        self.assertFalse(ans["result"]["isError"])
        text = ans["result"]["content"][0]["text"]
        self.assertIn("sqrt(1025) = 32.015621187164", text)
        self.assertIn("2^10 = 1024", text)
        self.assertIn("14389.25 * 38.2 = 549669.35", text)

    def test_tools_call_batch_partial_error(self):
        req = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "calculate", "arguments": {"expressions": ["10 + 2", "1 / 0"]}},
        }
        ans = mcp_math.handle_request(req)
        text = ans["result"]["content"][0]["text"]
        self.assertIn("10 + 2 = 12", text)
        self.assertIn("1 / 0 = error: division by zero", text)

    def test_tools_call_missing_args(self):
        req = {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": "calculate", "arguments": {}},
        }
        ans = mcp_math.handle_request(req)
        self.assertTrue(ans["result"]["isError"])
        self.assertIn("provide 'expression' or 'expressions'", ans["result"]["content"][0]["text"])


class TestStrataHubIntegration(unittest.TestCase):
    """Test loading and calling tools/mcp_math.py via Strata's McpHub."""

    def test_hub_with_math_server(self):
        hub = McpHub({"math": {"command": sys.executable, "args": [str(ROOT / "tools" / "mcp_math.py")]}})
        hub.start(wait=True)
        try:
            tools = [t["name"] for t in hub.template_tools()]
            self.assertIn("math__calculate", tools)
            res = hub.call("math__calculate", {"expression": "sqrt(256) + 10 * 3"})
            self.assertTrue(res["ok"])
            self.assertEqual(res["text"], "46")

            # Batch call via hub
            b_res = hub.call("math__calculate", {"expressions": ["2^8", "sqrt(100)"]})
            self.assertTrue(b_res["ok"])
            self.assertIn("2^8 = 256", b_res["text"])
            self.assertIn("sqrt(100) = 10", b_res["text"])
        finally:
            hub.close()

    def test_hub_from_test_env_json(self):
        env_file = ROOT / "tools" / "mcp_test_env.json"
        self.assertTrue(env_file.exists())
        hub = hub_from_config({}, mcp_config_path=str(env_file))
        self.assertIsNotNone(hub)
        self.assertIn("math", hub.servers)
        self.assertEqual(hub.settings["max_rounds"], 8)


if __name__ == "__main__":
    unittest.main()
