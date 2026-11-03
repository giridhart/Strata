"""tools/test_mcp_linter.py - unit tests for tools/mcp_linter.py and Strata MCP integration."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve.mcp import McpHub
from tools import mcp_linter


class TestHtmlValidation(unittest.TestCase):
    def test_valid_html_document(self):
        code = (
            "<!doctype html>\n"
            "<html lang='en'>\n"
            "<head><meta charset='utf-8'><title>Dashboard</title></head>\n"
            "<body>\n"
            "  <div class='card'>\n"
            "    <h1>Metric</h1>\n"
            "    <img src='chart.svg' alt='Chart'>\n"
            "    <input type='text' value='ok'>\n"
            "    <br>\n"
            "  </div>\n"
            "</body>\n"
            "</html>"
        )
        res = mcp_linter.lint_code(code, "html")
        self.assertTrue(res["valid"])
        self.assertEqual(len(res["errors"]), 0)
        self.assertEqual(res["unclosed_tags"], [])

    def test_unclosed_tags(self):
        code = "<div class='container'><section><h2>Title</h2><p>Paragraph"
        res = mcp_linter.lint_code(code, "html")
        self.assertFalse(res["valid"])
        self.assertIn("p", res["unclosed_tags"])
        self.assertIn("section", res["unclosed_tags"])
        self.assertIn("div", res["unclosed_tags"])

    def test_mismatched_tags(self):
        code = "<div><span>Text</div></span>"
        res = mcp_linter.lint_code(code, "html")
        self.assertFalse(res["valid"])
        self.assertTrue(any("Mismatched closing tag" in err["message"] for err in res["errors"]))

    def test_unclosed_script(self):
        code = "<div><script>const x = 10;\nconsole.log(x);<p>Text</p></div>"
        res = mcp_linter.lint_code(code, "html")
        self.assertFalse(res["valid"])
        self.assertTrue(any("Unclosed <script>" in err["message"] for err in res["errors"]))

    def test_auto_repair(self):
        code = "<main><div class='panel'><p>Hello"
        res = mcp_linter.lint_code(code, "html", auto_repair=True)
        self.assertFalse(res["valid"])
        self.assertIsNotNone(res["repaired_code"])
        self.assertTrue(res["repaired_code"].endswith("</p></div></main>"))


class TestPythonValidation(unittest.TestCase):
    def test_valid_python(self):
        code = "def add(a: int, b: int) -> int:\n    return a + b\n\nprint(add(2, 3))\n"
        res = mcp_linter.lint_code(code, "python")
        self.assertTrue(res["valid"])
        self.assertEqual(len(res["errors"]), 0)

    def test_invalid_python(self):
        code = "def broken(\n    return 42\n"
        res = mcp_linter.lint_code(code, "python")
        self.assertFalse(res["valid"])
        self.assertTrue(any("SyntaxError" in err["message"] for err in res["errors"]))


class TestJsonValidation(unittest.TestCase):
    def test_valid_json(self):
        code = '{"name": "test", "items": [1, 2, 3], "nested": {"ok": true}}'
        res = mcp_linter.lint_code(code, "json")
        self.assertTrue(res["valid"])

    def test_invalid_json_trailing_comma(self):
        code = '{"name": "test", "items": [1, 2, ], }'
        res = mcp_linter.lint_code(code, "json")
        self.assertFalse(res["valid"])
        self.assertTrue(any("JSONDecodeError" in err["message"] for err in res["errors"]))


class TestBracketValidation(unittest.TestCase):
    def test_balanced_javascript(self):
        code = "function test() { const a = [1, 2]; if (a.length > 0) { console.log('ok'); } }"
        res = mcp_linter.lint_code(code, "javascript")
        self.assertTrue(res["valid"])

    def test_unbalanced_javascript(self):
        code = "function test() { const a = [1, 2; }"
        res = mcp_linter.lint_code(code, "javascript")
        self.assertFalse(res["valid"])
        self.assertTrue(any("Mismatched bracket" in err["message"] for err in res["errors"]))


class TestJsonRpcProtocol(unittest.TestCase):
    def test_initialize(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        ans = mcp_linter.handle_request(req)
        self.assertEqual(ans["id"], 1)
        self.assertEqual(ans["result"]["serverInfo"]["name"], "linter")

    def test_tools_list(self):
        req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        ans = mcp_linter.handle_request(req)
        tools = [t["name"] for t in ans["result"]["tools"]]
        self.assertIn("lint_code", tools)

    def test_tool_call(self):
        req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "lint_code",
                "arguments": {
                    "code": "<div class='box'><p>Text</p></div>",
                    "language": "html",
                },
            },
        }
        ans = mcp_linter.handle_request(req)
        self.assertFalse(ans["result"]["isError"])
        text = ans["result"]["content"][0]["text"]
        self.assertIn("VALID", text)
        self.assertIn("Passed", text)


class TestHubIntegration(unittest.TestCase):
    def test_hub_with_linter(self):
        hub = McpHub({"linter": {"command": sys.executable, "args": [str(ROOT / "tools" / "mcp_linter.py")]}})
        hub.start(wait=True)
        try:
            tools = [t["name"] for t in hub.template_tools()]
            self.assertIn("linter__lint_code", tools)
            res = hub.call("linter__lint_code", {"code": "def foo(): pass", "language": "python"})
            self.assertTrue(res["ok"])
            self.assertIn("VALID", res["text"])
        finally:
            hub.close()


if __name__ == "__main__":
    unittest.main()
