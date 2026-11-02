"""tools/test_mcp_datetime.py - unit tests for tools/mcp_datetime.py and Strata MCP integration."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve.mcp import McpHub
from tools import mcp_datetime


class TestDateTimeTools(unittest.TestCase):
    def test_current_time(self):
        t = mcp_datetime.get_current_time("UTC")
        self.assertIn("iso", t)
        self.assertIn("formatted", t)
        self.assertIn("unix_timestamp", t)

    def test_date_difference(self):
        diff = mcp_datetime.date_difference("2026-01-01", "2026-01-15")
        self.assertEqual(diff["total_days"], 14)

    def test_date_add(self):
        res = mcp_datetime.date_add("2026-03-01", days=10, hours=2)
        self.assertIn("2026-03-11", res["result_iso"])

    def test_date_info(self):
        info = mcp_datetime.date_info("2024-02-29")
        self.assertTrue(info["is_leap_year"])
        self.assertEqual(info["day_of_week"], "Thursday")


class TestJsonRpcProtocol(unittest.TestCase):
    def test_initialize(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        ans = mcp_datetime.handle_request(req)
        self.assertEqual(ans["id"], 1)
        self.assertEqual(ans["result"]["serverInfo"]["name"], "datetime")

    def test_tools_list(self):
        req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        ans = mcp_datetime.handle_request(req)
        tools = [t["name"] for t in ans["result"]["tools"]]
        self.assertIn("get_current_time", tools)
        self.assertIn("date_difference", tools)
        self.assertIn("date_add", tools)
        self.assertIn("date_info", tools)

    def test_tool_call(self):
        req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "date_difference", "arguments": {"start_date": "2026-01-01", "end_date": "2026-01-11"}},
        }
        ans = mcp_datetime.handle_request(req)
        self.assertFalse(ans["result"]["isError"])
        data = json.loads(ans["result"]["content"][0]["text"])
        self.assertEqual(data["total_days"], 10)


class TestHubIntegration(unittest.TestCase):
    def test_hub_with_datetime(self):
        hub = McpHub({"clock": {"command": sys.executable, "args": [str(ROOT / "tools" / "mcp_datetime.py")]}})
        hub.start(wait=True)
        try:
            tools = [t["name"] for t in hub.template_tools()]
            self.assertIn("clock__get_current_time", tools)
            res = hub.call("clock__get_current_time", {"timezone": "UTC"})
            self.assertTrue(res["ok"])
            self.assertIn("unix_timestamp", res["text"])
        finally:
            hub.close()


if __name__ == "__main__":
    unittest.main()
