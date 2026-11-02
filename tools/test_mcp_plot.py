"""tools/test_mcp_plot.py - unit tests for tools/mcp_plot.py and Strata MCP integration."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve.mcp import McpHub
from tools import mcp_plot


class TestPlotGenerator(unittest.TestCase):
    def test_line_chart(self):
        series = [{"name": "Runs", "values": [10, 20, 15, 30]}]
        svg, summary = mcp_plot.generate_svg_chart(series, chart_type="line", title="Benchmark")
        self.assertIn("<svg", svg)
        self.assertIn("</svg>", svg)
        self.assertIn("<path", svg)
        self.assertIn("<circle", svg)
        self.assertIn("Benchmark", svg)
        self.assertIn("Generated line chart", summary)

    def test_bar_chart(self):
        series = [
            {"name": "Model A", "values": [100, 150]},
            {"name": "Model B", "values": [120, 140]},
        ]
        svg, summary = mcp_plot.generate_svg_chart(series, chart_type="bar", x_labels=["Prefill", "Decode"])
        self.assertIn("<rect", svg)
        self.assertIn("Model A", svg)
        self.assertIn("Model B", svg)
        self.assertIn("Generated bar chart", summary)

    def test_scatter_chart(self):
        series = [{"name": "Dots", "values": [5, 15, 25]}]
        svg, summary = mcp_plot.generate_svg_chart(series, chart_type="scatter")
        self.assertIn("<circle", svg)
        self.assertIn("Generated scatter chart", summary)

    def test_empty_series_error(self):
        with self.assertRaises(ValueError):
            mcp_plot.generate_svg_chart([])


class TestJsonRpcProtocol(unittest.TestCase):
    def test_initialize(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        ans = mcp_plot.handle_request(req)
        self.assertEqual(ans["id"], 1)
        self.assertEqual(ans["result"]["serverInfo"]["name"], "plot")

    def test_tools_list(self):
        req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        ans = mcp_plot.handle_request(req)
        tools = [t["name"] for t in ans["result"]["tools"]]
        self.assertIn("generate_plot", tools)

    def test_tool_call(self):
        req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "generate_plot",
                "arguments": {
                    "chart_type": "line",
                    "title": "Throughput",
                    "series": [{"name": "GPU", "values": [80, 95, 110]}],
                    "x": ["16K", "32K", "64K"],
                },
            },
        }
        ans = mcp_plot.handle_request(req)
        self.assertFalse(ans["result"]["isError"])
        content = ans["result"]["content"][0]["text"]
        self.assertIn("```xml", content)
        self.assertIn("<svg", content)
        self.assertIn("Throughput", content)


class TestHubIntegration(unittest.TestCase):
    def test_hub_with_plot(self):
        hub = McpHub({"plot": {"command": sys.executable, "args": [str(ROOT / "tools" / "mcp_plot.py")]}})
        hub.start(wait=True)
        try:
            tools = [t["name"] for t in hub.template_tools()]
            self.assertIn("plot__generate_plot", tools)
            res = hub.call(
                "plot__generate_plot",
                {"title": "Test", "series": [{"name": "S1", "values": [1, 2, 3]}]},
            )
            self.assertTrue(res["ok"])
            self.assertIn("<svg", res["text"])
        finally:
            hub.close()


if __name__ == "__main__":
    unittest.main()
