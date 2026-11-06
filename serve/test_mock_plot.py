"""End-to-end test of interactive Plotly plotting in the web UI without running full server/GPU.

    python -m unittest serve.test_mock_plot -v
    python -m serve.test_mock_plot --serve      # opens interactive browser test server
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from serve.frontend import ChatTemplate  # noqa: E402
from serve.mcp import McpHub  # noqa: E402
from serve.server import ByteTokenizer, Service, serve  # noqa: E402
from serve.test_mcp import ScriptedEngine, call_script  # noqa: E402

SAMPLE_PLOT_CODE = """df = pd.DataFrame({"Category": ["A", "B", "C"], "Values": [10, 25, 18]})
fig = px.bar(df, x="Category", y="Values", title="Interactive Sales Chart")
render_interactive_plot(fig)"""


class MockPlotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        first_turn = call_script("run_python", code=SAMPLE_PLOT_CODE)
        second_turn = "</think>\n\nHere is your interactive chart."

        cls.engine = ScriptedEngine(tok, [first_turn, second_turn])
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.svc.mcp = McpHub({})
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_plotly_vendor_asset_served(self):
        """Vendored plotly.min.js must be accessible via HTTP GET."""
        req = urllib.request.Request(f"{self.base}/web/vendor/plotly/plotly.min.js")
        with urllib.request.urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("javascript", resp.headers.get("Content-Type", ""))
            data = resp.read(1024)
            self.assertGreater(len(data), 0)

    def test_app_js_includes_tool_plots(self):
        """Web app script must include toolPlots, hydrateToolPlots, and DOM preservation hooks."""
        req = urllib.request.Request(f"{self.base}/web/app.js")
        with urllib.request.urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, 200)
            content = resp.read().decode("utf-8")
            self.assertIn("toolPlots", content)
            self.assertIn("hydrateToolPlots", content)
            self.assertIn("plotCache", content)
            self.assertIn("savedPlots", content)
            self.assertIn("replaceWith", content)

    def test_mock_chat_completion_streams_plotly_result(self):
        """Simulated model tool call generates Plotly JSON and streams SSE result with plot object."""
        req = urllib.request.Request(
            f"{self.base}/v1/chat/completions",
            data=json.dumps({
                "model": "qwen",
                "messages": [{"role": "user", "content": "draw a bar chart"}],
                "stream": True,
                "strata_mcp": True,
                "codemode": True
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            self.assertEqual(resp.status, 200)
            events = []
            for line in resp:
                line_str = line.decode("utf-8").strip()
                if line_str.startswith("data: ") and not line_str.endswith("[DONE]"):
                    chunk = json.loads(line_str[6:])
                    if "strata_mcp" in chunk:
                        events.append(chunk["strata_mcp"])

        result_events = [e for e in events if e.get("event") == "result"]
        self.assertGreaterEqual(len(result_events), 1, "Expected at least one tool result event")

        res = result_events[0]
        self.assertTrue(res.get("ok"), f"Tool result failed: {res.get('text')}")
        self.assertIn("plot", res, "Result event must contain 'plot' field")

        plot = res["plot"]
        self.assertEqual(plot.get("type"), "application/vnd.plotly.v1+json")
        self.assertIn("content", plot)

        content = plot["content"]
        self.assertIn("data", content)
        self.assertIn("layout", content)
        self.assertEqual(content["layout"].get("title", {}).get("text"), "Interactive Sales Chart")
        self.assertGreaterEqual(len(content["data"]), 1)
        self.assertEqual(content["data"][0].get("type"), "bar")

    def test_codemode_import_px_and_trailing_print_preserves_plot(self):
        """Calling run_python with import px, dependencies=['px'], and trailing print retains the plot."""
        code = (
            "import px\n"
            "df = pd.DataFrame({'x': [1, 2], 'y': [10, 20]})\n"
            "fig = px.scatter(df, x='x', y='y', title='Scatter Test')\n"
            "render_interactive_plot(fig)\n"
            "print('Plot created successfully')\n"
        )
        tok = ByteTokenizer()
        first_turn = call_script("run_python", code=code, dependencies=["px"])
        second_turn = "</think>\n\nHere is your scatter plot."
        engine = ScriptedEngine(tok, [first_turn, second_turn])
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.mcp = McpHub({})
        httpd = serve(svc, port=0)
        port = httpd.server_address[1]
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps({
                    "model": "qwen",
                    "messages": [{"role": "user", "content": "draw a scatter plot"}],
                    "stream": True,
                    "strata_mcp": True,
                    "codemode": True
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                self.assertEqual(resp.status, 200)
                events = []
                for line in resp:
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: ") and not line_str.endswith("[DONE]"):
                        chunk = json.loads(line_str[6:])
                        if "strata_mcp" in chunk:
                            events.append(chunk["strata_mcp"])

            results = [e for e in events if e.get("event") == "result"]
            self.assertGreaterEqual(len(results), 1)
            res = results[0]
            self.assertTrue(res.get("ok"), f"Tool call failed: {res.get('text')}")
            self.assertIn("plot", res)
            self.assertEqual(res["plot"]["content"]["layout"]["title"]["text"], "Scatter Test")
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_codemode_fig_show_preserves_plot(self):
        """Calling run_python with standard fig.show() routes to plot capture."""
        code = (
            "df = pd.DataFrame({'x': ['A', 'B'], 'y': [5, 15]})\n"
            "fig = px.bar(df, x='x', y='y', title='Bar Test')\n"
            "fig.show()\n"
        )
        tok = ByteTokenizer()
        first_turn = call_script("run_python", code=code)
        second_turn = "</think>\n\nHere is your chart."
        engine = ScriptedEngine(tok, [first_turn, second_turn])
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.mcp = McpHub({})
        httpd = serve(svc, port=0)
        port = httpd.server_address[1]
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps({
                    "model": "qwen",
                    "messages": [{"role": "user", "content": "draw a bar chart"}],
                    "stream": True,
                    "strata_mcp": True,
                    "codemode": True
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                self.assertEqual(resp.status, 200)
                events = []
                for line in resp:
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: ") and not line_str.endswith("[DONE]"):
                        chunk = json.loads(line_str[6:])
                        if "strata_mcp" in chunk:
                            events.append(chunk["strata_mcp"])

            results = [e for e in events if e.get("event") == "result"]
            self.assertGreaterEqual(len(results), 1)
            res = results[0]
            self.assertTrue(res.get("ok"), f"Tool call failed: {res.get('text')}")
            self.assertIn("plot", res)
            self.assertEqual(res["plot"]["content"]["layout"]["title"]["text"], "Bar Test")
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_codemode_subsequent_non_plot_turn_has_no_plot_field(self):
        """A subsequent tool call that does not generate a plot must not leak the previous turn's plot."""
        tok = ByteTokenizer()
        plot_turn = call_script("run_python", code="fig = px.bar(x=[1, 2], y=[3, 4])\nrender_interactive_plot(fig)")
        answer1 = "</think>\n\nHere is your chart."
        non_plot_turn = call_script("run_python", code="print('Non-plot computation: 42')")
        answer2 = "</think>\n\nHere is the calculation result."

        engine = ScriptedEngine(tok, [plot_turn, answer1, non_plot_turn, answer2], max_context=32768)
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.mcp = McpHub({})
        httpd = serve(svc, port=0)
        port = httpd.server_address[1]
        try:
            # 1. First request generates plot
            req1 = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps({
                    "model": "qwen",
                    "messages": [{"role": "user", "content": "draw plot"}],
                    "stream": True,
                    "strata_mcp": True,
                    "codemode": True
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req1, timeout=15) as resp:
                self.assertEqual(resp.status, 200)
                events1 = []
                for line in resp:
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: ") and not line_str.endswith("[DONE]"):
                        chunk = json.loads(line_str[6:])
                        if "strata_mcp" in chunk:
                            events1.append(chunk["strata_mcp"])

            results1 = [e for e in events1 if e.get("event") == "result"]
            self.assertEqual(len(results1), 1)
            self.assertIn("plot", results1[0], "First turn must have plot")

            # 2. Second request in new conversation does not ask for plot
            req2 = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps({
                    "model": "qwen",
                    "messages": [{"role": "user", "content": "compute numbers"}],
                    "stream": True,
                    "strata_mcp": True,
                    "codemode": True
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req2, timeout=15) as resp:
                self.assertEqual(resp.status, 200)
                events2 = []
                for line in resp:
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: ") and not line_str.endswith("[DONE]"):
                        chunk = json.loads(line_str[6:])
                        if "strata_mcp" in chunk:
                            events2.append(chunk["strata_mcp"])

            results2 = [e for e in events2 if e.get("event") == "result"]
            self.assertEqual(len(results2), 1)
            self.assertTrue(results2[0].get("ok"))
            self.assertNotIn("plot", results2[0], "Second turn must NOT contain plot from previous conversation")
            self.assertIn("Non-plot computation: 42", results2[0].get("text", ""))
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_chat_id_isolates_session_state(self):
        """Requests with different chat_id parameters operate in isolated Python runner sessions."""
        tok = ByteTokenizer()
        t1 = call_script("run_python", code="my_secret_var = 'isolated_secret'")
        ans1 = "</think>\n\nStored."
        t2 = call_script("run_python", code="print('found:', 'my_secret_var' in locals() or 'my_secret_var' in globals())")
        ans2 = "</think>\n\nChecked."

        engine = ScriptedEngine(tok, [t1, ans1, t2, ans2], max_context=32768)
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.mcp = McpHub({})
        httpd = serve(svc, port=0)
        port = httpd.server_address[1]
        try:
            # 1. Chat 1 defines my_secret_var
            req1 = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps({
                    "model": "qwen",
                    "messages": [{"role": "user", "content": "store secret"}],
                    "stream": True,
                    "strata_mcp": True,
                    "codemode": True,
                    "chat_id": "chat_session_alpha"
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req1, timeout=15) as resp:
                self.assertEqual(resp.status, 200)
                for _ in resp:
                    pass

            # 2. Chat 2 in a different chat_id should NOT see my_secret_var
            req2 = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps({
                    "model": "qwen",
                    "messages": [{"role": "user", "content": "check secret"}],
                    "stream": True,
                    "strata_mcp": True,
                    "codemode": True,
                    "chat_id": "chat_session_beta"
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req2, timeout=15) as resp:
                self.assertEqual(resp.status, 200)
                events2 = []
                for line in resp:
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: ") and not line_str.endswith("[DONE]"):
                        chunk = json.loads(line_str[6:])
                        if "strata_mcp" in chunk:
                            events2.append(chunk["strata_mcp"])

            results2 = [e for e in events2 if e.get("event") == "result"]
            self.assertEqual(len(results2), 1)
            self.assertTrue(results2[0].get("ok"))
            self.assertIn("found: False", results2[0].get("text", ""))
        finally:
            httpd.shutdown()
            httpd.server_close()


def run_interactive_server(port: int = 8080):
    """Launches a standalone mock server for instant browser previewing."""
    tok = ByteTokenizer()
    first_turn = call_script("run_python", code=SAMPLE_PLOT_CODE)
    second_turn = "</think>\n\nHere is your interactive chart. You can zoom, pan, hover, or reset the axes."

    engine = ScriptedEngine(tok, [first_turn, second_turn])
    svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    svc.mcp = McpHub({})
    httpd = serve(svc, port=port)
    actual_port = httpd.server_address[1]

    url = f"http://127.0.0.1:{actual_port}"
    print("\n=======================================================")
    print(f"  Strata Mock Plotting Server running at: {url}")
    print("  - No GPU or model weights needed (< 0.1s startup)")
    print(f"  - Open {url} in your browser")
    print("  - Send any message to trigger the interactive Plotly chart")
    print("  - Press Ctrl+C to stop")
    print("=======================================================\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping mock server...")
        httpd.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test or run Strata mock plot server")
    parser.add_argument("--serve", action="store_true", help="Start standalone mock web server")
    parser.add_argument("--port", type=int, default=8080, help="Port for standalone server (default: 8080)")
    args, remaining = parser.parse_known_args()

    if args.serve:
        run_interactive_server(args.port)
    else:
        unittest.main(argv=[sys.argv[0]] + remaining)
