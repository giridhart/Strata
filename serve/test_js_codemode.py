"""serve/test_js_codemode.py - Unit tests for QuickJS JavaScript Code Mode and language selection.

Validates:
- QuickJS execution: basic JavaScript, console.log capturing, expression auto-return.
- MCP proxy bridge: calling `mcp.<server>.<tool>(args)` in JavaScript through McpHub.
- Interactive plots: `render_interactive_plot(fig)` generating Plotly JSON payload.
- State persistence across turns for the same session_id in JavaScript.
- Timeout and error handling: catching while(true) and syntax errors in JavaScript.
- Selection mechanism: launchtime config vs per-request runtime language selection.
"""
from __future__ import annotations

import unittest
from serve.js_sandbox import execute_js_in_sandbox, QuickJSSandbox, HAS_QUICKJS
from serve.mcp import McpHub


class MockMcpHub:
    """Mock McpHub providing a simple test tool."""
    def __init__(self):
        self.settings = {"codemode": True, "codemode_lang": "javascript", "timeout_s": 5.0, "max_result_chars": 20000}
        self.call_log = []

    def call(self, name: str, arguments: dict, cancel=None) -> dict:
        self.call_log.append((name, arguments))
        if name == "math__add":
            a = arguments.get("a", 0)
            b = arguments.get("b", 0)
            return {"ok": True, "text": str(a + b), "result": a + b}
        elif name == "string__greet":
            name_val = arguments.get("name", "world")
            return {"ok": True, "text": f"Hello, {name_val}!"}
        return {"ok": False, "text": f"Unknown tool: {name}"}


@unittest.skipUnless(HAS_QUICKJS, "QuickJS module is required for JS CodeMode tests")
class TestQuickJSCodeMode(unittest.TestCase):
    def setUp(self):
        self.hub = MockMcpHub()

    def test_basic_js_execution_and_expression_return(self):
        """JavaScript expressions at the end must auto-return their evaluated value."""
        code = """
        const a = 15;
        const b = 27;
        a + b;
        """
        res = execute_js_in_sandbox(code, self.hub, session_id="test_basic")
        self.assertTrue(res["ok"])
        self.assertEqual(res["result"], 42)
        self.assertIn("42", res["text"])

    def test_console_log_capturing(self):
        """console.log, console.warn, and console.error output must be captured in text."""
        code = """
        console.log("Calculating step 1...");
        console.warn("Notice: low threshold");
        const val = 100;
        val * 2;
        """
        res = execute_js_in_sandbox(code, self.hub, session_id="test_log")
        self.assertTrue(res["ok"])
        self.assertIn("Calculating step 1...", res["text"])
        self.assertIn("[warn] Notice: low threshold", res["text"])
        self.assertIn("Result: 200", res["text"])
        self.assertEqual(res["result"], 200)

    def test_mcp_proxy_tool_invocation(self):
        """mcp.<server>.<tool>(args) must correctly route to McpHub and return parsed JSON."""
        code = """
        const addRes = mcp.math.add({ a: 18, b: 24 });
        const greetRes = mcp.string.greet({ name: "Alice" });
        ({ addResult: addRes.result, greeting: greetRes.text })
        """
        res = execute_js_in_sandbox(code, self.hub, session_id="test_mcp")
        self.assertTrue(res["ok"])
        self.assertEqual(res["result"]["addResult"], 42)
        self.assertEqual(res["result"]["greeting"], "Hello, Alice!")
        self.assertEqual(len(self.hub.call_log), 2)
        self.assertEqual(self.hub.call_log[0], ("math__add", {"a": 18, "b": 24}))
        self.assertEqual(self.hub.call_log[1], ("string__greet", {"name": "Alice"}))

    def test_js_state_persistence_across_turns(self):
        """Variables declared in a session must persist to subsequent turns."""
        sess = "persist_js_session"
        # Turn 1: define variables and function
        res1 = execute_js_in_sandbox("var x = 100;\nfunction double(n) { return n * 2; }", self.hub, session_id=sess)
        self.assertTrue(res1["ok"])

        # Turn 2: use variables and function defined in Turn 1
        res2 = execute_js_in_sandbox("double(x) + 5", self.hub, session_id=sess)
        self.assertTrue(res2["ok"])
        self.assertEqual(res2["result"], 205)

    def test_interactive_plot_rendering(self):
        """render_interactive_plot(fig) must produce standard plotly JSON schema."""
        code = """
        const fig = {
            data: [{ x: [1, 2, 3], y: [10, 20, 30], type: 'bar' }],
            layout: { title: 'JS Chart' }
        };
        render_interactive_plot(fig);
        """
        res = execute_js_in_sandbox(code, self.hub, session_id="test_plot")
        self.assertTrue(res["ok"])
        self.assertIsNotNone(res.get("plot"))
        self.assertEqual(res["plot"]["type"], "application/vnd.plotly.v1+json")
        self.assertEqual(res["plot"]["content"]["layout"]["title"], "JS Chart")

    def test_loop_timeout_protection(self):
        """Runaway loops in QuickJS must be terminated within the requested timeout_s."""
        code = "while(true) {}"
        res = execute_js_in_sandbox(code, self.hub, timeout_s=0.1, session_id="test_timeout")
        self.assertFalse(res["ok"])
        self.assertIn("timed out", res["text"])

    def test_syntax_error_handling(self):
        """Invalid JavaScript syntax must return clear error message."""
        code = "const a = ;"
        res = execute_js_in_sandbox(code, self.hub, session_id="test_syntax")
        self.assertFalse(res["ok"])
        self.assertIn("SyntaxError", res["text"])


class TestLanguageSelection(unittest.TestCase):
    def setUp(self):
        self.hub = McpHub({}, settings={"codemode": True, "codemode_lang": "javascript"})

    def test_default_tool_is_run_javascript(self):
        """Default codemode tools should offer run_javascript."""
        tools = self.hub.codemode_tools()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "run_javascript")
        self.assertIn("JavaScript (ES6)", tools[0]["description"])

    def test_language_selection_python_offers_run_python(self):
        """When lang='python', codemode_tools should offer run_python."""
        tools = self.hub.codemode_tools(lang="python")
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "run_python")
        self.assertIn("Python", tools[0]["description"])

    def test_hub_settings_language_switch(self):
        """Switching hub.settings['codemode_lang'] switches the default codemode tool."""
        self.hub.settings["codemode_lang"] = "python"
        tools = self.hub.codemode_tools()
        self.assertEqual(tools[0]["name"], "run_python")

    def test_direct_call_run_javascript(self):
        """Calling run_javascript directly executes via QuickJS sandbox."""
        res = self.hub.call("run_javascript", {"code": "3 * 14"})
        self.assertTrue(res["ok"])
        self.assertIn("42", res["text"])

    def test_run_code_with_language_parameter(self):
        """Calling run_code with language='javascript' routes to QuickJS."""
        res = self.hub.call("run_code", {"code": "'hello ' + 'world'", "language": "javascript"})
        self.assertTrue(res["ok"])
        self.assertIn("hello world", res["text"])

    def test_run_code_with_default_lang(self):
        """Calling run_code without language parameter respects default_lang."""
        res = self.hub.call("run_code", {"code": "'default ' + 'lang'"}, default_lang="javascript")
        self.assertTrue(res["ok"])
        self.assertIn("default lang", res["text"])

    def test_native_datetime_in_javascript(self):
        """Standard Date() in JavaScript should evaluate cleanly without needing Python datetime."""
        res = self.hub.call("run_javascript", {"code": "new Date().toISOString()"})
        self.assertTrue(res["ok"])
        self.assertIn("T", res["text"])
        self.assertIn("Z", res["text"])


class TestCodeModeFallbackIsolation(unittest.TestCase):
    """Verifies that an ad-hoc fallback execution does not globally alter session language."""

    def test_with_history_tools_blocks_fallback_runner_leakage(self):
        from serve.server import with_history_tools
        # Request declares run_javascript as the active codemode tool
        tools = [{"name": "run_javascript", "parameters": {}}]
        # Conversation history has a previous turn that fell back to run_python
        req = {
            "messages": [
                {"role": "user", "content": "what time is it?"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "run_python", "arguments": "{}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "2026-11-06T23:00:00Z"},
                {"role": "user", "content": "now plot a chart"}
            ]
        }
        res_tools = with_history_tools(tools, req)
        # run_python must NOT be added to tools; only run_javascript should be present
        names = [t["name"] for t in res_tools]
        self.assertEqual(names, ["run_javascript"])

    def test_steer_codemode_history_injects_steering_note(self):
        from serve.server import steer_codemode_history
        messages = [
            {"role": "user", "content": "what time is it?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "run_python", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "2026-11-06T23:00:00Z"},
            {"role": "user", "content": "now plot a chart"}
        ]
        steered = steer_codemode_history(messages, req_lang="javascript")
        tool_msg = next(m for m in steered if m.get("role") == "tool")
        self.assertIn("2026-11-06T23:00:00Z", tool_msg["content"])
        self.assertIn("[Note: This tool call executed Python as an ad-hoc fallback", tool_msg["content"])
        self.assertIn("The active Code Mode environment for this conversation remains javascript (run_javascript)", tool_msg["content"])

        # Idempotence: running it again does not duplicate the note
        steered_again = steer_codemode_history(steered, req_lang="javascript")
        tool_msg_again = next(m for m in steered_again if m.get("role") == "tool")
        self.assertEqual(tool_msg["content"], tool_msg_again["content"])

    def test_steer_codemode_history_clean_when_no_fallback(self):
        from serve.server import steer_codemode_history
        messages = [
            {"role": "user", "content": "calculate 2 + 2"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "run_javascript", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "4"}
        ]
        steered = steer_codemode_history(messages, req_lang="javascript")
        tool_msg = next(m for m in steered if m.get("role") == "tool")
        self.assertEqual(tool_msg["content"], "4")


if __name__ == "__main__":
    unittest.main()
