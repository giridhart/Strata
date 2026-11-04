"""serve/test_sandbox.py - Unit tests for LLMSandbox and AST execution.

    python -m unittest serve.test_sandbox -v
"""
from __future__ import annotations

import unittest
from serve.sandbox import LLMSandbox


class TestLLMSandbox(unittest.TestCase):
    """Test the Zero-Verbosity AST Execution Sandbox."""

    def setUp(self):
        self.sandbox = LLMSandbox()

    def test_pre_imported_globals_and_auto_return(self):
        # SymPy is pre-imported as `sp`, expression auto-returns without print()
        res = self.sandbox.execute("x = sp.Symbol('x')\neq = x**2 - 9\nsp.solve(eq, x)", session_id="s1")
        self.assertTrue(res["ok"], res.get("error"))
        self.assertIn("[-3, 3]", res["output"])

    def test_session_state_persistence(self):
        # Turn 1: define variables and symbols
        r1 = self.sandbox.execute("x = sp.Symbol('x')\nval = 42", session_id="persisted_session")
        self.assertTrue(r1["ok"])

        # Turn 2: use variables defined in Turn 1 without re-declaring them
        r2 = self.sandbox.execute("val + 8", session_id="persisted_session")
        self.assertTrue(r2["ok"])
        self.assertEqual(r2["output"], "50")

        # Turn 3: compute derivative using Symbol x defined in Turn 1
        r3 = self.sandbox.execute("sp.diff(sp.sin(x) * sp.exp(x), x)", session_id="persisted_session")
        self.assertTrue(r3["ok"])
        self.assertIn("exp(x)*sin(x) + exp(x)*cos(x)", r3["output"])

    def test_session_isolation(self):
        # Session A defines variable 'secret'
        self.sandbox.execute("secret = 12345", session_id="session_a")

        # Session B cannot see variable 'secret'
        r_b = self.sandbox.execute("secret", session_id="session_b")
        self.assertFalse(r_b["ok"])
        self.assertIn("NameError", r_b["output"])

    def test_markdown_code_fences(self):
        # Code wrapped in markdown fences runs seamlessly
        code = "```python\na = 15\nb = 25\na + b\n```"
        res = self.sandbox.execute(code)
        self.assertTrue(res["ok"])
        self.assertEqual(res["output"], "40")

    def test_pre_injected_symbolic_and_logic_tools(self):
        # Pre-injected tools callable directly without imports or 'mcp.' prefixes
        res1 = self.sandbox.execute("verify_equivalence('x^2 - 1', '(x - 1)*(x + 1)')")
        self.assertTrue(res1["ok"])
        self.assertIn('"equivalent": true', res1["output"])

        res2 = self.sandbox.execute("solve_constraints(['x', 'y'], ['x + y == 10', 'x - y == 2'])")
        self.assertTrue(res2["ok"])
        self.assertIn('"x": 6', res2["output"])
        self.assertIn('"y": 4', res2["output"])

    def test_syntax_error_reporting(self):
        res = self.sandbox.execute("def broken(:\n    pass")
        self.assertFalse(res["ok"])
        self.assertIn("SyntaxError", res["output"])

    def test_plotly_interactive_render(self):
        # User requested example: px.bar with DataFrame and render_interactive_plot
        code = (
            'df = pd.DataFrame({"Category": ["A", "B", "C"], "Values": [10, 20, 15]})\n'
            'fig = px.bar(df, x="Category", y="Values", title="Sample Chart")\n'
            'render_interactive_plot(fig)'
        )
        res = self.sandbox.execute(code, session_id="plotly_session")
        self.assertTrue(res["ok"], res.get("error"))
        result_dict = res.get("result")
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict.get("type"), "application/vnd.plotly.v1+json")
        content = result_dict.get("content", {})
        self.assertIn("data", content)
        self.assertIn("layout", content)
        self.assertIn("application/vnd.plotly.v1+json", res["output"])

    def test_plotly_auto_render_figure_expression(self):
        # Auto-return a Plotly Figure without explicit render_interactive_plot call
        code = (
            'df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})\n'
            'fig = px.line(df, x="x", y="y", title="Line Chart")\n'
            'fig'
        )
        res = self.sandbox.execute(code, session_id="plotly_auto")
        self.assertTrue(res["ok"], res.get("error"))
        result_dict = res.get("result")
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict.get("type"), "application/vnd.plotly.v1+json")

    def test_plotly_graph_objects(self):
        code = (
            'fig = go.Figure(data=go.Bar(x=["Jan", "Feb"], y=[100, 200]))\n'
            'render_interactive_plot(fig)'
        )
        res = self.sandbox.execute(code)
        self.assertTrue(res["ok"], res.get("error"))
        self.assertEqual(res["result"]["type"], "application/vnd.plotly.v1+json")

    def test_pseudo_imports_px_pd_np_sp_go(self):
        # LLMs often write `import px`, `import pd`, etc. due to fine-tuned import reflexes
        code = (
            "import px\n"
            "import pd\n"
            "import np\n"
            "import sp\n"
            "import go\n"
            "from px import bar\n"
            "df = pd.DataFrame({'x': np.array([1, 2]), 'y': [10, 20]})\n"
            "sym = sp.Symbol('val')\n"
            "fig = bar(df, x='x', y='y')\n"
            "render_interactive_plot(fig)"
        )
        res = self.sandbox.execute(code)
        self.assertTrue(res["ok"], res.get("error"))
        self.assertEqual(res["result"]["type"], "application/vnd.plotly.v1+json")

    def test_render_interactive_plot_error_handling(self):
        from serve.sandbox import render_interactive_plot
        import pandas as pd

        # Passing None
        with self.assertRaises(ValueError):
            render_interactive_plot(None)

        # Passing a DataFrame directly instead of a figure
        df = pd.DataFrame({"a": [1, 2]})
        with self.assertRaises(TypeError) as ctx:
            render_interactive_plot(df)
        self.assertIn("DataFrame", str(ctx.exception))

        # Passing a string
        with self.assertRaises(TypeError):
            render_interactive_plot("invalid")

        # Passing a number
        with self.assertRaises(TypeError):
            render_interactive_plot(123)


if __name__ == "__main__":
    unittest.main()
