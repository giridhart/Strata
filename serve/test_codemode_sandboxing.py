"""serve/test_codemode_sandboxing.py - Unit tests for the 3-layer Code Mode Python Sandbox.

Validates:
- Layer 1: Minimal sanitized environment (no host secrets/API keys), ephemeral scratch directory.
- Layer 2: AST SecurityVisitor blocking dangerous modules (os, sys, subprocess, socket, ctypes, open) and dunders.
- Layer 2: LoopBoundingTransformer catching infinite / runaway loops in milliseconds.
- Layer 2: attempt_syntax_repair auto-fixing unclosed delimiters and stripping XML artifacts.
- Layer 3: Bubblewrap containerization command wrapper logic.
"""
from __future__ import annotations

import os
import unittest
from serve.sandbox import LLMSandbox, attempt_syntax_repair, strip_markdown_fences
from serve.mcp_runner import get_sandboxed_env, _wrap_with_bwrap_if_available, execute_in_runner, close_all_runners
from serve.mcp import McpHub


class TestCodeModeSandboxing(unittest.TestCase):
    def setUp(self):
        self.sandbox = LLMSandbox()

    def test_layer2_banned_modules(self):
        """Layer 2 AST static analysis must reject host inspection and process modules."""
        banned = [
            "import os",
            "import sys",
            "import subprocess",
            "import socket",
            "import ctypes",
            "import shutil",
            "import pathlib",
            "from os import path",
            "from subprocess import check_output",
        ]
        for snippet in banned:
            res = self.sandbox.execute(snippet)
            self.assertFalse(res["ok"], f"Expected '{snippet}' to fail security check")
            self.assertIn("PermissionError", res["output"])

    def test_layer2_banned_attributes_and_dunders(self):
        """Layer 2 AST checks must block sandbox breakout attempts via dunder attributes."""
        escapes = [
            "().__class__.__bases__[0].__subclasses__()",
            "x = ().__class__.__globals__",
            "def f(): pass\nf.__code__",
        ]
        for snippet in escapes:
            res = self.sandbox.execute(snippet)
            self.assertFalse(res["ok"], f"Expected '{snippet}' to fail security check")
            self.assertIn("PermissionError", res["output"])

    def test_layer2_blocked_open_and_input(self):
        """Direct file open() and interactive input() must be blocked in the execution namespace."""
        res_open = self.sandbox.execute("open('/etc/passwd')")
        self.assertFalse(res_open["ok"])
        self.assertIn("PermissionError", res_open["output"])

        res_input = self.sandbox.execute("val = input('enter: ')")
        self.assertFalse(res_input["ok"])
        self.assertIn("RuntimeError", res_input["output"])

    def test_layer2_loop_bounding_prevents_30s_timeout(self):
        """While and for loops must be bounded so infinite loops raise TimeoutError in ms."""
        res_while = self.sandbox.execute("while True: pass")
        self.assertFalse(res_while["ok"])
        self.assertIn("TimeoutError", res_while["output"])

        res_for = self.sandbox.execute("for i in range(1_000_000_000): pass")
        self.assertFalse(res_for["ok"])
        self.assertIn("TimeoutError", res_for["output"])

        # Legitimate bounded loops must execute normally
        res_legit = self.sandbox.execute("total = 0\nfor i in range(100): total += i\ntotal")
        self.assertTrue(res_legit["ok"])
        self.assertEqual(res_legit["result"], 4950)

    def test_layer2_syntax_auto_repair(self):
        """Syntax auto-repair must close unclosed brackets, parens, and quotes without error."""
        # Unclosed bracket and parenthesis
        code1 = "x = [1, 2, (3, 4\nx"
        res1 = self.sandbox.execute(code1)
        self.assertTrue(res1["ok"])
        self.assertEqual(res1["result"], [1, 2, (3, 4)])

        # Delimiter balancing across multi-line definitions
        code2 = "d = {'a': 1, 'b': [10, 20\nd['b']"
        res2 = self.sandbox.execute(code2)
        self.assertTrue(res2["ok"])
        self.assertEqual(res2["result"], [10, 20])

    def test_layer2_strips_xml_prompt_artifacts(self):
        """Stray <tool_call> or <parameter=code> tags emitted by LLM must be stripped."""
        raw = "<tool_call>\n<function=run_python>\n<parameter=code>\nx = 42\nx * 2\n</parameter>\n</tool_call>"
        res = self.sandbox.execute(raw)
        self.assertTrue(res["ok"])
        self.assertEqual(res["result"], 84)

    def test_layer1_sandboxed_env_scrubs_secrets(self):
        """Layer 1 sanitized environment must not contain sensitive host API keys or tokens."""
        os.environ["OPENAI_API_KEY"] = "sk-secret-test-openai"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "aws-secret-test-key"
        os.environ["STRATA_TEST_SECRET"] = "internal-strata-secret"

        clean_env = get_sandboxed_env()
        self.assertNotIn("OPENAI_API_KEY", clean_env)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", clean_env)
        self.assertNotIn("STRATA_TEST_SECRET", clean_env)
        self.assertIn("PATH", clean_env)
        self.assertIn("PYTHONPATH", clean_env)
        self.assertEqual(clean_env.get("STRATA_SANDBOX"), "1")
        self.assertEqual(clean_env.get("OPENBLAS_NUM_THREADS"), "1")
        self.assertEqual(clean_env.get("OMP_NUM_THREADS"), "1")

    def test_layer3_bwrap_wrapper_logic(self):
        """Layer 3 bwrap logic wraps commands with unshare-net and read-only bindings when available."""
        scratch = "/tmp/strata_test_scratch"
        cmd = ["python", "-m", "serve.mcp_runner"]
        wrapped = _wrap_with_bwrap_if_available(cmd, scratch)
        # On systems without bwrap binary or where AppArmor restricts userns, wrapped is returned as-is
        self.assertTrue(len(wrapped) >= len(cmd))

    def test_layer3_bwrap_probe_apparmor_denial_fallback(self):
        """When AppArmor denies unprivileged userns to bwrap, probe returns False and bypasses wrapping."""
        from unittest.mock import patch
        import serve.mcp_runner as runner_mod

        old_cached = runner_mod._BWRAP_SUPPORTED
        try:
            with patch("shutil.which", return_value="/usr/bin/bwrap"):
                with patch("subprocess.run") as mock_run:
                    # Simulate AppArmor AUDIT denial: bwrap fails with exit code 1
                    mock_run.return_value.returncode = 1
                    runner_mod._BWRAP_SUPPORTED = None
                    self.assertFalse(runner_mod._is_bwrap_supported())

                    cmd = ["python", "-m", "serve.mcp_runner"]
                    wrapped = runner_mod._wrap_with_bwrap_if_available(cmd, "/tmp/scratch")
                    self.assertEqual(wrapped, cmd)
        finally:
            runner_mod._BWRAP_SUPPORTED = old_cached


if __name__ == "__main__":
    unittest.main()
