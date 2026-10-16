"""Tests for SYCL codebase invariants (doorbell cache-bypass, bounded spins, queue checks).
Runs offline without a GPU.

    python tools/test_sycl_invariants.py
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

class SyclKernelInvariantsTest(unittest.TestCase):
    def test_no_unbounded_flag_spins_in_sycl_kernels(self):
        """Device-side flag waits must use strata::sys_load and a bounded loop (spin_max) on Intel GPUs."""
        files = list((ROOT / "src/kernels/sycl").glob("*.cpp"))
        self.assertTrue(len(files) > 0, "No SYCL kernel files found")

        # Pattern matching raw unbounded while loop on pointer dereference: while (*flag < value)
        unbounded_spin_pat = re.compile(r"while\s*\(\s*\*\w+\s*(?:<|!=|==|>)\s*\w+\s*\)\s*(?:strata_spin_pause\(\)\s*;|\{\s*\})")

        violations = []
        for f in files:
            text = f.read_text(encoding="utf-8")
            for i, line in enumerate(text.splitlines(), 1):
                if unbounded_spin_pat.search(line):
                    violations.append(f"{f.name}:{i}: {line.strip()}")

        self.assertEqual(violations, [], f"Found raw unbounded spin loops in SYCL kernels:\n" + "\n".join(violations))

    def test_no_inverted_ext_oneapi_empty_checks(self):
        """ext_oneapi_empty() returns true when the queue is finished. Checking '!empty && *seq < want' prematurely errors while the GPU is computing."""
        files = list((ROOT / "src/core/sycl").glob("*.cpp"))
        self.assertTrue(len(files) > 0, "No SYCL core files found")

        inverted_pat = re.compile(r"if\s*\(\s*!is_empty\s*&&\s*\*seq\s*<\s*want\s*\)")

        violations = []
        for f in files:
            text = f.read_text(encoding="utf-8")
            for i, line in enumerate(text.splitlines(), 1):
                if inverted_pat.search(line):
                    violations.append(f"{f.name}:{i}: {line.strip()}")

        self.assertEqual(violations, [], f"Found inverted ext_oneapi_empty checks:\n" + "\n".join(violations))

    def test_doorbell_buffers_use_polled_allocator(self):
        """Mapped doorbell flags polled by GPU must use strata::host_malloc_polled (Level-Zero uncached) instead of plain malloc_host."""
        verify_cpp = (ROOT / "src/core/sycl/verify.cpp").read_text(encoding="utf-8")
        layer_cpp = (ROOT / "src/core/sycl/layer.cpp").read_text(encoding="utf-8")

        self.assertIn("host_malloc_polled", verify_cpp, "src/core/sycl/verify.cpp must use host_malloc_polled for mapped doorbells")
        self.assertIn("host_malloc_polled", layer_cpp, "src/core/sycl/layer.cpp must use host_malloc_polled for doorbell_init")

    def test_no_wait_calls_inside_graph_recording_helpers(self):
        """Synchronous .wait() calls on events/queues inside command graph recording break SYCL command graph capture."""
        ple_cpp = (ROOT / "src/kernels/sycl/ple.cpp").read_text(encoding="utf-8")
        self.assertNotIn("q->memcpy(stage + (size_t)h * HD, src, bytes).wait()", ple_cpp,
                         "Asynchronous device memcpy inside PLE staging must not call .wait() during command graph capture")


if __name__ == "__main__":
    unittest.main()
