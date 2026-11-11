"""Tests for tools/strata_bench.py with a fake engine behind a real HTTP server: the suite runner, the redaction, the report
writing and the comparison.  Light: no GPU, no Strata server, a few seconds."""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath, PureWindowsPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_bench as B  # noqa: E402


def text_of(messages) -> str:
    return "".join(m.get("content") or "" for m in messages)


class FakeStrata(ThreadingHTTPServer):
    """Speaks the parts of the Strata API the benchmark reads.  The conversation cache is the longest common prefix with the
    previous prompt (4 characters = 1 token)."""
    daemon_threads = True

    def __init__(self, ctx=32768, fail_after=None, runtime_payload=None):
        super().__init__(("127.0.0.1", 0), Handler)
        self.ctx, self.prev, self.requests, self.fail_after = ctx, "", 0, fail_after
        self.runtime_payload = runtime_payload
        self.lock = threading.Lock()
        threading.Thread(target=self.serve_forever, daemon=True).start()

    def stop(self):
        self.shutdown()
        self.server_close()

    @property
    def base(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        s = self.server
        if self.path == "/health":
            self.send_json({"status": "ok", "service": "strata", "loaded": True})
        elif self.path == "/v1/models":
            self.send_json({"object": "list", "data": [{"id": "fake", "meta": {"n_ctx": s.ctx}}]})
        elif self.path == "/props":
            self.send_json({"total_slots": 4, "model_alias": "fake-model", "build_info": "Strata 9.9.9"})
        elif self.path == "/v1/status":
            st = {"machine": {"gpu": {"name": "NVIDIA GeForce RTX 9999", "total_mib": 12288}, "ram": {"total_gib": 64.0}},
                  "engine": "9.9.9", "vram": {"free_mib": 1000}}
            if s.runtime_payload is not None:
                st["runtime"] = s.runtime_payload
            self.send_json(st)
        elif self.path == "/v1/runtime":
            if s.runtime_payload is not None:
                self.send_json(s.runtime_payload)
            else:
                self.send_json({}, 404)
        elif self.path == "/metrics":
            self.send_json({"engine": {"version": "9.9.9", "max_context": s.ctx, "model_path": r"C:\Users\Someone\models\x.gguf"},
                            "hardware_static": {"gpu_name": "NVIDIA GeForce RTX 9999", "cpu_name": "Fake CPU 9", "cores": 8, "threads": 16}})
        else:
            self.send_json({}, 404)

    def do_POST(self):
        s = self.server
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = text_of(req["messages"])
        with s.lock:
            s.requests += 1
            n = s.requests
            common = 0
            for x, y in zip(s.prev, text):
                if x != y:
                    break
                common += 1
            s.prev = text
        if s.fail_after is not None and n > s.fail_after:
            self.send_json({"error": "boom"}, 500)
            return
        prompt = max(1, len(text) // 4)
        cached = min(prompt, common // 4)
        gen = min(req.get("max_tokens", 16), 12)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def chunk(delta, finish=None, **extra):
            d = {"id": "x", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}
            self.wfile.write(b"data: " + json.dumps(d).encode() + b"\n\n")
            self.wfile.flush()
        chunk({"role": "assistant"})
        for _ in range(gen):
            chunk({"content": "word "})
        chunk({}, "stop", usage={"prompt_tokens": prompt, "completion_tokens": gen, "total_tokens": prompt + gen,
                                 "prompt_tokens_details": {"cached_tokens": cached}},
              timings={"cache_n": cached, "prompt_n": prompt - cached, "prompt_ms": 100.0,
                       "prompt_per_second": 1000.0 + n, "predicted_n": gen, "predicted_ms": 200.0, "predicted_per_second": 60.0})
        self.wfile.write(b"data: [DONE]\n\n")


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.srv = FakeStrata()
        self.addCleanup(self.srv.stop)

    def suite(self, **kw):
        return B.Suite(self.srv.base, "", kw.pop("ctx", 32768), kw.pop("runs", 3), False, log=lambda *a: None, **kw)

    def test_full_suite_shapes_and_medians(self):
        out = self.suite(want_clients=3).run()
        self.assertEqual(out["failures"], [])
        self.assertEqual(out["chat"]["decode_tps"]["median"], 60.0)
        self.assertEqual(out["chat"]["decode_tps"]["n"], 9)                 # 3 questions x 3 runs
        self.assertEqual(set(out["prompts"]), {"4K", "16K"})
        for v in out["prompts"].values():
            self.assertEqual(v["prompt_tps"]["n"], 3)
            self.assertLess(v["reused_tokens"]["median"], v["prompt_tokens"]["median"] / 100)   # a fresh tag at the start: next to nothing reused
        p4, p16 = (out["prompts"][k]["prompt_tokens"]["median"] for k in ("4K", "16K"))
        self.assertAlmostEqual(p16 / p4, 4, delta=0.6)                      # the sizes are calibrated, not guessed
        self.assertEqual(out["clients"]["clients"], 3)
        self.assertEqual(out["clients"]["errors"], 0)
        self.assertEqual(out["clients"]["rounds"][0]["tokens"], 3 * 12)

    def test_agent_session_reuses_the_conversation(self):
        out = self.suite(runs=2).run()
        ag = out["agent"]
        self.assertEqual(len(ag["sessions"]), 2)
        turns = ag["sessions"][0]
        self.assertEqual([t["turn"] for t in turns], list(range(B.AGENT_TURNS + 1)))
        self.assertEqual(turns[0]["reused_tokens"], 0)
        for t in turns[1:]:
            self.assertGreater(t["reused_tokens"], 0)
            self.assertLess(t["read_tokens"], t["prompt_tokens"])
        self.assertGreater(ag["hit_pct"]["median"], 50)
        self.assertLess(ag["reread_tokens"]["median"], ag["new_tokens"]["median"])      # new file text dominates, not re-reading
        self.assertGreater(turns[-1]["prompt_tokens"], turns[1]["prompt_tokens"])    # the conversation grows

    def test_small_context_skips_the_long_prompts(self):
        self.srv.ctx = 8192
        out = self.suite(ctx=8192, runs=1, want_long=True).run()
        self.assertIn("4K", out["prompts"])
        self.assertEqual(set(out["skipped"]), {"16K", "32K", "agent"})
        self.assertIn("below ~10K", out["skipped"]["agent"])
        self.assertEqual(out["agent"], {})                                  # no request was sent for the agent session

    def test_agent_runs_when_the_context_fits_it(self):
        out = self.suite(ctx=16384, runs=1).run()
        self.assertNotIn("agent", out["skipped"])
        self.assertTrue(out["agent"]["sessions"])

    def test_failures_are_reported_not_hidden(self):
        self.srv.fail_after = 3                  # warm-up and calibration succeed, then every request fails
        out = self.suite(runs=1).run()
        self.assertTrue(out["failures"])
        self.assertTrue(all("HTTP 500" in f["error"] for f in out["failures"]))
        self.assertIsNone(out["chat"]["decode_tps"])


class RedactionTests(unittest.TestCase):
    def test_paths_names_ips_and_keys(self):
        red = B.redactor(extra=["Jane Roe Laptop"], key="sk-secret-123456")
        user = os.environ.get("USERNAME") or os.environ.get("USER") or "tester"
        cases = [r"C:\Users\someone_else\models\a.gguf", "/home/bob/strata/x.json", "/Users/alice/Library",
                 "peer 192.168.1.20 and 10.0.0.7 and 100.101.102.103", "Authorization: Bearer abcdefghijkl12345",
                 "key sk-secret-123456 here", "name Jane Roe Laptop"]
        text = red("\n".join(cases))
        for s in ("someone_else", "bob", "alice", "192.168.1.20", "10.0.0.7", "100.101.102.103", "abcdefghijkl12345",
                  "sk-secret-123456", "Jane Roe Laptop"):
            self.assertNotIn(s, text)
        self.assertIn("<user>", text)
        self.assertIn("<ip>", text)
        self.assertEqual(red("127.0.0.1:8080 and version 0.1.40.3 and driver 32.0.15.7"), "127.0.0.1:8080 and version 0.1.40.3 and driver 32.0.15.7")
        self.assertNotIn(str(B.ROOT), red(f"loading {B.ROOT / 'strata-x.json'}"))
        self.assertIn("<strata>", red(f"loading {B.ROOT / 'strata-x.json'}"))
        self.assertNotIn(str(Path.home()), red(f"in {Path.home() / 'data'}"))
        if len(user) >= 4:
            self.assertNotIn(user, red(f"hello {user} world"))

    def test_a_folder_under_a_short_home_keeps_its_own_placeholder(self):
        # issue #3326 with a fixed machine (user name "dgbox", the repository directly under the home folder):
        # a name is wrapped in word bounds, which makes its pattern longer than the pattern of the folder
        # holding it, so the name won and the report kept the folder layout instead of <strata> and <home>.
        def red(root, home, user, host, tmp):
            with mock.patch.object(B, "ROOT", root), mock.patch.object(B.Path, "home", return_value=home), \
                 mock.patch.object(B.getpass, "getuser", return_value=user), \
                 mock.patch.object(B.socket, "gethostname", return_value=host), \
                 mock.patch.object(B.platform, "node", return_value=host), \
                 mock.patch.object(B.tempfile, "gettempdir", return_value=tmp):
                return B.redactor()

        linux = red(PurePosixPath("/home/dgbox/Strata"), PurePosixPath("/home/dgbox"), "dgbox", "dgbox-ubuntu-server", "/tmp")
        self.assertEqual(linux("loading /home/dgbox/Strata/strata-x.json"), "loading <strata>/strata-x.json")
        self.assertEqual(linux("dgbox-ubuntu-server wrote /home/dgbox/Strata/bench/results/run/results.json"),
                         "<host> wrote <strata>/bench/results/run/results.json")
        self.assertEqual(linux("in /home/dgbox/data"), "in <home>/data")
        self.assertEqual(linux("hello dgbox world"), "hello <user> world")
        self.assertEqual(linux("log /tmp/strata.log"), "log <tmp>/strata.log")
        win = red(PureWindowsPath(r"C:\Users\dgbox\Strata"), PureWindowsPath(r"C:\Users\dgbox"), "dgbox", "dgbox-pc",
                  r"C:\Users\dgbox\AppData\Local\Temp")
        self.assertEqual(win(r"in C:\Users\dgbox\data"), r"in <home>\data")
        self.assertEqual(win(r"loading C:\Users\dgbox\Strata\strata-x.json"), r"loading <strata>\strata-x.json")

    def test_objects_are_redacted_deeply(self):
        red = B.redactor()
        out = B.redact_obj({"a": [{"p": r"D:\Users\zed\x"}], "n": 3}, red)
        self.assertNotIn("zed", json.dumps(out))
        self.assertEqual(out["n"], 3)

    def test_config_view_drops_secrets_and_cuts_paths(self):
        cfg = {"exe": r"C:\Users\me\Strata\engine\strata.exe", "args": ["--native", "/home/me/models/m-IQ3_XXS.gguf", "--ctx", "65536"],
               "api_key": "hunter2", "model_name": "Fake Model", "log": "/tmp/x.log", "port": 8080,
               "env": {"STRATA_FOO": "/home/me/z", "PATH": "/usr/bin", "STRATA_TOKEN": "t"}}
        v = B.config_view(cfg)
        self.assertEqual(v["exe"], "strata.exe")
        self.assertEqual(v["args"], ["--native", "m-IQ3_XXS.gguf", "--ctx", "65536"])
        self.assertNotIn("api_key", v)
        self.assertNotIn("log", v)
        self.assertEqual(v["env"], {"STRATA_FOO": "z"})
        self.assertEqual(B.detect_quant(v), "IQ3_XXS")
        self.assertNotIn("hunter2", json.dumps(v))


class ReportTests(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(B, "gpu_info_local", return_value={"gpus": []})     # the server's own GPU name, not this PC's
        p.start()
        self.addCleanup(p.stop)

    def run_main(self, srv, out_dir, *extra):
        argv = ["--url", srv.base, "--out", out_dir, "--runs", "2", "--quiet", *extra]
        return B.main(argv)

    def test_main_writes_a_redacted_report(self):
        srv = FakeStrata()
        self.addCleanup(srv.stop)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.run_main(srv, d, "--clients", "2"), 0)
            dirs = list(Path(d).iterdir())
            self.assertEqual(len(dirs), 1)
            name = dirs[0].name
            self.assertRegex(name, r"^\d{4}-\d\d-\d\d-community-rtx-9999")
            readme = (dirs[0] / "README.md").read_text(encoding="utf-8")
            for part in ("# Community benchmark on NVIDIA GeForce RTX 9999", "## Hardware and software", "## Model and configuration",
                         "## Method", "## Results", "## Correctness and limitations", "### Agent session", "### Concurrency: 2 clients",
                         "| Configuration | Actual prompt tokens | Reused tokens |"):
                self.assertIn(part, readme)
            raw = json.loads((dirs[0] / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(raw["schema"], B.SCHEMA)
            self.assertEqual(raw["machine"]["engine_version"], "9.9.9")
            self.assertEqual(raw["results"]["chat"]["decode_tps"]["median"], 60.0)
            blob = readme + json.dumps(raw)
            self.assertNotIn("Someone", blob)                    # the engine's model_path never reaches the report
            # a second run the same day does not overwrite the first
            self.assertEqual(self.run_main(srv, d), 0)
            self.assertEqual(len(list(Path(d).iterdir())), 2)

    def test_compare(self):
        srv = FakeStrata()
        self.addCleanup(srv.stop)
        with tempfile.TemporaryDirectory() as d:
            self.run_main(srv, d)
            old = next(Path(d).iterdir())
            data = json.loads((old / "results.json").read_text(encoding="utf-8"))
            data["results"]["chat"]["decode_tps"]["median"] = 30.0
            data["machine"]["engine_version"] = "1.0"
            (old / "results.json").write_text(json.dumps(data), encoding="utf-8")
            self.run_main(srv, d)
            new = max(Path(d).iterdir(), key=lambda p: p.name)
            table = B.compare_table(B.load_results(old), B.load_results(new))
            self.assertIn("chat decode tok/s", table)
            self.assertIn("+100.0%", table)
            self.assertIn("engine: 1.0 -> 9.9.9", table)

    def test_only_the_gpus_the_engine_uses_are_listed(self):
        """#1985: a PC with two cards, the engine on one: the report names that one, with its VRAM."""
        srv = FakeStrata()
        self.addCleanup(srv.stop)
        two = {"gpus": [{"name": "NVIDIA GeForce RTX 3060", "vram_mib": 12288, "driver": "560.1"},
                        {"name": "NVIDIA GeForce RTX 9999", "vram_mib": 16384, "driver": "560.1"}]}
        with mock.patch.object(B, "gpu_info_local", return_value=two):
            m = B.collect_machine(srv.base, "", {"gpu": 1})
            self.assertEqual((m["gpu"], m["gpu_vram_mib"]), (["NVIDIA GeForce RTX 9999"], 16384))
            m = B.collect_machine(srv.base, "", {"gpu": [0, 1]})
            self.assertEqual((m["gpu"], m["gpu_vram_mib"]), (["NVIDIA GeForce RTX 3060", "NVIDIA GeForce RTX 9999"], 28672))
            m = B.collect_machine(srv.base, "", {"gpu": "0"}, "1")            # the bench's own --gpu wins
            self.assertEqual(m["gpu"], ["NVIDIA GeForce RTX 9999"])
            m = B.collect_machine(srv.base, "", {"args": ["--gpu", "0"]})
            self.assertEqual(m["gpu"], ["NVIDIA GeForce RTX 3060"])
            # nothing selected: one card, the one the server reports (not every card of the PC)
            m = B.collect_machine(srv.base, "", {})
            self.assertEqual((m["gpu"], m["gpu_vram_mib"]), (["NVIDIA GeForce RTX 9999"], 16384))
        self.assertEqual(B.gpu_selection({"gpu": "0,2"}), [0, 2])
        self.assertEqual(B.gpu_selection({}, None), [])

    def test_the_slug_names_mixed_cards_one_by_one(self):
        """#2062: two different cards are not N copies of the first."""
        a, b = "NVIDIA GeForce RTX 5070 Ti", "NVIDIA RTX A4000"
        self.assertEqual(B.gpu_slug({"gpu": [a]}), "rtx-5070-ti")
        self.assertEqual(B.gpu_slug({"gpu": [a, b]}), "rtx-5070-ti-plus-rtx-a4000")
        self.assertEqual(B.gpu_slug({"gpu": [b, a]}), "rtx-a4000-plus-rtx-5070-ti")
        self.assertEqual(B.gpu_slug({"gpu": [b, b]}), "2x-rtx-a4000")
        self.assertEqual(B.gpu_slug({"gpu": [b, a, b]}), "2x-rtx-a4000-plus-rtx-5070-ti")

    def test_a_card_the_pc_s_nvidia_smi_does_not_list_is_the_servers_own(self):
        """#2001: the engine on an Intel Arc beside a GeForce: the report names the Arc (the name the server reads)."""
        srv = FakeStrata()
        self.addCleanup(srv.stop)
        nv = {"gpus": [{"name": "NVIDIA GeForce RTX 3080 Ti", "vram_mib": 12288, "driver": "560.1"}], "cuda": "12.6"}
        with mock.patch.object(B, "gpu_info_local", return_value=nv), mock.patch.object(
                B, "try_json", side_effect=lambda base, path, *a, **k: (
                    {"hardware_static": {"gpu_name": "Intel Arc Pro B70"}} if path == "/metrics" else {})):
            m = B.collect_machine(srv.base, "", {"backend": "sycl"})
        self.assertEqual((m["gpu"], m["cuda"]), (["Intel Arc Pro B70"], None))

    def test_the_report_labels_the_driver_s_cuda_and_the_engine_s_build(self):
        """#2111: the CUDA line was the driver's ceiling read as the engine's toolkit, and storage said "not measured"."""
        srv = FakeStrata()
        self.addCleanup(srv.stop)
        nv = {"gpus": [{"name": "NVIDIA GeForce RTX 9999", "vram_mib": 16384, "driver": "616.92",
                        "power_limit_w": "165.00", "power_draw_w": "20.25", "pcie": "gen4 x8",
                        "pstate": "P3", "temp_c": "45"}], "cuda": "13.4"}
        with mock.patch.object(B, "gpu_info_local", return_value=nv), mock.patch.object(
                B, "storage_info", return_value="Samsung SSD 990 EVO Plus 2TB SSD/NVMe"), mock.patch.object(
                B, "engine_build", return_value={"version": "0.1.43", "cuda": "13.0"}):
            with tempfile.TemporaryDirectory() as d:
                self.assertEqual(self.run_main(srv, d), 0)
                readme = next(Path(d).iterdir()).joinpath("README.md").read_text(encoding="utf-8")
        for part in ("CUDA (driver supports up to): 13.4; engine built for CUDA 13.0",
                     "storage: Samsung SSD 990 EVO Plus 2TB SSD/NVMe",
                     "GPU detail (after the run, near idle): power_limit_w=165.00, power_draw_w=20.25, "
                     "pcie=gen4 x8, pstate=P3, temp_c=45",
                     "Background workloads: not recorded (GPU power limit/draw read above)"):
            self.assertIn(part, readme)
        self.assertNotIn("storage: not measured", readme)

    def test_a_bench_server_never_saves_over_the_users_profile(self):
        """#2088: --expert-profile-save writes the profile at quit; the bench's own server runs without it."""
        cfg = {"args": ["--expert-profile", "p.bin", "--expert-profile-save", "p.bin", "--spec", "4"]}
        safe, removed = B.without_profile_save(cfg)
        self.assertTrue(removed)
        self.assertEqual(safe["args"], ["--expert-profile", "p.bin", "--spec", "4"])
        self.assertEqual(cfg["args"][3], "p.bin")                       # the input is not changed
        safe, removed = B.without_profile_save({"args": ["--expert-profile-save=p.bin", "--spec", "4"]})
        self.assertEqual((safe["args"], removed), (["--spec", "4"], True))
        same = {"args": ["--spec", "4"]}
        self.assertEqual(B.without_profile_save(same), (same, False))
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            c = Path(d) / "strata-x.json"
            c.write_text(json.dumps(cfg))
            with mock.patch.object(B.subprocess, "Popen") as popen, contextlib.redirect_stderr(io.StringIO()):
                B.start_server(c, 8080, None, Path(d) / "run.log")
            used = Path(popen.call_args[0][0][popen.call_args[0][0].index("--config") + 1])
            self.assertNotIn("--expert-profile-save", json.loads(used.read_text())["args"])
            self.assertIn("--expert-profile-save", json.loads(c.read_text())["args"])   # the user's file is as it was

    def test_an_attached_server_is_not_stopped(self):
        """#2088: with --url the bench only reads; the server it did not start keeps running."""
        srv = FakeStrata()
        self.addCleanup(srv.stop)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(B, "stop_server") as stop:
            self.assertEqual(self.run_main(srv, d), 0)
        stop.assert_not_called()

    def test_names(self):
        m = {"gpu": ["NVIDIA GeForce RTX 3060"], "model": "Qwen3.8 Flash-Next", "quant": "IQ3_XXS"}
        self.assertEqual(B.results_dir_name(m, "2026-10-10"), "2026-10-10-community-rtx-3060-qwen3-8-flash-next-iq3-xxs")
        m["model"] = "qwen-iq3_xxs"                                      # a quant already in the name is not added twice
        self.assertEqual(B.results_dir_name(m, "2026-10-10"), "2026-10-10-community-rtx-3060-qwen-iq3-xxs")
        self.assertEqual(B.gpu_slug({"gpu": ["A B 4090", "A B 4090"]}), "2x-a-b-4090")
        m["model"] = "qwen-iq3_xxs"                                      # a quant already in the name is not added twice
        self.assertEqual(B.results_dir_name(m, "2026-10-10"), "2026-10-10-community-rtx-3060-qwen-iq3-xxs")
        self.assertEqual(B.gpu_slug({"gpu": []}), "cpu")

    def test_no_server_is_an_error_not_a_crash(self):
        self.assertEqual(B.main(["--url", "http://127.0.0.1:9", "--quiet"]), 2)

    def test_project_version_does_not_use_the_locale_code_page(self):
        # #2792: on Windows a text file read without an encoding is decoded with the locale's code page, and CMakeLists.txt
        # is UTF-8 with non-ASCII text.  Path.read_text here defaults to each Windows ANSI code page in turn.
        original = Path.read_text
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "CMakeLists.txt").write_bytes(
                "cmake_minimum_required(VERSION 3.24)\n# notes — § €\nproject(strata VERSION 9.8.7 LANGUAGES CXX)\n"
                .encode("utf-8"))
            for code_page in ("cp874", "cp932", "cp936", "cp949", "cp950", "cp1250", "cp1251", "cp1252", "cp1253",
                              "cp1254", "cp1255", "cp1256", "cp1257", "cp1258"):
                def locale_default(self, encoding=None, errors=None, **kw):
                    return original(self, encoding=encoding or code_page, errors=errors, **kw)

                with self.subTest(code_page=code_page), mock.patch.object(B, "ROOT", root), \
                        mock.patch.object(Path, "read_text", locale_default):
                    self.assertEqual(B.project_version(), "9.8.7")


class MachineDetailTests(unittest.TestCase):
    """#2111: what the report reads straight from nvidia-smi on the PC."""

    ROW = "NVIDIA GeForce RTX 4060 Ti, 16380, 616.92, 165.00, 20.25, 4, 8, P3, 45"

    @staticmethod
    def queries(cmd) -> bool:
        return any(str(c).startswith("--query-gpu") for c in cmd)

    def test_cuda_header_in_both_forms(self):
        """Current drivers print "CUDA UMD Version: 13.4"; the old "CUDA Version:" line still has to be read."""
        for head, want in (("Some header\n\nCUDA UMD Version: 13.4\n", "13.4"),
                           ("Some header\n\nCUDA Version: 12.6\n", "12.6")):
            def fake(cmd, *a, _head=head, **k):
                return self.ROW if self.queries(cmd) else _head
            with mock.patch.object(B, "run_cmd", side_effect=fake):
                info = B.gpu_info_local()
            self.assertEqual((info["cuda"], info["gpus"][0]["power_limit_w"]), (want, "165.00"))

    def test_power_pcie_pstate_and_temperature_are_read(self):
        with mock.patch.object(B, "run_cmd",
                               side_effect=lambda cmd, *a, **k: self.ROW if self.queries(cmd) else ""):
            g = B.gpu_info_local()["gpus"][0]
        self.assertEqual((g["name"], g["vram_mib"], g["driver"]), ("NVIDIA GeForce RTX 4060 Ti", 16380, "616.92"))
        self.assertEqual((g["power_limit_w"], g["pcie"], g["pstate"], g["temp_c"]), ("165.00", "gen4 x8", "P3", "45"))


class RuntimeInfoAndBenchTests(unittest.TestCase):
    """Tests for serve/runtime_info.py, GET /v1/runtime, and benchmark report/compare integration."""

    def test_runtime_info_snapshot_parses_cli_wrapper_info_and_env_defaults(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from serve import runtime_info as RI

        wrapper_path = str(Path(__file__).resolve().parents[1] / "sycl" / "serve" / "strata-sycl.sh")
        cfg = {
            "backend": "sycl",
            "model_name": "Qwen3.8-Flash-Next-AWQ",
            "exe": wrapper_path,
            "args": ["--native", "/home/user/models/model-00001-of-00003.gguf", "--kv", "q4_0", "--spec", "4", "--prefill", "auto"],
            "env": {"STRATA_XE2_SPLIT_K": "0", "SECRET_API_TOKEN": "hidden"},
        }
        eng_info = {
            "context": 32768,
            "kv": "q4_0",
            "expert_slots": 4560,
            "spec": 4,
            "prefill_cap": 2048,
            "vram_reserve_mib": 700,
            "adapt_every": 4,
            "adapt_swaps": 96,
            "ple_io": "direct",
            "pipeline_windows": 2,
            "no_prefill_borrow": 0,
            "mmap_experts": 1,
            "stream_experts": 0,
            "short_read": 64,
            "prompt_cache": 6,
            "prompt_cache_every": 16384,
            "adapt_async": 1,
        }
        fake_eng = mock.Mock()
        fake_eng.spawn = (
            wrapper_path,
            cfg["args"],
            "/home/user/projects/Strata",
            "/tmp/strata.log",
            {
                "STRATA_XE2_SPLIT_K": "0",
                "STRATA_MMVQ_SG": "32",
                "STRATA_API_KEY": "super-secret",
                "NEOReadDebugKeys": "1",
                "HOME": "/home/user",
            },
            False,
            None,
        )
        fake_eng.info = eng_info
        fake_svc = mock.Mock()
        fake_svc.engine = fake_eng
        fake_svc.runtime_cfg = cfg
        fake_svc.launch_argv = ["--engine", "strata", "--config", "/home/user/strata-qwen.json", "--port", "8080", "--api-key", "secret123"]
        fake_svc.launch_script = "/home/user/projects/Strata/sycl/serve/server_intel.py"
        fake_svc.parsed_args = {"engine": "strata", "config": "/home/user/strata-qwen.json", "port": 8080, "host": "127.0.0.1"}
        fake_svc.max_queue = None
        fake_svc.queue_timeout_s = None
        fake_svc.fit_max_tokens = True
        fake_svc.offline = False

        snap = RI.snapshot(fake_svc)

        # Server CLI & redacted config
        self.assertEqual(snap["server"]["script"], "sycl/serve/server_intel.py")
        self.assertIn("<redacted>", snap["server"]["argv"])
        self.assertNotIn("secret123", json.dumps(snap))
        self.assertNotIn("super-secret", json.dumps(snap))
        self.assertNotIn("hidden", json.dumps(snap))
        self.assertEqual(snap["server"]["config_path"], "strata-qwen.json")
        self.assertEqual(snap["server"]["options"]["--port"], 8080)

        # Engine CLI: combines defaults + --native preset + explicit CLI + INFO overrides
        self.assertEqual(snap["engine"]["exe"], "sycl/serve/strata-sycl.sh")
        self.assertEqual(snap["engine"]["sycl_bin"], "build-sycl-aot/strata")
        self.assertEqual(snap["engine"]["options"]["--kv"], "q4_0")
        self.assertEqual(snap["engine"]["options"]["--native-bf16"], True)   # from --native preset
        self.assertEqual(snap["engine"]["options"]["--prefill"], 2048)       # resolved from INFO prefill_cap=2048
        self.assertEqual(snap["engine"]["options"]["--pipeline-windows"], 2) # resolved from INFO pipeline_windows=2
        self.assertEqual(snap["engine"]["options"]["--mmap-experts"], True)  # resolved from INFO mmap_experts=1
        self.assertEqual(snap["engine"]["options"]["--adapt-async"], 1)

        # Environment variables: explicit, config, wrapper defaults, and code defaults
        env = snap["env"]
        self.assertEqual(env["explicit"]["STRATA_XE2_SPLIT_K"], "0")
        self.assertEqual(env["explicit"]["STRATA_MMVQ_SG"], "32")
        self.assertEqual(env["explicit"]["NEOReadDebugKeys"], "1")
        self.assertNotIn("HOME", env["explicit"])
        self.assertEqual(env["wrapper_defaults"].get("STRATA_VERIFY_DEVICE_PLAN"), "1")
        self.assertEqual(env["wrapper_defaults"].get("STRATA_VERIFY_NO_HOST"), "1")
        self.assertEqual(env["wrapper_defaults"].get("STRATA_STAGER_THREADS"), "12")
        self.assertEqual(env["defaults"].get("STRATA_XE2_XMX_MOE"), "1")
        self.assertEqual(env["effective"]["STRATA_XE2_SPLIT_K"], "0")
        self.assertEqual(env["effective"]["STRATA_STAGER_THREADS"], "12")
        self.assertEqual(env["sources"]["STRATA_XE2_SPLIT_K"], "config")
        self.assertEqual(env["sources"]["STRATA_MMVQ_SG"], "env")
        self.assertEqual(env["sources"]["STRATA_STAGER_THREADS"], "wrapper_default")
        self.assertEqual(env["sources"]["STRATA_XE2_XMX_MOE"], "default")

    def test_collect_machine_populates_from_v1_runtime_when_no_local_config(self):
        runtime_payload = {
            "server": {
                "script": "sycl/serve/server_intel.py",
                "argv": ["--engine", "strata", "--config", "strata-b580.json", "--port", "8080"],
                "explicit_options": {"--engine": "strata", "--config": "strata-b580.json", "--port": 8080},
                "defaults": {"--port": 8095},
                "options": {"--engine": "strata", "--config": "strata-b580.json", "--port": 8080, "--gpu": "0"},
                "config_path": "strata-b580.json",
                "config": {
                    "backend": "sycl",
                    "model_name": "Qwen3.8-Flash-Next",
                    "exe": "sycl/serve/strata-sycl.sh",
                    "args": ["--native", "Qwen3.8-Flash-Next-IQ2_XS.gguf", "--kv", "q4_0", "--spec", "4"],
                    "env": {"STRATA_MMVQ_SG": "16"},
                },
            },
            "engine": {
                "exe": "sycl/serve/strata-sycl.sh",
                "sycl_bin": "build-sycl/strata",
                "argv": ["--native", "Qwen3.8-Flash-Next-IQ2_XS.gguf", "--kv", "q4_0", "--spec", "4"],
                "explicit_options": {"--native": "Qwen3.8-Flash-Next-IQ2_XS.gguf", "--kv": "q4_0", "--spec": 4},
                "defaults": {"--kv": "fp16", "--prefill": 0, "--vram-reserve-mib": 700},
                "options": {"--native": "Qwen3.8-Flash-Next-IQ2_XS.gguf", "--kv": "q4_0", "--spec": 4, "--prefill": 2048, "--vram-reserve-mib": 700},
                "info": {"context": 32768, "prefill_cap": 2048},
            },
            "env": {
                "explicit": {"STRATA_MMVQ_SG": "16"},
                "config_env": {"STRATA_MMVQ_SG": "16"},
                "wrapper_defaults": {"SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS": "1"},
                "defaults": {"STRATA_XE2_SPLIT_K": "1", "STRATA_MMVQ_SG": "16"},
                "effective": {"STRATA_MMVQ_SG": "16", "STRATA_XE2_SPLIT_K": "1", "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS": "1"},
                "sources": {"STRATA_MMVQ_SG": "config", "STRATA_XE2_SPLIT_K": "default", "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS": "wrapper_default"},
            },
        }
        srv = FakeStrata(runtime_payload=runtime_payload)
        self.addCleanup(srv.stop)
        mc = B.collect_machine(srv.base, "", None)
        self.assertEqual(mc["model"], "Qwen3.8-Flash-Next")
        self.assertEqual(mc["quant"], "IQ2_XS")
        self.assertIn("strata-sycl.sh --native Qwen3.8-Flash-Next-IQ2_XS.gguf", mc["launch"])
        self.assertEqual(mc["runtime"]["engine"]["sycl_bin"], "build-sycl/strata")

        # Verify README renders config file, server command, engine binary, effective options, and env vars
        fake_res = {
            "date": "2026-11-12",
            "elapsed_s": 10.0,
            "params": {"runs": 1, "think": False, "command": "python tools/strata_bench.py --url ...", "started_server": False},
            "machine": mc,
            "results": {"chat": {}, "prompts": {}, "agent": {}, "clients": None, "skipped": {}, "failures": []},
        }
        readme = B.render_readme(fake_res)
        self.assertIn("Config file: `strata-b580.json`", readme)
        self.assertIn("Server command: `sycl/serve/server_intel.py --engine strata --config strata-b580.json --port 8080`", readme)
        self.assertIn("Engine binary: `build-sycl/strata`", readme)
        self.assertIn("Effective engine options (including defaults):", readme)
        self.assertIn("`--prefill=2048`", readme)
        self.assertIn("Explicit runtime environment: `STRATA_MMVQ_SG=16`", readme)
        self.assertIn("Wrapper default environment: `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1`", readme)
        self.assertIn("Code default environment: `STRATA_XE2_SPLIT_K=1`", readme)

        # Verify compare_table surfaces differences in both explicit options/env and code defaults
        mc2 = json.loads(json.dumps(mc))
        mc2["runtime"]["engine"]["options"]["--prefill"] = 4096
        mc2["runtime"]["env"]["effective"]["STRATA_XE2_SPLIT_K"] = "0"
        mc2["runtime"]["env"]["sources"]["STRATA_XE2_SPLIT_K"] = "default"
        mc2["runtime"]["env"]["effective"]["STRATA_MMVQ_SG"] = "32"
        mc2["runtime"]["env"]["sources"]["STRATA_MMVQ_SG"] = "env"
        fake_res2 = {**fake_res, "machine": mc2}
        diff_out = B.compare_table(fake_res, fake_res2)
        self.assertIn("engine --prefill: 2048 -> 4096 (default -> default)", diff_out)
        self.assertIn("env STRATA_XE2_SPLIT_K: 1 -> 0 (default -> default)", diff_out)
        self.assertIn("env STRATA_MMVQ_SG: 16 -> 32 (config -> env)", diff_out)

    def test_start_server_uses_server_intel_for_sycl_backend(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            sycl_cfg = Path(d) / "strata-sycl.json"
            sycl_cfg.write_text(json.dumps({"backend": "sycl", "exe": "sycl/serve/strata-sycl.sh", "args": ["--spec", "4"]}))
            cuda_cfg = Path(d) / "strata-cuda.json"
            cuda_cfg.write_text(json.dumps({"backend": "cuda", "exe": "build/strata", "args": ["--spec", "4"]}))
            with mock.patch.object(B.subprocess, "Popen") as popen:
                B.start_server(sycl_cfg, 8080, None, Path(d) / "sycl.log")
                sycl_cmd = popen.call_args[0][0]
                B.start_server(cuda_cfg, 8080, None, Path(d) / "cuda.log")
                cuda_cmd = popen.call_args[0][0]
            self.assertTrue(sycl_cmd[1].endswith("sycl/serve/server_intel.py"))
            self.assertTrue(cuda_cmd[1].endswith("serve/server.py"))
            self.assertFalse(cuda_cmd[1].endswith("sycl/serve/server_intel.py"))

    def test_live_server_intel_exposes_v1_runtime_endpoint(self):
        import socket
        import subprocess
        import time

        root = Path(__file__).resolve().parents[1]
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        env = dict(os.environ)
        env["STRATA_XE2_SPLIT_K"] = "0"
        env["STRATA_MMVQ_SG"] = "32"
        env["STRATA_SECRET_TOKEN"] = "do-not-expose"
        proc = subprocess.Popen(
            [
                sys.executable,
                str(root / "sycl" / "serve" / "server_intel.py"),
                "--engine",
                "mock",
                "--port",
                str(port),
                "--max-queue",
                "12",
                "--no-fit-max-tokens",
            ],
            cwd=str(root),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.addCleanup(lambda: (proc.terminate(), proc.wait(5)) if proc.poll() is None else None)

        base = f"http://127.0.0.1:{port}"
        self.assertTrue(B.wait_ready(base, "", proc, 15.0, log=lambda *a: None))

        rt = B.try_json(base, "/v1/runtime", "")
        st = B.try_json(base, "/v1/status", "")
        self.assertIsInstance(rt, dict)
        self.assertEqual(rt["server"]["script"], "sycl/serve/server_intel.py")
        self.assertEqual(rt["server"]["options"]["--port"], port)
        self.assertEqual(rt["server"]["options"]["--max-queue"], 12)
        self.assertEqual(rt["server"]["options"]["--fit-max-tokens"], False)
        self.assertEqual(rt["env"]["explicit"]["STRATA_XE2_SPLIT_K"], "0")
        self.assertEqual(rt["env"]["explicit"]["STRATA_MMVQ_SG"], "32")
        self.assertEqual(rt["env"]["explicit"]["STRATA_WATCHDOG_S"], "180")
        self.assertNotIn("STRATA_SECRET_TOKEN", rt["env"]["explicit"])
        self.assertNotIn("do-not-expose", json.dumps(rt))
        self.assertEqual(st.get("runtime", {}).get("server", {}).get("script"), "sycl/serve/server_intel.py")

    def test_running_server_runtime_config_overrides_local_newest_config(self):
        runtime_payload = {
            "server": {
                "script": "sycl/serve/server_intel.py",
                "argv": ["--engine", "strata", "--config", "strata-actual.json"],
                "config_path": "strata-actual.json",
                "config": {
                    "backend": "sycl",
                    "model_name": "Actual-Running-Model",
                    "exe": "build_sycl/strata",
                    "args": ["--native", "actual-IQ3_S.gguf", "--spec", "4"],
                    "env": {"STRATA_MMVQ_SG": "32"},
                },
            },
            "engine": {"exe": "build_sycl/strata", "argv": ["--native", "actual-IQ3_S.gguf", "--spec", "4"]},
            "env": {"explicit": {"STRATA_MMVQ_SG": "32"}, "effective": {"STRATA_MMVQ_SG": "32"}, "sources": {"STRATA_MMVQ_SG": "config"}},
        }
        srv = FakeStrata(runtime_payload=runtime_payload)
        self.addCleanup(srv.stop)
        port = srv.server_address[1]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            wrong1 = root / "strata-wrong-old.json"
            wrong2 = root / "strata-wrong-newest.json"
            wrong1.write_text(json.dumps({"model_name": "Wrong-Old", "port": port, "exe": "strata", "args": ["old.gguf"]}))
            wrong2.write_text(json.dumps({"model_name": "Wrong-Newest", "port": port, "exe": "strata", "args": ["newest.gguf"], "env": {"STRATA_MMVQ_SG": "8"}}))
            out_dir = root / "out"
            buf = io.StringIO()
            with mock.patch.object(B, "ROOT", root), mock.patch.object(B, "find_configs", return_value=[wrong2, wrong1]), \
                    contextlib.redirect_stdout(buf):
                rc = B.main(["--port", str(port), "--runs", "1", "--out", str(out_dir)])
            self.assertEqual(rc, 0)
            stdout_text = buf.getvalue()
            self.assertNotIn("the one used last", stdout_text)
            self.assertIn("Using the Strata server already running", stdout_text)
            self.assertIn("config: strata-actual.json", stdout_text)
            res_file = next(out_dir.iterdir()) / "results.json"
            saved = json.loads(res_file.read_text(encoding="utf-8"))
            self.assertEqual(saved["machine"]["model"], "Actual-Running-Model")
            self.assertEqual(saved["machine"]["config"]["env"], {"STRATA_MMVQ_SG": "32"})


if __name__ == "__main__":
    unittest.main()
