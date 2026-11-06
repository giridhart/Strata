"""serve/mcp.py - tools from MCP (Model Context Protocol) servers for the web app's chats.

The user lists servers in the run config (`"mcp_servers"`, or Claude Desktop's `"mcpServers"` block pasted as is) or
in a separate file (`--mcp-config`).  Each one is either a program Strata starts (`command`, `args`, `env`, `cwd`:
the stdio transport, newline-delimited JSON-RPC 2.0 on its stdin/stdout) or an address (`url`, `headers`: the
Streamable HTTP transport, JSON-RPC POSTed, answered as JSON or as an event stream).  Both are implemented here with
the standard library only; no MCP SDK is needed.

`McpHub` owns every configured server: it starts them in the background when the server starts, offers their tools to
the model as OpenAI function tools named `<server>__<tool>` (two servers may both have a `search`), and routes a call
back to the server that has it.  A server that does not start is left out with one log line; the chat works without
it.  A server that stops later (a crash) is started again on its next call.

Tool results reach the model as text: the MCP content blocks joined, capped at `max_result_chars` (a directory tree or
a web page can be megabytes, far more than a context holds) with a note saying so.  A failed call - the tool's own
error, a crash, a timeout - becomes a result that starts with "error:", so the model can react instead of the chat
breaking.
"""
from __future__ import annotations

import collections
import base64
import hashlib
from pathlib import Path
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from serve import lifeline

PROTOCOL = "2025-06-18"               # the MCP revision Strata asks for; the server's answer is used as given
DEFAULTS = {"timeout_s": 60.0, "max_result_chars": 20000, "max_rounds": 8, "start_timeout_s": 120.0, "image_results": False,
            "codemode": False, "codemode_lang": "javascript"}

CODEMODE_PYTHON_NAMES = ("run_python", "python", "py")
CODEMODE_JS_NAMES = ("run_javascript", "javascript", "js", "run_js")
CODEMODE_GENERIC_NAMES = ("run_code", "execute_code")
CODEMODE_CALL_NAMES = frozenset(CODEMODE_PYTHON_NAMES + CODEMODE_JS_NAMES + CODEMODE_GENERIC_NAMES)


class McpError(RuntimeError):
    """A server answered with an error, stopped, or could not be reached."""


class McpTimeout(McpError):
    pass


class McpCancelled(McpError):
    """The chat was stopped while the call ran."""


class _Slot:
    """One request waiting for its response."""

    def __init__(self):
        self.done = threading.Event()
        self.msg = None
        self.error = None


def _wait(slot: _Slot, timeout: float, cancel: threading.Event | None, what: str):
    """Wait for a response, checking the chat's stop button and the deadline a few times a second."""
    deadline = time.monotonic() + timeout
    while not slot.done.wait(0.05):
        if cancel is not None and cancel.is_set():
            raise McpCancelled("stopped")
        if time.monotonic() > deadline:
            raise McpTimeout(f"{what} timed out after {timeout:g} s")
    if slot.error:
        raise slot.error
    msg = slot.msg
    if "error" in msg:
        err = msg["error"] if isinstance(msg["error"], dict) else {"message": str(msg["error"])}
        raise McpError(f"{err.get('message') or 'error'} (code {err.get('code')})")
    return msg.get("result") or {}


# ------------------------------------------------------------------------------------------------ transports
class StdioTransport:
    """A server Strata starts: JSON-RPC messages one per line on its stdin and stdout; its stderr is its log."""
    kind = "stdio"

    def __init__(self, name: str, cfg: dict):
        self.name, self.cfg = name, cfg
        self.proc = None
        self.lock = threading.Lock()                    # the pending requests
        self.write_lock = threading.Lock()              # stdin; separate, so a full pipe can't block the reader
        self.pending: dict[int, _Slot] = {}
        self.next_id = 0
        self.stderr = collections.deque(maxlen=40)       # its last log lines, for the error message when it fails
        self.protocol = PROTOCOL

    def start(self):
        env = dict(os.environ)
        env.update({str(k): str(v) for k, v in (self.cfg.get("env") or {}).items()})
        command = str(self.cfg["command"])
        # On Windows `npx`, `uvx` and friends are .cmd files that CreateProcess finds only with their extension
        exe = shutil.which(command, path=env.get("PATH")) or command
        extra = {}
        if os.name == "nt":
            extra["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            extra["start_new_session"] = True           # its own process group, so close() ends its children too
        try:
            self.proc = lifeline.popen([exe, *[str(a) for a in self.cfg.get("args") or []]], "mcp", stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.cfg.get("cwd") or None,
                                       env=env, **extra)
        except OSError as e:
            raise McpError(f"could not start {command!r}: {e}") from None
        self._err_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._err_reader.start()
        threading.Thread(target=self._read, daemon=True).start()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None and not getattr(self, "ended", False)

    @staticmethod
    def _lines(f):
        """A pipe's lines until it closes (or close() closed it under us)."""
        try:
            yield from f
        except (OSError, ValueError):
            return

    def _read_stderr(self):
        for raw in self._lines(self.proc.stderr):
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                self.stderr.append(line)
                if os.environ.get("STRATA_DEBUG"):
                    print(f"[mcp {self.name}] {line}", flush=True)

    def _read(self):
        for raw in self._lines(self.proc.stdout):
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:                           # a server that prints to stdout; the spec forbids it
                self.stderr.append(line)
                continue
            for m in msg if isinstance(msg, list) else [msg]:
                if isinstance(m, dict):
                    self._dispatch(m)
        self.ended = True
        # the server's last log line says why it stopped: let the stderr reader take it first (the pipe closes with
        # the process), or the error raced it and said only "the server stopped"
        self._err_reader.join(1.0)
        code = self.proc.poll()
        err = McpError(f"the server stopped{f' (exit code {code})' if code is not None else ''}{self._tail()}")
        with self.lock:
            slots, self.pending = list(self.pending.values()), {}
        for s in slots:
            s.error = err
            s.done.set()

    def _tail(self) -> str:
        return f": {self.stderr[-1][:300]}" if self.stderr else ""

    def _dispatch(self, m: dict):
        if "method" in m:
            if "id" in m:                                # a request from the server: we offer no client features
                try:
                    if m["method"] == "ping":
                        self._send({"jsonrpc": "2.0", "id": m["id"], "result": {}})
                    else:
                        self._send({"jsonrpc": "2.0", "id": m["id"],
                                    "error": {"code": -32601, "message": f"method not supported: {m['method']}"}})
                except McpError:
                    pass
            return                                       # notifications (log messages, list changes): not used
        with self.lock:
            slot = self.pending.pop(m.get("id"), None)
        if slot is not None:                             # an answer to a call that timed out is simply dropped
            slot.msg = m
            slot.done.set()

    def _send(self, msg: dict):
        data = json.dumps(msg, ensure_ascii=False).encode("utf-8") + b"\n"
        try:
            with self.write_lock:
                self.proc.stdin.write(data)
                self.proc.stdin.flush()
        except (OSError, ValueError):
            raise McpError(f"the server stopped{self._tail()}") from None

    def request(self, method: str, params: dict, timeout: float, cancel=None):
        if not self.alive():
            raise McpError(f"the server is not running{self._tail()}")
        slot = _Slot()
        with self.lock:
            self.next_id += 1
            rid = self.next_id
            self.pending[rid] = slot
        try:
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
            return _wait(slot, timeout, cancel, method)
        except (McpTimeout, McpCancelled) as e:
            with self.lock:
                self.pending.pop(rid, None)
            try:                                         # tell the server to stop working on it
                self.notify("notifications/cancelled", {"requestId": rid, "reason": str(e)})
            except McpError:
                pass
            raise

    def notify(self, method: str, params: dict | None = None):
        self._send({"jsonrpc": "2.0", "method": method, **({"params": params} if params is not None else {})})

    def close(self):
        """As the spec says: close its stdin, give it a moment to exit, then end it (and what it started)."""
        p = self.proc
        if p is None:
            return
        try:
            p.stdin.close()
        except OSError:
            pass
        try:
            p.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._kill_tree(p)
        for f in (p.stdout, p.stderr):
            try:
                f.close()
            except OSError:
                pass

    @staticmethod
    def _kill_tree(p):
        try:
            if os.name == "nt":                          # npx.cmd -> node: end the whole tree
                subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=10)
            else:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            p.kill()
        except OSError:
            pass


class HttpTransport:
    """A server at an address (Streamable HTTP): every message is a POST; the answer is JSON or an event stream that
    carries it.  The session id the server hands out at `initialize` rides on every later request."""
    kind = "http"

    def __init__(self, name: str, cfg: dict):
        self.name, self.url = name, str(cfg["url"])
        self.headers = {str(k): str(v) for k, v in (cfg.get("headers") or {}).items()}
        self.session = None
        self.protocol = None                             # sent as MCP-Protocol-Version once negotiated
        self.next_id = 0
        self.lock = threading.Lock()
        self.broken = None

    def start(self):
        pass

    def alive(self) -> bool:
        return self.broken is None

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
             "User-Agent": "strata", **self.headers}
        if self.session:
            h["Mcp-Session-Id"] = self.session
        if self.protocol:
            h["MCP-Protocol-Version"] = self.protocol
        return h

    def _post(self, msg: dict, rid, timeout: float, slot: _Slot | None):
        try:
            req = urllib.request.Request(self.url, data=json.dumps(msg, ensure_ascii=False).encode("utf-8"),
                                         headers=self._headers(), method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if r.headers.get("Mcp-Session-Id"):
                    self.session = r.headers["Mcp-Session-Id"]
                if slot is None:                         # a notification: 202, nothing to read
                    return
                ctype = r.headers.get("Content-Type", "")
                if "text/event-stream" in ctype:
                    found = self._from_events(r, rid)
                else:
                    body = r.read()
                    found = None
                    parsed = json.loads(body) if body.strip() else None
                    for m in parsed if isinstance(parsed, list) else [parsed]:
                        if isinstance(m, dict) and m.get("id") == rid and "method" not in m:
                            found = m
                if found is None:
                    raise McpError("the server's answer did not contain a response")
                slot.msg = found
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except OSError:
                pass
            if e.code == 404 and self.session:           # the session expired: a restart starts a new one
                self.broken = McpError("the server ended the session")
            err = McpError(f"HTTP {e.code} {e.reason}{': ' + detail if detail else ''}")
            if slot is None:
                raise err from None
            slot.error = err
        except (OSError, ValueError, McpError) as e:
            err = e if isinstance(e, McpError) else McpError(f"could not reach {self.url}: {e}")
            if slot is None:
                raise err from None
            slot.error = err
        finally:
            if slot is not None:
                slot.done.set()

    @staticmethod
    def _from_events(r, rid):
        """Read the event stream until the event that answers `rid` (it may carry notifications first)."""
        data = []
        for raw in r:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
            elif not line and data:
                try:
                    m = json.loads("\n".join(data))
                except ValueError:
                    m = None
                data = []
                for x in m if isinstance(m, list) else [m]:
                    if isinstance(x, dict) and x.get("id") == rid and "method" not in x:
                        return x
        return None

    def request(self, method: str, params: dict, timeout: float, cancel=None):
        if self.broken:
            raise self.broken
        with self.lock:
            self.next_id += 1
            rid = self.next_id
        slot = _Slot()
        # the POST runs on its own thread so the stop button and the deadline are noticed while it waits
        threading.Thread(target=self._post, args=({"jsonrpc": "2.0", "id": rid, "method": method, "params": params},
                                                  rid, timeout + 5, slot), daemon=True).start()
        try:
            return _wait(slot, timeout, cancel, method)
        except (McpTimeout, McpCancelled) as e:
            threading.Thread(target=self._quiet_notify, args=("notifications/cancelled",
                                                              {"requestId": rid, "reason": str(e)}), daemon=True).start()
            raise

    def _quiet_notify(self, method, params):
        try:
            self.notify(method, params)
        except McpError:
            pass

    def notify(self, method: str, params: dict | None = None):
        self._post({"jsonrpc": "2.0", "method": method, **({"params": params} if params is not None else {})},
                   None, 10, None)

    def close(self):
        if not self.session:
            return
        try:                                             # end the session on the server (it may not allow that)
            req = urllib.request.Request(self.url, headers=self._headers(), method="DELETE")
            urllib.request.urlopen(req, timeout=3).close()
        except (OSError, ValueError):
            pass


# ------------------------------------------------------------------------------------------------ one server
class McpServer:
    """One configured server: started once (and again after it stopped), with the tools it listed."""

    def __init__(self, name: str, cfg: dict, settings: dict):
        self.name, self.cfg, self.settings = name, cfg, settings
        self.kind = "http" if cfg.get("url") else "stdio"
        self.status = "idle"             # idle -> starting -> ready | failed; ready -> stopped when it ends
        self.error = None
        self.tools: list[dict] = []
        self.info: dict = {}
        self.transport = None
        self.lock = threading.Lock()
        self.last_start = 0.0

    def start(self) -> bool:
        with self.lock:
            return self._start()

    def _start(self) -> bool:
        if self.transport is not None:
            self.transport.close()
        self.status, self.error, self.last_start = "starting", None, time.monotonic()
        t = HttpTransport(self.name, self.cfg) if self.kind == "http" else StdioTransport(self.name, self.cfg)
        self.transport = t
        timeout = float(self.settings["start_timeout_s"])
        try:
            t.start()
            res = t.request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                           "clientInfo": {"name": "strata", "title": "Strata", "version": "1"}},
                            timeout)
            t.protocol = str(res.get("protocolVersion") or PROTOCOL)
            self.info = {k: v for k, v in (res.get("serverInfo") or {}).items() if k in ("name", "title", "version")}
            self.info["protocol"] = t.protocol
            t.notify("notifications/initialized")
            tools, cursor = [], None
            for _ in range(100):                         # pages; a server that never stops paging is cut off
                try:
                    page = t.request("tools/list", {"cursor": cursor} if cursor else {}, timeout)
                except McpError as e:
                    if "code -32601" in str(e) and not tools:   # a server without tools (prompts/resources only)
                        break
                    raise
                tools += [x for x in page.get("tools") or [] if isinstance(x, dict) and x.get("name")]
                cursor = page.get("nextCursor")
                if not cursor:
                    break
            self.tools, self.status = tools, "ready"
            names = ", ".join(x["name"] for x in tools[:8]) + (", ..." if len(tools) > 8 else "")
            print(f"[strata] MCP server {self.name!r}: {len(tools)} tool{'s' * (len(tools) != 1)}"
                  f"{' (' + names + ')' if tools else ''}", flush=True)
            return True
        except McpError as e:
            self.status, self.error, self.tools = "failed", str(e), []
            t.close()
            print(f"[strata] MCP server {self.name!r} did not start: {e}. Its tools are left out; the chat works "
                  "without them.", flush=True)
            return False

    def call(self, tool: str, arguments: dict, timeout: float, cancel=None) -> dict:
        with self.lock:
            if self.transport is None or not self.transport.alive():
                # it stopped since its last call (a crash, an expired session): start it again; one that failed to
                # start is tried again at most every 10 s, so a broken command does not run on every call
                if self.status == "failed" and time.monotonic() - self.last_start < 10:
                    raise McpError(f"the server is not running ({self.error or 'it stopped'})")
                print(f"[strata] MCP server {self.name!r} had stopped; starting it again", flush=True)
                if not self._start():
                    raise McpError(f"the server could not be started again: {self.error}")
            t = self.transport
        try:
            return t.request("tools/call", {"name": tool, "arguments": arguments}, timeout, cancel)
        except McpError as e:
            if not t.alive():
                self.status, self.error = "stopped", str(e)
            raise

    def close(self):
        with self.lock:
            if self.transport is not None:
                self.transport.close()
            if self.status in ("ready", "starting"):
                self.status = "stopped"


# ------------------------------------------------------------------------------------------------ the hub
def result_text(result: dict) -> str:
    """An MCP tool result as the text the model reads: the text blocks as they are, other blocks named."""
    parts = []
    for block in result.get("content") or []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            parts.append(str(block.get("text", "")))
        elif kind == "resource":
            res = block.get("resource") or {}
            parts.append(str(res["text"]) if "text" in res else f"[resource {res.get('uri', '')} "
                                                                  f"({res.get('mimeType', 'binary')}), not shown]")
        elif kind == "resource_link":
            parts.append(f"[resource link: {block.get('name') or ''} {block.get('uri', '')}]".replace("  ", " "))
        elif kind in ("image", "audio"):
            size = len(block.get("data") or "") * 3 // 4
            parts.append(f"[{kind} ({block.get('mimeType', '?')}, {size:,} bytes): not shown to the model]")
        else:
            parts.append(json.dumps(block, ensure_ascii=False))
    if not parts and result.get("structuredContent") is not None:
        parts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
    return "\n".join(parts)


def cache_images(result, directory):
    """Keep raster tool outputs outside chat history; never expose arbitrary tool paths."""
    images = []
    directory = Path(directory)
    types = {"image/png": ("png", b"\x89PNG\r\n\x1a\n"), "image/jpeg": ("jpg", b"\xff\xd8\xff"),
             "image/webp": ("webp", b"RIFF")}
    for block in result.get("content") or []:
        if not isinstance(block, dict):
            continue
        mime = block.get("mimeType")
        if block.get("type") != "image" or mime not in types or len(images) >= 4:
            continue
        encoded = block.get("data")
        if not isinstance(encoded, str) or len(encoded) > 24_000_000:
            continue
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError):
            continue
        ext, magic = types[mime]
        if not data.startswith(magic) or (ext == "webp" and data[8:12] != b"WEBP"):
            continue
        directory.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(data).hexdigest() + "." + ext
        target = directory / name
        if not target.exists():
            temporary = directory / (name + "." + os.urandom(8).hex() + ".tmp")
            try:
                temporary.write_bytes(data)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        images.append({"url": "tool-media/" + name, "mimeType": mime, "name": "tool-image." + ext})
    return images


def _clean(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(name)) or "x"


class McpHub:
    """Every configured server, and the merged, namespaced tool list the model sees."""

    def __init__(self, servers: dict[str, dict], settings: dict | None = None):
        self.media_dir = None
        self.settings = {**DEFAULTS, **(settings or {})}
        self.servers = {name: McpServer(name, cfg, self.settings) for name, cfg in servers.items()}
        self.threads: list[threading.Thread] = []
        self._routes: dict[str, tuple[McpServer, str]] = {}

    def start(self, wait: bool = False):
        """Start every server on its own thread (npx may download a package first: that must not delay the chat)."""
        self.threads = [threading.Thread(target=s.start, daemon=True, name=f"mcp-{s.name}")
                        for s in self.servers.values()]
        for t in self.threads:
            t.start()
        if wait:
            self.wait(None)

    def wait(self, timeout: float | None) -> None:
        """Wait (up to `timeout` s in all) for servers that are still starting."""
        end = None if timeout is None else time.monotonic() + timeout
        for t in self.threads:
            t.join(None if end is None else max(0.0, end - time.monotonic()))

    def routes(self) -> dict[str, tuple[McpServer, str]]:
        """tool name the model sees -> (server, the tool's own name).  A server that stopped after it was ready keeps
        its tools here: a call starts it again."""
        out = {}
        for s in self.servers.values():
            if s.status not in ("ready", "stopped"):
                continue
            for t in s.tools:
                name = f"{_clean(s.name)}__{_clean(t['name'])}"[:64]
                n = 2
                while name in out:                       # two names that differ only in cleaned characters
                    name = f"{name[:60]}_{n}"
                    n += 1
                out[name] = (s, t["name"])
        self._routes = out
        return out

    def resolve_route(self, name: str) -> tuple[McpServer | None, str]:
        """Resolves a tool name or alias to (server, tool_name).
        Supports:
        - exact route: 'math__calculate'
        - dot notation: 'math.calculate'
        - mcp prefix: 'mcp.math.calculate', 'mcp.math__calculate', 'mcp__math__calculate'
        - bare tool: 'calculate' (if unambiguous across servers)
        """
        if name in ("run_python", "python", "py", "run_code", "execute_code") and self.settings.get("codemode"):
            return None, "run_python"
        if name in ("run_javascript", "javascript", "js", "run_js") and self.settings.get("codemode"):
            return None, "run_javascript"
        routes = self._routes or self.routes()
        if name in routes:
            return routes[name]
        norm = name
        if norm.startswith("mcp."):
            norm = norm[4:]
        elif norm.startswith("mcp__"):
            norm = norm[5:]
        if norm in routes:
            return routes[norm]
        dot_as_route = norm.replace(".", "__")
        if dot_as_route in routes:
            return routes[dot_as_route]
        for r_name, (s, tool) in routes.items():
            clean_s = _clean(s.name)
            clean_t = _clean(tool)
            if norm in (f"{clean_s}.{clean_t}", f"{clean_s}_{clean_t}", f"{clean_s}__{clean_t}"):
                return (s, tool)
        matches = [(s, tool) for r_name, (s, tool) in routes.items() if _clean(tool) == norm or tool == norm]
        if len(matches) == 1:
            return matches[0]
        return (None, name)

    def template_tools(self, exclude=()) -> list[dict]:
        """The tools in the chat template's form ({name, description, parameters}); `exclude`: names the request
        brought itself (those win)."""
        out = []
        for name, (s, tool) in self.routes().items():
            if name in exclude or tool == "generate_plot":
                continue
            t = next(x for x in s.tools if x["name"] == tool)
            schema = t.get("inputSchema") if isinstance(t.get("inputSchema"), dict) else {}
            out.append({"name": name, "description": str(t.get("description") or t.get("title") or ""),
                        "parameters": schema or {"type": "object", "properties": {}}})
        return out

    def openai_tools(self) -> list[dict]:
        return [{"type": "function", "function": t} for t in self.template_tools()]

    def codemode_tools(self, exclude=(), lang: str | None = None) -> list[dict]:
        """A single execution tool for Code Mode with compact signatures of all MCP tools."""
        effective_lang = (lang or self.settings.get("codemode_lang") or "javascript").lower()
        if effective_lang in ("javascript", "js"):
            tool_name = "run_javascript"
            if tool_name in exclude:
                return []
            sigs = []
            for name, (s, tool) in self.routes().items():
                if tool == "generate_plot" or name == "generate_plot" or name.endswith("__generate_plot"):
                    continue
                t = next((x for x in s.tools if x["name"] == tool), None)
                if not t:
                    continue
                schema = t.get("inputSchema") if isinstance(t.get("inputSchema"), dict) else {}
                props = schema.get("properties") or {}
                args_str = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in props.items())
                desc = str(t.get("description") or t.get("title") or "").strip().replace("\n", " ")
                if desc:
                    first_sent = desc.split(". ")[0].rstrip(".")
                    desc = f": {first_sent[:90]}"
                sigs.append(f"  - mcp.{_clean(s.name)}.{_clean(tool)}({{{args_str}}}){desc}")

            tools_list_text = "\n".join(sigs) if sigs else "  (no MCP tools currently registered)"
            desc = (
                "Execute JavaScript (ES6) code in an isolated QuickJS sandbox to interact with MCP tools, perform calculations, or process data.\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. `run_javascript` is the ONLY valid function name for tool calls. You MUST call `run_javascript`.\n"
                "2. NEVER emit `<function=mcp...>` or invoke MCP functions directly as tool names.\n"
                "3. Variables defined in global scope persist across turns within the same conversation session.\n"
                "4. Expressions at the end of the script auto-return their value. `console.log(...)` is also captured.\n"
                "5. MCP tools are callable via `mcp.<server>.<tool>({ arg1: value, ... })`.\n"
                "6. Native JavaScript builtins available:\n"
                "   - Date and time: `new Date().toISOString()`, `new Date().toLocaleString()`, etc.\n"
                "   - Math: `Math.sqrt()`, `Math.sin()`, `Math.PI`, `Math.round()`, etc.\n"
                "   - Standard ES6: Object, Array, String, Number, JSON, RegExp, Map, Set.\n"
                "7. To render interactive plots, pass a Plotly figure object to `render_interactive_plot(fig)`.\n\n"
                "EXAMPLES:\n"
                "# Date and time:\n"
                "run_javascript(code='new Date().toLocaleString()')\n\n"
                "# Interactive Plotting:\n"
                "run_javascript(code='render_interactive_plot({ data: [{ x: [\"A\", \"B\", \"C\"], y: [10, 25, 15], type: \"bar\" }], layout: { title: \"Sample Bar Chart\" } })')\n\n"
                "# Chaining MCP tools and calculating in JS:\n"
                "run_javascript(code='const data = mcp.filesystem.read_file({ path: \"data.json\" });\\nconst parsed = JSON.parse(data.text);\\nparsed.filter(x => x.active).length;')\n\n"
                f"MCP server tools callable via `mcp.<server>.<tool>({{...}})`:\n{tools_list_text}"
            )
            return [{
                "name": "run_javascript",
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Self-contained JavaScript (ES6) code to execute."
                        }
                    },
                    "required": ["code"]
                }
            }]

        if "run_python" in exclude:
            return []
        sigs = []
        for name, (s, tool) in self.routes().items():
            if tool == "generate_plot" or name == "generate_plot" or name.endswith("__generate_plot"):
                continue
            t = next((x for x in s.tools if x["name"] == tool), None)
            if not t:
                continue
            schema = t.get("inputSchema") if isinstance(t.get("inputSchema"), dict) else {}
            props = schema.get("properties") or {}
            args_str = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in props.items())
            desc = str(t.get("description") or t.get("title") or "").strip().replace("\n", " ")
            if desc:
                first_sent = desc.split(". ")[0].rstrip(".")
                desc = f": {first_sent[:90]}"
            sigs.append(f"  - mcp.{_clean(s.name)}.{_clean(tool)}({args_str}){desc}")

        tools_list_text = "\n".join(sigs) if sigs else "  (no MCP tools currently registered)"
        desc = (
            "Execute Python code to interact with MCP tools, run calculations, or process data locally.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. `run_python` is the ONLY valid function name for tool calls. You MUST call `run_python`.\n"
            "2. NEVER emit `<function=mcp...>` or invoke MCP functions directly as tool names.\n"
            "3. Variables and symbols persist across turns within the same conversation session.\n"
            "4. Expressions at the end of the script auto-return their value (Jupyter/REPL style). `print(...)` or `result = ...` is also supported.\n"
            "5. Pre-imported in your environment (zero tokens needed, DO NOT import them):\n"
            "   - sp / sympy (e.g. sp.Symbol, sp.solve, sp.diff, sp.simplify)\n"
            "   - np / numpy, math, cmath, pint, z3, astropy\n"
            "   - pd / pandas, px (plotly.express), go (plotly.graph_objects)\n"
            "   - render_interactive_plot(fig) -> dict\n"
            "   - verify_equivalence(expr1, expr2) -> dict\n"
            "   - dimensional_analysis(lhs, rhs, variables) -> dict\n"
            "   - solve_and_derive(equation, solve_for=None, derive_wrt=None) -> dict\n"
            "   - compile_equation_to_code(expr, target_lang) -> dict (c, rust, py, js, julia)\n"
            "   - solve_constraints(variables, constraints, types=None, maximize=None, minimize=None) -> dict\n"
            "   - lint_python_code(code) -> dict\n"
            "   - calculate_time_delta(start, end, timezone1, timezone2, business_days_only) -> dict\n"
            "6. To generate charts, use pre-imported `px` or `go` directly (e.g. `fig = px.bar(...)`). NEVER write `import px` (`px` is a pre-imported variable, not a PyPI package). Pass your final figure to `render_interactive_plot(fig)` (or leave `fig` as the final expression, or call `fig.show()`).\n"
            "7. You may also import any PyPI package (resolved dynamically via uv in milliseconds).\n\n"
            "EXAMPLES:\n"
            "# Interactive Plotting:\n"
            "run_python(code='df = pd.DataFrame({\"Category\": [\"A\", \"B\", \"C\"], \"Values\": [10, 20, 15]})\\nfig = px.bar(df, x=\"Category\", y=\"Values\", title=\"Sample Chart\")\\nrender_interactive_plot(fig)')\n\n"
            "# Symbolic derivation:\n"
            "run_python(code='x = sp.Symbol(\"x\")\\nsp.diff(sp.sin(x) * sp.exp(x), x)')\n\n"
            "# Dimensional analysis:\n"
            "run_python(code='dimensional_analysis(\"T\", \"2*pi*sqrt(L/g)\", {\"T\": \"s\", \"L\": \"m\", \"g\": \"m/s^2\"})')\n\n"
            "# Constraint solving:\n"
            "run_python(code='solve_constraints([\"Alice\", \"Bob\"], [\"Alice < Bob\", \"Alice >= 9\", \"Bob <= 17\"])')\n\n"
            f"MCP server tools callable via `mcp.<server>.<tool>(...)`:\n{tools_list_text}"
        )
        return [{
            "name": "run_python",
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Self-contained Python code to execute."
                    },
                    "dependencies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional PyPI packages needed for execution (e.g. ['tabulate', 'pandas']). Installed dynamically by uv in milliseconds."
                    }
                },
                "required": ["code"]
            }
        }]

    def call_codemode(self, arguments: dict | str, cancel: threading.Event | None = None,
                      timeout_s: float | None = None, lang: str | None = None) -> dict:
        code = ""
        deps = None
        arg_lang = None
        if isinstance(arguments, str):
            code = arguments
        elif isinstance(arguments, dict):
            code = (arguments.get("code") or arguments.get("script") or arguments.get("python") or
                    arguments.get("javascript") or arguments.get("js") or
                    arguments.get("command") or arguments.get("input") or arguments.get("source") or "")
            if isinstance(arguments.get("dependencies"), list):
                deps = [str(d) for d in arguments["dependencies"]]
            arg_lang = arguments.get("language") or arguments.get("lang")

        effective_lang = (lang or arg_lang or self.settings.get("codemode_lang") or "javascript").lower()
        tool_name = "run_javascript" if effective_lang in ("javascript", "js") else "run_python"

        timeout = float(timeout_s if timeout_s is not None else self.settings.get("timeout_s", 60.0))
        session_id = str(arguments.get("session_id") or "default") if isinstance(arguments, dict) else "default"

        if effective_lang in ("javascript", "js"):
            from serve import js_sandbox
            code = js_sandbox.strip_markdown_fences(code)
            if not isinstance(code, str) or not code.strip():
                return {"server": "codemode", "tool": tool_name, "ok": False,
                        "text": f"error: {tool_name} requires a 'code' string argument", "chars": 50,
                        "truncated": False, "ms": 0}
            r = js_sandbox.execute_js_in_sandbox(code, self, timeout_s=timeout, cancel=cancel, session_id=session_id)
        else:
            from serve import mcp_runner
            code = mcp_runner.strip_markdown_fences(code)
            if not isinstance(code, str) or not code.strip():
                return {"server": "codemode", "tool": tool_name, "ok": False,
                        "text": f"error: {tool_name} requires a 'code' string argument", "chars": 52,
                        "truncated": False, "ms": 0}
            r = mcp_runner.execute_in_runner(code, self, timeout_s=timeout, cancel=cancel, dependencies=deps, session_id=session_id)

        text, ok = r["text"], r["ok"]
        cap = int(self.settings["max_result_chars"])
        full = len(text)
        truncated = cap > 0 and full > cap
        if truncated:
            text = text[:cap] + (f"\n\n[... truncated: the tool returned {full:,} characters; only the first "
                                 f"{cap:,} are shown]")
        out = {"server": "codemode", "tool": tool_name, "ok": ok, "text": text, "chars": full,
               "truncated": truncated, "ms": r["ms"]}
        res = r.get("result")
        if isinstance(res, dict) and res.get("type") == "application/vnd.plotly.v1+json":
            out["plot"] = res
        elif isinstance(r.get("plot"), dict):
            out["plot"] = r["plot"]
        return out

    def call(self, name: str, arguments: dict, cancel: threading.Event | None = None,
             timeout_s: float | None = None, default_lang: str | None = None) -> dict:
        """Run one tool -> {"ok", "text" (what the model reads, capped), "chars" (its full length), "truncated",
        "ms", "server", "tool"}.  Errors become text starting with "error:"; only McpCancelled is raised."""
        if name in CODEMODE_PYTHON_NAMES:
            return self.call_codemode(arguments, cancel, timeout_s=timeout_s, lang="python")
        if name in CODEMODE_JS_NAMES:
            return self.call_codemode(arguments, cancel, timeout_s=timeout_s, lang="javascript")
        if name in CODEMODE_GENERIC_NAMES:
            arg_lang = arguments.get("language") if isinstance(arguments, dict) else None
            effective = (arg_lang or default_lang or self.settings.get("codemode_lang") or "javascript").lower()
            return self.call_codemode(arguments, cancel, timeout_s=timeout_s,
                                      lang="javascript" if effective in ("javascript", "js") else "python")
        t0 = time.monotonic()
        server, tool = self.resolve_route(name)
        out = {"server": server.name if server else None, "tool": tool}
        if server is None:
            text, ok = f"error: there is no tool named {name!r}", False
        else:
            try:
                effective_timeout = float(timeout_s if timeout_s is not None else self.settings["timeout_s"])
                res = server.call(tool, arguments if isinstance(arguments, dict) else {},
                                  effective_timeout, cancel)
                text, ok = result_text(res), not res.get("isError")
                if ok and self.settings["image_results"] and self.media_dir is not None:
                    try:
                        images = cache_images(res, self.media_dir)
                    except OSError as e:
                        print(f"[strata] MCP image cache unavailable: {e}", file=sys.stderr, flush=True)
                    else:
                        if images:
                            out["images"] = images
                if not ok:
                    text = "error: " + (text or "the tool reported an error")
            except McpCancelled:
                raise
            except McpError as e:
                text, ok = f"error: {e}", False
        cap = int(self.settings["max_result_chars"])
        full = len(text)
        if cap > 0 and full > cap:
            text = text[:cap] + (f"\n\n[... truncated: the tool returned {full:,} characters; only the first "
                                 f"{cap:,} are shown]")
        if ok and out.get("images"):
            out["model_content"] = [{"type": "text", "text": text.replace(": not shown to the model]", ": image attached]")}] + [
                {"type": "image", "source": str(Path(self.media_dir) / im["url"].split("/")[-1])}
                for im in out["images"]]
        out.update(ok=ok, text=text, chars=full, truncated=cap > 0 and full > cap,
                   ms=int((time.monotonic() - t0) * 1000))
        return out

    def status(self) -> dict:
        """GET /mcp: every server, its state and its tools (for the web app)."""
        routes = self.routes()
        servers = []
        for s in self.servers.values():
            names = {tool: n for n, (srv, tool) in routes.items() if srv is s}
            servers.append({"name": s.name, "transport": s.kind, "status": s.status, "error": s.error,
                            "info": s.info,
                            "tools": [{"name": names.get(t["name"], t["name"]), "tool": t["name"],
                                       "description": str(t.get("description") or "")[:300]} for t in s.tools]})
        return {"servers": servers, "tools": len(routes),
                "settings": {k: self.settings[k] for k in ("timeout_s", "max_result_chars", "max_rounds", "codemode", "codemode_lang")}}

    def close(self):
        for s in self.servers.values():
            try:
                s.close()
            except Exception:  # noqa: BLE001 - closing is best effort
                pass
        try:
            from serve import mcp_runner
            mcp_runner.close_all_runners()
        except Exception:
            pass


# ------------------------------------------------------------------------------------------------ config
def _check_server(name, cfg, where) -> dict:
    if not isinstance(cfg, dict):
        raise SystemExit(f"[strata] {where}: MCP server {name!r} must be an object with \"command\" or \"url\"")
    if cfg.get("url"):
        if cfg.get("type") == "sse":
            raise SystemExit(f"[strata] {where}: MCP server {name!r} uses the old SSE transport (\"type\": \"sse\"); "
                             "Strata speaks Streamable HTTP - most servers offer it at /mcp")
        if not isinstance(cfg.get("headers") or {}, dict):
            raise SystemExit(f"[strata] {where}: MCP server {name!r}: \"headers\" must be an object")
        return cfg
    if not isinstance(cfg.get("command"), str) or not cfg["command"].strip():
        raise SystemExit(f"[strata] {where}: MCP server {name!r} needs \"command\" (a program to start) or \"url\"")
    if not isinstance(cfg.get("args") or [], list):
        raise SystemExit(f"[strata] {where}: MCP server {name!r}: \"args\" must be a list")
    if not isinstance(cfg.get("env") or {}, dict):
        raise SystemExit(f"[strata] {where}: MCP server {name!r}: \"env\" must be an object")
    return cfg


def servers_from(block, where: str) -> dict[str, dict]:
    """{"name": {...}} with the entries checked; `"disabled": true` leaves one out."""
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise SystemExit(f"[strata] {where}: the MCP servers must be an object {{\"name\": {{...}}}}")
    return {str(n): _check_server(n, c, where) for n, c in block.items()
            if not (isinstance(c, dict) and c.get("disabled") is True)}


def settings_from(cfg: dict) -> dict:
    """The run config's optional `"mcp"` block: timeout_s, max_result_chars, max_rounds, start_timeout_s, image_results."""
    out = {}
    block = dict(cfg.get("mcp") or {})
    for k in ("codemode", "code_mode"):
        if k in cfg and "codemode" not in block and "code_mode" not in block:
            block["codemode"] = cfg[k]
    for k in ("codemode_lang", "codemode_language"):
        if k in cfg and "codemode_lang" not in block and "codemode_language" not in block:
            block["codemode_lang"] = cfg[k]
    for key, value in block.items():
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if key == "image_results":
            if not isinstance(value, bool):
                raise SystemExit(f"[strata] config mcp.{key}={value!r}: expected a boolean")
            out[key] = value
        elif key in ("codemode", "code_mode"):
            if not isinstance(value, bool):
                raise SystemExit(f"[strata] config mcp.{key}={value!r}: expected a boolean")
            out["codemode"] = value
        elif key in ("codemode_lang", "codemode_language"):
            lang_val = str(value).lower()
            if lang_val not in ("python", "javascript", "js"):
                raise SystemExit(f"[strata] config mcp.{key}={value!r}: expected 'python' or 'javascript'")
            out["codemode_lang"] = "javascript" if lang_val in ("javascript", "js") else "python"
        elif key in ("timeout_s", "start_timeout_s"):
            if not number or value <= 0:
                raise SystemExit(f"[strata] config mcp.{key}={value!r}: expected a number of seconds > 0")
            out[key] = float(value)
        elif key in ("max_result_chars", "max_rounds"):
            if not number or value != int(value) or value < 1:
                raise SystemExit(f"[strata] config mcp.{key}={value!r}: expected a positive integer")
            out[key] = int(value)
        else:
            print(f"[strata] config mcp.{key}={value!r}: unknown key, ignored", flush=True)
    return out


def hub_from_config(cfg: dict, mcp_config_path: str | None = None) -> McpHub | None:
    """The run config's `"mcp_servers"` (or `"mcpServers"`) plus the servers in the --mcp-config file (Claude
    Desktop's format: {"mcpServers": {...}}); a name in both takes the file's entry.  None when there are none."""
    servers = {}
    servers.update(servers_from(cfg.get("mcp_servers"), "config mcp_servers"))
    servers.update(servers_from(cfg.get("mcpServers"), "config mcpServers"))
    file_settings = {}
    if mcp_config_path:
        try:
            with open(mcp_config_path, encoding="utf-8-sig") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            raise SystemExit(f"[strata] --mcp-config {mcp_config_path}: {e}") from None
        block = data.get("mcpServers", data.get("mcp_servers")) if isinstance(data, dict) else None
        if block is None:
            raise SystemExit(f"[strata] --mcp-config {mcp_config_path}: expected {{\"mcpServers\": {{...}}}}")
        servers.update(servers_from(block, f"--mcp-config {mcp_config_path}"))
        if isinstance(data, dict):
            if isinstance(data.get("mcp"), dict):
                file_settings = settings_from({"mcp": data["mcp"]})
            elif "codemode" in data or "code_mode" in data:
                file_settings = settings_from(data)
    if not servers:
        return None
    merged = {**file_settings, **settings_from(cfg)}
    return McpHub(servers, merged)


if __name__ == "__main__":                               # python -m serve.mcp config.json: list what a config offers
    hub = hub_from_config(json.load(open(sys.argv[1], encoding="utf-8-sig")) if len(sys.argv) > 1 else {},
                          sys.argv[2] if len(sys.argv) > 2 else None)
    if hub is None:
        sys.exit("no MCP servers configured")
    hub.start(wait=True)
    print(json.dumps(hub.status(), indent=1))
    hub.close()
