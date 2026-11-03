"""tools/mcp_linter.py - Model Context Protocol (MCP) server for server-side code linting and structural validation.

Provides fast, zero-dependency validation and linting for LLM code generation (standard library only):
- HTML: Detects unclosed tags, mismatched nesting, unclosed <script>/<style> blocks, truncated tags,
  and void element misuse (<br>, <img>, <input>). Offers auto-repair suggestions for truncated HTML.
- Python: Full AST syntax parsing, detecting syntax errors, unclosed brackets, and indentation issues.
- JSON: Validates JSON structure, detects trailing commas, single-quoted strings, and unclosed brackets.
- JavaScript / CSS: Validates bracket/brace/parenthesis balance and string/comment termination.

Usage:
    python tools/mcp_linter.py          # Stdio MCP server (JSON-RPC 2.0 lines)
    python tools/mcp_linter.py --test   # Run self-tests
"""
from __future__ import annotations

import ast
from html.parser import HTMLParser
import json
import re
import sys
from typing import Any

PROTOCOL = "2025-06-18"

# HTML5 void elements that must not have closing tags
HTML5_VOID_ELEMENTS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}

# Elements that can optionally be self-closing in SVG / MathML inside HTML
SVG_MATHML_CONTAINERS = {"svg", "math"}


class HTMLStructureParser(HTMLParser):
    """Parses HTML and tracks tag nesting stacks, line positions, and unclosed blocks."""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.tag_stack: list[dict[str, Any]] = []  # [{tag, line, col}]
        self.errors: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []
        self.in_raw_block: str | None = None  # 'script' or 'style'
        self.raw_block_start: tuple[int, int] | None = None
        self.has_doctype = False

    def handle_decl(self, decl: str) -> None:
        if decl.lower().startswith("doctype"):
            self.has_doctype = True

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        line, col = self.getpos()

        # Check for unclosed script/style blocks
        if tag_lower in ("script", "style"):
            self.in_raw_block = tag_lower
            self.raw_block_start = (line, col)

        if tag_lower in HTML5_VOID_ELEMENTS:
            # Void element: does not need a closing tag
            return

        self.tag_stack.append({"tag": tag_lower, "line": line, "col": col})

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        line, col = self.getpos()

        if tag_lower in ("script", "style") and self.in_raw_block == tag_lower:
            self.in_raw_block = None
            self.raw_block_start = None

        if tag_lower in HTML5_VOID_ELEMENTS:
            self.warnings.append({
                "line": line,
                "column": col,
                "message": f"Void element <{tag_lower}> should not have a closing tag </{tag_lower}>.",
                "severity": "warning",
            })
            return

        if not self.tag_stack:
            self.errors.append({
                "line": line,
                "column": col,
                "message": f"Unexpected closing tag </{tag_lower}> with no matching open tag.",
                "severity": "error",
            })
            return

        # Check if matches the top of stack
        top = self.tag_stack[-1]
        if top["tag"] == tag_lower:
            self.tag_stack.pop()
            return

        # Tag mismatch: search down the stack to see if an ancestor matches
        ancestor_idx = -1
        for idx in range(len(self.tag_stack) - 1, -1, -1):
            if self.tag_stack[idx]["tag"] == tag_lower:
                ancestor_idx = idx
                break

        if ancestor_idx != -1:
            # We found a matching open tag higher up, meaning intervening tags were never closed!
            unclosed = self.tag_stack[ancestor_idx + 1 :]
            unclosed_names = ", ".join(f"<{item['tag']}> (line {item['line']})" for item in unclosed)
            self.errors.append({
                "line": line,
                "column": col,
                "message": (
                    f"Mismatched closing tag </{tag_lower}>; expected closing for intervening tags: {unclosed_names}."
                ),
                "severity": "error",
            })
            # Pop up through the matching ancestor
            self.tag_stack = self.tag_stack[:ancestor_idx]
        else:
            # Stray closing tag that was never opened
            self.errors.append({
                "line": line,
                "column": col,
                "message": f"Stray closing tag </{tag_lower}> found (expected </{top['tag']}> to match line {top['line']}).",
                "severity": "error",
            })


def validate_html(code: str, auto_repair: bool = False) -> dict[str, Any]:
    """Validates HTML markup structure and tag balance."""
    parser = HTMLStructureParser()
    lines = code.splitlines()

    try:
        parser.feed(code)
        parser.close()
    except Exception as e:
        parser.errors.append({
            "line": len(lines),
            "column": 0,
            "message": f"HTML parser encountered fatal syntax error: {e}",
            "severity": "error",
        })

    # Check for unclosed script/style block
    if parser.in_raw_block and parser.raw_block_start:
        r_line, r_col = parser.raw_block_start
        parser.errors.append({
            "line": r_line,
            "column": r_col,
            "message": (
                f"Critical: Unclosed <{parser.in_raw_block}> tag opened on line {r_line}. "
                f"All subsequent HTML is being parsed as raw code and will not render!"
            ),
            "severity": "error",
        })

    # Check for remaining unclosed tags
    unclosed_tags: list[dict[str, Any]] = []
    if parser.tag_stack:
        for item in reversed(parser.tag_stack):
            unclosed_tags.append(item)
            parser.errors.append({
                "line": item["line"],
                "column": item["col"],
                "message": f"Unclosed tag <{item['tag']}> opened on line {item['line']}.",
                "severity": "error",
            })

    # Check for mid-tag truncation at the end of the file
    truncated_tag_match = re.search(r"<[a-zA-Z0-9_\-]+(?:\s+[^>]*$|$)", code.rstrip())
    if truncated_tag_match and not code.rstrip().endswith(">"):
        parser.errors.append({
            "line": len(lines),
            "column": len(lines[-1]) if lines else 0,
            "message": "HTML appears to be cut off / truncated mid-tag at the end of output.",
            "severity": "error",
        })

    all_issues = parser.errors + parser.warnings
    is_valid = len(parser.errors) == 0

    # Auto-repair generation
    repaired_code = None
    if auto_repair or unclosed_tags or parser.in_raw_block:
        rep_parts = [code]
        if parser.in_raw_block:
            rep_parts.append(f"\n</{parser.in_raw_block}>")
        for item in unclosed_tags:
            rep_parts.append(f"</{item['tag']}>")
        repaired_code = "".join(rep_parts)

    unclosed_names = [t["tag"] for t in unclosed_tags]
    if is_valid:
        summary = f"HTML is structurally valid ({len(lines)} lines, {len(code)} chars, no tag errors detected)."
    else:
        err_count = len(parser.errors)
        summary = (
            f"HTML validation failed with {err_count} structural error(s). "
            f"Unclosed tags: {unclosed_names if unclosed_names else 'none'}."
        )

    return {
        "valid": is_valid,
        "language": "html",
        "lines": len(lines),
        "chars": len(code),
        "errors": all_issues,
        "unclosed_tags": unclosed_names,
        "summary": summary,
        "repaired_code": repaired_code if not is_valid else None,
    }


def validate_python(code: str) -> dict[str, Any]:
    """Validates Python syntax via ast.parse()."""
    lines = code.splitlines()
    errors = []

    try:
        ast.parse(code)
        is_valid = True
        summary = f"Python code is syntactically valid ({len(lines)} lines, no syntax errors detected)."
    except SyntaxError as e:
        is_valid = False
        line_num = e.lineno or len(lines)
        col_offset = e.offset or 0
        snippet = e.text.strip() if e.text else ""
        errors.append({
            "line": line_num,
            "column": col_offset,
            "message": f"SyntaxError: {e.msg}" + (f" -> `{snippet}`" if snippet else ""),
            "severity": "error",
        })
        summary = f"Python syntax error on line {line_num}: {e.msg}."
    except Exception as e:
        is_valid = False
        errors.append({"line": len(lines), "column": 0, "message": str(e), "severity": "error"})
        summary = f"Python parsing error: {e}"

    return {
        "valid": is_valid,
        "language": "python",
        "lines": len(lines),
        "chars": len(code),
        "errors": errors,
        "unclosed_tags": [],
        "summary": summary,
        "repaired_code": None,
    }


def validate_json(code: str) -> dict[str, Any]:
    """Validates JSON structure and detects common formatting defects."""
    lines = code.splitlines()
    errors = []

    try:
        json.loads(code)
        is_valid = True
        summary = f"JSON is syntactically valid ({len(lines)} lines, well-formed)."
    except json.JSONDecodeError as e:
        is_valid = False
        msg = e.msg
        # Check for common LLM JSON defects
        hint = ""
        if "Expecting property name enclosed in double quotes" in msg:
            hint = " (Check for single quotes around keys or trailing commas before '}' / ']')"
        elif "Extra data" in msg:
            hint = " (Multiple JSON objects or unparsed text after JSON end)"
        elif "Unterminated string" in msg:
            hint = " (Unescaped newline inside a string literal or missing closing quote)"

        errors.append({
            "line": e.lineno,
            "column": e.colno,
            "message": f"JSONDecodeError: {msg}{hint}",
            "severity": "error",
        })
        summary = f"JSON invalid on line {e.lineno}, col {e.colno}: {msg}."

    return {
        "valid": is_valid,
        "language": "json",
        "lines": len(lines),
        "chars": len(code),
        "errors": errors,
        "unclosed_tags": [],
        "summary": summary,
        "repaired_code": None,
    }


def validate_brackets(code: str, language: str = "javascript") -> dict[str, Any]:
    """Checks bracket/brace/parenthesis balance for JS/CSS and generic code."""
    lines = code.splitlines()
    stack = []
    pairs = {")": "(", "}": "{", "]": "["}
    openers = set(pairs.values())
    closers = set(pairs.keys())

    errors = []
    in_single_str = False
    in_double_str = False
    in_template_str = False
    in_line_comment = False
    in_block_comment = False

    i = 0
    line_no = 1
    col_no = 0
    n = len(code)

    while i < n:
        c = code[i]
        col_no += 1
        if c == "\n":
            line_no += 1
            col_no = 0
            in_line_comment = False
            i += 1
            continue

        # Handle comments
        if not (in_single_str or in_double_str or in_template_str):
            if in_block_comment:
                if c == "*" and i + 1 < n and code[i + 1] == "/":
                    in_block_comment = False
                    i += 2
                    col_no += 1
                    continue
                i += 1
                continue
            elif in_line_comment:
                i += 1
                continue
            elif c == "/" and i + 1 < n and code[i + 1] == "/":
                in_line_comment = True
                i += 2
                col_no += 1
                continue
            elif c == "/" and i + 1 < n and code[i + 1] == "*":
                in_block_comment = True
                i += 2
                col_no += 1
                continue

        # Handle strings
        if c == "\\" and (in_single_str or in_double_str or in_template_str):
            i += 2
            col_no += 1
            continue
        if c == "'" and not (in_double_str or in_template_str):
            in_single_str = not in_single_str
        elif c == '"' and not (in_single_str or in_template_str):
            in_double_str = not in_double_str
        elif c == "`" and not (in_single_str or in_double_str):
            in_template_str = not in_template_str

        # Check brackets outside strings/comments
        if not (in_single_str or in_double_str or in_template_str or in_block_comment or in_line_comment):
            if c in openers:
                stack.append((c, line_no, col_no))
            elif c in closers:
                if not stack:
                    errors.append({
                        "line": line_no,
                        "column": col_no,
                        "message": f"Unmatched closing bracket '{c}' with no matching opening bracket.",
                        "severity": "error",
                    })
                else:
                    op, op_line, op_col = stack.pop()
                    if pairs[c] != op:
                        errors.append({
                            "line": line_no,
                            "column": col_no,
                            "message": f"Mismatched bracket: expected '{op}' to match line {op_line}, but found '{c}'.",
                            "severity": "error",
                        })

        i += 1

    if in_template_str:
        errors.append({"line": line_no, "column": col_no, "message": "Unterminated template literal (`)", "severity": "error"})
    if in_block_comment:
        errors.append({"line": line_no, "column": col_no, "message": "Unterminated block comment (/* ... */)", "severity": "error"})

    for op, op_line, op_col in reversed(stack):
        errors.append({
            "line": op_line,
            "column": op_col,
            "message": f"Unclosed bracket '{op}' opened on line {op_line}.",
            "severity": "error",
        })

    is_valid = len(errors) == 0
    summary = (
        f"{language.capitalize()} brackets and blocks are balanced ({len(lines)} lines)."
        if is_valid
        else f"{language.capitalize()} has {len(errors)} bracket/structure error(s)."
    )

    return {
        "valid": is_valid,
        "language": language,
        "lines": len(lines),
        "chars": len(code),
        "errors": errors,
        "unclosed_tags": [],
        "summary": summary,
        "repaired_code": None,
    }


def lint_code(code: str, language: str = "html", auto_repair: bool = False) -> dict[str, Any]:
    """Main linting router."""
    lang = (language or "html").lower().strip()
    if lang in ("html", "htm", "xml", "svg"):
        return validate_html(code, auto_repair=auto_repair)
    if lang in ("python", "py"):
        return validate_python(code)
    if lang in ("json",):
        return validate_json(code)
    if lang in ("javascript", "js", "typescript", "ts", "css"):
        return validate_brackets(code, language=lang)
    # Default to HTML validator if unknown or contains HTML tags
    if "<" in code and ">" in code:
        return validate_html(code, auto_repair=auto_repair)
    return validate_brackets(code, language=lang)


TOOLS = [
    {
        "name": "lint_code",
        "description": (
            "Validate and lint generated code (HTML, Python, JSON, JavaScript, CSS). "
            "Detects structural defects, unclosed tags (like missing </div> or </script>), "
            "mismatched nesting, truncated blocks, and syntax errors. "
            "Ideal for checking large HTML files, full-stack templates, and complex scripts."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The complete source code or markup string to validate.",
                },
                "language": {
                    "type": "string",
                    "enum": ["html", "python", "json", "javascript", "css"],
                    "description": "Language of the code (default: 'html').",
                },
                "auto_repair": {
                    "type": "boolean",
                    "description": "If true, returns a repaired_code string with missing closing tags appended.",
                },
            },
            "required": ["code"],
        },
    }
]


def text_response(t: str, error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": t}], "isError": error}


def handle_request(msg: dict) -> dict | None:
    """Processes one JSON-RPC 2.0 message."""
    method = msg.get("method")
    rid = msg.get("id")
    params = msg.get("params") or {}

    if rid is None:
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {
                "protocolVersion": params.get("protocolVersion", PROTOCOL),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "linter", "version": "0.1.0"},
            },
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "lint_code":
            try:
                code = str(args.get("code") or "")
                language = str(args.get("language") or "html")
                auto_repair = bool(args.get("auto_repair", False))

                res = lint_code(code, language=language, auto_repair=auto_repair)

                # Format human & model readable summary
                err_count = len(res.get("errors", []))
                status_str = "Passed (0 errors)" if res["valid"] else f"Failed ({err_count} errors)"
                formatted_lines = [
                    f"### Code Linter Result: {language.upper()} ({'VALID' if res['valid'] else 'ISSUES FOUND'})",
                    f"- **Status**: {status_str}",
                    f"- **Length**: {res['lines']} lines ({res['chars']} characters)",
                ]
                if res.get("unclosed_tags"):
                    formatted_lines.append(f"- **Unclosed tags**: {', '.join(res['unclosed_tags'])}")

                if res.get("errors"):
                    formatted_lines.append("\n**Detected Errors:**")
                    for err in res["errors"]:
                        formatted_lines.append(f"- Line {err['line']}, Col {err['column']}: {err['message']}")

                if res.get("repaired_code"):
                    formatted_lines.append(
                        "\n**Auto-repair available:** Closing tags have been appended to resolve unclosed blocks."
                    )

                full_text = "\n".join(formatted_lines)
                print(f"[linter] validated {language} ({res['lines']} lines): valid={res['valid']}", file=sys.stderr, flush=True)
                return {"jsonrpc": "2.0", "id": rid, "result": text_response(full_text, error=False)}
            except Exception as e:
                print(f"[linter error] {e}", file=sys.stderr, flush=True)
                return {"jsonrpc": "2.0", "id": rid, "result": text_response(f"error: {e}", error=True)}

        return {
            "jsonrpc": "2.0",
            "id": rid,
            "error": {"code": -32602, "message": f"unknown tool: {name!r}"},
        }

    return {
        "jsonrpc": "2.0",
        "id": rid,
        "error": {"code": -32601, "message": f"unsupported method: {method!r}"},
    }


def run_stdio():
    for f in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(f, "reconfigure"):
            f.reconfigure(encoding="utf-8")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        ans = handle_request(msg)
        if ans is not None:
            sys.stdout.write(json.dumps(ans) + "\n")
            sys.stdout.flush()


def run_selftest():
    # Valid HTML
    html_ok = "<!doctype html><html><head><title>Test</title></head><body><p>Hello <img src='a.png'></p></body></html>"
    r = lint_code(html_ok, "html")
    assert r["valid"] is True, f"Expected valid, got {r}"

    # Unclosed tag
    html_bad = "<div class='container'><p>Unclosed paragraph"
    r2 = lint_code(html_bad, "html")
    assert r2["valid"] is False
    assert "p" in r2["unclosed_tags"] and "div" in r2["unclosed_tags"]

    # Unclosed script
    html_script = "<div><script>console.log('hi');<p>Text</p>"
    r3 = lint_code(html_script, "html")
    assert r3["valid"] is False
    assert any("script" in err["message"] for err in r3["errors"])

    # Python valid & invalid
    assert lint_code("def foo():\n    return 42\n", "python")["valid"] is True
    assert lint_code("def foo(\n", "python")["valid"] is False

    # JSON valid & invalid
    assert lint_code('{"name": "strata", "count": 10}', "json")["valid"] is True
    assert lint_code('{"name": "strata",}', "json")["valid"] is False  # trailing comma

    print("Self-test passed for mcp_linter.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_selftest()
    else:
        run_stdio()
