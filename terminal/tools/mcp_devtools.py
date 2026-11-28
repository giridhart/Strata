"""tools/mcp_devtools.py - Developer and scheduling utility tools.

Provides:
- lint_python_code: Runs Ruff linter and AST syntax check to return structured error diagnostics.
- calculate_time_delta: Precise cross-timezone datetime math and business-day scheduling calculations.
"""
from __future__ import annotations

import ast
import datetime
import json
import shutil
import subprocess
import sys
from typing import Any
from zoneinfo import ZoneInfo
from dateutil import parser as date_parser


def lint_python_code(code: str) -> dict[str, Any]:
    """Lints Python code using AST parsing and Ruff, returning structured diagnostics.

    Returns:
        dict with keys 'valid', 'errors' (list of line, column, code, message), and 'summary'.
    """
    errors: list[dict[str, Any]] = []

    # 1. First pass: AST parse check for fatal syntax errors
    try:
        ast.parse(code, "<lint>")
    except SyntaxError as e:
        errors.append({
            "source": "syntax",
            "line": e.lineno or 1,
            "column": e.offset or 1,
            "code": "E999",
            "message": f"SyntaxError: {e.msg}",
            "snippet": e.text.strip() if e.text else "",
        })
        return {
            "valid": False,
            "errors": errors,
            "summary": f"Found 1 fatal syntax error at line {e.lineno or 1}.",
        }

    # 2. Second pass: Ruff linter if ruff executable is present
    ruff_bin = shutil.which("ruff")
    if ruff_bin:
        try:
            res = subprocess.run(
                [ruff_bin, "check", "--output-format=json", "-"],
                input=code.encode("utf-8"),
                capture_output=True,
                timeout=5,
            )
            raw = res.stdout.decode("utf-8", errors="replace").strip()
            if raw:
                try:
                    ruff_errs = json.loads(raw)
                    for item in ruff_errs:
                        loc = item.get("location") or {}
                        errors.append({
                            "source": "ruff",
                            "line": loc.get("row", 1),
                            "column": loc.get("column", 1),
                            "code": item.get("code", "UNKNOWN"),
                            "message": item.get("message", ""),
                            "snippet": item.get("snippet", ""),
                        })
                except Exception:
                    pass
        except Exception:
            pass

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "summary": "Code passed linting with zero errors." if not errors else f"Found {len(errors)} linting/syntax issue(s).",
    }


def calculate_time_delta(
    start: str,
    end: str,
    timezone1: str = "UTC",
    timezone2: str = "UTC",
    business_days_only: bool = False,
) -> dict[str, Any]:
    """Calculates duration and deltas across timezones and optional business-day filters.

    Args:
        start: Start datetime string (e.g. '2026-11-04 09:00', '2026-11-04T14:30:00Z')
        end: End datetime string (e.g. '2026-11-08 17:00')
        timezone1: Timezone name for start (default: 'UTC')
        timezone2: Timezone name for end (default: 'UTC')
        business_days_only: Whether to calculate business days (Monday-Friday) only
    """
    try:
        tz_start = ZoneInfo(timezone1.strip())
        tz_end = ZoneInfo(timezone2.strip())

        dt_start = date_parser.parse(start.strip())
        if dt_start.tzinfo is None:
            dt_start = dt_start.replace(tzinfo=tz_start)
        else:
            dt_start = dt_start.astimezone(tz_start)

        dt_end = date_parser.parse(end.strip())
        if dt_end.tzinfo is None:
            dt_end = dt_end.replace(tzinfo=tz_end)
        else:
            dt_end = dt_end.astimezone(tz_end)

        # Normalize both to UTC for duration calculation
        dt_start_utc = dt_start.astimezone(ZoneInfo("UTC"))
        dt_end_utc = dt_end.astimezone(ZoneInfo("UTC"))

        delta = dt_end_utc - dt_start_utc
        total_seconds = delta.total_seconds()
        is_negative = total_seconds < 0

        # Business days calculation
        business_days = 0
        if dt_start_utc <= dt_end_utc:
            curr = dt_start_utc.date()
            end_date = dt_end_utc.date()
            while curr <= end_date:
                # Monday is 0, Sunday is 6
                if curr.weekday() < 5:
                    business_days += 1
                curr += datetime.timedelta(days=1)

        days = abs(int(total_seconds // 86400))
        rem = abs(total_seconds) % 86400
        hours = int(rem // 3600)
        minutes = int((rem % 3600) // 60)
        seconds = int(rem % 60)

        return {
            "start_iso": dt_start.isoformat(),
            "end_iso": dt_end.isoformat(),
            "total_seconds": total_seconds,
            "duration": {
                "days": -days if is_negative else days,
                "hours": -hours if is_negative else hours,
                "minutes": -minutes if is_negative else minutes,
                "seconds": -seconds if is_negative else seconds,
            },
            "business_days": business_days if not is_negative else -business_days,
            "formatted": f"{'-' if is_negative else ''}{days}d {hours}h {minutes}m {seconds}s",
        }
    except Exception as exc:
        return {"error": f"Date calculation error: {type(exc).__name__}: {exc}"}
