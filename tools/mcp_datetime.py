"""tools/mcp_datetime.py - Model Context Protocol (MCP) server for date and time operations.

Provides safe, deterministic time and date calculations over stdio transport:
- Get current date and time (local or specified timezone).
- Calculate difference between dates.
- Add or subtract intervals (days, hours, minutes) from a date.
- Determine day of week, week number, and leap year status.

Usage:
    python tools/mcp_datetime.py          # Stdio MCP server (JSON-RPC 2.0 lines)
    python tools/mcp_datetime.py --test   # Self-test
"""
from __future__ import annotations

import calendar
from datetime import datetime, timezone, timedelta
import json
import sys
from typing import Any

PROTOCOL = "2025-06-18"

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # type: ignore


def _parse_iso(s: str) -> datetime:
    """Parses various date/time formats into a datetime object."""
    s = s.strip()
    if s.lower() in ("now", "today", "current"):
        return datetime.now(timezone.utc).astimezone()
    # Try common formats
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    # ISO 8601 fallback
    return datetime.fromisoformat(s)


def get_current_time(tz_name: str | None = None) -> dict[str, Any]:
    """Returns the current date and time formatted in the given timezone or local time."""
    tz = None
    if tz_name and tz_name.strip():
        name = tz_name.strip()
        if name.upper() == "UTC":
            tz = timezone.utc
        elif ZoneInfo is not None:
            try:
                tz = ZoneInfo(name)
            except Exception:
                pass
    now = datetime.now(tz or timezone.utc).astimezone(tz) if tz else datetime.now().astimezone()
    return {
        "iso": now.isoformat(),
        "formatted": now.strftime("%A, %B %d, %Y %I:%M:%S %p %Z"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "timezone": str(now.tzinfo or "Local"),
        "utc_offset": now.strftime("%z"),
        "unix_timestamp": int(now.timestamp()),
    }


def date_difference(start_date: str, end_date: str) -> dict[str, Any]:
    """Calculates the duration between two dates."""
    d1 = _parse_iso(start_date)
    d2 = _parse_iso(end_date)
    diff = d2 - d1
    total_seconds = int(diff.total_seconds())
    days = diff.days
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    return {
        "start": d1.isoformat(),
        "end": d2.isoformat(),
        "total_days": days,
        "total_seconds": total_seconds,
        "summary": f"{days} days, {hours} hours, {minutes} minutes" if total_seconds >= 0 else f"-{abs(days)} days, {hours} hours",
    }


def date_add(date_str: str, days: int = 0, hours: int = 0, minutes: int = 0) -> dict[str, Any]:
    """Adds or subtracts days, hours, or minutes to a date."""
    base = _parse_iso(date_str)
    delta = timedelta(days=days, hours=hours, minutes=minutes)
    res = base + delta
    return {
        "input": base.isoformat(),
        "result_iso": res.isoformat(),
        "result_formatted": res.strftime("%A, %B %d, %Y %I:%M:%S %p"),
        "day_of_week": res.strftime("%A"),
    }


def date_info(date_str: str) -> dict[str, Any]:
    """Returns metadata about a specific date (leap year, day of week, day of year)."""
    d = _parse_iso(date_str)
    is_leap = calendar.isleap(d.year)
    day_of_year = d.timetuple().tm_yday
    week_num = d.isocalendar()[1]
    return {
        "date": d.strftime("%Y-%m-%d"),
        "day_of_week": d.strftime("%A"),
        "is_weekend": d.weekday() >= 5,
        "day_of_year": day_of_year,
        "week_number": week_num,
        "is_leap_year": is_leap,
    }


TOOLS = [
    {
        "name": "get_current_time",
        "description": "Get the current real-world date, time, and day of week. Optionally specify a timezone like 'UTC', 'America/New_York', or 'Asia/Tokyo'.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Optional timezone name (e.g. 'UTC', 'America/New_York', 'Europe/London'). Defaults to local time.",
                }
            },
        },
    },
    {
        "name": "date_difference",
        "description": "Calculate the exact duration, days, and hours between two dates/times.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "Starting date (e.g. '2026-01-15' or 'now')"},
                "end_date": {"type": "string", "description": "Ending date (e.g. '2026-11-02')"},
            },
            "required": ["start_date", "end_date"],
        },
    },
    {
        "name": "date_add",
        "description": "Add or subtract days, hours, or minutes from a date (use negative numbers to subtract).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Base date or 'now' (e.g. '2026-11-02')"},
                "days": {"type": "integer", "description": "Number of days to add (positive or negative)"},
                "hours": {"type": "integer", "description": "Number of hours to add"},
                "minutes": {"type": "integer", "description": "Number of minutes to add"},
            },
            "required": ["date"],
        },
    },
    {
        "name": "date_info",
        "description": "Get calendar details for a date (day of week, whether it is a weekend, leap year, week number).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Date to inspect, e.g. '2026-11-02'"}
            },
            "required": ["date"],
        },
    },
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
                "serverInfo": {"name": "datetime", "version": "0.1.0"},
            },
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        try:
            if name == "get_current_time":
                res = get_current_time(args.get("timezone"))
                return {"jsonrpc": "2.0", "id": rid, "result": text_response(json.dumps(res, indent=2), error=False)}
            if name == "date_difference":
                res = date_difference(str(args.get("start_date")), str(args.get("end_date")))
                return {"jsonrpc": "2.0", "id": rid, "result": text_response(json.dumps(res, indent=2), error=False)}
            if name == "date_add":
                res = date_add(str(args.get("date")), int(args.get("days") or 0), int(args.get("hours") or 0), int(args.get("minutes") or 0))
                return {"jsonrpc": "2.0", "id": rid, "result": text_response(json.dumps(res, indent=2), error=False)}
            if name == "date_info":
                res = date_info(str(args.get("date")))
                return {"jsonrpc": "2.0", "id": rid, "result": text_response(json.dumps(res, indent=2), error=False)}
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": f"unknown tool: {name!r}"}}
        except Exception as e:
            return {"jsonrpc": "2.0", "id": rid, "result": text_response(f"error: {e}", error=True)}

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
    t = get_current_time("UTC")
    assert "iso" in t and "unix_timestamp" in t
    diff = date_difference("2026-01-01", "2026-01-11")
    assert diff["total_days"] == 10
    add = date_add("2026-01-01", days=5)
    assert "2026-01-06" in add["result_iso"]
    info = date_info("2026-11-02")
    assert info["day_of_week"] == "Monday"
    print("Self-test passed for mcp_datetime.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_selftest()
    else:
        run_stdio()
