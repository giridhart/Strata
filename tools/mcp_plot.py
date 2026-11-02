"""tools/mcp_plot.py - Model Context Protocol (MCP) server for clean SVG chart generation.

Provides a fast, zero-dependency plotting tool using Python's standard library:
- Line charts, bar graphs, and scatter plots.
- Automatic axis scaling, ticks, grid lines, and legends.
- Generates clean, responsive SVG vector graphics (3-5 KB).
- Compatible with Strata's native web chat previewer.

Usage:
    python tools/mcp_plot.py          # Stdio MCP server (JSON-RPC 2.0 lines)
    python tools/mcp_plot.py --test   # Run self-tests
"""
from __future__ import annotations

import json
import math
import sys
from typing import Any

PROTOCOL = "2025-06-18"

PALETTE = [
    "#3b82f6",  # Blue
    "#10b981",  # Emerald Green
    "#8b5cf6",  # Purple
    "#f59e0b",  # Amber
    "#ef4444",  # Red
    "#06b6d4",  # Cyan
    "#ec4899",  # Pink
]


def _escape(s: Any) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _nice_bounds(y_min: float, y_max: float) -> tuple[float, float, list[float]]:
    """Calculates clean axis minimum, maximum, and tick intervals."""
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    span = y_max - y_min
    raw_step = span / 4.0
    power = 10.0 ** math.floor(math.log10(max(1e-9, raw_step)))
    fraction = raw_step / power

    if fraction < 1.5:
        step = 1.0 * power
    elif fraction < 3.0:
        step = 2.0 * power
    elif fraction < 7.0:
        step = 5.0 * power
    else:
        step = 10.0 * power

    nice_min = math.floor(y_min / step) * step
    nice_max = math.ceil(y_max / step) * step

    # Generate ticks
    ticks: list[float] = []
    curr = nice_min
    while curr <= nice_max + (step * 0.01):
        ticks.append(round(curr, 8))
        curr += step

    return nice_min, nice_max, ticks


def generate_svg_chart(
    series: list[dict[str, Any]],
    chart_type: str = "line",
    title: str = "",
    x_labels: list[Any] | None = None,
    x_label: str = "",
    y_label: str = "",
    width: int = 640,
    height: int = 380,
) -> tuple[str, str]:
    """Generates a responsive SVG chart string and text summary."""
    chart_type = chart_type.lower()
    if chart_type not in ("line", "bar", "scatter"):
        chart_type = "line"

    # Extract all values for bounds calculation
    all_vals: list[float] = []
    max_len = 0
    for s in series:
        vals = [float(v) for v in s.get("values", [])]
        all_vals.extend(vals)
        max_len = max(max_len, len(vals))

    if not all_vals:
        raise ValueError("no numeric values found in series")

    min_val, max_val = min(all_vals), max(all_vals)
    # For bar charts, baseline at 0 if positive
    if chart_type == "bar" and min_val > 0:
        min_val = 0.0

    y_min, y_max, ticks = _nice_bounds(min_val, max_val)

    # Margins and layout
    pad_left = 65
    pad_right = 30
    pad_top = 55
    pad_bottom = 50

    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    def to_y(val: float) -> float:
        denom = max(1e-9, y_max - y_min)
        ratio = (val - y_min) / denom
        return (pad_top + plot_h) - (ratio * plot_h)

    # SVG assembly
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;height:auto;max-width:{width}px;font-family:system-ui,-apple-system,sans-serif;">',
        '  <defs>',
        '    <style>',
        '      .grid { stroke: currentColor; stroke-opacity: 0.12; stroke-width: 1; }',
        '      .axis { stroke: currentColor; stroke-opacity: 0.35; stroke-width: 1.5; }',
        '      .tick-text { fill: currentColor; fill-opacity: 0.65; font-size: 11px; }',
        '      .axis-label { fill: currentColor; fill-opacity: 0.85; font-size: 12px; font-weight: 500; }',
        '      .chart-title { fill: currentColor; font-size: 15px; font-weight: 600; }',
        '      .legend-text { fill: currentColor; fill-opacity: 0.8; font-size: 12px; }',
        '    </style>',
        '  </defs>',
    ]

    # Title
    if title:
        svg.append(f'  <text x="{pad_left}" y="28" class="chart-title">{_escape(title)}</text>')

    # Legend (top right)
    leg_x = width - pad_right
    leg_items = []
    for i, s in enumerate(series):
        name = s.get("name") or (f"Series {i+1}" if len(series) > 1 else "")
        if name:
            color = PALETTE[i % len(PALETTE)]
            leg_items.append((name, color))

    if leg_items:
        lx = pad_left + 15
        for name, color in leg_items:
            svg.append(f'  <circle cx="{lx}" cy="45" r="4.5" fill="{color}"/>')
            svg.append(f'  <text x="{lx + 10}" y="49" class="legend-text">{_escape(name)}</text>')
            lx += len(name) * 8 + 28

    # Horizontal Gridlines & Y-ticks
    for t_val in ticks:
        cy = to_y(t_val)
        if pad_top <= cy <= (pad_top + plot_h):
            svg.append(f'  <line x1="{pad_left}" y1="{cy:.1f}" x2="{width - pad_right}" y2="{cy:.1f}" class="grid"/>')
            val_str = f"{t_val:.4g}"
            svg.append(f'  <text x="{pad_left - 8}" y="{cy + 4:.1f}" text-anchor="end" class="tick-text">{val_str}</text>')

    # X-labels
    num_x = max_len
    x_positions: list[float] = []
    if chart_type == "bar":
        slot_w = plot_w / max(1, num_x)
        for i in range(num_x):
            x_positions.append(pad_left + (i + 0.5) * slot_w)
    else:
        for i in range(num_x):
            denom = max(1, num_x - 1)
            x_positions.append(pad_left + (i / denom) * plot_w)

    for i, cx in enumerate(x_positions):
        label_text = str(x_labels[i]) if (x_labels and i < len(x_labels)) else str(i + 1)
        svg.append(f'  <text x="{cx:.1f}" y="{pad_top + plot_h + 18}" text-anchor="middle" class="tick-text">{_escape(label_text)}</text>')

    # Axes
    base_y = pad_top + plot_h
    svg.append(f'  <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{base_y}" class="axis"/>')
    svg.append(f'  <line x1="{pad_left}" y1="{base_y}" x2="{width - pad_right}" y2="{base_y}" class="axis"/>')

    # Axis Labels
    if y_label:
        rot = f'transform="rotate(-90 18 {pad_top + plot_h / 2})"'
        svg.append(f'  <text x="18" y="{pad_top + plot_h / 2}" text-anchor="middle" class="axis-label" {rot}>{_escape(y_label)}</text>')
    if x_label:
        svg.append(f'  <text x="{pad_left + plot_w / 2}" y="{height - 10}" text-anchor="middle" class="axis-label">{_escape(x_label)}</text>')

    # Data Rendering
    if chart_type == "bar":
        num_series = len(series)
        slot_w = plot_w / max(1, num_x)
        bar_group_w = slot_w * 0.72
        bar_w = bar_group_w / num_series

        for s_idx, s in enumerate(series):
            color = PALETTE[s_idx % len(PALETTE)]
            vals = [float(v) for v in s.get("values", [])]
            for i, val in enumerate(vals):
                bx = (pad_left + i * slot_w + (slot_w - bar_group_w) / 2) + s_idx * bar_w
                y_pos = to_y(val)
                zero_y = to_y(max(0.0, y_min))
                bh = abs(zero_y - y_pos)
                top_y = min(y_pos, zero_y)
                svg.append(f'  <rect x="{bx:.1f}" y="{top_y:.1f}" width="{max(1.0, bar_w - 2):.1f}" height="{bh:.1f}" rx="2" fill="{color}" opacity="0.9"/>')

    elif chart_type == "line":
        for s_idx, s in enumerate(series):
            color = PALETTE[s_idx % len(PALETTE)]
            vals = [float(v) for v in s.get("values", [])]
            pts = [(x_positions[i], to_y(v)) for i, v in enumerate(vals) if i < len(x_positions)]
            if not pts:
                continue

            # Path line
            d_cmd = " ".join([f"{'M' if k == 0 else 'L'} {px:.1f} {py:.1f}" for k, (px, py) in enumerate(pts)])
            svg.append(f'  <path d="{d_cmd}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>')

            # Dots
            for px, py in pts:
                svg.append(f'  <circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}" stroke="#fff" stroke-width="1.5"/>')

    elif chart_type == "scatter":
        for s_idx, s in enumerate(series):
            color = PALETTE[s_idx % len(PALETTE)]
            vals = [float(v) for v in s.get("values", [])]
            for i, val in enumerate(vals):
                if i < len(x_positions):
                    px = x_positions[i]
                    py = to_y(val)
                    svg.append(f'  <circle cx="{px:.1f}" cy="{py:.1f}" r="5.5" fill="{color}" fill-opacity="0.8" stroke="#fff" stroke-width="1.5"/>')

    svg.append("</svg>")
    svg_str = "\n".join(svg)

    summary = (
        f"Generated {chart_type} chart with {len(series)} series, {max_len} data points. "
        f"Range: [{min_val:.4g}, {max_val:.4g}]."
    )
    return svg_str, summary


TOOLS = [
    {
        "name": "generate_plot",
        "description": (
            "Generate a clean, publication-ready SVG vector chart (line, bar, or scatter plot). "
            "Returns an SVG image block that renders directly in Strata's web chat preview. "
            "Ideal for benchmarks, latency comparisons, distributions, and token rate metrics."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Chart title (e.g. 'Token Rate vs Context')"},
                "chart_type": {
                    "type": "string",
                    "enum": ["line", "bar", "scatter"],
                    "description": "Chart type: 'line', 'bar', or 'scatter' (default: 'line')",
                },
                "x": {
                    "type": "array",
                    "items": {"type": ["string", "number"]},
                    "description": "Labels or numbers for the X-axis (e.g. ['8K', '16K', '32K', '64K'] or [1, 2, 3])",
                },
                "series": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Series name for the legend"},
                            "values": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Numeric data points",
                            },
                        },
                        "required": ["values"],
                    },
                    "description": "One or more data series to plot",
                },
                "x_label": {"type": "string", "description": "Description for the X-axis"},
                "y_label": {"type": "string", "description": "Description for the Y-axis (e.g. 'tok/s' or 'ms')"},
            },
            "required": ["series"],
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
                "serverInfo": {"name": "plot", "version": "0.1.0"},
            },
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "generate_plot":
            try:
                series = args.get("series") or []
                chart_type = str(args.get("chart_type") or "line")
                title = str(args.get("title") or "")
                x_vals = args.get("x")
                x_label = str(args.get("x_label") or "")
                y_label = str(args.get("y_label") or "")

                svg_content, summary = generate_svg_chart(
                    series=series,
                    chart_type=chart_type,
                    title=title,
                    x_labels=x_vals,
                    x_label=x_label,
                    y_label=y_label,
                )

                # Return fenced XML block so Strata web preview recognizes it as an SVG image
                formatted = f"```xml\n{svg_content}\n```\n\n{summary}"
                print(f"[plot] generated {chart_type} chart: {summary}", file=sys.stderr, flush=True)
                return {"jsonrpc": "2.0", "id": rid, "result": text_response(formatted, error=False)}
            except Exception as e:
                print(f"[plot error] {e}", file=sys.stderr, flush=True)
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
    data = [{"name": "Xe2 B580", "values": [112, 108, 98, 85]}]
    svg, summary = generate_svg_chart(data, chart_type="line", title="Rate", x_labels=["8K", "16K", "32K", "64K"])
    assert "<svg" in svg and "</svg>" in svg
    assert "Xe2 B580" in svg

    # Bar chart
    b_svg, _ = generate_svg_chart(data, chart_type="bar", title="Bar")
    assert "<rect" in b_svg

    # Scatter chart
    s_svg, _ = generate_svg_chart(data, chart_type="scatter", title="Scatter")
    assert "<circle" in s_svg

    print("Self-test passed for mcp_plot.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_selftest()
    else:
        run_stdio()
