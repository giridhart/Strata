"""tools/mcp_plot.py - Model Context Protocol (MCP) server for clean SVG chart generation.

Provides a fast, zero-dependency plotting tool using Python's standard library:
- Line charts, area charts, bar graphs, and scatter plots.
- Intelligent layout engine: dynamic content-aware margins, automatic multi-row legend wrapping,
  smart label rotation (-35 deg), and tick stride to completely eliminate text collisions.
- Native browser hover tooltips (<title> tags) on every data point without requiring any JavaScript.
- Publication-quality modern SVG vector graphics with area gradients and rounded bars.
- Directly renders in Strata's native web chat previewer.

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
    "#06b6d4",  # Cyan
    "#ec4899",  # Pink
    "#f97316",  # Orange
    "#14b8a6",  # Teal
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


def _format_num(val: float) -> str:
    """Format numbers cleanly for axis ticks and labels."""
    if abs(val - round(val)) < 1e-9:
        val = round(val)
        if abs(val) >= 1_000_000:
            return f"{val / 1_000_000:.2g}M"
        if abs(val) >= 10_000:
            return f"{val / 1_000:.2g}k"
        return f"{int(val)}"
    if abs(val) >= 1_000_000:
        return f"{val / 1_000_000:.2f}M"
    if abs(val) >= 10_000:
        return f"{val / 1_000:.2f}k"
    if abs(val) < 0.001 and val != 0:
        return f"{val:.2e}"
    return f"{val:.3g}"


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
    width: int = 680,
    height: int = 420,
) -> tuple[str, str]:
    """Generates an anti-collision, responsive SVG chart string and text summary."""
    chart_type = chart_type.lower()
    if chart_type not in ("line", "bar", "scatter", "area"):
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
    num_x = max_len

    # Prepare formatted ticks and calculate max tick string length
    tick_strs = [_format_num(t) for t in ticks]
    max_tick_len = max(len(s) for s in tick_strs) if tick_strs else 3

    # Dynamic left margin based on tick width + optional Y-axis label
    pad_left = max(68, (30 if y_label else 14) + int(max_tick_len * 8.5) + 12)
    pad_right = 32

    # --- Title & Legend Layout (Collision-free wrapping) ---
    leg_items: list[tuple[str, str]] = []
    for i, s in enumerate(series):
        name = s.get("name") or (f"Series {i+1}" if len(series) > 1 else "")
        if name:
            leg_items.append((name, PALETTE[i % len(PALETTE)]))

    # Wrap legend items into rows if needed
    avail_legend_w = width - pad_left - pad_right
    legend_rows: list[list[tuple[str, str, float]]] = []
    current_row: list[tuple[str, str, float]] = []
    current_row_w = 0.0

    for name, color in leg_items:
        item_w = len(name) * 7.5 + 28
        if current_row and (current_row_w + item_w > avail_legend_w):
            legend_rows.append(current_row)
            current_row = [(name, color, item_w)]
            current_row_w = item_w
        else:
            current_row.append((name, color, item_w))
            current_row_w += item_w
    if current_row:
        legend_rows.append(current_row)

    num_leg_rows = len(legend_rows)
    pad_top = (36 if title else 16) + (num_leg_rows * 22) + 12

    # --- X-axis Layout & Collision Avoidance ---
    labels = [str(x_labels[i]) if (x_labels and i < len(x_labels)) else str(i + 1) for i in range(num_x)]
    max_x_len = max(len(l) for l in labels) if labels else 1

    temp_plot_w = width - pad_left - pad_right
    slot_w = temp_plot_w / max(1, num_x)
    needed_label_w = max_x_len * 7.2 + 14

    rotate_x = (slot_w < needed_label_w) or (num_x > 7 and max_x_len > 4)
    extra_bottom = min(95, int(max_x_len * 6.0)) if rotate_x else 0

    # Stride calculation for dense points to prevent overlapping
    label_stride = 1
    if slot_w < 18 or (not rotate_x and slot_w < 35 and num_x > 12):
        target_step = 28.0 if rotate_x else 50.0
        label_stride = max(1, math.ceil(target_step / max(1.0, slot_w)))

    pad_bottom = 36 + extra_bottom + (24 if x_label else 0)

    # Calculate final plot dimensions
    plot_w = width - pad_left - pad_right
    plot_h = max(180, height - pad_top - pad_bottom)
    total_height = pad_top + plot_h + pad_bottom

    def to_y(val: float) -> float:
        denom = max(1e-9, y_max - y_min)
        ratio = (val - y_min) / denom
        return (pad_top + plot_h) - (ratio * plot_h)

    # Compute X coordinates
    x_positions: list[float] = []
    if chart_type == "bar":
        b_slot = plot_w / max(1, num_x)
        for i in range(num_x):
            x_positions.append(pad_left + (i + 0.5) * b_slot)
    else:
        for i in range(num_x):
            denom = max(1, num_x - 1)
            x_positions.append(pad_left + (i / denom) * plot_w)

    # --- SVG Building ---
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {total_height}" '
        f'style="width:100%;height:auto;max-width:{width}px;font-family:system-ui,-apple-system,\'Segoe UI\',Roboto,sans-serif;">',
        '  <defs>',
        '    <style>',
        '      .grid { stroke: currentColor; stroke-opacity: 0.12; stroke-width: 1; stroke-dasharray: 3 3; }',
        '      .axis { stroke: currentColor; stroke-opacity: 0.35; stroke-width: 1.5; }',
        '      .tick-text { fill: currentColor; fill-opacity: 0.70; font-size: 11px; }',
        '      .axis-label { fill: currentColor; fill-opacity: 0.85; font-size: 12px; font-weight: 500; }',
        '      .chart-title { fill: currentColor; font-size: 16px; font-weight: 600; }',
        '      .legend-text { fill: currentColor; fill-opacity: 0.85; font-size: 12px; }',
        '      .val-label { fill: currentColor; fill-opacity: 0.80; font-size: 10.5px; font-weight: 600; font-family: ui-monospace,monospace; }',
        '      .plot-mark:hover circle { r: 6; stroke-width: 2.5; }',
        '      .plot-mark:hover rect { opacity: 1; filter: brightness(1.1); }',
        '    </style>',
    ]

    # Gradients for lines and bars
    for i, color in enumerate(PALETTE):
        svg.append(
            f'    <linearGradient id="area_grad_{i}" x1="0" y1="0" x2="0" y2="1">'
            f'      <stop offset="0%" stop-color="{color}" stop-opacity="0.28"/>'
            f'      <stop offset="100%" stop-color="{color}" stop-opacity="0.01"/>'
            f'    </linearGradient>'
        )
        svg.append(
            f'    <linearGradient id="bar_grad_{i}" x1="0" y1="0" x2="0" y2="1">'
            f'      <stop offset="0%" stop-color="{color}" stop-opacity="0.95"/>'
            f'      <stop offset="100%" stop-color="{color}" stop-opacity="0.80"/>'
            f'    </linearGradient>'
        )
    svg.append('  </defs>')

    # Title
    if title:
        svg.append(f'  <text x="{pad_left}" y="26" class="chart-title">{_escape(title)}</text>')

    # Legend rows
    if legend_rows:
        cur_y = 26 + (20 if title else 0)
        for row in legend_rows:
            lx = pad_left
            for name, color, item_w in row:
                svg.append(f'  <circle cx="{lx + 5}" cy="{cur_y + 11}" r="5" fill="{color}"/>')
                svg.append(f'  <text x="{lx + 16}" y="{cur_y + 15}" class="legend-text">{_escape(name)}</text>')
                lx += item_w
            cur_y += 22

    # Horizontal Gridlines & Y-ticks
    for t_val, t_str in zip(ticks, tick_strs):
        cy = to_y(t_val)
        if pad_top <= cy <= (pad_top + plot_h):
            svg.append(f'  <line x1="{pad_left}" y1="{cy:.1f}" x2="{width - pad_right}" y2="{cy:.1f}" class="grid"/>')
            svg.append(f'  <text x="{pad_left - 8}" y="{cy + 4:.1f}" text-anchor="end" class="tick-text">{t_str}</text>')

    # X-labels with collision avoidance (rotation & stride)
    base_y = pad_top + plot_h
    for i, cx in enumerate(x_positions):
        if (i % label_stride != 0) and (i != num_x - 1):
            continue
        label_text = labels[i]
        if rotate_x:
            rot = f'transform="rotate(-35 {cx:.1f} {base_y + 16})"'
            svg.append(
                f'  <text x="{cx:.1f}" y="{base_y + 16}" text-anchor="end" class="tick-text" {rot}>'
                f'{_escape(label_text)}</text>'
            )
        else:
            svg.append(
                f'  <text x="{cx:.1f}" y="{base_y + 18}" text-anchor="middle" class="tick-text">'
                f'{_escape(label_text)}</text>'
            )

    # Axes
    svg.append(f'  <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{base_y}" class="axis"/>')
    svg.append(f'  <line x1="{pad_left}" y1="{base_y}" x2="{width - pad_right}" y2="{base_y}" class="axis"/>')

    # Y-axis Label
    if y_label:
        y_label_x = max(16, pad_left - int(max_tick_len * 8.5) - 22)
        y_mid = pad_top + plot_h / 2
        rot = f'transform="rotate(-90 {y_label_x} {y_mid:.1f})"'
        svg.append(f'  <text x="{y_label_x}" y="{y_mid:.1f}" text-anchor="middle" class="axis-label" {rot}>{_escape(y_label)}</text>')

    # X-axis Label
    if x_label:
        x_label_y = total_height - 10
        svg.append(f'  <text x="{pad_left + plot_w / 2}" y="{x_label_y}" text-anchor="middle" class="axis-label">{_escape(x_label)}</text>')

    # --- Data Rendering ---
    show_direct_values = (num_x * len(series) <= 16) and (slot_w >= 26)

    if chart_type == "bar":
        num_series = len(series)
        b_slot = plot_w / max(1, num_x)
        bar_group_w = b_slot * 0.76
        bar_w = bar_group_w / num_series
        zero_y = to_y(max(0.0, y_min))

        for s_idx, s in enumerate(series):
            color = PALETTE[s_idx % len(PALETTE)]
            grad_id = f"bar_grad_{s_idx % len(PALETTE)}"
            s_name = s.get("name") or f"Series {s_idx + 1}"
            vals = [float(v) for v in s.get("values", [])]

            for i, val in enumerate(vals):
                bx = (pad_left + i * b_slot + (b_slot - bar_group_w) / 2) + s_idx * bar_w
                y_pos = to_y(val)
                bh = max(1.0, abs(zero_y - y_pos))
                top_y = min(y_pos, zero_y)
                val_fmt = _format_num(val)
                lbl_fmt = labels[i] if i < len(labels) else str(i + 1)

                svg.append('  <g class="plot-mark">')
                svg.append(f'    <title>{_escape(s_name)} [{_escape(lbl_fmt)}]: {val_fmt}</title>')
                svg.append(
                    f'    <rect x="{bx:.1f}" y="{top_y:.1f}" width="{max(1.0, bar_w - 2):.1f}" '
                    f'height="{bh:.1f}" rx="4" ry="4" fill="url(#{grad_id})"/>'
                )
                if show_direct_values and bar_w >= 22:
                    v_y = top_y - 6 if val >= 0 else top_y + bh + 12
                    svg.append(
                        f'    <text x="{bx + bar_w/2:.1f}" y="{v_y:.1f}" text-anchor="middle" class="val-label">'
                        f'{val_fmt}</text>'
                    )
                svg.append('  </g>')

    elif chart_type in ("line", "area"):
        is_area = (chart_type == "area")
        base_line_y = to_y(max(0.0, y_min))

        for s_idx, s in enumerate(series):
            color = PALETTE[s_idx % len(PALETTE)]
            area_grad = f"area_grad_{s_idx % len(PALETTE)}"
            s_name = s.get("name") or f"Series {s_idx + 1}"
            vals = [float(v) for v in s.get("values", [])]
            pts = [(x_positions[i], to_y(v), v, i) for i, v in enumerate(vals) if i < len(x_positions)]
            if not pts:
                continue

            # Area path
            first_x, last_x = pts[0][0], pts[-1][0]
            d_line = " ".join([f"{'M' if k == 0 else 'L'} {px:.1f} {py:.1f}" for k, (px, py, _, _) in enumerate(pts)])
            d_area = f"{d_line} L {last_x:.1f} {base_line_y:.1f} L {first_x:.1f} {base_line_y:.1f} Z"

            svg.append(f'  <path d="{d_area}" fill="url(#{area_grad})"/>')
            svg.append(
                f'  <path d="{d_line}" fill="none" stroke="{color}" stroke-width="2.5" '
                f'stroke-linecap="round" stroke-linejoin="round"/>'
            )

            # Dots & Tooltips
            for px, py, val, i in pts:
                val_fmt = _format_num(val)
                lbl_fmt = labels[i] if i < len(labels) else str(i + 1)
                svg.append('  <g class="plot-mark">')
                svg.append(f'    <title>{_escape(s_name)} [{_escape(lbl_fmt)}]: {val_fmt}</title>')
                svg.append(f'    <circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
                if show_direct_values and len(series) <= 2:
                    svg.append(f'    <text x="{px:.1f}" y="{py - 8:.1f}" text-anchor="middle" class="val-label">{val_fmt}</text>')
                svg.append('  </g>')

    elif chart_type == "scatter":
        for s_idx, s in enumerate(series):
            color = PALETTE[s_idx % len(PALETTE)]
            s_name = s.get("name") or f"Series {s_idx + 1}"
            vals = [float(v) for v in s.get("values", [])]

            for i, val in enumerate(vals):
                if i < len(x_positions):
                    px = x_positions[i]
                    py = to_y(val)
                    val_fmt = _format_num(val)
                    lbl_fmt = labels[i] if i < len(labels) else str(i + 1)
                    svg.append('  <g class="plot-mark">')
                    svg.append(f'    <title>{_escape(s_name)} [{_escape(lbl_fmt)}]: {val_fmt}</title>')
                    svg.append(f'    <circle cx="{px:.1f}" cy="{py:.1f}" r="5.5" fill="{color}" fill-opacity="0.85" stroke="#ffffff" stroke-width="1.5"/>')
                    svg.append('  </g>')

    svg.append("</svg>")
    svg_str = "\n".join(svg)

    summary = (
        f"Generated {chart_type} chart with {len(series)} series, {max_len} data points. "
        f"Range: [{_format_num(min_val)}, {_format_num(max_val)}]."
    )
    return svg_str, summary


TOOLS = [
    {
        "name": "generate_plot",
        "description": (
            "Generate a publication-ready, anti-collision SVG vector chart (line, area, bar, or scatter plot). "
            "Features smart auto-rotating labels, multi-row legend wrapping, native browser hover tooltips, "
            "and gradient fills. Returns an ```svg fenced block that automatically renders in Strata's web chat preview. "
            "Ideal for benchmark charts, latency comparisons, loss curves, and token rate metrics."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Chart title (e.g. 'Generation Speed (tok/s) vs Context Length')"},
                "chart_type": {
                    "type": "string",
                    "enum": ["line", "area", "bar", "scatter"],
                    "description": "Chart type: 'line', 'area', 'bar', or 'scatter' (default: 'line')",
                },
                "x": {
                    "type": "array",
                    "items": {"type": ["string", "number"]},
                    "description": "Labels or categories for the X-axis (e.g. ['4K', '8K', '16K', '32K', '64K'] or ['Model A', 'Model B'])",
                },
                "series": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Series name for the legend (e.g. 'Intel Arc B580')"},
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
                "x_label": {"type": "string", "description": "Axis label for the X-axis (e.g. 'Context Window')"},
                "y_label": {"type": "string", "description": "Axis label for the Y-axis (e.g. 'Tokens / Second')"},
                "width": {"type": "integer", "description": "Chart width in pixels (default: 680)"},
                "height": {"type": "integer", "description": "Chart height in pixels (default: 420)"},
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
                "serverInfo": {"name": "plot", "version": "0.2.0"},
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
                width = int(args.get("width") or 680)
                height = int(args.get("height") or 420)

                svg_content, summary = generate_svg_chart(
                    series=series,
                    chart_type=chart_type,
                    title=title,
                    x_labels=x_vals,
                    x_label=x_label,
                    y_label=y_label,
                    width=width,
                    height=height,
                )

                # Return fenced svg block so Strata web preview recognizes and auto-renders it
                formatted = f"```svg\n{svg_content}\n```\n\n{summary}"
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
    data = [{"name": "Intel Arc B580", "values": [112, 108, 98, 85]}]
    svg, summary = generate_svg_chart(data, chart_type="line", title="Rate", x_labels=["8K", "16K", "32K", "64K"])
    assert "<svg" in svg and "</svg>" in svg
    assert "Intel Arc B580" in svg
    assert "<title>" in svg

    # Bar chart with rotation test (many labels or tight slots)
    dense_labels = [f"Month-{i:02d}" for i in range(1, 10)]
    dense_data = [{"name": "Sales", "values": list(range(10, 19))}]
    b_svg, _ = generate_svg_chart(dense_data, chart_type="bar", title="Bar", x_labels=dense_labels)
    assert "<rect" in b_svg
    assert "rotate(-35" in b_svg

    # Area chart
    a_svg, _ = generate_svg_chart(data, chart_type="area", title="Area")
    assert "area_grad_" in a_svg

    # Scatter chart
    s_svg, _ = generate_svg_chart(data, chart_type="scatter", title="Scatter")
    assert "<circle" in s_svg

    print("Self-test passed for mcp_plot.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_selftest()
    else:
        run_stdio()
