from __future__ import annotations

import argparse
import html
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd

from gnprsid.common.io import ensure_dir, write_json
from gnprsid.common.paths import dataset_paths
from gnprsid.grpo.reward_trace import TRACE_GROUP_SIZE_ENV


TOP_LEVEL_FIELDS = [
    "format_reward",
    "reciprocal_rank_reward",
    "soft_hit_reward",
    "prefix_match_reward",
    "diversity_reward",
    "total_reward",
]

FORMAT_FIELDS = [
    "single_line_score",
    "valid_count_score",
    "exact_ten_score",
]

PAPER_PALETTE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
]


def _load_trace_rows(trace_path: str | Path) -> list[dict]:
    trace_path = Path(trace_path)
    trace_files = [trace_path] if trace_path.is_file() else sorted(trace_path.glob("*.jsonl"))
    rows: list[dict] = []
    for file_path in trace_files:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        raise FileNotFoundError(f"No reward trace rows found under {trace_path}")
    rows.sort(
        key=lambda row: (
            int(row.get("time_ns", 0)),
            int(row.get("pid", 0)),
            int(row.get("local_record_index", 0)),
        )
    )
    return rows


def _resolve_group_size(rows: list[dict], group_size: int | None) -> int:
    if group_size is not None:
        return int(group_size)
    hints = [int(row["group_size_hint"]) for row in rows if row.get("group_size_hint")]
    if not hints:
        raise ValueError(
            f"Could not infer group size from reward traces. Pass --group-size or set {TRACE_GROUP_SIZE_ENV} during training."
        )
    return Counter(hints).most_common(1)[0][0]


def _build_step_frame(rows: list[dict], group_size: int) -> pd.DataFrame:
    payload: list[dict] = []
    cumulative_sums = {field: 0.0 for field in TOP_LEVEL_FIELDS + FORMAT_FIELDS}
    cumulative_count = 0

    for start in range(0, len(rows), group_size):
        step_rows = rows[start : start + group_size]
        if not step_rows:
            continue

        cumulative_count += len(step_rows)
        row_payload: dict[str, float | int] = {
            "synthetic_step": (start // group_size) + 1,
            "sample_count": len(step_rows),
        }

        for field in TOP_LEVEL_FIELDS + FORMAT_FIELDS:
            step_sum = sum(float(item.get(field, 0.0) or 0.0) for item in step_rows)
            step_mean = step_sum / len(step_rows)
            cumulative_sums[field] += step_sum
            row_payload[f"step_sum_{field}"] = step_sum
            row_payload[f"step_mean_{field}"] = step_mean
            row_payload[f"cumulative_sum_{field}"] = cumulative_sums[field]
            row_payload[f"cumulative_mean_{field}"] = cumulative_sums[field] / cumulative_count

        payload.append(row_payload)

    return pd.DataFrame(payload)


def _nice_number(value: float, round_up: bool) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    fraction = value / (10**exponent)
    if round_up:
        if fraction <= 1:
            nice_fraction = 1
        elif fraction <= 2:
            nice_fraction = 2
        elif fraction <= 5:
            nice_fraction = 5
        else:
            nice_fraction = 10
    else:
        if fraction < 1.5:
            nice_fraction = 1
        elif fraction < 3:
            nice_fraction = 2
        elif fraction < 7:
            nice_fraction = 5
        else:
            nice_fraction = 10
    return nice_fraction * (10**exponent)


def _build_y_ticks(values: list[float], tick_count: int = 5) -> tuple[list[float], float, float]:
    min_value = min(values)
    max_value = max(values)
    if math.isclose(min_value, max_value):
        padding = max(abs(min_value) * 0.1, 0.1)
        min_value -= padding
        max_value += padding

    if min_value >= 0:
        min_value = 0.0

    span = max_value - min_value
    tick_spacing = _nice_number(span / max(tick_count - 1, 1), round_up=True)
    lower = math.floor(min_value / tick_spacing) * tick_spacing
    upper = math.ceil(max_value / tick_spacing) * tick_spacing
    ticks: list[float] = []
    value = lower
    limit = upper + tick_spacing * 0.5
    while value <= limit:
        ticks.append(round(value, 10))
        value += tick_spacing
    return ticks, lower, upper


def _build_x_ticks(x_values: list[int], max_ticks: int = 6) -> list[int]:
    if len(x_values) <= max_ticks:
        return x_values
    indices = {
        round(index * (len(x_values) - 1) / (max_ticks - 1))
        for index in range(max_ticks)
    }
    return [x_values[index] for index in sorted(indices)]


def _format_tick(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _build_polyline(
    x_values: list[int],
    y_values: list[float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: int,
    top: int,
    plot_width: int,
    plot_height: int,
) -> str:
    if len(x_values) == 1:
        x = left + plot_width / 2
        y = top + plot_height / 2
        return f"{x:.2f},{y:.2f}"

    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)
    points = []
    for x_value, y_value in zip(x_values, y_values):
        x = left + plot_width * ((x_value - x_min) / x_span)
        y = top + plot_height - plot_height * ((y_value - y_min) / y_span)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _render_svg_chart(
    title: str,
    x_values: list[int],
    series_map: dict[str, list[float]],
    y_axis_label: str,
) -> str:
    width = 980
    height = 420
    left = 84
    right = 28
    top = 28
    bottom = 62
    plot_width = width - left - right
    plot_height = height - top - bottom

    all_values = [value for values in series_map.values() for value in values]
    y_ticks, y_min, y_max = _build_y_ticks(all_values)
    x_ticks = _build_x_ticks(x_values)
    x_min = x_values[0]
    x_max = x_values[-1]

    legend_items = []
    polylines = []
    for index, (label, values) in enumerate(series_map.items()):
        color = PAPER_PALETTE[index % len(PAPER_PALETTE)]
        polyline = _build_polyline(
            x_values=x_values,
            y_values=values,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            left=left,
            top=top,
            plot_width=plot_width,
            plot_height=plot_height,
        )
        polylines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="1.6" '
            f'stroke-linecap="round" stroke-linejoin="round" points="{polyline}" />'
        )
        legend_items.append(
            f'<span style="display:inline-block;margin:0 18px 10px 0;font-size:14px;">'
            f'<span style="display:inline-block;width:18px;height:3px;background:{color};margin-right:8px;vertical-align:middle;"></span>'
            f"{html.escape(label)}</span>"
        )

    x_tick_lines = []
    x_tick_labels = []
    for tick in x_ticks:
        x = left + plot_width * ((tick - x_min) / max(x_max - x_min, 1e-9))
        x_tick_lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#e8e8e8" stroke-width="1" />'
        )
        x_tick_labels.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle" font-size="12" fill="#4a4a4a">{tick}</text>'
        )

    y_tick_lines = []
    y_tick_labels = []
    for tick in y_ticks:
        y = top + plot_height - plot_height * ((tick - y_min) / max(y_max - y_min, 1e-9))
        y_tick_lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e8e8e8" stroke-width="1" />'
        )
        y_tick_labels.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="#4a4a4a">{html.escape(_format_tick(tick))}</text>'
        )

    return (
        f"<section style=\"margin-bottom:36px;\">"
        f"<h2 style=\"font-size:18px;font-weight:600;margin:0 0 6px;\">{html.escape(title)}</h2>"
        f"<div style=\"font-size:13px;color:#555;margin-bottom:12px;\">Synthetic steps {x_min} to {x_max}</div>"
        f"<svg viewBox=\"0 0 {width} {height}\" style=\"width:100%;border:1px solid #d8d8d8;background:#fff;\">"
        f"{''.join(x_tick_lines)}"
        f"{''.join(y_tick_lines)}"
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#5a5a5a" stroke-width="1.2" />'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#5a5a5a" stroke-width="1.2" />'
        f"{''.join(polylines)}"
        f"{''.join(x_tick_labels)}"
        f"{''.join(y_tick_labels)}"
        f'<text x="{left + plot_width / 2:.2f}" y="{height - 14}" text-anchor="middle" font-size="13" fill="#333">Synthetic Step</text>'
        f'<text x="18" y="{top + plot_height / 2:.2f}" text-anchor="middle" font-size="13" fill="#333" transform="rotate(-90 18 {top + plot_height / 2:.2f})">{html.escape(y_axis_label)}</text>'
        f"</svg>"
        f"<div style=\"font-size:13px;margin-top:10px;line-height:1.5;\">{''.join(legend_items)}</div>"
        f"</section>"
    )


def _render_compact_svg_chart(
    title: str,
    x_values: list[int],
    y_values: list[float],
    y_axis_label: str,
    color: str,
) -> str:
    width = 440
    height = 260
    left = 62
    right = 18
    top = 24
    bottom = 48
    plot_width = width - left - right
    plot_height = height - top - bottom

    y_ticks, y_min, y_max = _build_y_ticks(y_values)
    x_ticks = _build_x_ticks(x_values, max_ticks=4)
    x_min = x_values[0]
    x_max = x_values[-1]
    polyline = _build_polyline(
        x_values=x_values,
        y_values=y_values,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        left=left,
        top=top,
        plot_width=plot_width,
        plot_height=plot_height,
    )

    x_tick_lines = []
    x_tick_labels = []
    for tick in x_ticks:
        x = left + plot_width * ((tick - x_min) / max(x_max - x_min, 1e-9))
        x_tick_lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#ededed" stroke-width="1" />'
        )
        x_tick_labels.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 20}" text-anchor="middle" font-size="11" fill="#4a4a4a">{tick}</text>'
        )

    y_tick_lines = []
    y_tick_labels = []
    for tick in y_ticks:
        y = top + plot_height - plot_height * ((tick - y_min) / max(y_max - y_min, 1e-9))
        y_tick_lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#ededed" stroke-width="1" />'
        )
        y_tick_labels.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11" fill="#4a4a4a">{html.escape(_format_tick(tick))}</text>'
        )

    svg = (
        f"<svg viewBox=\"0 0 {width} {height}\" style=\"width:100%;border:1px solid #d8d8d8;background:#fff;\">"
        f"{''.join(x_tick_lines)}"
        f"{''.join(y_tick_lines)}"
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#5a5a5a" stroke-width="1.1" />'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#5a5a5a" stroke-width="1.1" />'
        f'<polyline fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" points="{polyline}" />'
        f"{''.join(x_tick_labels)}"
        f"{''.join(y_tick_labels)}"
        f'<text x="{left + plot_width / 2:.2f}" y="{height - 10}" text-anchor="middle" font-size="12" fill="#333">Synthetic Step</text>'
        f'<text x="16" y="{top + plot_height / 2:.2f}" text-anchor="middle" font-size="12" fill="#333" transform="rotate(-90 16 {top + plot_height / 2:.2f})">{html.escape(y_axis_label)}</text>'
        f"</svg>"
    )
    return (
        "<div style=\"background:#fff;border:1px solid #ddd;padding:12px 12px 10px;box-shadow:0 1px 2px rgba(0,0,0,0.04);\">"
        f"<h3 style=\"font-size:15px;font-weight:600;margin:0 0 8px;\">{html.escape(title)}</h3>"
        f"{svg}"
        "</div>"
    )


def _render_split_component_panel(
    title: str,
    x_values: list[int],
    series_map: dict[str, list[float]],
    y_axis_label: str,
) -> str:
    cards = []
    for index, (label, values) in enumerate(series_map.items()):
        color = PAPER_PALETTE[index % len(PAPER_PALETTE)]
        cards.append(
            _render_compact_svg_chart(
                title=label,
                x_values=x_values,
                y_values=values,
                y_axis_label=y_axis_label,
                color=color,
            )
        )
    return (
        f"<section style=\"margin-bottom:12px;\">"
        f"<h2 style=\"font-size:18px;font-weight:600;margin:0 0 6px;\">{html.escape(title)}</h2>"
        f"<div style=\"font-size:13px;color:#555;margin-bottom:12px;\">Each reward component gets its own y-axis so lower-range signals stay readable.</div>"
        f"<div style=\"display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px;\">{''.join(cards)}</div>"
        f"</section>"
    )


def _resolve_default_report_output_path(trace_path: Path) -> Path:
    base_dir = trace_path if trace_path.is_dir() else trace_path.parent
    parts = base_dir.parts
    try:
        checkpoints_index = parts.index("checkpoints")
    except ValueError:
        return base_dir / "reward_trace_report.html"

    if len(parts) <= checkpoints_index + 3:
        return base_dir / "reward_trace_report.html"

    dataset = parts[checkpoints_index + 1]
    stage = parts[checkpoints_index + 2]
    run_name = parts[checkpoints_index + 3]
    return dataset_paths(dataset).outputs / "reports" / stage / run_name / "reward_trace_report.html"


def build_reward_trace_report(
    trace_path: str | Path,
    output_path: str | Path | None = None,
    csv_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    group_size: int | None = None,
) -> dict:
    trace_path = Path(trace_path)
    rows = _load_trace_rows(trace_path)
    resolved_group_size = _resolve_group_size(rows, group_size)
    frame = _build_step_frame(rows, resolved_group_size)

    if output_path is None:
        output_path = _resolve_default_report_output_path(trace_path)
    output_path = Path(output_path)

    if csv_path is None:
        csv_path = output_path.with_suffix(".csv")
    csv_path = Path(csv_path)

    if summary_path is None:
        summary_path = output_path.with_suffix(".summary.json")
    summary_path = Path(summary_path)

    ensure_dir(output_path.parent)
    frame.to_csv(csv_path, index=False)

    step_values = frame["synthetic_step"].astype(int).tolist()
    html_content = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>GRPO Reward Trace</title>"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;line-height:1.45;background:#f6f6f3;color:#222;}"
        "table{border-collapse:collapse;width:100%;background:#fff;}th,td{border:1px solid #ddd;padding:6px 8px;font-size:12px;}"
        "th{background:#f0f0f0;text-align:left;}code{background:#eceae4;padding:1px 4px;border-radius:3px;}"
        ".panel{background:#fff;border:1px solid #ddd;padding:16px 18px;margin-bottom:24px;box-shadow:0 1px 2px rgba(0,0,0,0.04);}</style>"
        "</head><body>"
        "<div class=\"panel\">"
        "<h1 style=\"margin-top:0;margin-bottom:10px;\">GRPO Reward Trace</h1>"
        f"<p style=\"margin:0;\">trace source: <code>{html.escape(str(trace_path))}</code><br>"
        f"synthetic group size: <code>{resolved_group_size}</code><br>"
        f"trace rows: <code>{len(rows)}</code>, synthetic steps: <code>{len(frame)}</code></p>"
        "</div>"
        "<div class=\"panel\">"
        + _render_svg_chart(
            "Per-Step Mean Reward Components",
            step_values,
            {field: frame[f"step_mean_{field}"].tolist() for field in TOP_LEVEL_FIELDS},
            y_axis_label="Mean Reward",
        )
        + _render_split_component_panel(
            "Per-Step Mean Reward Components (Split Panels)",
            step_values,
            {field: frame[f"step_mean_{field}"].tolist() for field in TOP_LEVEL_FIELDS},
            y_axis_label="Mean Reward",
        )
        + "</div><div class=\"panel\">"
        + _render_svg_chart(
            "Cumulative Mean Reward Components",
            step_values,
            {field: frame[f"cumulative_mean_{field}"].tolist() for field in TOP_LEVEL_FIELDS},
            y_axis_label="Cumulative Mean Reward",
        )
        + "</div><div class=\"panel\">"
        + _render_svg_chart(
            "Cumulative Total Reward Components",
            step_values,
            {field: frame[f"cumulative_sum_{field}"].tolist() for field in TOP_LEVEL_FIELDS},
            y_axis_label="Cumulative Total Reward",
        )
        + "</div><div class=\"panel\">"
        + _render_svg_chart(
            "Format Subscores",
            step_values,
            {field: frame[f"step_mean_{field}"].tolist() for field in FORMAT_FIELDS},
            y_axis_label="Mean Score",
        )
        + "</div><div class=\"panel\">"
        + frame.tail(20).to_html(index=False)
        + "</div>"
        + "</body></html>"
    )
    output_path.write_text(html_content, encoding="utf-8")

    summary = {
        "trace_path": str(trace_path),
        "output_path": str(output_path),
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
        "trace_row_count": len(rows),
        "synthetic_step_count": int(len(frame)),
        "group_size": resolved_group_size,
        "last_step": int(frame["synthetic_step"].iloc[-1]),
        "last_step_mean_total_reward": float(frame["step_mean_total_reward"].iloc[-1]),
        "last_cumulative_mean_total_reward": float(frame["cumulative_mean_total_reward"].iloc[-1]),
    }
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an HTML reward trace report from GNPR-SID GRPO reward logs.")
    parser.add_argument("--trace-path", required=True, help="Reward trace directory or a single JSONL trace file.")
    parser.add_argument("--output-path", default=None, help="HTML report output path.")
    parser.add_argument("--csv-path", default=None, help="CSV summary output path.")
    parser.add_argument("--summary-path", default=None, help="JSON summary output path.")
    parser.add_argument("--group-size", type=int, default=None, help="Synthetic step size override.")
    args = parser.parse_args()
    summary = build_reward_trace_report(
        trace_path=args.trace_path,
        output_path=args.output_path,
        csv_path=args.csv_path,
        summary_path=args.summary_path,
        group_size=args.group_size,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
