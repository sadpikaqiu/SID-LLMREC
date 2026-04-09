from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from gnprsid.common.io import ensure_dir, write_json
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


def _series_to_polyline(values: list[float], width: int, height: int, padding: int) -> str:
    if len(values) == 1:
        x = padding + (width - 2 * padding) / 2
        y = height / 2
        return f"{x:.2f},{y:.2f}"

    min_value = min(values)
    max_value = max(values)
    span = max(max_value - min_value, 1e-9)
    points = []
    usable_width = width - 2 * padding
    usable_height = height - 2 * padding
    for index, value in enumerate(values):
        x = padding + usable_width * (index / (len(values) - 1))
        y = height - padding - usable_height * ((value - min_value) / span)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _render_svg_chart(title: str, x_values: list[int], series_map: dict[str, list[float]]) -> str:
    width = 980
    height = 320
    padding = 36
    colors = [
        "#14532d",
        "#1d4ed8",
        "#b45309",
        "#be123c",
        "#6d28d9",
        "#0f766e",
    ]

    legend_items = []
    polylines = []
    for index, (label, values) in enumerate(series_map.items()):
        color = colors[index % len(colors)]
        polyline = _series_to_polyline(values, width, height, padding)
        polylines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{polyline}" />'
        )
        legend_items.append(
            f'<span style="display:inline-block;margin-right:14px;">'
            f'<span style="display:inline-block;width:10px;height:10px;background:{color};margin-right:6px;"></span>'
            f"{html.escape(label)}</span>"
        )

    x_start = x_values[0]
    x_end = x_values[-1]
    return (
        f"<section style=\"margin-bottom:28px;\">"
        f"<h2 style=\"font-size:18px;margin:0 0 8px;\">{html.escape(title)}</h2>"
        f"<div style=\"font-size:13px;color:#444;margin-bottom:10px;\">steps {x_start} to {x_end}</div>"
        f"<svg viewBox=\"0 0 {width} {height}\" style=\"width:100%;border:1px solid #ddd;background:#fff;\">"
        f"<line x1=\"{padding}\" y1=\"{height - padding}\" x2=\"{width - padding}\" y2=\"{height - padding}\" stroke=\"#666\" />"
        f"<line x1=\"{padding}\" y1=\"{padding}\" x2=\"{padding}\" y2=\"{height - padding}\" stroke=\"#666\" />"
        f"{''.join(polylines)}</svg>"
        f"<div style=\"font-size:13px;margin-top:8px;\">{''.join(legend_items)}</div>"
        f"</section>"
    )


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
        base_dir = trace_path if trace_path.is_dir() else trace_path.parent
        output_path = base_dir / "reward_trace_report.html"
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
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;line-height:1.4;background:#f7f7f7;}"
        "table{border-collapse:collapse;width:100%;background:#fff;}th,td{border:1px solid #ddd;padding:6px 8px;font-size:12px;}"
        "th{background:#f0f0f0;text-align:left;}code{background:#eee;padding:1px 4px;border-radius:3px;}</style>"
        "</head><body>"
        "<h1 style=\"margin-top:0;\">GRPO Reward Trace</h1>"
        f"<p>trace source: <code>{html.escape(str(trace_path))}</code><br>"
        f"synthetic group size: <code>{resolved_group_size}</code><br>"
        f"trace rows: <code>{len(rows)}</code>, synthetic steps: <code>{len(frame)}</code></p>"
        + _render_svg_chart(
            "Per-Step Mean Reward Components",
            step_values,
            {field: frame[f"step_mean_{field}"].tolist() for field in TOP_LEVEL_FIELDS},
        )
        + _render_svg_chart(
            "Cumulative Mean Reward Components",
            step_values,
            {field: frame[f"cumulative_mean_{field}"].tolist() for field in TOP_LEVEL_FIELDS},
        )
        + _render_svg_chart(
            "Format Subscores",
            step_values,
            {field: frame[f"step_mean_{field}"].tolist() for field in FORMAT_FIELDS},
        )
        + frame.tail(20).to_html(index=False)
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
