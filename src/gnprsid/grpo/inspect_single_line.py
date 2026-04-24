from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_trace_rows(trace_path: str | Path) -> list[dict[str, Any]]:
    trace_path = Path(trace_path)
    trace_files = [trace_path] if trace_path.is_file() else sorted(trace_path.glob("*.jsonl"))
    rows: list[dict[str, Any]] = []
    for file_path in trace_files:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        raise FileNotFoundError(f"No reward trace rows found under {trace_path}")
    return rows


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(key, 0.0) or 0.0) for row in rows) / len(rows)


def inspect_single_line_failures(trace_path: str | Path, top_k: int = 10) -> dict[str, Any]:
    rows = _load_trace_rows(trace_path)
    single_line_rows = [row for row in rows if float(row.get("single_line_score", 0.0) or 0.0) > 0.5]
    multi_line_rows = [row for row in rows if float(row.get("single_line_score", 0.0) or 0.0) <= 0.5]

    preview_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in multi_line_rows:
        preview_buckets[str(row.get("solution_preview", ""))].append(row)

    top_multi_line_previews: list[dict[str, Any]] = []
    for preview, bucket in sorted(preview_buckets.items(), key=lambda item: len(item[1]), reverse=True)[:top_k]:
        top_multi_line_previews.append(
            {
                "solution_preview": preview,
                "count": len(bucket),
                "mean_total_reward": _mean(bucket, "total_reward"),
                "mean_parsed_prediction_count": _mean(bucket, "parsed_prediction_count"),
                "mean_valid_count_score": _mean(bucket, "valid_count_score"),
                "mean_exact_ten_score": _mean(bucket, "exact_ten_score"),
                "mean_hit": _mean(bucket, "hit"),
            }
        )

    parsed_count_histogram = Counter(int(row.get("parsed_prediction_count", 0) or 0) for row in multi_line_rows)
    return {
        "trace_path": str(trace_path),
        "trace_row_count": len(rows),
        "single_line_rate": len(single_line_rows) / len(rows),
        "multi_line_rate": len(multi_line_rows) / len(rows),
        "single_line_summary": {
            "row_count": len(single_line_rows),
            "mean_total_reward": _mean(single_line_rows, "total_reward"),
            "mean_parsed_prediction_count": _mean(single_line_rows, "parsed_prediction_count"),
            "mean_valid_count_score": _mean(single_line_rows, "valid_count_score"),
            "mean_exact_ten_score": _mean(single_line_rows, "exact_ten_score"),
            "mean_hit": _mean(single_line_rows, "hit"),
        },
        "multi_line_summary": {
            "row_count": len(multi_line_rows),
            "mean_total_reward": _mean(multi_line_rows, "total_reward"),
            "mean_parsed_prediction_count": _mean(multi_line_rows, "parsed_prediction_count"),
            "mean_valid_count_score": _mean(multi_line_rows, "valid_count_score"),
            "mean_exact_ten_score": _mean(multi_line_rows, "exact_ten_score"),
            "mean_hit": _mean(multi_line_rows, "hit"),
            "parsed_prediction_count_histogram": dict(sorted(parsed_count_histogram.items())),
        },
        "top_multi_line_previews": top_multi_line_previews,
    }
