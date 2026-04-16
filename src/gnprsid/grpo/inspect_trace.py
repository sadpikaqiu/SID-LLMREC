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


def summarize_reward_traces(trace_path: str | Path, top_k: int = 10) -> dict[str, Any]:
    rows = _load_trace_rows(trace_path)
    total_reward = sum(float(row.get("total_reward", 0.0) or 0.0) for row in rows)
    total_parsed = sum(int(row.get("parsed_prediction_count", 0) or 0) for row in rows)
    total_single_line = sum(float(row.get("single_line_score", 0.0) or 0.0) for row in rows)
    zero_prediction_count = sum(1 for row in rows if int(row.get("parsed_prediction_count", 0) or 0) == 0)

    parsed_count_hist = Counter(int(row.get("parsed_prediction_count", 0) or 0) for row in rows)
    preview_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        preview_buckets[str(row.get("solution_preview", ""))].append(row)

    top_previews: list[dict[str, Any]] = []
    for preview, bucket in sorted(preview_buckets.items(), key=lambda item: len(item[1]), reverse=True)[:top_k]:
        bucket_rewards = [float(row.get("total_reward", 0.0) or 0.0) for row in bucket]
        bucket_parsed = [int(row.get("parsed_prediction_count", 0) or 0) for row in bucket]
        top_previews.append(
            {
                "solution_preview": preview,
                "count": len(bucket),
                "mean_total_reward": sum(bucket_rewards) / len(bucket_rewards),
                "mean_parsed_prediction_count": sum(bucket_parsed) / len(bucket_parsed),
                "single_line_rate": sum(float(row.get("single_line_score", 0.0) or 0.0) for row in bucket) / len(bucket),
            }
        )

    return {
        "trace_path": str(trace_path),
        "trace_row_count": len(rows),
        "mean_total_reward": total_reward / len(rows),
        "mean_parsed_prediction_count": total_parsed / len(rows),
        "single_line_rate": total_single_line / len(rows),
        "zero_prediction_rate": zero_prediction_count / len(rows),
        "parsed_prediction_count_histogram": dict(sorted(parsed_count_hist.items())),
        "top_solution_previews": top_previews,
    }
