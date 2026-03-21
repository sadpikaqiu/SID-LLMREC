from __future__ import annotations

from pathlib import Path

from gnprsid.common.io import read_json
from gnprsid.common.logging import get_logger
from gnprsid.common.paths import dataset_paths


logger = get_logger(__name__)


def summarize_evaluations(
    dataset: str,
    eval_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict:
    paths = dataset_paths(dataset)
    eval_dir = Path(eval_dir) if eval_dir else (paths.outputs / "eval")
    output_path = Path(output_path) if output_path else (paths.outputs / "reports" / "summary.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(eval_dir.glob("*.json")):
        payload = read_json(path)
        metadata = payload.get("metadata", {})
        metrics = payload.get("metrics", {})
        rows.append(
            {
                "run_name": path.stem,
                "repr": metadata.get("repr", "-"),
                "history_source": metadata.get("history_source", "-"),
                "samples": metrics.get("num_samples", 0),
                "acc1": metrics.get("acc_at_1", 0.0),
                "acc5": metrics.get("acc_at_5", 0.0),
                "acc10": metrics.get("acc_at_10", 0.0),
                "prompt_len": metrics.get("avg_prompt_char_length", 0.0),
                "exact10": metrics.get("exact_10_prediction_rate", 0.0),
            }
        )

    lines = [
        "# Evaluation Summary",
        "",
        "| run | repr | history | samples | ACC@1 | ACC@5 | ACC@10 | avg prompt chars | exact10 rate |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_name']} | {row['repr']} | {row['history_source']} | {row['samples']} | "
            f"{row['acc1']:.4f} | {row['acc5']:.4f} | {row['acc10']:.4f} | "
            f"{row['prompt_len']:.1f} | {row['exact10']:.4f} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "dataset": dataset,
        "eval_dir": str(eval_dir),
        "output_path": str(output_path),
        "num_runs": len(rows),
    }
    logger.info("Wrote evaluation summary to %s", output_path)
    return manifest
