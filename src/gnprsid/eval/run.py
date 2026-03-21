from __future__ import annotations

from pathlib import Path

from gnprsid.common.io import read_json, write_json
from gnprsid.common.logging import get_logger
from gnprsid.eval.metrics import evaluate_prediction_records


logger = get_logger(__name__)


def run_evaluation(predictions_path: str | Path, output_path: str | Path | None = None) -> dict:
    predictions_path = Path(predictions_path)
    payload = read_json(predictions_path)
    metadata = payload.get("metadata", {})
    records = payload.get("samples", payload)
    metrics, evaluated_records = evaluate_prediction_records(records)

    result = {
        "metadata": metadata,
        "metrics": metrics,
        "samples": evaluated_records,
    }
    if output_path is None:
        dataset = metadata.get("dataset", "unknown")
        repr_name = metadata.get("repr", "repr")
        history_source = metadata.get("history_source", "history")
        project_outputs = predictions_path.parents[1] / "eval"
        project_outputs.mkdir(parents=True, exist_ok=True)
        output_path = project_outputs / f"eval_{repr_name}_{history_source}.json"
    output_path = Path(output_path)
    write_json(output_path, result)
    logger.info("Saved evaluation metrics to %s", output_path)
    return result
