from __future__ import annotations

from statistics import mean
from typing import Iterable, Sequence

from gnprsid.prompts.render import extract_predictions


def accuracy_at_k(predictions: Sequence[str], target: str, k: int) -> bool:
    return target in list(predictions)[:k]


def evaluate_prediction_records(records: Iterable[dict]) -> tuple[dict, list[dict]]:
    evaluated = []
    correct_at_1 = 0
    correct_at_5 = 0
    correct_at_10 = 0
    prompt_lengths = []
    parsed_lengths = []
    exact_10_count = 0

    records = list(records)
    for record in records:
        repr_name = record.get("repr") or record.get("repr_name")
        parsed_predictions = record.get("parsed_predictions") or extract_predictions(record["prediction"], repr_name)
        top1 = accuracy_at_k(parsed_predictions, record["target"], 1)
        top5 = accuracy_at_k(parsed_predictions, record["target"], 5)
        top10 = accuracy_at_k(parsed_predictions, record["target"], 10)
        correct_at_1 += int(top1)
        correct_at_5 += int(top5)
        correct_at_10 += int(top10)
        prompt_lengths.append(int(record.get("prompt_char_length", len(record.get("prompt", "")))))
        parsed_lengths.append(len(parsed_predictions))
        if len(parsed_predictions) == 10:
            exact_10_count += 1

        enriched = dict(record)
        enriched["parsed_predictions"] = parsed_predictions
        enriched["top1_correct"] = top1
        enriched["top5_correct"] = top5
        enriched["top10_correct"] = top10
        evaluated.append(enriched)

    total = len(records)
    metrics = {
        "num_samples": total,
        "acc_at_1": correct_at_1 / total if total else 0.0,
        "acc_at_5": correct_at_5 / total if total else 0.0,
        "acc_at_10": correct_at_10 / total if total else 0.0,
        "correct_at_1": correct_at_1,
        "correct_at_5": correct_at_5,
        "correct_at_10": correct_at_10,
        "avg_prompt_char_length": mean(prompt_lengths) if prompt_lengths else 0.0,
        "avg_parsed_prediction_count": mean(parsed_lengths) if parsed_lengths else 0.0,
        "exact_10_prediction_rate": exact_10_count / total if total else 0.0,
    }
    return metrics, evaluated
