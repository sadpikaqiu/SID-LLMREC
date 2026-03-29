from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from gnprsid.alignment.semantic import (
    parse_profile_json,
    profile_to_json,
    validate_profile,
)
from gnprsid.common.io import iter_jsonl, write_json
from gnprsid.common.logging import get_logger
from gnprsid.common.paths import dataset_paths


logger = get_logger(__name__)
_PREFIX_PATTERNS = {
    "a": re.compile(r"<[a-zA-Z]_\d+>"),
    "ab": re.compile(r"(?:<[a-zA-Z]_\d+>){2}"),
    "abc": re.compile(r"(?:<[a-zA-Z]_\d+>){3}"),
}
_TASK_TO_TYPE = {
    "sid_to_abc_profile": ("phase_b", "full_sid_to_abc_profile"),
    "abc_profile_to_a": ("phase_a", "category_profile_to_a"),
    "abc_profile_to_ab": ("phase_a", "category_region_profile_to_ab"),
    "abc_profile_to_abc": ("phase_b", "abc_profile_to_abc"),
}
_TASK_TO_LEVEL = {
    "sid_to_abc_profile": "abc",
    "abc_profile_to_a": "a",
    "abc_profile_to_ab": "ab",
    "abc_profile_to_abc": "abc",
}


def format_alignment_prompt(record: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            f"### Instruction:\n{str(record['instruction']).strip()}",
            f"### Input:\n{str(record['input']).strip()}",
            "### Response:\n",
        ]
    )


def _default_data_path(dataset: str, split: str, task: str) -> Path:
    paths = dataset_paths(dataset)
    phase_name, _ = _TASK_TO_TYPE[task]
    suffix = "train" if split == "train" else "valid"
    return paths.artifacts / "alignment" / f"{suffix}_align_{phase_name}.jsonl"


def _load_task_records(data_path: Path, task: str, limit: int | None) -> list[dict[str, Any]]:
    _, task_type = _TASK_TO_TYPE[task]
    records = [record for record in iter_jsonl(data_path) if record["task_type"] == task_type]
    if limit is not None:
        records = records[:limit]
    return records


def _extract_prefix(text: str, level: str) -> str | None:
    match = _PREFIX_PATTERNS[level].search(str(text))
    return match.group(0) if match else None


def _evaluate_sid_to_abc_profile(
    records: Iterable[dict[str, Any]],
    predictions: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    materialized_records = list(records)
    samples: list[dict[str, Any]] = []
    valid_profile = 0
    category_match = 0
    region_match = 0
    geo_bucket_match = 0
    joint_match = 0
    predicted_categories: Counter[str] = Counter()
    predicted_regions: Counter[str] = Counter()
    predicted_profiles: Counter[str] = Counter()

    for record, prediction in zip(materialized_records, predictions):
        target_profile = parse_profile_json(str(record["output"]))
        predicted_profile = parse_profile_json(prediction)
        is_valid = validate_profile(predicted_profile, "abc")
        valid_profile += int(is_valid)
        if is_valid and predicted_profile is not None:
            predicted_category = str(predicted_profile["category"])
            predicted_region = str(predicted_profile["region"])
            predicted_profile_json = profile_to_json(predicted_profile, level="abc")
            predicted_categories[predicted_category] += 1
            predicted_regions[predicted_region] += 1
            predicted_profiles[predicted_profile_json] += 1
        else:
            predicted_profile_json = None

        category_ok = bool(
            is_valid
            and target_profile is not None
            and str(predicted_profile["category"]) == str(target_profile["category"])
        )
        region_ok = bool(
            is_valid
            and target_profile is not None
            and str(predicted_profile["region"]) == str(target_profile["region"])
        )
        geo_ok = bool(
            is_valid
            and target_profile is not None
            and str(predicted_profile["geo_bucket"]) == str(target_profile["geo_bucket"])
        )
        joint_ok = bool(category_ok and region_ok and geo_ok)
        category_match += int(category_ok)
        region_match += int(region_ok)
        geo_bucket_match += int(geo_ok)
        joint_match += int(joint_ok)

        samples.append(
            {
                "sample_id": record["sample_id"],
                "task_type": record["task_type"],
                "source_sid": record["source_sid"],
                "source_abc": record["source_abc"],
                "target": record["output"],
                "prediction": prediction,
                "parsed_profile": predicted_profile,
                "parsed_profile_json": predicted_profile_json,
                "valid_profile": is_valid,
                "category_match": category_ok,
                "region_match": region_ok,
                "geo_bucket_match": geo_ok,
                "joint_profile_match": joint_ok,
            }
        )

    total = len(materialized_records)
    metrics = {
        "num_samples": total,
        "valid_profile_rate": valid_profile / total if total else 0.0,
        "category_match_rate": category_match / total if total else 0.0,
        "region_match_rate": region_match / total if total else 0.0,
        "geo_bucket_match_rate": geo_bucket_match / total if total else 0.0,
        "joint_profile_match_rate": joint_match / total if total else 0.0,
        "dominant_category_share": (
            max(predicted_categories.values()) / total if total and predicted_categories else 0.0
        ),
        "dominant_region_share": (
            max(predicted_regions.values()) / total if total and predicted_regions else 0.0
        ),
        "unique_predicted_profiles": len(predicted_profiles),
    }
    return metrics, samples


def _evaluate_profile_to_prefix(
    task: str,
    records: Iterable[dict[str, Any]],
    predictions: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    level = _TASK_TO_LEVEL[task]
    materialized_records = list(records)
    samples: list[dict[str, Any]] = []
    valid_prefix = 0
    exact = 0

    for record, prediction in zip(materialized_records, predictions):
        parsed_prefix = _extract_prefix(prediction, level)
        candidates = list(record.get("candidate_prefixes", []))
        valid = bool(parsed_prefix is not None and parsed_prefix in candidates)
        is_exact = bool(parsed_prefix == str(record["output"]).strip())
        valid_prefix += int(valid)
        exact += int(is_exact)
        samples.append(
            {
                "sample_id": record["sample_id"],
                "task_type": record["task_type"],
                "source_sid": record["source_sid"],
                "source_abc": record["source_abc"],
                "target": record["output"],
                "candidate_prefixes": candidates,
                "prediction": prediction,
                "parsed_prefix": parsed_prefix,
                "valid_prefix": valid,
                "is_exact_match": is_exact,
            }
        )

    total = len(materialized_records)
    metrics = {
        "num_samples": total,
        "valid_prefix_rate": valid_prefix / total if total else 0.0,
        "exact_match_rate": exact / total if total else 0.0,
    }
    return metrics, samples


def evaluate_alignment(
    dataset: str,
    model_config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    split: str = "valid",
    task: str = "sid_to_abc_profile",
    data_path: str | Path | None = None,
    batch_size: int = 1,
    limit: int | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    from gnprsid.inference.modeling import generate_from_raw_prompts, load_generation_model

    if task not in _TASK_TO_TYPE:
        raise ValueError(f"Unsupported alignment evaluation task: {task}")

    paths = dataset_paths(dataset)
    data_path = Path(data_path) if data_path else _default_data_path(dataset, split, task)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing alignment evaluation data: {data_path}")

    if output_path is None:
        output_path = paths.outputs / "eval" / f"alignment_{task}_{split}.json"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = _load_task_records(data_path, task, limit=limit)
    prompts = [format_alignment_prompt(record) for record in records]
    model_cfg, tokenizer, model, model_source = load_generation_model(model_config_path, checkpoint_path=checkpoint_path)

    if task == "sid_to_abc_profile":
        candidate_space = sorted({str(record["output"]).strip() for record in records})
        predictions = generate_from_raw_prompts(
            model_cfg,
            tokenizer,
            model,
            prompts,
            batch_size=batch_size,
            allowed_completions=candidate_space,
            top_k_sequences=1,
        )
        decoding_mode = "candidate_constrained_global"
    else:
        predictions = []
        for prompt, record in zip(prompts, records):
            prediction = generate_from_raw_prompts(
                model_cfg,
                tokenizer,
                model,
                [prompt],
                batch_size=1,
                allowed_completions=list(record["candidate_prefixes"]),
                top_k_sequences=1,
            )[0]
            predictions.append(prediction)
        decoding_mode = "candidate_constrained_per_sample"

    if task == "sid_to_abc_profile":
        metrics, samples = _evaluate_sid_to_abc_profile(records, predictions)
    else:
        metrics, samples = _evaluate_profile_to_prefix(task, records, predictions)

    payload = {
        "metadata": {
            "dataset": dataset,
            "task": task,
            "split": split,
            "data_path": str(data_path),
            "model_config_path": str(model_config_path),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "model_source": model_source,
            "decoding_mode": decoding_mode,
            "num_samples": len(samples),
        },
        "metrics": metrics,
        "samples": samples,
    }
    write_json(output_path, payload)
    logger.info("Saved alignment evaluation to %s", output_path)
    return payload
