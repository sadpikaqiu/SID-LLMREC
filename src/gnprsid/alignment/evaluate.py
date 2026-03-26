from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, List

from gnprsid.common.io import read_json, write_json
from gnprsid.common.logging import get_logger
from gnprsid.common.paths import dataset_paths
from gnprsid.prompts.render import SID_PATTERN


logger = get_logger(__name__)
_SID_REGEX = re.compile(SID_PATTERN)


def format_alignment_prompt(record: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            f"### Instruction:\n{str(record['instruction']).strip()}",
            f"### Input:\n{str(record['input']).strip()}",
            "### Response:\n",
        ]
    )


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _extract_sid(text: str) -> str | None:
    match = _SID_REGEX.search(str(text))
    return match.group(0) if match else None


def _sid_segments(text: str) -> list[str]:
    return re.findall(r"<[a-zA-Z]_\d+>", text)


def _shared_prefix_length(pred_sid: str | None, target_sid: str) -> int:
    if not pred_sid:
        return 0
    pred_segments = _sid_segments(pred_sid)
    target_segments = _sid_segments(target_sid)
    shared = 0
    for pred_part, target_part in zip(pred_segments, target_segments):
        if pred_part != target_part:
            break
        shared += 1
    return shared


def _extract_attribute_field(text: str, label: str) -> str | None:
    match = re.search(rf"{re.escape(label)}:\s*([^;}}]+)", str(text))
    if not match:
        return None
    return match.group(1).strip()


def _generate_from_raw_prompts(
    model_cfg: dict[str, Any],
    tokenizer,
    model,
    prompts: list[str],
    batch_size: int,
) -> list[str]:
    import torch
    from tqdm import tqdm

    generation_cfg = dict(model_cfg.get("generation", {}))
    do_sample = bool(generation_cfg.get("do_sample", False))
    max_new_tokens = int(generation_cfg.get("max_new_tokens", 200))
    model_device = next(model.parameters()).device
    results: list[str] = []

    total_batches = (len(prompts) + batch_size - 1) // batch_size
    for start in tqdm(
        range(0, len(prompts), batch_size),
        total=total_batches,
        desc="Evaluating alignment",
    ):
        prompt_batch = prompts[start : start + batch_size]
        inputs = tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(model_cfg.get("max_length", 2048)),
        )
        prompt_padded_length = int(inputs["input_ids"].shape[1])
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = float(generation_cfg.get("temperature", 0.7))
            generate_kwargs["top_p"] = float(generation_cfg.get("top_p", 0.9))

        with torch.no_grad():
            outputs = model.generate(**inputs, **generate_kwargs)

        for output_ids in outputs:
            generated_ids = output_ids[prompt_padded_length:]
            results.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())

    return results


def _evaluate_attributes_to_sid(records: Iterable[dict[str, Any]], predictions: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    parsed = 0
    exact = 0
    prefix_1 = 0
    prefix_2 = 0
    prefix_3 = 0

    materialized_records = list(records)
    for record, prediction in zip(materialized_records, predictions):
        target_sid = str(record["output"]).strip()
        parsed_sid = _extract_sid(prediction)
        shared_prefix = _shared_prefix_length(parsed_sid, target_sid)
        parsed += int(parsed_sid is not None)
        exact += int(parsed_sid == target_sid)
        prefix_1 += int(shared_prefix >= 1)
        prefix_2 += int(shared_prefix >= 2)
        prefix_3 += int(shared_prefix >= 3)
        samples.append(
            {
                "instruction": record["instruction"],
                "input": record["input"],
                "target": target_sid,
                "prediction": prediction,
                "parsed_sid": parsed_sid,
                "shared_prefix_length": shared_prefix,
                "is_exact_match": parsed_sid == target_sid,
            }
        )

    total = len(materialized_records)
    metrics = {
        "num_samples": total,
        "parsed_sid_rate": parsed / total if total else 0.0,
        "exact_match_rate": exact / total if total else 0.0,
        "prefix_match_at_1": prefix_1 / total if total else 0.0,
        "prefix_match_at_2": prefix_2 / total if total else 0.0,
        "prefix_match_at_3": prefix_3 / total if total else 0.0,
    }
    return metrics, samples


def _evaluate_sid_to_attributes(records: Iterable[dict[str, Any]], predictions: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    exact = 0
    category = 0
    region = 0

    materialized_records = list(records)
    for record, prediction in zip(materialized_records, predictions):
        target_text = _normalize_text(str(record["output"]))
        pred_text = _normalize_text(prediction)
        target_category = _extract_attribute_field(target_text, "Category")
        target_region = _extract_attribute_field(target_text, "Region")
        exact += int(pred_text == target_text)
        category += int(target_category is not None and target_category in pred_text)
        region += int(target_region is not None and f"Region: {target_region}" in pred_text)
        samples.append(
            {
                "instruction": record["instruction"],
                "input": record["input"],
                "target": record["output"],
                "prediction": prediction,
                "is_exact_match": pred_text == target_text,
                "category_match": target_category is not None and target_category in pred_text,
                "region_match": target_region is not None and f"Region: {target_region}" in pred_text,
            }
        )

    total = len(materialized_records)
    metrics = {
        "num_samples": total,
        "exact_match_rate": exact / total if total else 0.0,
        "category_match_rate": category / total if total else 0.0,
        "region_match_rate": region / total if total else 0.0,
    }
    return metrics, samples


def evaluate_alignment(
    dataset: str,
    model_config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    split: str = "valid",
    task: str = "attributes_to_sid",
    data_path: str | Path | None = None,
    batch_size: int = 1,
    limit: int | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    from gnprsid.inference.modeling import load_generation_model

    paths = dataset_paths(dataset)
    if data_path is None:
        split_name = "valid" if split == "valid" else "train"
        data_path = paths.artifacts / "alignment" / f"{split_name}_align.json"
    else:
        data_path = Path(data_path)

    if output_path is None:
        output_path = paths.outputs / "eval" / f"alignment_{task}_{split}.json"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = read_json(data_path)
    if task == "attributes_to_sid":
        records = [record for record in records if _extract_sid(str(record["output"])) is not None]
    elif task == "sid_to_attributes":
        records = [record for record in records if _extract_sid(str(record["output"])) is None]
    else:
        raise ValueError(f"Unsupported alignment evaluation task: {task}")

    if limit is not None:
        records = records[:limit]

    prompts = [format_alignment_prompt(record) for record in records]
    model_cfg, tokenizer, model, model_source = load_generation_model(model_config_path, checkpoint_path=checkpoint_path)
    predictions = _generate_from_raw_prompts(model_cfg, tokenizer, model, prompts, batch_size=batch_size)

    if task == "attributes_to_sid":
        metrics, samples = _evaluate_attributes_to_sid(records, predictions)
    else:
        metrics, samples = _evaluate_sid_to_attributes(records, predictions)

    payload = {
        "metadata": {
            "dataset": dataset,
            "task": task,
            "split": split,
            "data_path": str(data_path),
            "model_config_path": str(model_config_path),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "model_source": model_source,
            "num_samples": len(samples),
        },
        "metrics": metrics,
        "samples": samples,
    }
    write_json(output_path, payload)
    logger.info("Saved alignment evaluation to %s", output_path)
    return payload
