from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from gnprsid.common.io import iter_jsonl, read_json, write_json
from gnprsid.common.logging import get_logger
from gnprsid.common.paths import dataset_paths
from gnprsid.prompts.render import (
    PROMPT_TEMPLATE_VERSION,
    build_prompt,
    extract_predictions,
    system_prompt,
)
from gnprsid.inference.modeling import generate_from_messages, load_generation_model


logger = get_logger(__name__)


def _load_history_map(path: Path) -> Dict[int, str]:
    payload = read_json(path)
    history_map: Dict[int, str] = {}
    for item in payload:
        import re

        match = re.search(r"User_(\d+)", item["input"])
        if match:
            history_map[int(match.group(1))] = item["input"]
    return history_map


def _load_split_rows(bank_path: Path, split: str, repr_name: str) -> List[dict]:
    return [
        row
        for row in iter_jsonl(bank_path)
        if row["split"] == split and row["repr"] == repr_name
    ]


def _load_bank_map(bank_path: Path, repr_name: str) -> Dict[str, dict]:
    return {
        row["sample_id"]: row
        for row in iter_jsonl(bank_path)
        if row["repr"] == repr_name
    }


def run_batch_inference(
    dataset: str,
    repr_name: str,
    history_source: str,
    model_config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    split: str = "test",
    retrieval_bank_path: str | Path | None = None,
    similar_map_path: str | Path | None = None,
    history_path: str | Path | None = None,
    top_k_retrieval: int | None = None,
    batch_size: int = 1,
    limit: int | None = None,
    output_path: str | Path | None = None,
) -> dict:
    paths = dataset_paths(dataset)
    retrieval_bank_path = Path(retrieval_bank_path) if retrieval_bank_path else (
        paths.artifacts / "retrieval" / f"retrieval_bank_{repr_name}.jsonl"
    )
    output_path = Path(output_path) if output_path else (
        paths.outputs / "predictions" / f"{repr_name}_{history_source}_{split}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = _load_split_rows(retrieval_bank_path, split, repr_name)
    if limit is not None:
        samples = samples[:limit]

    history_map: Optional[Dict[int, str]] = None
    similar_map: Optional[Dict[str, List[Dict[str, object]]]] = None
    bank_map: Optional[Dict[str, dict]] = None
    if history_source in {"original", "hybrid"}:
        history_path = Path(history_path) if history_path else (paths.processed / f"history_{repr_name}.json")
        history_map = _load_history_map(history_path)
    if history_source in {"retrieval", "hybrid"}:
        similar_map_path = Path(similar_map_path) if similar_map_path else (
            paths.artifacts / "retrieval" / f"similar_map_{split}_{repr_name}.json"
        )
        similar_map = read_json(similar_map_path)
        bank_map = _load_bank_map(retrieval_bank_path, repr_name)

    if top_k_retrieval is None:
        top_k_retrieval = 3 if history_source == "hybrid" else 5

    model_cfg, tokenizer, model, model_source = load_generation_model(model_config_path, checkpoint_path=checkpoint_path)
    sys_prompt = system_prompt(repr_name, history_source)
    prompts = []
    for sample in samples:
        user_prompt = build_prompt(
            sample,
            history_source,
            history_map=history_map,
            similar_map=similar_map,
            bank_map=bank_map,
            top_k_retrieval=top_k_retrieval,
            candidate_count=10,
        )
        prompts.append(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

    predictions = generate_from_messages(model_cfg, tokenizer, model, prompts, batch_size=batch_size)
    records = []
    for sample, message_pair, prediction in zip(samples, prompts, predictions):
        user_prompt = message_pair[1]["content"]
        records.append(
            {
                "sample_id": sample["sample_id"],
                "uid": sample["uid"],
                "split": sample["split"],
                "repr": repr_name,
                "history_source": history_source,
                "target": sample["target"],
                "target_time": sample["target_time"],
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "system_prompt": sys_prompt,
                "prompt": user_prompt,
                "prompt_char_length": len(user_prompt),
                "prediction": prediction,
                "parsed_predictions": extract_predictions(prediction, repr_name),
            }
        )

    payload = {
        "metadata": {
            "dataset": dataset,
            "repr": repr_name,
            "history_source": history_source,
            "split": split,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "model_config_path": str(model_config_path),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "model_source": model_source,
            "top_k_retrieval": top_k_retrieval,
            "num_samples": len(records),
        },
        "samples": records,
    }
    write_json(output_path, payload)
    logger.info("Saved %s predictions to %s", len(records), output_path)
    return payload
