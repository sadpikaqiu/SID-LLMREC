from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from gnprsid.common.io import iter_jsonl, write_json
from gnprsid.common.logging import get_logger
from gnprsid.common.paths import dataset_paths
from gnprsid.retrieval.encoder import encode_text, load_encoder


logger = get_logger(__name__)


def parse_target_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Encountered zero-norm embedding while normalizing retrieval matrix.")
    return matrix / norms


def build_candidate_mask(
    train_ids: np.ndarray,
    train_times: np.ndarray,
    current_id: str,
    current_time: float,
    split: str,
) -> np.ndarray:
    mask = train_times < current_time
    if split == "train":
        mask = mask & (train_ids != current_id)
    return mask


def _load_bank_rows(bank_path: Path, repr_name: str) -> list[dict]:
    return [row for row in iter_jsonl(bank_path) if row["repr"] == repr_name]


def build_similarity_map(
    dataset: str,
    repr_name: str,
    split: str = "test",
    retrieval_config_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    bank_path: str | Path | None = None,
    output_path: str | Path | None = None,
    model_name_or_path: str | None = None,
) -> dict:
    from gnprsid.common.config import load_yaml
    from gnprsid.common.profiles import load_model_profile, resolve_project_path

    paths = dataset_paths(dataset)
    retrieval_cfg = load_yaml(resolve_project_path(retrieval_config_path)) if retrieval_config_path else {}
    model_cfg = load_model_profile(model_config_path or retrieval_cfg.get("model_profile", "qwen2.5-7b-instruct"))

    bank_path = Path(bank_path) if bank_path else (paths.artifacts / "retrieval" / f"retrieval_bank_{repr_name}.jsonl")
    output_path = Path(output_path) if output_path else (paths.artifacts / "retrieval" / f"similar_map_{split}_{repr_name}.json")
    rows = _load_bank_rows(bank_path, repr_name)
    rows_by_split: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    for row in rows:
        rows_by_split.setdefault(row["split"], []).append(row)

    if not rows_by_split.get("train"):
        raise ValueError(f"Retrieval bank {bank_path} does not contain train rows for repr={repr_name}")
    if not rows_by_split.get(split):
        raise ValueError(f"Retrieval bank {bank_path} does not contain split={split} rows for repr={repr_name}")

    tokenizer, model = load_encoder(
        model_name_or_path or model_cfg["base_model"],
        dtype_name=str(retrieval_cfg.get("dtype", model_cfg.get("dtype", "auto"))),
        pooling=str(retrieval_cfg.get("pooling", "mean")),
        device_map=str(retrieval_cfg.get("device_map", "auto")),
        load_in_4bit=bool(retrieval_cfg.get("load_in_4bit", False)),
    )
    max_length = int(retrieval_cfg.get("max_length", model_cfg.get("max_length", 2048)))
    top_k = int(retrieval_cfg.get("top_k", 35))
    pooling = str(retrieval_cfg.get("pooling", "mean"))

    key_embeddings: Dict[str, np.ndarray] = {}
    query_embeddings: Dict[str, np.ndarray] = {}
    split_rows = rows_by_split[split]
    train_rows = rows_by_split["train"]

    for row in split_rows:
        key_embeddings[row["sample_id"]] = encode_text(model, tokenizer, row["key_text"], max_length=max_length, pooling=pooling)
    for row in train_rows:
        query_embeddings[row["sample_id"]] = encode_text(model, tokenizer, row["query_text"], max_length=max_length, pooling=pooling)

    train_ids = np.array([row["sample_id"] for row in train_rows])
    train_times = np.array([parse_target_time(row["target_time"]).timestamp() for row in train_rows])
    train_matrix = normalize_matrix(np.vstack([query_embeddings[sample_id] for sample_id in train_ids]))

    split_map: Dict[str, List[Dict[str, object]]] = {}
    score_values = []
    for row in split_rows:
        current_id = str(row["sample_id"])
        current_time = parse_target_time(row["target_time"]).timestamp()
        valid_mask = build_candidate_mask(train_ids, train_times, current_id, current_time, split)
        if not valid_mask.any():
            split_map[current_id] = []
            continue

        current_embedding = key_embeddings[current_id]
        current_norm = np.linalg.norm(current_embedding)
        if current_norm == 0:
            raise ValueError(f"Zero-norm query embedding for sample {current_id}")
        current_embedding = current_embedding / current_norm
        scores = np.dot(train_matrix[valid_mask], current_embedding)
        if not np.isfinite(scores).all():
            raise ValueError(f"Non-finite retrieval scores for sample {current_id}")
        candidate_ids = train_ids[valid_mask]
        actual_top_k = min(top_k, len(candidate_ids))
        top_indices = np.argsort(scores)[-actual_top_k:][::-1]
        neighbors = [
            {
                "sample_id": str(candidate_ids[index]),
                "score": float(scores[index]),
            }
            for index in top_indices
        ]
        split_map[current_id] = neighbors
        score_values.extend(item["score"] for item in neighbors)

    if score_values and not any(abs(value) > 0 for value in score_values):
        raise ValueError("All retrieval scores are zero. This indicates an invalid similarity run.")

    write_json(output_path, split_map)
    manifest = {
        "dataset": dataset,
        "repr": repr_name,
        "split": split,
        "bank_path": str(bank_path),
        "output_path": str(output_path),
        "top_k": top_k,
        "max_length": max_length,
        "pooling": pooling,
        "dtype": str(retrieval_cfg.get("dtype", model_cfg.get("dtype", "auto"))),
        "num_queries": len(split_map),
    }
    write_json(output_path.with_suffix(".manifest.json"), manifest)
    logger.info("Built similarity map for %s/%s/%s", dataset, repr_name, split)
    return manifest
