from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from gnprsid.common.config import load_yaml
from gnprsid.common.paths import dataset_paths
from gnprsid.common.profiles import resolve_model_profile_path, resolve_project_path
from gnprsid.grpo.reward_current_top10 import compute_score
from gnprsid.inference.modeling import (
    _resolve_chat_template_kwargs,
    generate_from_messages,
    load_generation_model,
    render_chat_prompts,
)
from gnprsid.prompts.render import extract_predictions


def _load_grpo_rows(grpo_path: Path) -> list[dict[str, Any]]:
    if not grpo_path.exists():
        raise FileNotFoundError(f"Missing GRPO parquet: {grpo_path}")
    return pd.read_parquet(grpo_path).to_dict(orient="records")


def inspect_grpo_sample(
    train_config_path: str | Path,
    split: str = "valid",
    sample_id: str | None = None,
    row_index: int = 0,
    grpo_data_path: str | Path | None = None,
) -> dict[str, Any]:
    train_config_path = resolve_project_path(train_config_path)
    cfg = load_yaml(train_config_path)
    dataset = str(cfg.get("dataset", "NYC"))
    paths = dataset_paths(dataset)

    if grpo_data_path is None:
        default_name = "valid.parquet" if split == "valid" else "train.parquet"
        grpo_path = resolve_project_path(cfg["valid_path"] if split == "valid" else cfg["train_path"])
        if grpo_path.name != default_name and not grpo_path.exists():
            grpo_path = paths.artifacts / "grpo" / "sid" / "current" / default_name
    else:
        grpo_path = resolve_project_path(grpo_data_path)

    rows = _load_grpo_rows(grpo_path)
    row: dict[str, Any] | None = None
    if sample_id is not None:
        for candidate in rows:
            extra_info = candidate.get("extra_info", {})
            if str(extra_info.get("sample_id")) == str(sample_id):
                row = candidate
                break
        if row is None:
            raise KeyError(f"Could not find sample_id={sample_id} in {grpo_path}")
    else:
        if row_index < 0 or row_index >= len(rows):
            raise IndexError(f"row_index={row_index} out of range for {grpo_path} with {len(rows)} rows")
        row = rows[row_index]

    model_profile_path = resolve_model_profile_path(cfg["model_profile"])
    checkpoint_path = resolve_project_path(cfg["init_model_path"])
    model_cfg, tokenizer, model, model_source = load_generation_model(
        model_profile_path,
        checkpoint_path=checkpoint_path,
    )

    messages = list(row["prompt"])
    rendered_prompt = render_chat_prompts(
        tokenizer,
        [messages],
        chat_template_kwargs=_resolve_chat_template_kwargs(model_cfg),
    )[0]
    prediction = generate_from_messages(
        model_cfg,
        tokenizer,
        model,
        [messages],
        batch_size=1,
        allowed_completions=None,
        top_k_sequences=10,
    )[0]

    parsed_predictions = extract_predictions(prediction, "sid")
    reward = compute_score(
        str(row.get("data_source", "")),
        prediction,
        str(row["reward_model"]["ground_truth"]),
        extra_info=row.get("extra_info"),
    )

    extra_info = dict(row.get("extra_info", {}))
    return {
        "train_config_path": str(train_config_path),
        "grpo_data_path": str(grpo_path),
        "split": split,
        "sample_id": str(extra_info.get("sample_id")),
        "row_index": row_index if sample_id is None else None,
        "model_profile_path": str(model_profile_path),
        "checkpoint_path": str(checkpoint_path),
        "model_source": model_source,
        "target": str(row["reward_model"]["ground_truth"]),
        "prompt_roles": [message["role"] for message in messages],
        "system_prompt": str(messages[0]["content"]) if messages else "",
        "user_prompt": str(messages[1]["content"]) if len(messages) > 1 else "",
        "rendered_prompt": rendered_prompt,
        "prediction": prediction,
        "parsed_predictions": parsed_predictions,
        "parsed_prediction_count": len(parsed_predictions),
        "reward": reward,
    }
