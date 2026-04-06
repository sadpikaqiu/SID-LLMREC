from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from gnprsid.common.io import ensure_dir, iter_jsonl, write_json
from gnprsid.common.logging import get_logger
from gnprsid.common.paths import dataset_paths
from gnprsid.common.profiles import resolve_project_path
from gnprsid.prompts.render import (
    PROMPT_TEMPLATE_VERSION,
    V2_NEXT_POI_INSTRUCTION,
    build_prompt_input_text,
    system_prompt,
)


logger = get_logger(__name__)

GRPO_DATA_SOURCE = "gnprsid_nyc_sid_current"
GRPO_ABILITY = "next_poi_current"


def _build_grpo_current_prompt(row: dict) -> str:
    return "\n\n".join(
        [
            "### Instruction:\n" + V2_NEXT_POI_INSTRUCTION,
            "### Input:\n" + build_prompt_input_text(row, "current"),
            "\n".join(
                [
                    "Output format:",
                    "1. Return exactly 1 complete semantic ID in one line.",
                    "2. The reply must contain only one complete semantic ID.",
                    "3. Start the reply immediately with the semantic ID.",
                    "4. Do not output explanations, numbering, commas, or any extra text.",
                ]
            ),
        ]
    )


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared sample file: {path}")
    return list(iter_jsonl(path))


def _to_verl_rows(rows: Iterable[dict]) -> list[dict]:
    payload: list[dict] = []
    sys_prompt = system_prompt("sid", "current", candidate_count=1)
    for row in rows:
        user_prompt = _build_grpo_current_prompt(row)
        payload.append(
            {
                "data_source": GRPO_DATA_SOURCE,
                "prompt": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "ability": GRPO_ABILITY,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(row["target"]),
                },
                "extra_info": {
                    "sample_id": str(row["sample_id"]),
                    "uid": int(row["uid"]),
                    "repr": "sid",
                    "history_source": "current",
                    "target": str(row["target"]),
                    "target_time": str(row["target_time"]),
                    "prompt_template_version": f"{PROMPT_TEMPLATE_VERSION}_grpo_constrained",
                },
            }
        )
    return payload


def build_grpo_data(
    dataset: str = "NYC",
    output_dir: str | Path | None = None,
) -> dict:
    paths = dataset_paths(dataset)
    source_dir = paths.processed
    output_dir = resolve_project_path(output_dir) if output_dir else (paths.artifacts / "grpo" / "sid" / "current")
    output_dir = ensure_dir(output_dir)

    train_rows = _load_rows(source_dir / "samples_sid_train.jsonl")
    valid_rows = _load_rows(source_dir / "samples_sid_val.jsonl")

    train_payload = _to_verl_rows(train_rows)
    valid_payload = _to_verl_rows(valid_rows)

    train_path = output_dir / "train.parquet"
    valid_path = output_dir / "valid.parquet"
    pd.DataFrame(train_payload).to_parquet(train_path, index=False)
    pd.DataFrame(valid_payload).to_parquet(valid_path, index=False)

    manifest = {
        "dataset": dataset,
        "repr": "sid",
        "history_source": "current",
        "data_source": GRPO_DATA_SOURCE,
        "ability": GRPO_ABILITY,
        "prompt_template_version": f"{PROMPT_TEMPLATE_VERSION}_grpo_constrained",
        "deployment_contract": "candidate_constrained_single_sid",
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "num_train": len(train_payload),
        "num_valid": len(valid_payload),
    }
    write_json(output_dir / "manifest.json", manifest)
    logger.info("Built GRPO parquet data for %s at %s", dataset, output_dir)
    return manifest
