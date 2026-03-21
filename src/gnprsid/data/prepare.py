from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

from gnprsid.common.io import ensure_dir, write_json, write_jsonl
from gnprsid.common.logging import get_logger
from gnprsid.common.paths import dataset_paths
from gnprsid.data.samples import (
    build_sample_rows,
    load_history_map,
    load_sid_token_map,
)
from gnprsid.prompts.render import build_supervised_prompt


logger = get_logger(__name__)


def _write_sft_jsonl(
    output_path: Path,
    rows: Iterable[dict],
    history_source: str,
    history_map: Optional[Dict[int, str]],
) -> None:
    payload = []
    for row in rows:
        payload.append(
            {
                "instruction": f"Predict the next POI for history source '{history_source}'.",
                "input": build_supervised_prompt(row, history_source, history_map=history_map),
                "output": row["target"],
                "sample_id": row["sample_id"],
            }
        )
    write_jsonl(output_path, payload)


def prepare_nyc(dataset: str, current_k: int, sid_map_path: Optional[str | Path] = None) -> dict:
    paths = dataset_paths(dataset)
    ensure_dir(paths.processed)
    ensure_dir(paths.artifacts / "llm")

    sid_map = None
    if sid_map_path:
        sid_map = load_sid_token_map(sid_map_path)
    else:
        default_sid_json = paths.artifacts / "sid" / "pid_to_sid.json"
        if default_sid_json.exists():
            sid_map = load_sid_token_map(default_sid_json)

    sample_manifest: dict[str, dict[str, str]] = {"dataset": dataset, "current_k": current_k, "samples": {}}
    history_id = paths.processed / "history_id.json"
    history_sid = paths.processed / "history_sid.json"
    history_id_map = load_history_map(history_id) if history_id.exists() else {}
    history_sid_map = load_history_map(history_sid) if history_sid.exists() else {}

    for repr_name, repr_sid_map, history_map in [
        ("id", None, history_id_map),
        ("sid", sid_map, history_sid_map),
    ]:
        if repr_name == "sid" and repr_sid_map is None:
            continue
        sample_manifest["samples"][repr_name] = {}
        for split in ["train", "val", "test"]:
            csv_path = paths.processed / f"{split}.csv"
            rows = build_sample_rows(split, csv_path, repr_name, current_k, repr_sid_map)
            sample_path = paths.processed / f"samples_{repr_name}_{split}.jsonl"
            write_jsonl(sample_path, rows)
            sample_manifest["samples"][repr_name][split] = str(sample_path)

            for history_source in ["current", "original"]:
                llm_path = paths.artifacts / "llm" / repr_name / history_source / f"{split}.jsonl"
                _write_sft_jsonl(
                    llm_path,
                    rows,
                    history_source,
                    history_map if history_source == "original" else None,
                )

    write_json(paths.interim / "dataset_manifest.json", sample_manifest)
    logger.info("Prepared NYC dataset with current_k=%s", current_k)
    return sample_manifest
