from __future__ import annotations

from pathlib import Path
from typing import Iterable

from gnprsid.common.io import ensure_dir, iter_jsonl, write_json
from gnprsid.common.logging import get_logger
from gnprsid.common.paths import dataset_paths


logger = get_logger(__name__)


def _load_split_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared sample file: {path}")
    return list(iter_jsonl(path))


def build_retrieval_bank(dataset: str, repr_name: str, output_path: str | Path | None = None) -> dict:
    paths = dataset_paths(dataset)
    rows: list[dict] = []
    split_counts = {}
    for split in ["train", "val", "test"]:
        sample_path = paths.processed / f"samples_{repr_name}_{split}.jsonl"
        split_rows = _load_split_rows(sample_path)
        rows.extend(split_rows)
        split_counts[split] = len(split_rows)

    output_path = Path(output_path) if output_path else (paths.artifacts / "retrieval" / f"retrieval_bank_{repr_name}.jsonl")
    ensure_dir(output_path.parent)
    from gnprsid.common.io import write_jsonl

    write_jsonl(output_path, rows)
    manifest = {
        "dataset": dataset,
        "repr": repr_name,
        "output_path": str(output_path),
        "num_rows": len(rows),
        "split_counts": split_counts,
    }
    write_json(output_path.with_suffix(".manifest.json"), manifest)
    logger.info("Built retrieval bank for %s/%s at %s", dataset, repr_name, output_path)
    return manifest
