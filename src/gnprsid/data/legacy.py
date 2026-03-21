from __future__ import annotations

from pathlib import Path
from typing import Iterable

from gnprsid.common.io import copy_file, copy_tree, ensure_dir, write_json
from gnprsid.common.logging import get_logger
from gnprsid.common.paths import dataset_paths


logger = get_logger(__name__)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    copy_file(src, dst)
    return True


def import_legacy_dataset(dataset: str, legacy_root: str | Path) -> dict:
    legacy_root = Path(legacy_root)
    old_dataset_root = legacy_root / "datasets" / dataset
    old_data_dir = old_dataset_root / "data"
    old_eval_dir = old_dataset_root / "eval"

    if not old_dataset_root.exists():
        raise FileNotFoundError(f"Legacy dataset root not found: {old_dataset_root}")

    paths = dataset_paths(dataset)
    ensure_dir(paths.raw)
    ensure_dir(paths.interim)
    ensure_dir(paths.processed)
    ensure_dir(paths.artifacts)
    ensure_dir(paths.outputs / "legacy_eval")

    legacy_snapshot = paths.raw / "legacy_snapshot"
    copy_tree(old_dataset_root, legacy_snapshot)

    canonical_copies = {
        old_data_dir / "train.csv": paths.processed / "train.csv",
        old_data_dir / "val.csv": paths.processed / "val.csv",
        old_data_dir / "test.csv": paths.processed / "test.csv",
        old_data_dir / "history.csv": paths.processed / "history.csv",
        old_dataset_root / "poi_info.csv": paths.processed / "poi_info.csv",
        old_dataset_root / "pid_mapping.csv": paths.processed / "pid_mapping.csv",
        old_dataset_root / "uid_mapping.csv": paths.processed / "uid_mapping.csv",
        old_dataset_root / "region_mapping.csv": paths.processed / "region_mapping.csv",
        old_dataset_root / "catname_mapping.csv": paths.processed / "catname_mapping.csv",
        old_data_dir / "history_id.json": paths.processed / "history_id.json",
        old_data_dir / "history_codebook.json": paths.processed / "history_sid.json",
        old_data_dir / "test_id.json": paths.processed / "test_id.json",
        old_data_dir / "test_codebook.json": paths.processed / "test_sid.json",
        old_data_dir / "retrieval_bank_id.jsonl": paths.artifacts / "retrieval" / "legacy" / "retrieval_bank_id.jsonl",
        old_data_dir / "retrieval_bank_codebook.jsonl": paths.artifacts / "retrieval" / "legacy" / "retrieval_bank_sid.jsonl",
        old_data_dir / "similar_map_test_id.json": paths.artifacts / "retrieval" / "legacy" / "similar_map_test_id.json",
        old_data_dir / "similar_map_test_codebook.json": paths.artifacts / "retrieval" / "legacy" / "similar_map_test_sid.json",
    }

    copied = []
    for src, dst in canonical_copies.items():
        if _copy_if_exists(src, dst):
            copied.append(str(dst))

    if old_eval_dir.exists():
        copy_tree(old_eval_dir, paths.outputs / "legacy_eval")

    manifest = {
        "dataset": dataset,
        "legacy_root": str(legacy_root),
        "legacy_dataset_root": str(old_dataset_root),
        "canonical_files": copied,
        "legacy_snapshot": str(legacy_snapshot),
    }
    write_json(paths.interim / "legacy_import_manifest.json", manifest)
    logger.info("Imported legacy dataset %s into %s", dataset, paths.root)
    return manifest
