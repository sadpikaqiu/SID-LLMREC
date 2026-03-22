from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

from gnprsid.common.io import copy_file, copy_tree, ensure_dir, write_json
from gnprsid.common.logging import get_logger
from gnprsid.common.paths import dataset_paths


logger = get_logger(__name__)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    copy_file(src, dst)
    return True


def _serialize_hour_counter(hours: Iterable[int]) -> dict[str, int]:
    counts = Counter(int(hour) for hour in hours)
    return {str(hour): int(count) for hour, count in sorted(counts.items())}


def _read_raw_checkin_table(path: Path) -> pd.DataFrame:
    common_kwargs = {
        "sep": "\t",
        "header": None,
        "names": [
            "original_uid",
            "Original_Pid",
            "category_id",
            "category",
            "latitude",
            "longitude",
            "timezone_offset",
            "utc_time",
        ],
        "on_bad_lines": "skip",
    }
    for encoding in ["utf-8", "latin1"]:
        try:
            return pd.read_csv(path, encoding=encoding, **common_kwargs)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("codec", b"", 0, 1, f"Could not decode raw checkin file: {path}")


def _enrich_poi_info_from_raw(dataset: str, legacy_root: Path, paths) -> Path | None:
    raw_txt_path = legacy_root / "datasets" / f"{dataset}.txt"
    pid_mapping_path = paths.processed / "pid_mapping.csv"
    poi_info_path = paths.processed / "poi_info.csv"
    if not raw_txt_path.exists() or not pid_mapping_path.exists() or not poi_info_path.exists():
        return None

    raw_copy_path = paths.raw / f"{dataset}.txt"
    copy_file(raw_txt_path, raw_copy_path)

    raw_df = _read_raw_checkin_table(raw_txt_path)
    raw_df["utc_time"] = pd.to_datetime(raw_df["utc_time"], utc=True, errors="coerce")
    raw_df = raw_df.dropna(subset=["Original_Pid", "latitude", "longitude", "utc_time"])
    raw_df["visit_hour"] = raw_df["utc_time"].dt.hour.astype(int)

    pid_mapping = pd.read_csv(pid_mapping_path)
    mapped = raw_df.merge(pid_mapping, on="Original_Pid", how="inner")
    mapped["Mapped_Pid"] = mapped["Mapped_Pid"].astype(int)

    grouped = (
        mapped.groupby("Mapped_Pid", sort=True)
        .agg(
            original_pid=("Original_Pid", "first"),
            category=("category", "first"),
            category_id=("category_id", "first"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            unique_user_count=("original_uid", "nunique"),
            visit_count=("visit_hour", "size"),
            visit_hours=("visit_hour", lambda values: sorted({int(item) for item in values})),
            visit_time_and_count=("visit_hour", lambda values: _serialize_hour_counter(values)),
        )
        .reset_index()
        .rename(columns={"Mapped_Pid": "Pid"})
    )

    legacy_poi_info = pd.read_csv(poi_info_path)
    legacy_copy_path = paths.processed / "poi_info_legacy.csv"
    legacy_poi_info.to_csv(legacy_copy_path, index=False, encoding="utf-8")

    enriched = legacy_poi_info.merge(grouped, on="Pid", how="left")
    enriched["pid"] = enriched["Pid"]
    if "Catname" in enriched.columns and "category" in enriched.columns:
        enriched["category"] = enriched["category"].fillna(enriched["Catname"].astype(str))
    if "Region" in enriched.columns:
        enriched["region"] = enriched["Region"].astype(str)
    if "visit_time_and_count" in enriched.columns:
        enriched["visit_time_and_count"] = enriched["visit_time_and_count"].apply(
            lambda value: value if isinstance(value, dict) else {}
        )
    enriched.to_csv(poi_info_path, index=False, encoding="utf-8")
    logger.info("Enriched poi_info.csv with latitude/longitude from %s", raw_txt_path)
    return poi_info_path


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

    enriched_poi_info = _enrich_poi_info_from_raw(dataset, legacy_root, paths)

    manifest = {
        "dataset": dataset,
        "legacy_root": str(legacy_root),
        "legacy_dataset_root": str(old_dataset_root),
        "canonical_files": copied,
        "legacy_snapshot": str(legacy_snapshot),
        "enriched_poi_info": str(enriched_poi_info) if enriched_poi_info else None,
    }
    write_json(paths.interim / "legacy_import_manifest.json", manifest)
    logger.info("Imported legacy dataset %s into %s", dataset, paths.root)
    return manifest
