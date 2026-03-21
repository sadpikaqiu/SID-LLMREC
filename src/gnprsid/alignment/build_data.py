from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pandas as pd

from gnprsid.common.io import ensure_dir, read_json, write_json
from gnprsid.common.logging import get_logger
from gnprsid.common.paths import dataset_paths
from gnprsid.data.samples import parse_literal_list


logger = get_logger(__name__)


def _first_present(row: pd.Series, candidates: list[str]) -> Any:
    for candidate in candidates:
        if candidate in row and pd.notna(row[candidate]):
            return row[candidate]
    return None


def _attribute_dict_from_row(row: pd.Series) -> dict[str, Any]:
    category = _first_present(row, ["category", "Category", "Catname", "catname"])
    region = _first_present(row, ["region", "Region"])
    latitude = _first_present(row, ["latitude", "Latitude", "lat", "Lat"])
    longitude = _first_present(row, ["longitude", "Longitude", "lon", "Lon"])
    visit_hours_raw = _first_present(row, ["visit_time_and_count", "Time", "time"])
    users_raw = _first_present(row, ["Uid", "uid", "users", "Users"])

    visit_hours = parse_literal_list(visit_hours_raw) if visit_hours_raw is not None else []
    user_ids = parse_literal_list(users_raw) if users_raw is not None else []

    attributes: dict[str, Any] = {}
    if category is not None:
        attributes["category"] = str(category)
    if region is not None:
        attributes["region"] = str(region)
    if latitude is not None:
        attributes["latitude"] = float(latitude)
    if longitude is not None:
        attributes["longitude"] = float(longitude)
    if visit_hours:
        attributes["visit_hours"] = [str(hour) for hour in visit_hours]
        attributes["visit_count"] = len(visit_hours)
    if user_ids:
        attributes["unique_user_count"] = len(set(str(value) for value in user_ids))
    return attributes


def _attributes_to_text(attributes: dict[str, Any]) -> str:
    ordered_fields = [
        ("category", "Category"),
        ("region", "Region"),
        ("latitude", "Latitude"),
        ("longitude", "Longitude"),
        ("visit_hours", "Visit hours"),
        ("visit_count", "Visit count"),
        ("unique_user_count", "Unique user count"),
    ]
    chunks = []
    for key, label in ordered_fields:
        if key in attributes:
            chunks.append(f"{label}: {attributes[key]}")
    return "{ " + "; ".join(chunks) + " }"


def build_alignment_data(
    dataset: str,
    sid_map_path: str | Path | None = None,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    paths = dataset_paths(dataset)
    poi_info_path = paths.processed / "poi_info.csv"
    sid_map_path = Path(sid_map_path) if sid_map_path else (paths.artifacts / "sid" / "pid_to_sid.json")

    if not poi_info_path.exists():
        raise FileNotFoundError(f"Missing poi_info.csv: {poi_info_path}")
    if not sid_map_path.exists():
        raise FileNotFoundError(f"Missing SID mapping: {sid_map_path}")

    poi_info = pd.read_csv(poi_info_path)
    sid_payload = read_json(sid_map_path)

    pid_column = "Pid" if "Pid" in poi_info.columns else "pid"
    mapping: dict[str, Any] = {}
    for _, row in poi_info.iterrows():
        pid = int(row[pid_column])
        sid_meta = sid_payload.get(str(pid))
        if not sid_meta:
            continue
        attributes = _attribute_dict_from_row(row)
        sid_token = str(sid_meta["sid_token"])
        mapping[sid_token] = {
            "pid": pid,
            "sid_indices": sid_meta.get("sid_indices", []),
            "attribute_text": _attributes_to_text(attributes),
            "attributes": attributes,
        }

    if not mapping:
        raise ValueError(f"No POIs could be matched between {poi_info_path} and {sid_map_path}")

    instruction_dataset = []
    for sid_token, meta in mapping.items():
        attribute_text = meta["attribute_text"]
        instruction_dataset.append(
            {
                "instruction": "Given POI attributes, describe its semantic ID.",
                "input": f"Attributes: {attribute_text}",
                "output": sid_token,
            }
        )
        instruction_dataset.append(
            {
                "instruction": "Given a semantic ID, describe its POI attributes.",
                "input": f"Semantic ID: {sid_token}",
                "output": attribute_text,
            }
        )

    rng = random.Random(seed)
    rng.shuffle(instruction_dataset)
    valid_size = max(1, int(len(instruction_dataset) * valid_ratio))
    valid_data = instruction_dataset[:valid_size]
    train_data = instruction_dataset[valid_size:]

    output_dir = ensure_dir(paths.artifacts / "alignment")
    mapping_path = output_dir / "semantic_code_mapping.json"
    full_dataset_path = output_dir / "semantic_instruction_dataset.json"
    train_path = output_dir / "train_align.json"
    valid_path = output_dir / "valid_align.json"
    manifest_path = output_dir / "alignment_manifest.json"

    write_json(mapping_path, mapping)
    write_json(full_dataset_path, instruction_dataset)
    write_json(train_path, train_data)
    write_json(valid_path, valid_data)

    manifest = {
        "dataset": dataset,
        "sid_map_path": str(sid_map_path),
        "poi_info_path": str(poi_info_path),
        "mapping_path": str(mapping_path),
        "instruction_dataset_path": str(full_dataset_path),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "num_semantic_ids": len(mapping),
        "num_examples": len(instruction_dataset),
        "num_train": len(train_data),
        "num_valid": len(valid_data),
        "valid_ratio": valid_ratio,
        "seed": seed,
    }
    write_json(manifest_path, manifest)
    logger.info("Built alignment dataset for %s with %s semantic IDs", dataset, len(mapping))
    return manifest
