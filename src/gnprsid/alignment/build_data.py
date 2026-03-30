from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

from gnprsid.alignment.semantic import (
    FULL_SID_LEVEL,
    PROFILE_FIELDS_BY_LEVEL,
    SEMANTIC_SCHEMA_NAME,
    SUPPORTED_PREFIX_LEVELS,
    choose_hard_negative_prefixes,
    compute_geo_bucket,
    deterministic_sample,
    forward_profile_sampling_weight,
    mode_with_global_tie_break,
    normalize_category,
    normalize_region,
    profile_for_level,
    profile_to_json,
    sid_prefix,
)
from gnprsid.common.io import ensure_dir, read_json, write_json, write_jsonl
from gnprsid.common.logging import get_logger
from gnprsid.common.paths import dataset_paths


logger = get_logger(__name__)
NEGATIVE_CANDIDATE_COUNT = 3
PHASE_B1_TASK_MIX = {
    "full_sid_to_abc_profile": 0.75,
    "abc_to_abc_profile": 0.25,
}
PHASE_B2_TASK_MIX = {
    "full_sid_to_abc_profile": 0.50,
    "abc_to_abc_profile": 0.20,
    "abc_profile_to_abc": 0.20,
    "phase_a_replay": 0.10,
}


def _first_present(row: pd.Series, candidates: list[str]) -> Any:
    for candidate in candidates:
        if candidate in row and pd.notna(row[candidate]):
            return row[candidate]
    return None


def _build_geo_bounds(rows: list[dict[str, Any]]) -> dict[str, float]:
    latitudes = [float(row["latitude"]) for row in rows]
    longitudes = [float(row["longitude"]) for row in rows]
    return {
        "lat_min": min(latitudes),
        "lat_max": max(latitudes),
        "lon_min": min(longitudes),
        "lon_max": max(longitudes),
    }


def _load_semantic_rows(
    poi_info_path: Path,
    sid_map_path: Path,
    grid_size: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    poi_info = pd.read_csv(poi_info_path)
    sid_payload = read_json(sid_map_path)
    pid_column = "Pid" if "Pid" in poi_info.columns else "pid"

    raw_rows: list[dict[str, Any]] = []
    for _, row in poi_info.iterrows():
        pid = int(row[pid_column])
        sid_meta = sid_payload.get(str(pid))
        if not sid_meta:
            continue

        category = normalize_category(_first_present(row, ["category", "Category", "Catname", "catname"]))
        region = normalize_region(_first_present(row, ["region", "Region"]))
        latitude = float(_first_present(row, ["latitude", "Latitude", "lat", "Lat"]))
        longitude = float(_first_present(row, ["longitude", "Longitude", "lon", "Lon"]))
        full_sid = str(sid_meta["sid_token"])
        raw_rows.append(
            {
                "pid": pid,
                "full_sid": full_sid,
                "a": sid_prefix(full_sid, "a"),
                "ab": sid_prefix(full_sid, "ab"),
                "abc": sid_prefix(full_sid, "abc"),
                "category": category,
                "region": region,
                "latitude": latitude,
                "longitude": longitude,
            }
        )

    if not raw_rows:
        raise ValueError(f"No POIs matched between {poi_info_path} and {sid_map_path}")

    bounds = _build_geo_bounds(raw_rows)
    for row in raw_rows:
        row["geo_bucket"] = compute_geo_bucket(
            row["latitude"],
            row["longitude"],
            bounds["lat_min"],
            bounds["lat_max"],
            bounds["lon_min"],
            bounds["lon_max"],
            grid_size=grid_size,
        )
    return raw_rows, bounds


def _build_prefix_prototypes(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], Counter, Counter, Counter]:
    category_counts = Counter(row["category"] for row in rows)
    region_counts = Counter(row["region"] for row in rows)
    geo_bucket_counts = Counter(row["geo_bucket"] for row in rows)
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        "a": defaultdict(list),
        "ab": defaultdict(list),
        "abc": defaultdict(list),
    }
    for row in rows:
        grouped["a"][row["a"]].append(row)
        grouped["ab"][row["ab"]].append(row)
        grouped["abc"][row["abc"]].append(row)

    prototypes: dict[str, dict[str, dict[str, Any]]] = {level: {} for level in SUPPORTED_PREFIX_LEVELS}
    for level in SUPPORTED_PREFIX_LEVELS:
        for prefix, members in grouped[level].items():
            category = mode_with_global_tie_break((member["category"] for member in members), category_counts)
            if level == "a":
                profile = profile_for_level(level, category=category)
            elif level == "ab":
                region = mode_with_global_tie_break((member["region"] for member in members), region_counts)
                profile = profile_for_level(level, category=category, region=region)
            else:
                region = mode_with_global_tie_break((member["region"] for member in members), region_counts)
                geo_counts = Counter(member["geo_bucket"] for member in members)
                geo_bucket = min(
                    geo_counts.keys(),
                    key=lambda value: (-geo_counts[value], str(value)),
                )
                profile = profile_for_level(level, category=category, region=region, geo_bucket=geo_bucket)
            prototypes[level][prefix] = {
                "target_prefix": prefix,
                "sid_level": level,
                "profile": profile,
                "profile_json": profile_to_json(profile, level=level),
                "member_count": len(members),
            }
    return prototypes, category_counts, region_counts, geo_bucket_counts


def _build_purity_report(
    rows: list[dict[str, Any]],
    bounds: dict[str, float],
    grid_size: int,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        "a": defaultdict(list),
        "ab": defaultdict(list),
        "abc": defaultdict(list),
    }
    for row in rows:
        grouped["a"][row["a"]].append(row)
        grouped["ab"][row["ab"]].append(row)
        grouped["abc"][row["abc"]].append(row)

    report: dict[str, Any] = {
        "semantic_schema": SEMANTIC_SCHEMA_NAME,
        "grid_size": grid_size,
        "bounds": bounds,
        "num_pois": len(rows),
        "levels": {},
    }
    for level in SUPPORTED_PREFIX_LEVELS:
        field_summary: dict[str, Any] = {}
        for field in PROFILE_FIELDS_BY_LEVEL[level]:
            purities = []
            for members in grouped[level].values():
                counter = Counter(member[field] for member in members)
                purities.append(max(counter.values()) / len(members))
            field_summary[field] = {
                "mean": sum(purities) / len(purities),
                "median": median(purities),
                "min": min(purities),
                "max": max(purities),
            }
        report["levels"][level] = {
            "num_groups": len(grouped[level]),
            "field_purity": field_summary,
        }
    return report


def _instruction_for_task(task_type: str) -> str:
    instructions = {
        "a_to_category_profile": "Given a semantic prefix, output its category profile JSON.",
        "category_profile_to_a": "Given a category profile JSON and candidate prefixes, output the single matching semantic prefix.",
        "ab_to_category_region_profile": "Given a semantic prefix, output its category-region profile JSON.",
        "category_region_profile_to_ab": "Given a category-region profile JSON and candidate prefixes, output the single matching semantic prefix.",
        "abc_to_abc_profile": "Given a semantic prefix, output its semantic profile JSON.",
        "abc_profile_to_abc": "Given a semantic profile JSON and candidate prefixes, output the single matching semantic prefix.",
        "full_sid_to_abc_profile": "Given a full semantic ID, output the semantic core profile JSON for its abc prefix.",
    }
    return instructions[task_type]


def _candidate_text(candidates: list[str]) -> str:
    return "Candidate prefixes: " + " ".join(candidates)


def _record(
    *,
    task_type: str,
    sid_level: str,
    source_sid: str,
    source_abc: str,
    target_prefix: str,
    category: str,
    region: int | str | None,
    geo_bucket: str | None,
    input_text: str,
    output_text: str,
    candidate_prefixes: list[str] | None = None,
    sample_id: str,
) -> dict[str, Any]:
    record = {
        "task_type": task_type,
        "sid_level": sid_level,
        "source_sid": source_sid,
        "source_abc": source_abc,
        "target_prefix": target_prefix,
        "category": category,
        "region": region,
        "geo_bucket": geo_bucket,
        "instruction": _instruction_for_task(task_type),
        "input": input_text,
        "output": output_text,
        "sample_id": sample_id,
    }
    if candidate_prefixes is not None:
        record["candidate_prefixes"] = list(candidate_prefixes)
    return record


def _build_phase_a_records(
    abc_prefixes: list[str],
    prefix_profiles: dict[str, dict[str, dict[str, Any]]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for abc_prefix in sorted(abc_prefixes):
        a_prefix = sid_prefix(abc_prefix, "a")
        ab_prefix = sid_prefix(abc_prefix, "ab")
        a_meta = prefix_profiles["a"][a_prefix]
        ab_meta = prefix_profiles["ab"][ab_prefix]

        records.append(
            _record(
                task_type="a_to_category_profile",
                sid_level="a",
                source_sid=a_prefix,
                source_abc=abc_prefix,
                target_prefix=a_prefix,
                category=a_meta["profile"]["category"],
                region=None,
                geo_bucket=None,
                input_text=f"Semantic prefix (level a): {a_prefix}",
                output_text=a_meta["profile_json"],
                sample_id=f"{abc_prefix}::a_to_profile",
            )
        )

        a_negatives = choose_hard_negative_prefixes(
            level="a",
            positive_prefix=a_prefix,
            positive_profile=a_meta["profile"],
            prefix_profiles=prefix_profiles["a"],
            negative_count=NEGATIVE_CANDIDATE_COUNT,
            rng=rng,
        )
        a_candidates = sorted([a_prefix, *a_negatives])
        records.append(
            _record(
                task_type="category_profile_to_a",
                sid_level="a",
                source_sid=a_prefix,
                source_abc=abc_prefix,
                target_prefix=a_prefix,
                category=a_meta["profile"]["category"],
                region=None,
                geo_bucket=None,
                input_text="\n".join(
                    [
                        f'Semantic profile: {a_meta["profile_json"]}',
                        _candidate_text(a_candidates),
                    ]
                ),
                output_text=a_prefix,
                candidate_prefixes=a_candidates,
                sample_id=f"{abc_prefix}::profile_to_a",
            )
        )

        records.append(
            _record(
                task_type="ab_to_category_region_profile",
                sid_level="ab",
                source_sid=ab_prefix,
                source_abc=abc_prefix,
                target_prefix=ab_prefix,
                category=ab_meta["profile"]["category"],
                region=ab_meta["profile"]["region"],
                geo_bucket=None,
                input_text=f"Semantic prefix (level ab): {ab_prefix}",
                output_text=ab_meta["profile_json"],
                sample_id=f"{abc_prefix}::ab_to_profile",
            )
        )

        ab_negatives = choose_hard_negative_prefixes(
            level="ab",
            positive_prefix=ab_prefix,
            positive_profile=ab_meta["profile"],
            prefix_profiles=prefix_profiles["ab"],
            negative_count=NEGATIVE_CANDIDATE_COUNT,
            rng=rng,
        )
        ab_candidates = sorted([ab_prefix, *ab_negatives])
        records.append(
            _record(
                task_type="category_region_profile_to_ab",
                sid_level="ab",
                source_sid=ab_prefix,
                source_abc=abc_prefix,
                target_prefix=ab_prefix,
                category=ab_meta["profile"]["category"],
                region=ab_meta["profile"]["region"],
                geo_bucket=None,
                input_text="\n".join(
                    [
                        f'Semantic profile: {ab_meta["profile_json"]}',
                        _candidate_text(ab_candidates),
                    ]
                ),
                output_text=ab_prefix,
                candidate_prefixes=ab_candidates,
                sample_id=f"{abc_prefix}::profile_to_ab",
            )
        )
    return records


def _build_phase_b_records(
    abc_prefixes: list[str],
    semantic_rows: list[dict[str, Any]],
    prefix_profiles: dict[str, dict[str, dict[str, Any]]],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    abc_records: list[dict[str, Any]] = []
    reverse_records: list[dict[str, Any]] = []
    full_sid_records: list[dict[str, Any]] = []

    for abc_prefix in sorted(abc_prefixes):
        abc_meta = prefix_profiles["abc"][abc_prefix]
        abc_records.append(
            _record(
                task_type="abc_to_abc_profile",
                sid_level="abc",
                source_sid=abc_prefix,
                source_abc=abc_prefix,
                target_prefix=abc_prefix,
                category=abc_meta["profile"]["category"],
                region=abc_meta["profile"]["region"],
                geo_bucket=abc_meta["profile"]["geo_bucket"],
                input_text=f"Semantic prefix (level abc): {abc_prefix}",
                output_text=abc_meta["profile_json"],
                sample_id=f"{abc_prefix}::abc_to_profile",
            )
        )

        abc_negatives = choose_hard_negative_prefixes(
            level="abc",
            positive_prefix=abc_prefix,
            positive_profile=abc_meta["profile"],
            prefix_profiles=prefix_profiles["abc"],
            negative_count=NEGATIVE_CANDIDATE_COUNT,
            rng=rng,
        )
        abc_candidates = sorted([abc_prefix, *abc_negatives])
        reverse_records.append(
            _record(
                task_type="abc_profile_to_abc",
                sid_level="abc",
                source_sid=abc_prefix,
                source_abc=abc_prefix,
                target_prefix=abc_prefix,
                category=abc_meta["profile"]["category"],
                region=abc_meta["profile"]["region"],
                geo_bucket=abc_meta["profile"]["geo_bucket"],
                input_text="\n".join(
                    [
                        f'Semantic profile: {abc_meta["profile_json"]}',
                        _candidate_text(abc_candidates),
                    ]
                ),
                output_text=abc_prefix,
                candidate_prefixes=abc_candidates,
                sample_id=f"{abc_prefix}::profile_to_abc",
            )
        )

    for row in semantic_rows:
        abc_meta = prefix_profiles["abc"][row["abc"]]
        full_sid_records.append(
            _record(
                task_type="full_sid_to_abc_profile",
                sid_level=FULL_SID_LEVEL,
                source_sid=row["full_sid"],
                source_abc=row["abc"],
                target_prefix=row["abc"],
                category=abc_meta["profile"]["category"],
                region=abc_meta["profile"]["region"],
                geo_bucket=abc_meta["profile"]["geo_bucket"],
                input_text=f"Semantic ID: {row['full_sid']}",
                output_text=abc_meta["profile_json"],
                sample_id=f"{row['full_sid']}::full_to_profile",
            )
        )
    return abc_records, reverse_records, full_sid_records


def _split_abc_prefixes(
    abc_prefixes: list[str],
    valid_ratio: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError("valid_ratio must be between 0 and 1")
    ordered = sorted(abc_prefixes)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    valid_size = max(1, int(len(ordered) * valid_ratio))
    valid_abc = set(ordered[:valid_size])
    train_abc = set(ordered[valid_size:])
    if not train_abc:
        raise ValueError("valid_ratio leaves no abc groups for training")
    return train_abc, valid_abc


def _shuffle_records(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _task_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(record["task_type"] for record in records)
    return dict(sorted(counts.items()))


def _sample_task_family(
    records: list[dict[str, Any]],
    desired_count: int,
    rng: random.Random,
    *,
    category_counts: Counter,
    region_counts: Counter,
    geo_bucket_counts: Counter,
    use_forward_weights: bool,
) -> list[dict[str, Any]]:
    if not records or desired_count <= 0:
        return []
    weights = None
    if use_forward_weights:
        weights = [
            forward_profile_sampling_weight(
                str(record["category"]),
                record.get("region"),
                record.get("geo_bucket"),
                category_counts,
                region_counts,
                geo_bucket_counts,
            )
            for record in records
        ]
    return deterministic_sample(records, desired_count, rng=rng, weights=weights)


def _mixed_phase_train_records(
    *,
    families: dict[str, list[dict[str, Any]]],
    task_mix: dict[str, float],
    category_counts: Counter,
    region_counts: Counter,
    geo_bucket_counts: Counter,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    total_target = sum(len(records) for records in families.values())
    desired_counts: dict[str, int] = {}
    assigned = 0
    family_names = list(task_mix.keys())
    for index, family_name in enumerate(family_names):
        ratio = task_mix[family_name]
        if index == len(family_names) - 1:
            count = total_target - assigned
        else:
            count = int(round(total_target * ratio))
            assigned += count
        desired_counts[family_name] = count

    phase_b_records: list[dict[str, Any]] = []
    for family_name, records in families.items():
        phase_b_records.extend(
            _sample_task_family(
                records,
                desired_counts[family_name],
                rng,
                category_counts=category_counts,
                region_counts=region_counts,
                geo_bucket_counts=geo_bucket_counts,
                use_forward_weights=family_name in {"full_sid_to_abc_profile", "abc_to_abc_profile"},
            )
        )
    rng.shuffle(phase_b_records)
    return phase_b_records


def build_alignment_data(
    dataset: str,
    sid_map_path: str | Path | None = None,
    valid_ratio: float = 0.1,
    seed: int = 42,
    semantic_schema: str = SEMANTIC_SCHEMA_NAME,
    grid_size: int = 8,
    split_by: str = "abc",
) -> dict[str, Any]:
    if semantic_schema != SEMANTIC_SCHEMA_NAME:
        raise ValueError(f"Unsupported semantic schema: {semantic_schema}")
    if split_by != "abc":
        raise ValueError("split_by must be 'abc' for semantic_spatial_v2")

    paths = dataset_paths(dataset)
    poi_info_path = paths.processed / "poi_info.csv"
    sid_map_path = Path(sid_map_path) if sid_map_path else (paths.artifacts / "sid" / "pid_to_sid.json")
    if not poi_info_path.exists():
        raise FileNotFoundError(f"Missing poi_info.csv: {poi_info_path}")
    if not sid_map_path.exists():
        raise FileNotFoundError(f"Missing SID mapping: {sid_map_path}")

    semantic_rows, bounds = _load_semantic_rows(poi_info_path, sid_map_path, grid_size=grid_size)
    prefix_profiles, category_counts, region_counts, geo_bucket_counts = _build_prefix_prototypes(semantic_rows)
    purity_report = _build_purity_report(semantic_rows, bounds=bounds, grid_size=grid_size)

    output_dir = ensure_dir(paths.artifacts / "alignment")
    purity_path = output_dir / "semantic_purity_report.json"
    profile_paths = {
        "a": output_dir / "prefix_profiles_a.json",
        "ab": output_dir / "prefix_profiles_ab.json",
        "abc": output_dir / "prefix_profiles_abc.json",
    }

    abc_prefixes = sorted(prefix_profiles["abc"])
    train_abc, valid_abc = _split_abc_prefixes(abc_prefixes, valid_ratio=valid_ratio, seed=seed)
    rng = random.Random(seed)

    phase_a_all = _build_phase_a_records(abc_prefixes, prefix_profiles, rng)
    abc_records, reverse_records, full_sid_records = _build_phase_b_records(
        abc_prefixes,
        semantic_rows,
        prefix_profiles,
        rng,
    )

    phase_a_train = [record for record in phase_a_all if record["source_abc"] in train_abc]
    phase_a_valid = [record for record in phase_a_all if record["source_abc"] in valid_abc]
    abc_train = [record for record in abc_records if record["source_abc"] in train_abc]
    abc_valid = [record for record in abc_records if record["source_abc"] in valid_abc]
    reverse_train = [record for record in reverse_records if record["source_abc"] in train_abc]
    reverse_valid = [record for record in reverse_records if record["source_abc"] in valid_abc]
    full_sid_train = [record for record in full_sid_records if record["source_abc"] in train_abc]
    full_sid_valid = [record for record in full_sid_records if record["source_abc"] in valid_abc]

    phase_b1_train = _mixed_phase_train_records(
        families={
            "full_sid_to_abc_profile": full_sid_train,
            "abc_to_abc_profile": abc_train,
        },
        task_mix=PHASE_B1_TASK_MIX,
        category_counts=category_counts,
        region_counts=region_counts,
        geo_bucket_counts=geo_bucket_counts,
        seed=seed,
    )
    phase_b2_train = _mixed_phase_train_records(
        families={
            "full_sid_to_abc_profile": full_sid_train,
            "abc_to_abc_profile": abc_train,
            "abc_profile_to_abc": reverse_train,
            "phase_a_replay": phase_a_train,
        },
        task_mix=PHASE_B2_TASK_MIX,
        category_counts=category_counts,
        region_counts=region_counts,
        geo_bucket_counts=geo_bucket_counts,
        seed=seed,
    )
    phase_b1_valid = _shuffle_records(
        [*abc_valid, *full_sid_valid],
        seed=seed,
    )
    phase_b2_valid = _shuffle_records(
        [*abc_valid, *reverse_valid, *full_sid_valid, *phase_a_valid],
        seed=seed,
    )
    phase_a_train = _shuffle_records(phase_a_train, seed=seed)
    phase_a_valid = _shuffle_records(phase_a_valid, seed=seed)

    train_phase_a_path = output_dir / "train_align_phase_a.jsonl"
    valid_phase_a_path = output_dir / "valid_align_phase_a.jsonl"
    train_phase_b1_path = output_dir / "train_align_phase_b1.jsonl"
    valid_phase_b1_path = output_dir / "valid_align_phase_b1.jsonl"
    train_phase_b2_path = output_dir / "train_align_phase_b2.jsonl"
    valid_phase_b2_path = output_dir / "valid_align_phase_b2.jsonl"
    manifest_path = output_dir / "alignment_manifest.json"

    write_json(purity_path, purity_report)
    for level, profile_path in profile_paths.items():
        write_json(profile_path, prefix_profiles[level])
    write_jsonl(train_phase_a_path, phase_a_train)
    write_jsonl(valid_phase_a_path, phase_a_valid)
    write_jsonl(train_phase_b1_path, phase_b1_train)
    write_jsonl(valid_phase_b1_path, phase_b1_valid)
    write_jsonl(train_phase_b2_path, phase_b2_train)
    write_jsonl(valid_phase_b2_path, phase_b2_valid)

    manifest = {
        "dataset": dataset,
        "semantic_schema": semantic_schema,
        "grid_size": grid_size,
        "split_by": split_by,
        "sid_map_path": str(sid_map_path),
        "poi_info_path": str(poi_info_path),
        "purity_report_path": str(purity_path),
        "profile_paths": {level: str(path) for level, path in profile_paths.items()},
        "phase_a_train_path": str(train_phase_a_path),
        "phase_a_valid_path": str(valid_phase_a_path),
        "phase_b1_train_path": str(train_phase_b1_path),
        "phase_b1_valid_path": str(valid_phase_b1_path),
        "phase_b2_train_path": str(train_phase_b2_path),
        "phase_b2_valid_path": str(valid_phase_b2_path),
        "num_pois": len(semantic_rows),
        "num_abc_groups": len(abc_prefixes),
        "num_train_abc_groups": len(train_abc),
        "num_valid_abc_groups": len(valid_abc),
        "phase_a_train_examples": len(phase_a_train),
        "phase_a_valid_examples": len(phase_a_valid),
        "phase_b1_train_examples": len(phase_b1_train),
        "phase_b1_valid_examples": len(phase_b1_valid),
        "phase_b2_train_examples": len(phase_b2_train),
        "phase_b2_valid_examples": len(phase_b2_valid),
        "phase_a_task_counts_train": _task_counts(phase_a_train),
        "phase_a_task_counts_valid": _task_counts(phase_a_valid),
        "phase_b1_task_counts_train": _task_counts(phase_b1_train),
        "phase_b1_task_counts_valid": _task_counts(phase_b1_valid),
        "phase_b2_task_counts_train": _task_counts(phase_b2_train),
        "phase_b2_task_counts_valid": _task_counts(phase_b2_valid),
        "seed": seed,
        "valid_ratio": valid_ratio,
        "negative_candidate_count": NEGATIVE_CANDIDATE_COUNT,
        "phase_b1_task_mix": PHASE_B1_TASK_MIX,
        "phase_b2_task_mix": PHASE_B2_TASK_MIX,
    }
    write_json(manifest_path, manifest)
    logger.info(
        "Built semantic alignment dataset for %s with %s abc groups",
        dataset,
        len(abc_prefixes),
    )
    return manifest
