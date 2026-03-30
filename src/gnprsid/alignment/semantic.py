from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence


SEMANTIC_SCHEMA_NAME = "semantic_spatial_v2"
SUPPORTED_PREFIX_LEVELS = ("a", "ab", "abc")
FULL_SID_LEVEL = "full_sid"
SID_SEGMENT_PATTERN = r"<[a-zA-Z]_\d+>"
_SID_SEGMENT_REGEX = re.compile(SID_SEGMENT_PATTERN)
PROFILE_FIELDS_BY_LEVEL: dict[str, tuple[str, ...]] = {
    "a": ("category",),
    "ab": ("category", "region"),
    "abc": ("category", "region", "geo_bucket"),
}


def sid_segments(sid_token: str) -> list[str]:
    segments = _SID_SEGMENT_REGEX.findall(str(sid_token))
    if len(segments) not in {3, 4}:
        raise ValueError(f"Expected SID with 3 or 4 segments, got {sid_token!r}")
    return segments


def sid_prefix(sid_token: str, level: str) -> str:
    segments = sid_segments(sid_token)
    if level == "a":
        return "".join(segments[:1])
    if level == "ab":
        return "".join(segments[:2])
    if level == "abc":
        return "".join(segments[:3])
    if level == FULL_SID_LEVEL:
        return "".join(segments)
    raise ValueError(f"Unsupported SID level: {level}")


def sid_level(sid_token: str) -> str:
    return FULL_SID_LEVEL if len(sid_segments(sid_token)) == 4 else "abc"


def normalize_category(value: Any) -> str:
    text = " ".join(str(value).strip().split())
    if not text:
        raise ValueError("Category must not be empty")
    return text


def normalize_region(value: Any) -> int | str:
    if value is None:
        raise ValueError("Region must not be null")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("Region must not be empty")
        value = stripped
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isfinite(numeric) and numeric.is_integer():
        return int(numeric)
    return str(value)


def compute_geo_bucket(
    latitude: float,
    longitude: float,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    grid_size: int = 8,
) -> str:
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")

    def _bucket_index(value: float, lower: float, upper: float) -> int:
        if upper <= lower:
            return 0
        normalized = (value - lower) / (upper - lower)
        scaled = int(normalized * grid_size)
        if value >= upper:
            return grid_size - 1
        return min(grid_size - 1, max(0, scaled))

    row = _bucket_index(float(latitude), float(lat_min), float(lat_max))
    col = _bucket_index(float(longitude), float(lon_min), float(lon_max))
    return f"G{row}_{col}"


def parse_geo_bucket(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"G(\d+)_(\d+)", str(value))
    if not match:
        raise ValueError(f"Invalid geo bucket: {value!r}")
    return int(match.group(1)), int(match.group(2))


def profile_for_level(
    level: str,
    category: str,
    region: int | str | None = None,
    geo_bucket: str | None = None,
) -> dict[str, Any]:
    if level not in PROFILE_FIELDS_BY_LEVEL:
        raise ValueError(f"Unsupported profile level: {level}")

    profile: dict[str, Any] = {"category": normalize_category(category)}
    if "region" in PROFILE_FIELDS_BY_LEVEL[level]:
        if region is None:
            raise ValueError(f"Region is required for level {level}")
        profile["region"] = normalize_region(region)
    if "geo_bucket" in PROFILE_FIELDS_BY_LEVEL[level]:
        if geo_bucket is None:
            raise ValueError(f"Geo bucket is required for level {level}")
        profile["geo_bucket"] = str(geo_bucket)
    return profile


def profile_to_json(profile: dict[str, Any], level: str | None = None) -> str:
    if level is None:
        keys = tuple(profile.keys())
    else:
        keys = PROFILE_FIELDS_BY_LEVEL[level]
    ordered = {key: profile[key] for key in keys if key in profile}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def parse_profile_json(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(str(text).strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def validate_profile(profile: dict[str, Any] | None, level: str) -> bool:
    if profile is None:
        return False
    expected = PROFILE_FIELDS_BY_LEVEL[level]
    if tuple(profile.keys()) != expected:
        return False
    try:
        normalize_category(profile["category"])
        if "region" in expected:
            normalize_region(profile["region"])
        if "geo_bucket" in expected:
            if not re.fullmatch(r"G\d+_\d+", str(profile["geo_bucket"])):
                return False
    except ValueError:
        return False
    return True


def mode_with_global_tie_break(
    values: Iterable[Any],
    global_counts: Counter,
) -> Any:
    counter = Counter(values)
    if not counter:
        raise ValueError("Cannot compute mode of empty values")
    return min(
        counter.keys(),
        key=lambda value: (
            -counter[value],
            -global_counts[value],
            str(value),
        ),
    )


def candidate_sampling_weight(
    category: str,
    region: int | str | None,
    category_counts: Counter,
    region_counts: Counter,
) -> float:
    category_count = max(1, int(category_counts[category]))
    region_count = max(1, int(region_counts[region])) if region is not None else 1
    return 1.0 / math.sqrt(category_count * region_count)


def forward_profile_sampling_weight(
    category: str,
    region: int | str | None,
    geo_bucket: str | None,
    category_counts: Counter,
    region_counts: Counter,
    geo_bucket_counts: Counter,
) -> float:
    category_count = max(1, int(category_counts[category]))
    region_count = max(1, int(region_counts[region])) if region is not None else 1
    geo_count = max(1, int(geo_bucket_counts[geo_bucket])) if geo_bucket is not None else 1
    # Keep a mild anti-frequency bias so rare regions get extra exposure
    # without erasing the true popularity prior of common regions.
    weight = 1.0 / (region_count ** 0.35)
    weight *= 1.0 / (category_count ** 0.15)
    weight *= 1.0 / (geo_count ** 0.20)
    return max(0.1, min(weight, 1.0))


def deterministic_sample(
    items: Sequence[Any],
    sample_size: int,
    rng: random.Random,
    weights: Sequence[float] | None = None,
) -> list[Any]:
    if sample_size <= 0 or not items:
        return []
    if weights is None:
        return [items[rng.randrange(len(items))] for _ in range(sample_size)]
    return rng.choices(list(items), weights=list(weights), k=sample_size)


def choose_negative_prefixes(
    all_prefixes: Sequence[str],
    positive_prefix: str,
    incompatible_profiles: set[str],
    prefix_profile_map: dict[str, str],
    negative_count: int,
    rng: random.Random,
) -> list[str]:
    candidates = [
        prefix
        for prefix in all_prefixes
        if prefix != positive_prefix and prefix_profile_map[prefix] not in incompatible_profiles
    ]
    if len(candidates) <= negative_count:
        return sorted(candidates)
    return sorted(rng.sample(candidates, negative_count))


def choose_hard_negative_prefixes(
    *,
    level: str,
    positive_prefix: str,
    positive_profile: Mapping[str, Any],
    prefix_profiles: Mapping[str, Mapping[str, Any]],
    negative_count: int,
    rng: random.Random,
) -> list[str]:
    if level not in PROFILE_FIELDS_BY_LEVEL:
        raise ValueError(f"Unsupported prefix level for negatives: {level}")

    def _geo_distance(left: str | None, right: str | None) -> float:
        if left is None or right is None:
            return float("inf")
        left_row, left_col = parse_geo_bucket(left)
        right_row, right_col = parse_geo_bucket(right)
        return abs(left_row - right_row) + abs(left_col - right_col)

    positive_category = positive_profile["category"]
    positive_region = positive_profile.get("region")
    positive_geo = positive_profile.get("geo_bucket")

    candidates = []
    for prefix, meta in prefix_profiles.items():
        if prefix == positive_prefix:
            continue
        profile = meta["profile"]
        same_category = str(profile["category"]) == str(positive_category)
        same_region = str(profile.get("region")) == str(positive_region)
        same_geo = str(profile.get("geo_bucket")) == str(positive_geo)
        geo_distance = _geo_distance(profile.get("geo_bucket"), positive_geo)
        candidates.append(
            {
                "prefix": prefix,
                "profile": profile,
                "same_category": same_category,
                "same_region": same_region,
                "same_geo": same_geo,
                "geo_distance": geo_distance,
            }
        )

    picked: list[str] = []

    def _take(pool: list[dict[str, Any]], *, key) -> None:
        if not pool or len(picked) >= negative_count:
            return
        ordered = sorted(pool, key=key)
        for item in ordered:
            prefix = str(item["prefix"])
            if prefix not in picked:
                picked.append(prefix)
                return

    if level == "a":
        pool = [item for item in candidates if not item["same_category"]]
        _take(pool, key=lambda item: (str(item["profile"]["category"]), str(item["prefix"])))
    elif level == "ab":
        same_category_diff_region = [
            item for item in candidates if item["same_category"] and not item["same_region"]
        ]
        diff_category_same_region = [
            item for item in candidates if not item["same_category"] and item["same_region"]
        ]
        _take(
            same_category_diff_region,
            key=lambda item: (str(item["profile"].get("region")), str(item["prefix"])),
        )
        _take(
            diff_category_same_region,
            key=lambda item: (str(item["profile"]["category"]), str(item["prefix"])),
        )
    else:
        same_category_diff_region = [
            item for item in candidates if item["same_category"] and not item["same_region"]
        ]
        same_category_region_diff_geo = [
            item
            for item in candidates
            if item["same_category"] and item["same_region"] and not item["same_geo"]
        ]
        diff_category_close_geo = [
            item for item in candidates if not item["same_category"] and item["geo_distance"] < float("inf")
        ]
        _take(
            same_category_diff_region,
            key=lambda item: (item["geo_distance"], str(item["profile"].get("region")), str(item["prefix"])),
        )
        _take(
            same_category_region_diff_geo,
            key=lambda item: (item["geo_distance"], str(item["prefix"])),
        )
        _take(
            diff_category_close_geo,
            key=lambda item: (item["geo_distance"], str(item["profile"]["category"]), str(item["prefix"])),
        )

    remaining = [str(item["prefix"]) for item in candidates if str(item["prefix"]) not in picked]
    rng.shuffle(remaining)
    picked.extend(remaining[: max(0, negative_count - len(picked))])
    return sorted(picked[:negative_count])
