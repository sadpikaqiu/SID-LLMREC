from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
from openlocationcode import openlocationcode as olc

from gnprsid.common.io import copy_file, ensure_dir, write_json
from gnprsid.common.logging import get_logger
from gnprsid.common.paths import DatasetPaths, dataset_paths


logger = get_logger(__name__)


RAW_COLUMNS = [
    "original_uid",
    "original_pid",
    "category_id",
    "category",
    "latitude",
    "longitude",
    "timezone_offset",
    "utc_time",
]


@dataclass(frozen=True)
class RawBuildConfig:
    dataset: str = "NYC"
    poi_min_freq: int = 10
    user_min_freq: int = 10
    train_ratio: float = 0.8
    window_size: int = 50
    step_size: int = 10
    mask_prob: float = 0.1
    max_user_train_events: int = 80
    min_sequence_len: int = 10
    seed: int = 42


def _resolve_paths(dataset: str, output_root: str | Path | None = None) -> DatasetPaths:
    if output_root is None:
        return dataset_paths(dataset)

    root = Path(output_root)
    return DatasetPaths(
        dataset=dataset,
        root=root,
        raw=root / "raw",
        interim=root / "interim",
        processed=root / "processed",
        artifacts=root / "artifacts",
        checkpoints=root / "checkpoints",
        outputs=root / "outputs",
    )


def _copy_raw_if_needed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return
    copy_file(src, dst)


def _read_raw_checkins(raw_path: Path) -> pd.DataFrame:
    kwargs = {
        "sep": "\t",
        "header": None,
        "names": RAW_COLUMNS,
        "on_bad_lines": "skip",
    }
    for encoding in ["utf-8", "latin1"]:
        try:
            return pd.read_csv(raw_path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("codec", b"", 0, 1, f"Could not decode raw checkin file: {raw_path}")


def _plus_code_prefix(latitude: float, longitude: float) -> str:
    return olc.encode(latitude, longitude)[:6]


def _prepare_raw_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["utc_time"] = pd.to_datetime(df["utc_time"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce")
    df = df.dropna(subset=["original_uid", "original_pid", "latitude", "longitude", "timezone_offset", "utc_time"])

    df["region"] = df.apply(
        lambda row: _plus_code_prefix(float(row["latitude"]), float(row["longitude"])),
        axis=1,
    )
    df["time_dt"] = df["utc_time"] + df["timezone_offset"].apply(lambda value: timedelta(minutes=int(value)))
    df["Time"] = df["time_dt"].dt.strftime("%Y-%m-%d %H:%M")
    df["visit_hour"] = df["time_dt"].dt.hour.astype(int)
    df["original_uid"] = df["original_uid"].astype(int)
    df["original_pid"] = df["original_pid"].astype(str)
    df["category"] = df["category"].astype(str)
    df["category_id"] = df["category_id"].astype(str)
    df["region"] = df["region"].astype(str)
    return df


def _filter_min_frequency(df: pd.DataFrame, poi_min_freq: int, user_min_freq: int) -> pd.DataFrame:
    filtered = df.copy()
    filtered["poi_freq"] = filtered.groupby("original_pid")["original_uid"].transform("count")
    filtered = filtered[filtered["poi_freq"] >= poi_min_freq].copy()
    filtered["user_freq"] = filtered.groupby("original_uid")["original_pid"].transform("count")
    filtered = filtered[filtered["user_freq"] >= user_min_freq].copy()
    return filtered.drop(columns=["poi_freq", "user_freq"]).reset_index(drop=True)


def _build_mapping_frame(values: Iterable, original_col: str, mapped_col: str) -> pd.DataFrame:
    ordered = list(values)
    return pd.DataFrame(
        {
            original_col: ordered,
            mapped_col: list(range(1, len(ordered) + 1)),
        }
    )


def _build_mappings(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "uid": _build_mapping_frame(sorted(df["original_uid"].unique().tolist()), "Original_Uid", "Mapped_Uid"),
        "pid": _build_mapping_frame(sorted(df["original_pid"].unique().tolist()), "Original_Pid", "Mapped_Pid"),
        "cat": _build_mapping_frame(sorted(df["category"].unique().tolist()), "Original_Catname", "Mapped_Catname"),
        "region": _build_mapping_frame(sorted(df["region"].unique().tolist()), "Original_Region", "Mapped_Region"),
    }


def _apply_mappings(df: pd.DataFrame, mappings: dict[str, pd.DataFrame]) -> pd.DataFrame:
    uid_map = dict(zip(mappings["uid"]["Original_Uid"], mappings["uid"]["Mapped_Uid"]))
    pid_map = dict(zip(mappings["pid"]["Original_Pid"], mappings["pid"]["Mapped_Pid"]))
    cat_map = dict(zip(mappings["cat"]["Original_Catname"], mappings["cat"]["Mapped_Catname"]))
    region_map = dict(zip(mappings["region"]["Original_Region"], mappings["region"]["Mapped_Region"]))

    mapped = df.copy()
    mapped["Uid"] = mapped["original_uid"].map(uid_map).astype(int)
    mapped["Pid"] = mapped["original_pid"].map(pid_map).astype(int)
    mapped["Catname"] = mapped["category"].map(cat_map).astype(int)
    mapped["Region"] = mapped["region"].map(region_map).astype(int)
    return mapped


def _unique_preserve_order(values: Iterable) -> list:
    seen = set()
    ordered = []
    for value in values:
        normalized = int(value) if isinstance(value, bool | int) else value
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _serialize_hour_counter(hours: Iterable[int]) -> dict[str, int]:
    counts = Counter(int(hour) for hour in hours)
    return {str(hour): int(count) for hour, count in sorted(counts.items())}


def _rank_neighbors(sequences: Iterable[list[int]], forward_only: bool) -> dict[int, list[int]]:
    neighbor_counts: dict[int, Counter[int]] = defaultdict(Counter)
    all_pois: set[int] = set()

    for sequence in sequences:
        all_pois.update(sequence)
        for idx, current in enumerate(sequence):
            if forward_only:
                if idx < len(sequence) - 1:
                    neighbor_counts[current][sequence[idx + 1]] += 1
                continue
            if idx > 0:
                neighbor_counts[current][sequence[idx - 1]] += 1
            if idx < len(sequence) - 1:
                neighbor_counts[current][sequence[idx + 1]] += 1

    ranked = {}
    for poi in sorted(all_pois):
        items = neighbor_counts.get(poi, Counter()).items()
        ranked[poi] = [neighbor for neighbor, _ in sorted(items, key=lambda item: (-item[1], item[0]))]
    return ranked


def _build_poi_info(mapped_df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = mapped_df.sort_values(["Uid", "time_dt"], kind="stable").reset_index(drop=True)
    sequences = sorted_df.groupby("Uid", sort=True)["Pid"].agg(list).tolist()
    neighbors = _rank_neighbors(sequences, forward_only=False)
    forward_neighbors = _rank_neighbors(sequences, forward_only=True)

    poi_info = (
        sorted_df.groupby("Pid", sort=True)
        .agg(
            Uid=("Uid", lambda values: _unique_preserve_order(values)),
            Catname=("Catname", "first"),
            Region=("Region", "first"),
            Time=("visit_hour", lambda values: _unique_preserve_order(int(value) for value in values)),
            original_pid=("original_pid", "first"),
            category=("category", "first"),
            category_id=("category_id", "first"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            unique_user_count=("Uid", "nunique"),
            visit_count=("visit_hour", "size"),
            visit_hours=("visit_hour", lambda values: sorted({int(value) for value in values})),
            visit_time_and_count=("visit_hour", _serialize_hour_counter),
        )
        .reset_index()
    )
    poi_info["neighbors"] = poi_info["Pid"].map(neighbors).apply(lambda value: value or [])
    poi_info["forward_neighbors"] = poi_info["Pid"].map(forward_neighbors).apply(lambda value: value or [])
    poi_info["pid"] = poi_info["Pid"]
    poi_info["region"] = poi_info["Region"].astype(str)

    ordered_columns = [
        "Pid",
        "Uid",
        "Catname",
        "Region",
        "Time",
        "neighbors",
        "forward_neighbors",
        "original_pid",
        "category",
        "category_id",
        "latitude",
        "longitude",
        "unique_user_count",
        "visit_count",
        "visit_hours",
        "visit_time_and_count",
        "pid",
        "region",
    ]
    return poi_info[ordered_columns]


def _split_train_test_base(mapped_df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = mapped_df[["Uid", "Pid", "Time"]].sort_values("Time").reset_index(drop=True)
    train_size = int(train_ratio * len(working))
    train_df = working.iloc[:train_size].copy()
    test_df = working.iloc[train_size:].copy()

    train_users = set(train_df["Uid"].unique().tolist())
    train_pois = set(train_df["Pid"].unique().tolist())
    test_df = test_df[test_df["Uid"].isin(train_users) & test_df["Pid"].isin(train_pois)].copy()
    test_users = test_df["Uid"].unique().tolist()
    expanded_df = working[working["Uid"].isin(test_users)].copy()
    return train_df.reset_index(drop=True), expanded_df.reset_index(drop=True)


def _format_sequence_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Uid", "Pids", "Times", "Target", "Target_time"])
    out = df.copy()
    out["Pids"] = out["Pids"].apply(lambda values: [int(value) for value in values])
    out["Times"] = out["Times"].apply(lambda values: [value.strftime("%Y-%m-%d %H:%M") for value in values])
    out["Target_time"] = out["Target_time"].apply(lambda value: value.strftime("%Y-%m-%d %H:%M"))
    return out


def _generate_train_sequences(
    df: pd.DataFrame,
    *,
    window_size: int,
    step_size: int,
    mask_prob: float,
    max_user_events: int,
    min_sequence_len: int,
    seed: int,
) -> pd.DataFrame:
    working = df.copy()
    working["Time"] = pd.to_datetime(working["Time"])
    rng = random.Random(seed)
    rows: list[dict] = []

    for uid, group in working.groupby("Uid", sort=True):
        group = group.sort_values("Time").reset_index(drop=True)
        if len(group) > max_user_events:
            group = group.iloc[-max_user_events:].reset_index(drop=True)
        n = len(group)

        if n < window_size:
            if n >= min_sequence_len:
                rows.append(
                    {
                        "Uid": int(uid),
                        "Pids": group["Pid"].iloc[:-1].astype(int).tolist(),
                        "Times": group["Time"].iloc[:-1].tolist(),
                        "Target": int(group["Pid"].iloc[-1]),
                        "Target_time": group["Time"].iloc[-1],
                    }
                )
            continue

        for start in range(n - 1, window_size - 2, -step_size):
            window = group.iloc[start - window_size + 1 : start + 1]
            input_pids = window["Pid"].iloc[:-1].astype(int).tolist()
            input_times = window["Time"].iloc[:-1].tolist()
            original_target_pid = int(window["Pid"].iloc[-1])
            original_target_time = window["Time"].iloc[-1]

            if rng.random() < mask_prob and input_pids:
                drop_idx = rng.randint(0, len(input_pids) - 1)
                target_pid = input_pids[drop_idx]
                target_time = input_times[drop_idx]
                input_pids = input_pids[:drop_idx] + input_pids[drop_idx + 1 :] + [original_target_pid]
                input_times = input_times[:drop_idx] + input_times[drop_idx + 1 :] + [original_target_time]
            else:
                target_pid = original_target_pid
                target_time = original_target_time

            rows.append(
                {
                    "Uid": int(uid),
                    "Pids": input_pids,
                    "Times": input_times,
                    "Target": target_pid,
                    "Target_time": target_time,
                }
            )

    return _format_sequence_df(pd.DataFrame(rows))


def _generate_eval_sequences(df: pd.DataFrame, *, window_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.copy()
    working["Time"] = pd.to_datetime(working["Time"])

    val_rows: list[dict] = []
    test_rows: list[dict] = []

    for uid, group in working.groupby("Uid", sort=True):
        group = group.sort_values("Time").reset_index(drop=True)
        n = len(group)

        if n < window_size:
            if n > 2:
                test_rows.append(
                    {
                        "Uid": int(uid),
                        "Pids": group["Pid"].iloc[:-1].astype(int).tolist(),
                        "Times": group["Time"].iloc[:-1].tolist(),
                        "Target": int(group["Pid"].iloc[-1]),
                        "Target_time": group["Time"].iloc[-1],
                    }
                )
                val_rows.append(
                    {
                        "Uid": int(uid),
                        "Pids": group["Pid"].iloc[:-2].astype(int).tolist(),
                        "Times": group["Time"].iloc[:-2].tolist(),
                        "Target": int(group["Pid"].iloc[-2]),
                        "Target_time": group["Time"].iloc[-2],
                    }
                )
            continue

        if n >= window_size + 1:
            val_window = group.iloc[n - window_size - 1 : n - 1]
            val_rows.append(
                {
                    "Uid": int(uid),
                    "Pids": val_window["Pid"].iloc[:-1].astype(int).tolist(),
                    "Times": val_window["Time"].iloc[:-1].tolist(),
                    "Target": int(val_window["Pid"].iloc[-1]),
                    "Target_time": val_window["Time"].iloc[-1],
                }
            )

        test_window = group.iloc[n - window_size :]
        test_rows.append(
            {
                "Uid": int(uid),
                "Pids": test_window["Pid"].iloc[:-1].astype(int).tolist(),
                "Times": test_window["Time"].iloc[:-1].tolist(),
                "Target": int(test_window["Pid"].iloc[-1]),
                "Target_time": test_window["Time"].iloc[-1],
            }
        )

    return _format_sequence_df(pd.DataFrame(val_rows)), _format_sequence_df(pd.DataFrame(test_rows))


def _generate_history_sequences(train_base: pd.DataFrame) -> pd.DataFrame:
    working = train_base.copy()
    working["Time"] = pd.to_datetime(working["Time"])
    rows: list[dict] = []

    for uid, group in working.groupby("Uid", sort=True):
        group = group.sort_values("Time").reset_index(drop=True)
        if len(group) <= 1:
            continue
        rows.append(
            {
                "Uid": int(uid),
                "Pids": group["Pid"].iloc[:-1].astype(int).tolist(),
                "Times": group["Time"].iloc[:-1].tolist(),
                "Target": int(group["Pid"].iloc[-1]),
                "Target_time": group["Time"].iloc[-1],
            }
        )

    return _format_sequence_df(pd.DataFrame(rows))


def build_nyc_from_raw(
    *,
    dataset: str = "NYC",
    raw_path: str | Path | None = None,
    output_root: str | Path | None = None,
    config: RawBuildConfig | None = None,
) -> dict:
    cfg = config or RawBuildConfig(dataset=dataset)
    paths = _resolve_paths(dataset, output_root=output_root)

    ensure_dir(paths.raw)
    ensure_dir(paths.interim)
    ensure_dir(paths.processed)

    source_raw_path = Path(raw_path) if raw_path else (paths.raw / f"{dataset}.txt")
    if not source_raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {source_raw_path}")

    canonical_raw_path = paths.raw / f"{dataset}.txt"
    _copy_raw_if_needed(source_raw_path, canonical_raw_path)

    raw_df = _prepare_raw_table(_read_raw_checkins(source_raw_path))
    filtered_df = _filter_min_frequency(raw_df, cfg.poi_min_freq, cfg.user_min_freq)
    mappings = _build_mappings(filtered_df)
    mapped_df = _apply_mappings(filtered_df, mappings)

    train_base, test_base = _split_train_test_base(mapped_df, cfg.train_ratio)
    train_df = _generate_train_sequences(
        train_base,
        window_size=cfg.window_size,
        step_size=cfg.step_size,
        mask_prob=cfg.mask_prob,
        max_user_events=cfg.max_user_train_events,
        min_sequence_len=cfg.min_sequence_len,
        seed=cfg.seed,
    )
    val_df, test_df = _generate_eval_sequences(test_base, window_size=cfg.window_size)
    history_df = _generate_history_sequences(train_base)
    poi_info_df = _build_poi_info(mapped_df)

    output_frames = {
        paths.processed / "train.csv": train_df,
        paths.processed / "val.csv": val_df,
        paths.processed / "test.csv": test_df,
        paths.processed / "history.csv": history_df,
        paths.processed / "poi_info.csv": poi_info_df,
        paths.processed / "uid_mapping.csv": mappings["uid"],
        paths.processed / "pid_mapping.csv": mappings["pid"],
        paths.processed / "catname_mapping.csv": mappings["cat"],
        paths.processed / "region_mapping.csv": mappings["region"],
    }
    for path, frame in output_frames.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8")

    manifest = {
        "dataset": dataset,
        "raw_path": str(canonical_raw_path),
        "processed_root": str(paths.processed),
        "counts": {
            "raw_rows": int(len(raw_df)),
            "filtered_rows": int(len(filtered_df)),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "history_rows": int(len(history_df)),
            "poi_rows": int(len(poi_info_df)),
            "uid_count": int(len(mappings["uid"])),
            "pid_count": int(len(mappings["pid"])),
            "cat_count": int(len(mappings["cat"])),
            "region_count": int(len(mappings["region"])),
        },
        "config": {
            "poi_min_freq": cfg.poi_min_freq,
            "user_min_freq": cfg.user_min_freq,
            "train_ratio": cfg.train_ratio,
            "window_size": cfg.window_size,
            "step_size": cfg.step_size,
            "mask_prob": cfg.mask_prob,
            "max_user_train_events": cfg.max_user_train_events,
            "min_sequence_len": cfg.min_sequence_len,
            "seed": cfg.seed,
        },
    }
    write_json(paths.interim / "raw_build_manifest.json", manifest)
    logger.info("Built %s processed files from raw checkins at %s", dataset, source_raw_path)
    return manifest
