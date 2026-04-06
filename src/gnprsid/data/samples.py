from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import pandas as pd

from gnprsid.common.io import iter_jsonl, write_jsonl


SID_PATTERN = r"<a_\d+><b_\d+><c_\d+>(?:<d_\d+>)?"
ID_PATTERN = r"<\d+>"


def sanitize_literal_string(value: str) -> str:
    value = re.sub(r"np\.int64\(([-]?\d+)\)", r"\1", value)
    value = re.sub(r"numpy\.int64\(([-]?\d+)\)", r"\1", value)
    return value


def parse_literal_list(value: object) -> List:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    if isinstance(value, str):
        value = sanitize_literal_string(value)
    return list(ast.literal_eval(value))


def sid_indices_to_token(indices: Sequence[int]) -> str:
    return "".join(f"<{chr(97 + idx)}_{value}>" for idx, value in enumerate(indices))


def load_sid_token_map(path: str | Path) -> Dict[int, str]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {int(pid): str(meta["sid_token"]) for pid, meta in payload.items()}
    if path.suffix.lower() == ".csv":
        mapping: Dict[int, str] = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                token = row.get("sid_token")
                if not token and row.get("sid_indices"):
                    token = sid_indices_to_token(parse_literal_list(row["sid_indices"]))
                if not token and row.get("sid"):
                    token = sid_indices_to_token(parse_literal_list(row["sid"]))
                if not token:
                    raise ValueError(f"Could not determine sid token in row: {row}")
                mapping[int(row["pid"])] = token
        return mapping
    raise ValueError(f"Unsupported SID mapping format: {path}")


def format_poi_token(pid: int, repr_name: str, sid_token_map: Optional[Dict[int, str]] = None) -> str:
    if repr_name == "id":
        return f"<{pid}>"
    if repr_name == "sid":
        if sid_token_map is None:
            raise ValueError("sid_token_map is required for SID representation")
        return sid_token_map[int(pid)]
    raise ValueError(f"Unsupported representation: {repr_name}")


def format_sequence_text(
    uid: int,
    pids: Sequence[int],
    times: Sequence[str],
    repr_name: str,
    sid_token_map: Optional[Dict[int, str]] = None,
) -> str:
    parts = []
    for pid, visit_time in zip(pids, times):
        poi_token = format_poi_token(pid, repr_name, sid_token_map)
        parts.append(f"{poi_token} at {visit_time}")
    return f"User_{uid} visited: " + ", ".join(parts)


def format_prediction_input(
    uid: int,
    pids: Sequence[int],
    times: Sequence[str],
    target_time: str,
    repr_name: str,
    sid_token_map: Optional[Dict[int, str]] = None,
) -> str:
    prefix = format_sequence_text(uid, pids, times, repr_name, sid_token_map)
    return f"{prefix}. When {target_time} user_{uid} is likely to visit:"


def build_sample_rows(
    split: str,
    csv_path: str | Path,
    repr_name: str,
    current_k: int,
    sid_token_map: Optional[Dict[int, str]] = None,
) -> List[dict]:
    df = pd.read_csv(csv_path)
    rows: List[dict] = []
    for row_index, row in df.iterrows():
        uid = int(row["Uid"])
        all_pids = [int(pid) for pid in parse_literal_list(row["Pids"])]
        all_times = [str(value) for value in parse_literal_list(row["Times"])]
        observed_pids = all_pids[-current_k:] if current_k > 0 else list(all_pids)
        observed_times = all_times[-current_k:] if current_k > 0 else list(all_times)
        target_pid = int(row["Target"])
        target_time = str(row["Target_time"])
        target = format_poi_token(target_pid, repr_name, sid_token_map)
        sample_id = f"{split}-{row_index:05d}-u{uid}"

        key_text = format_sequence_text(uid, observed_pids, observed_times, repr_name, sid_token_map)
        query_text = format_sequence_text(
            uid,
            list(observed_pids) + [target_pid],
            list(observed_times) + [target_time],
            repr_name,
            sid_token_map,
        )

        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "row_index": int(row_index),
                "uid": uid,
                "repr": repr_name,
                "instruction": "Predict the next POI from historical and current trajectory evidence.",
                "target_pid": target_pid,
                "target": target,
                "target_time": target_time,
                "key_text": key_text,
                "query_text": query_text,
                "input_text": format_prediction_input(
                    uid,
                    observed_pids,
                    observed_times,
                    target_time,
                    repr_name,
                    sid_token_map,
                ),
                "observed_pids": observed_pids,
                "observed_times": observed_times,
            }
        )
    return rows


def write_sample_rows(path: str | Path, rows: Iterable[dict]) -> None:
    write_jsonl(path, rows)


def load_sample_rows(path: str | Path) -> List[dict]:
    return list(iter_jsonl(path))


def filter_samples(path: str | Path, split: str, repr_name: str) -> List[dict]:
    return [
        row
        for row in iter_jsonl(path)
        if row["split"] == split and row["repr"] == repr_name
    ]


def load_history_map(path: str | Path) -> Dict[int, str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    history_map: Dict[int, str] = {}
    for item in data:
        text = item["input"]
        match = re.search(r"User_(\d+)", text)
        if match:
            history_map[int(match.group(1))] = text
    return history_map
