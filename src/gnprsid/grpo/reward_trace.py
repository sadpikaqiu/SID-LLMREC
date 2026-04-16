from __future__ import annotations

import json
import os
import time
from itertools import count
from pathlib import Path
from typing import Any, Mapping


TRACE_DIR_ENV = "GNPRSID_REWARD_TRACE_DIR"
TRACE_GROUP_SIZE_ENV = "GNPRSID_REWARD_TRACE_GROUP_SIZE"

_LOCAL_RECORD_COUNTER = count()


def _coerce_trace_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _coerce_trace_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_trace_value(item) for item in value]
    return str(value)


def append_reward_trace(extra_info: Mapping[str, Any] | None, payload: Mapping[str, Any]) -> None:
    trace_dir = os.environ.get(TRACE_DIR_ENV)
    if not trace_dir:
        return

    trace_path = Path(trace_dir) / f"reward_trace_pid{os.getpid()}.jsonl"
    record: dict[str, Any] = {
        "time_ns": time.time_ns(),
        "pid": os.getpid(),
        "local_record_index": next(_LOCAL_RECORD_COUNTER),
    }

    group_size_hint = os.environ.get(TRACE_GROUP_SIZE_ENV)
    if group_size_hint:
        try:
            record["group_size_hint"] = int(group_size_hint)
        except ValueError:
            pass

    if extra_info:
        for key in (
            "sample_id",
            "uid",
            "repr",
            "history_source",
            "target",
            "target_time",
            "prompt_template_version",
        ):
            if key in extra_info:
                record[key] = _coerce_trace_value(extra_info[key])

    for key, value in payload.items():
        record[key] = _coerce_trace_value(value)

    try:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        # Reward tracing is observational only and must never break training.
        return
