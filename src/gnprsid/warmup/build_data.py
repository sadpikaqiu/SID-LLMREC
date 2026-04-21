from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from gnprsid.alignment.semantic import sid_prefix
from gnprsid.common.io import ensure_dir, iter_jsonl, read_json, write_json, write_jsonl
from gnprsid.common.logging import get_logger
from gnprsid.common.paths import dataset_paths
from gnprsid.common.profiles import resolve_project_path
from gnprsid.prompts.render import (
    PROMPT_TEMPLATE_VERSION,
    build_next_poi_instruction,
    build_output_requirements,
    build_prompt_input_text,
    system_prompt,
)


logger = get_logger(__name__)


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared sample file: {path}")
    return list(iter_jsonl(path))


def _load_sid_space(dataset: str) -> list[str]:
    paths = dataset_paths(dataset)
    sid_map_path = paths.artifacts / "sid" / "pid_to_sid.json"
    if not sid_map_path.exists():
        raise FileNotFoundError(f"Missing SID mapping file: {sid_map_path}")
    sid_payload = read_json(sid_map_path)
    sid_space = sorted({str(meta["sid_token"]) for meta in sid_payload.values()})
    if len(sid_space) < 10:
        raise ValueError("Warmup candidate space must contain at least 10 semantic IDs.")
    return sid_space


def _build_prefix_groups(sid_space: list[str]) -> dict[str, dict[str, list[str]]]:
    groups: dict[str, dict[str, list[str]]] = {
        "a": defaultdict(list),
        "ab": defaultdict(list),
        "abc": defaultdict(list),
    }
    for sid in sid_space:
        groups["a"][sid_prefix(sid, "a")].append(sid)
        groups["ab"][sid_prefix(sid, "ab")].append(sid)
        groups["abc"][sid_prefix(sid, "abc")].append(sid)
    return groups


def _ranking_key(sid: str, target_counts: Counter[str]) -> tuple[int, str]:
    return (-int(target_counts.get(sid, 0)), sid)


def build_ranked_sid_targets(
    target_sid: str,
    sid_space: list[str],
    target_counts: Counter[str],
    prefix_groups: dict[str, dict[str, list[str]]],
    top_k: int = 10,
) -> list[str]:
    if target_sid not in sid_space:
        raise ValueError(f"Target SID {target_sid!r} is not present in SID space.")

    ranked = [target_sid]
    seen = {target_sid}
    target_a = sid_prefix(target_sid, "a")
    target_ab = sid_prefix(target_sid, "ab")
    target_abc = sid_prefix(target_sid, "abc")

    same_abc_candidates = sorted(prefix_groups["abc"][target_abc], key=lambda sid: _ranking_key(sid, target_counts))
    for candidate in same_abc_candidates:
        if candidate in seen:
            continue
        ranked.append(candidate)
        seen.add(candidate)
        if len(ranked) == 2 or len(ranked) == top_k:
            break
    if len(ranked) == top_k:
        return ranked

    same_ab_candidates = sorted(prefix_groups["ab"][target_ab], key=lambda sid: _ranking_key(sid, target_counts))
    for candidate in same_ab_candidates:
        if candidate in seen or sid_prefix(candidate, "abc") == target_abc:
            continue
        ranked.append(candidate)
        seen.add(candidate)
        if len(ranked) == top_k:
            return ranked

    same_a_candidates = sorted(prefix_groups["a"][target_a], key=lambda sid: _ranking_key(sid, target_counts))
    for candidate in same_a_candidates:
        if candidate in seen or sid_prefix(candidate, "ab") == target_ab:
            continue
        ranked.append(candidate)
        seen.add(candidate)
        if len(ranked) == top_k:
            return ranked

    global_candidates = sorted(sid_space, key=lambda sid: _ranking_key(sid, target_counts))
    for candidate in global_candidates:
        if candidate in seen or sid_prefix(candidate, "abc") == target_abc:
            continue
        ranked.append(candidate)
        seen.add(candidate)
        if len(ranked) == top_k:
            return ranked

    for candidate in global_candidates:
        if candidate in seen:
            continue
        ranked.append(candidate)
        seen.add(candidate)
        if len(ranked) == top_k:
            return ranked

    raise ValueError(f"Could not construct {top_k} ranked SIDs for target {target_sid!r}.")


def _build_warmup_rows(
    rows: list[dict],
    sid_space: list[str],
    target_counts: Counter[str],
    prefix_groups: dict[str, dict[str, list[str]]],
    history_source: str,
) -> list[dict]:
    instruction = (
        system_prompt("sid", history_source, candidate_count=10)
        + "\n\n"
        + build_next_poi_instruction(history_source)
    )
    payload: list[dict] = []
    for row in rows:
        ranked_targets = build_ranked_sid_targets(
            str(row["target"]),
            sid_space=sid_space,
            target_counts=target_counts,
            prefix_groups=prefix_groups,
            top_k=10,
        )
        payload.append(
            {
                "instruction": instruction,
                "input": (
                    build_prompt_input_text(row, history_source)
                    + "\n\n"
                    + build_output_requirements("sid", candidate_count=10)
                ),
                "output": " ".join(ranked_targets),
                "sample_id": row["sample_id"],
                "target": str(row["target"]),
                "ranked_targets": ranked_targets,
            }
        )
    return payload


def build_warmup_data(
    dataset: str = "NYC",
    history_source: str = "current",
    output_dir: str | Path | None = None,
) -> dict:
    if history_source != "current":
        raise ValueError("The first direct-10 warmup version only supports history_source='current'.")

    paths = dataset_paths(dataset)
    source_dir = paths.processed
    output_dir = resolve_project_path(output_dir) if output_dir else (paths.artifacts / "warmup" / "sid" / history_source)
    output_dir = ensure_dir(output_dir)

    train_rows = _load_rows(source_dir / "samples_sid_train.jsonl")
    valid_rows = _load_rows(source_dir / "samples_sid_val.jsonl")
    sid_space = _load_sid_space(dataset)
    target_counts = Counter(str(row["target"]) for row in train_rows)
    prefix_groups = _build_prefix_groups(sid_space)

    train_payload = _build_warmup_rows(train_rows, sid_space, target_counts, prefix_groups, history_source)
    valid_payload = _build_warmup_rows(valid_rows, sid_space, target_counts, prefix_groups, history_source)

    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    write_jsonl(train_path, train_payload)
    write_jsonl(valid_path, valid_payload)

    manifest = {
        "dataset": dataset,
        "repr": "sid",
        "history_source": history_source,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "num_train": len(train_payload),
        "num_valid": len(valid_payload),
        "sid_space_size": len(sid_space),
    }
    write_json(output_dir / "manifest.json", manifest)
    logger.info("Built direct-10 warmup data for %s at %s", dataset, output_dir)
    return manifest
