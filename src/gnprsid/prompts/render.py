from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence


PROMPT_TEMPLATE_VERSION = "v1"
SID_PATTERN = r"<[a-zA-Z]_\d+>(?:<[a-zA-Z]_\d+>){2,3}"
ID_PATTERN = r"<\d+>"


def deduplicate_preserve_order(values: Sequence[str]) -> List[str]:
    seen = set()
    unique_values: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def _target_count_text(count: int) -> str:
    return "exactly 1" if count == 1 else f"exactly {count}"


def build_output_requirements(repr_name: str, candidate_count: int = 10) -> str:
    count_text = _target_count_text(candidate_count)
    if repr_name == "id":
        example = "<123>" if candidate_count == 1 else (
            "<123> <456> <789> <1011> <1213> <1415> <1617> <1819> <2021> <2223>"
        )
        return "\n".join(
            [
                "Output format:",
                f"1. Return {count_text} candidate POI IDs in one line.",
                "2. Use descending likelihood order.",
                "3. Each candidate must use the POI ID format like <123>.",
                "4. Separate candidates with a single space only.",
                "5. Do not output explanations, numbering, commas, or any extra text.",
                "6. Do not repeat candidates.",
                f"Example: {example}",
            ]
        )

    example = (
        "<a_1><b_2><c_3>"
        if candidate_count == 1
        else (
            "<a_1><b_2><c_3> <a_1><b_2><c_8> <a_4><b_1><c_0><d_0> "
            "<a_4><b_1><c_0><d_1> <a_9><b_3><c_7> <a_2><b_5><c_1> "
            "<a_7><b_0><c_4> <a_5><b_6><c_2> <a_8><b_9><c_3> <a_6><b_2><c_1>"
        )
    )
    return "\n".join(
        [
            "Semantic ID notes:",
            "1. A semantic ID looks like <a_1><b_2><c_3> or <a_1><b_2><c_3><d_0>.",
            "2. Prefix a is a coarse semantic group shared by broadly similar POIs.",
            "3. Prefix b refines the group under a, and c refines it further.",
            "4. Prefix d is only used to separate POIs when a, b, and c are identical.",
            "5. Longer shared prefixes usually indicate more similar POIs.",
            "",
            "Output format:",
            f"1. Return {count_text} complete semantic IDs in one line.",
            "2. Use descending likelihood order.",
            "3. Each candidate must be a complete semantic ID.",
            "4. Separate candidates with a single space only.",
            "5. Do not output explanations, numbering, commas, or any extra text.",
            "6. Do not repeat candidates.",
            f"Example: {example}",
        ]
    )


def build_current_sequence_block(sample: Dict[str, object]) -> str:
    return "\n".join(
        [
            "Current trajectory to predict:",
            str(sample["key_text"]),
            f"Prediction time: {sample['target_time']}",
        ]
    )


def build_task_header(history_source: str) -> str:
    if history_source == "current":
        evidence_hint = "Use only the current observed trajectory."
    elif history_source == "original":
        evidence_hint = "Use the user's full historical record together with the current trajectory."
    elif history_source == "hybrid":
        evidence_hint = (
            "Use the user's full historical record, retrieved similar cases, and the current trajectory together."
        )
    else:
        evidence_hint = "Use the retrieved similar cases together with the current trajectory."

    return "\n".join(
        [
            "Task:",
            "Predict the next POI that the user is most likely to visit at the prediction time.",
            evidence_hint,
        ]
    )


def build_original_history_block(sample: Dict[str, object], history_map: Dict[int, str]) -> str:
    history_text = history_map.get(int(sample["uid"]))
    if not history_text:
        return "Full historical record:\nNo historical record is available."
    return f"Full historical record:\n{history_text}"


def format_retrieved_trajectory(index: int, sample: Dict[str, object]) -> str:
    return "\n".join(
        [
            f"Retrieved case {index}:",
            "Observed trajectory:",
            str(sample["key_text"]),
            f"Ground-truth next POI at {sample['target_time']}:",
            str(sample["target"]),
        ]
    )


def build_retrieval_history_block(
    sample: Dict[str, object],
    similar_map: Dict[str, List[Dict[str, object]]],
    bank_map: Dict[str, Dict[str, object]],
    top_k_retrieval: int,
) -> str:
    neighbors = similar_map.get(str(sample["sample_id"]), [])[:top_k_retrieval]
    rendered: List[str] = []
    for index, item in enumerate(neighbors, start=1):
        neighbor = bank_map.get(str(item["sample_id"]))
        if neighbor is None:
            continue
        rendered.append(format_retrieved_trajectory(index, neighbor))

    if not rendered:
        return "Retrieved similar cases:\nNo retrieved case is available."
    return "Retrieved similar cases:\n" + "\n\n".join(rendered)


def build_prompt_body(
    sample: Dict[str, object],
    history_source: str,
    history_map: Optional[Dict[int, str]] = None,
    similar_map: Optional[Dict[str, List[Dict[str, object]]]] = None,
    bank_map: Optional[Dict[str, Dict[str, object]]] = None,
    top_k_retrieval: int = 5,
) -> str:
    sections = [build_task_header(history_source)]
    if history_source in {"original", "hybrid"}:
        sections.append(build_original_history_block(sample, history_map or {}))
    if history_source in {"retrieval", "hybrid"}:
        sections.append(build_retrieval_history_block(sample, similar_map or {}, bank_map or {}, top_k_retrieval))
    sections.append(build_current_sequence_block(sample))
    return "\n\n".join(sections)


def build_prompt(
    sample: Dict[str, object],
    history_source: str,
    history_map: Optional[Dict[int, str]] = None,
    similar_map: Optional[Dict[str, List[Dict[str, object]]]] = None,
    bank_map: Optional[Dict[str, Dict[str, object]]] = None,
    top_k_retrieval: int = 5,
    candidate_count: int = 10,
) -> str:
    return "\n\n".join(
        [
            build_prompt_body(
                sample,
                history_source,
                history_map=history_map,
                similar_map=similar_map,
                bank_map=bank_map,
                top_k_retrieval=top_k_retrieval,
            ),
            build_output_requirements(str(sample["repr"]), candidate_count=candidate_count),
        ]
    )


def build_supervised_prompt(
    sample: Dict[str, object],
    history_source: str,
    history_map: Optional[Dict[int, str]] = None,
    similar_map: Optional[Dict[str, List[Dict[str, object]]]] = None,
    bank_map: Optional[Dict[str, Dict[str, object]]] = None,
    top_k_retrieval: int = 5,
) -> str:
    return build_prompt(
        sample,
        history_source,
        history_map=history_map,
        similar_map=similar_map,
        bank_map=bank_map,
        top_k_retrieval=top_k_retrieval,
        candidate_count=1,
    )


def system_prompt(repr_name: str, history_source: str, candidate_count: int = 10) -> str:
    history_hint = {
        "current": "You only have the current observed trajectory.",
        "original": "You must use the user's full historical record and the current observed trajectory.",
        "retrieval": "You must use the retrieved similar cases and the current observed trajectory.",
        "hybrid": "You must use the user's full historical record, the retrieved similar cases, and the current observed trajectory.",
    }[history_source]

    count_text = _target_count_text(candidate_count)
    base_rules = (
        "You are a POI recommendation assistant. "
        f"{history_hint} "
        f"Return one line only. Return {count_text} candidates ordered by likelihood. "
        "Separate candidates with a single space. Do not add explanations or duplicate candidates."
    )
    if repr_name == "id":
        return base_rules + " Each candidate must use the POI ID format like <123>."
    return (
        base_rules
        + " Each candidate must be a complete semantic ID. "
        "Semantic IDs have hierarchical prefixes a/b/c/(d): a is coarse, b and c refine the semantics, "
        "and d only disambiguates when a, b, and c are identical."
    )


def extract_predictions(text: str, repr_name: str) -> List[str]:
    pattern = ID_PATTERN if repr_name == "id" else SID_PATTERN
    return deduplicate_preserve_order(re.findall(pattern, text.strip()))
