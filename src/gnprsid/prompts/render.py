from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence


PROMPT_TEMPLATE_VERSION = "v2"
SID_PATTERN = r"<[a-zA-Z]_\d+>(?:<[a-zA-Z]_\d+>){2,3}"
ID_PATTERN = r"<\d+>"
V2_NEXT_POI_INSTRUCTION = (
    "Here is a record of a user's POI accesses, your task is based on the history "
    "to predict the POI that the user is likely to access at the specified time."
)


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


def build_next_poi_instruction(history_source: str) -> str:
    return V2_NEXT_POI_INSTRUCTION


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


def _fallback_current_query(sample: Dict[str, object]) -> str:
    return "\n".join(
        [
            str(sample["key_text"]),
            f"Prediction time: {sample['target_time']}",
        ]
    )


def build_current_input_block(sample: Dict[str, object]) -> str:
    return str(sample.get("input_text") or _fallback_current_query(sample))


def build_original_history_block(sample: Dict[str, object], history_map: Dict[int, str]) -> str:
    history_text = history_map.get(int(sample["uid"]))
    if not history_text:
        return "Historical trajectory:\nNo historical trajectory is available."
    return f"Historical trajectory:\n{history_text}"


def format_retrieved_trajectory(index: int, sample: Dict[str, object]) -> str:
    return "\n".join(
        [
            f"Case {index}:",
            f"Trajectory: {sample['key_text']}",
            f"Next POI: {sample['target']}",
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


def build_prompt_input_text(
    sample: Dict[str, object],
    history_source: str,
    history_map: Optional[Dict[int, str]] = None,
    similar_map: Optional[Dict[str, List[Dict[str, object]]]] = None,
    bank_map: Optional[Dict[str, Dict[str, object]]] = None,
    top_k_retrieval: int = 5,
) -> str:
    sections: List[str] = []
    if history_source in {"original", "hybrid"}:
        sections.append(build_original_history_block(sample, history_map or {}))
    if history_source in {"retrieval", "hybrid"}:
        sections.append(build_retrieval_history_block(sample, similar_map or {}, bank_map or {}, top_k_retrieval))
    sections.append(build_current_input_block(sample))
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
    sections = [
        "### Instruction:\n" + build_next_poi_instruction(history_source),
        "### Input:\n"
        + build_prompt_input_text(
            sample,
            history_source,
            history_map=history_map,
            similar_map=similar_map,
            bank_map=bank_map,
            top_k_retrieval=top_k_retrieval,
        ),
    ]
    if candidate_count != 1:
        sections.append(build_output_requirements(str(sample["repr"]), candidate_count=candidate_count))
    return "\n\n".join(sections)


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
    if repr_name == "id":
        return "You are a helpful assistant for next POI prediction. Reply with POI IDs only."
    return "You are a helpful assistant for next POI prediction. Reply with semantic IDs only."


def extract_predictions(text: str, repr_name: str) -> List[str]:
    pattern = ID_PATTERN if repr_name == "id" else SID_PATTERN
    return deduplicate_preserve_order(re.findall(pattern, text.strip()))
