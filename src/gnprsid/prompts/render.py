from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence


PROMPT_TEMPLATE_VERSION = "v3"
SID_PATTERN = r"<a_\d+><b_\d+><c_\d+>(?:<d_\d+>)?"
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
        lines = [
            "Output format:",
            f"1. Return {count_text} candidate POI IDs in one line.",
            "2. Use descending likelihood order.",
            "3. Each candidate must use the POI ID format like <123>.",
            "4. Separate candidates with a single space only.",
            "5. Do not output explanations, numbering, commas, or any extra text.",
            "6. Do not repeat candidates.",
        ]
        if candidate_count != 1:
            lines.append("7. Start the reply immediately with the first POI ID.")
        return "\n".join(lines)

    lines = [
        "Output format:",
        f"1. Return {count_text} complete semantic IDs.",
        "2. Use descending likelihood order.",
        "3. Output one line with single spaces only.",
        "4. Do not output explanations, numbering, commas, or duplicate IDs.",
    ]
    if candidate_count != 1:
        lines.append("5. Start the reply immediately with the first semantic ID.")
        return "\n".join(lines)

    return "\n".join(
        lines + ["Example: <a_1><b_2><c_3>"]
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
        if candidate_count != 1:
            return (
                "You are a next-POI prediction assistant. "
                "Reply with exactly 10 POI IDs in descending likelihood order. "
                "Output one line only. Separate IDs with a single space. "
                "Do not output explanations, numbering, commas, or duplicate IDs. "
                "Start immediately with the first POI ID."
            )
        return "You are a helpful assistant for next POI prediction. Reply with POI IDs only."
    if candidate_count != 1:
        return (
            "You are a next-POI prediction assistant. "
            "Reply with exactly 10 complete semantic IDs in descending likelihood order. "
            "Output one line only. Separate IDs with a single space. "
            "Do not output explanations, numbering, commas, or duplicate IDs. "
            "Start immediately with the first semantic ID."
        )
    return "You are a helpful assistant for next POI prediction. Reply with semantic IDs only."


def extract_predictions(text: str, repr_name: str) -> List[str]:
    pattern = ID_PATTERN if repr_name == "id" else SID_PATTERN
    return deduplicate_preserve_order(re.findall(pattern, text.strip()))
