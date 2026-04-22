from __future__ import annotations

import re
from typing import Iterable

from gnprsid.grpo.reward_trace import append_reward_trace


SID_PATTERN = r"<a_\d+><b_\d+><c_\d+>(?:<d_\d+>)?"
SID_TOKEN_PATTERN = r"<[a-z]_\d+>"

FORMAT_WEIGHT = 0.5
SINGLE_LINE_COMPONENT_WEIGHT = 0.2
VALID_COUNT_COMPONENT_WEIGHT = 0.5
EXACT_TEN_COMPONENT_WEIGHT = 0.3
RECIPROCAL_RANK_WEIGHT = 1.0
SOFT_HIT_WEIGHT = 1.0
PREFIX_MATCH_WEIGHT = 0.2
DIVERSITY_WEIGHT = 0.1


def _extract_predictions(text: str) -> list[str]:
    seen: set[str] = set()
    predictions: list[str] = []
    for match in re.findall(SID_PATTERN, text.strip()):
        if match in seen:
            continue
        seen.add(match)
        predictions.append(match)
    return predictions[:10]


def _is_single_line_output(solution_str: str) -> bool:
    stripped = solution_str.strip()
    return "\n" not in stripped and "\r" not in stripped


def _preview_text(text: str, limit: int = 240) -> str:
    normalized = " ".join(text.strip().split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _common_prefix_depth(prediction: str, target: str) -> int:
    prediction_tokens = re.findall(SID_TOKEN_PATTERN, prediction)
    target_tokens = re.findall(SID_TOKEN_PATTERN, target)
    depth = 0
    for pred_token, target_token in zip(prediction_tokens[:3], target_tokens[:3]):
        if pred_token != target_token:
            break
        depth += 1
    return depth


def _prefix_match_score(predictions: Iterable[str], target: str) -> float:
    total = sum(_common_prefix_depth(prediction, target) for prediction in predictions)
    return total / 30.0


def _reciprocal_rank_boost(rank: int) -> float:
    capped_rank = min(max(rank, 1), 10)
    return 1.0 + (10 - capped_rank) / 10.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> float:
    del data_source

    parsed_predictions = _extract_predictions(solution_str)
    single_line_score = float(_is_single_line_output(solution_str))
    valid_count_score = len(parsed_predictions) / 10.0
    exact_ten_score = float(len(parsed_predictions) == 10)
    single_line_reward = FORMAT_WEIGHT * SINGLE_LINE_COMPONENT_WEIGHT * single_line_score
    valid_count_reward = FORMAT_WEIGHT * VALID_COUNT_COMPONENT_WEIGHT * valid_count_score
    exact_ten_reward = FORMAT_WEIGHT * EXACT_TEN_COMPONENT_WEIGHT * exact_ten_score
    format_score = (
        SINGLE_LINE_COMPONENT_WEIGHT * single_line_score
        + VALID_COUNT_COMPONENT_WEIGHT * valid_count_score
        + EXACT_TEN_COMPONENT_WEIGHT * exact_ten_score
    )
    format_reward = single_line_reward + valid_count_reward + exact_ten_reward

    reciprocal_rank_reward = 0.0
    soft_hit_reward = 0.0
    reciprocal_rank_boost = 0.0
    if ground_truth in parsed_predictions:
        rank = parsed_predictions.index(ground_truth) + 1
        reciprocal_rank_boost = _reciprocal_rank_boost(rank)
        reciprocal_rank_reward = RECIPROCAL_RANK_WEIGHT * (1.0 / rank) * reciprocal_rank_boost
        soft_hit_reward = SOFT_HIT_WEIGHT

    prefix_match_reward = PREFIX_MATCH_WEIGHT * _prefix_match_score(parsed_predictions, str(ground_truth))
    diversity_reward = DIVERSITY_WEIGHT * (len(parsed_predictions) / 10.0)
    total_reward = (
        format_reward
        + reciprocal_rank_reward
        + soft_hit_reward
        + prefix_match_reward
        + diversity_reward
    )

    append_reward_trace(
        extra_info,
        {
            "solution_preview": _preview_text(solution_str),
            "solution_char_length": len(solution_str),
            "parsed_predictions": parsed_predictions,
            "single_line_score": single_line_score,
            "valid_count_score": valid_count_score,
            "exact_ten_score": exact_ten_score,
            "single_line_reward": single_line_reward,
            "valid_count_reward": valid_count_reward,
            "exact_ten_reward": exact_ten_reward,
            "parsed_prediction_count": len(parsed_predictions),
            "hit": float(ground_truth in parsed_predictions),
            "rank": parsed_predictions.index(ground_truth) + 1 if ground_truth in parsed_predictions else None,
            "reciprocal_rank_boost": reciprocal_rank_boost,
            "format_reward": format_reward,
            "reciprocal_rank_reward": reciprocal_rank_reward,
            "soft_hit_reward": soft_hit_reward,
            "prefix_match_reward": prefix_match_reward,
            "diversity_reward": diversity_reward,
            "total_reward": total_reward,
        },
    )

    return total_reward
