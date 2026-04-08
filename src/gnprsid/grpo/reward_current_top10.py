from __future__ import annotations

import re
from typing import Iterable


SID_PATTERN = r"<a_\d+><b_\d+><c_\d+>(?:<d_\d+>)?"
SID_TOKEN_PATTERN = r"<[a-z]_\d+>"

FORMAT_WEIGHT = 1.0
SINGLE_LINE_COMPONENT_WEIGHT = 0.2
VALID_COUNT_COMPONENT_WEIGHT = 0.5
EXACT_TEN_COMPONENT_WEIGHT = 0.3
RECIPROCAL_RANK_WEIGHT = 1.0
SOFT_HIT_WEIGHT = 1.0
PREFIX_MATCH_WEIGHT = 0.2
DIVERSITY_WEIGHT = 0.2


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


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> float:
    del data_source, extra_info

    parsed_predictions = _extract_predictions(solution_str)
    single_line_score = float(_is_single_line_output(solution_str))
    valid_count_score = len(parsed_predictions) / 10.0
    exact_ten_score = float(len(parsed_predictions) == 10)
    format_score = (
        SINGLE_LINE_COMPONENT_WEIGHT * single_line_score
        + VALID_COUNT_COMPONENT_WEIGHT * valid_count_score
        + EXACT_TEN_COMPONENT_WEIGHT * exact_ten_score
    )
    format_reward = FORMAT_WEIGHT * format_score

    reciprocal_rank_reward = 0.0
    soft_hit_reward = 0.0
    if ground_truth in parsed_predictions:
        rank = parsed_predictions.index(ground_truth) + 1
        reciprocal_rank_reward = RECIPROCAL_RANK_WEIGHT * (1.0 / rank)
        soft_hit_reward = SOFT_HIT_WEIGHT

    prefix_match_reward = PREFIX_MATCH_WEIGHT * _prefix_match_score(parsed_predictions, str(ground_truth))
    diversity_reward = DIVERSITY_WEIGHT * (len(parsed_predictions) / 10.0)

    return format_reward + reciprocal_rank_reward + soft_hit_reward + prefix_match_reward + diversity_reward
