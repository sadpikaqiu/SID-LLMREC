from __future__ import annotations

import re


SID_PATTERN = r"<a_\d+><b_\d+><c_\d+>(?:<d_\d+>)?"
SID_TOKEN_PATTERN = r"<[a-z]_\d+>"
STRICT_FORMAT_WEIGHT = 0.1
EXACT_MATCH_WEIGHT = 1.0
PREFIX_MATCH_WEIGHT = 0.2


def _extract_predictions(text: str) -> list[str]:
    return re.findall(SID_PATTERN, text.strip())


def _is_strict_valid_output(solution_str: str, parsed_predictions: list[str]) -> bool:
    if len(parsed_predictions) != 1:
        return False
    stripped = solution_str.strip()
    if "\n" in stripped or "\r" in stripped:
        return False
    return stripped == parsed_predictions[0]


def _common_prefix_depth(prediction: str, target: str) -> int:
    prediction_tokens = re.findall(SID_TOKEN_PATTERN, prediction)
    target_tokens = re.findall(SID_TOKEN_PATTERN, target)
    depth = 0
    for pred_token, target_token in zip(prediction_tokens[:3], target_tokens[:3]):
        if pred_token != target_token:
            break
        depth += 1
    return depth


def _prefix_match_score(prediction: str | None, target: str) -> float:
    if not prediction:
        return 0.0
    depth = _common_prefix_depth(prediction, target)
    return 2 ** (depth - 3)


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    parsed_predictions = _extract_predictions(solution_str)
    strict_valid = _is_strict_valid_output(solution_str, parsed_predictions)
    prediction = parsed_predictions[0] if parsed_predictions else None

    format_reward = STRICT_FORMAT_WEIGHT * float(strict_valid)
    exact_match_reward = EXACT_MATCH_WEIGHT * float(strict_valid and prediction == str(ground_truth))
    prefix_match_reward = PREFIX_MATCH_WEIGHT * _prefix_match_score(prediction, str(ground_truth))
    return format_reward + exact_match_reward + prefix_match_reward
